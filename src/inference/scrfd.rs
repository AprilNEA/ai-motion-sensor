use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use ort::session::Session;

use crate::config::FaceConfig;
use crate::geometry::{
    BBox, Detection, FaceDetection, LetterboxInfo, Point2D, letterbox_params,
    nms, unletterbox_bbox, unletterbox_point,
};
use crate::inference::engine::load_model;

const INPUT_SIZE: u32 = 640;
const STRIDES: [u32; 3] = [8, 16, 32];

pub struct ScrfdDetector {
    session: Session,
    config: FaceConfig,
}

impl ScrfdDetector {
    pub fn new(model_path: &str, config: FaceConfig) -> Result<Self> {
        let session = load_model(model_path)?;
        Ok(Self { session, config })
    }

    /// Detect faces and 5-point landmarks.
    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<FaceDetection>> {
        let (img_w, img_h) = image.dimensions();
        let info = letterbox_params(img_w, img_h, INPUT_SIZE, INPUT_SIZE);
        let input = self.preprocess(image, &info);
        let tensor = ort::value::Tensor::from_array(input)?;

        let outputs = self.session.run(ort::inputs![tensor])?;

        // Discover output layout by inspecting shapes and names.
        // SCRFD models output 9 tensors.  The buffalo_l det_10g.onnx uses
        // the ordering:
        //   [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
        // We classify outputs by their last dimension:
        //   dim[-1]==1  → score
        //   dim[-1]==4  → bbox
        //   dim[-1]==10 → kps
        let num_outputs = outputs.len();
        let mut score_outputs: Vec<usize> = Vec::new();
        let mut bbox_outputs: Vec<usize> = Vec::new();
        let mut kps_outputs: Vec<usize> = Vec::new();

        for i in 0..num_outputs {
            let arr = outputs[i].try_extract_array::<f32>()?;
            let shape = arr.shape();
            tracing::debug!(output_idx = i, ?shape, "SCRFD output");
            match shape.last().copied() {
                Some(1) => score_outputs.push(i),
                Some(4) => bbox_outputs.push(i),
                Some(10) => kps_outputs.push(i),
                _ => {} // skip unknown
            }
        }

        // Sort each group by descending number of anchors (stride 8 has the most).
        // Output shape is 2D: [num_anchors, features], so shape[0] is anchor count.
        let sort_by_anchors_desc = |indices: &mut Vec<usize>| {
            indices.sort_by(|&a, &b| {
                let sa = outputs[a].try_extract_array::<f32>().map(|v| v.shape()[0]).unwrap_or(0);
                let sb = outputs[b].try_extract_array::<f32>().map(|v| v.shape()[0]).unwrap_or(0);
                sb.cmp(&sa)
            });
        };
        sort_by_anchors_desc(&mut score_outputs);
        sort_by_anchors_desc(&mut bbox_outputs);
        sort_by_anchors_desc(&mut kps_outputs);

        let num_strides = score_outputs.len().min(bbox_outputs.len()).min(kps_outputs.len());
        let mut raw_faces: Vec<FaceDetection> = Vec::new();

        for si in 0..num_strides {
            let scores = outputs[score_outputs[si]].try_extract_array::<f32>()?;
            let bboxes = outputs[bbox_outputs[si]].try_extract_array::<f32>()?;
            let kps = outputs[kps_outputs[si]].try_extract_array::<f32>()?;

            let stride = STRIDES[si];
            let feat_w = (INPUT_SIZE / stride) as usize;
            let num_anchors = scores.shape()[0];
            // Number of anchors per grid cell (typically 2 for SCRFD).
            let num_anchors_per_cell = num_anchors / (feat_w * feat_w).max(1);
            let num_anchors_per_cell = num_anchors_per_cell.max(1);

            // Output tensors are 2D [num_anchors, features] (no batch dim).
            for idx in 0..num_anchors {
                let score = scores[[idx, 0]];
                if score < self.config.detection_confidence {
                    continue;
                }

                let cell_idx = idx / num_anchors_per_cell;
                let anchor_cx = ((cell_idx % feat_w) as f32 + 0.5) * stride as f32;
                let anchor_cy = ((cell_idx / feat_w) as f32 + 0.5) * stride as f32;

                let dx1 = bboxes[[idx, 0]] * stride as f32;
                let dy1 = bboxes[[idx, 1]] * stride as f32;
                let dx2 = bboxes[[idx, 2]] * stride as f32;
                let dy2 = bboxes[[idx, 3]] * stride as f32;

                let bbox_letterbox = BBox::from_xyxy(
                    anchor_cx - dx1,
                    anchor_cy - dy1,
                    anchor_cx + dx2,
                    anchor_cy + dy2,
                );
                let bbox = unletterbox_bbox(&bbox_letterbox, &info)
                    .clamp(img_w as f32, img_h as f32);

                let mut landmarks = [Point2D::default(); 5];
                for k in 0..5 {
                    let kx = anchor_cx + kps[[idx, k * 2]] * stride as f32;
                    let ky = anchor_cy + kps[[idx, k * 2 + 1]] * stride as f32;
                    landmarks[k] = unletterbox_point(Point2D::new(kx, ky), &info);
                }

                raw_faces.push(FaceDetection {
                    bbox,
                    confidence: score,
                    landmarks,
                });
            }
        }

        // NMS.
        let dets: Vec<Detection> = raw_faces
            .iter()
            .map(|f| Detection {
                bbox: f.bbox,
                confidence: f.confidence,
                class_id: 0,
            })
            .collect();
        let keep = nms(&dets, 0.4);
        let result = keep.into_iter().map(|i| raw_faces[i].clone()).collect();
        Ok(result)
    }

    fn preprocess(&self, image: &DynamicImage, info: &LetterboxInfo) -> Array4<f32> {
        let rgb = image.to_rgb8();

        let new_w = (rgb.width() as f32 * info.scale).round() as u32;
        let new_h = (rgb.height() as f32 * info.scale).round() as u32;

        let resized = image::imageops::resize(
            &rgb,
            new_w,
            new_h,
            image::imageops::FilterType::Triangle,
        );

        let size = INPUT_SIZE as usize;
        let mut tensor = Array4::<f32>::zeros((1, 3, size, size));

        let x_off = info.pad_x.round() as usize;
        let y_off = info.pad_y.round() as usize;

        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let p = resized.get_pixel(x as u32, y as u32);
                tensor[[0, 0, y + y_off, x + x_off]] = (p[0] as f32 - 127.5) / 128.0;
                tensor[[0, 1, y + y_off, x + x_off]] = (p[1] as f32 - 127.5) / 128.0;
                tensor[[0, 2, y + y_off, x + x_off]] = (p[2] as f32 - 127.5) / 128.0;
            }
        }

        tensor
    }
}
