use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use ort::session::Session;

use crate::config::DetectionConfig;
use crate::geometry::{
    BBox, Detection, LetterboxInfo, letterbox_params, nms, unletterbox_bbox,
};
use crate::inference::engine::load_model;

/// COCO class id for "person".
const PERSON_CLASS: usize = 0;

pub struct YoloDetector {
    session: Session,
    config: DetectionConfig,
}

impl YoloDetector {
    pub fn new(model_path: &str, config: DetectionConfig) -> Result<Self> {
        let session = load_model(model_path)?;
        Ok(Self { session, config })
    }

    /// Run detection on a single image.  Returns person detections in original
    /// image coordinates.
    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<Detection>> {
        let size = self.config.input_size;
        let (img_w, img_h) = image.dimensions();
        let info = letterbox_params(img_w, img_h, size, size);

        // ---- pre-process: letterbox + normalize to [0,1] NCHW ----
        let input = self.preprocess(image, size, &info);
        let tensor = ort::value::Tensor::from_array(input)?;

        // ---- inference ----
        let outputs = self.session.run(ort::inputs![tensor])?;
        let output = outputs[0].try_extract_array::<f32>()?;

        // ---- post-process ----
        // YOLO11 ONNX output shape: [1, 4+num_classes, num_preds]
        let shape = output.shape();
        let num_classes = shape[1] - 4;
        let num_preds = shape[2];

        let mut raw_dets: Vec<Detection> = Vec::new();

        for i in 0..num_preds {
            let cx = output[[0, 0, i]];
            let cy = output[[0, 1, i]];
            let w = output[[0, 2, i]];
            let h = output[[0, 3, i]];

            // Find best class.
            let mut max_score: f32 = 0.0;
            let mut class_id: usize = 0;
            for c in 0..num_classes {
                let s = output[[0, 4 + c, i]];
                if s > max_score {
                    max_score = s;
                    class_id = c;
                }
            }

            if max_score < self.config.confidence {
                continue;
            }
            if self.config.person_only && class_id != PERSON_CLASS {
                continue;
            }

            let bbox = BBox::from_cxcywh(cx, cy, w, h);
            let bbox = unletterbox_bbox(&bbox, &info).clamp(img_w as f32, img_h as f32);

            raw_dets.push(Detection {
                bbox,
                confidence: max_score,
                class_id,
            });
        }

        // NMS
        let keep = nms(&raw_dets, self.config.nms_iou);
        let detections = keep.into_iter().map(|i| raw_dets[i].clone()).collect();
        Ok(detections)
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn preprocess(
        &self,
        image: &DynamicImage,
        size: u32,
        info: &LetterboxInfo,
    ) -> Array4<f32> {
        let rgb = image.to_rgb8();
        let (img_w, img_h) = (rgb.width(), rgb.height());

        let new_w = (img_w as f32 * info.scale).round() as u32;
        let new_h = (img_h as f32 * info.scale).round() as u32;

        let resized = image::imageops::resize(
            &rgb,
            new_w,
            new_h,
            image::imageops::FilterType::Triangle,
        );

        let pad_val = 114.0 / 255.0;
        let mut tensor = Array4::<f32>::from_elem((1, 3, size as usize, size as usize), pad_val);

        let x_offset = info.pad_x.round() as usize;
        let y_offset = info.pad_y.round() as usize;

        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                tensor[[0, 0, y + y_offset, x + x_offset]] = pixel[0] as f32 / 255.0;
                tensor[[0, 1, y + y_offset, x + x_offset]] = pixel[1] as f32 / 255.0;
                tensor[[0, 2, y + y_offset, x + x_offset]] = pixel[2] as f32 / 255.0;
            }
        }

        tensor
    }
}
