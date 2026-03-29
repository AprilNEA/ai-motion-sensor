use anyhow::Result;
use image::{DynamicImage, RgbImage};
use nalgebra::{Matrix2, Matrix2x3, Vector2};
use ndarray::Array4;
use ort::session::Session;

use crate::geometry::{FaceDetection, Point2D};
use crate::inference::engine::load_model;

const ARCFACE_SIZE: u32 = 112;

/// ArcFace standard reference landmarks for 112x112 aligned face.
const REF_LANDMARKS: [[f32; 2]; 5] = [
    [38.2946, 51.6963],  // left eye
    [73.5318, 51.5014],  // right eye
    [56.0252, 71.7366],  // nose tip
    [41.5493, 92.3655],  // left mouth corner
    [70.7299, 92.2041],  // right mouth corner
];

pub struct ArcFaceExtractor {
    session: Session,
}

impl ArcFaceExtractor {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = load_model(model_path)?;
        Ok(Self { session })
    }

    /// Extract a 512-dimensional face embedding.
    pub fn extract(
        &mut self,
        image: &DynamicImage,
        face: &FaceDetection,
    ) -> Result<Vec<f32>> {
        let aligned = align_face(image, &face.landmarks);
        let input = self.preprocess(&aligned);
        let tensor = ort::value::Tensor::from_array(input)?;

        let outputs = self.session.run(ort::inputs![tensor])?;
        let embedding = outputs[0].try_extract_array::<f32>()?;

        // L2-normalise the embedding.
        let raw: Vec<f32> = embedding.iter().copied().collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-8 {
            return Ok(raw);
        }
        Ok(raw.iter().map(|x| x / norm).collect())
    }

    fn preprocess(&self, face: &RgbImage) -> Array4<f32> {
        let s = ARCFACE_SIZE as usize;
        let mut tensor = Array4::<f32>::zeros((1, 3, s, s));
        for y in 0..s {
            for x in 0..s {
                let p = face.get_pixel(x as u32, y as u32);
                tensor[[0, 0, y, x]] = (p[0] as f32 - 127.5) / 127.5;
                tensor[[0, 1, y, x]] = (p[1] as f32 - 127.5) / 127.5;
                tensor[[0, 2, y, x]] = (p[2] as f32 - 127.5) / 127.5;
            }
        }
        tensor
    }
}

// ---------------------------------------------------------------------------
// Face alignment via similarity transform
// ---------------------------------------------------------------------------

fn align_face(image: &DynamicImage, landmarks: &[Point2D; 5]) -> RgbImage {
    let src: Vec<Vector2<f32>> = landmarks
        .iter()
        .map(|p| Vector2::new(p.x, p.y))
        .collect();
    let dst: Vec<Vector2<f32>> = REF_LANDMARKS
        .iter()
        .map(|p| Vector2::new(p[0], p[1]))
        .collect();

    let m = estimate_similarity_transform(&src, &dst);
    let rgb = image.to_rgb8();
    let size = ARCFACE_SIZE;
    let mut aligned = RgbImage::new(size, size);

    for out_y in 0..size {
        for out_x in 0..size {
            let src_pt = affine_inverse_map(&m, out_x as f32, out_y as f32);
            let sx = src_pt.x as i32;
            let sy = src_pt.y as i32;
            if sx >= 0 && sy >= 0 && (sx as u32) < rgb.width() && (sy as u32) < rgb.height() {
                aligned.put_pixel(out_x, out_y, *rgb.get_pixel(sx as u32, sy as u32));
            }
        }
    }
    aligned
}

fn estimate_similarity_transform(
    src: &[Vector2<f32>],
    dst: &[Vector2<f32>],
) -> Matrix2x3<f32> {
    let n = src.len() as f32;
    let src_mean: Vector2<f32> = src.iter().sum::<Vector2<f32>>() / n;
    let dst_mean: Vector2<f32> = dst.iter().sum::<Vector2<f32>>() / n;

    let src_c: Vec<Vector2<f32>> = src.iter().map(|p| p - src_mean).collect();
    let dst_c: Vec<Vector2<f32>> = dst.iter().map(|p| p - dst_mean).collect();

    let mut cov = Matrix2::<f32>::zeros();
    for (s, d) in src_c.iter().zip(dst_c.iter()) {
        cov += d * s.transpose();
    }
    cov /= n;

    let svd = cov.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut d_sign = Matrix2::identity();
    if (u * vt).determinant() < 0.0 {
        d_sign[(1, 1)] = -1.0;
    }
    let r = u * d_sign * vt;

    let src_var: f32 = src_c.iter().map(|p| p.dot(p)).sum::<f32>() / n;
    let scale = if src_var.abs() < 1e-8 {
        1.0
    } else {
        svd.singular_values.sum() / src_var
    };

    let t = dst_mean - scale * (r * src_mean);

    Matrix2x3::new(
        scale * r[(0, 0)],
        scale * r[(0, 1)],
        t.x,
        scale * r[(1, 0)],
        scale * r[(1, 1)],
        t.y,
    )
}

fn affine_inverse_map(m: &Matrix2x3<f32>, x: f32, y: f32) -> Point2D {
    let a = Matrix2::new(m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]);
    let t = Vector2::new(m[(0, 2)], m[(1, 2)]);
    let p = Vector2::new(x, y);
    let inv_a = a.try_inverse().unwrap_or(Matrix2::identity());
    let src = inv_a * (p - t);
    Point2D::new(src.x, src.y)
}
