use serde::Deserialize;

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

impl Point2D {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Axis-aligned bounding box in pixel coordinates (x1, y1 = top-left).
#[derive(Debug, Clone, Copy, Default)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub fn from_xyxy(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    pub fn from_cxcywh(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self {
            x1: cx - w / 2.0,
            y1: cy - h / 2.0,
            x2: cx + w / 2.0,
            y2: cy + h / 2.0,
        }
    }

    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn center(&self) -> Point2D {
        Point2D::new((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    pub fn aspect_ratio(&self) -> f32 {
        let h = self.height();
        if h < 1e-6 {
            return 0.0;
        }
        self.width() / h
    }

    /// Convert to [cx, cy, aspect_ratio, height] for Kalman filter.
    pub fn to_xyah(&self) -> [f32; 4] {
        let c = self.center();
        [c.x, c.y, self.aspect_ratio(), self.height()]
    }

    /// Construct from [cx, cy, aspect_ratio, height].
    pub fn from_xyah(cx: f32, cy: f32, a: f32, h: f32) -> Self {
        let w = a * h;
        Self::from_cxcywh(cx, cy, w, h)
    }

    /// Clamp coordinates to image bounds.
    pub fn clamp(&self, img_w: f32, img_h: f32) -> Self {
        Self {
            x1: self.x1.clamp(0.0, img_w),
            y1: self.y1.clamp(0.0, img_h),
            x2: self.x2.clamp(0.0, img_w),
            y2: self.y2.clamp(0.0, img_h),
        }
    }
}

// ---------------------------------------------------------------------------
// Detection result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox,
    pub confidence: f32,
    pub class_id: usize,
}

/// Face detection with 5 landmarks (left eye, right eye, nose, left mouth, right mouth).
#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub bbox: BBox,
    pub confidence: f32,
    pub landmarks: [Point2D; 5],
}

// ---------------------------------------------------------------------------
// Geometric helpers
// ---------------------------------------------------------------------------

/// Intersection-over-Union between two bounding boxes.
pub fn iou(a: &BBox, b: &BBox) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let union_area = a.area() + b.area() - inter_area;
    if union_area < 1e-6 {
        return 0.0;
    }
    inter_area / union_area
}

/// Non-Maximum Suppression. Returns indices of kept detections (sorted by
/// descending confidence).
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..detections.len()).collect();
    indices.sort_by(|&a, &b| {
        detections[b]
            .confidence
            .partial_cmp(&detections[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &indices {
            if j == i || suppressed[j] {
                continue;
            }
            if iou(&detections[i].bbox, &detections[j].bbox) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// Check whether a point lies inside a polygon (ray-casting algorithm).
pub fn point_in_polygon(p: Point2D, polygon: &[Point2D]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];
        if ((pi.y > p.y) != (pj.y > p.y))
            && (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Euclidean distance between two points.
pub fn distance(a: Point2D, b: Point2D) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// Normalise a 2D vector to unit length; returns zero vector on degenerate input.
pub fn normalize_vec(x: f32, y: f32) -> (f32, f32) {
    let len = (x * x + y * y).sqrt();
    if len < 1e-8 {
        return (0.0, 0.0);
    }
    (x / len, y / len)
}

/// Dot product of two 2D vectors.
pub fn dot2d(a: (f32, f32), b: (f32, f32)) -> f32 {
    a.0 * b.0 + a.1 * b.1
}

/// Cosine similarity between two equal-length slices.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    dot / (na * nb)
}

// ---------------------------------------------------------------------------
// Letterbox helpers (for model pre-processing)
// ---------------------------------------------------------------------------

/// Parameters for reversing a letterbox transform.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxInfo {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

/// Compute letterbox parameters for resizing `(src_w, src_h)` into
/// `(dst_w, dst_h)` while preserving aspect ratio.
pub fn letterbox_params(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> LetterboxInfo {
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round();
    let new_h = (src_h as f32 * scale).round();
    LetterboxInfo {
        scale,
        pad_x: (dst_w as f32 - new_w) / 2.0,
        pad_y: (dst_h as f32 - new_h) / 2.0,
    }
}

/// Map a point from letterboxed coordinates back to original image coordinates.
pub fn unletterbox_point(p: Point2D, info: &LetterboxInfo) -> Point2D {
    Point2D {
        x: (p.x - info.pad_x) / info.scale,
        y: (p.y - info.pad_y) / info.scale,
    }
}

/// Map a bbox from letterboxed coordinates back to original image coordinates.
pub fn unletterbox_bbox(b: &BBox, info: &LetterboxInfo) -> BBox {
    let tl = unletterbox_point(Point2D::new(b.x1, b.y1), info);
    let br = unletterbox_point(Point2D::new(b.x2, b.y2), info);
    BBox::from_xyxy(tl.x, tl.y, br.x, br.y)
}

// ---------------------------------------------------------------------------
// Polygon deserialization helper
// ---------------------------------------------------------------------------

/// Deserialize a polygon from a list of `[x, y]` pairs.
#[derive(Debug, Clone, Deserialize)]
#[serde(transparent)]
pub struct Polygon(pub Vec<[f32; 2]>);

impl Polygon {
    pub fn to_points(&self) -> Vec<Point2D> {
        self.0.iter().map(|p| Point2D::new(p[0], p[1])).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_identical() {
        let b = BBox::from_xyxy(0.0, 0.0, 10.0, 10.0);
        assert!((iou(&b, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = BBox::from_xyxy(0.0, 0.0, 5.0, 5.0);
        let b = BBox::from_xyxy(6.0, 6.0, 10.0, 10.0);
        assert!(iou(&a, &b) < 1e-5);
    }

    #[test]
    fn test_iou_partial() {
        let a = BBox::from_xyxy(0.0, 0.0, 10.0, 10.0);
        let b = BBox::from_xyxy(5.0, 5.0, 15.0, 15.0);
        // intersection = 5*5 = 25, union = 100+100-25 = 175
        let expected = 25.0 / 175.0;
        assert!((iou(&a, &b) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_nms() {
        let dets = vec![
            Detection {
                bbox: BBox::from_xyxy(0.0, 0.0, 10.0, 10.0),
                confidence: 0.9,
                class_id: 0,
            },
            Detection {
                bbox: BBox::from_xyxy(1.0, 1.0, 11.0, 11.0),
                confidence: 0.8,
                class_id: 0,
            },
            Detection {
                bbox: BBox::from_xyxy(50.0, 50.0, 60.0, 60.0),
                confidence: 0.7,
                class_id: 0,
            },
        ];
        let kept = nms(&dets, 0.5);
        // First two boxes overlap heavily → only keep index 0; third is independent.
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0], 0);
        assert_eq!(kept[1], 2);
    }

    #[test]
    fn test_point_in_polygon() {
        let poly = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(10.0, 10.0),
            Point2D::new(0.0, 10.0),
        ];
        assert!(point_in_polygon(Point2D::new(5.0, 5.0), &poly));
        assert!(!point_in_polygon(Point2D::new(15.0, 5.0), &poly));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }
}
