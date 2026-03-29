use serde::Deserialize;

use crate::geometry::{Point2D, Polygon};

// ---------------------------------------------------------------------------
// Top-level configuration (deserialized from TOML)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub models: ModelPaths,
    pub detection: DetectionConfig,
    pub face: FaceConfig,
    pub tracking: TrackingConfig,
    pub intent: IntentConfig,
    pub door_zones: Vec<DoorZoneConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelPaths {
    pub yolo_path: String,
    pub scrfd_path: String,
    pub arcface_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetectionConfig {
    pub confidence: f32,
    pub nms_iou: f32,
    pub person_only: bool,
    pub input_size: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FaceConfig {
    pub enabled: bool,
    pub detection_confidence: f32,
    pub match_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrackingConfig {
    pub high_thresh: f32,
    pub low_thresh: f32,
    pub new_track_thresh: f32,
    pub match_iou: f32,
    pub max_lost: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntentConfig {
    pub alert_threshold: f32,
    pub confirm_frames: usize,
    pub confirm_ratio: f32,
    pub trajectory_length: usize,
    pub weights: IntentWeights,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntentWeights {
    pub direction: f32,
    pub distance: f32,
    pub in_zone: f32,
    pub facing: f32,
    pub walking: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DoorZoneConfig {
    pub name: String,
    pub polygon: Polygon,
    pub direction: [f32; 2],
}

// ---------------------------------------------------------------------------
// Runtime door zone (pre-computed from config)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DoorZone {
    pub name: String,
    pub polygon: Vec<Point2D>,
    pub direction: (f32, f32),
    pub center: Point2D,
}

impl DoorZone {
    pub fn from_config(cfg: &DoorZoneConfig) -> Self {
        let points = cfg.polygon.to_points();
        let n = points.len() as f32;
        let cx = points.iter().map(|p| p.x).sum::<f32>() / n;
        let cy = points.iter().map(|p| p.y).sum::<f32>() / n;
        Self {
            name: cfg.name.clone(),
            polygon: points,
            direction: (cfg.direction[0], cfg.direction[1]),
            center: Point2D::new(cx, cy),
        }
    }
}

impl AppConfig {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn door_zones(&self) -> Vec<DoorZone> {
        self.door_zones.iter().map(DoorZone::from_config).collect()
    }
}
