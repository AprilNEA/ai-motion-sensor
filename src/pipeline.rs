use anyhow::Result;
use image::{DynamicImage, GenericImageView};

use std::collections::HashMap;
use std::sync::Arc;

use crate::analysis::face_db::FaceDatabase;
use crate::analysis::intent::{ExitIntentScorer, IntentResult};
use crate::analysis::spatial::SpatialSignals;
use crate::config::{AppConfig, DoorControlConfig, DoorZone};
use crate::door::DoorController;
use crate::door::unifi_access::UnifiAccessController;
use crate::geometry::{Point2D, distance, dot2d, normalize_vec, point_in_polygon};
use crate::inference::arcface::ArcFaceExtractor;
use crate::inference::scrfd::ScrfdDetector;
use crate::inference::yolo::YoloDetector;
use crate::tracking::byte_track::ByteTracker;
use crate::tracking::track::{Track, TrackState};
use crate::video::source::FrameSource;

/// Central processing pipeline.
pub struct Pipeline {
    yolo: YoloDetector,
    scrfd: Option<ScrfdDetector>,
    arcface: Option<ArcFaceExtractor>,
    tracker: ByteTracker,
    scorer: ExitIntentScorer,
    door_zones: Vec<DoorZone>,
    face_db: FaceDatabase,
    frame_count: u64,
    detect_interval: u64,
    max_input_dim: u32,
    fps: f64,
    /// Door controller (Arc for thread-safe sharing) + zone-name-to-door-name mapping.
    door_ctrl: Option<(Arc<dyn DoorController + Sync>, HashMap<String, String>)>,
}

/// Summary returned after processing a single frame.
#[derive(Debug)]
pub struct FrameResult {
    pub frame_id: u64,
    pub num_persons: usize,
    pub alerts: Vec<IntentResult>,
}

impl Pipeline {
    pub fn new(config: AppConfig) -> Result<Self> {
        tracing::info!("initialising pipeline");

        let yolo = YoloDetector::new(&config.models.yolo_path, config.detection.clone())?;

        let (scrfd, arcface) = if config.face.enabled {
            let s = ScrfdDetector::new(&config.models.scrfd_path, config.face.clone())?;
            let a = ArcFaceExtractor::new(&config.models.arcface_path)?;
            (Some(s), Some(a))
        } else {
            (None, None)
        };

        let tracker = ByteTracker::new(config.tracking.clone());
        let scorer = ExitIntentScorer::new(config.intent.clone());
        let door_zones = config.door_zones();
        let face_db = FaceDatabase::new(config.face.match_threshold);
        let detect_interval = config.detection.detect_interval.max(1);
        let max_input_dim = config.detection.max_input_dim;

        // Door controller.
        let door_ctrl: Option<(Arc<dyn DoorController + Sync>, HashMap<String, String>)> =
            match &config.door_control {
                Some(DoorControlConfig::UnifiAccess {
                    host,
                    token,
                    door_name_map,
                }) => {
                    match UnifiAccessController::new(host, token) {
                        Ok(ctrl) => {
                            tracing::info!(host, "UniFi Access door controller connected");
                            Some((Arc::new(ctrl), door_name_map.clone()))
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "failed to connect door controller, continuing without");
                            None
                        }
                    }
                }
                None => None,
            };

        tracing::info!(
            doors = door_zones.len(),
            face_enabled = config.face.enabled,
            detect_interval,
            max_input_dim,
            door_control = door_ctrl.is_some(),
            "pipeline ready"
        );

        Ok(Self {
            yolo,
            scrfd,
            arcface,
            tracker,
            scorer,
            door_zones,
            face_db,
            frame_count: 0,
            detect_interval,
            max_input_dim,
            fps: 30.0, // updated in run()
            door_ctrl,
        })
    }

    /// Process a single frame through the full pipeline.
    pub fn process_frame(&mut self, image: &DynamicImage) -> Result<FrameResult> {
        self.frame_count += 1;
        let timestamp = self.frame_count as f64;
        let (img_w, img_h) = image.dimensions();

        // ---- 0. Downscale if configured ----
        let image = if self.max_input_dim > 0 {
            let max_dim = img_w.max(img_h);
            if max_dim > self.max_input_dim {
                let scale = self.max_input_dim as f32 / max_dim as f32;
                let new_w = (img_w as f32 * scale) as u32;
                let new_h = (img_h as f32 * scale) as u32;
                image.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle)
            } else {
                image.clone()
            }
        } else {
            image.clone()
        };
        let (proc_w, proc_h) = image.dimensions();

        // ---- 1. Detection (every N frames) or prediction-only ----
        let is_detect_frame = self.frame_count % self.detect_interval == 0
            || self.frame_count <= 2; // always detect first frames

        let detections = if is_detect_frame {
            self.yolo.detect(&image)?
        } else {
            Vec::new()
        };

        // ---- 2. Tracking ----
        if is_detect_frame {
            self.tracker.update(&detections, timestamp);
        } else {
            // Predict-only: advance Kalman filters without new measurements.
            self.tracker.predict_only();
        }

        // ---- 3. Optional face recognition (on detect frames only) ----
        if is_detect_frame && self.frame_count % (self.detect_interval * 5) == 0 {
            self.run_face_recognition(&image, proc_w, proc_h);
        }

        // ---- 4 & 5. Spatial analysis + intent scoring ----
        let alerts = self.run_intent_analysis(proc_w, proc_h);

        // Prune stale history.
        let active_ids: Vec<u64> = self.tracker.active_tracks().map(|t| t.id).collect();
        self.scorer.prune(&active_ids);

        let num_persons = self.tracker.active_tracks().count();

        Ok(FrameResult {
            frame_id: self.frame_count,
            num_persons,
            alerts,
        })
    }

    /// Mutable access to the face database (for registration).
    pub fn face_db_mut(&mut self) -> &mut FaceDatabase {
        &mut self.face_db
    }

    /// Run the full pipeline on a video source until it is exhausted.
    pub fn run(&mut self, source: &mut dyn FrameSource) -> Result<()> {
        self.fps = source.fps();
        tracing::info!(fps = self.fps, "starting pipeline loop");

        loop {
            let frame = source.next_frame()?;
            let Some(image) = frame else {
                tracing::info!("video source exhausted");
                break;
            };
            let result = self.process_frame(&image)?;
            if !result.alerts.is_empty() {
                for alert in &result.alerts {
                    if alert.alert {
                        tracing::warn!(
                            frame = result.frame_id,
                            track = alert.track_id,
                            score = format!("{:.2}", alert.score),
                            door = ?alert.door_name,
                            "EXIT INTENT"
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn run_face_recognition(&mut self, image: &DynamicImage, img_w: u32, img_h: u32) {
        let (Some(scrfd), Some(arcface)) = (&mut self.scrfd, &mut self.arcface) else {
            return;
        };

        let faces = match scrfd.detect(image) {
            Ok(f) => f,
            Err(_) => return,
        };

        for face in &faces {
            let embedding = match arcface.extract(image, face) {
                Ok(e) => e,
                Err(_) => continue,
            };

            let matched = self.face_db.search(&embedding);

            let face_center = face.bbox.center();
            let face_norm = Point2D::new(
                face_center.x / img_w as f32,
                face_center.y / img_h as f32,
            );

            let mut best_tid = None;
            let mut best_dist = f32::MAX;
            for track in self.tracker.active_tracks() {
                let tc = track.predicted_bbox().center();
                let tc_norm = Point2D::new(tc.x / img_w as f32, tc.y / img_h as f32);
                let d = crate::geometry::distance(face_norm, tc_norm);
                if d < best_dist {
                    best_dist = d;
                    best_tid = Some(track.id);
                }
            }

            if let Some(tid) = best_tid {
                if best_dist < 0.15 {
                    if let Some(track) = self.tracker.get_track_mut(tid) {
                        track.face_embedding = Some(embedding);
                        if let Some(m) = &matched {
                            track.identity = Some(m.identity.clone());
                            tracing::info!(
                                track_id = tid,
                                identity = %m.identity,
                                similarity = m.similarity,
                                "face recognised"
                            );
                        }
                    }
                }
            }
        }
    }

    fn run_intent_analysis(&mut self, img_w: u32, img_h: u32) -> Vec<IntentResult> {
        let mut alerts = Vec::new();

        let track_snapshot: Vec<(u64, TrackState)> = self
            .tracker
            .active_tracks()
            .map(|t| (t.id, t.state))
            .collect();

        let door_zones = self.door_zones.clone();
        let fps = self.fps;
        let frame = self.frame_count;

        for door in &door_zones {
            for &(track_id, state) in &track_snapshot {
                if state != TrackState::Active {
                    continue;
                }
                if let Some(track) = self.tracker.get_track_mut(track_id) {
                    let signals = extract_signals_normalised(track, door, img_w, img_h);
                    let result = self.scorer.evaluate(track, &signals, door, frame, fps);

                    if result.alert {
                        let identity =
                            track.identity.clone().unwrap_or_else(|| "unknown".into());
                        tracing::warn!(
                            track_id,
                            identity,
                            score = result.score,
                            door = door.name,
                            "EXIT INTENT DETECTED"
                        );

                        // Trigger door unlock in background thread (non-blocking).
                        if let Some((ctrl, name_map)) = &self.door_ctrl {
                            let door_name = name_map
                                .get(&door.name)
                                .cloned()
                                .unwrap_or_else(|| door.name.clone());
                            let ctrl = Arc::clone(ctrl);
                            std::thread::spawn(move || {
                                if let Err(e) = ctrl.unlock(&door_name) {
                                    tracing::error!(
                                        door = %door_name,
                                        error = %e,
                                        "door unlock failed"
                                    );
                                }
                            });
                        }

                        alerts.push(result);
                    }
                }
            }
        }

        alerts
    }
}

// ---------------------------------------------------------------------------
// Helper: extract spatial signals with on-the-fly normalisation
// ---------------------------------------------------------------------------

fn extract_signals_normalised(
    track: &Track,
    door: &DoorZone,
    img_w: u32,
    img_h: u32,
) -> SpatialSignals {
    let traj = &track.trajectory;
    let n = traj.len();

    let norm = |p: Point2D| -> Point2D {
        Point2D::new(p.x / img_w as f32, p.y / img_h as f32)
    };

    let current = if n > 0 {
        norm(traj.back().unwrap().center)
    } else {
        Point2D::new(0.5, 0.5)
    };

    let direction_score = if n >= 2 {
        let w = 10.min(n);
        let start = norm(traj[n - w].center);
        let end = current;
        let move_dir = normalize_vec(end.x - start.x, end.y - start.y);
        let to_door = normalize_vec(door.center.x - end.x, door.center.y - end.y);
        dot2d(move_dir, to_door)
    } else {
        0.0
    };

    let distance_decreasing = if n >= 3 {
        let w = 8.min(n);
        let dists: Vec<f32> = (n - w..n)
            .map(|i| distance(norm(traj[i].center), door.center))
            .collect();
        let pairs = dists.windows(2).count();
        let dec = dists.windows(2).filter(|p| p[1] < p[0]).count();
        pairs > 0 && (dec as f32 / pairs as f32) > 0.7
    } else {
        false
    };

    let distance_to_door = distance(current, door.center).min(1.0);
    let in_door_zone = point_in_polygon(current, &door.polygon);

    let speed = if n >= 2 {
        let prev = norm(traj[n - 2].center);
        distance(prev, current)
    } else {
        0.0
    };

    SpatialSignals {
        direction_score,
        distance_decreasing,
        distance_to_door,
        in_door_zone,
        speed,
    }
}
