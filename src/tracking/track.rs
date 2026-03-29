use std::collections::VecDeque;

use crate::geometry::{BBox, Detection, Point2D};
use crate::tracking::kalman::{KalmanFilter, KalmanState};

/// Lifecycle state of a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    /// Newly created, not yet confirmed.
    Tentative,
    /// Actively tracked.
    Active,
    /// Lost (not matched for some frames but not yet removed).
    Lost,
}

/// A single tracked object.
pub struct Track {
    pub id: u64,
    pub state: TrackState,
    pub kalman: KalmanState,

    /// Number of consecutive frames without a match.
    pub time_since_update: usize,
    /// Total number of successful matches.
    pub hits: usize,

    /// Most recent matched detection confidence.
    pub confidence: f32,
    /// Class id from the last matched detection.
    pub class_id: usize,

    /// Position history (center points in original image coords).
    pub trajectory: VecDeque<TrackPoint>,
    /// Maximum trajectory length.
    max_traj_len: usize,

    /// Optional face embedding (set when a face is recognised).
    pub face_embedding: Option<Vec<f32>>,
    /// Optional identity label.
    pub identity: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TrackPoint {
    pub center: Point2D,
    pub timestamp: f64,
}

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Track {
    /// Create a new track from a detection.
    pub fn new(
        det: &Detection,
        kf: &KalmanFilter,
        timestamp: f64,
        max_traj_len: usize,
    ) -> Self {
        let xyah = det.bbox.to_xyah();
        let kalman = kf.initiate(xyah);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let center = det.bbox.center();
        let mut trajectory = VecDeque::with_capacity(max_traj_len);
        trajectory.push_back(TrackPoint {
            center,
            timestamp,
        });

        Self {
            id,
            state: TrackState::Tentative,
            kalman,
            time_since_update: 0,
            hits: 1,
            confidence: det.confidence,
            class_id: det.class_id,
            trajectory,
            max_traj_len,
            face_embedding: None,
            identity: None,
        }
    }

    /// Predict one step forward.
    pub fn predict(&mut self, kf: &KalmanFilter) {
        kf.predict(&mut self.kalman);
        self.time_since_update += 1;
    }

    /// Update the track with a matched detection.
    pub fn update(
        &mut self,
        det: &Detection,
        kf: &KalmanFilter,
        timestamp: f64,
    ) {
        let xyah = det.bbox.to_xyah();
        kf.update(&mut self.kalman, xyah);

        self.time_since_update = 0;
        self.hits += 1;
        self.confidence = det.confidence;
        self.class_id = det.class_id;

        // Promote tentative → active after enough hits.
        if self.state == TrackState::Tentative && self.hits >= 3 {
            self.state = TrackState::Active;
        }
        if self.state == TrackState::Lost {
            self.state = TrackState::Active;
        }

        // Append to trajectory.
        let center = det.bbox.center();
        self.trajectory.push_back(TrackPoint {
            center,
            timestamp,
        });
        while self.trajectory.len() > self.max_traj_len {
            self.trajectory.pop_front();
        }
    }

    /// Mark as lost.
    pub fn mark_lost(&mut self) {
        if self.state != TrackState::Tentative {
            self.state = TrackState::Lost;
        }
    }

    /// Whether this track should be removed (lost for too long, or tentative
    /// that was never confirmed).
    pub fn should_remove(&self, max_lost: usize) -> bool {
        match self.state {
            TrackState::Tentative => self.time_since_update > 2,
            TrackState::Lost => self.time_since_update > max_lost,
            TrackState::Active => false,
        }
    }

    /// Get the current predicted bounding box.
    pub fn predicted_bbox(&self) -> BBox {
        let x = &self.kalman.x;
        BBox::from_xyah(x[0], x[1], x[2], x[3])
    }
}
