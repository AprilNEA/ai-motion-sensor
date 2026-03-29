use crate::config::TrackingConfig;
use crate::geometry::{Detection, iou};
use crate::tracking::kalman::KalmanFilter;
use crate::tracking::track::{Track, TrackState};

/// ByteTrack multi-object tracker.
///
/// Two-stage association:
///   1. Match high-confidence detections with existing tracks (IoU).
///   2. Match remaining tracks with low-confidence detections (IoU).
///   3. Create new tracks from unmatched high-confidence detections.
pub struct ByteTracker {
    kf: KalmanFilter,
    tracks: Vec<Track>,
    config: TrackingConfig,
    frame_id: u64,
}

impl ByteTracker {
    pub fn new(config: TrackingConfig) -> Self {
        Self {
            kf: KalmanFilter::new(),
            tracks: Vec::new(),
            config,
            frame_id: 0,
        }
    }

    /// Process one frame of detections.  Returns a reference to all current
    /// tracks (including lost ones that haven't been pruned yet).
    pub fn update(&mut self, detections: &[Detection], timestamp: f64) -> &[Track] {
        self.frame_id += 1;

        // Predict all existing tracks forward.
        for track in &mut self.tracks {
            track.predict(&self.kf);
        }

        // Split detections by confidence.
        let mut high_dets: Vec<(usize, &Detection)> = Vec::new();
        let mut low_dets: Vec<(usize, &Detection)> = Vec::new();
        for (i, det) in detections.iter().enumerate() {
            if det.confidence >= self.config.high_thresh {
                high_dets.push((i, det));
            } else if det.confidence >= self.config.low_thresh {
                low_dets.push((i, det));
            }
        }

        // Separate confirmed and unconfirmed tracks.
        let mut confirmed_indices: Vec<usize> = Vec::new();
        let mut unconfirmed_indices: Vec<usize> = Vec::new();
        for (i, track) in self.tracks.iter().enumerate() {
            if track.state == TrackState::Active || track.state == TrackState::Lost {
                confirmed_indices.push(i);
            } else {
                unconfirmed_indices.push(i);
            }
        }

        // ---- First association: high-confidence dets ↔ confirmed tracks ----
        let (matched_1, unmatched_tracks_1, unmatched_dets_1) = self.associate(
            &confirmed_indices,
            &high_dets,
            self.config.match_iou,
        );

        for (track_idx, det_idx) in &matched_1 {
            self.tracks[*track_idx].update(detections.get(*det_idx).unwrap(), &self.kf, timestamp);
        }

        // ---- Second association: remaining tracks ↔ low-confidence dets ----
        let remaining_tracks: Vec<usize> = unmatched_tracks_1
            .iter()
            .filter(|&&i| self.tracks[i].state == TrackState::Active)
            .copied()
            .collect();

        let (matched_2, unmatched_tracks_2, _) = self.associate(
            &remaining_tracks,
            &low_dets,
            self.config.match_iou,
        );

        for (track_idx, det_idx) in &matched_2 {
            self.tracks[*track_idx].update(detections.get(*det_idx).unwrap(), &self.kf, timestamp);
        }

        // ---- Third association: unconfirmed tracks ↔ unmatched high dets ----
        let unmatched_high_as_dets: Vec<(usize, &Detection)> = unmatched_dets_1
            .iter()
            .map(|&i| high_dets[i])
            .collect();
        let (matched_3, unmatched_unconf, unmatched_high_final) = self.associate(
            &unconfirmed_indices,
            &unmatched_high_as_dets,
            self.config.match_iou,
        );

        for (track_idx, det_idx) in &matched_3 {
            let original_det_idx = unmatched_high_as_dets[*det_idx].0;
            self.tracks[*track_idx].update(&detections[original_det_idx], &self.kf, timestamp);
        }

        // Mark unmatched tracks as lost.
        for &i in &unmatched_tracks_2 {
            self.tracks[i].mark_lost();
        }
        for &i in &unmatched_unconf {
            self.tracks[i].mark_lost();
        }
        // Also mark Lost from first association (those that were already Lost).
        for &i in &unmatched_tracks_1 {
            if !remaining_tracks.contains(&i) {
                self.tracks[i].mark_lost();
            }
        }

        // ---- Create new tracks from remaining high-confidence detections ----
        let traj_len = 90; // will be overridden by pipeline config
        for &local_idx in &unmatched_high_final {
            let original_det_idx = unmatched_high_as_dets[local_idx].0;
            let det = &detections[original_det_idx];
            if det.confidence >= self.config.new_track_thresh {
                let track = Track::new(det, &self.kf, timestamp, traj_len);
                self.tracks.push(track);
            }
        }

        // ---- Prune dead tracks ----
        self.tracks
            .retain(|t| !t.should_remove(self.config.max_lost));

        &self.tracks
    }

    /// Get all active tracks (Active state only).
    pub fn active_tracks(&self) -> impl Iterator<Item = &Track> {
        self.tracks
            .iter()
            .filter(|t| t.state == TrackState::Active)
    }

    /// Get mutable reference to a track by id.
    pub fn get_track_mut(&mut self, id: u64) -> Option<&mut Track> {
        self.tracks.iter_mut().find(|t| t.id == id)
    }

    // -----------------------------------------------------------------------
    // Greedy IoU-based association
    // -----------------------------------------------------------------------

    /// Returns (matched_pairs, unmatched_track_indices, unmatched_det_indices).
    /// Each matched pair is (track_index_in_original_tracks, det_local_index).
    fn associate(
        &self,
        track_indices: &[usize],
        dets: &[(usize, &Detection)],
        iou_thresh: f32,
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        if track_indices.is_empty() || dets.is_empty() {
            return (
                Vec::new(),
                track_indices.to_vec(),
                (0..dets.len()).collect(),
            );
        }

        // Compute IoU cost matrix.
        let num_tracks = track_indices.len();
        let num_dets = dets.len();
        let mut cost = vec![vec![0.0f32; num_dets]; num_tracks];

        for (ti, &track_idx) in track_indices.iter().enumerate() {
            let track_bbox = self.tracks[track_idx].predicted_bbox();
            for (di, (_, det)) in dets.iter().enumerate() {
                cost[ti][di] = iou(&track_bbox, &det.bbox);
            }
        }

        // Greedy matching (highest IoU first).
        let mut matched = Vec::new();
        let mut used_tracks = vec![false; num_tracks];
        let mut used_dets = vec![false; num_dets];

        // Collect all (iou, track_local_idx, det_local_idx) and sort descending.
        let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
        for ti in 0..num_tracks {
            for di in 0..num_dets {
                if cost[ti][di] > 1.0 - iou_thresh {
                    candidates.push((cost[ti][di], ti, di));
                }
            }
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for (_, ti, di) in candidates {
            if used_tracks[ti] || used_dets[di] {
                continue;
            }
            matched.push((track_indices[ti], di));
            used_tracks[ti] = true;
            used_dets[di] = true;
        }

        let unmatched_tracks: Vec<usize> = track_indices
            .iter()
            .enumerate()
            .filter(|(ti, _)| !used_tracks[*ti])
            .map(|(_, &idx)| idx)
            .collect();

        let unmatched_dets: Vec<usize> = (0..num_dets)
            .filter(|di| !used_dets[*di])
            .collect();

        (matched, unmatched_tracks, unmatched_dets)
    }
}
