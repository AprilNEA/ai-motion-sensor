use std::collections::HashMap;

use crate::analysis::spatial::SpatialSignals;
use crate::config::{DoorZone, IntentConfig};
use crate::tracking::track::Track;

/// Per-track rolling intent score history.
pub struct ExitIntentScorer {
    config: IntentConfig,
    /// track_id → ring buffer of recent scores.
    history: HashMap<u64, Vec<f32>>,
}

/// Result of intent evaluation for a single track.
#[derive(Debug, Clone)]
pub struct IntentResult {
    pub track_id: u64,
    /// Raw intent score for the current frame (0..1).
    pub score: f32,
    /// Whether the sliding-window confirmation threshold is met.
    pub alert: bool,
    /// Which door zone triggered (if any).
    pub door_name: Option<String>,
}

impl ExitIntentScorer {
    pub fn new(config: IntentConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
        }
    }

    /// Score a single track against the closest / most relevant door zone.
    pub fn evaluate(
        &mut self,
        track: &Track,
        signals: &SpatialSignals,
        door: &DoorZone,
    ) -> IntentResult {
        let w = &self.config.weights;

        // ---- Weighted signal fusion ----
        let mut score = 0.0f32;
        let mut weight_sum = 0.0f32;

        // Direction toward door.
        score += w.direction * signals.direction_score.max(0.0);
        weight_sum += w.direction;

        // Distance decreasing + proximity.
        if signals.distance_decreasing {
            let proximity = (1.0 - signals.distance_to_door).max(0.0);
            score += w.distance * proximity;
        }
        weight_sum += w.distance;

        // Inside door zone.
        if signals.in_door_zone {
            score += w.in_zone;
        }
        weight_sum += w.in_zone;

        // Facing (placeholder – requires pose data, contributes 0 for now).
        weight_sum += w.facing;

        // Walking (heuristic: speed above a threshold).
        let walking_thresh = 0.005; // normalised units per frame
        if signals.speed > walking_thresh {
            score += w.walking;
        }
        weight_sum += w.walking;

        let score = if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.0
        };

        // ---- Rolling history ----
        let history = self.history.entry(track.id).or_default();
        history.push(score);
        // Keep only the last `confirm_frames` entries.
        let max_len = self.config.confirm_frames;
        if history.len() > max_len {
            let drain_count = history.len() - max_len;
            history.drain(..drain_count);
        }

        // ---- Sliding-window confirmation ----
        let alert = if history.len() >= max_len {
            let above = history
                .iter()
                .filter(|&&s| s > self.config.alert_threshold)
                .count();
            above as f32 / max_len as f32 >= self.config.confirm_ratio
        } else {
            false
        };

        IntentResult {
            track_id: track.id,
            score,
            alert,
            door_name: if alert {
                Some(door.name.clone())
            } else {
                None
            },
        }
    }

    /// Prune history for tracks that no longer exist.
    pub fn prune(&mut self, active_ids: &[u64]) {
        self.history.retain(|id, _| active_ids.contains(id));
    }
}
