use std::collections::HashMap;

use crate::analysis::spatial::SpatialSignals;
use crate::config::{DoorZone, IntentConfig};
use crate::tracking::track::Track;

/// Per-track rolling intent score history.
pub struct ExitIntentScorer {
    config: IntentConfig,
    /// track_id → per-track state.
    state: HashMap<u64, TrackIntentState>,
}

struct TrackIntentState {
    /// Rolling score history.
    history: Vec<f32>,
    /// Timestamp (frame count) of the last fired alert.
    last_alert_frame: Option<u64>,
    /// Whether the alert is currently in cooldown.
    in_cooldown: bool,
}

/// Result of intent evaluation for a single track.
#[derive(Debug, Clone)]
pub struct IntentResult {
    pub track_id: u64,
    /// Raw intent score for the current frame (0..1).
    pub score: f32,
    /// Whether a NEW alert should be emitted this frame.
    pub alert: bool,
    /// Which door zone triggered (if any).
    pub door_name: Option<String>,
}

impl ExitIntentScorer {
    pub fn new(config: IntentConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
        }
    }

    /// Score a single track against the closest / most relevant door zone.
    pub fn evaluate(
        &mut self,
        track: &Track,
        signals: &SpatialSignals,
        door: &DoorZone,
        current_frame: u64,
        fps: f64,
    ) -> IntentResult {
        let w = &self.config.weights;

        // ---- Weighted signal fusion ----
        let mut score = 0.0f32;
        let mut weight_sum = 0.0f32;

        score += w.direction * signals.direction_score.max(0.0);
        weight_sum += w.direction;

        if signals.distance_decreasing {
            let proximity = (1.0 - signals.distance_to_door).max(0.0);
            score += w.distance * proximity;
        }
        weight_sum += w.distance;

        if signals.in_door_zone {
            score += w.in_zone;
        }
        weight_sum += w.in_zone;

        // Facing (placeholder – requires pose data).
        weight_sum += w.facing;

        let walking_thresh = 0.005;
        if signals.speed > walking_thresh {
            score += w.walking;
        }
        weight_sum += w.walking;

        let score = if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.0
        };

        // ---- Per-track state ----
        let ts = self.state.entry(track.id).or_insert_with(|| TrackIntentState {
            history: Vec::new(),
            last_alert_frame: None,
            in_cooldown: false,
        });

        ts.history.push(score);
        let max_len = self.config.confirm_frames;
        if ts.history.len() > max_len {
            let drain = ts.history.len() - max_len;
            ts.history.drain(..drain);
        }

        // ---- Sliding-window confirmation ----
        let window_met = if ts.history.len() >= max_len {
            let above = ts
                .history
                .iter()
                .filter(|&&s| s > self.config.alert_threshold)
                .count();
            above as f32 / max_len as f32 >= self.config.confirm_ratio
        } else {
            false
        };

        // ---- Debounce: cooldown + hysteresis ----
        let cooldown_frames = (self.config.cooldown_secs * fps).max(1.0) as u64;

        // Check if cooldown has expired.
        if ts.in_cooldown {
            if let Some(last) = ts.last_alert_frame {
                let elapsed = current_frame.saturating_sub(last);
                if elapsed >= cooldown_frames && score < self.config.rearm_threshold {
                    // Cooldown expired AND score dropped low enough → re-arm.
                    ts.in_cooldown = false;
                }
            }
        }

        let alert = window_met && !ts.in_cooldown;

        if alert {
            ts.last_alert_frame = Some(current_frame);
            ts.in_cooldown = true;
        }

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

    /// Prune state for tracks that no longer exist.
    pub fn prune(&mut self, active_ids: &[u64]) {
        self.state.retain(|id, _| active_ids.contains(id));
    }
}
