use crate::config::DoorZone;
use crate::geometry::{Point2D, distance, dot2d, normalize_vec, point_in_polygon};
use crate::tracking::track::{Track, TrackPoint};

/// Spatial signals extracted from a track's trajectory relative to a door zone.
#[derive(Debug, Clone)]
pub struct SpatialSignals {
    /// Cosine similarity of recent movement direction vs. direction toward door.
    /// Range [-1, 1].  Positive = moving toward the door.
    pub direction_score: f32,

    /// Whether the distance to the door center is consistently decreasing over
    /// the recent window.
    pub distance_decreasing: bool,

    /// Current distance from person center to door center (normalised 0..1).
    pub distance_to_door: f32,

    /// Whether the person's bbox center is inside the door polygon.
    pub in_door_zone: bool,

    /// Instantaneous speed in normalised units per frame.
    pub speed: f32,
}

/// Number of recent points to use for direction estimation.
const DIRECTION_WINDOW: usize = 10;

/// Number of recent points to use for distance-decreasing check.
const DISTANCE_WINDOW: usize = 8;

/// Extract spatial signals from a track's trajectory w.r.t. a door zone.
///
/// Coordinates are assumed normalised to [0, 1] (the caller should normalise
/// by image width/height before appending to the trajectory).
pub fn extract_spatial_signals(track: &Track, door: &DoorZone) -> SpatialSignals {
    let traj = &track.trajectory;
    let n = traj.len();

    // Current position (latest point).
    let current = if n > 0 {
        traj.back().unwrap().center
    } else {
        Point2D::new(0.5, 0.5)
    };

    // ---- direction score ----
    let direction_score = if n >= 2 {
        let window = &traj_slice(traj, DIRECTION_WINDOW);
        let start = window.first().unwrap().center;
        let end = window.last().unwrap().center;
        let move_dir = normalize_vec(end.x - start.x, end.y - start.y);
        let to_door = normalize_vec(
            door.center.x - end.x,
            door.center.y - end.y,
        );
        dot2d(move_dir, to_door)
    } else {
        0.0
    };

    // ---- distance decreasing ----
    let distance_decreasing = if n >= 3 {
        let window = traj_slice(traj, DISTANCE_WINDOW);
        let distances: Vec<f32> = window
            .iter()
            .map(|p| distance(p.center, door.center))
            .collect();
        // Check if at least 70% of consecutive pairs are decreasing.
        let pairs = distances.windows(2).count();
        let decreasing = distances
            .windows(2)
            .filter(|w| w[1] < w[0])
            .count();
        pairs > 0 && (decreasing as f32 / pairs as f32) > 0.7
    } else {
        false
    };

    // ---- distance to door ----
    let distance_to_door = distance(current, door.center).min(1.0);

    // ---- in door zone ----
    let in_door_zone = point_in_polygon(current, &door.polygon);

    // ---- speed ----
    let speed = if n >= 2 {
        let prev = traj[n - 2].center;
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

/// Get the last `window` elements from the trajectory as a slice.
fn traj_slice(
    traj: &std::collections::VecDeque<TrackPoint>,
    window: usize,
) -> Vec<&TrackPoint> {
    let n = traj.len();
    let start = n.saturating_sub(window);
    (start..n).map(|i| &traj[i]).collect()
}
