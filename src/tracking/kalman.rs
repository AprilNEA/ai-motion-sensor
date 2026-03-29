/// Linear Kalman filter for object tracking.
///
/// State vector (8D):  [cx, cy, aspect_ratio, height, v_cx, v_cy, v_a, v_h]
/// Measurement (4D):   [cx, cy, aspect_ratio, height]
///
/// Uses nalgebra fixed-size matrices for zero-allocation, inlineable math.
use nalgebra::{SMatrix, SVector};

const N: usize = 8; // state dimension
const M: usize = 4; // measurement dimension

type StateVec = SVector<f32, N>;
type StateMat = SMatrix<f32, N, N>;
type MeasVec = SVector<f32, M>;
type MeasMat = SMatrix<f32, M, M>;
type ObsMat = SMatrix<f32, M, N>;
type KalmanGain = SMatrix<f32, N, M>;

/// Standard deviation multipliers for the process noise.
const STD_WEIGHT_POSITION: f32 = 1.0 / 20.0;
const STD_WEIGHT_VELOCITY: f32 = 1.0 / 160.0;

pub struct KalmanFilter {
    /// State transition matrix (constant velocity model).
    f: StateMat,
    /// Observation matrix.
    h: ObsMat,
}

#[derive(Clone)]
pub struct KalmanState {
    pub x: StateVec,
    pub p: StateMat,
}

impl KalmanFilter {
    pub fn new() -> Self {
        // F = [[I_4, I_4], [0, I_4]]  (constant velocity)
        let mut f = StateMat::identity();
        for i in 0..4 {
            f[(i, i + 4)] = 1.0;
        }

        // H = [I_4, 0_4]
        let mut h = ObsMat::zeros();
        for i in 0..4 {
            h[(i, i)] = 1.0;
        }

        Self { f, h }
    }

    /// Initialise a new track from a measurement [cx, cy, a, h].
    pub fn initiate(&self, measurement: [f32; 4]) -> KalmanState {
        let mut x = StateVec::zeros();
        for i in 0..4 {
            x[i] = measurement[i];
        }
        // Velocity is initialised to zero.

        // Initial covariance: large uncertainty on velocity.
        let h = measurement[3];
        let std_pos = [
            2.0 * STD_WEIGHT_POSITION * h,
            2.0 * STD_WEIGHT_POSITION * h,
            1e-2,
            2.0 * STD_WEIGHT_POSITION * h,
        ];
        let std_vel = [
            10.0 * STD_WEIGHT_VELOCITY * h,
            10.0 * STD_WEIGHT_VELOCITY * h,
            1e-5,
            10.0 * STD_WEIGHT_VELOCITY * h,
        ];
        let mut diag = [0.0f32; N];
        for i in 0..4 {
            diag[i] = std_pos[i] * std_pos[i];
        }
        for i in 0..4 {
            diag[i + 4] = std_vel[i] * std_vel[i];
        }
        let p = StateMat::from_diagonal(&StateVec::from_column_slice(&diag));

        KalmanState { x, p }
    }

    /// Predict the next state (one timestep forward).
    pub fn predict(&self, state: &mut KalmanState) {
        let h = state.x[3].max(1.0);

        // Process noise Q.
        let std_pos = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-2,
            STD_WEIGHT_POSITION * h,
        ];
        let std_vel = [
            STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * h,
            1e-5,
            STD_WEIGHT_VELOCITY * h,
        ];
        let mut q_diag = [0.0f32; N];
        for i in 0..4 {
            q_diag[i] = std_pos[i] * std_pos[i];
        }
        for i in 0..4 {
            q_diag[i + 4] = std_vel[i] * std_vel[i];
        }
        let q = StateMat::from_diagonal(&StateVec::from_column_slice(&q_diag));

        state.x = self.f * state.x;
        state.p = self.f * state.p * self.f.transpose() + q;
    }

    /// Update the state with a new measurement [cx, cy, a, h].
    pub fn update(&self, state: &mut KalmanState, measurement: [f32; 4]) {
        let z = MeasVec::from_column_slice(&measurement);

        // Innovation.
        let y = z - self.h * state.x;

        // Measurement noise R.
        let h = state.x[3].max(1.0);
        let std_meas = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-1,
            STD_WEIGHT_POSITION * h,
        ];
        let mut r_diag = [0.0f32; M];
        for i in 0..M {
            r_diag[i] = std_meas[i] * std_meas[i];
        }
        let r = MeasMat::from_diagonal(&MeasVec::from_column_slice(&r_diag));

        // Innovation covariance.
        let s = self.h * state.p * self.h.transpose() + r;
        let s_inv = s.try_inverse().unwrap_or(MeasMat::identity());

        // Kalman gain.
        let k: KalmanGain = state.p * self.h.transpose() * s_inv;

        // Posterior update.
        state.x += k * y;
        let i_kh = StateMat::identity() - k * self.h;
        state.p = i_kh * state.p;
    }

    /// Projected measurement mean and covariance (used for gating / distance).
    pub fn project(&self, state: &KalmanState) -> (MeasVec, MeasMat) {
        let h = state.x[3].max(1.0);
        let std_meas = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-1,
            STD_WEIGHT_POSITION * h,
        ];
        let mut r_diag = [0.0f32; M];
        for i in 0..M {
            r_diag[i] = std_meas[i] * std_meas[i];
        }
        let r = MeasMat::from_diagonal(&MeasVec::from_column_slice(&r_diag));

        let mean = self.h * state.x;
        let cov = self.h * state.p * self.h.transpose() + r;
        (mean, cov)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_update_roundtrip() {
        let kf = KalmanFilter::new();
        let mut state = kf.initiate([100.0, 200.0, 0.5, 300.0]);

        // Predict forward.
        kf.predict(&mut state);
        assert!((state.x[0] - 100.0).abs() < 1.0); // cx should not move much

        // Update with same measurement.
        kf.update(&mut state, [100.0, 200.0, 0.5, 300.0]);
        assert!((state.x[0] - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_velocity_estimation() {
        let kf = KalmanFilter::new();
        let mut state = kf.initiate([0.0, 0.0, 1.0, 100.0]);

        // Simulate object moving right at 10 px/frame.
        for t in 1..=10 {
            kf.predict(&mut state);
            kf.update(&mut state, [t as f32 * 10.0, 0.0, 1.0, 100.0]);
        }

        // Velocity estimate should converge toward 10.
        assert!(state.x[4] > 5.0, "v_cx = {} should be > 5", state.x[4]);
    }
}
