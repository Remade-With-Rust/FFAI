//! Constant-velocity Kalman filter for box tracking — the SORT/ByteTrack model.
//!
//! State is 8-dimensional: `[cx, cy, a, h, vx, vy, va, vh]` where `a` is the
//! aspect ratio `w/h`. Observations are the first four.
//!
//! # Why the noise scales with height
//!
//! Every noise term is proportional to the box HEIGHT rather than constant. A
//! person 400 px tall and one 40 px tall do not have the same positional
//! uncertainty in pixels — the far one's centre moves several pixels between
//! frames for the same real-world motion. A constant noise term makes the filter
//! over-trust distant boxes and under-trust near ones, which shows up as ID
//! switches in crowds rather than as anything visible in a per-frame metric.
//!
//! The two weights below (1/20 position, 1/160 velocity) are the reference's,
//! kept rather than re-derived so the tracker's behaviour can be compared to
//! published numbers.

/// Position noise weight — reference `_std_weight_position`.
const STD_POS: f32 = 1.0 / 20.0;
/// Velocity noise weight — reference `_std_weight_velocity`.
const STD_VEL: f32 = 1.0 / 160.0;

/// `[cx, cy, a, h, vx, vy, va, vh]`.
pub type State = [f32; 8];

/// Diagonal covariance. The full filter keeps an 8x8, but every matrix in this
/// model is diagonal or block-diagonal with a constant off-diagonal from the
/// motion coupling, and carrying 64 numbers to represent 8 was measurably
/// slower for no accuracy the metrics could see.
///
/// The coupling IS kept, in [`predict`]: velocity uncertainty is folded into
/// position each step, which is the part that matters.
pub type Cov = [f32; 8];

/// A box as the filter sees it: centre, aspect, height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xyah {
    pub cx: f32,
    pub cy: f32,
    pub a: f32,
    pub h: f32,
}

impl Xyah {
    /// From `[x0, y0, x1, y1]`.
    #[must_use] 
    pub fn from_xyxy(b: [f32; 4]) -> Self {
        let (w, h) = ((b[2] - b[0]).max(1e-6), (b[3] - b[1]).max(1e-6));
        Self { cx: b[0] + w * 0.5, cy: b[1] + h * 0.5, a: w / h, h }
    }

    /// Back to `[x0, y0, x1, y1]`.
    #[must_use] 
    pub fn to_xyxy(self) -> [f32; 4] {
        let w = self.a * self.h;
        [self.cx - w * 0.5, self.cy - self.h * 0.5, self.cx + w * 0.5, self.cy + self.h * 0.5]
    }
}

/// Initial state and covariance for a newly observed box.
#[must_use] 
pub fn initiate(m: Xyah) -> (State, Cov) {
    let s: State = [m.cx, m.cy, m.a, m.h, 0.0, 0.0, 0.0, 0.0];
    // Velocity starts UNKNOWN, so its variance starts large — 10x the position
    // term. Seeding it small would make the filter confident the object is
    // stationary and lag every real motion for several frames.
    let p: Cov = [
        (2.0 * STD_POS * m.h).powi(2),
        (2.0 * STD_POS * m.h).powi(2),
        1e-2,
        (2.0 * STD_POS * m.h).powi(2),
        (10.0 * STD_VEL * m.h).powi(2),
        (10.0 * STD_VEL * m.h).powi(2),
        1e-5,
        (10.0 * STD_VEL * m.h).powi(2),
    ];
    (s, p)
}

/// Advance one frame: `x += v`, and grow the covariance by the process noise.
pub fn predict(s: &mut State, p: &mut Cov) {
    for i in 0..4 {
        s[i] += s[i + 4];
    }
    let h = s[3].max(1e-6);
    let qp = [
        (STD_POS * h).powi(2),
        (STD_POS * h).powi(2),
        1e-4,
        (STD_POS * h).powi(2),
    ];
    let qv = [
        (STD_VEL * h).powi(2),
        (STD_VEL * h).powi(2),
        1e-10,
        (STD_VEL * h).powi(2),
    ];
    // Position uncertainty inherits the velocity's — this is the motion
    // coupling the diagonal representation would otherwise drop, and without it
    // a track that has not been seen for several frames stays falsely
    // confident about where it is.
    for i in 0..4 {
        p[i] += p[i + 4] + qp[i];
        p[i + 4] += qv[i];
    }
}

/// Fold in a measurement. Returns the posterior mean.
pub fn update(s: &mut State, p: &mut Cov, m: Xyah) -> Xyah {
    let h = s[3].max(1e-6);
    let r = [
        (STD_POS * h).powi(2),
        (STD_POS * h).powi(2),
        1e-2,
        (STD_POS * h).powi(2),
    ];
    let z = [m.cx, m.cy, m.a, m.h];
    for i in 0..4 {
        // Scalar Kalman gain per dimension: K = P / (P + R).
        let k = p[i] / (p[i] + r[i]).max(1e-12);
        let innov = z[i] - s[i];
        s[i] += k * innov;
        // The velocity estimate is corrected by the same innovation, scaled by
        // its own share of the uncertainty. Dropping this is the difference
        // between a filter that learns motion and one that only smooths.
        let kv = p[i + 4] / (p[i] + r[i]).max(1e-12);
        s[i + 4] += kv * innov;
        p[i] *= 1.0 - k;
        p[i + 4] *= 1.0 - kv;
    }
    Xyah { cx: s[0], cy: s[1], a: s[2], h: s[3] }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box moving at constant velocity must be TRACKED, not merely smoothed:
    /// after a few updates the filter's prediction should land near the truth
    /// before it sees the measurement.
    #[test]
    fn learns_constant_velocity() {
        let start = Xyah { cx: 100.0, cy: 100.0, a: 0.5, h: 80.0 };
        let (mut s, mut p) = initiate(start);
        let (vx, vy) = (6.0f32, 2.0f32);

        for k in 1..=12 {
            predict(&mut s, &mut p);
            let m = Xyah {
                cx: 100.0 + vx * k as f32,
                cy: 100.0 + vy * k as f32,
                a: 0.5,
                h: 80.0,
            };
            update(&mut s, &mut p, m);
        }
        // Velocity should have converged near the truth.
        assert!((s[4] - vx).abs() < 1.5, "vx {} vs {vx}", s[4]);
        assert!((s[5] - vy).abs() < 1.5, "vy {} vs {vy}", s[5]);

        // And the NEXT predict, with no measurement, should land close.
        predict(&mut s, &mut p);
        let want_x = 100.0 + vx * 13.0;
        assert!((s[0] - want_x).abs() < 6.0, "predicted {} vs {want_x}", s[0]);
    }

    /// Covariance must SHRINK on update and GROW on predict. A filter that has
    /// these backwards still produces plausible boxes and silently stops
    /// trusting its measurements.
    #[test]
    fn covariance_shrinks_on_update_and_grows_on_predict() {
        let (mut s, mut p) = initiate(Xyah { cx: 50.0, cy: 50.0, a: 0.5, h: 100.0 });
        let before = p[0];
        update(&mut s, &mut p, Xyah { cx: 50.0, cy: 50.0, a: 0.5, h: 100.0 });
        let after_update = p[0];
        assert!(after_update < before, "{after_update} !< {before}");
        predict(&mut s, &mut p);
        assert!(p[0] > after_update, "{} !> {after_update}", p[0]);
    }

    #[test]
    fn xyxy_round_trips() {
        let b = [10.0f32, 20.0, 40.0, 100.0];
        let got = Xyah::from_xyxy(b).to_xyxy();
        for i in 0..4 {
            assert!((got[i] - b[i]).abs() < 1e-3, "{got:?} vs {b:?}");
        }
    }

    /// Noise scales with height: a tall box must end up with a LARGER positional
    /// variance than a short one, or the filter over-trusts distant objects.
    #[test]
    fn noise_scales_with_height() {
        let (_, tall) = initiate(Xyah { cx: 0.0, cy: 0.0, a: 0.5, h: 400.0 });
        let (_, short) = initiate(Xyah { cx: 0.0, cy: 0.0, a: 0.5, h: 40.0 });
        assert!(tall[0] > short[0] * 50.0, "tall {} short {}", tall[0], short[0]);
    }
}
