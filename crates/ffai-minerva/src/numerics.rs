//! The float arithmetic, written once, in one order, so a file gives the same tokens on every
//! device. These scalar functions are the ORACLES: every SIMD twin in [`crate::simd`]
//! reproduces them bit for bit (on every non-NaN input), and is tested `assert_eq!` on bits.
//!
//! Rules:
//!
//! - **A float dot product is [`dot16`]'s order**: 16 lanes, lane `j` accumulating elements
//!   `j, j + 16, j + 32, ...` with a multiply then an add (never a fused multiply-add), then a
//!   fixed pairwise reduction. AVX2 (2 x 8), SSE2, NEON and WASM-SIMD (4 x 4) all express it.
//! - **A sum that sets a scale (RMS norm, softmax denominator) is [`sum4_f64`]'s order**: four
//!   `f64` lanes, then `(l0 + l2) + (l1 + l3)`. (bitnet.cpp sums sequentially in `f64`; four lanes
//!   is as exact in practice and is an order every ISA can reproduce.)
//! - **`exp` is [`exp_canon`]**: Cephes' `expf` polynomial behind a round-to-nearest-even range
//!   reduction, in plain `f32` multiplies and adds — the same operations on every ISA, where a
//!   platform `expf` differs in the last bit from libm to libm.
//! - **Per-token setup (the `RoPE` angles) uses the pure-Rust `libm` crate**, never the
//!   platform's. `sqrt` is correctly rounded by IEEE 754 everywhere and is used directly.
//! - **`f16` conversions round to nearest even**, as the reference's `F16C` path does. A NaN's
//!   payload is not specified (it never reaches a token decision: every NaN compares alike).
//!
//! Where the reference rounds to `f16` (the KV cache; a query and the attention weights before
//! their products; the hidden state before the head), this engine rounds too.

use std::sync::OnceLock;

use half::f16;

/// Lanes in the canonical dot product.
pub const LANES: usize = 16;
/// Lanes in the canonical `f64` sum.
pub const F64_LANES: usize = 4;

/// Every `f16` bit pattern's `f32` value (exact): one 256 KB table, built once.
#[must_use]
pub fn f16_table() -> &'static [f32] {
    static T: OnceLock<Vec<f32>> = OnceLock::new();
    T.get_or_init(|| (0..=u16::MAX).map(|b| f16::from_bits(b).to_f32()).collect())
}

/// `x` rounded to the nearest `f16` and back.
#[inline]
#[must_use]
pub fn round_f16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// `x` as `f16` bits (round to nearest even).
#[inline]
#[must_use]
pub fn to_f16_bits(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

/// The canonical 16-lane reduction: halve the width, adding lane `j + w` into lane `j`.
#[inline]
#[must_use]
pub fn reduce16(acc: [f32; LANES]) -> f32 {
    let mut a = acc;
    let mut w = LANES / 2;
    while w > 0 {
        for j in 0..w {
            a[j] += a[j + w];
        }
        w /= 2;
    }
    a[0]
}

/// The canonical 4-lane `f64` reduction.
#[inline]
#[must_use]
pub fn reduce4_f64(a: [f64; F64_LANES]) -> f64 {
    (a[0] + a[2]) + (a[1] + a[3])
}

/// The canonical `f32` dot product (see the module rules). Lengths must be equal.
#[must_use]
pub fn dot16(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = [0f32; LANES];
    let (ac, at) = a.as_chunks::<LANES>();
    let (bc, bt) = b.as_chunks::<LANES>();
    for (x, y) in ac.iter().zip(bc) {
        for j in 0..LANES {
            acc[j] += x[j] * y[j];
        }
    }
    for (j, (x, y)) in at.iter().zip(bt).enumerate() {
        acc[j] += x * y;
    }
    reduce16(acc)
}

/// [`dot16`] with `a` as `f16` bits, converted exactly.
#[must_use]
pub fn dot16_f16(a: &[u16], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let t = f16_table();
    let mut acc = [0f32; LANES];
    let (ac, at) = a.as_chunks::<LANES>();
    let (bc, bt) = b.as_chunks::<LANES>();
    for (x, y) in ac.iter().zip(bc) {
        for j in 0..LANES {
            acc[j] += t[usize::from(x[j])] * y[j];
        }
    }
    for (j, (x, y)) in at.iter().zip(bt).enumerate() {
        acc[j] += t[usize::from(*x)] * y;
    }
    reduce16(acc)
}

/// [`dot16_f16`] with `a` as little-endian `f16` bytes (two a value).
#[must_use]
pub fn dot16_f16_bytes(a: &[u8], b: &[f32]) -> f32 {
    assert_eq!(a.len(), 2 * b.len());
    let t = f16_table();
    let mut acc = [0f32; LANES];
    let (ac, at) = a.as_chunks::<{ 2 * LANES }>();
    let (bc, bt) = b.as_chunks::<LANES>();
    for (x, y) in ac.iter().zip(bc) {
        for j in 0..LANES {
            acc[j] += t[usize::from(u16::from_le_bytes([x[2 * j], x[2 * j + 1]]))] * y[j];
        }
    }
    for (j, (x, y)) in at.chunks_exact(2).zip(bt).enumerate() {
        acc[j] += t[usize::from(u16::from_le_bytes([x[0], x[1]]))] * y;
    }
    reduce16(acc)
}

/// `out[d] = Σ_t p[t] · v_t[d]`, `t` in order from `out = 0` (each element a sequence of
/// multiply-then-add), where `v_t` is `v[t * stride .. t * stride + out.len()]` as `f16` bits.
pub fn weighted_sum_f16(p: &[f32], v: &[u16], stride: usize, out: &mut [f32]) {
    let t16 = f16_table();
    out.fill(0.0);
    for (t, &pt) in p.iter().enumerate() {
        let row = &v[t * stride..t * stride + out.len()];
        for (o, &x) in out.iter_mut().zip(row) {
            *o += pt * t16[usize::from(x)];
        }
    }
}

/// `Σ x²` in the canonical 4-lane `f64` order (each square exact in `f64`).
#[must_use]
pub fn sum_sq_f64(x: &[f32]) -> f64 {
    let mut acc = [0f64; F64_LANES];
    let (c, t) = x.as_chunks::<F64_LANES>();
    for v in c {
        for j in 0..F64_LANES {
            acc[j] += f64::from(v[j]) * f64::from(v[j]);
        }
    }
    for (j, &v) in t.iter().enumerate() {
        acc[j] += f64::from(v) * f64::from(v);
    }
    reduce4_f64(acc)
}

/// `Σ x` in the canonical 4-lane `f64` order.
#[must_use]
pub fn sum4_f64(x: &[f32]) -> f64 {
    let mut acc = [0f64; F64_LANES];
    let (c, t) = x.as_chunks::<F64_LANES>();
    for v in c {
        for j in 0..F64_LANES {
            acc[j] += f64::from(v[j]);
        }
    }
    for (j, &v) in t.iter().enumerate() {
        acc[j] += f64::from(v);
    }
    reduce4_f64(acc)
}

/// The scale an RMS norm multiplies by: `1 / sqrt(mean(x²) + eps)`, the mean rounded to `f32`
/// as the reference does.
#[must_use]
pub fn rms_scale(sum_sq: f64, n: usize, eps: f32) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "the reference's float mean"
    )]
    let mean = (sum_sq / n as f64) as f32;
    1.0 / (mean + eps).sqrt()
}

/// RMS norm: `out = (x · scale) · w`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    assert!(x.len() == w.len() && x.len() == out.len());
    let scale = rms_scale(sum_sq_f64(x), x.len(), eps);
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = (v * scale) * g;
    }
}

/// `exp` constants (Cephes `expf`): Cody-Waite split of `ln 2` (`C1` has 9 significant bits, so
/// `n · C1` is exact for every `n` the clamp allows) and the minimax polynomial.
pub mod exp_consts {
    /// Inputs below are clamped here (`e^-87` is still a normal `f32`).
    pub const LO: f32 = -87.0;
    /// Inputs above are clamped here (`e^88` is finite).
    pub const HI: f32 = 88.0;
    /// `log2(e)`.
    pub const LOG2E: f32 = std::f32::consts::LOG2_E;
    /// High part of `ln 2`.
    pub const C1: f32 = 0.693_359_4;
    /// Low part of `ln 2`.
    pub const C2: f32 = -2.121_944_4e-4;
    /// The polynomial, highest degree first.
    pub const P: [f32; 6] = [
        1.987_569_1e-4,
        1.398_199_9e-3,
        8.333_452e-3,
        4.166_579_6e-2,
        1.666_666_5e-1,
        0.5,
    ];
    // (Cephes writes the last as 5.0000001201e-1; in f32 that is exactly 0.5.)
}

/// The canonical `exp`: `x` clamped to `[-87, 88]`, `n = round_ties_even(x · log2 e)`,
/// `r = (x - n·C1) - n·C2`, `e^r ≈ ((p(r) · r²) + r) + 1`, times `2^n` built in the exponent
/// field. Every step is one `f32` operation, so every ISA computes the same bits. Within ~1 ulp
/// of the true value. A NaN becomes `e^-87`.
#[must_use]
pub fn exp_canon(x: f32) -> f32 {
    use exp_consts::{C1, C2, HI, LO, LOG2E, P};
    #[expect(
        clippy::manual_clamp,
        reason = "max/min map a NaN to LO, as every twin does; clamp would keep it"
    )]
    let x = x.max(LO).min(HI);
    let n = (x * LOG2E).round_ties_even();
    let r = x - n * C1;
    let r = r - n * C2;
    let mut p = P[0];
    for &c in &P[1..] {
        p = p * r + c;
    }
    let y = p * (r * r);
    let y = y + r;
    let y = y + 1.0;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "n is in -126..=127 after the clamp"
    )]
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    y * scale
}

/// Softmax in place, as the reference's `soft_max_ext(x, scale)`: multiply by `scale`, subtract
/// the max, [`exp_canon`] each, sum in the canonical `f64` order, multiply by `(1 / sum)`
/// rounded to `f32`.
pub fn softmax(x: &mut [f32], scale: f32) {
    for v in x.iter_mut() {
        *v *= scale;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in x.iter_mut() {
        *v = exp_canon(*v - max);
    }
    let sum = sum4_f64(x);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the reference scales by an f32"
    )]
    let inv = (1.0 / sum) as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// The `RoPE` angles for one position: `cos` and `sin` of `pos · base^(-2i/dims)` for each pair
/// `i`, the angle advanced by repeated `f32` multiplication as the reference does. Per-token
/// setup (`dims / 2` values), from the pure-Rust `libm`.
pub fn rope_table(pos: usize, dims: usize, base: f32, cos: &mut [f32], sin: &mut [f32]) {
    assert!(cos.len() == dims / 2 && sin.len() == dims / 2);
    #[expect(clippy::cast_precision_loss, reason = "head width, small")]
    let theta_scale = libm::powf(base, -2.0 / dims as f32);
    #[expect(clippy::cast_precision_loss, reason = "positions are far below 2^24")]
    let mut theta = pos as f32;
    for (c, s) in cos.iter_mut().zip(sin.iter_mut()) {
        *c = libm::cosf(theta);
        *s = libm::sinf(theta);
        theta *= theta_scale;
    }
}

/// `RoPE` in the `NEOX` arrangement (the reference's type for `BitNet`): pair `i` is
/// `(x[i], x[i + dims/2])`.
pub fn rope_neox(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    assert!(cos.len() == half && sin.len() == half);
    let (lo, hi) = x.split_at_mut(half);
    for i in 0..half {
        let (x0, x1) = (lo[i], hi[i]);
        lo[i] = x0 * cos[i] - x1 * sin[i];
        hi[i] = x0 * sin[i] + x1 * cos[i];
    }
}

/// A projection's float output: `out[r] = (sum[r] as f32 / act_scale) · weight_scale`
/// (`rusty_ternary::kernel::finish`, elementwise).
pub fn finish_rows(sum: &[i32], act_scale: f32, weight_scale: f32, out: &mut [f32]) {
    for (o, &s) in out.iter_mut().zip(sum) {
        *o = rusty_ternary::kernel::finish(s, act_scale, weight_scale);
    }
}

/// The feed-forward gate: `g = relu(g)² · u`, with `relu(g) = g > 0 ? g : 0`.
pub fn relu2_mul(g: &mut [f32], u: &[f32]) {
    for (g, &u) in g.iter_mut().zip(u) {
        let r = if *g > 0.0 { *g } else { 0.0 };
        *g = r * r * u;
    }
}

/// `x += y`.
pub fn add_assign(x: &mut [f32], y: &[f32]) {
    for (x, &y) in x.iter_mut().zip(y) {
        *x += y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot16_is_the_lane_order() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32).mul_add(0.37, -3.0)).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 / (i as f32 + 1.5)).collect();
        let mut acc = [0f32; LANES];
        for i in 0..37 {
            acc[i % LANES] += a[i] * b[i];
        }
        assert_eq!(dot16(&a, &b).to_bits(), reduce16(acc).to_bits());
        let h: Vec<u16> = a.iter().map(|&v| to_f16_bits(v)).collect();
        let hb: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        let hr: Vec<f32> = a.iter().map(|&v| round_f16(v)).collect();
        assert_eq!(dot16_f16(&h, &b).to_bits(), dot16(&hr, &b).to_bits());
        assert_eq!(dot16_f16_bytes(&hb, &b).to_bits(), dot16(&hr, &b).to_bits());
    }

    #[test]
    fn f16_table_is_exact() {
        let t = f16_table();
        for b in [0u16, 1, 0x3C00, 0x7BFF, 0x8001, 0xFC00] {
            assert_eq!(
                t[usize::from(b)].to_bits(),
                f16::from_bits(b).to_f32().to_bits()
            );
        }
        assert!(t[0x7E00].is_nan());
    }

    #[test]
    fn exp_canon_is_close_to_exp_and_monotone() {
        let mut worst = 0f64;
        let mut prev = 0f32;
        for i in 0..=200_000 {
            let x = -87.0 + 87.0 * (i as f32 / 200_000.0);
            let y = exp_canon(x);
            let want = f64::from(x).exp();
            worst = worst.max(((f64::from(y) - want) / want).abs());
            assert!(y >= prev, "not monotone at {x}");
            prev = y;
        }
        assert!(worst < 3e-7, "worst relative error {worst:e}");
        assert_eq!(exp_canon(0.0), 1.0);
        assert_eq!(exp_canon(-1000.0), exp_canon(-87.0));
        assert!(exp_canon(-87.0) > 0.0);
    }

    #[test]
    fn softmax_sums_to_one_and_keeps_order() {
        let mut x = [1.0f32, 3.0, 2.0, -50.0, 0.5];
        softmax(&mut x, 1.0);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[1] > x[2] && x[2] > x[0] && x[0] > x[4] && x[4] > x[3]);
    }

    #[test]
    fn sums_are_the_lane_order() {
        let x: Vec<f32> = (0..23).map(|i| (i as f32 * 0.731).sin()).collect();
        let mut acc = [0f64; 4];
        for (i, &v) in x.iter().enumerate() {
            acc[i % 4] += f64::from(v);
        }
        assert_eq!(sum4_f64(&x).to_bits(), reduce4_f64(acc).to_bits());
    }

    #[test]
    fn rope_at_zero_is_identity_and_preserves_norm() {
        let (mut c, mut s) = (vec![0.0; 4], vec![0.0; 4]);
        rope_table(0, 8, 500_000.0, &mut c, &mut s);
        let x0 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = x0;
        rope_neox(&mut x, &c, &s);
        assert_eq!(x, x0);
        rope_table(17, 8, 500_000.0, &mut c, &mut s);
        rope_neox(&mut x, &c, &s);
        let n0: f32 = x0.iter().map(|v| v * v).sum();
        let n1: f32 = x.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-3);
    }
}
