//! The float arithmetic, written once, in one order, so a file gives the same tokens on every
//! device.
//!
//! Rules (every SIMD twin must reproduce these bit for bit, not within a tolerance):
//!
//! - **Transcendentals come from the pure-Rust `libm` crate**, never the platform's (MSVC's CRT,
//!   glibc and a wasm runtime return different last bits for `expf`, `cosf`, `powf`).
//!   `sqrt` is correctly rounded by IEEE 754 everywhere and is used directly.
//! - **A float dot product is [`dot16`]'s order**: 16 lanes, lane `j` accumulating elements
//!   `j, j + 16, j + 32, ...` with a multiply then an add (never a fused multiply-add), then a
//!   fixed pairwise reduction. AVX2 (2 x 8), NEON and WASM-SIMD (4 x 4) can all express exactly
//!   this order.
//! - **Sums that set a scale (RMS norm, softmax denominator) run in `f64`, sequentially**,
//!   as the reference (`ggml_float`) does.
//! - **`f16` conversions round to nearest even** (`half`), as the reference's `F16C` path does.
//!
//! Where the reference rounds to `f16` (the KV cache; a query and the attention weights before
//! their products; the hidden state before the head), this engine rounds too.

use std::sync::OnceLock;

use half::f16;

/// Lanes in the canonical dot product.
pub const LANES: usize = 16;

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

/// `x` as `f16` bits.
#[inline]
#[must_use]
pub fn to_f16_bits(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

#[inline]
fn reduce(acc: [f32; LANES]) -> f32 {
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
    reduce(acc)
}

/// [`dot16`] with `a` given as `f16` bits (little-endian bytes, two a value), converted exactly.
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
    reduce(acc)
}

/// [`dot16`] with `a` given as `f16` bits.
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
    reduce(acc)
}

/// RMS norm as the reference computes it: `sum(x^2)` in `f64`, `mean` back to `f32`,
/// `scale = 1 / sqrt(mean + eps)`, then `(x * scale) * w`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    assert!(x.len() == w.len() && x.len() == out.len());
    let sum: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    #[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss, reason = "the reference's float mean")]
    let mean = (sum / x.len() as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = (v * scale) * g;
    }
}

/// Softmax in place, as the reference's `soft_max`: subtract the max, `expf` each, sum in
/// `f64`, multiply by `(1 / sum)` rounded to `f32`.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max);
        sum += f64::from(*v);
    }
    #[expect(clippy::cast_possible_truncation, reason = "the reference scales by an f32")]
    let inv = (1.0 / sum) as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// The `RoPE` table for one position: `(cos, sin)` of `pos * base^(-2i/dims)` for each pair
/// `i`, with the angle advanced by repeated `f32` multiplication as the reference does.
pub fn rope_table(pos: usize, dims: usize, base: f32, out: &mut [(f32, f32)]) {
    assert_eq!(out.len(), dims / 2);
    #[expect(clippy::cast_precision_loss, reason = "head width, small")]
    let theta_scale = libm::powf(base, -2.0 / dims as f32);
    #[expect(clippy::cast_precision_loss, reason = "positions are far below 2^24")]
    let mut theta = pos as f32;
    for c in out.iter_mut() {
        *c = (libm::cosf(theta), libm::sinf(theta));
        theta *= theta_scale;
    }
}

/// `RoPE` in the `NEOX` arrangement (the reference's type for `BitNet`): pair `i` is
/// `(x[i], x[i + dims/2])`.
pub fn rope_neox(x: &mut [f32], table: &[(f32, f32)]) {
    let half = x.len() / 2;
    assert_eq!(table.len(), half);
    for (i, &(c, s)) in table.iter().enumerate() {
        let (x0, x1) = (x[i], x[i + half]);
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
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
        assert_eq!(dot16(&a, &b).to_bits(), reduce(acc).to_bits());
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
            assert_eq!(t[usize::from(b)].to_bits(), f16::from_bits(b).to_f32().to_bits());
        }
        assert!(t[0x7E00].is_nan());
    }

    #[test]
    fn softmax_sums_to_one_and_keeps_order() {
        let mut x = [1.0f32, 3.0, 2.0, -50.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[1] > x[2] && x[2] > x[0] && x[0] > x[3]);
    }

    #[test]
    fn rope_at_zero_is_identity_and_preserves_norm() {
        let mut t = vec![(0.0, 0.0); 4];
        rope_table(0, 8, 500_000.0, &mut t);
        let x0 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = x0;
        rope_neox(&mut x, &t);
        assert_eq!(x, x0);
        rope_table(17, 8, 500_000.0, &mut t);
        rope_neox(&mut x, &t);
        let n0: f32 = x0.iter().map(|v| v * v).sum();
        let n1: f32 = x.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-3);
    }
}
