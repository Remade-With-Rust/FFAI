//! aarch64 NEON twins. NEON is baseline on aarch64; `super::Isa` still checks it at run time.
//!
//! The `f16` conversions are integer bit manipulation (stable Rust has no `float16x4_t`): exact,
//! and the same algorithm as the SSSE3 and WASM tiers. Maxima use `maxnm` (a NaN loses, as
//! `f32::max`); the ReLU uses compare-and-select (as the oracle's `g > 0 ? g : 0`). No fused
//! multiply-add anywhere: `vmulq` then `vaddq`.

use std::arch::aarch64::*;

use super::{act_scale, quant_one};
use crate::numerics::{self as num, LANES, exp_consts as E};

#[target_feature(enable = "neon")]
#[inline]
fn h2f4(h: uint32x4_t) -> float32x4_t {
    let me = vshlq_n_u32::<13>(vandq_u32(h, vdupq_n_u32(0x7fff)));
    let e = vandq_u32(me, vdupq_n_u32(0x0f80_0000));
    let mut o = vaddq_u32(me, vdupq_n_u32(0x3800_0000));
    let infnan = vceqq_u32(e, vdupq_n_u32(0x0f80_0000));
    o = vaddq_u32(o, vandq_u32(infnan, vdupq_n_u32(0x3800_0000)));
    let denorm = vceqq_u32(e, vdupq_n_u32(0));
    let od = vreinterpretq_u32_f32(vsubq_f32(
        vreinterpretq_f32_u32(vaddq_u32(o, vdupq_n_u32(0x0080_0000))),
        vdupq_n_f32(f32::from_bits(0x3880_0000)),
    ));
    o = vbslq_u32(denorm, od, o);
    vreinterpretq_f32_u32(vorrq_u32(
        o,
        vshlq_n_u32::<16>(vandq_u32(h, vdupq_n_u32(0x8000))),
    ))
}

/// Eight halves at `p` (any alignment) to two `f32x4`.
///
/// # Safety
/// `p` readable for 16 bytes.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load8h(p: *const u8) -> (float32x4_t, float32x4_t) {
    // SAFETY: the caller guarantees 16 readable bytes; a byte load has no alignment demand.
    let h = vreinterpretq_u16_u8(unsafe { vld1q_u8(p) });
    (h2f4(vmovl_u16(vget_low_u16(h))), h2f4(vmovl_high_u16(h)))
}

#[target_feature(enable = "neon")]
#[inline]
fn f2h4(v: float32x4_t) -> uint32x4_t {
    let x0 = vreinterpretq_u32_f32(v);
    let sign = vandq_u32(x0, vdupq_n_u32(0x8000_0000));
    let x = veorq_u32(x0, sign);
    let big = vcgtq_u32(x, vdupq_n_u32(0x477f_ffff));
    let nan = vcgtq_u32(x, vdupq_n_u32(0x7f80_0000));
    let o_big = vbslq_u32(nan, vdupq_n_u32(0x7e00), vdupq_n_u32(0x7c00));
    let small = vcltq_u32(x, vdupq_n_u32(0x3880_0000));
    let magic = vdupq_n_u32(0x3f00_0000);
    let o_small = vsubq_u32(
        vreinterpretq_u32_f32(vaddq_f32(
            vreinterpretq_f32_u32(x),
            vreinterpretq_f32_u32(magic),
        )),
        magic,
    );
    let odd = vandq_u32(vshrq_n_u32::<13>(x), vdupq_n_u32(1));
    let o_norm = vshrq_n_u32::<13>(vaddq_u32(vaddq_u32(x, vdupq_n_u32(0xc800_0fff)), odd));
    let o = vbslq_u32(small, o_small, vbslq_u32(big, o_big, o_norm));
    vorrq_u32(o, vshrq_n_u32::<16>(sign))
}

#[target_feature(enable = "neon")]
#[inline]
fn reduce16(a0: float32x4_t, a1: float32x4_t, a2: float32x4_t, a3: float32x4_t) -> f32 {
    let s = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
    let t = vaddq_f32(s, vextq_f32::<2>(s, s));
    vgetq_lane_f32::<0>(t) + vgetq_lane_f32::<1>(t)
}

#[target_feature(enable = "neon")]
#[inline]
fn exp4(v: float32x4_t) -> float32x4_t {
    let x = vminnmq_f32(vmaxnmq_f32(v, vdupq_n_f32(E::LO)), vdupq_n_f32(E::HI));
    let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(E::LOG2E)));
    let r = vsubq_f32(x, vmulq_f32(n, vdupq_n_f32(E::C1)));
    let r = vsubq_f32(r, vmulq_f32(n, vdupq_n_f32(E::C2)));
    let mut p = vdupq_n_f32(E::P[0]);
    for &c in &E::P[1..] {
        p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(c));
    }
    let y = vmulq_f32(p, vmulq_f32(r, r));
    let y = vaddq_f32(y, r);
    let y = vaddq_f32(y, vdupq_n_f32(1.0));
    let scale = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
    vmulq_f32(y, vreinterpretq_f32_s32(scale))
}

/// # Safety
/// NEON; `codes.len() == y.len() * a.len() / 4`, `a.len() % 128 == 0`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn gemv(codes: &[u8], a: &[i8], a_sum: i32, y: &mut [i32]) {
    let cols = a.len();
    let rpc = cols / 4;
    let mask = vdupq_n_u8(3);
    for (r, out) in y.iter_mut().enumerate() {
        // SAFETY: block `b` of row `r` reads codes `32b..32b+32` and activations
        // `128b..128b+128`, inside the slices.
        unsafe {
            let row = codes.as_ptr().add(r * rpc);
            let mut acc = vdupq_n_s32(0);
            for b in 0..cols / 128 {
                let c0 = vld1q_u8(row.add(b * 32));
                let c1 = vld1q_u8(row.add(b * 32 + 16));
                let ab = a.as_ptr().add(b * 128);
                // Codes 0..=3 times activations: a product is at most 384 in magnitude; eight per
                // 16-bit lane per block stay under 3,072.
                let (mut lo, mut hi) = (vdupq_n_s16(0), vdupq_n_s16(0));
                macro_rules! group {
                    ($q0:expr, $q1:expr, $g:literal) => {
                        let q0 = vreinterpretq_s8_u8(vandq_u8($q0, mask));
                        let q1 = vreinterpretq_s8_u8(vandq_u8($q1, mask));
                        let a0 = vld1q_s8(ab.add($g * 32));
                        let a1 = vld1q_s8(ab.add($g * 32 + 16));
                        lo = vmlal_s8(lo, vget_low_s8(q0), vget_low_s8(a0));
                        hi = vmlal_high_s8(hi, q0, a0);
                        lo = vmlal_s8(lo, vget_low_s8(q1), vget_low_s8(a1));
                        hi = vmlal_high_s8(hi, q1, a1);
                    };
                }
                group!(vshrq_n_u8::<6>(c0), vshrq_n_u8::<6>(c1), 0);
                group!(vshrq_n_u8::<4>(c0), vshrq_n_u8::<4>(c1), 1);
                group!(vshrq_n_u8::<2>(c0), vshrq_n_u8::<2>(c1), 2);
                group!(c0, c1, 3);
                acc = vpadalq_s16(acc, lo);
                acc = vpadalq_s16(acc, hi);
            }
            *out = vaddvq_s32(acc) - a_sum;
        }
    }
}

/// # Safety
/// NEON; `row` readable for `cols / 4` bytes, `a` for `T` tokens of `cols`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn gemm_tile<const T: usize>(row: *const u8, a: *const i8, cols: usize, out: &mut [i32]) {
    let mask = vdupq_n_u8(3);
    let mut acc = [vdupq_n_s32(0); T];
    for b in 0..cols / 128 {
        // SAFETY: block `b` is inside the row and inside every token's activations.
        unsafe {
            let c0 = vld1q_u8(row.add(b * 32));
            let c1 = vld1q_u8(row.add(b * 32 + 16));
            let q = [
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<6>(c0), mask)),
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<6>(c1), mask)),
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(c0), mask)),
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(c1), mask)),
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(c0), mask)),
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(c1), mask)),
                vreinterpretq_s8_u8(vandq_u8(c0, mask)),
                vreinterpretq_s8_u8(vandq_u8(c1, mask)),
            ];
            for (t, acc) in acc.iter_mut().enumerate() {
                let ab = a.add(t * cols + b * 128);
                let (mut lo, mut hi) = (vdupq_n_s16(0), vdupq_n_s16(0));
                for (k, &qk) in q.iter().enumerate() {
                    let v = vld1q_s8(ab.add(16 * k));
                    lo = vmlal_s8(lo, vget_low_s8(qk), vget_low_s8(v));
                    hi = vmlal_high_s8(hi, qk, v);
                }
                *acc = vpadalq_s16(vpadalq_s16(*acc, lo), hi);
            }
        }
    }
    for (o, v) in out.iter_mut().zip(acc) {
        *o = vaddvq_s32(v);
    }
}

/// # Safety
/// NEON; `codes` whole rows of `cols / 4` bytes, `a` `m · cols`, `a_sums` `m`, `y` `rows · m`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn gemm(
    codes: &[u8],
    cols: usize,
    m: usize,
    a: &[i8],
    a_sums: &[i32],
    y: &mut [i32],
) {
    let rpc = cols / 4;
    for (r, yr) in y.chunks_exact_mut(m).enumerate() {
        // SAFETY: row `r` and every tile's tokens are inside the slices.
        unsafe {
            let row = codes.as_ptr().add(r * rpc);
            let mut t = 0;
            while t + 8 <= m {
                gemm_tile::<8>(row, a.as_ptr().add(t * cols), cols, &mut yr[t..t + 8]);
                t += 8;
            }
            while t < m {
                gemm_tile::<1>(row, a.as_ptr().add(t * cols), cols, &mut yr[t..=t]);
                t += 1;
            }
        }
        for (o, &s) in yr.iter_mut().zip(a_sums) {
            *o -= s;
        }
    }
}

/// int8 x f32 mat-vec in the canonical 16-lane order.
///
/// # Safety
/// NEON; `q.len() == out.len() * x.len()`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn i8f_matvec(q: &[i8], x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let n16 = n / LANES;
    for (r, o) in out.iter_mut().enumerate() {
        let row = &q[r * n..(r + 1) * n];
        // SAFETY: chunk `i < n16` reads row bytes and `x` floats `16i..16i+16`.
        unsafe {
            let z = vdupq_n_f32(0.0);
            let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
            for i in 0..n16 {
                let v = vld1q_s8(row.as_ptr().add(16 * i));
                let (l, h) = (vmovl_s8(vget_low_s8(v)), vmovl_high_s8(v));
                let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(l)));
                let w1 = vcvtq_f32_s32(vmovl_high_s16(l));
                let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(h)));
                let w3 = vcvtq_f32_s32(vmovl_high_s16(h));
                let xp = x.as_ptr().add(16 * i);
                a0 = vaddq_f32(a0, vmulq_f32(w0, vld1q_f32(xp)));
                a1 = vaddq_f32(a1, vmulq_f32(w1, vld1q_f32(xp.add(4))));
                a2 = vaddq_f32(a2, vmulq_f32(w2, vld1q_f32(xp.add(8))));
                a3 = vaddq_f32(a3, vmulq_f32(w3, vld1q_f32(xp.add(12))));
            }
            if n.is_multiple_of(LANES) {
                *o = reduce16(a0, a1, a2, a3);
            } else {
                let mut acc = [0f32; LANES];
                vst1q_f32(acc.as_mut_ptr(), a0);
                vst1q_f32(acc.as_mut_ptr().add(4), a1);
                vst1q_f32(acc.as_mut_ptr().add(8), a2);
                vst1q_f32(acc.as_mut_ptr().add(12), a3);
                *o = super::i8f_finish(acc, row, x, n16 * LANES);
            }
        }
    }
}

/// int8 x int8 GEMV: every product widened to `i16` (`vmull`), every pair to `i32` (`vpadal`).
///
/// # Safety
/// NEON; `q.len() == y.len() * a.len()`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn i8_gemv(q: &[i8], a: &[i8], y: &mut [i32]) {
    let n = a.len();
    for (r, out) in y.iter_mut().enumerate() {
        let row = &q[r * n..(r + 1) * n];
        let mut acc = vdupq_n_s32(0);
        for i in 0..n / 16 {
            // SAFETY: bytes `16i..16i+16` of the row and of `a`.
            unsafe {
                let w = vld1q_s8(row.as_ptr().add(16 * i));
                let v = vld1q_s8(a.as_ptr().add(16 * i));
                acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w), vget_low_s8(v)));
                acc = vpadalq_s16(acc, vmull_high_s8(w, v));
            }
        }
        let tail: i32 = row[n / 16 * 16..]
            .iter()
            .zip(&a[n / 16 * 16..])
            .map(|(&w, &v)| i32::from(w) * i32::from(v))
            .sum();
        *out = vaddvq_s32(acc) + tail;
    }
}

/// # Safety
/// NEON; rows `r < out.len()` at `base + r·row_bytes` readable for `2 * x.len()` bytes.
#[target_feature(enable = "neon")]
pub(super) unsafe fn f16_matvec(base: *const u8, row_bytes: usize, x: &[f32], out: &mut [f32]) {
    let n16 = x.len() / LANES;
    let t16 = num::f16_table();
    for (r, o) in out.iter_mut().enumerate() {
        // SAFETY: chunk `i < n16` reads row bytes `32i..32i+32` and `x[16i..16i+16]`; the tail
        // reads the rest of the row.
        unsafe {
            let row = base.add(r * row_bytes);
            let z = vdupq_n_f32(0.0);
            let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
            for i in 0..n16 {
                let p = row.add(32 * i);
                let (h0, h1) = load8h(p);
                let (h2, h3) = load8h(p.add(16));
                let xp = x.as_ptr().add(16 * i);
                a0 = vaddq_f32(a0, vmulq_f32(h0, vld1q_f32(xp)));
                a1 = vaddq_f32(a1, vmulq_f32(h1, vld1q_f32(xp.add(4))));
                a2 = vaddq_f32(a2, vmulq_f32(h2, vld1q_f32(xp.add(8))));
                a3 = vaddq_f32(a3, vmulq_f32(h3, vld1q_f32(xp.add(12))));
            }
            if x.len().is_multiple_of(LANES) {
                *o = reduce16(a0, a1, a2, a3);
            } else {
                let mut acc = [0f32; LANES];
                vst1q_f32(acc.as_mut_ptr(), a0);
                vst1q_f32(acc.as_mut_ptr().add(4), a1);
                vst1q_f32(acc.as_mut_ptr().add(8), a2);
                vst1q_f32(acc.as_mut_ptr().add(12), a3);
                for (j, i) in (n16 * LANES..x.len()).enumerate() {
                    let h = u16::from_le_bytes([*row.add(2 * i), *row.add(2 * i + 1)]);
                    acc[j] += t16[usize::from(h)] * x[i];
                }
                *o = num::reduce16(acc);
            }
        }
    }
}

/// # Safety
/// NEON; rows `t < p.len()` at `base + t·stride` readable for `out.len()` values.
#[target_feature(enable = "neon")]
pub(super) unsafe fn weighted_sum(p: &[f32], base: *const u16, stride: usize, out: &mut [f32]) {
    let n16 = out.len() / 16;
    for blk in 0..n16 {
        // SAFETY: values `16·blk..16·blk+16` of each row and of `out` are in bounds.
        unsafe {
            let z = vdupq_n_f32(0.0);
            let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
            for (t, &pt) in p.iter().enumerate() {
                let q = base.add(t * stride + 16 * blk).cast::<u8>();
                let pv = vdupq_n_f32(pt);
                let (h0, h1) = load8h(q);
                let (h2, h3) = load8h(q.add(16));
                a0 = vaddq_f32(a0, vmulq_f32(pv, h0));
                a1 = vaddq_f32(a1, vmulq_f32(pv, h1));
                a2 = vaddq_f32(a2, vmulq_f32(pv, h2));
                a3 = vaddq_f32(a3, vmulq_f32(pv, h3));
            }
            let o = out.as_mut_ptr().add(16 * blk);
            vst1q_f32(o, a0);
            vst1q_f32(o.add(4), a1);
            vst1q_f32(o.add(8), a2);
            vst1q_f32(o.add(12), a3);
        }
    }
    let t16 = num::f16_table();
    for (d, o) in out.iter_mut().enumerate().skip(n16 * 16) {
        let mut acc = 0f32;
        for (t, &pt) in p.iter().enumerate() {
            // SAFETY: row `t` spans `out.len()` values.
            acc += pt * t16[usize::from(unsafe { *base.add(t * stride + d) })];
        }
        *o = acc;
    }
}

/// # Safety
/// NEON; equal lengths.
#[target_feature(enable = "neon")]
pub(super) unsafe fn act_quant(x: &[f32], out: &mut [i8]) -> f32 {
    let n = x.len();
    let mut m = vdupq_n_f32(0.0);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        m = vmaxnmq_f32(m, vabsq_f32(unsafe { vld1q_f32(x.as_ptr().add(4 * i)) }));
    }
    let amax = [
        vgetq_lane_f32::<0>(m),
        vgetq_lane_f32::<1>(m),
        vgetq_lane_f32::<2>(m),
        vgetq_lane_f32::<3>(m),
    ]
    .iter()
    .chain(x[n / 4 * 4..].iter())
    .fold(0f32, |a, v| a.max(v.abs()));
    let s = act_scale(amax);
    let sv = vdupq_n_f32(s);
    let q = |v: float32x4_t| {
        let v = vmulq_f32(v, sv);
        let v = vbslq_f32(vceqq_f32(v, v), v, vdupq_n_f32(0.0));
        let v = vminq_f32(
            vmaxq_f32(vrndnq_f32(v), vdupq_n_f32(-128.0)),
            vdupq_n_f32(127.0),
        );
        vcvtq_s32_f32(v)
    };
    for i in 0..n / 16 {
        // SAFETY: `16i + 16 <= n` for both slices.
        unsafe {
            let p = x.as_ptr().add(16 * i);
            let a = vcombine_s16(
                vqmovn_s32(q(vld1q_f32(p))),
                vqmovn_s32(q(vld1q_f32(p.add(4)))),
            );
            let b = vcombine_s16(
                vqmovn_s32(q(vld1q_f32(p.add(8)))),
                vqmovn_s32(q(vld1q_f32(p.add(12)))),
            );
            vst1q_s8(
                out.as_mut_ptr().add(16 * i),
                vcombine_s8(vqmovn_s16(a), vqmovn_s16(b)),
            );
        }
    }
    for i in n / 16 * 16..n {
        out[i] = quant_one(x[i], s);
    }
    s
}

/// # Safety
/// NEON; equal lengths.
#[target_feature(enable = "neon")]
pub(super) unsafe fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let (mut a01, mut a23) = (vdupq_n_f64(0.0), vdupq_n_f64(0.0));
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        let v = unsafe { vld1q_f32(x.as_ptr().add(4 * i)) };
        let (lo, hi) = (vcvt_f64_f32(vget_low_f32(v)), vcvt_high_f64_f32(v));
        a01 = vaddq_f64(a01, vmulq_f64(lo, lo));
        a23 = vaddq_f64(a23, vmulq_f64(hi, hi));
    }
    let mut acc = [
        vgetq_lane_f64::<0>(a01),
        vgetq_lane_f64::<1>(a01),
        vgetq_lane_f64::<0>(a23),
        vgetq_lane_f64::<1>(a23),
    ];
    for (j, &v) in x[n / 4 * 4..].iter().enumerate() {
        acc[j] += f64::from(v) * f64::from(v);
    }
    let s = num::rms_scale(num::reduce4_f64(acc), n, eps);
    let sv = vdupq_n_f32(s);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for all three slices.
        unsafe {
            let v = vmulq_f32(
                vmulq_f32(vld1q_f32(x.as_ptr().add(4 * i)), sv),
                vld1q_f32(w.as_ptr().add(4 * i)),
            );
            vst1q_f32(out.as_mut_ptr().add(4 * i), v);
        }
    }
    for i in n / 4 * 4..n {
        out[i] = (x[i] * s) * w[i];
    }
}

/// # Safety
/// NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax(x: &mut [f32], scale: f32) {
    let n = x.len();
    let sc = vdupq_n_f32(scale);
    let mut m = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            let v = vmulq_f32(vld1q_f32(p), sc);
            vst1q_f32(p, v);
            m = vmaxnmq_f32(m, v);
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v *= scale;
    }
    let max = [
        vgetq_lane_f32::<0>(m),
        vgetq_lane_f32::<1>(m),
        vgetq_lane_f32::<2>(m),
        vgetq_lane_f32::<3>(m),
    ]
    .iter()
    .chain(x[n / 4 * 4..].iter())
    .copied()
    .fold(f32::NEG_INFINITY, f32::max);
    let mv = vdupq_n_f32(max);
    let (mut a01, mut a23) = (vdupq_n_f64(0.0), vdupq_n_f64(0.0));
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            let e = exp4(vsubq_f32(vld1q_f32(p), mv));
            vst1q_f32(p, e);
            a01 = vaddq_f64(a01, vcvt_f64_f32(vget_low_f32(e)));
            a23 = vaddq_f64(a23, vcvt_high_f64_f32(e));
        }
    }
    let mut acc = [
        vgetq_lane_f64::<0>(a01),
        vgetq_lane_f64::<1>(a01),
        vgetq_lane_f64::<0>(a23),
        vgetq_lane_f64::<1>(a23),
    ];
    for (j, v) in x[n / 4 * 4..].iter_mut().enumerate() {
        *v = num::exp_canon(*v - max);
        acc[j] += f64::from(*v);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the oracle scales by an f32"
    )]
    let inv = (1.0 / num::reduce4_f64(acc)) as f32;
    let iv = vdupq_n_f32(inv);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            vst1q_f32(p, vmulq_f32(vld1q_f32(p), iv));
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v *= inv;
    }
}

/// # Safety
/// NEON; `src.len() == 2 * dst.len()`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn h2f(src: &[u8], dst: &mut [f32]) {
    let n = dst.len();
    for i in 0..n / 8 {
        // SAFETY: bytes `16i..16i+16`, floats `8i..8i+8`.
        unsafe {
            let (a, b) = load8h(src.as_ptr().add(16 * i));
            vst1q_f32(dst.as_mut_ptr().add(8 * i), a);
            vst1q_f32(dst.as_mut_ptr().add(8 * i + 4), b);
        }
    }
    let t = num::f16_table();
    for i in n / 8 * 8..n {
        dst[i] = t[usize::from(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]))];
    }
}

/// # Safety
/// NEON; equal lengths.
#[target_feature(enable = "neon")]
pub(super) unsafe fn f2h(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    for i in 0..n / 8 {
        // SAFETY: floats `8i..8i+8`, halves `8i..8i+8`.
        unsafe {
            let p = src.as_ptr().add(8 * i);
            let h = vcombine_u16(
                vmovn_u32(f2h4(vld1q_f32(p))),
                vmovn_u32(f2h4(vld1q_f32(p.add(4)))),
            );
            vst1q_u8(
                dst.as_mut_ptr().add(8 * i).cast::<u8>(),
                vreinterpretq_u8_u16(h),
            );
        }
    }
    for i in n / 8 * 8..n {
        dst[i] = num::to_f16_bits(src[i]);
    }
}

/// # Safety
/// NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn round_f16(x: &mut [f32]) {
    let n = x.len();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            vst1q_f32(p, h2f4(f2h4(vld1q_f32(p))));
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v = num::round_f16(*v);
    }
}

/// # Safety
/// NEON; equal lengths.
#[target_feature(enable = "neon")]
pub(super) unsafe fn finish(sum: &[i32], act: f32, w: f32, out: &mut [f32]) {
    let n = sum.len();
    let (av, wv) = (vdupq_n_f32(act), vdupq_n_f32(w));
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for both slices.
        unsafe {
            let f = vcvtq_f32_s32(vld1q_s32(sum.as_ptr().add(4 * i)));
            vst1q_f32(out.as_mut_ptr().add(4 * i), vmulq_f32(vdivq_f32(f, av), wv));
        }
    }
    num::finish_rows(&sum[n / 4 * 4..], act, w, &mut out[n / 4 * 4..]);
}

/// # Safety
/// NEON; equal lengths.
#[target_feature(enable = "neon")]
pub(super) unsafe fn relu2_mul(g: &mut [f32], u: &[f32]) {
    let n = g.len();
    let z = vdupq_n_f32(0.0);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for both slices.
        unsafe {
            let p = g.as_mut_ptr().add(4 * i);
            let v = vld1q_f32(p);
            let r = vbslq_f32(vcgtq_f32(v, z), v, z);
            vst1q_f32(
                p,
                vmulq_f32(vmulq_f32(r, r), vld1q_f32(u.as_ptr().add(4 * i))),
            );
        }
    }
    num::relu2_mul(&mut g[n / 4 * 4..], &u[n / 4 * 4..]);
}

/// # Safety
/// NEON; equal lengths.
#[target_feature(enable = "neon")]
pub(super) unsafe fn add_assign(x: &mut [f32], y: &[f32]) {
    let n = x.len();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for both slices.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            vst1q_f32(p, vaddq_f32(vld1q_f32(p), vld1q_f32(y.as_ptr().add(4 * i))));
        }
    }
    num::add_assign(&mut x[n / 4 * 4..], &y[n / 4 * 4..]);
}

/// # Safety
/// NEON; `x.len() == 2 * cos.len() == 2 * sin.len()`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    let (lo, hi) = x.split_at_mut(half);
    for i in 0..half / 4 {
        // SAFETY: `4i + 4 <= half` for all four slices.
        unsafe {
            let (pl, ph) = (lo.as_mut_ptr().add(4 * i), hi.as_mut_ptr().add(4 * i));
            let (x0, x1) = (vld1q_f32(pl), vld1q_f32(ph));
            let (c, s) = (
                vld1q_f32(cos.as_ptr().add(4 * i)),
                vld1q_f32(sin.as_ptr().add(4 * i)),
            );
            vst1q_f32(pl, vsubq_f32(vmulq_f32(x0, c), vmulq_f32(x1, s)));
            vst1q_f32(ph, vaddq_f32(vmulq_f32(x0, s), vmulq_f32(x1, c)));
        }
    }
    for i in half / 4 * 4..half {
        let (x0, x1) = (lo[i], hi[i]);
        lo[i] = x0 * cos[i] - x1 * sin[i];
        hi[i] = x0 * sin[i] + x1 * cos[i];
    }
}
