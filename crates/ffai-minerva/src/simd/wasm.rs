//! wasm32 SIMD128 twins (the module is compiled only when the build enables `simd128`: wasm has
//! no run-time detection, and one v128 instruction makes the whole module require SIMD).
//!
//! Same algorithms as the SSSE3 tier: integer `f16` conversions, `extmul` products widened by
//! pairwise adds, `pmax` / `pmin` for maxima (`pmax(a, b) = a < b ? b : a`, so a NaN loses, as
//! `f32::max`, and `pmax(0, g)` is the oracle's `g > 0 ? g : 0`), `nearest` for ties-to-even.

use core::arch::wasm32::*;

use super::{act_scale, quant_one};
use crate::numerics::{self as num, LANES, exp_consts as E};

#[target_feature(enable = "simd128")]
#[inline]
fn h2f4(h: v128) -> v128 {
    let me = i32x4_shl(v128_and(h, u32x4_splat(0x7fff)), 13);
    let e = v128_and(me, u32x4_splat(0x0f80_0000));
    let mut o = i32x4_add(me, u32x4_splat(0x3800_0000));
    let infnan = i32x4_eq(e, u32x4_splat(0x0f80_0000));
    o = i32x4_add(o, v128_and(infnan, u32x4_splat(0x3800_0000)));
    let denorm = i32x4_eq(e, u32x4_splat(0));
    let od = f32x4_sub(
        i32x4_add(o, u32x4_splat(0x0080_0000)),
        f32x4_splat(f32::from_bits(0x3880_0000)),
    );
    o = v128_bitselect(od, o, denorm);
    v128_or(o, i32x4_shl(v128_and(h, u32x4_splat(0x8000)), 16))
}

/// # Safety
/// `p` readable for 16 bytes.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn load8h(p: *const u8) -> (v128, v128) {
    // SAFETY: the caller guarantees 16 readable bytes; `v128_load` has no alignment demand.
    let h = unsafe { v128_load(p.cast()) };
    (
        h2f4(u32x4_extend_low_u16x8(h)),
        h2f4(u32x4_extend_high_u16x8(h)),
    )
}

#[target_feature(enable = "simd128")]
#[inline]
fn f2h4(v: v128) -> v128 {
    let sign = v128_and(v, u32x4_splat(0x8000_0000));
    let x = v128_xor(v, sign);
    let big = u32x4_gt(x, u32x4_splat(0x477f_ffff));
    let nan = u32x4_gt(x, u32x4_splat(0x7f80_0000));
    let o_big = v128_bitselect(u32x4_splat(0x7e00), u32x4_splat(0x7c00), nan);
    let small = u32x4_lt(x, u32x4_splat(0x3880_0000));
    let magic = u32x4_splat(0x3f00_0000);
    let o_small = i32x4_sub(f32x4_add(x, magic), magic);
    let odd = v128_and(u32x4_shr(x, 13), u32x4_splat(1));
    let o_norm = u32x4_shr(i32x4_add(i32x4_add(x, u32x4_splat(0xc800_0fff)), odd), 13);
    let o = v128_bitselect(o_small, v128_bitselect(o_big, o_norm, big), small);
    v128_or(o, u32x4_shr(sign, 16))
}

#[target_feature(enable = "simd128")]
#[inline]
fn pack_lo16(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(a, b)
}

#[target_feature(enable = "simd128")]
#[inline]
fn reduce16(a0: v128, a1: v128, a2: v128, a3: v128) -> f32 {
    let s = f32x4_add(f32x4_add(a0, a2), f32x4_add(a1, a3));
    let t = f32x4_add(s, i32x4_shuffle::<2, 3, 2, 3>(s, s));
    f32x4_extract_lane::<0>(t) + f32x4_extract_lane::<1>(t)
}

#[target_feature(enable = "simd128")]
#[inline]
fn lanes_f32(v: v128) -> [f32; 4] {
    [
        f32x4_extract_lane::<0>(v),
        f32x4_extract_lane::<1>(v),
        f32x4_extract_lane::<2>(v),
        f32x4_extract_lane::<3>(v),
    ]
}

#[target_feature(enable = "simd128")]
#[inline]
fn exp4(v: v128) -> v128 {
    let x = f32x4_pmin(f32x4_splat(E::HI), f32x4_pmax(f32x4_splat(E::LO), v));
    let n = f32x4_nearest(f32x4_mul(x, f32x4_splat(E::LOG2E)));
    let r = f32x4_sub(x, f32x4_mul(n, f32x4_splat(E::C1)));
    let r = f32x4_sub(r, f32x4_mul(n, f32x4_splat(E::C2)));
    let mut p = f32x4_splat(E::P[0]);
    for &c in &E::P[1..] {
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(c));
    }
    let y = f32x4_mul(p, f32x4_mul(r, r));
    let y = f32x4_add(y, r);
    let y = f32x4_add(y, f32x4_splat(1.0));
    let scale = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(n), i32x4_splat(127)), 23);
    f32x4_mul(y, scale)
}

/// The two halves of an `f32x4` widened to `f64x2` (lanes 0,1 and 2,3).
#[target_feature(enable = "simd128")]
#[inline]
fn widen(v: v128) -> (v128, v128) {
    (
        f64x2_promote_low_f32x4(v),
        f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(v, v)),
    )
}

/// # Safety
/// `codes.len() == y.len() * a.len() / 4`, `a.len() % 128 == 0`.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn gemv(codes: &[u8], a: &[i8], a_sum: i32, y: &mut [i32]) {
    let cols = a.len();
    let rpc = cols / 4;
    let mask = u8x16_splat(3);
    for (r, out) in y.iter_mut().enumerate() {
        // SAFETY: block `b` of row `r` reads codes `32b..32b+32` and activations
        // `128b..128b+128`, inside the slices.
        unsafe {
            let row = codes.as_ptr().add(r * rpc);
            let mut acc = i32x4_splat(0);
            for b in 0..cols / 128 {
                let c0 = v128_load(row.add(b * 32).cast());
                let c1 = v128_load(row.add(b * 32 + 16).cast());
                let ab = a.as_ptr().add(b * 128);
                let (mut lo, mut hi) = (i16x8_splat(0), i16x8_splat(0));
                for (g, sh) in [(0usize, 6u32), (1, 4), (2, 2), (3, 0)] {
                    let q0 = v128_and(u8x16_shr(c0, sh), mask);
                    let q1 = v128_and(u8x16_shr(c1, sh), mask);
                    let a0 = v128_load(ab.add(g * 32).cast());
                    let a1 = v128_load(ab.add(g * 32 + 16).cast());
                    lo = i16x8_add(lo, i16x8_extmul_low_i8x16(q0, a0));
                    hi = i16x8_add(hi, i16x8_extmul_high_i8x16(q0, a0));
                    lo = i16x8_add(lo, i16x8_extmul_low_i8x16(q1, a1));
                    hi = i16x8_add(hi, i16x8_extmul_high_i8x16(q1, a1));
                }
                acc = i32x4_add(acc, i32x4_extadd_pairwise_i16x8(lo));
                acc = i32x4_add(acc, i32x4_extadd_pairwise_i16x8(hi));
            }
            let s = i32x4_extract_lane::<0>(acc)
                + i32x4_extract_lane::<1>(acc)
                + i32x4_extract_lane::<2>(acc)
                + i32x4_extract_lane::<3>(acc);
            *out = s - a_sum;
        }
    }
}

/// # Safety
/// `row` readable for `cols / 4` bytes, `a` for `T` tokens of `cols`.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn gemm_tile<const T: usize>(row: *const u8, a: *const i8, cols: usize, out: &mut [i32]) {
    let mask = u8x16_splat(3);
    let mut acc = [i32x4_splat(0); T];
    for b in 0..cols / 128 {
        // SAFETY: block `b` is inside the row and inside every token's activations.
        unsafe {
            let c0 = v128_load(row.add(b * 32).cast());
            let c1 = v128_load(row.add(b * 32 + 16).cast());
            let mut q = [i8x16_splat(0); 8];
            for (g, sh) in [6u32, 4, 2, 0].into_iter().enumerate() {
                q[2 * g] = v128_and(u8x16_shr(c0, sh), mask);
                q[2 * g + 1] = v128_and(u8x16_shr(c1, sh), mask);
            }
            for (t, acc) in acc.iter_mut().enumerate() {
                let ab = a.add(t * cols + b * 128);
                let (mut lo, mut hi) = (i16x8_splat(0), i16x8_splat(0));
                for (k, &qk) in q.iter().enumerate() {
                    let v = v128_load(ab.add(16 * k).cast());
                    lo = i16x8_add(lo, i16x8_extmul_low_i8x16(qk, v));
                    hi = i16x8_add(hi, i16x8_extmul_high_i8x16(qk, v));
                }
                *acc = i32x4_add(
                    *acc,
                    i32x4_add(
                        i32x4_extadd_pairwise_i16x8(lo),
                        i32x4_extadd_pairwise_i16x8(hi),
                    ),
                );
            }
        }
    }
    for (o, v) in out.iter_mut().zip(acc) {
        *o = i32x4_extract_lane::<0>(v)
            + i32x4_extract_lane::<1>(v)
            + i32x4_extract_lane::<2>(v)
            + i32x4_extract_lane::<3>(v);
    }
}

/// # Safety
/// `codes` whole rows of `cols / 4` bytes, `a` `m · cols`, `a_sums` `m`, `y` `rows · m`.
#[target_feature(enable = "simd128")]
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
/// `q.len() == out.len() * x.len()`.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn i8f_matvec(q: &[i8], x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let n16 = n / LANES;
    for (r, o) in out.iter_mut().enumerate() {
        let row = &q[r * n..(r + 1) * n];
        // SAFETY: chunk `i < n16` reads row bytes and `x` floats `16i..16i+16`.
        unsafe {
            let z = f32x4_splat(0.0);
            let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
            for i in 0..n16 {
                let v = v128_load(row.as_ptr().add(16 * i).cast());
                let (l, h) = (i16x8_extend_low_i8x16(v), i16x8_extend_high_i8x16(v));
                let w0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(l));
                let w1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(l));
                let w2 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(h));
                let w3 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(h));
                let xp = x.as_ptr().add(16 * i);
                a0 = f32x4_add(a0, f32x4_mul(w0, v128_load(xp.cast())));
                a1 = f32x4_add(a1, f32x4_mul(w1, v128_load(xp.add(4).cast())));
                a2 = f32x4_add(a2, f32x4_mul(w2, v128_load(xp.add(8).cast())));
                a3 = f32x4_add(a3, f32x4_mul(w3, v128_load(xp.add(12).cast())));
            }
            if n.is_multiple_of(LANES) {
                *o = reduce16(a0, a1, a2, a3);
            } else {
                let mut acc = [0f32; LANES];
                for (k, a) in [a0, a1, a2, a3].into_iter().enumerate() {
                    acc[4 * k..4 * k + 4].copy_from_slice(&lanes_f32(a));
                }
                *o = super::i8f_finish(acc, row, x, n16 * LANES);
            }
        }
    }
}

/// int8 x int16 mat-vec: the weights widened to `i16`, `dot` (pairs into i32).
///
/// # Safety
/// `q.len() == y.len() * a.len()`, `a.len() <= I16_MAX_COLS`, `|a| <= I16_ACT_MAX`.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn i8i16_matvec(q: &[i8], a: &[i16], y: &mut [i64]) {
    let n = a.len();
    for (r, out) in y.iter_mut().enumerate() {
        let row = &q[r * n..(r + 1) * n];
        let mut acc = i32x4_splat(0);
        for i in 0..n / 16 {
            // SAFETY: elements `16i..16i+16` of the row and of `a`.
            unsafe {
                let w = v128_load(row.as_ptr().add(16 * i).cast());
                let p = a.as_ptr().add(16 * i);
                let d0 = i32x4_dot_i16x8(i16x8_extend_low_i8x16(w), v128_load(p.cast()));
                let d1 = i32x4_dot_i16x8(i16x8_extend_high_i8x16(w), v128_load(p.add(8).cast()));
                acc = i32x4_add(acc, i32x4_add(d0, d1));
            }
        }
        *out = i64::from(i32x4_extract_lane::<0>(acc))
            + i64::from(i32x4_extract_lane::<1>(acc))
            + i64::from(i32x4_extract_lane::<2>(acc))
            + i64::from(i32x4_extract_lane::<3>(acc))
            + super::i8i16_tail(row, a, n / 16 * 16);
    }
}

/// # Safety
/// Rows `r < out.len()` at `base + r·row_bytes` readable for `2 * x.len()` bytes.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn f16_matvec(base: *const u8, row_bytes: usize, x: &[f32], out: &mut [f32]) {
    let n16 = x.len() / LANES;
    let t16 = num::f16_table();
    for (r, o) in out.iter_mut().enumerate() {
        // SAFETY: chunk `i < n16` reads row bytes `32i..32i+32` and `x[16i..16i+16]`; the tail
        // reads the rest of the row.
        unsafe {
            let row = base.add(r * row_bytes);
            let z = f32x4_splat(0.0);
            let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
            for i in 0..n16 {
                let p = row.add(32 * i);
                let (h0, h1) = load8h(p);
                let (h2, h3) = load8h(p.add(16));
                let xp = x.as_ptr().add(16 * i);
                a0 = f32x4_add(a0, f32x4_mul(h0, v128_load(xp.cast())));
                a1 = f32x4_add(a1, f32x4_mul(h1, v128_load(xp.add(4).cast())));
                a2 = f32x4_add(a2, f32x4_mul(h2, v128_load(xp.add(8).cast())));
                a3 = f32x4_add(a3, f32x4_mul(h3, v128_load(xp.add(12).cast())));
            }
            if x.len().is_multiple_of(LANES) {
                *o = reduce16(a0, a1, a2, a3);
            } else {
                let mut acc = [0f32; LANES];
                for (k, a) in [a0, a1, a2, a3].into_iter().enumerate() {
                    acc[4 * k..4 * k + 4].copy_from_slice(&lanes_f32(a));
                }
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
/// Rows `t < p.len()` at `base + t·stride` readable for `out.len()` values.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn weighted_sum(p: &[f32], base: *const u16, stride: usize, out: &mut [f32]) {
    let n16 = out.len() / 16;
    for blk in 0..n16 {
        // SAFETY: values `16·blk..16·blk+16` of each row and of `out` are in bounds.
        unsafe {
            let z = f32x4_splat(0.0);
            let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
            for (t, &pt) in p.iter().enumerate() {
                let q = base.add(t * stride + 16 * blk).cast::<u8>();
                let pv = f32x4_splat(pt);
                let (h0, h1) = load8h(q);
                let (h2, h3) = load8h(q.add(16));
                a0 = f32x4_add(a0, f32x4_mul(pv, h0));
                a1 = f32x4_add(a1, f32x4_mul(pv, h1));
                a2 = f32x4_add(a2, f32x4_mul(pv, h2));
                a3 = f32x4_add(a3, f32x4_mul(pv, h3));
            }
            let o = out.as_mut_ptr().add(16 * blk);
            v128_store(o.cast(), a0);
            v128_store(o.add(4).cast(), a1);
            v128_store(o.add(8).cast(), a2);
            v128_store(o.add(12).cast(), a3);
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
/// Equal lengths.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn act_quant(x: &[f32], out: &mut [i8]) -> (f32, i32) {
    let n = x.len();
    let mut m = f32x4_splat(0.0);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        m = f32x4_pmax(
            m,
            f32x4_abs(unsafe { v128_load(x.as_ptr().add(4 * i).cast()) }),
        );
    }
    let amax = lanes_f32(m)
        .iter()
        .chain(x[n / 4 * 4..].iter())
        .fold(0f32, |a, v| a.max(v.abs()));
    let s = act_scale(amax);
    let sv = f32x4_splat(s);
    // `trunc_sat` maps a NaN to 0 and saturates; the narrows saturate to the i8 range.
    let q = |v: v128| i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_mul(v, sv)));
    // The codes' sum rides along on the i32 lanes (already in -128..=127).
    let mut acc = i32x4_splat(0);
    for i in 0..n / 16 {
        // SAFETY: `16i + 16 <= n` for both slices.
        unsafe {
            let p = x.as_ptr().add(16 * i);
            let (q0, q1) = (q(v128_load(p.cast())), q(v128_load(p.add(4).cast())));
            let (q2, q3) = (
                q(v128_load(p.add(8).cast())),
                q(v128_load(p.add(12).cast())),
            );
            acc = i32x4_add(acc, i32x4_add(i32x4_add(q0, q1), i32x4_add(q2, q3)));
            let a = i16x8_narrow_i32x4(q0, q1);
            let b = i16x8_narrow_i32x4(q2, q3);
            v128_store(
                out.as_mut_ptr().add(16 * i).cast(),
                i8x16_narrow_i16x8(a, b),
            );
        }
    }
    let mut sum = i32x4_extract_lane::<0>(acc)
        + i32x4_extract_lane::<1>(acc)
        + i32x4_extract_lane::<2>(acc)
        + i32x4_extract_lane::<3>(acc);
    for i in n / 16 * 16..n {
        out[i] = quant_one(x[i], s);
        sum += i32::from(out[i]);
    }
    (s, sum)
}

/// # Safety
/// Equal lengths.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let (mut a01, mut a23) = (f64x2_splat(0.0), f64x2_splat(0.0));
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        let (lo, hi) = widen(unsafe { v128_load(x.as_ptr().add(4 * i).cast()) });
        a01 = f64x2_add(a01, f64x2_mul(lo, lo));
        a23 = f64x2_add(a23, f64x2_mul(hi, hi));
    }
    let mut acc = [
        f64x2_extract_lane::<0>(a01),
        f64x2_extract_lane::<1>(a01),
        f64x2_extract_lane::<0>(a23),
        f64x2_extract_lane::<1>(a23),
    ];
    for (j, &v) in x[n / 4 * 4..].iter().enumerate() {
        acc[j] += f64::from(v) * f64::from(v);
    }
    let s = num::rms_scale(num::reduce4_f64(acc), n, eps);
    let sv = f32x4_splat(s);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for all three slices.
        unsafe {
            let v = f32x4_mul(
                f32x4_mul(v128_load(x.as_ptr().add(4 * i).cast()), sv),
                v128_load(w.as_ptr().add(4 * i).cast()),
            );
            v128_store(out.as_mut_ptr().add(4 * i).cast(), v);
        }
    }
    for i in n / 4 * 4..n {
        out[i] = (x[i] * s) * w[i];
    }
}

/// # Safety
/// None beyond the tier.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn softmax(x: &mut [f32], scale: f32) {
    let n = x.len();
    let sc = f32x4_splat(scale);
    let mut m = f32x4_splat(f32::NEG_INFINITY);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            let v = f32x4_mul(v128_load(p.cast()), sc);
            v128_store(p.cast(), v);
            m = f32x4_pmax(m, v);
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v *= scale;
    }
    let max = lanes_f32(m)
        .iter()
        .chain(x[n / 4 * 4..].iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mv = f32x4_splat(max);
    let (mut a01, mut a23) = (f64x2_splat(0.0), f64x2_splat(0.0));
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            let e = exp4(f32x4_sub(v128_load(p.cast()), mv));
            v128_store(p.cast(), e);
            let (lo, hi) = widen(e);
            a01 = f64x2_add(a01, lo);
            a23 = f64x2_add(a23, hi);
        }
    }
    let mut acc = [
        f64x2_extract_lane::<0>(a01),
        f64x2_extract_lane::<1>(a01),
        f64x2_extract_lane::<0>(a23),
        f64x2_extract_lane::<1>(a23),
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
    let iv = f32x4_splat(inv);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            v128_store(p.cast(), f32x4_mul(v128_load(p.cast()), iv));
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v *= inv;
    }
}

/// # Safety
/// `src.len() == 2 * dst.len()`.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn h2f(src: &[u8], dst: &mut [f32]) {
    let n = dst.len();
    for i in 0..n / 8 {
        // SAFETY: bytes `16i..16i+16`, floats `8i..8i+8`.
        unsafe {
            let (a, b) = load8h(src.as_ptr().add(16 * i));
            v128_store(dst.as_mut_ptr().add(8 * i).cast(), a);
            v128_store(dst.as_mut_ptr().add(8 * i + 4).cast(), b);
        }
    }
    let t = num::f16_table();
    for i in n / 8 * 8..n {
        dst[i] = t[usize::from(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]))];
    }
}

/// # Safety
/// Equal lengths.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn f2h(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    for i in 0..n / 8 {
        // SAFETY: floats `8i..8i+8`, halves `8i..8i+8`.
        unsafe {
            let p = src.as_ptr().add(8 * i);
            let h = pack_lo16(f2h4(v128_load(p.cast())), f2h4(v128_load(p.add(4).cast())));
            v128_store(dst.as_mut_ptr().add(8 * i).cast(), h);
        }
    }
    for i in n / 8 * 8..n {
        dst[i] = num::to_f16_bits(src[i]);
    }
}

/// # Safety
/// None beyond the tier.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn round_f16(x: &mut [f32]) {
    let n = x.len();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            v128_store(p.cast(), h2f4(f2h4(v128_load(p.cast()))));
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v = num::round_f16(*v);
    }
}

/// # Safety
/// Equal lengths.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn finish(sum: &[i32], act: f32, w: f32, out: &mut [f32]) {
    let n = sum.len();
    let (av, wv) = (f32x4_splat(act), f32x4_splat(w));
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for both slices.
        unsafe {
            let f = f32x4_convert_i32x4(v128_load(sum.as_ptr().add(4 * i).cast()));
            v128_store(
                out.as_mut_ptr().add(4 * i).cast(),
                f32x4_mul(f32x4_div(f, av), wv),
            );
        }
    }
    num::finish_rows(&sum[n / 4 * 4..], act, w, &mut out[n / 4 * 4..]);
}

/// # Safety
/// Equal lengths.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn relu2_mul(g: &mut [f32], u: &[f32]) {
    let n = g.len();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for both slices.
        unsafe {
            let p = g.as_mut_ptr().add(4 * i);
            let r = f32x4_pmax(f32x4_splat(0.0), v128_load(p.cast()));
            v128_store(
                p.cast(),
                f32x4_mul(f32x4_mul(r, r), v128_load(u.as_ptr().add(4 * i).cast())),
            );
        }
    }
    num::relu2_mul(&mut g[n / 4 * 4..], &u[n / 4 * 4..]);
}

/// # Safety
/// Equal lengths.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn add_assign(x: &mut [f32], y: &[f32]) {
    let n = x.len();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for both slices.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            v128_store(
                p.cast(),
                f32x4_add(v128_load(p.cast()), v128_load(y.as_ptr().add(4 * i).cast())),
            );
        }
    }
    num::add_assign(&mut x[n / 4 * 4..], &y[n / 4 * 4..]);
}

/// # Safety
/// `x.len() == 2 * cos.len() == 2 * sin.len()`.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    let (lo, hi) = x.split_at_mut(half);
    for i in 0..half / 4 {
        // SAFETY: `4i + 4 <= half` for all four slices.
        unsafe {
            let (pl, ph) = (lo.as_mut_ptr().add(4 * i), hi.as_mut_ptr().add(4 * i));
            let (x0, x1) = (v128_load(pl.cast()), v128_load(ph.cast()));
            let (c, s) = (
                v128_load(cos.as_ptr().add(4 * i).cast()),
                v128_load(sin.as_ptr().add(4 * i).cast()),
            );
            v128_store(pl.cast(), f32x4_sub(f32x4_mul(x0, c), f32x4_mul(x1, s)));
            v128_store(ph.cast(), f32x4_add(f32x4_mul(x0, s), f32x4_mul(x1, c)));
        }
    }
    for i in half / 4 * 4..half {
        let (x0, x1) = (lo[i], hi[i]);
        lo[i] = x0 * cos[i] - x1 * sin[i];
        hi[i] = x0 * sin[i] + x1 * cos[i];
    }
}
