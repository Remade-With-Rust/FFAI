//! aarch64 NEON twins. NEON is baseline on aarch64; `super::Isa` still checks it at run time.
//!
//! The `f16` conversions are `fcvtl` / `fcvtn` by inline assembly (stable Rust has no
//! `float16x4_t`): exact and ties to even, as the oracles. Maxima use `maxnm` (a NaN loses, as
//! `f32::max`); the ReLU uses compare-and-select (as the oracle's `g > 0 ? g : 0`). No fused
//! multiply-add anywhere: `vmulq` then `vaddq`.

use std::arch::aarch64::*;

use super::{act_scale, quant_one};
use crate::numerics::{self as num, LANES, exp_consts as E};

/// Eight halves to two `f32x4`, by `fcvtl` (baseline aarch64, exact, subnormals included; a NaN
/// comes out quiet). Inline assembly: the `float16x4_t` intrinsics are unstable on 1.95.
#[target_feature(enable = "neon")]
#[inline]
fn h2f8(h: uint16x8_t) -> (float32x4_t, float32x4_t) {
    let (lo, hi): (float32x4_t, float32x4_t);
    // SAFETY: both read `h` and write one vector register each; baseline on aarch64.
    unsafe {
        std::arch::asm!(
            "fcvtl {lo:v}.4s, {h:v}.4h",
            "fcvtl2 {hi:v}.4s, {h:v}.8h",
            h = in(vreg) h,
            lo = out(vreg) lo,
            hi = lateout(vreg) hi,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (lo, hi)
}

/// Two `f32x4` to eight halves, by `fcvtn` (ties to even, overflow to infinity).
#[target_feature(enable = "neon")]
#[inline]
fn f2h8(a: float32x4_t, b: float32x4_t) -> uint16x8_t {
    let h: uint16x8_t;
    // SAFETY: reads `a` and `b`, writes one vector register; baseline on aarch64.
    unsafe {
        std::arch::asm!(
            "fcvtn {h:v}.4h, {a:v}.4s",
            "fcvtn2 {h:v}.8h, {b:v}.4s",
            h = out(vreg) h,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    h
}

/// Eight halves at `p` (any alignment) to two `f32x4`.
///
/// # Safety
/// `p` readable for 16 bytes.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load8h(p: *const u8) -> (float32x4_t, float32x4_t) {
    // SAFETY: the caller guarantees 16 readable bytes; a byte load has no alignment demand.
    h2f8(vreinterpretq_u16_u8(unsafe { vld1q_u8(p) }))
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

/// `acc + sdot(a, b)`: four signed products summed into each i32 lane.
///
/// Inline assembly, not the intrinsic: `vdotq_s32` is unstable (`stdarch_neon_dotprod`) on the
/// pinned 1.95 toolchain. One register-only instruction under `dotprod`.
#[target_feature(enable = "neon,dotprod")]
#[inline]
fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // SAFETY: `sdot` reads and writes only the three vector registers named; the function's
    // `dotprod` feature (detected by `super::Isa`) makes it valid to execute.
    unsafe {
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    acc
}

/// NEON dot-product GEMV: `sdot` of codes (0..=2, as signed bytes) against activations, straight
/// into i32. Rows four at a time: an activation vector is loaded once for the four, and the four
/// accumulators are independent chains (one chain held the core to one `sdot` a cycle).
///
/// # Safety
/// NEON + dotprod; shapes as [`gemv`].
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn gemv_dot(codes: &[u8], a: &[i8], a_sum: i32, y: &mut [i32]) {
    let rpc = a.len() / 4;
    let quads = y.len() / 4 * 4;
    let (y4, y1) = y.split_at_mut(quads);
    for (i, o) in y4.chunks_exact_mut(4).enumerate() {
        // SAFETY: rows `4i..4i+4` are inside `codes`.
        unsafe { gemv_rows_dot::<4>(codes.as_ptr().add(4 * i * rpc), rpc, a, a_sum, o) };
    }
    for (i, o) in y1.chunks_exact_mut(1).enumerate() {
        // SAFETY: row `quads + i` is inside `codes`.
        unsafe { gemv_rows_dot::<1>(codes.as_ptr().add((quads + i) * rpc), rpc, a, a_sum, o) };
    }
}

/// # Safety
/// NEON + dotprod; `R` rows of `rpc = a.len() / 4` bytes readable at `base`.
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn gemv_rows_dot<const R: usize>(
    base: *const u8,
    rpc: usize,
    a: &[i8],
    a_sum: i32,
    out: &mut [i32],
) {
    let mask = vdupq_n_u8(3);
    let mut acc = [vdupq_n_s32(0); R];
    for b in 0..a.len() / 128 {
        // SAFETY: block `b` reads codes `32b..32b+32` of each row and activations
        // `128b..128b+128`.
        unsafe {
            let ab = a.as_ptr().add(b * 128);
            let v = [
                vld1q_s8(ab),
                vld1q_s8(ab.add(16)),
                vld1q_s8(ab.add(32)),
                vld1q_s8(ab.add(48)),
                vld1q_s8(ab.add(64)),
                vld1q_s8(ab.add(80)),
                vld1q_s8(ab.add(96)),
                vld1q_s8(ab.add(112)),
            ];
            for (r, acc) in acc.iter_mut().enumerate() {
                let row = base.add(r * rpc + b * 32);
                let (c0, c1) = (vld1q_u8(row), vld1q_u8(row.add(16)));
                let q = [
                    vshrq_n_u8::<6>(c0),
                    vshrq_n_u8::<6>(c1),
                    vandq_u8(vshrq_n_u8::<4>(c0), mask),
                    vandq_u8(vshrq_n_u8::<4>(c1), mask),
                    vandq_u8(vshrq_n_u8::<2>(c0), mask),
                    vandq_u8(vshrq_n_u8::<2>(c1), mask),
                    vandq_u8(c0, mask),
                    vandq_u8(c1, mask),
                ];
                for (qk, vk) in q.into_iter().zip(v) {
                    *acc = sdot(*acc, vreinterpretq_s8_u8(qk), vk);
                }
            }
        }
    }
    for (o, v) in out.iter_mut().zip(acc) {
        *o = vaddvq_s32(v) - a_sum;
    }
}

/// # Safety
/// NEON + dotprod; as [`gemm_tile`].
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn gemm_tile_dot<const T: usize>(
    row: *const u8,
    a: *const i8,
    cols: usize,
    out: &mut [i32],
) {
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
                for (k, &qk) in q.iter().enumerate() {
                    *acc = sdot(*acc, qk, vld1q_s8(ab.add(16 * k)));
                }
            }
        }
    }
    for (o, v) in out.iter_mut().zip(acc) {
        *o = vaddvq_s32(v);
    }
}

/// # Safety
/// NEON + dotprod; shapes as [`gemm`].
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn gemm_dot(
    codes: &[u8],
    cols: usize,
    m: usize,
    a: &[i8],
    a_sums: &[i32],
    y: &mut [i32],
) {
    let rpc = cols / 4;
    // Tokens outside, rows inside, as `gemm_i8mm`.
    let mut t = 0;
    while t < m {
        let n = if m - t >= 8 { 8 } else { 1 };
        for (r, yr) in y.chunks_exact_mut(m).enumerate() {
            // SAFETY: row `r` and tokens `t..t + n` are inside the slices.
            unsafe {
                let row = codes.as_ptr().add(r * rpc);
                let at = a.as_ptr().add(t * cols);
                if n == 8 {
                    gemm_tile_dot::<8>(row, at, cols, &mut yr[t..t + 8]);
                } else {
                    gemm_tile_dot::<1>(row, at, cols, &mut yr[t..=t]);
                }
            }
        }
        t += n;
    }
    for yr in y.chunks_exact_mut(m) {
        for (o, &s) in yr.iter_mut().zip(a_sums) {
            *o -= s;
        }
    }
}

/// `acc + smmla(q, a)`: `q` two rows' 8 bytes, `a` two tokens' 8 bytes; lanes
/// `[r0·t0, r0·t1, r1·t0, r1·t1]`.
///
/// Inline assembly, not the intrinsic: `vmmlaq_s32` is unstable on the pinned 1.95 toolchain.
#[target_feature(enable = "neon,i8mm")]
#[inline]
fn smmla(mut acc: int32x4_t, q: int8x16_t, a: int8x16_t) -> int32x4_t {
    // SAFETY: `smmla` reads and writes only the three vector registers named; the function's
    // `i8mm` feature (detected by `super::Isa`) makes it valid to execute.
    unsafe {
        std::arch::asm!(
            "smmla {acc:v}.4s, {q:v}.16b, {a:v}.16b",
            acc = inout(vreg) acc,
            q = in(vreg) q,
            a = in(vreg) a,
            options(pure, nomem, nostack),
        );
    }
    acc
}

/// Two rows times `P` token pairs (`packed`: `P` pairs of `2 · cols`, see [`gemm_i8mm`]), the
/// sums into `acc`.
///
/// # Safety
/// NEON + i8mm; `row0` and `row1` readable for `cols / 4` bytes, `packed` for `P · 2 · cols`.
#[target_feature(enable = "neon,i8mm")]
#[inline]
unsafe fn gemm_tile_i8mm<const P: usize>(
    row0: *const u8,
    row1: *const u8,
    packed: *const i8,
    cols: usize,
) -> [int32x4_t; P] {
    let mask = vdupq_n_u8(3);
    let mut acc = [vdupq_n_s32(0); P];
    for b in 0..cols / 128 {
        // SAFETY: block `b` is inside both rows and inside every pair's packed activations.
        unsafe {
            let unpack = |row: *const u8| {
                let (c0, c1) = (vld1q_u8(row.add(b * 32)), vld1q_u8(row.add(b * 32 + 16)));
                [
                    vshrq_n_u8::<6>(c0),
                    vshrq_n_u8::<6>(c1),
                    vandq_u8(vshrq_n_u8::<4>(c0), mask),
                    vandq_u8(vshrq_n_u8::<4>(c1), mask),
                    vandq_u8(vshrq_n_u8::<2>(c0), mask),
                    vandq_u8(vshrq_n_u8::<2>(c1), mask),
                    vandq_u8(c0, mask),
                    vandq_u8(c1, mask),
                ]
            };
            let (u, v) = (unpack(row0), unpack(row1));
            for (k, (&u, &v)) in u.iter().zip(&v).enumerate() {
                let (u, v) = (vreinterpretq_s64_u8(u), vreinterpretq_s64_u8(v));
                let lo = vreinterpretq_s8_s64(vzip1q_s64(u, v));
                let hi = vreinterpretq_s8_s64(vzip2q_s64(u, v));
                for (p, acc) in acc.iter_mut().enumerate() {
                    let ab = packed.add(p * 2 * cols + b * 256 + k * 32);
                    *acc = smmla(*acc, lo, vld1q_s8(ab));
                    *acc = smmla(*acc, hi, vld1q_s8(ab.add(16)));
                }
            }
        }
    }
    acc
}

/// NEON i8mm GEMM: rows and tokens in pairs through `smmla` (32 products an instruction to
/// `sdot`'s 16). A pair of tokens is first packed as alternating 8-byte runs; an odd last row or
/// token takes the `sdot` tile.
///
/// # Safety
/// NEON + dotprod + i8mm; shapes as [`gemm`].
#[target_feature(enable = "neon,dotprod,i8mm")]
pub(super) unsafe fn gemm_i8mm(
    codes: &[u8],
    cols: usize,
    m: usize,
    a: &[i8],
    a_sums: &[i32],
    y: &mut [i32],
) {
    const PAIRS: usize = 4;
    let rpc = cols / 4;
    let rows = y.len() / m;
    let pairs = m / 2;
    let row_pairs = rows / 2;
    let (y2, y1) = y.split_at_mut(row_pairs * 2 * m);
    // Tokens outside, rows inside: a tile of 8 tokens' activations (20 to 55 KB) stays in L1 for
    // every row of the chunk. The other order walks the whole batch's activations (328 KB and
    // up) once a row pair.
    let mut packed = vec![0i8; PAIRS * 2 * cols];
    let mut p0 = 0;
    while p0 < pairs {
        let np = if pairs - p0 >= PAIRS { PAIRS } else { 1 };
        for (dst, src) in packed
            .chunks_exact_mut(2 * cols)
            .zip(a[p0 * 2 * cols..].chunks_exact(2 * cols))
            .take(np)
        {
            let (t0, t1) = src.split_at(cols);
            for ((d, a0), a1) in dst
                .chunks_exact_mut(16)
                .zip(t0.chunks_exact(8))
                .zip(t1.chunks_exact(8))
            {
                d[..8].copy_from_slice(a0);
                d[8..].copy_from_slice(a1);
            }
        }
        for (i, yy) in y2.chunks_exact_mut(2 * m).enumerate() {
            let (yr0, yr1) = yy.split_at_mut(m);
            let mut put = |p: usize, v: int32x4_t| {
                yr0[2 * p] = vgetq_lane_s32::<0>(v);
                yr0[2 * p + 1] = vgetq_lane_s32::<1>(v);
                yr1[2 * p] = vgetq_lane_s32::<2>(v);
                yr1[2 * p + 1] = vgetq_lane_s32::<3>(v);
            };
            // SAFETY: rows `2i` and `2i + 1` and the `np` packed pairs are in bounds.
            unsafe {
                let row0 = codes.as_ptr().add(2 * i * rpc);
                let row1 = row0.add(rpc);
                if np == PAIRS {
                    let acc = gemm_tile_i8mm::<PAIRS>(row0, row1, packed.as_ptr(), cols);
                    for (j, v) in acc.into_iter().enumerate() {
                        put(p0 + j, v);
                    }
                } else {
                    put(
                        p0,
                        gemm_tile_i8mm::<1>(row0, row1, packed.as_ptr(), cols)[0],
                    );
                }
            }
        }
        p0 += np;
    }
    for (i, yr) in y2.chunks_exact_mut(m).enumerate() {
        if m % 2 == 1 {
            // SAFETY: row `i` and the last token are in bounds.
            unsafe {
                let last = a.as_ptr().add((m - 1) * cols);
                gemm_tile_dot::<1>(codes.as_ptr().add(i * rpc), last, cols, &mut yr[m - 1..]);
            }
        }
        for (o, &s) in yr.iter_mut().zip(a_sums) {
            *o -= s;
        }
    }
    if !y1.is_empty() {
        // SAFETY: the odd last row, with the same shapes.
        unsafe { gemm_dot(&codes[row_pairs * 2 * rpc..], cols, m, a, a_sums, y1) };
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

/// int8 x int16 mat-vec: the weights widened to `i16`, `smlal` into i32.
///
/// # Safety
/// NEON; `q.len() == y.len() * a.len()`, `a.len() <= I16_MAX_COLS`, `|a| <= I16_ACT_MAX`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn i8i16_matvec(q: &[i8], a: &[i16], y: &mut [i64]) {
    let n = a.len();
    for (r, out) in y.iter_mut().enumerate() {
        let row = &q[r * n..(r + 1) * n];
        let mut acc = [vdupq_n_s32(0); 4];
        for i in 0..n / 16 {
            // SAFETY: elements `16i..16i+16` of the row and of `a`.
            unsafe {
                let w = vld1q_s8(row.as_ptr().add(16 * i));
                let (wl, wh) = (vmovl_s8(vget_low_s8(w)), vmovl_high_s8(w));
                let p = a.as_ptr().add(16 * i);
                let (vl, vh) = (vld1q_s16(p), vld1q_s16(p.add(8)));
                acc[0] = vmlal_s16(acc[0], vget_low_s16(wl), vget_low_s16(vl));
                acc[1] = vmlal_high_s16(acc[1], wl, vl);
                acc[2] = vmlal_s16(acc[2], vget_low_s16(wh), vget_low_s16(vh));
                acc[3] = vmlal_high_s16(acc[3], wh, vh);
            }
        }
        let mut sum = super::i8i16_tail(row, a, n / 16 * 16);
        for v in acc {
            sum += vaddlvq_s32(v);
        }
        *out = sum;
    }
}

/// int8 x int16 mat-vec by `sdot`: `a = 128·hi + lo`, two `sdot`s per 16 weights, four rows at a
/// time (the activation loads shared, eight independent chains).
///
/// # Safety
/// NEON + dotprod; as [`i8i16_matvec`], and `hi`, `lo` are `a`'s split.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn i8i16_matvec_dot(q: &[i8], a: &[i16], hi: &[i8], lo: &[i8], y: &mut [i64]) {
    let n = a.len();
    assert!(hi.len() == n && lo.len() == n && q.len() == y.len() * n);
    let quads = y.len() / 4 * 4;
    let (y4, y1) = y.split_at_mut(quads);
    for (i, o) in y4.chunks_exact_mut(4).enumerate() {
        let q = &q[4 * i * n..4 * (i + 1) * n];
        // SAFETY: four rows of `n`.
        unsafe { i8i16_rows_dot::<4>(q, a, hi, lo, o) };
    }
    for (i, o) in y1.chunks_exact_mut(1).enumerate() {
        let q = &q[(quads + i) * n..(quads + i + 1) * n];
        // SAFETY: one row of `n`.
        unsafe { i8i16_rows_dot::<1>(q, a, hi, lo, o) };
    }
}

/// # Safety
/// NEON + dotprod; `q` is `R` rows of `a.len()`; `hi` and `lo` as long as `a`.
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn i8i16_rows_dot<const R: usize>(q: &[i8], a: &[i16], hi: &[i8], lo: &[i8], y: &mut [i64]) {
    let n = a.len();
    let z = vdupq_n_s32(0);
    let mut acc = [(z, z); R];
    for i in 0..n / 16 {
        // SAFETY: elements `16i..16i+16` of `hi`, `lo` and each row.
        unsafe {
            let vh = vld1q_s8(hi.as_ptr().add(16 * i));
            let vl = vld1q_s8(lo.as_ptr().add(16 * i));
            for (r, (h, l)) in acc.iter_mut().enumerate() {
                let w = vld1q_s8(q.as_ptr().add(r * n + 16 * i));
                *h = sdot(*h, w, vh);
                *l = sdot(*l, w, vl);
            }
        }
    }
    for ((o, (h, l)), row) in y.iter_mut().zip(acc).zip(q.chunks_exact(n)) {
        *o = 128 * vaddlvq_s32(h) + vaddlvq_s32(l) + super::i8i16_tail(row, a, n / 16 * 16);
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
    // 64 outputs a pass: each pass walks every row, so a narrow pass walks the cache `stride`
    // apart once per 16 outputs. Per output the additions run in the same order.
    let n64 = out.len() / 64;
    for blk in 0..n64 {
        // SAFETY: values `64·blk..64·blk+64` of each row and of `out` are in bounds.
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); 16];
            for (t, &pt) in p.iter().enumerate() {
                let q = base.add(t * stride + 64 * blk).cast::<u8>();
                let pv = vdupq_n_f32(pt);
                for (k, pair) in acc.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let (h0, h1) = load8h(q.add(16 * k));
                    pair[0] = vaddq_f32(pair[0], vmulq_f32(pv, h0));
                    pair[1] = vaddq_f32(pair[1], vmulq_f32(pv, h1));
                }
            }
            let o = out.as_mut_ptr().add(64 * blk);
            for (k, v) in acc.into_iter().enumerate() {
                vst1q_f32(o.add(4 * k), v);
            }
        }
    }
    let n16 = out.len() / 16;
    for blk in 4 * n64..n16 {
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
pub(super) unsafe fn act_quant(x: &[f32], out: &mut [i8]) -> (f32, i32) {
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
        // No clamp: |x·s| <= 127·(1 + 2u) because s = 127 / max|x| (or |x| < 1e-5 under the
        // floor), so the round-to-nearest-even conversion lands in -127..=127 and the clamp the
        // oracle applies never acts. A NaN is zeroed first (the oracle maps it to 0).
        // `vcvtnq` converts rounding to nearest even.
        vcvtnq_s32_f32(v)
    };
    // The codes' sum rides along on the i32 lanes (already in -128..=127).
    let mut acc = vdupq_n_s32(0);
    for i in 0..n / 16 {
        // SAFETY: `16i + 16 <= n` for both slices.
        unsafe {
            let p = x.as_ptr().add(16 * i);
            let (q0, q1) = (q(vld1q_f32(p)), q(vld1q_f32(p.add(4))));
            let (q2, q3) = (q(vld1q_f32(p.add(8))), q(vld1q_f32(p.add(12))));
            acc = vaddq_s32(acc, vaddq_s32(vaddq_s32(q0, q1), vaddq_s32(q2, q3)));
            let a = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            let b = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            vst1q_s8(
                out.as_mut_ptr().add(16 * i),
                vcombine_s8(vqmovn_s16(a), vqmovn_s16(b)),
            );
        }
    }
    let mut sum = vaddvq_s32(acc);
    for i in n / 16 * 16..n {
        out[i] = quant_one(x[i], s);
        sum += i32::from(out[i]);
    }
    (s, sum)
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
        // f32 squared is exact in f64 (48 of 53 bits), so the fused multiply-add rounds exactly as
        // `acc + v * v` (the oracle's order and bits).
        a01 = vfmaq_f64(a01, lo, lo);
        a23 = vfmaq_f64(a23, hi, hi);
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
            let h = f2h8(vld1q_f32(p), vld1q_f32(p.add(4)));
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
    for i in 0..n / 8 {
        // SAFETY: `8i + 8 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(8 * i);
            let (a, b) = h2f8(f2h8(vld1q_f32(p), vld1q_f32(p.add(4))));
            vst1q_f32(p, a);
            vst1q_f32(p.add(4), b);
        }
    }
    for v in &mut x[n / 8 * 8..] {
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
