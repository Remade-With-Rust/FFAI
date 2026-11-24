//! x86-64 twins: `ssse3` (SSE2 plus `pmaddubsw` / `pshufb`) and `avx2` (plus `f16c`).
//!
//! Every function here is `unsafe` for one reason: it must only run on a CPU with its
//! `target_feature`s, which `super::Isa` guarantees. Every pointer read is inside a bound the
//! safe entry point in `super` asserted, restated at each kernel.

use std::arch::x86_64::*;

use super::{act_scale, quant_one};
use crate::numerics::{self as num, LANES, exp_consts as E};

// ---------------------------------------------------------------------------------------------
// helpers (SSE2: baseline on x86-64, declared so the intrinsics are callable)

#[target_feature(enable = "sse2")]
#[inline]
fn hsum_epi32(v: __m128i) -> i32 {
    let s = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s)
}

/// The canonical reduction's last three steps on lanes `[0..4)` already holding
/// `a[j] + a[j+8]` + `a[j+4] + a[j+12]` folded as `num::reduce16` folds them.
#[target_feature(enable = "sse2")]
#[inline]
fn reduce4_canon(s: __m128) -> f32 {
    let t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps::<0b01>(t, t)))
}

/// Four `f16` (low 16 bits of each `u32` lane) to `f32`, exactly: the exponent rebiased, a
/// subnormal half renormalised by one exact `f32` subtraction, Inf/NaN kept.
#[target_feature(enable = "sse2")]
#[inline]
fn h2f4(h: __m128i) -> __m128 {
    let me = _mm_slli_epi32::<13>(_mm_and_si128(h, _mm_set1_epi32(0x7fff)));
    let e = _mm_and_si128(me, _mm_set1_epi32(0x0f80_0000));
    let mut o = _mm_add_epi32(me, _mm_set1_epi32(0x3800_0000));
    let infnan = _mm_cmpeq_epi32(e, _mm_set1_epi32(0x0f80_0000));
    o = _mm_add_epi32(o, _mm_and_si128(infnan, _mm_set1_epi32(0x3800_0000)));
    let denorm = _mm_cmpeq_epi32(e, _mm_setzero_si128());
    let od = _mm_castps_si128(_mm_sub_ps(
        _mm_castsi128_ps(_mm_add_epi32(o, _mm_set1_epi32(0x0080_0000))),
        _mm_set1_ps(f32::from_bits(0x3880_0000)),
    ));
    o = _mm_or_si128(_mm_and_si128(denorm, od), _mm_andnot_si128(denorm, o));
    _mm_castsi128_ps(_mm_or_si128(
        o,
        _mm_slli_epi32::<16>(_mm_and_si128(h, _mm_set1_epi32(0x8000))),
    ))
}

/// Eight `f16` at `p` to two `f32x4`.
///
/// # Safety
/// `p` must be readable for 16 bytes.
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn load8h(p: *const u8) -> (__m128, __m128) {
    // SAFETY: the caller guarantees 16 readable bytes.
    let h = unsafe { _mm_loadu_si128(p.cast()) };
    let z = _mm_setzero_si128();
    (
        h2f4(_mm_unpacklo_epi16(h, z)),
        h2f4(_mm_unpackhi_epi16(h, z)),
    )
}

/// Four `f32` to `f16` bits (in the low half of each `u32` lane), round to nearest even
/// (ryg's `float_to_half_fast3_rtne`, branch-free).
#[target_feature(enable = "sse2")]
#[inline]
fn f2h4(v: __m128) -> __m128i {
    let x0 = _mm_castps_si128(v);
    let sign = _mm_and_si128(x0, _mm_set1_epi32(i32::MIN));
    let x = _mm_xor_si128(x0, sign);
    let big = _mm_cmpgt_epi32(x, _mm_set1_epi32(0x477f_ffff));
    let nan = _mm_cmpgt_epi32(x, _mm_set1_epi32(0x7f80_0000));
    let o_big = _mm_or_si128(
        _mm_and_si128(nan, _mm_set1_epi32(0x7e00)),
        _mm_andnot_si128(nan, _mm_set1_epi32(0x7c00)),
    );
    let small = _mm_cmplt_epi32(x, _mm_set1_epi32(0x3880_0000));
    let magic = _mm_set1_epi32(0x3f00_0000);
    let o_small = _mm_sub_epi32(
        _mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(x), _mm_castsi128_ps(magic))),
        magic,
    );
    let odd = _mm_and_si128(_mm_srli_epi32::<13>(x), _mm_set1_epi32(1));
    #[expect(
        clippy::cast_possible_wrap,
        reason = "a bit pattern: (15 - 127) << 23, plus 0xfff"
    )]
    let rebias = _mm_set1_epi32(0xc800_0fff_u32 as i32);
    let o_norm = _mm_srli_epi32::<13>(_mm_add_epi32(_mm_add_epi32(x, rebias), odd));
    let o = _mm_or_si128(_mm_and_si128(big, o_big), _mm_andnot_si128(big, o_norm));
    let o = _mm_or_si128(_mm_and_si128(small, o_small), _mm_andnot_si128(small, o));
    _mm_or_si128(o, _mm_srli_epi32::<16>(sign))
}

/// The low 16 bits of each `u32` lane of `a` then `b`, as eight `u16`.
#[target_feature(enable = "ssse3")]
#[inline]
fn pack_lo16(a: __m128i, b: __m128i) -> __m128i {
    let m = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    _mm_unpacklo_epi64(_mm_shuffle_epi8(a, m), _mm_shuffle_epi8(b, m))
}

/// The tail of a 16-lane dot product: lanes stored, the rest added in the oracle's order, the
/// canonical reduction.
///
/// # Safety
/// `row` must be readable for `2 * x.len()` bytes.
unsafe fn finish_dot(acc: [f32; LANES], row: *const u8, x: &[f32], from: usize) -> f32 {
    let mut acc = acc;
    let t = num::f16_table();
    for (j, i) in (from..x.len()).enumerate() {
        // SAFETY: `i < x.len()`, so bytes `2i, 2i+1` are inside the row.
        let h = unsafe { u16::from_le_bytes([*row.add(2 * i), *row.add(2 * i + 1)]) };
        acc[j] += t[usize::from(h)] * x[i];
    }
    num::reduce16(acc)
}

#[target_feature(enable = "sse2")]
#[inline]
fn exp4(v: __m128) -> __m128 {
    let x = _mm_min_ps(_mm_max_ps(v, _mm_set1_ps(E::LO)), _mm_set1_ps(E::HI));
    let ni = _mm_cvtps_epi32(_mm_mul_ps(x, _mm_set1_ps(E::LOG2E)));
    let n = _mm_cvtepi32_ps(ni);
    let r = _mm_sub_ps(x, _mm_mul_ps(n, _mm_set1_ps(E::C1)));
    let r = _mm_sub_ps(r, _mm_mul_ps(n, _mm_set1_ps(E::C2)));
    let mut p = _mm_set1_ps(E::P[0]);
    for &c in &E::P[1..] {
        p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(c));
    }
    let y = _mm_mul_ps(p, _mm_mul_ps(r, r));
    let y = _mm_add_ps(y, r);
    let y = _mm_add_ps(y, _mm_set1_ps(1.0));
    let scale = _mm_slli_epi32::<23>(_mm_add_epi32(ni, _mm_set1_epi32(127)));
    _mm_mul_ps(y, _mm_castsi128_ps(scale))
}

#[target_feature(enable = "avx2")]
#[inline]
fn exp8(v: __m256) -> __m256 {
    let x = _mm256_min_ps(
        _mm256_max_ps(v, _mm256_set1_ps(E::LO)),
        _mm256_set1_ps(E::HI),
    );
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(E::LOG2E),
    ));
    let r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(E::C1)));
    let r = _mm256_sub_ps(r, _mm256_mul_ps(n, _mm256_set1_ps(E::C2)));
    let mut p = _mm256_set1_ps(E::P[0]);
    for &c in &E::P[1..] {
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(c));
    }
    let y = _mm256_mul_ps(p, _mm256_mul_ps(r, r));
    let y = _mm256_add_ps(y, r);
    let y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
    let scale = _mm256_slli_epi32::<23>(_mm256_add_epi32(
        _mm256_cvtps_epi32(n),
        _mm256_set1_epi32(127),
    ));
    _mm256_mul_ps(y, _mm256_castsi256_ps(scale))
}

// ---------------------------------------------------------------------------------------------
// I2_S GEMV: codes 0/1/2 times int8 activations by `pmaddubsw` (a pair is at most 2·2·128 =
// 512, or 768 for an invalid code 3; eight pairs per 16-bit lane per block stay under 6,144),
// widened to i32 by `pmaddwd` once a block, then `Σ code·a − Σ a = Σ trit·a`.

/// # Safety
/// SSSE3; `codes.len() == y.len() * a.len() / 4`, `a.len() % 128 == 0`.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn gemv_ssse3(codes: &[u8], a: &[i8], a_sum: i32, y: &mut [i32]) {
    let cols = a.len();
    let rpc = cols / 4;
    let mask = _mm_set1_epi8(3);
    let ones = _mm_set1_epi16(1);
    for (r, out) in y.iter_mut().enumerate() {
        // SAFETY: row `r < y.len()` spans `rpc` bytes inside `codes`; block `b` reads bytes
        // `32b..32b+32` of it and activations `128b..128b+128`, all in bounds.
        unsafe {
            let row = codes.as_ptr().add(r * rpc);
            let mut acc = _mm_setzero_si128();
            for b in 0..cols / 128 {
                let c0 = _mm_loadu_si128(row.add(b * 32).cast());
                let c1 = _mm_loadu_si128(row.add(b * 32 + 16).cast());
                let ab = a.as_ptr().add(b * 128);
                let mut s = _mm_setzero_si128();
                macro_rules! group {
                    ($sh:literal, $g:literal) => {
                        let q0 = _mm_and_si128(_mm_srli_epi16::<$sh>(c0), mask);
                        let q1 = _mm_and_si128(_mm_srli_epi16::<$sh>(c1), mask);
                        s = _mm_add_epi16(
                            s,
                            _mm_maddubs_epi16(q0, _mm_loadu_si128(ab.add($g * 32).cast())),
                        );
                        s = _mm_add_epi16(
                            s,
                            _mm_maddubs_epi16(q1, _mm_loadu_si128(ab.add($g * 32 + 16).cast())),
                        );
                    };
                }
                group!(6, 0);
                group!(4, 1);
                group!(2, 2);
                group!(0, 3);
                acc = _mm_add_epi32(acc, _mm_madd_epi16(s, ones));
            }
            *out = hsum_epi32(acc) - a_sum;
        }
    }
}

/// # Safety
/// AVX2; `codes.len() == y.len() * a.len() / 4`, `a.len() % 128 == 0`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn gemv_avx2(codes: &[u8], a: &[i8], a_sum: i32, y: &mut [i32]) {
    let cols = a.len();
    let rpc = cols / 4;
    let mask = _mm256_set1_epi8(3);
    let ones = _mm256_set1_epi16(1);
    for (r, out) in y.iter_mut().enumerate() {
        // SAFETY: as `gemv_ssse3`: every read is inside row `r` and the activations.
        unsafe {
            let row = codes.as_ptr().add(r * rpc);
            let mut acc = _mm256_setzero_si256();
            for b in 0..cols / 128 {
                let c = _mm256_loadu_si256(row.add(b * 32).cast());
                let ab = a.as_ptr().add(b * 128);
                let q = |v: __m256i| _mm256_and_si256(v, mask);
                let mut s = _mm256_maddubs_epi16(
                    q(_mm256_srli_epi16::<6>(c)),
                    _mm256_loadu_si256(ab.cast()),
                );
                s = _mm256_add_epi16(
                    s,
                    _mm256_maddubs_epi16(
                        q(_mm256_srli_epi16::<4>(c)),
                        _mm256_loadu_si256(ab.add(32).cast()),
                    ),
                );
                s = _mm256_add_epi16(
                    s,
                    _mm256_maddubs_epi16(
                        q(_mm256_srli_epi16::<2>(c)),
                        _mm256_loadu_si256(ab.add(64).cast()),
                    ),
                );
                s = _mm256_add_epi16(
                    s,
                    _mm256_maddubs_epi16(q(c), _mm256_loadu_si256(ab.add(96).cast())),
                );
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s, ones));
            }
            let h = _mm_add_epi32(
                _mm256_castsi256_si128(acc),
                _mm256_extracti128_si256::<1>(acc),
            );
            *out = hsum_epi32(h) - a_sum;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// int8 x int8 GEMV (the head screen): `pmaddubsw(|q|, a·sign(q))`, both in -127..=127, so a
// pair is at most 2·127² = 32,258 (no saturation); `pmaddwd` widens every block.

fn i8_tail(q: &[i8], a: &[i8], from: usize) -> i32 {
    q[from..]
        .iter()
        .zip(&a[from..])
        .map(|(&w, &v)| i32::from(w) * i32::from(v))
        .sum()
}

/// # Safety
/// SSSE3; `q.len() == y.len() * a.len()`.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn i8_gemv_ssse3(q: &[i8], a: &[i8], y: &mut [i32]) {
    let n = a.len();
    let ones = _mm_set1_epi16(1);
    for (r, out) in y.iter_mut().enumerate() {
        let row = &q[r * n..(r + 1) * n];
        let mut acc = _mm_setzero_si128();
        for i in 0..n / 16 {
            // SAFETY: bytes `16i..16i+16` of the row and of `a`.
            unsafe {
                let w = _mm_loadu_si128(row.as_ptr().add(16 * i).cast());
                let v = _mm_loadu_si128(a.as_ptr().add(16 * i).cast());
                let p = _mm_maddubs_epi16(_mm_abs_epi8(w), _mm_sign_epi8(v, w));
                acc = _mm_add_epi32(acc, _mm_madd_epi16(p, ones));
            }
        }
        *out = hsum_epi32(acc) + i8_tail(row, a, n / 16 * 16);
    }
}

/// # Safety
/// AVX2; `q.len() == y.len() * a.len()`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn i8_gemv_avx2(q: &[i8], a: &[i8], y: &mut [i32]) {
    let n = a.len();
    let ones = _mm256_set1_epi16(1);
    for (r, out) in y.iter_mut().enumerate() {
        let row = &q[r * n..(r + 1) * n];
        let mut acc = _mm256_setzero_si256();
        for i in 0..n / 32 {
            // SAFETY: bytes `32i..32i+32` of the row and of `a`.
            unsafe {
                let w = _mm256_loadu_si256(row.as_ptr().add(32 * i).cast());
                let v = _mm256_loadu_si256(a.as_ptr().add(32 * i).cast());
                let p = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(v, w));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
            }
        }
        let h = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        *out = hsum_epi32(h) + i8_tail(row, a, n / 32 * 32);
    }
}

// ---------------------------------------------------------------------------------------------
// I2_S GEMM (prefill): each 32-byte code block unpacked once, applied to a tile of `T` tokens
// held in `T` i32 accumulators. Codes are <= 2 (refused at load otherwise), so a block's four
// `pmaddubsw` sums stay under 4·512 = 2,048 per i16 lane before the `pmaddwd` widen.

/// AVX-VNNI GEMV: `vpdpbusd` (unsigned codes <= 2 times signed activations, four products summed
/// straight into each i32 lane, no saturating step) replaces `pmaddubsw` + `paddw` + `pmaddwd` +
/// `paddd`.
///
/// # Safety
/// AVX2 + AVX-VNNI; shapes as [`gemv_avx2`].
#[target_feature(enable = "avx2,avxvnni")]
pub(super) unsafe fn gemv_vnni(codes: &[u8], a: &[i8], a_sum: i32, y: &mut [i32]) {
    let cols = a.len();
    let rpc = cols / 4;
    let mask = _mm256_set1_epi8(3);
    for (r, out) in y.iter_mut().enumerate() {
        // SAFETY: as `gemv_avx2`: every read is inside row `r` and the activations.
        unsafe {
            let row = codes.as_ptr().add(r * rpc);
            let mut acc = _mm256_setzero_si256();
            for b in 0..cols / 128 {
                let c = _mm256_loadu_si256(row.add(b * 32).cast());
                let ab = a.as_ptr().add(b * 128);
                acc = _mm256_dpbusd_avx_epi32(
                    acc,
                    _mm256_and_si256(_mm256_srli_epi16::<6>(c), mask),
                    _mm256_loadu_si256(ab.cast()),
                );
                acc = _mm256_dpbusd_avx_epi32(
                    acc,
                    _mm256_and_si256(_mm256_srli_epi16::<4>(c), mask),
                    _mm256_loadu_si256(ab.add(32).cast()),
                );
                acc = _mm256_dpbusd_avx_epi32(
                    acc,
                    _mm256_and_si256(_mm256_srli_epi16::<2>(c), mask),
                    _mm256_loadu_si256(ab.add(64).cast()),
                );
                acc = _mm256_dpbusd_avx_epi32(
                    acc,
                    _mm256_and_si256(c, mask),
                    _mm256_loadu_si256(ab.add(96).cast()),
                );
            }
            let h = _mm_add_epi32(
                _mm256_castsi256_si128(acc),
                _mm256_extracti128_si256::<1>(acc),
            );
            *out = hsum_epi32(h) - a_sum;
        }
    }
}

/// # Safety
/// AVX2 + AVX-VNNI; as [`gemm_tile_avx2`].
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
unsafe fn gemm_tile_vnni<const T: usize>(
    row: *const u8,
    a: *const i8,
    cols: usize,
    out: &mut [i32],
) {
    let mask = _mm256_set1_epi8(3);
    let mut acc = [_mm256_setzero_si256(); T];
    for b in 0..cols / 128 {
        // SAFETY: block `b` is inside the row and inside every token's activations.
        unsafe {
            let c = _mm256_loadu_si256(row.add(b * 32).cast());
            let q0 = _mm256_and_si256(_mm256_srli_epi16::<6>(c), mask);
            let q1 = _mm256_and_si256(_mm256_srli_epi16::<4>(c), mask);
            let q2 = _mm256_and_si256(_mm256_srli_epi16::<2>(c), mask);
            let q3 = _mm256_and_si256(c, mask);
            for (t, acc) in acc.iter_mut().enumerate() {
                let ab = a.add(t * cols + b * 128);
                let mut v = _mm256_dpbusd_avx_epi32(*acc, q0, _mm256_loadu_si256(ab.cast()));
                v = _mm256_dpbusd_avx_epi32(v, q1, _mm256_loadu_si256(ab.add(32).cast()));
                v = _mm256_dpbusd_avx_epi32(v, q2, _mm256_loadu_si256(ab.add(64).cast()));
                *acc = _mm256_dpbusd_avx_epi32(v, q3, _mm256_loadu_si256(ab.add(96).cast()));
            }
        }
    }
    for (o, v) in out.iter_mut().zip(acc) {
        *o = hsum_epi32(_mm_add_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256::<1>(v),
        ));
    }
}

/// Two rows by `T` tokens: each activation load feeds two `vpdpbusd` (one a row) and each
/// unpacked code vector `T`, against one a load in the 1-row tile (V1: the 1-row tile is
/// load-bound). 2 x 4 holds 8 accumulators, 2 code and 2 unpacked vectors, one activation and
/// the mask in 16 `ymm`.
///
/// # Safety
/// AVX2 + AVX-VNNI; `r0`, `r1` readable for `cols / 4` bytes, `a` for `T` tokens of `cols`.
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
unsafe fn gemm_tile_vnni_2r<const T: usize>(
    r0: *const u8,
    r1: *const u8,
    a: *const i8,
    cols: usize,
    o0: &mut [i32],
    o1: &mut [i32],
) {
    let mask = _mm256_set1_epi8(3);
    let mut acc0 = [_mm256_setzero_si256(); T];
    let mut acc1 = [_mm256_setzero_si256(); T];
    for b in 0..cols / 128 {
        // SAFETY: block `b` is inside both rows and inside every token's activations.
        unsafe {
            let c0 = _mm256_loadu_si256(r0.add(b * 32).cast());
            let c1 = _mm256_loadu_si256(r1.add(b * 32).cast());
            macro_rules! group {
                ($sh:literal, $g:literal) => {
                    let q0 = _mm256_and_si256(_mm256_srli_epi16::<$sh>(c0), mask);
                    let q1 = _mm256_and_si256(_mm256_srli_epi16::<$sh>(c1), mask);
                    for t in 0..T {
                        let v = _mm256_loadu_si256(a.add(t * cols + b * 128 + $g * 32).cast());
                        acc0[t] = _mm256_dpbusd_avx_epi32(acc0[t], q0, v);
                        acc1[t] = _mm256_dpbusd_avx_epi32(acc1[t], q1, v);
                    }
                };
            }
            group!(6, 0);
            group!(4, 1);
            group!(2, 2);
            group!(0, 3);
        }
    }
    let h = |v: __m256i| {
        hsum_epi32(_mm_add_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256::<1>(v),
        ))
    };
    for t in 0..T {
        o0[t] = h(acc0[t]);
        o1[t] = h(acc1[t]);
    }
}

/// # Safety
/// AVX2 + AVX-VNNI; shapes as [`gemm_avx2`].
#[target_feature(enable = "avx2,avxvnni")]
pub(super) unsafe fn gemm_vnni(
    codes: &[u8],
    cols: usize,
    m: usize,
    a: &[i8],
    a_sums: &[i32],
    y: &mut [i32],
) {
    let rpc = cols / 4;
    let rows = y.len() / m;
    let mut r = 0;
    while r + 2 <= rows {
        let (y0, y1) = y[r * m..(r + 2) * m].split_at_mut(m);
        // SAFETY: rows `r`, `r + 1` and every tile's tokens are inside the slices.
        unsafe {
            let (p0, p1) = (
                codes.as_ptr().add(r * rpc),
                codes.as_ptr().add((r + 1) * rpc),
            );
            let mut t = 0;
            while t + 4 <= m {
                gemm_tile_vnni_2r::<4>(
                    p0,
                    p1,
                    a.as_ptr().add(t * cols),
                    cols,
                    &mut y0[t..t + 4],
                    &mut y1[t..t + 4],
                );
                t += 4;
            }
            while t < m {
                gemm_tile_vnni_2r::<1>(
                    p0,
                    p1,
                    a.as_ptr().add(t * cols),
                    cols,
                    &mut y0[t..=t],
                    &mut y1[t..=t],
                );
                t += 1;
            }
        }
        for (o, &s) in y0
            .iter_mut()
            .chain(y1.iter_mut())
            .zip(a_sums.iter().chain(a_sums))
        {
            *o -= s;
        }
        r += 2;
    }
    if r < rows {
        let yr = &mut y[r * m..(r + 1) * m];
        // SAFETY: the last row and every tile's tokens are inside the slices.
        unsafe {
            let row = codes.as_ptr().add(r * rpc);
            let mut t = 0;
            while t + GEMM_TILE <= m {
                gemm_tile_vnni::<GEMM_TILE>(
                    row,
                    a.as_ptr().add(t * cols),
                    cols,
                    &mut yr[t..t + GEMM_TILE],
                );
                t += GEMM_TILE;
            }
            while t < m {
                gemm_tile_vnni::<1>(row, a.as_ptr().add(t * cols), cols, &mut yr[t..=t]);
                t += 1;
            }
        }
        for (o, &s) in yr.iter_mut().zip(a_sums) {
            *o -= s;
        }
    }
}

/// Tokens a GEMM tile holds.
const GEMM_TILE: usize = 8;

/// # Safety
/// AVX2; `row` readable for `cols / 4` bytes, `a` for `T` tokens of `cols`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn gemm_tile_avx2<const T: usize>(
    row: *const u8,
    a: *const i8,
    cols: usize,
    out: &mut [i32],
) {
    let mask = _mm256_set1_epi8(3);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_si256(); T];
    for b in 0..cols / 128 {
        // SAFETY: block `b` is inside the row and inside every token's activations.
        unsafe {
            let c = _mm256_loadu_si256(row.add(b * 32).cast());
            let q0 = _mm256_and_si256(_mm256_srli_epi16::<6>(c), mask);
            let q1 = _mm256_and_si256(_mm256_srli_epi16::<4>(c), mask);
            let q2 = _mm256_and_si256(_mm256_srli_epi16::<2>(c), mask);
            let q3 = _mm256_and_si256(c, mask);
            for (t, acc) in acc.iter_mut().enumerate() {
                let ab = a.add(t * cols + b * 128);
                let mut s = _mm256_maddubs_epi16(q0, _mm256_loadu_si256(ab.cast()));
                s = _mm256_add_epi16(
                    s,
                    _mm256_maddubs_epi16(q1, _mm256_loadu_si256(ab.add(32).cast())),
                );
                s = _mm256_add_epi16(
                    s,
                    _mm256_maddubs_epi16(q2, _mm256_loadu_si256(ab.add(64).cast())),
                );
                s = _mm256_add_epi16(
                    s,
                    _mm256_maddubs_epi16(q3, _mm256_loadu_si256(ab.add(96).cast())),
                );
                *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(s, ones));
            }
        }
    }
    for (o, v) in out.iter_mut().zip(acc) {
        *o = hsum_epi32(_mm_add_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256::<1>(v),
        ));
    }
}

/// # Safety
/// AVX2; `codes` whole rows of `cols / 4` bytes, `a` `m · cols`, `a_sums` `m`, `y` `rows · m`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn gemm_avx2(
    codes: &[u8],
    cols: usize,
    m: usize,
    a: &[i8],
    a_sums: &[i32],
    y: &mut [i32],
) {
    let rpc = cols / 4;
    for (r, yr) in y.chunks_exact_mut(m).enumerate() {
        // SAFETY: row `r` and every tile's tokens are inside the slices (asserted by the caller).
        unsafe {
            let row = codes.as_ptr().add(r * rpc);
            let mut t = 0;
            while t + GEMM_TILE <= m {
                gemm_tile_avx2::<GEMM_TILE>(
                    row,
                    a.as_ptr().add(t * cols),
                    cols,
                    &mut yr[t..t + GEMM_TILE],
                );
                t += GEMM_TILE;
            }
            while t < m {
                gemm_tile_avx2::<1>(row, a.as_ptr().add(t * cols), cols, &mut yr[t..=t]);
                t += 1;
            }
        }
        for (o, &s) in yr.iter_mut().zip(a_sums) {
            *o -= s;
        }
    }
}

/// # Safety
/// SSSE3; as [`gemm_tile_avx2`].
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn gemm_tile_ssse3<const T: usize>(
    row: *const u8,
    a: *const i8,
    cols: usize,
    out: &mut [i32],
) {
    let mask = _mm_set1_epi8(3);
    let ones = _mm_set1_epi16(1);
    let mut acc = [_mm_setzero_si128(); T];
    for b in 0..cols / 128 {
        // SAFETY: block `b` is inside the row and inside every token's activations.
        unsafe {
            let c0 = _mm_loadu_si128(row.add(b * 32).cast());
            let c1 = _mm_loadu_si128(row.add(b * 32 + 16).cast());
            let q = [
                _mm_and_si128(_mm_srli_epi16::<6>(c0), mask),
                _mm_and_si128(_mm_srli_epi16::<6>(c1), mask),
                _mm_and_si128(_mm_srli_epi16::<4>(c0), mask),
                _mm_and_si128(_mm_srli_epi16::<4>(c1), mask),
                _mm_and_si128(_mm_srli_epi16::<2>(c0), mask),
                _mm_and_si128(_mm_srli_epi16::<2>(c1), mask),
                _mm_and_si128(c0, mask),
                _mm_and_si128(c1, mask),
            ];
            for (t, acc) in acc.iter_mut().enumerate() {
                let ab = a.add(t * cols + b * 128);
                let mut s = _mm_setzero_si128();
                for (k, &qk) in q.iter().enumerate() {
                    s = _mm_add_epi16(
                        s,
                        _mm_maddubs_epi16(qk, _mm_loadu_si128(ab.add(16 * k).cast())),
                    );
                }
                *acc = _mm_add_epi32(*acc, _mm_madd_epi16(s, ones));
            }
        }
    }
    for (o, v) in out.iter_mut().zip(acc) {
        *o = hsum_epi32(v);
    }
}

/// # Safety
/// SSSE3; shapes as [`gemm_avx2`].
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn gemm_ssse3(
    codes: &[u8],
    cols: usize,
    m: usize,
    a: &[i8],
    a_sums: &[i32],
    y: &mut [i32],
) {
    let rpc = cols / 4;
    for (r, yr) in y.chunks_exact_mut(m).enumerate() {
        // SAFETY: as `gemm_avx2`.
        unsafe {
            let row = codes.as_ptr().add(r * rpc);
            let mut t = 0;
            while t + GEMM_TILE <= m {
                gemm_tile_ssse3::<GEMM_TILE>(
                    row,
                    a.as_ptr().add(t * cols),
                    cols,
                    &mut yr[t..t + GEMM_TILE],
                );
                t += GEMM_TILE;
            }
            while t < m {
                gemm_tile_ssse3::<1>(row, a.as_ptr().add(t * cols), cols, &mut yr[t..=t]);
                t += 1;
            }
        }
        for (o, &s) in yr.iter_mut().zip(a_sums) {
            *o -= s;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// int8 x f32 mat-vec (the head screen) in the canonical 16-lane order.

/// Sixteen int8 at `p` sign-extended to four `f32x4` (SSE2: unpack with itself, shift down).
///
/// # Safety
/// `p` readable for 16 bytes.
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn load16_i8_f32(p: *const i8) -> [__m128; 4] {
    // SAFETY: the caller guarantees 16 readable bytes.
    let v = unsafe { _mm_loadu_si128(p.cast()) };
    let lo16 = _mm_srai_epi16::<8>(_mm_unpacklo_epi8(v, v));
    let hi16 = _mm_srai_epi16::<8>(_mm_unpackhi_epi8(v, v));
    let w = |h: __m128i, low: bool| {
        let u = if low {
            _mm_unpacklo_epi16(h, h)
        } else {
            _mm_unpackhi_epi16(h, h)
        };
        _mm_cvtepi32_ps(_mm_srai_epi32::<16>(u))
    };
    [w(lo16, true), w(lo16, false), w(hi16, true), w(hi16, false)]
}

/// # Safety
/// SSSE3; `q.len() == out.len() * x.len()`.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn i8f_matvec_ssse3(q: &[i8], x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let n16 = n / LANES;
    for (r, o) in out.iter_mut().enumerate() {
        let row = &q[r * n..(r + 1) * n];
        // SAFETY: chunk `i < n16` reads row bytes and `x` floats `16i..16i+16`.
        unsafe {
            let (mut a0, mut a1, mut a2, mut a3) = (
                _mm_setzero_ps(),
                _mm_setzero_ps(),
                _mm_setzero_ps(),
                _mm_setzero_ps(),
            );
            for i in 0..n16 {
                let [w0, w1, w2, w3] = load16_i8_f32(row.as_ptr().add(16 * i));
                let xp = x.as_ptr().add(16 * i);
                a0 = _mm_add_ps(a0, _mm_mul_ps(w0, _mm_loadu_ps(xp)));
                a1 = _mm_add_ps(a1, _mm_mul_ps(w1, _mm_loadu_ps(xp.add(4))));
                a2 = _mm_add_ps(a2, _mm_mul_ps(w2, _mm_loadu_ps(xp.add(8))));
                a3 = _mm_add_ps(a3, _mm_mul_ps(w3, _mm_loadu_ps(xp.add(12))));
            }
            if n.is_multiple_of(LANES) {
                *o = reduce4_canon(_mm_add_ps(_mm_add_ps(a0, a2), _mm_add_ps(a1, a3)));
            } else {
                let mut acc = [0f32; LANES];
                _mm_storeu_ps(acc.as_mut_ptr(), a0);
                _mm_storeu_ps(acc.as_mut_ptr().add(4), a1);
                _mm_storeu_ps(acc.as_mut_ptr().add(8), a2);
                _mm_storeu_ps(acc.as_mut_ptr().add(12), a3);
                *o = super::i8f_finish(acc, row, x, n16 * LANES);
            }
        }
    }
}

/// # Safety
/// AVX2; `q.len() == out.len() * x.len()`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn i8f_matvec_avx2(q: &[i8], x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let n16 = n / LANES;
    for (r, o) in out.iter_mut().enumerate() {
        let row = &q[r * n..(r + 1) * n];
        // SAFETY: chunk `i < n16` reads row bytes and `x` floats `16i..16i+16`.
        unsafe {
            let (mut lo, mut hi) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            for i in 0..n16 {
                let v = _mm_loadu_si128(row.as_ptr().add(16 * i).cast());
                let w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
                let w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(v)));
                let xp = x.as_ptr().add(16 * i);
                lo = _mm256_add_ps(lo, _mm256_mul_ps(w0, _mm256_loadu_ps(xp)));
                hi = _mm256_add_ps(hi, _mm256_mul_ps(w1, _mm256_loadu_ps(xp.add(8))));
            }
            if n.is_multiple_of(LANES) {
                let v = _mm256_add_ps(lo, hi);
                *o = reduce4_canon(_mm_add_ps(
                    _mm256_castps256_ps128(v),
                    _mm256_extractf128_ps::<1>(v),
                ));
            } else {
                let mut acc = [0f32; LANES];
                _mm256_storeu_ps(acc.as_mut_ptr(), lo);
                _mm256_storeu_ps(acc.as_mut_ptr().add(8), hi);
                *o = super::i8f_finish(acc, row, x, n16 * LANES);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// f16 matrix-vector in the canonical 16-lane order.

/// # Safety
/// SSSE3; rows `r < out.len()` at `base + r·row_bytes` readable for `2 * x.len()` bytes.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn f16_matvec_ssse3(
    base: *const u8,
    row_bytes: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let n16 = x.len() / LANES;
    for (r, o) in out.iter_mut().enumerate() {
        // SAFETY: chunk `i < n16` reads bytes `32i..32i+32` of row `r` and `x[16i..16i+16]`.
        unsafe {
            let row = base.add(r * row_bytes);
            let (mut a0, mut a1, mut a2, mut a3) = (
                _mm_setzero_ps(),
                _mm_setzero_ps(),
                _mm_setzero_ps(),
                _mm_setzero_ps(),
            );
            for i in 0..n16 {
                let p = row.add(32 * i);
                let (h0, h1) = load8h(p);
                let (h2, h3) = load8h(p.add(16));
                let xp = x.as_ptr().add(16 * i);
                a0 = _mm_add_ps(a0, _mm_mul_ps(h0, _mm_loadu_ps(xp)));
                a1 = _mm_add_ps(a1, _mm_mul_ps(h1, _mm_loadu_ps(xp.add(4))));
                a2 = _mm_add_ps(a2, _mm_mul_ps(h2, _mm_loadu_ps(xp.add(8))));
                a3 = _mm_add_ps(a3, _mm_mul_ps(h3, _mm_loadu_ps(xp.add(12))));
            }
            if x.len().is_multiple_of(LANES) {
                *o = reduce4_canon(_mm_add_ps(_mm_add_ps(a0, a2), _mm_add_ps(a1, a3)));
            } else {
                let mut acc = [0f32; LANES];
                _mm_storeu_ps(acc.as_mut_ptr(), a0);
                _mm_storeu_ps(acc.as_mut_ptr().add(4), a1);
                _mm_storeu_ps(acc.as_mut_ptr().add(8), a2);
                _mm_storeu_ps(acc.as_mut_ptr().add(12), a3);
                *o = finish_dot(acc, row, x, n16 * LANES);
            }
        }
    }
}

/// # Safety
/// AVX2 + F16C; rows as [`f16_matvec_ssse3`].
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn f16_matvec_avx2(
    base: *const u8,
    row_bytes: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let n16 = x.len() / LANES;
    for (r, o) in out.iter_mut().enumerate() {
        // SAFETY: as `f16_matvec_ssse3`.
        unsafe {
            let row = base.add(r * row_bytes);
            let (mut lo, mut hi) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            for i in 0..n16 {
                let h = _mm256_loadu_si256(row.add(32 * i).cast());
                let xp = x.as_ptr().add(16 * i);
                lo = _mm256_add_ps(
                    lo,
                    _mm256_mul_ps(
                        _mm256_cvtph_ps(_mm256_castsi256_si128(h)),
                        _mm256_loadu_ps(xp),
                    ),
                );
                hi = _mm256_add_ps(
                    hi,
                    _mm256_mul_ps(
                        _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(h)),
                        _mm256_loadu_ps(xp.add(8)),
                    ),
                );
            }
            if x.len().is_multiple_of(LANES) {
                let v = _mm256_add_ps(lo, hi);
                *o = reduce4_canon(_mm_add_ps(
                    _mm256_castps256_ps128(v),
                    _mm256_extractf128_ps::<1>(v),
                ));
            } else {
                let mut acc = [0f32; LANES];
                _mm256_storeu_ps(acc.as_mut_ptr(), lo);
                _mm256_storeu_ps(acc.as_mut_ptr().add(8), hi);
                *o = finish_dot(acc, row, x, n16 * LANES);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Attention values: out[d] = Σ_t p[t]·v_t[d], lanes across d, t sequential per lane.

fn weighted_tail(p: &[f32], base: *const u16, stride: usize, out: &mut [f32], from: usize) {
    let t16 = num::f16_table();
    for (d, o) in out.iter_mut().enumerate().skip(from) {
        let mut acc = 0f32;
        for (t, &pt) in p.iter().enumerate() {
            // SAFETY: the entry point asserted every row `t` spans `out.len()` values.
            acc += pt * t16[usize::from(unsafe { *base.add(t * stride + d) })];
        }
        *o = acc;
    }
}

/// # Safety
/// SSSE3; rows `t < p.len()` at `base + t·stride` readable for `out.len()` values.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn weighted_sum_ssse3(
    p: &[f32],
    base: *const u16,
    stride: usize,
    out: &mut [f32],
) {
    let n16 = out.len() / 16;
    for blk in 0..n16 {
        // SAFETY: values `16·blk..16·blk+16` of each row are inside it; `out` likewise.
        unsafe {
            let (mut a0, mut a1, mut a2, mut a3) = (
                _mm_setzero_ps(),
                _mm_setzero_ps(),
                _mm_setzero_ps(),
                _mm_setzero_ps(),
            );
            for (t, &pt) in p.iter().enumerate() {
                let q = base.add(t * stride + 16 * blk).cast::<u8>();
                let pv = _mm_set1_ps(pt);
                let (h0, h1) = load8h(q);
                let (h2, h3) = load8h(q.add(16));
                a0 = _mm_add_ps(a0, _mm_mul_ps(pv, h0));
                a1 = _mm_add_ps(a1, _mm_mul_ps(pv, h1));
                a2 = _mm_add_ps(a2, _mm_mul_ps(pv, h2));
                a3 = _mm_add_ps(a3, _mm_mul_ps(pv, h3));
            }
            let o = out.as_mut_ptr().add(16 * blk);
            _mm_storeu_ps(o, a0);
            _mm_storeu_ps(o.add(4), a1);
            _mm_storeu_ps(o.add(8), a2);
            _mm_storeu_ps(o.add(12), a3);
        }
    }
    weighted_tail(p, base, stride, out, n16 * 16);
}

/// # Safety
/// AVX2 + F16C; rows as [`weighted_sum_ssse3`].
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn weighted_sum_avx2(
    p: &[f32],
    base: *const u16,
    stride: usize,
    out: &mut [f32],
) {
    let n32 = out.len() / 32;
    for blk in 0..n32 {
        // SAFETY: values `32·blk..32·blk+32` of each row are inside it; `out` likewise.
        unsafe {
            let (mut a0, mut a1, mut a2, mut a3) = (
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );
            for (t, &pt) in p.iter().enumerate() {
                let q = base.add(t * stride + 32 * blk);
                let pv = _mm256_set1_ps(pt);
                let h01 = _mm256_loadu_si256(q.cast());
                let h23 = _mm256_loadu_si256(q.add(16).cast());
                a0 = _mm256_add_ps(
                    a0,
                    _mm256_mul_ps(pv, _mm256_cvtph_ps(_mm256_castsi256_si128(h01))),
                );
                a1 = _mm256_add_ps(
                    a1,
                    _mm256_mul_ps(pv, _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(h01))),
                );
                a2 = _mm256_add_ps(
                    a2,
                    _mm256_mul_ps(pv, _mm256_cvtph_ps(_mm256_castsi256_si128(h23))),
                );
                a3 = _mm256_add_ps(
                    a3,
                    _mm256_mul_ps(pv, _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(h23))),
                );
            }
            let o = out.as_mut_ptr().add(32 * blk);
            _mm256_storeu_ps(o, a0);
            _mm256_storeu_ps(o.add(8), a1);
            _mm256_storeu_ps(o.add(16), a2);
            _mm256_storeu_ps(o.add(24), a3);
        }
    }
    weighted_tail(p, base, stride, out, n32 * 32);
}

// ---------------------------------------------------------------------------------------------
// Activation quantisation.

/// # Safety
/// SSSE3; equal lengths.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn act_quant_ssse3(x: &[f32], out: &mut [i8]) -> (f32, i32) {
    let n = x.len();
    let abs = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
    let mut m = _mm_setzero_ps();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`. `max(v, m)` returns `m` for a NaN `v`, as the oracle's fold.
        m = _mm_max_ps(
            _mm_and_ps(unsafe { _mm_loadu_ps(x.as_ptr().add(4 * i)) }, abs),
            m,
        );
    }
    let mut lanes = [0f32; 4];
    // SAFETY: four floats into a four-float array.
    unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), m) };
    let amax = lanes
        .iter()
        .chain(x[n / 4 * 4..].iter())
        .fold(0f32, |a, v| a.max(v.abs()));
    let s = act_scale(amax);
    let sv = _mm_set1_ps(s);
    let q = |v: __m128| {
        let v = _mm_mul_ps(v, sv);
        let v = _mm_and_ps(v, _mm_cmpord_ps(v, v));
        // No clamp: |x·s| <= 127·(1 + 2u) because s = 127 / max|x| (or |x| < 1e-5 under the
        // floor), so the round-to-nearest-even conversion lands in -127..=127 and the clamp the
        // oracle applies never acts. A NaN is zeroed first (the oracle maps it to 0).
        _mm_cvtps_epi32(v)
    };
    // The codes' sum rides along: the i32 lanes are already in -128..=127, so the saturating
    // packs store exactly them.
    let mut acc = _mm_setzero_si128();
    for i in 0..n / 16 {
        // SAFETY: `16i + 16 <= n` for both slices.
        unsafe {
            let p = x.as_ptr().add(16 * i);
            let (q0, q1) = (q(_mm_loadu_ps(p)), q(_mm_loadu_ps(p.add(4))));
            let (q2, q3) = (q(_mm_loadu_ps(p.add(8))), q(_mm_loadu_ps(p.add(12))));
            acc = _mm_add_epi32(
                acc,
                _mm_add_epi32(_mm_add_epi32(q0, q1), _mm_add_epi32(q2, q3)),
            );
            let a = _mm_packs_epi32(q0, q1);
            let b = _mm_packs_epi32(q2, q3);
            _mm_storeu_si128(out.as_mut_ptr().add(16 * i).cast(), _mm_packs_epi16(a, b));
        }
    }
    let mut sum = hsum_epi32(acc);
    for i in n / 16 * 16..n {
        out[i] = quant_one(x[i], s);
        sum += i32::from(out[i]);
    }
    (s, sum)
}

/// # Safety
/// AVX2; equal lengths.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn act_quant_avx2(x: &[f32], out: &mut [i8]) -> (f32, i32) {
    let n = x.len();
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut m = _mm256_setzero_ps();
    for i in 0..n / 8 {
        // SAFETY: `8i + 8 <= n`.
        m = _mm256_max_ps(
            _mm256_and_ps(unsafe { _mm256_loadu_ps(x.as_ptr().add(8 * i)) }, abs),
            m,
        );
    }
    let mut lanes = [0f32; 8];
    // SAFETY: eight floats into an eight-float array.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), m) };
    let amax = lanes
        .iter()
        .chain(x[n / 8 * 8..].iter())
        .fold(0f32, |a, v| a.max(v.abs()));
    let s = act_scale(amax);
    let sv = _mm256_set1_ps(s);
    let q = |v: __m256| {
        let v = _mm256_mul_ps(v, sv);
        let v = _mm256_and_ps(v, _mm256_cmp_ps::<_CMP_ORD_Q>(v, v));
        // No clamp: |x·s| <= 127·(1 + 2u) because s = 127 / max|x| (or |x| < 1e-5 under the
        // floor), so the round-to-nearest-even conversion lands in -127..=127 and the clamp the
        // oracle applies never acts. A NaN is zeroed first (the oracle maps it to 0).
        // `cvtps2dq` rounds to nearest even itself (MXCSR default), so no separate round.
        _mm256_cvtps_epi32(v)
    };
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    // The codes' sum rides along: the i32 lanes are already in -128..=127, so the saturating
    // packs store exactly them.
    let mut acc = _mm256_setzero_si256();
    for i in 0..n / 32 {
        // SAFETY: `32i + 32 <= n` for both slices.
        unsafe {
            let p = x.as_ptr().add(32 * i);
            let (q0, q1) = (q(_mm256_loadu_ps(p)), q(_mm256_loadu_ps(p.add(8))));
            let (q2, q3) = (q(_mm256_loadu_ps(p.add(16))), q(_mm256_loadu_ps(p.add(24))));
            acc = _mm256_add_epi32(
                acc,
                _mm256_add_epi32(_mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3)),
            );
            let a = _mm256_packs_epi32(q0, q1);
            let b = _mm256_packs_epi32(q2, q3);
            // The packs work within 128-bit lanes; this puts the 4-byte groups back in order.
            let v = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(a, b), order);
            _mm256_storeu_si256(out.as_mut_ptr().add(32 * i).cast(), v);
        }
    }
    let mut sum = hsum_epi32(_mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    ));
    for i in n / 32 * 32..n {
        out[i] = quant_one(x[i], s);
        sum += i32::from(out[i]);
    }
    (s, sum)
}

// ---------------------------------------------------------------------------------------------
// RMS norm and softmax: f64 sums in the canonical 4-lane order.

/// # Safety
/// SSSE3; equal lengths.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn rms_norm_ssse3(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let (mut a01, mut a23) = (_mm_setzero_pd(), _mm_setzero_pd());
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        let v = unsafe { _mm_loadu_ps(x.as_ptr().add(4 * i)) };
        let (lo, hi) = (_mm_cvtps_pd(v), _mm_cvtps_pd(_mm_movehl_ps(v, v)));
        a01 = _mm_add_pd(a01, _mm_mul_pd(lo, lo));
        a23 = _mm_add_pd(a23, _mm_mul_pd(hi, hi));
    }
    let mut acc = [0f64; 4];
    // SAFETY: two doubles into each half of a four-double array.
    unsafe {
        _mm_storeu_pd(acc.as_mut_ptr(), a01);
        _mm_storeu_pd(acc.as_mut_ptr().add(2), a23);
    }
    for (j, &v) in x[n / 4 * 4..].iter().enumerate() {
        acc[j] += f64::from(v) * f64::from(v);
    }
    let sv = _mm_set1_ps(num::rms_scale(num::reduce4_f64(acc), n, eps));
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for all three slices.
        unsafe {
            let v = _mm_mul_ps(
                _mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(4 * i)), sv),
                _mm_loadu_ps(w.as_ptr().add(4 * i)),
            );
            _mm_storeu_ps(out.as_mut_ptr().add(4 * i), v);
        }
    }
    let s = _mm_cvtss_f32(sv);
    for i in n / 4 * 4..n {
        out[i] = (x[i] * s) * w[i];
    }
}

/// # Safety
/// AVX2 + FMA; equal lengths.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn rms_norm_avx2(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut a = _mm256_setzero_pd();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        let v = _mm256_cvtps_pd(unsafe { _mm_loadu_ps(x.as_ptr().add(4 * i)) });
        // f32 squared is exact in f64 (48 of 53 bits), so the fused multiply-add rounds exactly as
        // `acc + v * v` (the oracle's order and bits).
        a = _mm256_fmadd_pd(v, v, a);
    }
    let mut acc = [0f64; 4];
    // SAFETY: four doubles into a four-double array.
    unsafe { _mm256_storeu_pd(acc.as_mut_ptr(), a) };
    for (j, &v) in x[n / 4 * 4..].iter().enumerate() {
        acc[j] += f64::from(v) * f64::from(v);
    }
    let s = num::rms_scale(num::reduce4_f64(acc), n, eps);
    let sv = _mm256_set1_ps(s);
    for i in 0..n / 8 {
        // SAFETY: `8i + 8 <= n` for all three slices.
        unsafe {
            let v = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(8 * i)), sv),
                _mm256_loadu_ps(w.as_ptr().add(8 * i)),
            );
            _mm256_storeu_ps(out.as_mut_ptr().add(8 * i), v);
        }
    }
    for i in n / 8 * 8..n {
        out[i] = (x[i] * s) * w[i];
    }
}

fn hmax_tail(lanes: &[f32], rest: &[f32]) -> f32 {
    lanes
        .iter()
        .chain(rest)
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
}

/// # Safety
/// SSSE3.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn softmax_ssse3(x: &mut [f32], scale: f32) {
    let n = x.len();
    let sc = _mm_set1_ps(scale);
    let mut m = _mm_set1_ps(f32::NEG_INFINITY);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            let v = _mm_mul_ps(_mm_loadu_ps(p), sc);
            _mm_storeu_ps(p, v);
            m = _mm_max_ps(v, m);
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v *= scale;
    }
    let mut lanes = [0f32; 4];
    // SAFETY: four floats into a four-float array.
    unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), m) };
    let max = hmax_tail(&lanes, &x[n / 4 * 4..]);
    let mv = _mm_set1_ps(max);
    let (mut a01, mut a23) = (_mm_setzero_pd(), _mm_setzero_pd());
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            let e = exp4(_mm_sub_ps(_mm_loadu_ps(p), mv));
            _mm_storeu_ps(p, e);
            a01 = _mm_add_pd(a01, _mm_cvtps_pd(e));
            a23 = _mm_add_pd(a23, _mm_cvtps_pd(_mm_movehl_ps(e, e)));
        }
    }
    let mut acc = [0f64; 4];
    // SAFETY: two doubles into each half of a four-double array.
    unsafe {
        _mm_storeu_pd(acc.as_mut_ptr(), a01);
        _mm_storeu_pd(acc.as_mut_ptr().add(2), a23);
    }
    for (j, v) in x[n / 4 * 4..].iter_mut().enumerate() {
        *v = num::exp_canon(*v - max);
        acc[j] += f64::from(*v);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the oracle scales by an f32"
    )]
    let inv = (1.0 / num::reduce4_f64(acc)) as f32;
    let iv = _mm_set1_ps(inv);
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            _mm_storeu_ps(p, _mm_mul_ps(_mm_loadu_ps(p), iv));
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v *= inv;
    }
}

/// # Safety
/// AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn softmax_avx2(x: &mut [f32], scale: f32) {
    let n = x.len();
    let sc = _mm256_set1_ps(scale);
    let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..n / 8 {
        // SAFETY: `8i + 8 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(8 * i);
            let v = _mm256_mul_ps(_mm256_loadu_ps(p), sc);
            _mm256_storeu_ps(p, v);
            m = _mm256_max_ps(v, m);
        }
    }
    for v in &mut x[n / 8 * 8..] {
        *v *= scale;
    }
    let mut lanes = [0f32; 8];
    // SAFETY: eight floats into an eight-float array.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), m) };
    let max = hmax_tail(&lanes, &x[n / 8 * 8..]);
    let mv = _mm256_set1_ps(max);
    let mut a = _mm256_setzero_pd();
    for i in 0..n / 8 {
        // SAFETY: `8i + 8 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(8 * i);
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(p), mv));
            _mm256_storeu_ps(p, e);
            // Elements 8i..8i+4 then 8i+4..8i+8 into lanes 0..4: each lane's sequence is the
            // oracle's.
            a = _mm256_add_pd(a, _mm256_cvtps_pd(_mm256_castps256_ps128(e)));
            a = _mm256_add_pd(a, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(e)));
        }
    }
    let mut acc = [0f64; 4];
    // SAFETY: four doubles into a four-double array.
    unsafe { _mm256_storeu_pd(acc.as_mut_ptr(), a) };
    for (j, v) in x[n / 8 * 8..].iter_mut().enumerate() {
        *v = num::exp_canon(*v - max);
        acc[j % 4] += f64::from(*v);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the oracle scales by an f32"
    )]
    let inv = (1.0 / num::reduce4_f64(acc)) as f32;
    let iv = _mm256_set1_ps(inv);
    for i in 0..n / 8 {
        // SAFETY: `8i + 8 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(8 * i);
            _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(p), iv));
        }
    }
    for v in &mut x[n / 8 * 8..] {
        *v *= inv;
    }
}

// ---------------------------------------------------------------------------------------------
// f16 conversions.

/// # Safety
/// SSSE3; `src.len() == 2 * dst.len()`.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn h2f_ssse3(src: &[u8], dst: &mut [f32]) {
    let n = dst.len();
    for i in 0..n / 8 {
        // SAFETY: bytes `16i..16i+16` and floats `8i..8i+8` are in bounds.
        unsafe {
            let (a, b) = load8h(src.as_ptr().add(16 * i));
            _mm_storeu_ps(dst.as_mut_ptr().add(8 * i), a);
            _mm_storeu_ps(dst.as_mut_ptr().add(8 * i + 4), b);
        }
    }
    let t = num::f16_table();
    for i in n / 8 * 8..n {
        dst[i] = t[usize::from(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]))];
    }
}

/// # Safety
/// AVX2 + F16C; `src.len() == 2 * dst.len()`.
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn h2f_avx2(src: &[u8], dst: &mut [f32]) {
    let n = dst.len();
    for i in 0..n / 8 {
        // SAFETY: bytes `16i..16i+16` and floats `8i..8i+8` are in bounds.
        unsafe {
            let h = _mm_loadu_si128(src.as_ptr().add(16 * i).cast());
            _mm256_storeu_ps(dst.as_mut_ptr().add(8 * i), _mm256_cvtph_ps(h));
        }
    }
    let t = num::f16_table();
    for i in n / 8 * 8..n {
        dst[i] = t[usize::from(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]))];
    }
}

/// # Safety
/// SSSE3; equal lengths.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn f2h_ssse3(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    for i in 0..n / 8 {
        // SAFETY: floats `8i..8i+8` and halves `8i..8i+8` are in bounds.
        unsafe {
            let p = src.as_ptr().add(8 * i);
            let v = pack_lo16(f2h4(_mm_loadu_ps(p)), f2h4(_mm_loadu_ps(p.add(4))));
            _mm_storeu_si128(dst.as_mut_ptr().add(8 * i).cast(), v);
        }
    }
    for i in n / 8 * 8..n {
        dst[i] = num::to_f16_bits(src[i]);
    }
}

/// # Safety
/// AVX2 + F16C; equal lengths.
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn f2h_avx2(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    for i in 0..n / 8 {
        // SAFETY: floats `8i..8i+8` and halves `8i..8i+8` are in bounds.
        unsafe {
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(
                src.as_ptr().add(8 * i),
            ));
            _mm_storeu_si128(dst.as_mut_ptr().add(8 * i).cast(), h);
        }
    }
    for i in n / 8 * 8..n {
        dst[i] = num::to_f16_bits(src[i]);
    }
}

/// # Safety
/// SSSE3.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn round_f16_ssse3(x: &mut [f32]) {
    let n = x.len();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            _mm_storeu_ps(p, h2f4(f2h4(_mm_loadu_ps(p))));
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v = num::round_f16(*v);
    }
}

/// # Safety
/// AVX2 + F16C.
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn round_f16_avx2(x: &mut [f32]) {
    let n = x.len();
    for i in 0..n / 8 {
        // SAFETY: `8i + 8 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(8 * i);
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(p));
            _mm256_storeu_ps(p, _mm256_cvtph_ps(h));
        }
    }
    for v in &mut x[n / 8 * 8..] {
        *v = num::round_f16(*v);
    }
}

// ---------------------------------------------------------------------------------------------
// Elementwise.

/// # Safety
/// SSSE3; equal lengths.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn finish_ssse3(sum: &[i32], act: f32, w: f32, out: &mut [f32]) {
    let n = sum.len();
    let (av, wv) = (_mm_set1_ps(act), _mm_set1_ps(w));
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for both slices.
        unsafe {
            let f = _mm_cvtepi32_ps(_mm_loadu_si128(sum.as_ptr().add(4 * i).cast()));
            _mm_storeu_ps(
                out.as_mut_ptr().add(4 * i),
                _mm_mul_ps(_mm_div_ps(f, av), wv),
            );
        }
    }
    num::finish_rows(&sum[n / 4 * 4..], act, w, &mut out[n / 4 * 4..]);
}

/// # Safety
/// AVX2; equal lengths.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn finish_avx2(sum: &[i32], act: f32, w: f32, out: &mut [f32]) {
    let n = sum.len();
    let (av, wv) = (_mm256_set1_ps(act), _mm256_set1_ps(w));
    for i in 0..n / 8 {
        // SAFETY: `8i + 8 <= n` for both slices.
        unsafe {
            let f = _mm256_cvtepi32_ps(_mm256_loadu_si256(sum.as_ptr().add(8 * i).cast()));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(8 * i),
                _mm256_mul_ps(_mm256_div_ps(f, av), wv),
            );
        }
    }
    num::finish_rows(&sum[n / 8 * 8..], act, w, &mut out[n / 8 * 8..]);
}

/// # Safety
/// SSSE3; equal lengths.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn relu2_mul_ssse3(g: &mut [f32], u: &[f32]) {
    let n = g.len();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for both slices. `max(g, 0)` returns 0 for a NaN or a -0 `g`,
        // as the oracle's `g > 0 ? g : 0`.
        unsafe {
            let p = g.as_mut_ptr().add(4 * i);
            let r = _mm_max_ps(_mm_loadu_ps(p), _mm_setzero_ps());
            _mm_storeu_ps(
                p,
                _mm_mul_ps(_mm_mul_ps(r, r), _mm_loadu_ps(u.as_ptr().add(4 * i))),
            );
        }
    }
    num::relu2_mul(&mut g[n / 4 * 4..], &u[n / 4 * 4..]);
}

/// # Safety
/// AVX2; equal lengths.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn relu2_mul_avx2(g: &mut [f32], u: &[f32]) {
    let n = g.len();
    for i in 0..n / 8 {
        // SAFETY: as `relu2_mul_ssse3`.
        unsafe {
            let p = g.as_mut_ptr().add(8 * i);
            let r = _mm256_max_ps(_mm256_loadu_ps(p), _mm256_setzero_ps());
            _mm256_storeu_ps(
                p,
                _mm256_mul_ps(_mm256_mul_ps(r, r), _mm256_loadu_ps(u.as_ptr().add(8 * i))),
            );
        }
    }
    num::relu2_mul(&mut g[n / 8 * 8..], &u[n / 8 * 8..]);
}

/// # Safety
/// SSSE3; equal lengths.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn add_assign_ssse3(x: &mut [f32], y: &[f32]) {
    let n = x.len();
    for i in 0..n / 4 {
        // SAFETY: `4i + 4 <= n` for both slices.
        unsafe {
            let p = x.as_mut_ptr().add(4 * i);
            _mm_storeu_ps(
                p,
                _mm_add_ps(_mm_loadu_ps(p), _mm_loadu_ps(y.as_ptr().add(4 * i))),
            );
        }
    }
    num::add_assign(&mut x[n / 4 * 4..], &y[n / 4 * 4..]);
}

/// # Safety
/// AVX2; equal lengths.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn add_assign_avx2(x: &mut [f32], y: &[f32]) {
    let n = x.len();
    for i in 0..n / 8 {
        // SAFETY: `8i + 8 <= n` for both slices.
        unsafe {
            let p = x.as_mut_ptr().add(8 * i);
            _mm256_storeu_ps(
                p,
                _mm256_add_ps(_mm256_loadu_ps(p), _mm256_loadu_ps(y.as_ptr().add(8 * i))),
            );
        }
    }
    num::add_assign(&mut x[n / 8 * 8..], &y[n / 8 * 8..]);
}

/// # Safety
/// SSSE3; `x.len() == 2 * cos.len() == 2 * sin.len()`.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn rope_ssse3(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    let (lo, hi) = x.split_at_mut(half);
    for i in 0..half / 4 {
        // SAFETY: `4i + 4 <= half` for all four slices.
        unsafe {
            let (pl, ph) = (lo.as_mut_ptr().add(4 * i), hi.as_mut_ptr().add(4 * i));
            let (x0, x1) = (_mm_loadu_ps(pl), _mm_loadu_ps(ph));
            let (c, s) = (
                _mm_loadu_ps(cos.as_ptr().add(4 * i)),
                _mm_loadu_ps(sin.as_ptr().add(4 * i)),
            );
            _mm_storeu_ps(pl, _mm_sub_ps(_mm_mul_ps(x0, c), _mm_mul_ps(x1, s)));
            _mm_storeu_ps(ph, _mm_add_ps(_mm_mul_ps(x0, s), _mm_mul_ps(x1, c)));
        }
    }
    for i in half / 4 * 4..half {
        let (x0, x1) = (lo[i], hi[i]);
        lo[i] = x0 * cos[i] - x1 * sin[i];
        hi[i] = x0 * sin[i] + x1 * cos[i];
    }
}

/// # Safety
/// AVX2; lengths as [`rope_ssse3`].
#[target_feature(enable = "avx2")]
pub(super) unsafe fn rope_avx2(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    let (lo, hi) = x.split_at_mut(half);
    for i in 0..half / 8 {
        // SAFETY: `8i + 8 <= half` for all four slices.
        unsafe {
            let (pl, ph) = (lo.as_mut_ptr().add(8 * i), hi.as_mut_ptr().add(8 * i));
            let (x0, x1) = (_mm256_loadu_ps(pl), _mm256_loadu_ps(ph));
            let (c, s) = (
                _mm256_loadu_ps(cos.as_ptr().add(8 * i)),
                _mm256_loadu_ps(sin.as_ptr().add(8 * i)),
            );
            _mm256_storeu_ps(
                pl,
                _mm256_sub_ps(_mm256_mul_ps(x0, c), _mm256_mul_ps(x1, s)),
            );
            _mm256_storeu_ps(
                ph,
                _mm256_add_ps(_mm256_mul_ps(x0, s), _mm256_mul_ps(x1, c)),
            );
        }
    }
    for i in half / 8 * 8..half {
        let (x0, x1) = (lo[i], hi[i]);
        lo[i] = x0 * cos[i] - x1 * sin[i];
        hi[i] = x0 * sin[i] + x1 * cos[i];
    }
}
