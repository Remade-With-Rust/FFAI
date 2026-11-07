//! AVX2 int8 GEMV for the vocabulary projection.
//!
//! The projection is ~27 % of decode and pure bandwidth: `(1,d) @ (d,vocab)`
//! streams the whole embedding matrix per generated token with no reuse. The
//! f16 path moves 39.8 MB/token at ~24 GB/s, already close to what this
//! machine's memory system gives a read-only stream.
//!
//! candle's `Q8_0` halves the bytes and measured **no faster** (20/41,
//! z = -0.2) — it spends on dequantization what it saves on traffic, turning
//! a memory-bound op into a compute-bound one. That is a verdict on that
//! kernel, not on int8: AVX2's `_mm256_maddubs_epi16` + `_mm256_madd_epi16`
//! do **32 multiply-accumulates per two instructions against 8 for an f32
//! FMA**, so a direct int8 GEMV wins on compute *and* bytes.
//!
//! Measured against the shipped f16+pad arm at the real shape:
//!
//! | | ms | GB/s |
//! |---|---:|---:|
//! | AVX2 int8 (20.3 MB) | 0.379 | 53.6 |
//! | f16 + pad (39.8 MB) | 1.645 | 24.2 |
//!
//! **4.34x, 41/41 paired rounds, z = +6.4.**
//!
//! **Saturation is the design constraint.** `maddubs` folds two u8xi8 products
//! into a *saturating* i16; at full range 255x127x2 = 64770 overflows.
//! Quantizing activations to 7 bits (offset into [0,127]) caps the pair sum at
//! 32258 < 32767. The lost bit is cheap: the activation is one d-vector whose
//! quantization error is negligible beside the weight matrix's.
//!
//! Weights use one scale per **32-weight block**. A per-row scale was tried
//! first and shipped 4.34x, but corpus WER moved 7.87 -> 7.93 %, leaving only
//! 0.027 pp of gate margin: one outlier sets the row scale and coarsens every
//! weight beside it. Block scales cost a per-block horizontal reduction and
//! ~6 % more bytes, and are why `Q8_0` uses them.

//! Cast policy (gate H-15): `cast_possible_truncation`, `cast_sign_loss` and
//! `cast_possible_wrap` are allowed in this module. Every value converted here
//! is a MODEL-INTERNAL dimension, index or accumulator - bounded by weights the
//! loader has already validated - not a number read from caller input. The lint
//! stays DENIED in the untrusted-surface modules (`mel`, `fbank`, `onnx`,
//! `normalize`, `lexicon`, `chunk`, `phonemize`, `phoneme_ids`), which is where
//! this audit's arithmetic defects were actually found.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    // `_mm256_loadu_si256` is the UNALIGNED load, so casting a byte pointer to
    // *const __m256i for it is correct; clippy cannot see the consumer.
    clippy::cast_ptr_alignment
)]

use crate::par::prelude::*;
use ffai_core::candle::{
    CpuStorage, CustomOp1, Device, Layout, Result as CandleResult, Shape, Tensor,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtepi32_ps, _mm_cvtsi128_si32, _mm_hadd_epi32,
    _mm_loadu_ps, _mm_mul_ps, _mm_shuffle_epi32, _mm_storeu_ps, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_setzero_si256,
};

/// Block-symmetric int8 vocabulary weights.
pub struct Int8Vocab {
    shared: std::sync::Arc<Shared>,
}

struct Shared {
    qw: Vec<i8>,
    /// One scale per 32-weight BLOCK, not per row. A single row scale is set
    /// by that row's largest magnitude, so one outlier coarsens all 384
    /// weights beside it; corpus WER moved 7.87 -> 7.93 % and left only
    /// 0.027 pp of gate margin. Per-block scales are why `Q8_0` uses them.
    bscale: Vec<f32>,
    /// Per-row activation-offset correction, `64 * sum_b scale_b * blocksum_b`.
    /// Independent of the activation, so it is folded once at load rather than
    /// carried as a per-block sum the kernel would have to read.
    corr: Vec<f32>,
    vocab: usize,
    d: usize,
    nblocks: usize,
    /// f32 weights, retained ONLY under `FFAI_VOCAB_AUDIT` so each token can be
    /// scored against an exact oracle. Costs ~80 MB; never allocated otherwise.
    #[allow(dead_code)] // exact-oracle tensor: ~80 MB, allocated only when scoring against it
    oracle: Option<Tensor>,
}

/// Weights per scale. Overridable with `FFAI_VOCAB_BLK` so the block-vs-row
/// tradeoff can be measured rather than argued: setting it to `d` reproduces a
/// single per-row scale exactly.
#[inline]
fn blk() -> usize {
    super::knobs::VOCAB_BLK.get_usize()
}

impl Int8Vocab {
    /// Build from the embedding matrix, which is already (vocab, d) row-major
    /// — so unlike the f16 path this needs no transpose at all.
    ///
    /// Returns `None` when the shape or machine is not one this serves.
    pub fn new(embeddings: &Tensor) -> CandleResult<Option<Self>> {
        if !have_avx2() || !matches!(embeddings.device(), Device::Cpu) {
            return Ok(None);
        }
        let (vocab, d) = match embeddings.dims2() {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        // The kernel steps 32 lanes at a time; every Whisper n_state is a
        // multiple of 32, but do not assume it.
        //
        // All three conditions are MEMORY-SAFETY invariants for `dot_i8_blocked`,
        // not tuning preferences, so they are checked HERE - at the point where
        // the kernel becomes reachable - rather than trusted from the knob's
        // env-var validator:
        //
        //   blk != 0        `d % blk` is a divide-by-zero panic otherwise, and
        //                   `Num::get_usize` maps any negative override to 0.
        //   blk % 32 == 0   the inner loop tests `o < blk` but READS 32 BYTES per
        //                   step, so a blk of 48 reads 16 bytes past the block -
        //                   and past the whole allocation on the final block.
        //   d % blk == 0    blocks must tile the row exactly.
        //
        // `FFAI_VOCAB_BLK` is validated on the env path, but `Num::set` stores
        // without validating, so that guarantee does not reach here. Falling back
        // to `None` picks the safe non-SIMD path instead of trusting the caller.
        let blk = blk();
        if blk == 0 || !blk.is_multiple_of(32) || d % blk != 0 {
            return Ok(None);
        }
        let w: Vec<f32> = embeddings
            .to_dtype(ffai_core::candle::DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        let nblocks = d / blk;
        let mut q = vec![0i8; vocab * d];
        let mut bscale = vec![0f32; vocab * nblocks];
        let mut corr = vec![0f32; vocab];
        for v in 0..vocab {
            let row = &w[v * d..(v + 1) * d];
            let mut c = 0f32;
            for b in 0..nblocks {
                let bw = &row[b * blk..(b + 1) * blk];
                let amax = bw.iter().fold(0f32, |m, &x| m.max(x.abs()));
                let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                bscale[v * nblocks + b] = sc;
                let mut bsum = 0i32;
                let qblk = &mut q[v * d + b * blk..][..blk];
                for (slot, &bwk) in qblk.iter_mut().zip(bw) {
                    let qi = (bwk / sc).round().clamp(-127.0, 127.0) as i8;
                    *slot = qi;
                    bsum += i32::from(qi);
                }
                c += sc * bsum as f32;
            }
            corr[v] = 64.0 * c;
        }
        let oracle = if audit_enabled() {
            Some(embeddings.to_dtype(ffai_core::candle::DType::F32)?)
        } else {
            None
        };
        let _ = &oracle;
        Ok(Some(Self {
            shared: std::sync::Arc::new(Shared {
                qw: q,
                bscale,
                corr,
                vocab,
                d,
                nblocks,
                oracle,
            }),
        }))
    }

    /// `x` is (1, d) f32 → logits (1, vocab) f32.
    ///
    /// Delivered as `CustomOp1`: the first version copied the activation out
    /// with `to_vec1()` and rebuilt a 207 KB logit vector with
    /// `Tensor::from_vec` on every generated token.
    ///
    /// **Measured 1.013x, 13/21, z = +1.1 — INSIDE THE NOISE.** Kept anyway,
    /// and the speed claim is NOT made: it strictly removes a per-token
    /// 207 KB allocate-and-copy, and it matches how the other two kernels in
    /// this crate are delivered, so a third one on `to_vec1` would just be a
    /// trap for the next reader.
    ///
    /// Why so much smaller than the decoder GEMV's 1.21x swing from the same
    /// change: this op READS ~20 MB per call, so the 207 KB copy is ~1 % of
    /// its traffic. On the decoder projections the copy was a large fraction
    /// of a tiny operation. **Marshalling cost is only decisive when the
    /// operation it wraps is small** — predicted before measuring, and the
    /// measurement agreed.
    ///
    /// Rayon STAYS here, unlike the decoder GEMV where it was removed. The
    /// distinction is per-call work: this is 19.9 M FMAs ≈ 622 us, which
    /// dwarfs a 10-20 us fork-join, where the decoder's projections were only
    /// 4.6-18.4 us and the fork-join cost more than the arithmetic it split.
    /// Same insight, opposite conclusion, because the shape is different.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        x.apply_op1(Int8VocabOp {
            q: self.shared.clone(),
        })
    }

    /// The projection straight into `out` (resized to the vocabulary), with
    /// no tensor built around the result. The decode loop reads the logits as
    /// a `Vec<f32>` anyway; routing through a `Tensor` only to `to_vec1()` it
    /// back out was a 4·vocab-byte allocation and copy per token (vein 2).
    /// `x` is the one activation row, `d` floats. Same kernel, same order.
    pub fn forward_into(&self, x: &[f32], out: &mut Vec<f32>) -> CandleResult<()> {
        if x.len() != self.shared.d {
            ffai_core::candle::bail!(
                "int8-vocab: activation has {} elements, expected {}",
                x.len(),
                self.shared.d
            );
        }
        out.resize(self.shared.vocab, 0.0);
        project(&self.shared, x, out);
        Ok(())
    }
}

/// The GEMV body shared by the op and `forward_into`: quantize the activation
/// row, then one block-scaled int8 dot per vocabulary row.
fn project(q: &Shared, x: &[f32], out: &mut [f32]) {
    let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let sx = if amax > 0.0 { amax / 63.0 } else { 1.0 };
    let mut xu = vec![0u8; q.d];
    // Zipped rather than indexed: same elements in the same order, and no
    // `panic_bounds_check` in a loop that runs `d` times per decode step.
    //
    // Vein 8: `f32::round` is a libm CALL (ties-away has no x86 instruction),
    // the one thing in this unit-stride loop that would not widen.
    // `round_ties_away_i32` reproduces it exactly — same integer for every
    // input, including the `.5` ties and the `0.499_999_97 + 0.5` carry —
    // in pure arithmetic, so the loop vectorises.
    for (slot, &v) in xu.iter_mut().zip(x) {
        let qi = ffai_core::fastmath::round_ties_away_i32((v / sx).clamp(-64.0, 63.0));
        *slot = (qi + 64) as u8;
    }

    let (d, nb) = (q.d, q.nblocks);
    let blk = d / nb;
    let legacy = super::knobs::VEINS_OFF.get();
    out.par_chunks_mut(512).enumerate().for_each(|(ci, o)| {
        let v0 = ci * 512;
        let mut i = 0;
        // Vein 8: two vocabulary rows per pass share every load of the
        // activation (`xu`); the arithmetic per row is untouched, so this is
        // bit-identical and simply issues fewer instructions.
        while i + 2 <= o.len() && !legacy {
            let (va, vb) = (v0 + i, v0 + i + 1);
            // SAFETY: avx2 - `Int8Vocab::new` only constructs this when `have_avx2()`
            // passes, and both entry points go through it. The slices handed to the
            // kernel are derived from `nblocks`/`blk`, which partition the quantized
            // weight buffer exactly.
            let (acc_a, acc_b) = unsafe {
                dot_i8_blocked_x2(
                    &q.qw[va * d..(va + 1) * d],
                    &q.qw[vb * d..(vb + 1) * d],
                    &q.bscale[va * nb..(va + 1) * nb],
                    &q.bscale[vb * nb..(vb + 1) * nb],
                    &xu,
                    nb,
                    blk,
                )
            };
            o[i] = (acc_a - q.corr[va]) * sx;
            o[i + 1] = (acc_b - q.corr[vb]) * sx;
            i += 2;
        }
        for (i, slot) in o.iter_mut().enumerate().skip(i) {
            let v = v0 + i;
            // SAFETY: as above.
            let acc = unsafe {
                if legacy {
                    dot_i8_blocked_legacy(
                        &q.qw[v * d..(v + 1) * d],
                        &q.bscale[v * nb..(v + 1) * nb],
                        &xu,
                        nb,
                        blk,
                    )
                } else {
                    dot_i8_blocked(
                        &q.qw[v * d..(v + 1) * d],
                        &q.bscale[v * nb..(v + 1) * nb],
                        &xu,
                        nb,
                        blk,
                    )
                }
            };
            *slot = (acc - q.corr[v]) * sx;
        }
    });
}

struct Int8VocabOp {
    q: std::sync::Arc<Shared>,
}

impl CustomOp1 for Int8VocabOp {
    fn name(&self) -> &'static str {
        "int8-vocab"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        let xf = match s {
            CpuStorage::F32(v) => v,
            _ => ffai_core::candle::bail!("int8-vocab: f32 activations only"),
        };
        if !l.is_contiguous() {
            ffai_core::candle::bail!("int8-vocab: contiguous only")
        }
        let q = &*self.q;
        let x = &xf[l.start_offset()..l.start_offset() + q.d];
        let mut out = vec![0f32; q.vocab];
        project(q, x, &mut out);
        Ok((CpuStorage::F32(out), Shape::from_dims(&[1, q.vocab])))
    }
}

#[cfg(target_arch = "x86_64")]
fn have_avx2() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| is_x86_feature_detected!("avx2"))
}

#[cfg(not(target_arch = "x86_64"))]
fn have_avx2() -> bool {
    false
}

/// Block-scaled int8 dot: each 32-lane block reduces to an i32, is scaled by
/// its own f32, and accumulates in float.
///
/// The per-block horizontal reduction is the price of the finer scale — the
/// row-scaled version avoided it by reducing once per row. This is the
/// original form, kept as the oracle and as the `FFAI_VEINS=off` arm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: caller must ensure avx2, and that `w`/`bscale`/`xu` are long enough
// for `nblocks*blk`. Guarded by `have_avx2()` at the single call site.
unsafe fn dot_i8_blocked_legacy(
    w: &[i8],
    bscale: &[f32],
    xu: &[u8],
    nblocks: usize,
    blk: usize,
) -> f32 {
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let wp = w.as_ptr();
        let xp = xu.as_ptr();
        let mut acc = 0f32;
        for b in 0..nblocks {
            // One block may span several 32-lane steps when FFAI_VOCAB_BLK > 32.
            let mut p = _mm256_setzero_si256();
            let base = b * blk;
            let mut o = 0;
            while o < blk {
                let x0 = _mm256_loadu_si256(xp.add(base + o).cast::<__m256i>());
                let w0 = _mm256_loadu_si256(wp.add(base + o).cast::<__m256i>());
                p = _mm256_add_epi32(p, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, w0), ones));
                o += 32;
            }
            let s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_01_00_01));
            acc += _mm_cvtsi128_si32(s) as f32 * *bscale.get_unchecked(b);
        }
        acc
    }
}

/// The block sums of one row, as 8-lane i32 partials: `blk` lanes of
/// `maddubs`+`madd` per block, exact integers throughout.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: caller must ensure avx2 and that `wp`/`xp` address `blk` bytes from
// `base`. Only called from the two kernels below, which hold that contract.
unsafe fn block_partials(wp: *const i8, xp: *const u8, base: usize, blk: usize) -> __m256i {
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut p = _mm256_setzero_si256();
        let mut o = 0;
        while o < blk {
            let x0 = _mm256_loadu_si256(xp.add(base + o).cast::<__m256i>());
            let w0 = _mm256_loadu_si256(wp.add(base + o).cast::<__m256i>());
            p = _mm256_add_epi32(p, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, w0), ones));
            o += 32;
        }
        p
    }
}

/// Four blocks' 8-lane i32 partials -> their four exact block sums, in one
/// 128-bit lane each, with 3 `hadd`s instead of 4 x (2 shuffles + 2 adds).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: avx2 (for `_mm256_extracti128_si256`); pure register arithmetic.
unsafe fn sum4(p0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i) -> __m128i {
    let q0 = _mm_add_epi32(_mm256_castsi256_si128(p0), _mm256_extracti128_si256(p0, 1));
    let q1 = _mm_add_epi32(_mm256_castsi256_si128(p1), _mm256_extracti128_si256(p1, 1));
    let q2 = _mm_add_epi32(_mm256_castsi256_si128(p2), _mm256_extracti128_si256(p2, 1));
    let q3 = _mm_add_epi32(_mm256_castsi256_si128(p3), _mm256_extracti128_si256(p3, 1));
    // [q0a+q0b, q0c+q0d, q1a+q1b, q1c+q1d] and the same for q2/q3, then
    // one more pairwise add leaves [sum q0, sum q1, sum q2, sum q3].
    _mm_hadd_epi32(_mm_hadd_epi32(q0, q1), _mm_hadd_epi32(q2, q3))
}

/// Fold four block sums into the running f32 accumulator EXACTLY as the
/// legacy kernel did: `acc += (sum_b as f32) * bscale[b]` in block order.
/// The four products are formed in one packed multiply (each lane is the
/// same IEEE product the scalar form computes); only the four adds stay
/// sequential, because they must.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: avx2/sse; `bscale` must hold 4 readable floats.
unsafe fn fold4(acc: f32, sums: __m128i, bscale: *const f32) -> f32 {
    unsafe {
        let prod = _mm_mul_ps(_mm_cvtepi32_ps(sums), _mm_loadu_ps(bscale));
        let mut out = [0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), prod);
        (((acc + out[0]) + out[1]) + out[2]) + out[3]
    }
}

/// Block-scaled int8 dot with the per-block horizontal reductions GROUPED
/// four at a time (vein 8). The i32 block sums are exact whichever way the
/// lanes are added, and the f32 products and their summation order match
/// `dot_i8_blocked_legacy` term for term, so the result is bit-identical;
/// what changes is ~11 reduction instructions per four blocks instead of ~40.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: caller must ensure avx2, and that `w`/`bscale`/`xu` are long enough
// for `nblocks*blk`. Guarded by `have_avx2()` at the call sites.
unsafe fn dot_i8_blocked(w: &[i8], bscale: &[f32], xu: &[u8], nblocks: usize, blk: usize) -> f32 {
    unsafe {
        let wp = w.as_ptr();
        let xp = xu.as_ptr();
        let mut acc = 0f32;
        let mut b = 0;
        while b + 4 <= nblocks {
            let p0 = block_partials(wp, xp, b * blk, blk);
            let p1 = block_partials(wp, xp, (b + 1) * blk, blk);
            let p2 = block_partials(wp, xp, (b + 2) * blk, blk);
            let p3 = block_partials(wp, xp, (b + 3) * blk, blk);
            acc = fold4(acc, sum4(p0, p1, p2, p3), bscale.as_ptr().add(b));
            b += 4;
        }
        while b < nblocks {
            let p = block_partials(wp, xp, b * blk, blk);
            let s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_01_00_01));
            acc += _mm_cvtsi128_si32(s) as f32 * *bscale.get_unchecked(b);
            b += 1;
        }
        acc
    }
}

/// Two rows at once (vein 8): one load of each activation block feeds both
/// rows' `maddubs`. Per-row arithmetic and order are exactly
/// [`dot_i8_blocked`]'s.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
// SAFETY: as `dot_i8_blocked`, for both rows.
unsafe fn dot_i8_blocked_x2(
    wa: &[i8],
    wb: &[i8],
    sa: &[f32],
    sb: &[f32],
    xu: &[u8],
    nblocks: usize,
    blk: usize,
) -> (f32, f32) {
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let (wpa, wpb) = (wa.as_ptr(), wb.as_ptr());
        let xp = xu.as_ptr();
        let (mut acc_a, mut acc_b) = (0f32, 0f32);
        let mut b = 0;
        while b + 4 <= nblocks {
            let mut pa = [_mm256_setzero_si256(); 4];
            let mut pb = [_mm256_setzero_si256(); 4];
            for g in 0..4 {
                let base = (b + g) * blk;
                let mut o = 0;
                while o < blk {
                    let x0 = _mm256_loadu_si256(xp.add(base + o).cast::<__m256i>());
                    let w0 = _mm256_loadu_si256(wpa.add(base + o).cast::<__m256i>());
                    let w1 = _mm256_loadu_si256(wpb.add(base + o).cast::<__m256i>());
                    pa[g] = _mm256_add_epi32(
                        pa[g],
                        _mm256_madd_epi16(_mm256_maddubs_epi16(x0, w0), ones),
                    );
                    pb[g] = _mm256_add_epi32(
                        pb[g],
                        _mm256_madd_epi16(_mm256_maddubs_epi16(x0, w1), ones),
                    );
                    o += 32;
                }
            }
            acc_a = fold4(acc_a, sum4(pa[0], pa[1], pa[2], pa[3]), sa.as_ptr().add(b));
            acc_b = fold4(acc_b, sum4(pb[0], pb[1], pb[2], pb[3]), sb.as_ptr().add(b));
            b += 4;
        }
        while b < nblocks {
            let base = b * blk;
            let mut pa = _mm256_setzero_si256();
            let mut pb = _mm256_setzero_si256();
            let mut o = 0;
            while o < blk {
                let x0 = _mm256_loadu_si256(xp.add(base + o).cast::<__m256i>());
                let w0 = _mm256_loadu_si256(wpa.add(base + o).cast::<__m256i>());
                let w1 = _mm256_loadu_si256(wpb.add(base + o).cast::<__m256i>());
                pa = _mm256_add_epi32(pa, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, w0), ones));
                pb = _mm256_add_epi32(pb, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, w1), ones));
                o += 32;
            }
            for (p, acc, sc) in [(pa, &mut acc_a, sa), (pb, &mut acc_b, sb)] {
                let s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_01_00_01));
                *acc += _mm_cvtsi128_si32(s) as f32 * *sc.get_unchecked(b);
            }
            b += 1;
        }
        (acc_a, acc_b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
// SAFETY: non-x86 stub. It is `unsafe fn` only to match the signature of the
// x86 kernel it stands in for, and its body is `unreachable!` - the runtime
// feature check that guards every call site can never pass on this target,
// so there is no reachable code here to be unsound.
unsafe fn dot_i8_blocked(_w: &[i8], _b: &[f32], _xu: &[u8], _n: usize, _k: usize) -> f32 {
    unreachable!("guarded by have_avx2()")
}

#[cfg(not(target_arch = "x86_64"))]
// SAFETY: non-x86 stub, as above.
unsafe fn dot_i8_blocked_legacy(_w: &[i8], _b: &[f32], _xu: &[u8], _n: usize, _k: usize) -> f32 {
    unreachable!("guarded by have_avx2()")
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
// SAFETY: non-x86 stub, as above.
unsafe fn dot_i8_blocked_x2(
    _wa: &[i8],
    _wb: &[i8],
    _sa: &[f32],
    _sb: &[f32],
    _xu: &[u8],
    _n: usize,
    _k: usize,
) -> (f32, f32) {
    unreachable!("guarded by have_avx2()")
}

#[cfg(all(test, target_arch = "x86_64"))]
mod reduction_tests {
    use super::*;

    /// The grouped and two-row kernels must reproduce the legacy kernel to
    /// the bit on every block count, including the <4 remainder and
    /// multi-step blocks.
    #[test]
    fn grouped_and_paired_kernels_match_legacy_bitwise() {
        if !have_avx2() {
            return;
        }
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for &(nblocks, blk) in &[
            (12usize, 32usize),
            (13, 32),
            (5, 64),
            (3, 32),
            (1, 32),
            (40, 32),
        ] {
            let d = nblocks * blk;
            let wa: Vec<i8> = (0..d).map(|_| (next() % 255) as i8).collect();
            let wb: Vec<i8> = (0..d).map(|_| (next() % 255) as i8).collect();
            let xu: Vec<u8> = (0..d).map(|_| (next() % 128) as u8).collect();
            let sa: Vec<f32> = (0..nblocks)
                .map(|_| (next() % 1000) as f32 * 1e-3 + 1e-3)
                .collect();
            let sb: Vec<f32> = (0..nblocks)
                .map(|_| (next() % 1000) as f32 * 1e-3 + 1e-3)
                .collect();
            // SAFETY: avx2 checked above; every slice is exactly nblocks*blk.
            let (la, lb, ga, gb, (xa, xb)) = unsafe {
                (
                    dot_i8_blocked_legacy(&wa, &sa, &xu, nblocks, blk),
                    dot_i8_blocked_legacy(&wb, &sb, &xu, nblocks, blk),
                    dot_i8_blocked(&wa, &sa, &xu, nblocks, blk),
                    dot_i8_blocked(&wb, &sb, &xu, nblocks, blk),
                    dot_i8_blocked_x2(&wa, &wb, &sa, &sb, &xu, nblocks, blk),
                )
            };
            assert_eq!(
                la.to_bits(),
                ga.to_bits(),
                "grouped row a, nblocks {nblocks} blk {blk}"
            );
            assert_eq!(
                lb.to_bits(),
                gb.to_bits(),
                "grouped row b, nblocks {nblocks} blk {blk}"
            );
            assert_eq!(
                la.to_bits(),
                xa.to_bits(),
                "paired row a, nblocks {nblocks} blk {blk}"
            );
            assert_eq!(
                lb.to_bits(),
                xb.to_bits(),
                "paired row b, nblocks {nblocks} blk {blk}"
            );
        }
    }

    /// The branchless quantiser rounds exactly like `f32::round` on the
    /// clamped range the kernel feeds it, ties included.
    #[test]
    fn quantiser_rounding_matches_round() {
        let mut v = -64.0f32;
        while v <= 63.0 {
            for dv in [0.0f32, 0.25, 0.5, 0.75, 0.499_999_97, 0.500_000_06] {
                let x = (v + dv).clamp(-64.0, 63.0);
                let legacy = x.round() as i32;
                let fast = ffai_core::fastmath::round_ties_away_i32(x);
                assert_eq!(legacy, fast, "x = {x}");
            }
            v += 1.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Argmax-disagreement audit
// ---------------------------------------------------------------------------
//
// WER is a lagging, noisy indicator of what quantization does. The row-scaled
// version of this kernel moved corpus WER by 0.06 pp — inside the run-to-run
// noise band — while being materially worse at the only thing this op decides:
// which token wins. This counts that directly.
//
// Enable with `FFAI_VOCAB_AUDIT=1`. Resolved once, not per call: an earlier
// optimization in this crate regressed itself 3x by reading an env var and
// taking a mutex inside the per-token path.

use std::sync::atomic::{AtomicU64, Ordering};

static AUDIT_TOTAL: AtomicU64 = AtomicU64::new(0);
static AUDIT_FLIPS: AtomicU64 = AtomicU64::new(0);
/// Sum of |top1 - top2| margins, in milli-units, over flipped tokens — a flip
/// where the top two logits were far apart is a much worse sign than one where
/// they were tied.
static AUDIT_FLIP_MARGIN: AtomicU64 = AtomicU64::new(0);

pub fn audit_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FFAI_VOCAB_AUDIT").is_ok_and(|v| v != "0"))
}

/// Record one token's verdict: did int8 pick the same row as the f32 oracle?
pub fn audit_record(ours: &[f32], oracle: &[f32]) {
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    let (a, b) = (top(ours), top(oracle));
    AUDIT_TOTAL.fetch_add(1, Ordering::Relaxed);
    if a != b {
        AUDIT_FLIPS.fetch_add(1, Ordering::Relaxed);
        // How decisive was the oracle? A flip against a clear winner is worse
        // than one against a near-tie.
        let mut s: Vec<f32> = oracle.to_vec();
        s.sort_by(|x, y| y.total_cmp(x));
        let margin = (s[0] - s[1]).abs();
        AUDIT_FLIP_MARGIN.fetch_add((margin * 1000.0) as u64, Ordering::Relaxed);
    }
}

/// Human-readable audit result, or `None` if the audit never ran.
pub fn audit_report() -> Option<String> {
    let total = AUDIT_TOTAL.load(Ordering::Relaxed);
    if total == 0 {
        return None;
    }
    let flips = AUDIT_FLIPS.load(Ordering::Relaxed);
    let margin = AUDIT_FLIP_MARGIN.load(Ordering::Relaxed) as f64 / 1000.0;
    Some(format!(
        "VOCAB ARGMAX AUDIT
  tokens {total}
  argmax flips vs f32 oracle {flips} ({:.3}%)
           mean oracle top1-top2 margin on flips {:.4}",
        flips as f64 / total as f64 * 100.0,
        if flips > 0 {
            margin / flips as f64
        } else {
            0.0
        }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What matters is the ARGMAX — it is what selects a token.
    ///
    /// This asserted plain argmax equality against unseeded `randn` and was
    /// FLAKY: on random logits the top two can be arbitrarily close, and int8
    /// legitimately flips them. That is not a defect, so the test must not
    /// claim it is one. What quantization must actually guarantee is that the
    /// row it picks is within quantization error of the true best — a wrong
    /// pick among near-ties is harmless, a wrong pick against a clear winner
    /// is not.
    #[test]
    fn selects_a_row_within_quantization_error() -> CandleResult<()> {
        let dev = Device::Cpu;
        let (vocab, d) = (4096usize, 384usize);
        let w = Tensor::randn(0f32, 1., (vocab, d), &dev)?;
        let x = Tensor::randn(0f32, 1., (1, d), &dev)?;

        let Some(q) = Int8Vocab::new(&w)? else {
            return Ok(());
        };
        let ours: Vec<f32> = q.forward(&x)?.flatten_all()?.to_vec1()?;
        let want: Vec<f32> = x.matmul(&w.t()?)?.flatten_all()?.to_vec1()?;

        let am = |v: &Vec<f32>| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        let (picked, best) = (am(&ours), am(&want));
        let scale = want.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let shortfall = (want[best] - want[picked]) / scale;
        assert!(
            shortfall < 0.02,
            "int8 picked row {picked}, which is {shortfall:.4} of full scale              below the true best (row {best}) — beyond quantization error"
        );
        Ok(())
    }

    #[test]
    fn declines_shapes_it_cannot_serve() -> CandleResult<()> {
        let dev = Device::Cpu;
        // d not a multiple of the 32-lane step.
        let w = Tensor::randn(0f32, 1., (128usize, 20usize), &dev)?;
        assert!(Int8Vocab::new(&w)?.is_none());
        Ok(())
    }
}
