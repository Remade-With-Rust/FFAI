//! Fused ("flash") attention for the encoder shape, AVX2 + FMA.
//!
//! Encoder attention was ~67 % of the encoder and the single largest remaining
//! item in the speed gap. The three-op path (`q@kᵀ` → softmax → `@v`)
//! materializes a 54 MB score matrix per layer and runs at ~88 GFLOP/s; this
//! kernel never materializes it and measures **331 GFLOP/s, 3.75× faster,
//! 21/21 paired rounds (z = +4.6)**, output matching to 6.9e-7.
//!
//! Two earlier attempts are worth recording because the difference between
//! them *is* the optimization:
//!
//! | attempt | GFLOP/s |
//! |---|---:|
//! | naive dot products | 46 |
//! | contraction-outermost AXPY | 73 |
//! | **AVX2 register tiling + vectorized `exp`** | **331** |
//!
//! The first lost because a dot product ends in a horizontal reduction. The
//! second still moved the score tile through memory once per contraction step.
//! This one keeps a 4×16 output tile in registers across the whole 64-step
//! contraction, and replaces the scalar `exp` — 13.5 M of them per 30 s
//! window, enough to rival the rest of the kernel — with a vectorized one.
//!
//! Applies only where all of these hold, falling back to the three-op path
//! otherwise: CPU, f32, no mask, `head_dim` 64 (every Whisper size), a query
//! sequence long enough to amortize the tiling, and AVX2+FMA present.

// `_mm_loadu_si128` is the UNALIGNED load - the `u` is the whole point - so
// casting a *const u16 to *const __m128i for it is correct, not a latent
// misalignment. clippy cannot see the consumer.
#![allow(clippy::cast_ptr_alignment)]

use ffai_core::candle::{
    CpuStorage, CustomOp3, DType, Device, Layout, Result as CandleResult, Shape, Tensor,
};
use crate::par::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _mm_add_ps, _mm_add_ss,
    _mm_cvtss_f32, _mm_loadu_si128, _mm_max_ps, _mm_max_ss, _mm_movehl_ps, _mm_shuffle_ps,
    _mm256_add_epi32, _mm256_add_ps, _mm256_castps256_ps128, _mm256_castsi256_ps, _mm256_cvtph_ps,
    _mm256_cvtps_epi32, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps,
    _mm256_mul_ps, _mm256_round_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_slli_epi32, _mm256_storeu_ps, _mm256_sub_ps,
};

const BQ: usize = 64;
const BK: usize = 256;
const HD: usize = 64;

/// Below this query length the three-op path wins — there is not enough work
/// per tile to amortize packing, and the decoder's single-row queries have no
/// reuse to exploit at all.
const MIN_SEQ: usize = 256;

/// Minimum cached keys before the decode-shape kernel takes over.
///
/// Decoder self-attention moves only ~138 KB per call but costs 274 us —
/// **2 % of memory bandwidth, so ~98 % of it is framework dispatch**. The
/// traffic argument that justifies the cross-attention threshold does not
/// apply here; the case for fusing is collapsing op count, which pays at far
/// smaller key counts. Tunable so the crossover is measured, not guessed.
#[inline]
fn min_decode_keys() -> usize {
    super::knobs::DECODE_MIN_KEYS.get_usize()
}

/// `FFAI_PAR_HEADS=on` fans the decode-shape heads across cores.
#[inline]
fn par_heads() -> bool {
    super::knobs::PAR_HEADS.get()
}

#[cfg(target_arch = "x86_64")]
fn have_f16c() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| is_x86_feature_detected!("f16c"))
}

#[cfg(not(target_arch = "x86_64"))]
fn have_f16c() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn have_avx2() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

#[cfg(not(target_arch = "x86_64"))]
fn have_avx2() -> bool {
    false
}

/// Whether the fused kernel can serve this K/V pair at the decode shape.
///
/// Callers use this to decide the cache dtype *before* building it: keeping
/// the cache f32 is only right if the kernel will actually read it.
#[must_use]
pub fn serves(kt: &Tensor, v: &Tensor) -> bool {
    if !have_avx2() || !matches!(kt.device(), Device::Cpu) {
        return false;
    }
    match (kt.dims4(), v.dims4()) {
        (Ok((1, h1, HD, keys)), Ok((1, h2, k2, HD))) => {
            // The floor must be the one the CONSUMER will apply, or this
            // predicate promises a kernel that then declines.
            //
            // It used to be `keys >= MIN_SEQ` alone while `attend_prepared`
            // gates the decode shape on `min_decode_keys()`. Two predicates
            // for one decision: callers use `serves` to choose the cache dtype
            // (f16) AND to leave q in f32, so raising `FFAI_DECODE_MIN_KEYS`
            // above the key count left an f16 cache being read by the f32
            // three-op fallback — `dtype mismatch in matmul, lhs: F32,
            // rhs: F16`. At the shipped default (8) both floors pass at 1500
            // keys, so this was invisible in production and fatal to the
            // escape hatch: the A/B that produced the fused self-attention
            // kernel's 24/25, z = +4.6 could no longer be re-run.
            //
            // A toggle that looks available but panics is worse than none.
            h1 == h2 && keys == k2 && keys >= MIN_SEQ.max(min_decode_keys())
        }
        _ => false,
    }
}

/// Encoder attention returning (1, seq, heads*HD) — already merged.
///
/// Saves the caller a `transpose(1,2) + flatten_from(2)`, which is a strided
/// copy of 2.3 MB per layer measured at 13 ms in context across the encoder.
pub fn attend_merged(q: &Tensor, kt: &Tensor, v: &Tensor) -> CandleResult<Option<Tensor>> {
    let (b, heads, qlen, hd) = match q.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    if b != 1 || hd != HD || !have_avx2() || !matches!(q.device(), Device::Cpu) {
        return Ok(None);
    }
    if q.dtype() != DType::F32 || kt.dtype() != DType::F32 || v.dtype() != DType::F32 {
        return Ok(None);
    }
    let keys = match kt.dims4() {
        Ok((1, h, d, k)) if h == heads && d == HD => k,
        _ => return Ok(None),
    };
    if v.dims4()? != (1, heads, keys, HD) || qlen != keys || keys < MIN_SEQ {
        return Ok(None);
    }
    Ok(Some(q.apply_op3(
        kt,
        v,
        MergedAttnOp { heads, seq: keys },
    )?))
}

struct MergedAttnOp {
    heads: usize,
    seq: usize,
}

impl CustomOp3 for MergedAttnOp {
    fn name(&self) -> &'static str {
        "flash-attn-merged"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let (q, kt, v) = match (s1, s2, s3) {
            (CpuStorage::F32(a), CpuStorage::F32(b), CpuStorage::F32(c)) => (a, b, c),
            _ => ffai_core::candle::bail!("flash-attn-merged: f32 only"),
        };
        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
            ffai_core::candle::bail!("flash-attn-merged: contiguous only")
        }
        let (heads, seq) = (self.heads, self.seq);
        let n = heads * seq * HD;
        let q = &q[l1.start_offset()..l1.start_offset() + n];
        let kt = &kt[l2.start_offset()..l2.start_offset() + n];
        let v = &v[l3.start_offset()..l3.start_offset() + n];

        let width = heads * HD;
        let mut out = vec![0f32; seq * width];
        // Heads write disjoint COLUMN BANDS of the same rows, so the slices
        // cannot be handed out by `par_chunks_mut`. Split by head over raw
        // pointers instead; each head touches only `[.. + h*HD .. + h*HD+HD]`
        // of every row, so no two threads ever write the same element.
        let base = out.as_mut_ptr() as usize;
        (0..heads).into_par_iter().for_each(|h| {
            let per = seq * HD;
            let r = h * per..(h + 1) * per;
            // SAFETY: disjoint column bands, as argued above.
            let slice = unsafe {
                std::slice::from_raw_parts_mut((base as *mut f32).add(h * HD), seq * width - h * HD)
            };
            flash_head_strided(&q[r.clone()], &kt[r.clone()], &v[r], slice, seq, width);
        });
        Ok((CpuStorage::F32(out), Shape::from_dims(&[1, seq, width])))
    }
}

/// Fused attention, or `Ok(None)` when this shape is not one the kernel serves.
///
/// `q` is (1, heads, seq, 64); `kt` is (1, heads, 64, seq) — already
/// transposed, which is what the caller's prepared path produces; `v` is
/// (1, heads, seq, 64). Whisper folds 1/sqrt(d) into q and k before this
/// point, so no scaling happens here.
pub fn attend(q: &Tensor, kt: &Tensor, v: &Tensor, masked: bool) -> CandleResult<Option<Tensor>> {
    if masked || !have_avx2() || !matches!(q.device(), Device::Cpu) || q.dtype() != DType::F32 {
        return Ok(None);
    }
    // K/V may be f16 (decode shape only); q always stays f32 so the
    // accumulation keeps full precision.
    let kv_half = kt.dtype() == DType::F16;
    if kv_half && (v.dtype() != DType::F16 || !have_f16c()) {
        return Ok(None);
    }
    if !kv_half && (kt.dtype() != DType::F32 || v.dtype() != DType::F32) {
        return Ok(None);
    }
    let (b, heads, qlen, hd) = match q.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    if b != 1 || hd != HD {
        return Ok(None);
    }
    let keys = match kt.dims4() {
        Ok((1, h, d, k)) if h == heads && d == HD => k,
        _ => return Ok(None),
    };
    if v.dims4()? != (1, heads, keys, HD) {
        return Ok(None);
    }
    let floor = if qlen == 1 {
        min_decode_keys()
    } else {
        MIN_SEQ
    };
    if keys < floor {
        return Ok(None);
    }
    // Two shapes, two kernels. The encoder tiles 64 query rows and holds a
    // 4x16 register block; the decoder has ONE query row, so there is no reuse
    // to tile and the right kernel is a streaming one instead.
    match qlen {
        1 => {}
        n if n == keys => {}
        _ => return Ok(None),
    }

    let out = q.apply_op3(kt, v, FlashAttnOp { heads, qlen, keys })?;
    Ok(Some(out))
}

struct FlashAttnOp {
    heads: usize,
    qlen: usize,
    keys: usize,
}

impl CustomOp3 for FlashAttnOp {
    fn name(&self) -> &'static str {
        "flash-attn"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        // f16 K/V is served at the decode shape only: the cache is read in
        // full every token, so halving it pays there. The encoder shape reads
        // its K/V once per pass and would gain nothing for the precision.
        if let (CpuStorage::F32(q), CpuStorage::F16(kt), CpuStorage::F16(v)) = (s1, s2, s3)
            && self.qlen == 1
            && l1.is_contiguous()
            && l2.is_contiguous()
            && l3.is_contiguous()
        {
            let (heads, keys) = (self.heads, self.keys);
            let q = &q[l1.start_offset()..l1.start_offset() + heads * HD];
            // SAFETY: reinterpreting candle's F16 storage as the u16 bit pattern the kernel
            // consumes. `half::f16` is repr(transparent) over u16, so the cast is
            // layout-compatible and not a reinterpretation of unrelated types. The length
            // heads*HD*keys is exactly this tensor's element count, and `l2.is_contiguous()`
            // was checked above, so start_offset + that length stays inside the allocation.
            let ktb: &[u16] = unsafe {
                std::slice::from_raw_parts(
                    kt.as_ptr().add(l2.start_offset()).cast::<u16>(),
                    heads * HD * keys,
                )
            };
            // SAFETY: as for `ktb` above - repr(transparent) f16 -> u16 over a contiguous
            // tensor whose length was checked, offset by its own start_offset.
            let vb: &[u16] = unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr().add(l3.start_offset()).cast::<u16>(),
                    heads * keys * HD,
                )
            };
            let mut out = vec![0f32; heads * HD];
            if par_heads() {
                let mut scratch = vec![0f32; heads * keys];
                out.par_chunks_mut(HD)
                    .zip(scratch.par_chunks_mut(keys))
                    .enumerate()
                    // SAFETY: two obligations. (1) avx2+fma+f16c: `have_avx2()` and `have_f16c()`
                    // were both checked at the top of this function, which is the only path here.
                    // (2) Aliasing: `par_chunks_mut(HD)` hands each task a distinct output chunk and
                    // `scratch.par_chunks_mut(keys)` a distinct scratch row, so no two tasks write
                    // the same element.
                    .for_each(|(h, (o, sc))| unsafe {
                        xattn_head_f16(
                            &q[h * HD..(h + 1) * HD],
                            &ktb[h * HD * keys..(h + 1) * HD * keys],
                            &vb[h * keys * HD..(h + 1) * keys * HD],
                            o,
                            keys,
                            sc,
                        );
                    });
            } else {
                let mut scratch = vec![0f32; keys];
                for h in 0..heads {
                    // SAFETY: same feature guard as the parallel arm above. This branch is serial,
                    // so the `&mut out[..]` slices are exclusive by the borrow checker rather than
                    // by argument.
                    unsafe {
                        xattn_head_f16(
                            &q[h * HD..(h + 1) * HD],
                            &ktb[h * HD * keys..(h + 1) * keys * HD / HD * HD],
                            &vb[h * keys * HD..(h + 1) * keys * HD],
                            &mut out[h * HD..(h + 1) * HD],
                            keys,
                            &mut scratch,
                        );
                    }
                }
            }
            return Ok((CpuStorage::F32(out), Shape::from_dims(&[1, heads, 1, HD])));
        }
        let (q, kt, v) = match (s1, s2, s3) {
            (CpuStorage::F32(a), CpuStorage::F32(b), CpuStorage::F32(c)) => (a, b, c),
            _ => ffai_core::candle::bail!("flash-attn: f32 only"),
        };
        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
            ffai_core::candle::bail!("flash-attn: contiguous operands only")
        }
        let (heads, qlen, keys) = (self.heads, self.qlen, self.keys);
        let q = &q[l1.start_offset()..l1.start_offset() + heads * qlen * HD];
        let kt = &kt[l2.start_offset()..l2.start_offset() + heads * HD * keys];
        let v = &v[l3.start_offset()..l3.start_offset() + heads * keys * HD];

        let mut out = vec![0f32; heads * qlen * HD];
        if qlen == 1 {
            // Decode step: one streaming pass per head, heads SERIAL.
            //
            // MEASURED AND REVERTED: rayon across the 6 heads. The stage got
            // faster — cross-attn 0.073 -> 0.064 s, 1.14x — and the pipeline
            // got SLOWER: 0.355 -> 0.364 s total, 7/21 paired rounds,
            // z = -1.5. Each head streams ~512 KB, so there is real work to
            // split, but the fork-join contends with the rest of the pipeline
            // and gives back more than the stage wins.
            //
            // This is the chain-level rule with a fresh data point: an
            // optimization is applied to an op and paid for by its
            // NEIGHBOURS. A stage-level win is not a result until the level
            // above confirms it.
            let mut scratch = vec![0f32; keys];
            for h in 0..heads {
                // SAFETY: `have_avx2()` was checked at the top of this function. Serial loop, so
                // each `&mut out[h*HD..]` borrow is exclusive; `scratch` is reused across heads
                // and fully overwritten by the callee before it is read.
                unsafe {
                    xattn_head(
                        &q[h * HD..(h + 1) * HD],
                        &kt[h * HD * keys..(h + 1) * HD * keys],
                        &v[h * keys * HD..(h + 1) * keys * HD],
                        &mut out[h * HD..(h + 1) * HD],
                        keys,
                        &mut scratch,
                    );
                }
            }
        } else {
            let per_head = keys * HD;
            out.par_chunks_mut(per_head).enumerate().for_each(|(h, o)| {
                let r = h * per_head..(h + 1) * per_head;
                flash_head(&q[r.clone()], &kt[r.clone()], &v[r], o, keys);
            });
        }

        Ok((
            CpuStorage::F32(out),
            Shape::from_dims(&[1, heads, qlen, HD]),
        ))
    }
}

#[cfg(target_arch = "x86_64")]
fn flash_head(q: &[f32], kt: &[f32], v: &[f32], out: &mut [f32], seq: usize) {
    flash_head_strided(q, kt, v, out, seq, HD);
}

/// `out_stride` is the distance between consecutive query rows in `out`.
///
/// With `HD` the head writes a contiguous (seq, HD) block, which the caller
/// then has to transpose into (seq, heads*HD) — measured at 13 ms in context
/// across the encoder, 7x what a standalone probe suggested. With
/// `heads * HD` the head writes straight into its own column band of the
/// merged buffer and the transpose disappears: the kernel was already storing
/// element by element, so a different offset costs nothing.
#[cfg(target_arch = "x86_64")]
fn flash_head_strided(
    q: &[f32],
    kt: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq: usize,
    out_stride: usize,
) {
    let mut scores = vec![0f32; BQ * BK];
    let mut acc = vec![0f32; BQ * HD];
    let mut run_max = vec![0f32; BQ];
    let mut run_sum = vec![0f32; BQ];

    for q0 in (0..seq).step_by(BQ) {
        let rows = BQ.min(seq - q0);
        acc[..rows * HD].fill(0.0);
        run_max[..rows].fill(f32::NEG_INFINITY);
        run_sum[..rows].fill(0.0);
        let qblock = &q[q0 * HD..(q0 + rows) * HD];

        for k0 in (0..seq).step_by(BK) {
            let cols = BK.min(seq - k0);
            // SAFETY: every callee here (`scores_tile`, `softmax_row`, `accum_pv`) is
            // avx2+fma-gated, and this function is only reached once `have_avx2()` has
            // passed. Slice bounds: `rows`/`cols` are clamped to the block size against
            // `seq` immediately above, so no tile read crosses the end of `q`/`kt`.
            unsafe {
                scores_tile(qblock, kt, &mut scores, rows, k0, cols, seq);
                for i in 0..rows {
                    let (new_max, correction, block_sum) =
                        softmax_row(&mut scores[i * BK..(i + 1) * BK], cols, run_max[i]);
                    run_sum[i] = run_sum[i].mul_add(correction, block_sum);
                    run_max[i] = new_max;
                    // Exact compare on purpose: 1.0 means softmax produced no
                    // rescale, so the whole accumulator pass is skipped. A near-1.0
                    // value simply does the (harmless) multiply.
                    #[allow(clippy::float_cmp)]
                    if correction != 1.0 {
                        for x in &mut acc[i * HD..(i + 1) * HD] {
                            *x *= correction;
                        }
                    }
                }
                accum_pv(&scores, v, &mut acc, rows, k0, cols);
            }
        }

        for i in 0..rows {
            let inv = 1.0 / run_sum[i];
            let base = (q0 + i) * out_stride;
            for (o, &a) in out[base..base + HD]
                .iter_mut()
                .zip(&acc[i * HD..(i + 1) * HD])
            {
                *o = a * inv;
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
// SAFETY: non-x86 stub. It is `unsafe fn` only to match the signature of the
// x86 kernel it stands in for, and its body is `unreachable!` - the runtime
// feature check that guards every call site can never pass on this target,
// so there is no reachable code here to be unsound.
unsafe fn xattn_head(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _o: &mut [f32],
    _n: usize,
    _s: &mut [f32],
) {
    unreachable!("guarded by have_avx2()")
}

#[cfg(not(target_arch = "x86_64"))]
fn flash_head(_q: &[f32], _kt: &[f32], _v: &[f32], _out: &mut [f32], _seq: usize) {
    unreachable!("guarded by have_avx2()")
}

// The two stubs below were MISSING, and mercury therefore did not compile at
// all on any non-x86_64 target. Nothing caught it: every developer box and every
// CI job was x86_64 until the release matrix built for aarch64-apple-darwin.
//
// Same contract as the stubs above - the runtime feature check that guards each
// call site can never pass here, so no reachable code exists to be unsound.
#[cfg(not(target_arch = "x86_64"))]
fn flash_head_strided(
    _q: &[f32],
    _kt: &[f32],
    _v: &[f32],
    _out: &mut [f32],
    _seq: usize,
    _out_stride: usize,
) {
    unreachable!("guarded by have_avx2()")
}

#[cfg(not(target_arch = "x86_64"))]
// SAFETY: non-x86 stub, `unsafe fn` only to match the kernel it stands in for.
unsafe fn xattn_head_f16(
    _q: &[f32],
    _kt: &[u16],
    _v: &[u16],
    _out: &mut [f32],
    _keys: usize,
    _s: &mut [f32],
) {
    unreachable!("guarded by have_f16c()")
}

// ---------------------------------------------------------------------------
// AVX2 kernels
// ---------------------------------------------------------------------------

/// `exp` to ~1e-7 relative: `exp(x) = 2^(x·log2e)`, integer part written
/// straight into the f32 exponent field, fractional part by a degree-5
/// minimax polynomial on [-0.5, 0.5].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: caller must ensure avx2+fma are available. Every call site is
// dominated by a `have_avx2()` check; the non-x86 build replaces this with an
// `unreachable!` stub.
unsafe fn exp256(x: __m256) -> __m256 {
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
    let t = _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E));
    let k = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let f = _mm256_sub_ps(t, k);

    let mut p = _mm256_set1_ps(0.001_333_355_8);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.009_618_129));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.055_504_11));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.240_226_5));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.693_147_2));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));

    let ki = _mm256_cvtps_epi32(k);
    let pow2k = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ki, _mm256_set1_epi32(127)),
        23,
    ));
    _mm256_mul_ps(p, pow2k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: avx2+fma required; see `exp256`. Pure register shuffle - no memory
// access, so feature availability is the only obligation.
unsafe fn hmax256(v: __m256) -> f32 {
    let m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    let m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    _mm_cvtss_f32(m)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: avx2+fma required; see `exp256`. Pure register reduction.
unsafe fn hsum256(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// One head. `q` is HD floats (already scaled); `kt` is (HD, keys) head-dim
/// major; `v` is (keys, HD). `out` is HD floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: avx2+fma required. Caller must also pass `q`/`kt`/`v` of at least
// `keys`-derived length and an `out` of HD elements; the strided callers above
// slice exactly those extents.
unsafe fn xattn_head(
    q: &[f32],
    kt: &[f32],
    v: &[f32],
    out: &mut [f32],
    keys: usize,
    s: &mut [f32],
) {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        // ---- scores = q @ K, contraction outermost so the inner loop is an AXPY
        // over keys with no horizontal reduction ----
        s[..keys].fill(0.0);
        for t in 0..HD {
            let qv = _mm256_set1_ps(*q.get_unchecked(t));
            let krow = kt.as_ptr().add(t * keys);
            let sp = s.as_mut_ptr();
            let mut j = 0;
            while j + 8 <= keys {
                let acc =
                    _mm256_fmadd_ps(qv, _mm256_loadu_ps(krow.add(j)), _mm256_loadu_ps(sp.add(j)));
                _mm256_storeu_ps(sp.add(j), acc);
                j += 8;
            }
            let qs = *q.get_unchecked(t);
            for jj in j..keys {
                *s.get_unchecked_mut(jj) += qs * *krow.add(jj);
            }
        }

        // ---- softmax over the whole key axis ----
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut j = 0;
        while j + 8 <= keys {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(s.as_ptr().add(j)));
            j += 8;
        }
        let mut mx = if keys >= 8 {
            hmax256(vmax)
        } else {
            f32::NEG_INFINITY
        };
        for &x in &s[j..keys] {
            mx = mx.max(x);
        }
        let vmx = _mm256_set1_ps(mx);
        let mut vsum = _mm256_setzero_ps();
        let mut j = 0;
        while j + 8 <= keys {
            let p = exp256(_mm256_sub_ps(_mm256_loadu_ps(s.as_ptr().add(j)), vmx));
            _mm256_storeu_ps(s.as_mut_ptr().add(j), p);
            vsum = _mm256_add_ps(vsum, p);
            j += 8;
        }
        let mut sum = if keys >= 8 { hsum256(vsum) } else { 0.0 };
        for x in &mut s[j..keys] {
            *x = ffai_core::fastmath::exp(*x - mx);
            sum += *x;
        }
        let inv = 1.0 / sum;

        // ---- out = weights @ V, AXPY over HD (8 vectors), V read once ----
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut a4 = _mm256_setzero_ps();
        let mut a5 = _mm256_setzero_ps();
        let mut a6 = _mm256_setzero_ps();
        let mut a7 = _mm256_setzero_ps();
        for jj in 0..keys {
            let w = _mm256_set1_ps(*s.get_unchecked(jj) * inv);
            let vp = v.as_ptr().add(jj * HD);
            a0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp), a0);
            a1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(8)), a1);
            a2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(16)), a2);
            a3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(24)), a3);
            a4 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(32)), a4);
            a5 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(40)), a5);
            a6 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(48)), a6);
            a7 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(56)), a7);
        }
        let op = out.as_mut_ptr();
        _mm256_storeu_ps(op, a0);
        _mm256_storeu_ps(op.add(8), a1);
        _mm256_storeu_ps(op.add(16), a2);
        _mm256_storeu_ps(op.add(24), a3);
        _mm256_storeu_ps(op.add(32), a4);
        _mm256_storeu_ps(op.add(40), a5);
        _mm256_storeu_ps(op.add(48), a6);
        _mm256_storeu_ps(op.add(56), a7);
    }
}

/// Decode-shape head with an **f16 K/V cache**, widened on the fly.
///
/// The cross-attention cache is read in full on every generated token —
/// 18.4 MB per token at f32 across the stack, which after the decoder weights
/// went to f16 is the largest remaining per-token read in decode. Storing it
/// f16 halves that to 9.2 MB; `_mm256_cvtph_ps` widens 8 values per
/// instruction as they stream, so the accumulation still happens in f32.
///
/// This is the same trick that took the f16 GEMV from 0.946x to 1.144x: pay
/// f16's traffic, keep f32's arithmetic. NOT f16 arithmetic — an earlier
/// attempt at an f16 K/V chain (§6.16) lost precisely because the softmax
/// between the two matmuls ran 82 % slower in half precision.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
// SAFETY: avx2+fma+f16c required (this variant widens f16 K/V in-register).
// Length contract as for `xattn_head`, with `kt`/`v` in f16 bit patterns.
unsafe fn xattn_head_f16(
    q: &[f32],
    kt: &[u16],
    v: &[u16],
    out: &mut [f32],
    keys: usize,
    s: &mut [f32],
) {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        s[..keys].fill(0.0);
        for t in 0..HD {
            let qv = _mm256_set1_ps(*q.get_unchecked(t));
            let krow = kt.as_ptr().add(t * keys);
            let sp = s.as_mut_ptr();
            let mut j = 0;
            while j + 8 <= keys {
                let kf = _mm256_cvtph_ps(_mm_loadu_si128(krow.add(j).cast::<__m128i>()));
                _mm256_storeu_ps(
                    sp.add(j),
                    _mm256_fmadd_ps(qv, kf, _mm256_loadu_ps(sp.add(j))),
                );
                j += 8;
            }
            let qs = *q.get_unchecked(t);
            for jj in j..keys {
                *s.get_unchecked_mut(jj) += qs * f32::from(half::f16::from_bits(*krow.add(jj)));
            }
        }

        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut j = 0;
        while j + 8 <= keys {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(s.as_ptr().add(j)));
            j += 8;
        }
        let mut mx = if keys >= 8 {
            hmax256(vmax)
        } else {
            f32::NEG_INFINITY
        };
        for &x in &s[j..keys] {
            mx = mx.max(x);
        }
        let vmx = _mm256_set1_ps(mx);
        let mut vsum = _mm256_setzero_ps();
        let mut j = 0;
        while j + 8 <= keys {
            let p = exp256(_mm256_sub_ps(_mm256_loadu_ps(s.as_ptr().add(j)), vmx));
            _mm256_storeu_ps(s.as_mut_ptr().add(j), p);
            vsum = _mm256_add_ps(vsum, p);
            j += 8;
        }
        let mut sum = if keys >= 8 { hsum256(vsum) } else { 0.0 };
        for x in &mut s[j..keys] {
            *x = ffai_core::fastmath::exp(*x - mx);
            sum += *x;
        }
        let inv = 1.0 / sum;

        let mut a: [__m256; 8] = [_mm256_setzero_ps(); 8];
        for jj in 0..keys {
            let w = _mm256_set1_ps(*s.get_unchecked(jj) * inv);
            let vp = v.as_ptr().add(jj * HD);
            for (c, acc) in a.iter_mut().enumerate() {
                let vf = _mm256_cvtph_ps(_mm_loadu_si128(vp.add(c * 8).cast::<__m128i>()));
                *acc = _mm256_fmadd_ps(w, vf, *acc);
            }
        }
        for (c, acc) in a.iter().enumerate() {
            _mm256_storeu_ps(out.as_mut_ptr().add(c * 8), *acc);
        }
    }
}

/// `S = Q_block @ K_block`, 4 query rows × 16 key columns held in registers
/// across the whole 64-step contraction. Each pair of K loads feeds 8 FMAs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: avx2+fma required. `rows`/`cols` must not exceed the caller's tile
// bounds - the flash loop clamps both against `seq` before calling.
unsafe fn scores_tile(
    qs: &[f32],
    kt: &[f32],
    scores: &mut [f32],
    rows: usize,
    k0: usize,
    cols: usize,
    seq: usize,
) {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        let kbase = kt.as_ptr();
        let mut i0 = 0;
        while i0 + 4 <= rows {
            let mut j0 = 0;
            while j0 + 16 <= cols {
                let (mut a00, mut a01) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                let (mut a10, mut a11) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                let (mut a20, mut a21) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                let (mut a30, mut a31) = (_mm256_setzero_ps(), _mm256_setzero_ps());

                for t in 0..HD {
                    let kp = kbase.add(t * seq + k0 + j0);
                    let kv0 = _mm256_loadu_ps(kp);
                    let kv1 = _mm256_loadu_ps(kp.add(8));

                    let q0 = _mm256_set1_ps(*qs.get_unchecked(i0 * HD + t));
                    let q1 = _mm256_set1_ps(*qs.get_unchecked((i0 + 1) * HD + t));
                    let q2 = _mm256_set1_ps(*qs.get_unchecked((i0 + 2) * HD + t));
                    let q3 = _mm256_set1_ps(*qs.get_unchecked((i0 + 3) * HD + t));

                    a00 = _mm256_fmadd_ps(q0, kv0, a00);
                    a01 = _mm256_fmadd_ps(q0, kv1, a01);
                    a10 = _mm256_fmadd_ps(q1, kv0, a10);
                    a11 = _mm256_fmadd_ps(q1, kv1, a11);
                    a20 = _mm256_fmadd_ps(q2, kv0, a20);
                    a21 = _mm256_fmadd_ps(q2, kv1, a21);
                    a30 = _mm256_fmadd_ps(q3, kv0, a30);
                    a31 = _mm256_fmadd_ps(q3, kv1, a31);
                }

                let sp = scores.as_mut_ptr();
                _mm256_storeu_ps(sp.add(i0 * BK + j0), a00);
                _mm256_storeu_ps(sp.add(i0 * BK + j0 + 8), a01);
                _mm256_storeu_ps(sp.add((i0 + 1) * BK + j0), a10);
                _mm256_storeu_ps(sp.add((i0 + 1) * BK + j0 + 8), a11);
                _mm256_storeu_ps(sp.add((i0 + 2) * BK + j0), a20);
                _mm256_storeu_ps(sp.add((i0 + 2) * BK + j0 + 8), a21);
                _mm256_storeu_ps(sp.add((i0 + 3) * BK + j0), a30);
                _mm256_storeu_ps(sp.add((i0 + 3) * BK + j0 + 8), a31);
                j0 += 16;
            }
            for j in j0..cols {
                for m in 0..4 {
                    let mut a = 0f32;
                    for t in 0..HD {
                        a += qs.get_unchecked((i0 + m) * HD + t)
                            * kt.get_unchecked(t * seq + k0 + j);
                    }
                    *scores.get_unchecked_mut((i0 + m) * BK + j) = a;
                }
            }
            i0 += 4;
        }
        for i in i0..rows {
            for j in 0..cols {
                let mut a = 0f32;
                for t in 0..HD {
                    a += qs.get_unchecked(i * HD + t) * kt.get_unchecked(t * seq + k0 + j);
                }
                *scores.get_unchecked_mut(i * BK + j) = a;
            }
        }
    }
}

/// `acc += P_block @ V_block`, 4 rows × 16 head-dims in registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: avx2+fma required. `scores` must hold `rows*BK` elements and `acc`
// `rows*HD`; both are allocated at those sizes by the flash loop.
unsafe fn accum_pv(
    scores: &[f32],
    v: &[f32],
    acc: &mut [f32],
    rows: usize,
    k0: usize,
    cols: usize,
) {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        let vbase = v.as_ptr();
        let mut i0 = 0;
        while i0 + 4 <= rows {
            let mut d0 = 0;
            while d0 < HD {
                let ap = acc.as_mut_ptr();
                let mut a00 = _mm256_loadu_ps(ap.add(i0 * HD + d0));
                let mut a01 = _mm256_loadu_ps(ap.add(i0 * HD + d0 + 8));
                let mut a10 = _mm256_loadu_ps(ap.add((i0 + 1) * HD + d0));
                let mut a11 = _mm256_loadu_ps(ap.add((i0 + 1) * HD + d0 + 8));
                let mut a20 = _mm256_loadu_ps(ap.add((i0 + 2) * HD + d0));
                let mut a21 = _mm256_loadu_ps(ap.add((i0 + 2) * HD + d0 + 8));
                let mut a30 = _mm256_loadu_ps(ap.add((i0 + 3) * HD + d0));
                let mut a31 = _mm256_loadu_ps(ap.add((i0 + 3) * HD + d0 + 8));

                for j in 0..cols {
                    let vp = vbase.add((k0 + j) * HD + d0);
                    let vv0 = _mm256_loadu_ps(vp);
                    let vv1 = _mm256_loadu_ps(vp.add(8));

                    let p0 = _mm256_set1_ps(*scores.get_unchecked(i0 * BK + j));
                    let p1 = _mm256_set1_ps(*scores.get_unchecked((i0 + 1) * BK + j));
                    let p2 = _mm256_set1_ps(*scores.get_unchecked((i0 + 2) * BK + j));
                    let p3 = _mm256_set1_ps(*scores.get_unchecked((i0 + 3) * BK + j));

                    a00 = _mm256_fmadd_ps(p0, vv0, a00);
                    a01 = _mm256_fmadd_ps(p0, vv1, a01);
                    a10 = _mm256_fmadd_ps(p1, vv0, a10);
                    a11 = _mm256_fmadd_ps(p1, vv1, a11);
                    a20 = _mm256_fmadd_ps(p2, vv0, a20);
                    a21 = _mm256_fmadd_ps(p2, vv1, a21);
                    a30 = _mm256_fmadd_ps(p3, vv0, a30);
                    a31 = _mm256_fmadd_ps(p3, vv1, a31);
                }

                _mm256_storeu_ps(ap.add(i0 * HD + d0), a00);
                _mm256_storeu_ps(ap.add(i0 * HD + d0 + 8), a01);
                _mm256_storeu_ps(ap.add((i0 + 1) * HD + d0), a10);
                _mm256_storeu_ps(ap.add((i0 + 1) * HD + d0 + 8), a11);
                _mm256_storeu_ps(ap.add((i0 + 2) * HD + d0), a20);
                _mm256_storeu_ps(ap.add((i0 + 2) * HD + d0 + 8), a21);
                _mm256_storeu_ps(ap.add((i0 + 3) * HD + d0), a30);
                _mm256_storeu_ps(ap.add((i0 + 3) * HD + d0 + 8), a31);
                d0 += 16;
            }
            i0 += 4;
        }
        for i in i0..rows {
            for j in 0..cols {
                let p = *scores.get_unchecked(i * BK + j);
                for d in 0..HD {
                    *acc.get_unchecked_mut(i * HD + d) += p * v.get_unchecked((k0 + j) * HD + d);
                }
            }
        }
    }
}

/// Online softmax over one score row → (new running max, correction, sum).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: avx2+fma required. `row` must hold at least `cols` elements; the
// caller slices it to exactly BK and passes `cols <= BK`.
unsafe fn softmax_row(row: &mut [f32], cols: usize, run_max: f32) -> (f32, f32, f32) {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut j = 0;
        while j + 8 <= cols {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row.as_ptr().add(j)));
            j += 8;
        }
        let mut block_max = if cols >= 8 {
            hmax256(vmax)
        } else {
            f32::NEG_INFINITY
        };
        for &s in &row[j..cols] {
            block_max = block_max.max(s);
        }

        let new_max = run_max.max(block_max);
        let correction = if run_max.is_finite() {
            ffai_core::fastmath::exp(run_max - new_max)
        } else {
            0.0
        };

        let vnm = _mm256_set1_ps(new_max);
        let mut vsum = _mm256_setzero_ps();
        let mut j = 0;
        while j + 8 <= cols {
            let p = exp256(_mm256_sub_ps(_mm256_loadu_ps(row.as_ptr().add(j)), vnm));
            _mm256_storeu_ps(row.as_mut_ptr().add(j), p);
            vsum = _mm256_add_ps(vsum, p);
            j += 8;
        }
        let mut block_sum = if cols >= 8 { hsum256(vsum) } else { 0.0 };
        for s in &mut row[j..cols] {
            *s = ffai_core::fastmath::exp(*s - new_max);
            block_sum += *s;
        }
        (new_max, correction, block_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel must agree with the three-op path it replaces.
    #[test]
    fn matches_three_op_path() -> CandleResult<()> {
        let dev = Device::Cpu;
        let (h, seq) = (4usize, 512usize);
        let q = Tensor::randn(0f32, 1., (1, h, seq, HD), &dev)?.contiguous()?;
        let k = Tensor::randn(0f32, 1., (1, h, HD, seq), &dev)?.contiguous()?;
        let v = Tensor::randn(0f32, 1., (1, h, seq, HD), &dev)?.contiguous()?;

        let Some(ours) = attend(&q, &k, &v, false)? else {
            return Ok(()); // no AVX2 here; nothing to check
        };
        let want = candle_nn::ops::softmax_last_dim(&q.matmul(&k)?)?.matmul(&v)?;

        let d: f32 = (ours - want)?.abs()?.max_all()?.to_scalar()?;
        assert!(d < 1e-4, "max |delta| {d:e} exceeds tolerance");
        Ok(())
    }

    /// Shapes the kernel does not serve must decline, not produce garbage.
    #[test]
    fn declines_unsupported_shapes() -> CandleResult<()> {
        let dev = Device::Cpu;
        // Short query sequence (the decoder's case).
        let q = Tensor::randn(0f32, 1., (1, 4, 1, HD), &dev)?;
        let k = Tensor::randn(0f32, 1., (1, 4, HD, 1), &dev)?;
        let v = Tensor::randn(0f32, 1., (1, 4, 1, HD), &dev)?;
        assert!(attend(&q, &k, &v, false)?.is_none());

        // Masked attention.
        let q = Tensor::randn(0f32, 1., (1, 4, 512, HD), &dev)?;
        let k = Tensor::randn(0f32, 1., (1, 4, HD, 512), &dev)?;
        let v = Tensor::randn(0f32, 1., (1, 4, 512, HD), &dev)?;
        assert!(attend(&q, &k, &v, true)?.is_none());
        Ok(())
    }
}
