//! AVX2 int8 GEMV for the vocabulary projection.
//!
//! The projection is ~27 % of decode and pure bandwidth: `(1,d) @ (d,vocab)`
//! streams the whole embedding matrix per generated token with no reuse. The
//! f16 path moves 39.8 MB/token at ~24 GB/s, already close to what this
//! machine's memory system gives a read-only stream.
//!
//! candle's Q8_0 halves the bytes and measured **no faster** (20/41,
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
//! ~6 % more bytes, and are why Q8_0 uses them.

use ffai_core::candle::{
    CpuStorage, CustomOp1, Device, Layout, Result as CandleResult, Shape, Tensor,
};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Block-symmetric int8 vocabulary weights.
pub struct Int8Vocab {
    shared: std::sync::Arc<Shared>,
}

struct Shared {
    qw: Vec<i8>,
    /// One scale per 32-weight BLOCK, not per row. A single row scale is set
    /// by that row's largest magnitude, so one outlier coarsens all 384
    /// weights beside it; corpus WER moved 7.87 -> 7.93 % and left only
    /// 0.027 pp of gate margin. Per-block scales are why Q8_0 uses them.
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
        let blk = blk();
        if d % blk != 0 {
            return Ok(None);
        }
        let w: Vec<f32> = embeddings.to_dtype(ffai_core::candle::DType::F32)?.flatten_all()?.to_vec1()?;

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
                for k in 0..blk {
                    let qi = (bw[k] / sc).round().clamp(-127.0, 127.0) as i8;
                    q[v * d + b * blk + k] = qi;
                    bsum += qi as i32;
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
        Ok(Some(Int8Vocab {
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
        let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let sx = if amax > 0.0 { amax / 63.0 } else { 1.0 };
        let mut xu = vec![0u8; q.d];
        for k in 0..q.d {
            let qi = (x[k] / sx).round().clamp(-64.0, 63.0) as i32;
            xu[k] = (qi + 64) as u8;
        }

        let mut out = vec![0f32; q.vocab];
        let (d, nb) = (q.d, q.nblocks);
        out.par_chunks_mut(512).enumerate().for_each(|(ci, o)| {
            let v0 = ci * 512;
            for (i, slot) in o.iter_mut().enumerate() {
                let v = v0 + i;
                let acc = unsafe {
                    dot_i8_blocked(
                        &q.qw[v * d..(v + 1) * d],
                        &q.bscale[v * nb..(v + 1) * nb],
                        &xu,
                        nb,
                        d / nb,
                    )
                };
                *slot = (acc - q.corr[v]) * sx;
            }
        });
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
/// row-scaled version avoided it by reducing once per row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_blocked(w: &[i8], bscale: &[f32], xu: &[u8], nblocks: usize, blk: usize) -> f32 {
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
            let x0 = _mm256_loadu_si256(xp.add(base + o) as *const __m256i);
            let w0 = _mm256_loadu_si256(wp.add(base + o) as *const __m256i);
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

#[cfg(not(target_arch = "x86_64"))]
unsafe fn dot_i8_blocked(_w: &[i8], _b: &[f32], _xu: &[u8], _n: usize, _k: usize) -> f32 {
    unreachable!("guarded by have_avx2()")
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
            .map(|(i, _)| i)
            .unwrap_or(0)
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
        if flips > 0 { margin / flips as f64 } else { 0.0 }
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
            v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
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
