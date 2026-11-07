//! Mercury's own Whisper text decoder, with **incremental** decoding.
//!
//! # Why this exists
//!
//! The M2 profile put 77 % of transcription time in the decoder, and showed
//! per-token cost climbing with sequence length (19 ms at 10 tokens, 27.8 ms
//! at 50). That climb is the cost of recomputing the forward pass for every
//! position on every step when only the last position's logits are used.
//!
//! candle-transformers' `TextDecoder` cannot avoid it: it narrows the
//! positional embedding from index 0 and recomputes self-attention K/V from
//! the whole input each call, so it is structurally a full-sequence decoder.
//! Fixing that means owning the decoder blocks — the boundary the mission
//! plan always said we would move when speed demanded it.
//!
//! What this does instead:
//!
//! - after the prompt, **one token per step** enters the network;
//! - **self-attention K/V are cached and appended**, so step n attends over n
//!   cached keys instead of recomputing all of them;
//! - **cross-attention K/V are computed once per 30 s window** — they depend
//!   only on the encoder output, so recomputing them per token was pure waste;
//! - the positional embedding is offset by the number of tokens already
//!   consumed, which is what makes feeding a single token correct.
//!
//! Weights load from the same HF safetensors layout candle uses, so this is a
//! drop-in twin — and `tests/oracle_decoder.rs` holds it to producing
//! *identical token ids* to the candle path. A faster decoder that changes the
//! output is not a faster decoder, it is a different model.

use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use ffai_core::candle::quantized::{GgmlDType, QMatMul, QTensor};
use ffai_core::candle::{D, DType, IndexOp, Result as CandleResult, Tensor};

/// Whisper uses eps 1e-5 throughout.
const LAYER_NORM_EPS: f64 = 1e-5;

/// Weight precision for the decoder's matmuls.
///
/// Quantization is applied **at load time from the same f32 safetensors** we
/// already fetch — there is no second set of GGUF weights to download, keep
/// in sync, or license separately. One weight source, several precisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    F32,
    /// 8-bit, 32-element blocks with an f16 scale — the precision
    /// faster-whisper runs by default.
    Q8_0,
    /// 4-bit k-quant: smaller and faster still, at more accuracy risk.
    Q4K,
}

impl Precision {
    const fn ggml(self) -> Option<GgmlDType> {
        match self {
            Self::F32 => None,
            Self::Q8_0 => Some(GgmlDType::Q8_0),
            Self::Q4K => Some(GgmlDType::Q4K),
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4k",
        }
    }
}

/// A linear layer that is either full-precision or quantized.
///
/// Quantized matmul carries its own bias separately because `QMatMul` is a
/// pure `x @ Wᵀ`.
pub enum QLinear {
    // MEASURED AND REVERTED: GEMV padding on these projections.
    //
    // The cliff microbenchmark promised 1.46x on mlp fc1 at m=2, and the
    // isolated numbers were real. In context it lost: process-level paired
    // A/B over the whole engine gave **0/13 rounds, z = -3.6, 0.847x**. The
    // pre-transposed weight copies cost memory the decoder then has to stream,
    // and on cold cache the padded matmul does not recover it.
    //
    // Padding stays where it was PROVEN end to end — the vocabulary
    // projection (VocabProj), +5.7 % at 15/15 rounds. A cliff that exists in a
    // microbenchmark is not a cliff that exists in the pipeline.
    Full(Linear),
    Quant {
        matmul: QMatMul,
        bias: Option<Tensor>,
    },
    /// int8 GEMV for layers that only ever see ONE input row.
    ///
    /// The decoder MLP is 18.9 MB of weights per token across the stack and
    /// measured 16.9 GB/s — the same bandwidth-bound single-row shape the
    /// vocabulary projection was, where this kernel gave 4.34x. The encoder
    /// MLP has the same weights but M=1500, which is a real GEMM that
    /// candle's `gemm` already does well, so `full` stays live and serves
    /// anything wider than one row.
    Int8 {
        q: super::vocab_int8::Int8Vocab,
        bias: Option<Tensor>,
    },
    /// f16 weights, pre-transposed, for layers fed ONE row at a time.
    ///
    /// This is the precision the reference actually runs — its tiny.en file is
    /// 77.7 MB, i.e. f16, against 156 MB for f32 — and matching it is the
    /// honest way to close the decode gap. int8 undercuts their traffic (8.3
    /// vs 16.5 MB/token) but pays in error that COMPOUNDS through the residual
    /// stream, and failed the corpus quality gate at 8.39 % WER. f16 keeps
    /// ~3 decimal digits, which is why the reference reaches 7.58 % on it.
    Half {
        k: super::f16_gemv::F16Gemv,
        bias: Option<Tensor>,
        /// The same bias as plain f32, for the kernel to add in place
        /// (`F16Gemv::forward_fused`).
        bias_f32: Option<std::sync::Arc<Vec<f32>>>,
    },
}

/// `FFAI_KV_F16=off` keeps the cross-attention cache at f32 — the A/B arm.
#[inline]
fn kv_f16_disabled() -> bool {
    super::knobs::KV_F16_DISABLED.get()
}

/// `FFAI_DEC_F16=off` forces the decoder's single-row projections back to f32.
#[inline]
fn dec_f16_disabled() -> bool {
    super::knobs::DEC_F16_DISABLED.get()
}

/// `FFAI_MLP_INT8=on` puts the decoder MLP on int8 instead of f16.
#[inline]
fn mlp_int8_enabled() -> bool {
    super::knobs::MLP_INT8_ENABLED.get()
}

/// `FFAI_MLP_INT8=off` forces the decoder MLP back to the GEMM path.
#[inline]
fn mlp_int8_disabled() -> bool {
    super::knobs::MLP_INT8_DISABLED.get()
}

/// Apply a single-row kernel over `rows` rows.
///
/// These layers previously kept `full: Linear` — a second, f32 copy of the
/// whole weight — purely to serve inputs wider than one row. That fallback
/// made quantization ADD memory instead of replacing it: measured, `q8_0` held
/// 373 MiB against f32's 392, a 4.7 % saving where the variant is sold on
/// "~4x smaller weights". The vocabulary projection alone kept ~80 MB of f32
/// beside ~20 MB of int8.
///
/// Every caller of these constructors feeds ONE row (a decoder step projects a
/// single token; the 1500-row cross-attention K/V projections take the plain
/// `Linear` path via `linear_no_bias`), so the fallback was cold weight. The
/// multi-row case still works — it loops the kernel — it is simply no longer
/// paid for in resident memory by every model that never uses it.
fn gemv_rows(
    rows: usize,
    flat: &Tensor,
    mut f: impl FnMut(&Tensor) -> CandleResult<Tensor>,
) -> CandleResult<Tensor> {
    if rows == 1 {
        return f(flat);
    }
    let mut outs = Vec::with_capacity(rows);
    for r in 0..rows {
        outs.push(f(&flat.narrow(0, r, 1)?)?);
    }
    Tensor::cat(&outs, 0)
}

impl QLinear {
    /// Load a linear layer from a `VarBuilder` at the given precision.
    pub fn from_vb(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        bias: bool,
        precision: Precision,
    ) -> CandleResult<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = if bias {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        Self::new(weight, bias, precision)
    }

    /// Load a layer that will only ever be fed one row at a time.
    /// **Currently unreachable — retained deliberately, not stranded.**
    ///
    /// The int8 decoder path it builds was measured and reverted twice: alone
    /// on the MLP it was inside the noise (1.005x), and across every
    /// projection it failed the corpus quality gate at 8.39 % WER. Those
    /// projections now use f16 (`from_vb_gemv_f16`), which passes.
    ///
    /// Kept because re-testing a deleted experiment costs its whole rebuild.
    /// Note the case has since got WORSE, not better: int8 was measured
    /// against an f32 baseline (4x traffic advantage) and the baseline is now
    /// f16, so it would buy 2x for the same quality risk that already failed.
    pub fn from_vb_gemv(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        bias: bool,
        precision: Precision,
    ) -> CandleResult<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = if bias {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        // A/B escape hatch. Resolved ONCE here at load, never per call — an
        // earlier optimization in this crate regressed itself 3x by reading an
        // env var inside the per-token path.
        if mlp_int8_disabled() {
            return Self::new(weight, bias, precision);
        }
        // Weight is already (out, in) row-major — the layout the kernel reads.
        match super::vocab_int8::Int8Vocab::new(&weight)? {
            Some(q) => Ok(Self::Int8 { q, bias }),
            None => Self::new(weight, bias, precision),
        }
    }

    /// f16 GEMV loader: match the reference's precision on the single-row
    /// paths. Pays the transpose once here rather than once per token.
    pub fn from_vb_gemv_f16(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        bias: bool,
        precision: Precision,
    ) -> CandleResult<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = if bias {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        // A/B escape hatch, resolved once at load.
        if dec_f16_disabled() {
            return Self::new(weight, bias, precision);
        }
        match super::f16_gemv::F16Gemv::new(&weight)? {
            Some(k) => {
                let bias_f32 = match &bias {
                    Some(b) => Some(std::sync::Arc::new(
                        b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    )),
                    None => None,
                };
                Ok(Self::Half { k, bias, bias_f32 })
            }
            None => Self::new(weight, bias, precision),
        }
    }

    fn new(weight: Tensor, bias: Option<Tensor>, precision: Precision) -> CandleResult<Self> {
        match precision.ggml() {
            // Quantization needs the contracted dimension to be a multiple of
            // the block size (32). Whisper's d_model values (384, 512, 768,
            // 1024, 1280) all satisfy this; anything else falls back to f32
            // rather than failing to load.
            Some(dtype) if weight.dim(D::Minus1)? % dtype.block_size() == 0 => {
                let q = QTensor::quantize(&weight, dtype)?;
                Ok(Self::Quant {
                    matmul: QMatMul::from_qtensor(q)?,
                    bias,
                })
            }
            _ => Ok(Self::Full(Linear::new(weight, bias))),
        }
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Full(linear) => linear.forward(x),
            Self::Quant { matmul, bias } => {
                let y = matmul.forward(x)?;
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
            Self::Half { k, bias, bias_f32 } => {
                let dims = x.dims();
                let rows: usize = dims[..dims.len() - 1].iter().product();
                // Vein 3: the one-row case — every decoder projection — is a
                // single op. The kernel adds the bias as it writes and emits
                // the caller's rank, so the reshape pair and the
                // `broadcast_add` pass (4 ops, two 1.5 KB tensors) are gone.
                if rows == 1 && x.is_contiguous() && !super::knobs::VEINS_OFF.get() {
                    super::profile::counters().plumbing_ops.add(1);
                    return k.forward_fused(x, bias_f32.as_ref(), dims);
                }
                super::profile::counters().plumbing_ops.add(4);
                let flat = x.reshape((rows, dims[dims.len() - 1]))?;
                let y = gemv_rows(rows, &flat, |r| k.forward(r))?;
                let y = match bias {
                    Some(b) => y.broadcast_add(b)?,
                    None => y,
                };
                let mut out = dims.to_vec();
                let n = out.len();
                out[n - 1] = y.dim(1)?;
                y.reshape(out)
            }
            Self::Int8 { q, bias } => {
                let dims = x.dims();
                let rows: usize = dims[..dims.len() - 1].iter().product();
                let flat = x.reshape((rows, dims[dims.len() - 1]))?;
                let y = gemv_rows(rows, &flat, |r| q.forward(r))?;
                let y = match bias {
                    Some(b) => y.broadcast_add(b)?,
                    None => y,
                };
                // Restore the caller's rank, e.g. (1,1,d).
                let mut out = dims.to_vec();
                let n = out.len();
                out[n - 1] = y.dim(1)?;
                y.reshape(out)
            }
        }
    }
}

/// GELU without a transcendental call.
///
/// The encoder anatomy bench (mission plan §6.10) found GELU to be **39 % of
/// all encoder time** — 90 ms per pass, running at 0.8 GB/s against a
/// measured 15 GB/s memory ceiling. An elementwise activation has no business
/// being the most expensive operation in a transformer; the cause is `tanh`,
/// which is a scalar libm call the compiler cannot vectorize. A hand-written
/// loop using `tanh` measured the same as candle's, confirming the framework
/// was not at fault.
///
/// This replaces `tanh` with a degree-7/8 Padé rational approximation —
/// multiplies, adds and one divide, all of which auto-vectorize. Measured on
/// the encoder's (1500, 1536) activation:
///
/// | implementation | ms | GB/s | max abs error |
/// |---|---:|---:|---:|
/// | candle `.gelu()` (tanh) | 22.33 | 0.8 | — |
/// | low-order rational | 2.28 | 8.1 | 2.2e-2 |
/// | **this (Padé 7/8)** | **2.58** | **7.2** | **1.8e-4** |
/// | memcpy floor | 1.28 | 14.4 | — |
///
/// The low-order form was rejected despite being marginally faster: 2.2e-2
/// of error compounds across four layers, and 0.3 ms is not worth risking
/// the transcript for.
/// The scalar kernel. Kept separate so the custom op, the fallback path and
/// the tests all evaluate exactly the same arithmetic.
#[inline(always)]
fn gelu_scalar(t: f32) -> f32 {
    // Whisper's GELU: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
    let u = 0.797_884_6_f32 * (0.044715 * t * t).mul_add(t, t);
    // Clamp before squaring: the rational diverges outside its range, and
    // tanh has saturated to +/-1 long before |u| = 8 anyway.
    let u2 = (u * u).min(64.0);
    let num = u * (135135.0 + u2 * (17325.0 + u2 * (378.0 + u2)));
    let den = 135135.0 + u2 * (62370.0 + u2 * (3150.0 + u2 * 28.0));
    0.5 * t * (1.0 + (num / den).clamp(-1.0, 1.0))
}

/// A candle custom op, so the kernel reads the tensor's storage directly.
///
/// The first version round-tripped through `to_vec1` → `Vec` → `from_vec`,
/// which copied the 9 MB activation twice per call. On an op whose whole cost
/// is memory traffic that overhead was larger than the arithmetic: 2.58 ms
/// isolated became 4.7 ms in context.
struct FastGelu;

impl ffai_core::candle::CustomOp1 for FastGelu {
    fn name(&self) -> &'static str {
        "ffai-fast-gelu"
    }

    fn cpu_fwd(
        &self,
        storage: &ffai_core::candle::CpuStorage,
        layout: &ffai_core::candle::Layout,
    ) -> CandleResult<(ffai_core::candle::CpuStorage, ffai_core::candle::Shape)> {
        use crate::par::prelude::*;

        use ffai_core::candle::CpuStorage;

        // Chunked so each core owns a contiguous span; the inner loop is pure
        // arithmetic and vectorizes.
        const CHUNK: usize = 8192;
        let (start, end) = layout.contiguous_offsets().unwrap_or_else(|| {
            let s = layout.start_offset();
            (s, s + layout.shape().elem_count())
        });

        let out = match storage {
            CpuStorage::F32(src) => {
                let src = &src[start..end];
                // Every element is written below (vein 5): no zero pass.
                let out = crate::buf::filled_f32(src.len(), |out| {
                    out.par_chunks_mut(CHUNK)
                        .zip(src.par_chunks(CHUNK))
                        .for_each(|(o, i)| {
                            for (o, &t) in o.iter_mut().zip(i.iter()) {
                                *o = gelu_scalar(t);
                            }
                        });
                });
                CpuStorage::F32(out)
            }
            // f16 storage, f32 arithmetic — the same split ggml uses. Halving
            // the bytes is the point; computing the activation in half
            // precision would only lose accuracy for nothing.
            CpuStorage::F16(src) => {
                let src = &src[start..end];
                let out = crate::buf::filled_f16(src.len(), |out| {
                    out.par_chunks_mut(CHUNK)
                        .zip(src.par_chunks(CHUNK))
                        .for_each(|(o, i)| {
                            for (o, &t) in o.iter_mut().zip(i.iter()) {
                                *o = half::f16::from_f32(gelu_scalar(t.to_f32()));
                            }
                        });
                });
                CpuStorage::F16(out)
            }
            _ => {
                return Err(ffai_core::candle::Error::Msg(
                    "fast_gelu supports f32 and f16 storage only".to_string(),
                ));
            }
        };
        Ok((out, layout.shape().clone()))
    }
}

/// Softmax over the last dimension, with a real f16 path.
///
/// candle's `softmax_last_dim` is generic over `T: Float` and evaluates
/// `(*s - max).exp()` per element. For f16 that is a scalar `exp` wrapped in
/// **two conversions per element**, and it measured 82 % slower than the f32
/// version — which is what made the f16 cross-attention K/V cache a net loss
/// (§6.16): half precision won both matmuls and gave it all back here.
///
/// This converts each row to f32 **once**, computes in f32 (where the
/// exponential belongs numerically anyway), and converts back — so f16 costs
/// one pass, not one conversion per element.
struct FastSoftmax;

impl ffai_core::candle::CustomOp1 for FastSoftmax {
    fn name(&self) -> &'static str {
        "ffai-fast-softmax"
    }

    fn cpu_fwd(
        &self,
        storage: &ffai_core::candle::CpuStorage,
        layout: &ffai_core::candle::Layout,
    ) -> CandleResult<(ffai_core::candle::CpuStorage, ffai_core::candle::Shape)> {
        use crate::par::prelude::*;
        use ffai_core::candle::CpuStorage;

        let dims = layout.shape().dims();
        let last = *dims.last().unwrap_or(&1);
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            ffai_core::candle::Error::Msg("fast_softmax needs a contiguous input".into())
        })?;

        /// Three passes, all three vectorised.
        ///
        /// The shape this replaces is the one Argus's vision tower was found on
        /// and measured 8.8x slower for: `fold(NEG_INFINITY, f32::max)` is a
        /// loop-carried reduction on a NaN-aware function, `(s - max).exp()` is
        /// a libm CALL per element, and `sum += *d` is a loop-carried
        /// dependency on a non-associative add. LLVM lanes none of the three,
        /// and a scalar rewrite does not help — only explicit intrinsics do.
        ///
        /// `copy_from_slice` is `memcpy`, which dispatches to the widest store
        /// the target has; the hand-written copy loop it replaces did not.
        #[inline(always)]
        fn softmax_row(src: &[f32], dst: &mut [f32]) {
            let max = ffai_core::fastmath::max_f32(src);
            dst.copy_from_slice(src);
            softmax_inplace(dst, max);
        }

        /// The same three passes on a row that is already where it belongs.
        #[inline(always)]
        fn softmax_inplace(row: &mut [f32], max: f32) {
            let sum = ffai_core::fastmath::exp_sub_sum_inplace(row, max);
            let inv = 1.0 / sum;
            for d in row.iter_mut() {
                *d *= inv;
            }
        }

        let out = match storage {
            CpuStorage::F32(src) => {
                let src = &src[start..end];
                // Every row is written below (vein 5): no zero pass.
                let out = crate::buf::filled_f32(src.len(), |out| {
                    out.par_chunks_mut(last)
                        .zip(src.par_chunks(last))
                        .for_each(|(d, s)| softmax_row(s, d));
                });
                CpuStorage::F32(out)
            }
            CpuStorage::F16(src) => {
                let src = &src[start..end];
                let out = crate::buf::filled_f16(src.len(), |out| {
                    out.par_chunks_mut(last)
                        .zip(src.par_chunks(last))
                        .for_each(|(d, s)| {
                            // One conversion per row, not per element — and the
                            // widened row is the work buffer itself (vein 5): one
                            // allocation per row, no zeroed `tmp` beside it.
                            let mut row: Vec<f32> = s.iter().map(|x| x.to_f32()).collect();
                            let max = ffai_core::fastmath::max_f32(&row);
                            softmax_inplace(&mut row, max);
                            for (d, &t) in d.iter_mut().zip(row.iter()) {
                                *d = half::f16::from_f32(t);
                            }
                        });
                });
                CpuStorage::F16(out)
            }
            _ => {
                return Err(ffai_core::candle::Error::Msg(
                    "fast_softmax supports f32 and f16 storage only".into(),
                ));
            }
        };
        Ok((out, layout.shape().clone()))
    }
}

/// Softmax over the last dimension — see [`FastSoftmax`].
pub fn fast_softmax(x: &Tensor) -> CandleResult<Tensor> {
    x.apply_op1_no_bwd(&FastSoftmax)
}

/// GELU without a transcendental call — see [`FastGelu`].
pub fn fast_gelu(x: &Tensor) -> CandleResult<Tensor> {
    x.apply_op1_no_bwd(&FastGelu)
}

fn layer_norm(size: usize, vb: VarBuilder) -> CandleResult<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(size, "weight")?,
        vb.get(size, "bias")?,
        LAYER_NORM_EPS,
    ))
}

fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder, p: Precision) -> CandleResult<QLinear> {
    QLinear::new(
        vb.get((out_dim, in_dim), "weight")?,
        Some(vb.get(out_dim, "bias")?),
        p,
    )
}

fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    p: Precision,
) -> CandleResult<QLinear> {
    QLinear::new(vb.get((out_dim, in_dim), "weight")?, None, p)
}

/// Multi-head attention with a K/V cache.
/// Self-attention K/V kept in the fused decode kernel's own layout.
///
/// K is `[heads][HD][cap]` — the head-split, transposed operand the kernel
/// streams — and V is `[heads][cap][HD]`; `cap` grows geometrically up to the
/// model's position limit, so a short window never pays for 448 slots.
///
/// This replaces `Option<(Tensor, Tensor)>` + `Tensor::cat` for the decoder's
/// self-attention. That path cost, per layer per token, two `cat`s that
/// copied the whole cache and a `split_heads().transpose().contiguous()` pair
/// that copied it again — four passes over O(n) data for an O(1) append, plus
/// eleven candle ops of plumbing before the kernel saw a byte. Here a token
/// is written into place (2·n_state floats) and the kernel reads the cache
/// directly with a row stride of `cap`.
struct FlatKv {
    k: Vec<f32>,
    v: Vec<f32>,
    heads: usize,
    hd: usize,
    cap: usize,
    len: usize,
    max: usize,
    /// The kernel's score row, reused across heads and tokens.
    scratch: Vec<f32>,
}

/// A copied prefix of a [`FlatKv`], for beam search's save/restore.
#[derive(Clone)]
struct FlatKvState {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
}

impl FlatKv {
    const INITIAL_CAP: usize = 64;

    fn new(heads: usize, hd: usize, max: usize) -> Self {
        let cap = Self::INITIAL_CAP.min(max.max(1));
        Self {
            k: vec![0.0; heads * hd * cap],
            v: vec![0.0; heads * cap * hd],
            heads,
            hd,
            cap,
            len: 0,
            max,
            scratch: vec![0.0; cap],
        }
    }

    const fn clear(&mut self) {
        self.len = 0;
    }

    /// Re-lay both buffers for a new capacity, keeping the used prefix.
    fn regrow(&mut self, cap: usize) {
        let (heads, hd, len, old) = (self.heads, self.hd, self.len, self.cap);
        let mut k = vec![0f32; heads * hd * cap];
        for r in 0..heads * hd {
            k[r * cap..r * cap + len].copy_from_slice(&self.k[r * old..r * old + len]);
        }
        let mut v = vec![0f32; heads * cap * hd];
        for h in 0..heads {
            v[h * cap * hd..h * cap * hd + len * hd]
                .copy_from_slice(&self.v[h * old * hd..h * old * hd + len * hd]);
        }
        self.k = k;
        self.v = v;
        self.cap = cap;
        self.scratch = vec![0.0; cap];
    }

    /// Room for `extra` more positions; grows by doubling, amortized O(1).
    fn reserve(&mut self, extra: usize) -> CandleResult<()> {
        let need = self.len + extra;
        if need <= self.cap {
            return Ok(());
        }
        if need > self.max {
            ffai_core::candle::bail!(
                "self-attention cache: {need} positions exceed the model's {}",
                self.max
            );
        }
        let mut cap = self.cap;
        while cap < need {
            cap = (cap * 2).min(self.max);
        }
        self.regrow(cap);
        Ok(())
    }

    /// Append `n` positions. `k_rows` / `v_rows` are position-major
    /// `(n, heads*hd)` — the projections exactly as the linear layers emit
    /// them, so the only data movement is this scatter into kernel layout.
    fn append(&mut self, k_rows: &[f32], v_rows: &[f32], n: usize) -> CandleResult<()> {
        self.reserve(n)?;
        let (heads, hd, cap) = (self.heads, self.hd, self.cap);
        let n_state = heads * hd;
        for p in 0..n {
            let pos = self.len + p;
            let kr = &k_rows[p * n_state..(p + 1) * n_state];
            let vr = &v_rows[p * n_state..(p + 1) * n_state];
            for h in 0..heads {
                let kbase = h * hd * cap + pos;
                for (t, &kv) in kr[h * hd..(h + 1) * hd].iter().enumerate() {
                    self.k[kbase + t * cap] = kv;
                }
                let vdst = h * cap * hd + pos * hd;
                self.v[vdst..vdst + hd].copy_from_slice(&vr[h * hd..(h + 1) * hd]);
            }
        }
        self.len += n;
        Ok(())
    }

    /// Head `h` of K: a `[hd][cap]` block, `len` live columns per row.
    fn k_head(&self, h: usize) -> &[f32] {
        &self.k[h * self.hd * self.cap..(h + 1) * self.hd * self.cap]
    }

    /// Head `h` of V: the `len` live rows, contiguous.
    fn v_head(&self, h: usize) -> &[f32] {
        let base = h * self.cap * self.hd;
        &self.v[base..base + self.len * self.hd]
    }

    /// The used prefix, copied tight.
    fn prefix(&self) -> (Vec<f32>, Vec<f32>) {
        let (heads, hd, cap, len) = (self.heads, self.hd, self.cap, self.len);
        let mut k = Vec::with_capacity(heads * hd * len);
        for r in 0..heads * hd {
            k.extend_from_slice(&self.k[r * cap..r * cap + len]);
        }
        let mut v = Vec::with_capacity(heads * len * hd);
        for h in 0..heads {
            v.extend_from_slice(&self.v[h * cap * hd..h * cap * hd + len * hd]);
        }
        (k, v)
    }

    /// The used prefix as the `(1, heads, hd, len)` / `(1, heads, len, hd)`
    /// tensors the three-op path consumes — prefill and the few first tokens
    /// under the kernel's key floor. Same values the legacy
    /// `split_heads().transpose().contiguous()` produced.
    fn materialize(&self, device: &ffai_core::candle::Device) -> CandleResult<(Tensor, Tensor)> {
        let (k, v) = self.prefix();
        Ok((
            Tensor::from_vec(k, (1, self.heads, self.hd, self.len), device)?,
            Tensor::from_vec(v, (1, self.heads, self.len, self.hd), device)?,
        ))
    }

    fn snapshot(&self) -> FlatKvState {
        let (k, v) = self.prefix();
        FlatKvState {
            k,
            v,
            len: self.len,
        }
    }

    /// Put a snapshot back. Infallible: a snapshot came out of a cache that
    /// held it, so it fits the position limit by construction.
    fn restore(&mut self, s: &FlatKvState) {
        self.len = 0;
        if s.len > self.cap {
            let mut cap = self.cap.max(1);
            while cap < s.len {
                cap *= 2;
            }
            self.regrow(cap.min(self.max.max(s.len)));
        }
        let (heads, hd, cap, len) = (self.heads, self.hd, self.cap, s.len);
        for r in 0..heads * hd {
            self.k[r * cap..r * cap + len].copy_from_slice(&s.k[r * len..(r + 1) * len]);
        }
        for h in 0..heads {
            self.v[h * cap * hd..h * cap * hd + len * hd]
                .copy_from_slice(&s.v[h * len * hd..(h + 1) * len * hd]);
        }
        self.len = len;
    }
}

use crate::buf::with_f32;

struct Attention {
    query: QLinear,
    key: QLinear,
    value: QLinear,
    out: QLinear,
    n_head: usize,
    /// Cached keys/values. For cross-attention it is filled once per window
    /// and then reused; for self-attention it is the legacy (`FFAI_VEINS=off`)
    /// arm, superseded by `flat`.
    cache: Option<(Tensor, Tensor)>,
    /// Self-attention K/V in kernel layout (vein 1). `None` until the first
    /// token of the first window; cleared, not freed, between windows.
    flat: Option<FlatKv>,
    /// The model's position limit — the flat cache's ceiling. Zero for
    /// attention that never holds a growing cache (encoder, cross).
    max_positions: usize,
    /// Whisper's `head_dim^-0.25` applied to BOTH q and k, folded onto q
    /// alone as its square — a constant of the layer geometry, computed once
    /// here instead of a `powf` per call (vein 3).
    scale2: f64,
    /// Whether the fused kernel serves the cross-attention cache as built.
    /// Decided when the cache is built, not re-asked on every token (vein 3).
    kernel_serves: bool,
}

// MEASURED AND REVERTED: int8 GEMV across every decoder projection.
//
// This came from the last open D6 question — *are both arms doing the same
// work?* — and the answer was no. The reference ships **f16 weights** (its
// tiny.en file is 77.7 MB; f32 would be 156 MB) while we run f32, so it reads
// HALF the weight bytes we do on every generated token. On one identical
// 10.44 s clip:
//
//   stage    ours          whisper.cpp
//   mel      7.0 ms        4.4 ms       1.6x slower
//   encode   160 ms        117 ms       1.37x slower
//   decode   4.11 ms/tok   2.22 ms/tok  1.85x slower
//   sample   12 ms         19.9 ms      we are 1.66x FASTER
//
// Decoder linear traffic: f32 33.0 MB/token, their f16 16.5, int8 8.3. A 2.0x
// traffic ratio against a 1.85x measured decode gap — **the decode gap is the
// precision, not the code.**
//
// int8 on every single-row projection delivered exactly what that predicts:
// decoder 0.149 -> 0.125 s, total 0.316 -> 0.289 s, **19/21 paired rounds,
// z = +3.7**. It also explains why the MLP alone measured 1.005x (inside
// noise): the traffic saving only becomes resolvable when every projection
// contributes.
//
// **It fails the quality gate.** Corpus WER:
//
//   test-clean   7.77 % -> 8.39 %   FAIL (band is 7.959 %)
//   test-other  16.83 % -> 16.88 %  pass
//
// Unlike the vocabulary projection — where a quantization error either flips
// an argmax or vanishes — error here feeds the residual stream and the next
// layer's attention, so it COMPOUNDS. The argmax-flip instrument cannot see
// this class at all: it measures the final projection, which is downstream of
// the damage.
//
// Note the asymmetry, which is the second time this campaign has produced it
// and is still mechanically unexplained: the HARDER corpus is the LESS
// sensitive one (+0.05 pp vs +0.62 pp), matching the earlier flip-rate
// finding (1.07 % on test-other vs 1.77 % on test-clean). I predicted the
// opposite both times.
//
// The route to matching their traffic without their error profile is f16
// weights — the precision they actually use — not int8. `Precision` has no
// F16 variant today; that is the next experiment, not this one.
impl Attention {
    /// Decoder attention: projections fed ONE row per generated token take
    /// f16 weights, matching the reference's precision. `kv_single_row`
    /// distinguishes self-attention (all four project one token) from
    /// cross-attention (k/v project the 1500-row encoder output once, a real
    /// GEMM that belongs on candle's gemm).
    fn load_decode(
        n_state: usize,
        n_head: usize,
        vb: VarBuilder,
        p: Precision,
        kv_single_row: bool,
        max_positions: usize,
    ) -> CandleResult<Self> {
        let half =
            |name: &str, b: bool| QLinear::from_vb_gemv_f16(n_state, n_state, vb.pp(name), b, p);
        Ok(Self {
            query: half("q_proj", true)?,
            key: if kv_single_row {
                half("k_proj", false)?
            } else {
                linear_no_bias(n_state, n_state, vb.pp("k_proj"), p)?
            },
            value: if kv_single_row {
                half("v_proj", true)?
            } else {
                linear(n_state, n_state, vb.pp("v_proj"), p)?
            },
            out: half("out_proj", true)?,
            n_head,
            cache: None,
            flat: None,
            max_positions: if kv_single_row { max_positions } else { 0 },
            scale2: Self::scale2(n_state, n_head),
            kernel_serves: false,
        })
    }

    fn load(n_state: usize, n_head: usize, vb: VarBuilder, p: Precision) -> CandleResult<Self> {
        Ok(Self {
            query: linear(n_state, n_state, vb.pp("q_proj"), p)?,
            // Whisper's key projection has no bias — it would be redundant
            // under the softmax.
            key: linear_no_bias(n_state, n_state, vb.pp("k_proj"), p)?,
            value: linear(n_state, n_state, vb.pp("v_proj"), p)?,
            out: linear(n_state, n_state, vb.pp("out_proj"), p)?,
            n_head,
            cache: None,
            flat: None,
            max_positions: 0,
            scale2: Self::scale2(n_state, n_head),
            kernel_serves: false,
        })
    }

    /// Whisper folds `1/sqrt(head_dim)` into q and k as `^-0.25` each; the
    /// product of the two halves rides on q alone. Same expression the
    /// per-call `scale()` evaluated, so the bits are identical.
    fn scale2(n_state: usize, n_head: usize) -> f64 {
        let scale = ((n_state / n_head) as f64).powf(-0.25);
        scale * scale
    }

    /// (batch, seq, `n_state`) -> (batch, `n_head`, seq, `head_dim`)
    fn split_heads(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, seq, n_state) = x.dims3()?;
        x.reshape((b, seq, self.n_head, n_state / self.n_head))?
            .transpose(1, 2)
    }

    /// Self-attention over the cache, appending this step's keys/values.
    ///
    /// `offset` is the absolute position of the first query token, which the
    /// causal mask needs: query i sits at `offset + i` and may attend to any
    /// key at or before that. With one query token and a populated cache the
    /// mask is unnecessary — every cached key is in the past — which is why
    /// the steady-state path skips building it entirely.
    fn forward_self(&mut self, x: &Tensor, offset: usize) -> CandleResult<Tensor> {
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Arm choice is sticky for the window: whichever cache already holds
        // tokens continues, so a knob flipped mid-window cannot split one
        // history across two layouts.
        let flat_live = self.flat.as_ref().is_some_and(|f| f.len > 0);
        let use_flat = flat_live
            || (self.cache.is_none()
                && self.max_positions > 0
                && !super::knobs::VEINS_OFF.get()
                && k.dtype() == DType::F32
                && matches!(k.device(), ffai_core::candle::Device::Cpu)
                && k.dim(0)? == 1
                && k.dim(D::Minus1)? / self.n_head == super::flash_attn::HD);
        if use_flat {
            return self.forward_self_flat(&q, &k, &v, offset);
        }

        let (k, v) = match self.cache.take() {
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[&prev_k, &k], 1)?,
                Tensor::cat(&[&prev_v, &v], 1)?,
            ),
            None => (k, v),
        };
        {
            // Work census (vein 1): two cats, each copying the WHOLE cache.
            let c = super::profile::counters();
            c.kv_cache_ops.add(2);
            c.kv_bytes_copied
                .add_n((k.elem_count() + v.elem_count()) * k.dtype().size_in_bytes());
        }
        self.cache = Some((k.clone(), v.clone()));

        let q_len = x.dim(1)?;
        let mask = if q_len > 1 {
            Some(causal_mask(
                q_len,
                k.dim(1)?,
                offset,
                x.dtype(),
                x.device(),
            )?)
        } else {
            None
        };
        self.attend(&q, &k, &v, mask.as_ref())
    }

    /// Self-attention on the flat cache (vein 1).
    ///
    /// Steady state (one query, cache at or above the kernel's key floor):
    /// append in place, scale q, run the kernel per head straight off the
    /// cache, wrap the merged context in ONE tensor for the output
    /// projection. Per layer per token that is 1 candle op and 2·n_state
    /// floats of copying, against 16 ops and four whole-cache passes before.
    ///
    /// Prefill (several queries) and the first few tokens under the floor
    /// take the three-op path on a tight copy of the used prefix — the same
    /// operands the legacy arm built, so the same arithmetic.
    fn forward_self_flat(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        offset: usize,
    ) -> CandleResult<Tensor> {
        let (_, seq, n_state) = k.dims3()?;
        let heads = self.n_head;
        let hd = n_state / heads;
        let max = self.max_positions;
        let flat = self.flat.get_or_insert_with(|| FlatKv::new(heads, hd, max));
        with_f32(k, |kr| with_f32(v, |vr| flat.append(kr, vr, seq)))???;
        let c = super::profile::counters();
        c.kv_bytes_copied.add_n(2 * seq * n_state * 4);
        let keys = flat.len;

        if seq == 1
            && keys >= super::flash_attn::decode_floor()
            && super::flash_attn::decode_kernel_available()
        {
            // q scaled exactly as candle's `Tensor * f64` does it on f32:
            // `v * (s as f32) + 0.0`.
            // f64 -> f32 on purpose: candle's `affine` narrows the same way.
            #[allow(clippy::cast_possible_truncation)]
            let s = self.scale2 as f32;
            let qs = with_f32(q, |qv| {
                qv.iter().map(|&x| x * s + 0.0).collect::<Vec<f32>>()
            })?;
            c.memset_bytes.add_n(n_state * 4);
            let mut out = vec![0f32; n_state];
            let mut scratch = std::mem::take(&mut flat.scratch);
            for h in 0..heads {
                let served = super::flash_attn::decode_head(
                    &qs[h * hd..(h + 1) * hd],
                    flat.k_head(h),
                    flat.cap,
                    flat.v_head(h),
                    &mut out[h * hd..(h + 1) * hd],
                    keys,
                    &mut scratch,
                );
                debug_assert!(served, "decode_kernel_available() was checked above");
            }
            flat.scratch = scratch;
            c.kv_cache_ops.add(1);
            let ctx = Tensor::from_vec(out, (1, 1, n_state), q.device())?;
            return self.out.forward(&ctx);
        }

        let (kt, vt) = flat.materialize(q.device())?;
        c.kv_cache_ops.add(2);
        c.kv_bytes_copied.add_n(2 * keys * n_state * 4);
        let qp = (self.split_heads(q)? * self.scale2)?.contiguous()?;
        c.kv_cache_ops.add(4);
        let mask = if seq > 1 {
            Some(causal_mask(seq, keys, offset, q.dtype(), q.device())?)
        } else {
            None
        };
        self.attend_prepared(&qp, &kt, &vt, mask.as_ref())
    }

    /// Cross-attention: keys/values come from the encoder and are constant for
    /// the whole window, so they are projected once and reused.
    ///
    /// The cache holds them **already split into heads, transposed and
    /// scaled** — the finished operands, not the raw projections. Caching the
    /// raw projections still leaves a reshape + transpose + scale + copy over
    /// 1500 encoder frames on every token, which measured as 63 % of decoder
    /// time once the vocabulary projection was fixed.
    fn forward_cross(&mut self, x: &Tensor, audio: &Tensor) -> CandleResult<Tensor> {
        let p = super::profile::profile();
        let q = super::profile::timed(&p.xa_qproj, || self.query.forward(x))?;
        let (k, v) = if let Some((k, v)) = &self.cache {
            (k.clone(), v.clone())
        } else {
            let k = self.key.forward(audio)?;
            let v = self.value.forward(audio)?;
            // Prepared once per window, reused for every token.
            // Whisper folds 1/sqrt(d) as ^-0.25 into BOTH q and k. That is
            // symmetric but not required: (q*s)·(k*s) = q·k·s^2, so the
            // whole factor can ride on q alone. Here q is ONE row per
            // token and k is 1500x384 per layer, so scaling k meant a
            // 2.3 MB multiply-and-allocate per layer at cache build —
            // paid once per clip, and these corpus clips average ~6 s, so
            // a fixed cost lands heavily on the throughput figure.
            let k = self.split_heads(&k)?.transpose(2, 3)?.contiguous()?;
            let v = self.split_heads(&v)?.contiguous()?;
            // The cross-attention cache is read IN FULL on every generated
            // token — 4.6 MB per layer, 18.4 MB per token across the stack
            // — which the three-pass breakdown showed is what
            // cross-attention actually costs.
            //
            // Storing it at f16 halves that. This was tried once and
            // REVERTED (§6.16): the matmuls got faster and candle's f16
            // softmax, 82 % slower, gave it all back. With `fast_softmax`
            // that blocker is gone and the FULL CHAIN now wins 1.18x
            // (31/41 paired rounds, z = +3.3). The link was never the
            // problem; the chain was.
            // ...and with the fused f32 kernel that argument changes
            // again. f16 bought its 1.18x by halving traffic through a
            // three-op path; the fused kernel reads K and V exactly once
            // in f32 and beats the f16 three-op path outright:
            //
            //   default (adaptive)      cross-attn 0.107 s   total 0.440
            //   forced f16 (no kernel)  cross-attn 0.083 s   total 0.368
            //   forced f32 (kernel)     cross-attn 0.075 s   total 0.360
            //
            // The adaptive path was ALSO losing to both of its own
            // options — the known run-to-run flip (margin ~1.0-1.17x
            // against a 1.10 threshold) meant it kept re-deciding. When
            // the fused kernel can serve this shape the choice is no
            // longer a balance to calibrate, so stop calibrating it.
            // Decided once per cache build (vein 3): the kernel's answer is
            // a function of the cache's shape, which does not change until
            // the next window rebuilds it.
            self.kernel_serves = super::flash_attn::serves(&k, &v);
            let kv_dtype = if self.kernel_serves {
                // f16 cache: the fused kernel widens it with F16C as it
                // streams and still accumulates in f32, so this halves the
                // 18.4 MB/token read without halving the arithmetic. That
                // is the distinction §6.16's failed f16 K/V chain missed —
                // it did f16 MATH, and the softmax between the matmuls ran
                // 82 % slower for it.
                // f16 cache, widened by F16C as it streams so the
                // accumulation still happens in f32. Halves the largest
                // remaining per-token read in decode, 18.4 -> 9.2 MB.
                //
                // NOT the f16 K/V chain that failed in 6.16 — that did f16
                // MATH, and the softmax between the matmuls ran 82 %
                // slower for it. Here only the bytes are half.
                //
                // Measured: cross-attn 0.064 -> 0.055 s (1.16x on the
                // stage); pipeline 13/21, z = +1.1, ratio 1.003x —
                // INCONCLUSIVE at the total level, because cross-attention
                // is ~18 % of the pipeline so 1.16x there is ~2.5 %, under
                // this harness's noise floor. Kept on the strength of the
                // stage measurement and the halved cache memory; the
                // pipeline claim is NOT made.
                if kv_f16_disabled() {
                    DType::F32
                } else {
                    DType::F16
                }
            } else {
                super::adaptive::attention_kv_dtype(k.dim(1)?, k.dim(3)?, k.dim(2)?, k.device())
            };
            let (k, v) = if kv_dtype == DType::F16 {
                (k.to_dtype(DType::F16)?, v.to_dtype(DType::F16)?)
            } else {
                (k, v)
            };
            self.cache = Some((k.clone(), v.clone()));
            (k, v)
        };
        let q = super::profile::timed(&p.xa_prep, || {
            // scale^2 because k no longer carries its half — see the cache
            // build above. Match the cache's dtype; the query is one row, so
            // the conversion is free against the megabytes it unlocks.
            // NOT k.dtype(): with an f16 cache the kernel wants q in f32 so
            // the accumulation stays full precision.
            let q = (self.split_heads(&q)? * self.scale2)?.contiguous()?;
            if self.kernel_serves {
                Ok(q)
            } else {
                q.to_dtype(k.dtype())
            }
        })?;

        // Inlined from attend_prepared so each step is separately timed in
        // context — the microbenchmark of these same ops disagreed with the
        // stage total by 79 %, and when they disagree the stage wins.
        // Fused streaming kernel for the decode shape (one query row, many
        // keys) — 2.37x the three ops below, 41/41, z=+6.4. Declines when the
        // KV cache is f16, so the fallback stays live.
        if let Some(ctx) =
            super::profile::timed(&p.xa_qk, || super::flash_attn::attend(&q, &k, &v, false))?
        {
            let merged = super::profile::timed(&p.xa_merge, || {
                ctx.transpose(1, 2)?.flatten_from(2)?.to_dtype(x.dtype())
            })?;
            return super::profile::timed(&p.xa_out, || self.out.forward(&merged));
        }
        // A multi-token prompt prefill (previous-window conditioning) has
        // 1 < qlen < keys, a shape the fused kernel declines — and q
        // deliberately stays f32 while the cache may be f16, so the fallback
        // matmul would face mixed dtypes. Widen K/V for this one call: it
        // runs once per window, and every per-token step after it still
        // reads the f16 cache through the kernel.
        let (k, v) = if q.dtype() == k.dtype() {
            (k, v)
        } else {
            (k.to_dtype(q.dtype())?, v.to_dtype(q.dtype())?)
        };
        let qk = super::profile::timed(&p.xa_qk, || q.matmul(&k))?;
        let w = super::profile::timed(&p.xa_softmax, || fast_softmax(&qk))?;
        let ctx = super::profile::timed(&p.xa_wv, || w.matmul(&v))?;
        let merged = super::profile::timed(&p.xa_merge, || {
            ctx.transpose(1, 2)?.flatten_from(2)?.to_dtype(x.dtype())
        })?;
        super::profile::timed(&p.xa_out, || self.out.forward(&merged))
    }

    /// Split heads, scale, and attend. Used by self-attention, whose keys and
    /// values change every step.
    pub(crate) fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Whisper folds 1/sqrt(d) as ^-0.25 into BOTH operands. Symmetric, but
        // (q*s)(k*s) = q*k*s^2, so one side can carry all of it. Both sides
        // here are 1500x384, and the k-side multiply is a 2.3 MB
        // allocate-and-write per layer measured at 0.72 ms -- 5.8 ms across
        // the encoder for arithmetic q can absorb for free.
        let q = (self.split_heads(q)? * self.scale2)?.contiguous()?;
        let k = self.split_heads(k)?.transpose(2, 3)?.contiguous()?;
        let v = self.split_heads(v)?.contiguous()?;
        {
            // Work census (vein 1): reshape/transpose/scale/contiguous on q
            // (4), k (4), v (3); the two contiguous() calls copy K and V whole.
            let c = super::profile::counters();
            c.kv_cache_ops.add(11);
            c.kv_bytes_copied
                .add_n((k.elem_count() + v.elem_count()) * k.dtype().size_in_bytes());
        }
        self.attend_prepared(&q, &k, &v, mask)
    }

    /// Attend with operands already split into heads, scaled and contiguous.
    fn attend_prepared(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Fused AVX2 kernel where the shape allows it — 3.75x the three-op
        // path on the encoder shape (super::flash_attn). Declines and falls
        // through for masked, short-query, non-f32 or non-CPU cases.
        if let Some(wv) = super::flash_attn::attend(q, k, v, mask.is_some())? {
            super::profile::counters().kv_cache_ops.add(3);
            return wv
                .transpose(1, 2)?
                .flatten_from(2)
                .and_then(|wv| self.out.forward(&wv));
        }
        super::profile::counters().kv_cache_ops.add(6);
        let mut qk = q.matmul(k)?;
        if let Some(mask) = mask {
            qk = qk.broadcast_add(mask)?;
        }
        let wv = candle_nn::ops::softmax_last_dim(&qk)?.matmul(v)?;
        wv.transpose(1, 2)?
            .flatten_from(2)
            .and_then(|wv| self.out.forward(&wv))
    }

    fn reset(&mut self) {
        self.cache = None;
        // Cleared, not dropped: the next window reuses the allocation.
        if let Some(f) = &mut self.flat {
            f.clear();
        }
    }
}

// MEASURED AND REVERTED: query-tiled ("flash") attention.
//
// The encoder profile (mission plan §6.6) showed the stage stops scaling past
// ~4 threads on 24 cores, and the score matrix is 54 MB per layer — the
// classic memory-bound signature, whose textbook fix is to tile attention so
// the scores never materialize. Because softmax runs over the KEY axis,
// tiling queries only is exact and needs no online rescaling, so this was
// cheap to build and safe to try.
//
// It is monotonically SLOWER. Encoder attention, tiny.en, 1500 positions:
//
//   untiled 0.161 s | tile=512 0.207 | 256 0.230 | 128 0.283 | 64 0.353 | 32 0.441
//
// Smaller tiles are worse, and no size beats untiled — so there is no sweet
// spot to tune toward. Splitting one large GEMM into several smaller ones
// costs more in matmul efficiency, rayon scheduling, and the final concat
// than it recovers in memory traffic. The cache reasoning that motivated it
// was also wrong: a 256-query block spans all 6 heads (~9 MB), never L2
// resident as assumed.
//
// Reverted rather than kept behind a flag: it is dead weight with a measured
// negative. The memory-bound diagnosis may still be correct — but tiling at
// this scale is not the lever that exploits it.

/// Causal mask for `q_len` queries starting at absolute position `offset`,
/// against `k_len` keys at absolute positions `0..k_len`.
fn causal_mask(
    q_len: usize,
    k_len: usize,
    offset: usize,
    dtype: DType,
    device: &ffai_core::candle::Device,
) -> CandleResult<Tensor> {
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            (0..k_len).map(move |j| {
                if j > offset + i {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (q_len, k_len), device)?.to_dtype(dtype)
}

/// The projection from hidden state to vocabulary logits.
///
/// Deliberately NOT a `candle_nn::Linear` in the f32 case: `Linear::forward`
/// calls `weight.t()` internally, and matmul against that non-contiguous view
/// materializes the whole 51864 × `d_model` matrix *per token* — measured at
/// 91.5 % of decoder time before it was fixed (mission plan §6.3). The f32
/// arm therefore holds a pre-transposed contiguous copy; the quantized arm
/// needs no transpose at all, since `QMatMul` contracts the last dimension.
enum VocabProj {
    /// Pre-transposed and stored in **half precision**, when
    /// [`super::adaptive`] measures that as faster on this machine.
    ///
    /// This matmul streams the entire ~20 M-parameter embedding matrix on
    /// every generated token — 80 MB at f32 — which makes it the only truly
    /// bandwidth-bound op in the decoder, and the one place half precision
    /// paid: 0.231 s -> 0.162 s over 50 tokens. Applying it across the whole
    /// decoder instead made cross-attention 1.13x *slower*. Precision is a
    /// per-op property, and which way it falls depends on the machine — hence
    /// the calibration rather than a constant.
    Half {
        w: Tensor,
        pad: usize,
    },
    Full {
        w: Tensor,
        pad: usize,
    },
    Quant(QMatMul),
    /// Hand-written AVX2 int8 GEMV — 4.34x the f16+pad arm (41/41, z=+6.4).
    /// See [`super::vocab_int8`].
    Int8(super::vocab_int8::Int8Vocab),
}

impl VocabProj {
    /// Logits into the caller's buffer. The int8 arm writes them there
    /// directly; the tensor arms compute as before and copy out once.
    fn forward_into(&self, x: &Tensor, out: &mut Vec<f32>) -> CandleResult<()> {
        if let Self::Int8(q) = self
            && x.dtype() == DType::F32
            && x.is_contiguous()
        {
            return with_f32(x, |xs| q.forward_into(xs, out))?;
        }
        let logits = self.forward(x)?.to_dtype(DType::F32)?;
        with_f32(&logits.contiguous()?, |l| {
            out.clear();
            out.extend_from_slice(l);
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            // The activation is a single row; converting it costs nothing
            // against the 40 MB weight read it enables.
            Self::Half { w, pad } => {
                matmul_padded(&x.to_dtype(DType::F16)?, w, *pad)?.to_dtype(DType::F32)
            }
            Self::Full { w, pad } => matmul_padded(x, w, *pad),
            Self::Quant(q) => q.forward(x),
            Self::Int8(q) => q.forward(x),
        }
    }
}

/// Matmul, padding a single output row up to whatever [`super::adaptive`]
/// measured as fastest for this shape.
///
/// candle routes `m == 1` to a matrix-VECTOR path that is dramatically worse
/// than its matrix-matrix one on wide outputs. Measured on the vocabulary
/// projection (1x384 @ 384x51864, f16), *absolute* wall time:
///
/// | rows | ms | GB/s |
/// |---|---:|---:|
/// | 1 | 1.94 | 20.5 |
/// | 2 | 0.95 | 41.7 |
/// | **4** | **0.78** | **51.1** |
/// | 8 | 1.11 | 36.0 |
///
/// Computing four rows costs 2.5x *less* than computing one, because m >= 2
/// reaches the tuned GEMM micro-kernel. Thread count is irrelevant (20 GB/s
/// at both 1 and 24 threads), so it is a code-path cliff, not parallelism.
/// The padded rows are duplicates — no extra weight traffic, the 40 MB read
/// is shared. The cliff reverses on square shapes, which is why the row count
/// is measured per shape instead of fixed.
/// `rows` is resolved ONCE at load ([`VocabProj`] stores it), never per call.
///
/// An earlier version called the calibration helper here, which meant a
/// `std::env::var` (allocating, and locking the process environment) plus a
/// mutex-guarded `HashMap` lookup **on every generated token**. That cost more
/// than the optimization saved and showed up as the vocabulary projection
/// regressing from 0.088 s to 0.263 s. A per-call lookup in a per-token path
/// is a bug even when the lookup is "cheap".
pub fn matmul_padded(x: &Tensor, w: &Tensor, rows: usize) -> CandleResult<Tensor> {
    if rows <= 1 || x.dim(0)? != 1 {
        return x.matmul(w);
    }
    Tensor::cat(&vec![x; rows], 0)?.matmul(w)?.narrow(0, 0, 1)
}

/// Bidirectional self-attention for the audio encoder: no causal mask and no
/// K/V cache, because each call is one fresh pass over a whole window.
pub struct EncoderAttention {
    inner: Attention,
    /// K's projection weight, pre-scaled, applied TRANSPOSED. Two bricks in
    /// one tensor (priced in `examples/enc_prep_ceiling.rs`):
    ///
    /// 1. **The layout.** `Linear` stores its weight as `(out, in)` and
    ///    computes `x @ Wᵀ`, giving k as `(seq, d)`. The fused kernel reads
    ///    `(heads, HD, seq)`, so every layer paid `transpose(2,3).contiguous()`
    ///    — a strided gather over 2.3 MB, measured at 1.33 ms/layer. But
    ///    `kᵀ = W @ xᵀ` directly, `x.t()` is a view candle's matmul accepts
    ///    without materializing, and the result reshapes into the kernel's
    ///    layout for free: **0.951 ms against 2.265 ms for projection +
    ///    gather.** The transposed projection is cheaper than the plain one.
    ///
    ///    `OPEN.md` §2 refuted fusing ALL of q/k/v into one transposed
    ///    projection — correctly, because the kernel wants q as `(seq, HD)`,
    ///    where `q[i*64 + t]` walks 4 contiguous cache lines. That refutation
    ///    does not reach K, which is the one tensor the kernel genuinely wants
    ///    transposed, and K alone was never measured.
    ///
    /// 2. **The scale.** Attention's `head_dim^-0.5` depends on nothing but
    ///    the model geometry, yet rode a per-call multiply over 2.3 MB of q.
    ///    Since k is only ever consumed by `q@k`, scaling k's WEIGHT at load
    ///    is equivalent and free — the first question
    ///    `codec-eliminate-redundancy` says to ask of a hot loop.
    key_wt_scaled: Tensor,
}

impl EncoderAttention {
    pub fn load(n_state: usize, n_head: usize, vb: VarBuilder, p: Precision) -> CandleResult<Self> {
        let inner = Attention::load(n_state, n_head, vb.clone(), p)?;
        // Whisper's k projection carries no bias (see `load_decode`), so the
        // fold is a plain scalar multiply with nothing else to carry.
        let head_dim = n_state / n_head;
        let scale = (head_dim as f64).powf(-0.25);
        let key_w = vb.pp("k_proj").get((n_state, n_state), "weight")?;
        let key_wt_scaled = (key_w * (scale * scale))?.contiguous()?;
        Ok(Self {
            inner,
            key_wt_scaled,
        })
    }

    /// Inlined from `Attention::attend` so each step is timed IN CONTEXT.
    ///
    /// The stage carried a 16.6 ms residue — 20 % of itself — that standalone
    /// probes could only guess at. Cross-attention already had this treatment;
    /// the encoder did not, which is how the residue stayed unnamed while I
    /// twice called the stage exhausted.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // The fused kernels below are written for a single window: the
        // transposed-K reshape is `(1, heads, hd, seq)` and the merge writes
        // `(1, seq, heads*hd)`. Rather than let a batched call silently take
        // the first item's answer — which is exactly the bug `conv1d_gemm`
        // carried — run each item through the identical path and stack.
        //
        // Nothing is lost by looping: the batching ceiling was measured at
        // 1.00-1.04x (examples/batch_ceiling.rs), so there is no shared work
        // between windows here worth restructuring the kernels to capture.
        let batch = x.dim(0)?;
        if batch > 1 {
            let mut outs = Vec::with_capacity(batch);
            for b in 0..batch {
                outs.push(self.forward(&x.narrow(0, b, 1)?.contiguous()?)?);
            }
            return Tensor::cat(&outs, 0);
        }

        let p = super::profile::profile();
        let (seq, d) = (x.dim(1)?, x.dim(2)?);
        let heads = self.inner.n_head;
        let hd = d / heads;
        let kt_on = !super::knobs::ENC_KT_DISABLED.get();
        let (q, kt, v) = super::profile::timed(&p.ea_proj, || -> CandleResult<_> {
            Ok((
                self.inner.query.forward(x)?,
                // k, already transposed and already scaled — see the field's
                // documentation. `x.t()` is a view; candle's matmul takes it
                // strided rather than materializing it.
                if kt_on {
                    self.key_wt_scaled.matmul(&x.reshape((seq, d))?.t()?)?
                } else {
                    self.inner.key.forward(x)?
                },
                self.inner.value.forward(x)?,
            ))
        })?;

        // Vein 14: with K already in kernel layout, q and v can be read by
        // the merged kernel exactly as the projections emit them — one row
        // per position, heads side by side — so the two `split_heads()
        // .contiguous()` copies (2 x 2.3 MB per layer on tiny) never happen.
        if kt_on && !super::knobs::VEINS_OFF.get() {
            let kt = super::profile::timed(&p.ea_prep, || kt.reshape((1, heads, hd, seq)))?;
            if let Some(merged) = super::profile::timed(&p.ea_kernel, || {
                super::flash_attn::attend_merged_unsplit(&q, &kt, &v)
            })? {
                return super::profile::timed(&p.ea_merge, || self.inner.out.forward(&merged));
            }
        }
        let (q, k, v) = super::profile::timed(&p.ea_prep, || -> CandleResult<_> {
            super::profile::counters()
                .enc_copy_bytes
                .add_n(2 * seq * d * 4);
            Ok((
                if kt_on {
                    self.inner.split_heads(&q)?.contiguous()?
                } else {
                    // Whole scale on q — see `Attention::attend`.
                    (self.inner.split_heads(&q)? * self.inner.scale2)?.contiguous()?
                },
                if kt_on {
                    // A contiguous reshape into exactly the layout the kernel
                    // reads: element (h, j, s) is k[s, h*hd + j], which is what
                    // `split_heads(k).transpose(2,3)` produced by gathering.
                    kt.reshape((1, heads, hd, seq))?
                } else {
                    self.inner.split_heads(&kt)?.transpose(2, 3)?.contiguous()?
                },
                self.inner.split_heads(&v)?.contiguous()?,
            ))
        })?;

        // Merged path: the kernel writes (1, seq, heads*HD) directly, so the
        // strided transpose in `ea_merge` never happens.
        if let Some(merged) = super::profile::timed(&p.ea_kernel, || {
            super::flash_attn::attend_merged(&q, &k, &v)
        })? {
            return super::profile::timed(&p.ea_merge, || self.inner.out.forward(&merged));
        }
        let wv = super::profile::timed(&p.ea_kernel, || -> CandleResult<Tensor> {
            if let Some(wv) = super::flash_attn::attend(&q, &k, &v, false)? {
                return Ok(wv);
            }
            let qk = q.matmul(&k)?;
            candle_nn::ops::softmax_last_dim(&qk)?.matmul(&v)
        })?;

        super::profile::timed(&p.ea_merge, || {
            wv.transpose(1, 2)?
                .flatten_from(2)
                .and_then(|wv| self.inner.out.forward(&wv))
        })
    }
}

struct Block {
    attn: Attention,
    attn_ln: LayerNorm,
    cross_attn: Attention,
    cross_attn_ln: LayerNorm,
    mlp_ln: LayerNorm,
    mlp1: QLinear,
    mlp2: QLinear,
}

impl Block {
    fn load(
        n_state: usize,
        n_head: usize,
        vb: VarBuilder,
        p: Precision,
        max_positions: usize,
    ) -> CandleResult<Self> {
        Ok(Self {
            attn: Attention::load_decode(
                n_state,
                n_head,
                vb.pp("self_attn"),
                p,
                true,
                max_positions,
            )?,
            attn_ln: layer_norm(n_state, vb.pp("self_attn_layer_norm"))?,
            cross_attn: Attention::load_decode(
                n_state,
                n_head,
                vb.pp("encoder_attn"),
                p,
                false,
                0,
            )?,
            cross_attn_ln: layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?,
            mlp_ln: layer_norm(n_state, vb.pp("final_layer_norm"))?,
            // MEASURED AND REVERTED: int8 GEMV on the decoder MLP.
            //
            // The shape is right — 18.9 MB of weights per token at 16.9 GB/s,
            // the same bandwidth-bound single-row case where the vocabulary
            // projection gained 4.34x — and the stage did speed up, 0.046 ->
            // 0.037 s (1.36x per call). It bought nothing end to end:
            // **8/15 paired rounds, z = +0.3, ratio 1.005x**, transcription
            // time only, load excluded.
            //
            // Arithmetic explains it. The MLP is ~20 % of decode and decode is
            // ~50 % of the pipeline, so 1.36x on it is worth ~2.6 % overall —
            // below what any harness here can resolve. Against that: it
            // quantizes 4.7 M weights at load, keeps both int8 and f32 copies
            // resident, and unlike the vocabulary projection its error
            // COMPOUNDS, feeding the residual stream and the next layer's
            // attention rather than either flipping an argmax or vanishing.
            //
            // Reverted because the delta sat inside the noise — NOT because it
            // measured worse. `QLinear::Int8` and `FFAI_MLP_INT8` stay for the
            // re-test if the surrounding stages ever shrink enough to make
            // 2.6 % resolvable.
            mlp1: if mlp_int8_enabled() {
                QLinear::from_vb_gemv(n_state, n_state * 4, vb.pp("fc1"), true, p)?
            } else {
                QLinear::from_vb_gemv_f16(n_state, n_state * 4, vb.pp("fc1"), true, p)?
            },
            mlp2: if mlp_int8_enabled() {
                QLinear::from_vb_gemv(n_state * 4, n_state, vb.pp("fc2"), true, p)?
            } else {
                QLinear::from_vb_gemv_f16(n_state * 4, n_state, vb.pp("fc2"), true, p)?
            },
        })
    }

    fn forward(&mut self, x: &Tensor, audio: &Tensor, offset: usize) -> CandleResult<Tensor> {
        let p = super::profile::profile();
        let attn_ln = self.attn_ln.forward(x)?;
        let self_out = super::profile::timed(&p.dec_self_attn, || {
            self.attn.forward_self(&attn_ln, offset)
        })?;
        let x = (x + self_out)?;

        let cross_ln = self.cross_attn_ln.forward(&x)?;
        let cross_out = super::profile::timed(&p.dec_cross_attn, || {
            self.cross_attn.forward_cross(&cross_ln, audio)
        })?;
        let x = (&x + cross_out)?;

        let mlp = super::profile::timed(&p.dec_mlp, || {
            self.mlp2
                // `gelu_tanh`, NOT `gelu_erf` — candle's `.gelu()` is the tanh
                // approximation and the two differ by ~1e-3, which is above
                // what Whisper's oracle tolerates.
                .forward(&ffai_core::fastops::gelu_tanh(
                    &self.mlp1.forward(&self.mlp_ln.forward(&x)?)?,
                )?)
        })?;
        x + mlp
    }

    fn reset(&mut self) {
        self.attn.reset();
        self.cross_attn.reset();
    }
}

/// A saved decoder cache, for restoring one beam's history.
///
/// Opaque by design: the shape of the caches is the decoder's business, and
/// beam search only needs to save one and put it back.
#[derive(Clone)]
pub struct DecoderState {
    /// Per block: legacy self-attention cache, cross-attention cache, and
    /// the flat self-attention cache's used prefix.
    blocks: Vec<(
        Option<(Tensor, Tensor)>,
        Option<(Tensor, Tensor)>,
        Option<FlatKvState>,
    )>,
    offset: usize,
}

impl DecoderState {
    /// A state holding no cached history — equivalent to a freshly
    /// [`TextDecoder::reset`] decoder. Restoring it into a decoder with more
    /// blocks than this holds leaves the extra blocks untouched, so it is
    /// only meaningful as a placeholder (tests, an unused beam slot).
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            blocks: Vec::new(),
            offset: 0,
        }
    }
}

/// Whisper's text decoder, decoding one token per step.
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<Block>,
    ln: LayerNorm,
    /// The output projection over the tied token-embedding matrix.
    vocab_proj: VocabProj,
    n_state: usize,
    /// Absolute position of the next token to be fed.
    offset: usize,
}

impl TextDecoder {
    /// The projection used when the int8 GEMV declines this shape or machine.
    fn fallback_proj(
        embeddings: &Tensor,
        cfg: &candle_transformers::models::whisper::Config,
        n_state: usize,
        precision: Precision,
    ) -> CandleResult<VocabProj> {
        Ok(match precision.ggml() {
            Some(dtype) if n_state.is_multiple_of(dtype.block_size()) => {
                // ~4x smaller than the f32 transposed copy it replaces, and no
                // transpose is needed at all.
                VocabProj::Quant(QMatMul::from_qtensor(QTensor::quantize(
                    embeddings, dtype,
                )?)?)
            }
            // Pay the transpose once, here, instead of once per generated token.
            _ => {
                let w = embeddings.t()?.contiguous()?;
                // Both decisions measured once, here, and then stored.
                let chosen = super::adaptive::matmul_dtype(1, n_state, cfg.vocab_size, w.device());
                if let (DType::F16, Ok(half)) = (chosen, w.to_dtype(DType::F16)) {
                    let pad = super::adaptive::matmul_pad_rows(
                        n_state,
                        cfg.vocab_size,
                        DType::F16,
                        half.device(),
                    );
                    VocabProj::Half { w: half, pad }
                } else {
                    let pad = super::adaptive::matmul_pad_rows(
                        n_state,
                        cfg.vocab_size,
                        DType::F32,
                        w.device(),
                    );
                    VocabProj::Full { w, pad }
                }
            }
        })
    }

    pub fn load(
        vb: VarBuilder,
        cfg: &candle_transformers::models::whisper::Config,
        precision: Precision,
    ) -> CandleResult<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let token_embedding = Embedding::new(
            vb.pp("embed_tokens")
                .get((cfg.vocab_size, n_state), "weight")?,
            n_state,
        );
        let positional_embedding = vb.get(
            (cfg.max_target_positions, n_state),
            "embed_positions.weight",
        )?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                Block::load(
                    n_state,
                    n_head,
                    vb.pp(format!("layers.{i}")),
                    precision,
                    cfg.max_target_positions,
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let ln = layer_norm(n_state, vb.pp("layer_norm"))?;
        let embeddings = token_embedding.embeddings();
        // The int8 GEMV reads the embedding matrix in its stored (vocab,
        // n_state) layout, so unlike the f16 arm it needs no transposed copy.
        let vocab_proj = match super::vocab_int8::Int8Vocab::new(embeddings)? {
            Some(q) => VocabProj::Int8(q),
            None => Self::fallback_proj(embeddings, cfg, n_state, precision)?,
        };
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            vocab_proj,
            n_state,
            offset: 0,
        })
    }

    /// Start a new 30 s window: drop every cache and rewind positions.
    ///
    /// Must be called between windows. Carrying a self-attention cache across
    /// windows would let one window's tokens attend into the next; carrying
    /// the cross-attention cache would decode the *previous* window's audio.
    pub fn reset(&mut self) {
        for block in &mut self.blocks {
            block.reset();
        }
        self.offset = 0;
    }

    /// Snapshot every cache plus the position counter.
    ///
    /// This is what makes beam search affordable. Each hypothesis needs its
    /// own self-attention K/V history, and the naive alternative — reset and
    /// re-feed each beam's whole prefix every step — is O(steps²·beams),
    /// roughly a hundredfold more decoder work at beam 5.
    ///
    /// The tensor clones are SHALLOW: a candle `Tensor` is a handle onto
    /// refcounted storage, and those caches are only ever *replaced*, never
    /// written in place, so snapshots share storage with the live cache. The
    /// cross-attention cache is identical across beams — same audio — so
    /// sharing it costs nothing. The flat self-attention cache IS written in
    /// place, so its used prefix is copied: 2·len·n_state floats per layer,
    /// a few percent of the weight traffic one decode step already pays.
    #[must_use]
    pub fn save(&self) -> DecoderState {
        DecoderState {
            blocks: self
                .blocks
                .iter()
                .map(|b| {
                    (
                        b.attn.cache.clone(),
                        b.cross_attn.cache.clone(),
                        b.attn
                            .flat
                            .as_ref()
                            .filter(|f| f.len > 0)
                            .map(FlatKv::snapshot),
                    )
                })
                .collect(),
            offset: self.offset,
        }
    }

    /// Restore a snapshot taken by [`Self::save`].
    pub fn restore(&mut self, state: &DecoderState) {
        for (block, (attn, cross, flat)) in self.blocks.iter_mut().zip(state.blocks.iter()) {
            block.attn.cache = attn.clone();
            block.cross_attn.cache = cross.clone();
            match (flat, &mut block.attn.flat) {
                (Some(s), Some(f)) => f.restore(s),
                (Some(s), None) => {
                    let heads = block.attn.n_head;
                    let hd = s.k.len() / (heads * s.len).max(1);
                    let mut f = FlatKv::new(heads, hd, block.attn.max_positions.max(s.len));
                    f.restore(s);
                    block.attn.flat = Some(f);
                }
                (None, Some(f)) => f.clear(),
                (None, None) => {}
            }
        }
        self.offset = state.offset;
    }

    /// [`Self::forward`], with the final position's logits written into
    /// `out` (resized to the vocabulary) instead of returned as a tensor.
    ///
    /// The decode loop reads logits as a `Vec<f32>`; `forward` + `to_vec1`
    /// built a 51 865-float tensor only to copy it out again on every token.
    /// With the int8 projection the kernel now writes into `out` itself
    /// (vein 2) — same values, one less full-vocabulary pass and one less
    /// 207 KB allocation per token.
    pub fn forward_into(
        &mut self,
        tokens: &[u32],
        audio: &Tensor,
        out: &mut Vec<f32>,
    ) -> CandleResult<()> {
        let p = super::profile::profile();
        let x = self.hidden(tokens, audio)?;
        let seq_len = tokens.len();
        super::profile::timed(&p.dec_final, || {
            let last = x.i((.., seq_len - 1.., ..))?.reshape((1, self.n_state))?;
            self.vocab_proj.forward_into(&last, out)
        })
    }

    /// The transformer stack: embeddings through the final layer norm,
    /// `(1, seq, n_state)`.
    fn hidden(&mut self, tokens: &[u32], audio: &Tensor) -> CandleResult<Tensor> {
        let p = super::profile::profile();
        let device = audio.device();
        let seq_len = tokens.len();
        let mut x = super::profile::timed(&p.dec_embed, || -> CandleResult<Tensor> {
            let ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
            let embedded = self.token_embedding.forward(&ids)?;
            // The offset is what makes single-token steps correct: token n
            // must receive position n's embedding, not position 0's.
            let positions = self.positional_embedding.narrow(0, self.offset, seq_len)?;
            embedded.broadcast_add(&positions)
        })?;

        for block in &mut self.blocks {
            x = block.forward(&x, audio, self.offset)?;
        }
        self.offset += seq_len;
        self.ln.forward(&x)
    }

    /// Feed `tokens` (the whole prompt on the first call, one token after) and
    /// return the logits for the final position only.
    pub fn forward(&mut self, tokens: &[u32], audio: &Tensor) -> CandleResult<Tensor> {
        let p = super::profile::profile();
        let x = self.hidden(tokens, audio)?;
        let seq_len = tokens.len();
        super::profile::timed(&p.dec_final, || {
            // Only the last position's logits matter; projecting the rest to a
            // ~51k vocabulary would be the most expensive wasted op here.
            //
            // See VocabProj: the f32 arm is pre-transposed at load because
            // transposing per token cost more than the whole transformer.
            let last = x.i((.., seq_len - 1.., ..))?.reshape((1, self.n_state))?;
            self.vocab_proj.forward(&last)?.i(0)
        })
    }
}

#[cfg(test)]
mod flat_kv_tests {
    //! Vein 1: the flat cache holds exactly what `Tensor::cat` +
    //! `split_heads().transpose().contiguous()` would have produced.
    use super::*;

    fn rows(n: usize, n_state: usize, seed: &mut u64) -> Vec<f32> {
        (0..n * n_state)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn reference(
        all_k: &[f32],
        all_v: &[f32],
        len: usize,
        heads: usize,
        hd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let dev = ffai_core::candle::Device::Cpu;
        let k = Tensor::from_vec(all_k.to_vec(), (1, len, heads * hd), &dev).unwrap();
        let v = Tensor::from_vec(all_v.to_vec(), (1, len, heads * hd), &dev).unwrap();
        let kt = k
            .reshape((1, len, heads, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .transpose(2, 3)
            .unwrap()
            .contiguous()
            .unwrap();
        let vs = v
            .reshape((1, len, heads, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        (
            kt.flatten_all().unwrap().to_vec1().unwrap(),
            vs.flatten_all().unwrap().to_vec1().unwrap(),
        )
    }

    #[test]
    fn append_grow_materialize_and_snapshot_match_the_tensor_layout() {
        let (heads, hd, max) = (6usize, 64usize, 448usize);
        let n_state = heads * hd;
        let mut seed = 3u64;
        let mut flat = FlatKv::new(heads, hd, max);
        assert_eq!(flat.cap, FlatKv::INITIAL_CAP);
        let mut all_k = Vec::new();
        let mut all_v = Vec::new();
        // A 5-token prefill, then single tokens past the first capacity.
        for n in std::iter::once(5).chain(std::iter::repeat_n(1, 70)) {
            let k = rows(n, n_state, &mut seed);
            let v = rows(n, n_state, &mut seed);
            flat.append(&k, &v, n).unwrap();
            all_k.extend_from_slice(&k);
            all_v.extend_from_slice(&v);
        }
        assert_eq!(flat.len, 75);
        assert!(flat.cap >= 75 && flat.cap <= max, "cap {}", flat.cap);
        let (want_k, want_v) = reference(&all_k, &all_v, flat.len, heads, hd);
        let (kt, vt) = flat.materialize(&ffai_core::candle::Device::Cpu).unwrap();
        assert_eq!(kt.dims(), &[1, heads, hd, 75]);
        assert_eq!(vt.dims(), &[1, heads, 75, hd]);
        let got_k: Vec<f32> = kt.flatten_all().unwrap().to_vec1().unwrap();
        let got_v: Vec<f32> = vt.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got_k, want_k);
        assert_eq!(got_v, want_v);
        // Per-head views the kernel reads: row stride `cap`, `len` live.
        for h in 0..heads {
            let kh = flat.k_head(h);
            for t in 0..hd {
                assert_eq!(
                    &kh[t * flat.cap..t * flat.cap + 75],
                    &want_k[(h * hd + t) * 75..(h * hd + t + 1) * 75]
                );
            }
            assert_eq!(flat.v_head(h), &want_v[h * 75 * hd..(h + 1) * 75 * hd]);
        }
        // Snapshot, advance, restore: the restored cache is the snapshot.
        let snap = flat.snapshot();
        let (k2, v2) = (rows(3, n_state, &mut seed), rows(3, n_state, &mut seed));
        flat.append(&k2, &v2, 3).unwrap();
        assert_eq!(flat.len, 78);
        flat.restore(&snap);
        assert_eq!(flat.len, 75);
        let (kt2, _) = flat.materialize(&ffai_core::candle::Device::Cpu).unwrap();
        let got_k2: Vec<f32> = kt2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got_k2, want_k);
        // Restoring into a smaller, fresh cache grows it to fit.
        let mut fresh = FlatKv::new(heads, hd, max);
        fresh.restore(&snap);
        assert_eq!(fresh.len, 75);
        assert_eq!(fresh.snapshot().k, snap.k);
        // Clearing keeps the allocation and forgets the tokens.
        let cap = flat.cap;
        flat.clear();
        assert_eq!((flat.len, flat.cap), (0, cap));
        // Beyond the model's limit is an error, not a silent overrun.
        let mut tiny = FlatKv::new(heads, hd, 4);
        assert!(
            tiny.append(
                &rows(5, n_state, &mut seed),
                &rows(5, n_state, &mut seed),
                5
            )
            .is_err()
        );
    }
}
