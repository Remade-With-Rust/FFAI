//! Mercury's own Whisper audio encoder.
//!
//! # Why this exists
//!
//! Quantizing the decoder alone bought nothing (mission plan §6.5): int8 made
//! its matmuls 2–3× faster, but the encoder is ~42 % of transcription time and
//! stayed f32 because it belonged to candle. Amdahl's law did the rest.
//!
//! The encoder analysis (§6.4) says what this needs to be: it is **O(n) in
//! sequence length and not cache-bound**, so its cost is plain matmul work —
//! the MLP path dominates attention at these sizes — which is precisely what
//! int8 addresses. There is no clever algorithm to find here, only precision.
//!
//! Structure (Whisper's, unchanged):
//!
//! ```text
//! conv1(n_mels -> d_model, k=3, pad=1) + gelu
//! conv2(d_model -> d_model, k=3, stride=2, pad=1) + gelu   <- halves the sequence
//! transpose -> + sinusoidal positional embedding
//! N x (self-attention + MLP)
//! layer_norm
//! ```
//!
//! Convolutions stay f32: they are two layers against N transformer blocks,
//! and `QMatMul` quantizes matmuls, not convolutions. The blocks — where the
//! time is — quantize.

use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Module, VarBuilder};
use ffai_core::candle::{Result as CandleResult, Tensor};

use super::text_decoder::{fast_gelu, EncoderAttention, Precision, QLinear};

/// Conv1d as im2col + GEMM, in CHANNEL-MAJOR orientation.
///
/// candle's conv1d ran the encoder front end at 95-214 GFLOP/s while every
/// matmul in this pipeline reaches 350-550. The arithmetic is a GEMM; it just
/// was not being run as one.
///
/// Two things make this work, and I had both wrong on the first pass:
///
/// 1. **The im2col does not need a strided gather.** With channel-major
///    (C, L) input, row `c*3+k` of the im2col matrix is `in[c][k..k+L]` — a
///    contiguous slice. Building it is 240 `copy_from_slice` calls, 0.75 ms.
///    I had assumed a transpose-shaped gather and refuted the idea on that.
///
/// 2. **Orientation decides everything.** `(1500,1152)@(1152,384)` measures
///    **54 GFLOP/s**; the transpose `(384,1152)@(1152,1500)` measures **539**
///    — 10x, for identical arithmetic. I benchmarked the slow orientation and
///    concluded conv2 was already beating a GEMM.
///
/// | | candle | im2col+GEMM |
/// |---|---:|---:|
/// | conv1 (80->384, k3, s1) | 5.84 ms | 2.34 ms (2.07x) |
/// | conv2 (384->384, k3, s2) | 6.20 ms | 3.87 ms (1.60x) |
fn conv1d_gemm(
    x: &Tensor,
    wm: &Tensor,
    bias: &Tensor,
    cin: usize,
    stride: usize,
) -> CandleResult<Tensor> {
    let batch = x.dim(0)?;
    let l_in = x.dim(2)?;
    let l_out = (l_in + 2 * 1 - 3) / stride + 1;
    let src: Vec<f32> = x.flatten_all()?.to_vec1()?;

    // im2col for every batch item into ONE buffer, laid out so the columns of
    // item b occupy `[b*l_out, (b+1)*l_out)`. That makes the whole batch a
    // single `(cin*3) x (batch*l_out)` GEMM instead of `batch` small ones —
    // the weight matrix is read once rather than per item.
    //
    // This used to index `src[c*l_in..]` with no batch stride while the
    // caller's signature advertised `(batch, n_mels, frames)`. With batch > 1
    // it silently produced item 0's answer for the whole batch: no error, no
    // shape mismatch, just wrong features for every window after the first.
    let mut col = vec![0f32; cin * 3 * batch * l_out];
    let row_stride = batch * l_out;
    for b in 0..batch {
        let base = b * cin * l_in;
        for c in 0..cin {
            let s = &src[base + c * l_in..base + (c + 1) * l_in];
            for k in 0..3 {
                let row = (c * 3 + k) * row_stride + b * l_out;
                let dst = &mut col[row..row + l_out];
                if stride == 1 {
                    // Contiguous slices — padding 1 means tap k reads t+k-1.
                    match k {
                        0 => dst[1..].copy_from_slice(&s[..l_out - 1]),
                        1 => dst.copy_from_slice(&s[..l_out]),
                        _ => dst[..l_out - 1].copy_from_slice(&s[1..l_out]),
                    }
                } else {
                    for (i, o) in dst.iter_mut().enumerate() {
                        let idx = stride * i + k;
                        *o = if idx == 0 { 0.0 } else { *s.get(idx - 1).unwrap_or(&0.0) };
                    }
                }
            }
        }
    }
    let cout = wm.dim(0)?;
    let colt = Tensor::from_vec(col, (cin * 3, batch * l_out), x.device())?;
    let y = wm
        .matmul(&colt)?
        .broadcast_add(&bias.reshape((bias.dim(0)?, 1))?)?
        // (cout, batch*l_out) -> (cout, batch, l_out) -> (batch, cout, l_out)
        .reshape((cout, batch, l_out))?;
    if batch == 1 {
        // Keep the single-window path allocation-identical to before.
        y.reshape((1, cout, l_out))
    } else {
        y.transpose(0, 1)?.contiguous()
    }
}

fn layer_norm(size: usize, vb: VarBuilder) -> CandleResult<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(size, "weight")?,
        vb.get(size, "bias")?,
        1e-5,
    ))
}

struct Block {
    attn: EncoderAttention,
    attn_ln: LayerNorm,
    mlp_ln: LayerNorm,
    mlp1: QLinear,
    mlp2: QLinear,
}

impl Block {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder, p: Precision) -> CandleResult<Self> {
        Ok(Block {
            attn: EncoderAttention::load(n_state, n_head, vb.pp("self_attn"), p)?,
            attn_ln: layer_norm(n_state, vb.pp("self_attn_layer_norm"))?,
            mlp_ln: layer_norm(n_state, vb.pp("final_layer_norm"))?,
            mlp1: QLinear::from_vb(n_state, n_state * 4, vb.pp("fc1"), true, p)?,
            mlp2: QLinear::from_vb(n_state * 4, n_state, vb.pp("fc2"), true, p)?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // The encoder is bidirectional: no causal mask, no KV cache — every
        // position sees every other, and each call is a fresh full-sequence
        // pass over one 30 s window.
        let p = super::profile::profile();
        let ln = self.attn_ln.forward(x)?;
        let attn = super::profile::timed(&p.enc_attn, || self.attn.forward(&ln))?;
        let x = (x + attn)?;
        let mlp = super::profile::timed(&p.enc_mlp, || {
            // Broken out because the stage carried a ~10 ms residue that only
            // standalone probes had estimated — and on encoder attention those
            // probes were 7x wrong about which op owned the time.
            let n = super::profile::timed(&p.em_ln, || self.mlp_ln.forward(&x))?;
            let h = super::profile::timed(&p.em_fc1, || self.mlp1.forward(&n))?;
            let g = super::profile::timed(&p.em_gelu, || fast_gelu(&h))?;
            super::profile::timed(&p.em_fc2, || self.mlp2.forward(&g))
        })?;
        x + mlp
    }
}

/// Whisper's audio encoder: log-mel spectrogram → audio features.
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    /// (cout, cin*3) reshapes of the conv weights, for the GEMM path.
    conv1_wm: Tensor,
    conv2_wm: Tensor,
    conv1_b: Tensor,
    conv2_b: Tensor,
    positional_embedding: Tensor,
    blocks: Vec<Block>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    pub fn load(
        vb: VarBuilder,
        cfg: &candle_transformers::models::whisper::Config,
        precision: Precision,
    ) -> CandleResult<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let conv1 = candle_nn::conv1d(
            cfg.num_mel_bins,
            n_state,
            3,
            Conv1dConfig { padding: 1, ..Default::default() },
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(
            n_state,
            n_state,
            3,
            Conv1dConfig { padding: 1, stride: 2, ..Default::default() },
            vb.pp("conv2"),
        )?;
        let positional_embedding =
            vb.get((cfg.max_source_positions, n_state), "embed_positions.weight")?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| Block::load(n_state, n_head, vb.pp(format!("layers.{i}")), precision))
            .collect::<CandleResult<Vec<_>>>()?;
        let ln_post = layer_norm(n_state, vb.pp("layer_norm"))?;
        // Pay the (cout, cin, 3) -> (cout, cin*3) reshape once, at load.
        let conv1_wm = {
            let w = conv1.weight();
            w.reshape((w.dim(0)?, w.dim(1)? * 3))?.contiguous()?
        };
        let conv2_wm = {
            let w = conv2.weight();
            w.reshape((w.dim(0)?, w.dim(1)? * 3))?.contiguous()?
        };
        let conv1_b = vb.pp("conv1").get(n_state, "bias")?;
        let conv2_b = vb.pp("conv2").get(n_state, "bias")?;
        Ok(AudioEncoder {
            conv1,
            conv2,
            conv1_wm,
            conv2_wm,
            conv1_b,
            conv2_b,
            positional_embedding,
            blocks,
            ln_post,
        })
    }

    /// `mel`: (batch, n_mels, frames) → features (batch, frames/2, d_model).
    pub fn forward(&self, mel: &Tensor) -> CandleResult<Tensor> {
        let p = super::profile::profile();
        let x = super::profile::timed(&p.enc_conv, || -> CandleResult<Tensor> {
            // GEMM path where it applies; candle's conv1d otherwise.
            let c1 = if mel.dim(1)? == self.conv1_wm.dim(1)? / 3 {
                conv1d_gemm(mel, &self.conv1_wm, &self.conv1_b, mel.dim(1)?, 1)?
            } else {
                self.conv1.forward(mel)?
            };
            let x = fast_gelu(&c1)?;
            let c2 = conv1d_gemm(&x, &self.conv2_wm, &self.conv2_b, x.dim(1)?, 2)?;
            fast_gelu(&c2)
        })?;
        let x = x.transpose(1, 2)?;
        let seq_len = x.dim(1)?;
        let positions = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&positions)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.ln_post.forward(&x)
    }
}
