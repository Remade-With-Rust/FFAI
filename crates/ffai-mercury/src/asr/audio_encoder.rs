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
            self.mlp2
                .forward(&fast_gelu(&self.mlp1.forward(&self.mlp_ln.forward(&x)?)?)?)
        })?;
        x + mlp
    }
}

/// Whisper's audio encoder: log-mel spectrogram → audio features.
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
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
        Ok(AudioEncoder { conv1, conv2, positional_embedding, blocks, ln_post })
    }

    /// `mel`: (batch, n_mels, frames) → features (batch, frames/2, d_model).
    pub fn forward(&self, mel: &Tensor) -> CandleResult<Tensor> {
        let p = super::profile::profile();
        let x = super::profile::timed(&p.enc_conv, || -> CandleResult<Tensor> {
            let x = fast_gelu(&self.conv1.forward(mel)?)?;
            fast_gelu(&self.conv2.forward(&x)?)
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
