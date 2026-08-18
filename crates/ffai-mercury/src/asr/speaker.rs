//! ECAPA-TDNN speaker embeddings, in candle.
//!
//! Turns a window of audio into a 192-dimensional vector whose direction
//! identifies a voice. [`super::diarize`] does the rest; this module knows
//! nothing about clustering and clustering knows nothing about this.
//!
//! **Architecture** — transcribed from `SpeechBrain`'s `ECAPA_TDNN` with the
//! `spkrec-ecapa-voxceleb` hyperparameters, so a stock checkpoint maps
//! straight on:
//!
//! ```text
//!   channels    [1024, 1024, 1024, 1024, 3072]
//!   kernels     [   5,    3,    3,    3,    1]
//!   dilations   [   1,    2,    3,    4,    1]
//!   attention 128    embedding 192    n_mels 80
//! ```
//!
//! A TDNN block, three SE-Res2Net blocks at rising dilation, multi-layer
//! feature aggregation over the three, attentive statistics pooling, and a
//! linear projection to 192 dims.
//!
//! **Why this one.** The obvious choice is pyannote's embedding model, which
//! is MIT-licensed and **gated** — access walled behind terms acceptance,
//! which cannot live in a manifest under principle 4. `SpeechBrain`'s is
//! Apache-2.0 and ungated.
//!
//! **Verification status.** Shapes and the tensor-name mapping are tested
//! here against randomly-initialised weights. That proves wiring, not
//! numerics — the wav2vec2 port passed every shape test with a
//! weight-normalisation axis wrong, and only the real checkpoint caught it.
//! Two things must land before this is `stable`: a real checkpoint, and a
//! filterbank front end that matches `SpeechBrain`'s `Fbank` rather than
//! Whisper's log-mel. **They are not the same features**, and feeding
//! Whisper's mel into this network produces confident nonsense — embeddings
//! that cluster, but not by speaker.

use candle_nn::ops::softmax;
// `ModuleT` carries `forward_t`, which is how a BatchNorm is told it is in
// eval mode — running statistics rather than batch statistics. Getting that
// wrong on a batch of one would normalise every channel to zero.
use candle_nn::{BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Module, ModuleT, VarBuilder};

use ffai_core::candle::{DType, Device, Result as CandleResult, Tensor};
use ffai_core::error::{Error, Result};

fn model_err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Model(format!("ecapa {what}: {e}"))
}

/// Architecture constants.
#[derive(Debug, Clone)]
pub struct Config {
    pub n_mels: usize,
    pub channels: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub dilations: Vec<usize>,
    pub attention_channels: usize,
    pub embedding_dim: usize,
    /// `Res2Net` cardinality — how many channel groups the block splits into.
    pub res2net_scale: usize,
    pub se_channels: usize,
}

impl Default for Config {
    fn default() -> Self {
        // speechbrain/spkrec-ecapa-voxceleb, from its hyperparams.yaml.
        Self {
            n_mels: 80,
            channels: vec![1024, 1024, 1024, 1024, 3072],
            kernel_sizes: vec![5, 3, 3, 3, 1],
            dilations: vec![1, 2, 3, 4, 1],
            attention_channels: 128,
            embedding_dim: 192,
            res2net_scale: 8,
            se_channels: 128,
        }
    }
}

/// `same` padding for a dilated convolution: keeps the time axis length.
///
/// Off-by-one here does not error — it shifts every frame, so pooling
/// statistics come from a window misaligned with the audio.
const fn same_padding(kernel: usize, dilation: usize) -> usize {
    dilation * (kernel - 1) / 2
}

fn batch_norm_1d(size: usize, vb: VarBuilder) -> Result<BatchNorm> {
    // SpeechBrain's BatchNorm1d defaults: eps 1e-5, affine, running stats.
    candle_nn::batch_norm(
        size,
        BatchNormConfig {
            eps: 1e-5,
            ..Default::default()
        },
        vb,
    )
    .map_err(|e| model_err("batch_norm", e))
}

/// Conv1d → `ReLU` → `BatchNorm`. The unit the whole network is built from.
struct TdnnBlock {
    conv: Conv1d,
    norm: BatchNorm,
}

impl TdnnBlock {
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = candle_nn::conv1d(
            in_ch,
            out_ch,
            kernel,
            Conv1dConfig {
                padding: same_padding(kernel, dilation),
                dilation,
                ..Default::default()
            },
            // `conv.conv`, not `conv`. SpeechBrain's `Conv1d` and
            // `BatchNorm1d` are wrapper classes that hold the torch module as
            // a field, so every parameter sits one level deeper than the
            // attribute name suggests. Read from the checkpoint, not guessed:
            // `blocks.0.conv.conv.weight`, `blocks.0.norm.norm.weight`.
            vb.pp("conv").pp("conv"),
        )
        .map_err(|e| model_err("tdnn conv", e))?;
        Ok(Self {
            conv,
            norm: batch_norm_1d(out_ch, vb.pp("norm").pp("norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.norm.forward_t(&self.conv.forward(x)?.relu()?, false)
    }
}

/// `Res2Net`: split the channels into `scale` groups and process them in a
/// cascade, each group summed with the previous group's output. Gives the
/// block several effective receptive-field sizes for the cost of one.
struct Res2NetBlock {
    blocks: Vec<TdnnBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn load(
        channels: usize,
        scale: usize,
        kernel: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let width = channels / scale;
        let mut blocks = Vec::with_capacity(scale - 1);
        for i in 0..(scale - 1) {
            blocks.push(TdnnBlock::load(
                width,
                width,
                kernel,
                dilation,
                vb.pp("blocks").pp(i.to_string()),
            )?);
        }
        Ok(Self { blocks, scale })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let channels = x.dim(1)?;
        let width = channels / self.scale;
        let mut out: Vec<Tensor> = Vec::with_capacity(self.scale);
        let mut carry: Option<Tensor> = None;
        for i in 0..self.scale {
            let part = x.narrow(1, i * width, width)?;
            if i == 0 {
                // The first group passes through untouched — that is what
                // makes this cheaper than a full-width convolution.
                out.push(part);
                continue;
            }
            let input = match &carry {
                Some(prev) => (&part + prev)?,
                None => part,
            };
            let y = self.blocks[i - 1].forward(&input)?;
            carry = Some(y.clone());
            out.push(y);
        }
        Tensor::cat(&out, 1)
    }
}

/// Squeeze-and-excitation: pool over time, learn a per-channel gain.
struct SeBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SeBlock {
    fn load(channels: usize, se_channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: candle_nn::conv1d(
                channels,
                se_channels,
                1,
                Default::default(),
                vb.pp("conv1").pp("conv"),
            )
            .map_err(|e| model_err("se conv1", e))?,
            conv2: candle_nn::conv1d(
                se_channels,
                channels,
                1,
                Default::default(),
                vb.pp("conv2").pp("conv"),
            )
            .map_err(|e| model_err("se conv2", e))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let pooled = x.mean_keepdim(2)?;
        let gain =
            candle_nn::ops::sigmoid(&self.conv2.forward(&self.conv1.forward(&pooled)?.relu()?)?)?;
        x.broadcast_mul(&gain)
    }
}

struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net: Res2NetBlock,
    tdnn2: TdnnBlock,
    se: SeBlock,
    shortcut: Option<Conv1d>,
}

impl SeRes2NetBlock {
    fn load(
        cfg: &Config,
        in_ch: usize,
        out_ch: usize,
        index: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let kernel = cfg.kernel_sizes[index];
        let dilation = cfg.dilations[index];
        let shortcut = if in_ch == out_ch {
            None
        } else {
            Some(
                candle_nn::conv1d(in_ch, out_ch, 1, Default::default(), vb.pp("shortcut"))
                    .map_err(|e| model_err("shortcut", e))?,
            )
        };
        Ok(Self {
            // The 1x1 blocks around the Res2Net core are always kernel 1,
            // dilation 1 — the dilation belongs to the core.
            tdnn1: TdnnBlock::load(in_ch, out_ch, 1, 1, vb.pp("tdnn1"))?,
            res2net: Res2NetBlock::load(
                out_ch,
                cfg.res2net_scale,
                kernel,
                dilation,
                vb.pp("res2net_block"),
            )?,
            tdnn2: TdnnBlock::load(out_ch, out_ch, 1, 1, vb.pp("tdnn2"))?,
            se: SeBlock::load(out_ch, cfg.se_channels, vb.pp("se_block"))?,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = match &self.shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        let y = self.tdnn1.forward(x)?;
        let y = self.res2net.forward(&y)?;
        let y = self.tdnn2.forward(&y)?;
        let y = self.se.forward(&y)?;
        &y + residual
    }
}

/// Attentive statistics pooling: a learned weighted mean and standard
/// deviation over time, so a loud frame does not automatically dominate.
struct AttentiveStatsPooling {
    tdnn: TdnnBlock,
    conv: Conv1d,
}

impl AttentiveStatsPooling {
    fn load(channels: usize, attention_channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            // Input is 3x channels: the features plus broadcast global mean
            // and std (SpeechBrain's `global_context=True`, its default).
            tdnn: TdnnBlock::load(channels * 3, attention_channels, 1, 1, vb.pp("tdnn"))?,
            conv: candle_nn::conv1d(
                attention_channels,
                channels,
                1,
                Default::default(),
                vb.pp("conv").pp("conv"),
            )
            .map_err(|e| model_err("asp conv", e))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let t = x.dim(2)?;
        let eps = 1e-12f64;

        let mean = x.mean_keepdim(2)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(2)?;
        let std = (var + eps)?.sqrt()?;

        // Global context: every frame sees the utterance-level statistics.
        let mean_b = mean.expand((x.dim(0)?, x.dim(1)?, t))?;
        let std_b = std.expand((x.dim(0)?, x.dim(1)?, t))?;
        let ctx = Tensor::cat(&[x, &mean_b, &std_b], 1)?;

        let attn = self.conv.forward(&self.tdnn.forward(&ctx)?.tanh()?)?;
        // Softmax over TIME, not channels: the weights answer "which frames
        // matter", and normalising the wrong axis produces a plausible tensor
        // with no meaning.
        let attn = softmax(&attn, 2)?;

        let w_mean = x.broadcast_mul(&attn)?.sum_keepdim(2)?;
        let w_sq = x.sqr()?.broadcast_mul(&attn)?.sum_keepdim(2)?;
        let w_std = (w_sq - w_mean.sqr()?)?.clamp(eps, f64::INFINITY)?.sqrt()?;
        Tensor::cat(&[&w_mean, &w_std], 1)
    }
}

/// The embedding network.
pub struct EcapaTdnn {
    cfg: Config,
    blocks: Vec<SeRes2NetBlock>,
    first: TdnnBlock,
    mfa: TdnnBlock,
    asp: AttentiveStatsPooling,
    asp_bn: BatchNorm,
    fc: Conv1d,
    device: Device,
}

impl EcapaTdnn {
    pub fn load(cfg: Config, vb: VarBuilder, device: Device) -> Result<Self> {
        let first = TdnnBlock::load(
            cfg.n_mels,
            cfg.channels[0],
            cfg.kernel_sizes[0],
            cfg.dilations[0],
            vb.pp("blocks").pp("0"),
        )?;
        let mut blocks = Vec::new();
        for i in 1..cfg.channels.len() - 1 {
            blocks.push(SeRes2NetBlock::load(
                &cfg,
                cfg.channels[i - 1],
                cfg.channels[i],
                i,
                vb.pp("blocks").pp(i.to_string()),
            )?);
        }
        // Multi-layer feature aggregation over the three SE-Res2Net outputs.
        let mfa_in = cfg.channels[1..cfg.channels.len() - 1]
            .iter()
            .sum::<usize>();
        let mfa_out = *cfg.channels.last().expect("channels is non-empty");
        let mfa = TdnnBlock::load(
            mfa_in,
            mfa_out,
            *cfg.kernel_sizes.last().expect("non-empty"),
            *cfg.dilations.last().expect("non-empty"),
            vb.pp("mfa"),
        )?;
        let asp = AttentiveStatsPooling::load(mfa_out, cfg.attention_channels, vb.pp("asp"))?;
        // `asp_bn.norm` and `fc.conv` are only ONE level deep — the wrapper
        // is the attribute itself here rather than a field inside a block.
        // Uniformly applying the double nesting above would miss both.
        let asp_bn = batch_norm_1d(mfa_out * 2, vb.pp("asp_bn").pp("norm"))?;
        let fc = candle_nn::conv1d(
            mfa_out * 2,
            cfg.embedding_dim,
            1,
            Default::default(),
            vb.pp("fc").pp("conv"),
        )
        .map_err(|e| model_err("fc", e))?;
        Ok(Self {
            cfg,
            blocks,
            first,
            mfa,
            asp,
            asp_bn,
            fc,
            device,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.cfg
    }

    /// Features `(frames, n_mels)` → a single `embedding_dim` vector.
    pub fn embed(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        if frames == 0 || features.len() != frames * self.cfg.n_mels {
            return Err(Error::Model(format!(
                "ecapa expects {frames} x {} features, got {}",
                self.cfg.n_mels,
                features.len()
            )));
        }
        // (1, n_mels, frames) — channels-first, as the convolutions want.
        let x = Tensor::from_slice(features, (1, frames, self.cfg.n_mels), &self.device)
            .and_then(|t| t.transpose(1, 2))
            .and_then(|t| t.contiguous())
            .map_err(|e| model_err("input tensor", e))?;

        let mut h = self
            .first
            .forward(&x)
            .map_err(|e| model_err("block 0", e))?;
        let mut outs = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            h = block
                .forward(&h)
                .map_err(|e| model_err(&format!("block {}", i + 1), e))?;
            outs.push(h.clone());
        }
        let cat = Tensor::cat(&outs, 1).map_err(|e| model_err("mfa concat", e))?;
        let h = self.mfa.forward(&cat).map_err(|e| model_err("mfa", e))?;
        let h = self.asp.forward(&h).map_err(|e| model_err("asp", e))?;
        let h = self
            .asp_bn
            .forward_t(&h, false)
            .map_err(|e| model_err("asp_bn", e))?;
        let h = self.fc.forward(&h).map_err(|e| model_err("fc", e))?;
        h.flatten_all()
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err("embedding readback", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// A scaled-down configuration: wiring is what is under test, and the
    /// real 1024-channel network would make this a benchmark.
    fn tiny() -> Config {
        Config {
            n_mels: 16,
            channels: vec![32, 32, 32, 32, 96],
            kernel_sizes: vec![5, 3, 3, 3, 1],
            dilations: vec![1, 2, 3, 4, 1],
            attention_channels: 8,
            embedding_dim: 24,
            res2net_scale: 8,
            se_channels: 8,
        }
    }

    fn build(cfg: &Config) -> (EcapaTdnn, VarMap) {
        let map = VarMap::new();
        let vb = VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu);
        (
            EcapaTdnn::load(cfg.clone(), vb, Device::Cpu).expect("loads"),
            map,
        )
    }

    #[test]
    fn same_padding_preserves_length_for_every_configured_dilation() {
        // Every (kernel, dilation) pair the real config uses must keep the
        // time axis, or the MFA concat cannot line up.
        for (k, d) in [(5, 1), (3, 2), (3, 3), (3, 4), (1, 1)] {
            let pad = same_padding(k, d);
            let l_in = 100;
            let l_out = (l_in + 2 * pad - d * (k - 1) - 1) + 1;
            assert_eq!(l_out, l_in, "kernel {k} dilation {d} changed length");
        }
    }

    #[test]
    fn embedding_has_the_configured_width() {
        let cfg = tiny();
        let (model, _map) = build(&cfg);
        let frames = 150;
        let e = model
            .embed(&vec![0.1f32; frames * cfg.n_mels], frames)
            .expect("embeds");
        assert_eq!(e.len(), cfg.embedding_dim);
        assert!(e.iter().all(|v| v.is_finite()), "non-finite embedding");
    }

    #[test]
    fn embedding_is_independent_of_input_length() {
        // Pooling is over time, so a 1 s and a 3 s window must both give one
        // vector of the same width — this is what lets windows be compared.
        let cfg = tiny();
        let (model, _map) = build(&cfg);
        let a = model
            .embed(&vec![0.1f32; 100 * cfg.n_mels], 100)
            .expect("a");
        let b = model
            .embed(&vec![0.1f32; 300 * cfg.n_mels], 300)
            .expect("b");
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn mismatched_feature_count_is_rejected() {
        let (model, _map) = build(&tiny());
        assert!(model.embed(&[0.0; 10], 7).is_err());
        assert!(model.embed(&[], 0).is_err());
    }

    #[test]
    fn different_inputs_give_different_embeddings() {
        // Guards the degenerate case where pooling collapses everything to a
        // constant, which would cluster perfectly and mean nothing.
        let cfg = tiny();
        let (model, _map) = build(&cfg);
        let a = model
            .embed(&vec![0.1f32; 200 * cfg.n_mels], 200)
            .expect("a");
        let ramp: Vec<f32> = (0..200 * cfg.n_mels)
            .map(|i| (i % 17) as f32 * 0.01)
            .collect();
        let b = model.embed(&ramp, 200).expect("b");
        let dist = super::super::diarize::cosine_distance(&a, &b);
        assert!(dist > 1e-6, "embeddings collapsed: distance {dist}");
    }

    #[test]
    fn default_config_matches_the_published_hyperparams() {
        let cfg = Config::default();
        assert_eq!(cfg.channels, vec![1024, 1024, 1024, 1024, 3072]);
        assert_eq!(cfg.kernel_sizes, vec![5, 3, 3, 3, 1]);
        assert_eq!(cfg.dilations, vec![1, 2, 3, 4, 1]);
        assert_eq!(cfg.embedding_dim, 192);
        assert_eq!(cfg.n_mels, 80);
        // MFA input is the three SE-Res2Net outputs concatenated.
        assert_eq!(cfg.channels[1..4].iter().sum::<usize>(), 3072);
    }
}
