//! wav2vec2-CTC in candle — the acoustic model behind word timestamps.
//!
//! Produces per-frame character log-probabilities for [`super::align`] to
//! align a known transcript against. Nothing here does recognition: the CTC
//! head's output is only ever consumed as emissions by the forced aligner.
//!
//! **Why this is written out by hand.** `candle-transformers` 0.11 ships 125
//! models and exactly two of them are audio — `whisper` and `encodec`. There
//! is no wav2vec2, so there is nothing to call. The architecture below is
//! transcribed from HuggingFace's `Wav2Vec2ForCTC`, using their tensor names
//! so a stock `facebook/wav2vec2-*` checkpoint maps straight onto it.
//!
//! **Frame rate is the load-bearing number.** The convolutional front end
//! strides 5·2·2·2·2·2·2 = 320 samples, so at 16 kHz one output frame is
//! exactly 20 ms. That constant is what turns an aligned frame index into a
//! timestamp, and it is asserted in the tests rather than trusted.
//!
//! **Verification status, stated plainly.** Shapes, strides and the frame-rate
//! arithmetic are tested here against randomly-initialised weights. That
//! proves the wiring, *not* the numerics — a transposed projection or a
//! swapped norm would pass every test in this file. Oracle verification
//! against HuggingFace outputs on a real checkpoint is Mercury-X §C3 and this
//! model is not `stable` until that lands.

use candle_nn::ops::softmax_last_dim;
use candle_nn::{Conv1d, Conv1dConfig, GroupNorm, LayerNorm, Linear, Module, VarBuilder};

use ffai_core::candle::quantized::{GgmlDType, QMatMul, QTensor};
use ffai_core::candle::{DType, Device, Result as CandleResult, Tensor};
use ffai_core::error::{Error, Result};

use super::align::Emissions;

fn model_err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Model(format!("wav2vec2 {what}: {e}"))
}

/// Architecture constants. `base` and `large` differ only in these.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
    /// `true` for base checkpoints: group-norm on the first conv layer only.
    /// `false` for large: layer-norm on every conv layer.
    pub conv_group_norm: bool,
    /// Pre-norm (`true`, large) vs post-norm (`false`, base) transformer.
    pub stable_layer_norm: bool,
    pub pos_conv_kernel: usize,
    pub pos_conv_groups: usize,
    pub vocab: usize,
    pub layer_norm_eps: f64,
}

impl Config {
    /// `facebook/wav2vec2-base-960h`.
    pub fn base_960h() -> Self {
        Config {
            hidden: 768,
            layers: 12,
            heads: 12,
            intermediate: 3072,
            conv_dim: vec![512; 7],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            conv_group_norm: true,
            stable_layer_norm: false,
            pos_conv_kernel: 128,
            pos_conv_groups: 16,
            vocab: 32,
            layer_norm_eps: 1e-5,
        }
    }

    /// Total downsampling factor of the convolutional front end.
    pub fn total_stride(&self) -> usize {
        self.conv_stride.iter().product()
    }

    /// Seconds of audio per output frame at the given sample rate.
    pub fn frame_secs(&self, sample_rate: usize) -> f64 {
        self.total_stride() as f64 / sample_rate as f64
    }

    /// Output frames for an input of `n` samples — each conv layer floors.
    pub fn output_frames(&self, n: usize) -> usize {
        let mut len = n;
        for (k, s) in self.conv_kernel.iter().zip(self.conv_stride.iter()) {
            len = if len < *k { 0 } else { (len - k) / s + 1 };
        }
        len
    }
}

/// A projection that is either plain f32 or int8-quantized.
///
/// **Why only the projections.** The transformer stack is ~85M of this
/// model's 95M parameters; the convolutional front end is the other 10M and
/// stays f32. Quantizing 90 % of the weights gets essentially all of the
/// saving without touching the part whose activations have the widest range.
///
/// **Why Q8_0 and not f16.** f16 was tried first and refuted: the forward
/// pass runs but emissions go degenerate, because wav2vec2-base was not
/// trained for half precision and its conv-stack activations do not survive
/// the range. Q8_0 carries a per-32-element scale, so the representable range
/// follows the data instead of being fixed — which is the property f16 lacked.
enum Projection {
    Plain(Linear),
    Quantized { w: QMatMul, bias: Option<Tensor> },
}

impl Projection {
    fn load(in_dim: usize, out_dim: usize, vb: VarBuilder, quantized: bool) -> Result<Self> {
        if !quantized {
            return candle_nn::linear(in_dim, out_dim, vb)
                .map(Projection::Plain)
                .map_err(|e| model_err("linear", e));
        }
        let weight = vb
            .get((out_dim, in_dim), "weight")
            .map_err(|e| model_err("quantized weight", e))?;
        let bias = vb.get(out_dim, "bias").ok();
        // Quantization needs f32 input whatever the VarBuilder was opened as.
        let weight = weight.to_dtype(DType::F32).map_err(|e| model_err("weight dtype", e))?;
        let q = QTensor::quantize(&weight, GgmlDType::Q8_0)
            .map_err(|e| model_err("quantize", e))?;
        let w = QMatMul::from_qtensor(q).map_err(|e| model_err("qmatmul", e))?;
        Ok(Projection::Quantized { w, bias })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Projection::Plain(l) => l.forward(x),
            Projection::Quantized { w, bias } => {
                let y = w.forward(x)?;
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
        }
    }
}

/// One layer of the convolutional feature extractor.
struct ConvLayer {
    conv: Conv1d,
    group_norm: Option<GroupNorm>,
    layer_norm: Option<LayerNorm>,
}

impl ConvLayer {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.conv.forward(x)?;
        let x = match (&self.group_norm, &self.layer_norm) {
            (Some(gn), _) => gn.forward(&x)?,
            // The "layer" variant normalises over the channel axis, so the
            // tensor is transposed into (batch, time, channel) and back.
            (_, Some(ln)) => ln.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?,
            _ => x,
        };
        x.gelu_erf()
    }
}

struct FeatureExtractor {
    layers: Vec<ConvLayer>,
}

impl FeatureExtractor {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("conv_layers");
        let mut layers = Vec::with_capacity(cfg.conv_dim.len());
        for i in 0..cfg.conv_dim.len() {
            let vbl = vb.pp(i.to_string());
            let in_ch = if i == 0 { 1 } else { cfg.conv_dim[i - 1] };
            let out_ch = cfg.conv_dim[i];
            let conv = candle_nn::conv1d_no_bias(
                in_ch,
                out_ch,
                cfg.conv_kernel[i],
                Conv1dConfig { stride: cfg.conv_stride[i], ..Default::default() },
                vbl.pp("conv"),
            )
            .map_err(|e| model_err(&format!("conv layer {i}"), e))?;

            // Base checkpoints normalise the first conv layer only, with
            // groups == channels (instance norm). Applying it to every layer,
            // or to none, silently changes the feature scale.
            let (group_norm, layer_norm) = if cfg.conv_group_norm {
                if i == 0 {
                    let gn = candle_nn::group_norm(out_ch, out_ch, cfg.layer_norm_eps, vbl.pp("layer_norm"))
                        .map_err(|e| model_err("conv group_norm", e))?;
                    (Some(gn), None)
                } else {
                    (None, None)
                }
            } else {
                let ln = candle_nn::layer_norm(out_ch, cfg.layer_norm_eps, vbl.pp("layer_norm"))
                    .map_err(|e| model_err(&format!("conv layer_norm {i}"), e))?;
                (None, Some(ln))
            };
            layers.push(ConvLayer { conv, group_norm, layer_norm });
        }
        Ok(FeatureExtractor { layers })
    }

    /// `(batch, samples)` → `(batch, channels, frames)`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut x = x.unsqueeze(1)?; // add the single input channel
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

struct Attention {
    q: Projection,
    k: Projection,
    v: Projection,
    out: Projection,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(cfg: &Config, vb: VarBuilder, quantized: bool) -> Result<Self> {
        let h = cfg.hidden;
        let head_dim = h / cfg.heads;
        let lin = |name: &str, vb: &VarBuilder| {
            Projection::load(h, h, vb.pp(name), quantized)
        };
        Ok(Attention {
            q: lin("q_proj", &vb)?,
            k: lin("k_proj", &vb)?,
            v: lin("v_proj", &vb)?,
            out: lin("out_proj", &vb)?,
            heads: cfg.heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, t, _) = x.dims3()?;
        let split = |t_: Tensor| -> CandleResult<Tensor> {
            t_.reshape((b, t, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()
        };
        let q = split(self.q.forward(x)?)?;
        let k = split(self.k.forward(x)?)?;
        let v = split(self.v.forward(x)?)?;

        let scores = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let probs = softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, t, self.heads * self.head_dim))?;
        self.out.forward(&ctx)
    }
}

struct EncoderLayer {
    attn: Attention,
    attn_norm: LayerNorm,
    ff_in: Projection,
    ff_out: Projection,
    final_norm: LayerNorm,
    stable: bool,
}

impl EncoderLayer {
    fn load(cfg: &Config, vb: VarBuilder, quantized: bool) -> Result<Self> {
        Ok(EncoderLayer {
            attn: Attention::load(cfg, vb.pp("attention"), quantized)?,
            attn_norm: candle_nn::layer_norm(cfg.hidden, cfg.layer_norm_eps, vb.pp("layer_norm"))
                .map_err(|e| model_err("layer_norm", e))?,
            ff_in: Projection::load(
                cfg.hidden,
                cfg.intermediate,
                vb.pp("feed_forward").pp("intermediate_dense"),
                quantized,
            )?,
            ff_out: Projection::load(
                cfg.intermediate,
                cfg.hidden,
                vb.pp("feed_forward").pp("output_dense"),
                quantized,
            )?,
            final_norm: candle_nn::layer_norm(
                cfg.hidden,
                cfg.layer_norm_eps,
                vb.pp("final_layer_norm"),
            )
            .map_err(|e| model_err("final_layer_norm", e))?,
            stable: cfg.stable_layer_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        if self.stable {
            // Pre-norm (large): normalise going in, residual stays clean.
            let h = (x + self.attn.forward(&self.attn_norm.forward(x)?)?)?;
            let ff = self.ff_out.forward(&self.ff_in.forward(&self.final_norm.forward(&h)?)?.gelu_erf()?)?;
            &h + ff
        } else {
            // Post-norm (base): normalise after each residual.
            let h = self.attn_norm.forward(&(x + self.attn.forward(x)?)?)?;
            let ff = self.ff_out.forward(&self.ff_in.forward(&h)?.gelu_erf()?)?;
            self.final_norm.forward(&(&h + ff)?)
        }
    }
}

/// The relative positional embedding: a wide grouped convolution added to the
/// features. Its weights are stored weight-normalised (`weight_g`/`weight_v`,
/// or `parametrizations.weight.original0/original1` in newer exports), so the
/// effective kernel has to be reconstructed at load time.
struct PosConv {
    conv: Conv1d,
    trim_last: bool,
}

impl PosConv {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("conv");
        let v_shape = (cfg.hidden, cfg.hidden / cfg.pos_conv_groups, cfg.pos_conv_kernel);
        // HF applies `weight_norm(conv, name="weight", dim=2)`, so the gain is
        // per KERNEL POSITION — `[1, 1, k]`, one scalar per tap — not per
        // output channel. Reading it as `[c, 1, 1]` is the same size only by
        // coincidence of the numbers involved and gives a silently wrong
        // convolution. Verified against the checkpoint header, not assumed:
        //   weight_g [1, 1, 128]   weight_v [768, 48, 128]
        let g_shape = (1, 1, cfg.pos_conv_kernel);
        // Two export vintages, same tensor. Trying both is cheaper than
        // telling a user their checkpoint is unsupported because HF renamed a
        // parametrisation.
        let (g, v) = match (vb.get(g_shape, "weight_g"), vb.get(v_shape, "weight_v")) {
            (Ok(g), Ok(v)) => (g, v),
            _ => (
                vb.get(g_shape, "parametrizations.weight.original0")
                    .map_err(|e| model_err("pos_conv weight_g", e))?,
                vb.get(v_shape, "parametrizations.weight.original1")
                    .map_err(|e| model_err("pos_conv weight_v", e))?,
            ),
        };
        // weight = g * v / ||v||, with the norm over the axes `dim=2` leaves
        // out — channels and taps-per-group — keeping the kernel axis.
        //
        // The epsilon is not decoration. A zero `weight_v` slice makes this
        // 0/0, and a NaN here does not announce itself — it propagates
        // silently through the whole encoder and surfaces as "no valid
        // alignment path" from the aligner, a hundred lines away from the
        // cause. Cheap guard, and it costs nothing on a real checkpoint where
        // the norm is O(1).
        let norm = v
            .sqr()
            .and_then(|t| t.sum_keepdim(0))
            .and_then(|t| t.sum_keepdim(1))
            .and_then(|t| (t + 1e-12)?.sqrt())
            .map_err(|e| model_err("pos_conv weight norm", e))?;
        let weight = v
            .broadcast_div(&norm)
            .and_then(|t| t.broadcast_mul(&g))
            .map_err(|e| model_err("pos_conv weight_norm reconstruction", e))?;
        let bias = vb.get(cfg.hidden, "bias").map_err(|e| model_err("pos_conv bias", e))?;

        let conv = Conv1d::new(
            weight,
            Some(bias),
            Conv1dConfig {
                padding: cfg.pos_conv_kernel / 2,
                groups: cfg.pos_conv_groups,
                ..Default::default()
            },
        );
        // An even kernel with `padding = k/2` returns one frame too many;
        // HF's `Wav2Vec2SamePadLayer` drops the last. Off by one here shifts
        // every timestamp by 20 ms.
        Ok(PosConv { conv, trim_last: cfg.pos_conv_kernel % 2 == 0 })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // (b, t, c) -> (b, c, t) for convolution, and back.
        let h = self.conv.forward(&x.transpose(1, 2)?.contiguous()?)?;
        let h = if self.trim_last {
            let t = h.dim(2)?;
            h.narrow(2, 0, t - 1)?
        } else {
            h
        };
        h.gelu_erf()?.transpose(1, 2)?.contiguous()
    }
}

/// `Wav2Vec2ForCTC`: features → transformer → character logits.
pub struct Wav2Vec2Ctc {
    cfg: Config,
    features: FeatureExtractor,
    feat_norm: LayerNorm,
    feat_proj: Linear,
    pos_conv: PosConv,
    encoder_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    lm_head: Linear,
    device: Device,
    /// The weights' dtype. Inputs are built in f32 and must be cast to match:
    /// candle's conv1d refuses a dtype mismatch rather than promoting, which
    /// is the right call and made an f16 build fail on every segment.
    dtype: DType,
}

impl Wav2Vec2Ctc {
    pub fn load(cfg: Config, vb: VarBuilder, device: Device) -> Result<Self> {
        Self::load_with(cfg, vb, device, false)
    }

    /// `quantized` puts the transformer projections through Q8_0.
    pub fn load_with(
        cfg: Config,
        vb: VarBuilder,
        device: Device,
        quantized: bool,
    ) -> Result<Self> {
        let root = vb.pp("wav2vec2");
        let features = FeatureExtractor::load(&cfg, root.pp("feature_extractor"))?;
        let last_conv = *cfg.conv_dim.last().expect("conv_dim is never empty");
        let feat_norm = candle_nn::layer_norm(
            last_conv,
            cfg.layer_norm_eps,
            root.pp("feature_projection").pp("layer_norm"),
        )
        .map_err(|e| model_err("feature_projection.layer_norm", e))?;
        let feat_proj = candle_nn::linear(
            last_conv,
            cfg.hidden,
            root.pp("feature_projection").pp("projection"),
        )
        .map_err(|e| model_err("feature_projection.projection", e))?;

        let enc = root.pp("encoder");
        let pos_conv = PosConv::load(&cfg, enc.pp("pos_conv_embed"))?;
        let encoder_norm =
            candle_nn::layer_norm(cfg.hidden, cfg.layer_norm_eps, enc.pp("layer_norm"))
                .map_err(|e| model_err("encoder.layer_norm", e))?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            layers.push(EncoderLayer::load(&cfg, enc.pp("layers").pp(i.to_string()), quantized)?);
        }
        let lm_head = candle_nn::linear(cfg.hidden, cfg.vocab, vb.pp("lm_head"))
            .map_err(|e| model_err("lm_head", e))?;

        let dtype = vb.dtype();
        Ok(Wav2Vec2Ctc {
            dtype,
            cfg,
            features,
            feat_norm,
            feat_proj,
            pos_conv,
            encoder_norm,
            layers,
            lm_head,
            device,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Character logits for one mono waveform: `(frames, vocab)`.
    pub fn logits(&self, samples: &[f32]) -> Result<Tensor> {
        if self.cfg.output_frames(samples.len()) == 0 {
            return Err(Error::Model(format!(
                "wav2vec2 needs at least {} samples to produce one frame, got {}",
                self.cfg.total_stride(),
                samples.len()
            )));
        }
        let x = Tensor::from_slice(samples, (1, samples.len()), &self.device)
            .and_then(|t| t.to_dtype(self.dtype))
            .map_err(|e| model_err("input tensor", e))?;

        let feats = self.features.forward(&x).map_err(|e| model_err("feature extractor", e))?;
        // (b, c, t) -> (b, t, c)
        let feats = feats.transpose(1, 2).and_then(|t| t.contiguous()).map_err(|e| model_err("transpose", e))?;
        let h = self
            .feat_norm
            .forward(&feats)
            .and_then(|t| self.feat_proj.forward(&t))
            .map_err(|e| model_err("feature projection", e))?;

        let pos = self.pos_conv.forward(&h).map_err(|e| model_err("pos_conv", e))?;
        let mut h = (&h + pos).map_err(|e| model_err("pos_conv residual", e))?;
        if !self.cfg.stable_layer_norm {
            h = self.encoder_norm.forward(&h).map_err(|e| model_err("encoder norm", e))?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h).map_err(|e| model_err(&format!("encoder layer {i}"), e))?;
        }
        if self.cfg.stable_layer_norm {
            h = self.encoder_norm.forward(&h).map_err(|e| model_err("encoder norm", e))?;
        }
        self.lm_head
            .forward(&h)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| model_err("lm_head", e))
    }

    /// Log-softmaxed emissions, ready for [`super::align::forced_align`].
    pub fn emissions(&self, samples: &[f32], sample_rate: usize) -> Result<Emissions> {
        let logits = self.logits(samples)?;
        let logp = candle_nn::ops::log_softmax(&logits, 1)
            .map_err(|e| model_err("log_softmax", e))?
            .to_dtype(DType::F32)
            .map_err(|e| model_err("emissions dtype", e))?;
        let (frames, vocab) = logp.dims2().map_err(|e| model_err("emissions shape", e))?;
        let data = logp.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| model_err("emissions readback", e))?;
        Emissions::new(frames, vocab, data, self.cfg.frame_secs(sample_rate))
            .map_err(Error::Model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// A deliberately tiny configuration — the wiring is what is under test,
    /// and a 12-layer 768-wide model would make this a benchmark instead.
    fn tiny() -> Config {
        Config {
            hidden: 32,
            layers: 2,
            heads: 4,
            intermediate: 64,
            conv_dim: vec![16; 7],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            conv_group_norm: true,
            stable_layer_norm: false,
            pos_conv_kernel: 16,
            pos_conv_groups: 4,
            vocab: 32,
            layer_norm_eps: 1e-5,
        }
    }

    fn build(cfg: &Config) -> (Wav2Vec2Ctc, VarMap) {
        let map = VarMap::new();
        let vb = VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu);
        let model = Wav2Vec2Ctc::load(cfg.clone(), vb, Device::Cpu).expect("tiny model loads");
        (model, map)
    }

    #[test]
    fn base_frame_rate_is_exactly_20ms() {
        let cfg = Config::base_960h();
        assert_eq!(cfg.total_stride(), 320);
        assert!((cfg.frame_secs(16_000) - 0.02).abs() < 1e-12, "{}", cfg.frame_secs(16_000));
    }

    #[test]
    fn output_frames_matches_the_conv_arithmetic() {
        let cfg = Config::base_960h();
        // One second of 16 kHz audio through a 320-sample stride.
        let f = cfg.output_frames(16_000);
        assert!((49..=50).contains(&f), "expected ~49 frames for 1 s, got {f}");
        // Too short for the first kernel at all.
        assert_eq!(cfg.output_frames(4), 0);
    }

    #[test]
    fn logits_have_one_row_per_frame() {
        let cfg = tiny();
        let (model, _map) = build(&cfg);
        let samples = vec![0.01f32; 8000];
        let out = model.logits(&samples).expect("forward runs");
        let (frames, vocab) = out.dims2().expect("2-D logits");
        assert_eq!(vocab, cfg.vocab);
        assert_eq!(frames, cfg.output_frames(samples.len()), "frame count must match the strides");
    }

    #[test]
    fn emissions_are_log_probabilities() {
        let cfg = tiny();
        let (model, _map) = build(&cfg);
        let e = model.emissions(&vec![0.01f32; 8000], 16_000).expect("emissions");
        assert_eq!(e.frames * e.vocab, e.data.len());
        assert!((e.frame_secs - 0.02).abs() < 1e-12);
        // Every row must sum to 1 in probability space.
        for f in 0..e.frames {
            let row = &e.data[f * e.vocab..(f + 1) * e.vocab];
            assert!(row.iter().all(|v| *v <= 1e-6), "log-probs must be <= 0: {row:?}");
            let total: f32 = row.iter().map(|v| v.exp()).sum();
            assert!((total - 1.0).abs() < 1e-3, "row {f} sums to {total}");
        }
    }

    #[test]
    fn audio_too_short_for_one_frame_is_an_error() {
        let (model, _map) = build(&tiny());
        assert!(model.logits(&[0.0; 8]).is_err());
    }

    #[test]
    fn emissions_feed_the_aligner() {
        // The contract that matters: this model's output is consumable by the
        // forced aligner without any adaptation between them.
        let (model, _map) = build(&tiny());
        let e = model.emissions(&vec![0.01f32; 16_000], 16_000).expect("emissions");
        let spans = super::super::align::forced_align(&e, &[4, 5, 6], 0).expect("aligns");
        assert_eq!(spans.len(), 3);
        for pair in spans.windows(2) {
            assert!(pair[0].end <= pair[1].start);
        }
    }

    #[test]
    fn pre_norm_variant_also_runs() {
        let mut cfg = tiny();
        cfg.stable_layer_norm = true;
        cfg.conv_group_norm = false;
        let (model, _map) = build(&cfg);
        let out = model.logits(&vec![0.01f32; 8000]).expect("large-style forward runs");
        assert_eq!(out.dims2().expect("2-D").1, cfg.vocab);
    }
}
