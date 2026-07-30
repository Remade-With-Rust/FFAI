//! VITS inference on candle: phoneme ids → waveform, running the SAME voice
//! files piper runs (weights extracted by `corpora/refs/dump_piper_weights.py`
//! into the model cache — data, not code).
//!
//! Four stages, mirroring the graph piper exports, each independently
//! callable so the stage oracles (`examples/vits_oracle.rs`) can pin every
//! boundary against onnxruntime's own intermediates:
//!
//! | Stage | Contract | Oracle tensor |
//! |---|---|---|
//! | [`Vits::text_encoder`] | ids → prior stats + hidden | `m_p`, `logs_p` |
//! | [`Vits::durations`] | hidden → frames per phoneme | `w_ceil` (integer-exact) |
//! | [`Vits::flow_reverse`] | expanded prior → latent | `dec_in` |
//! | [`Vits::decode`] | latent → waveform | `audio` |
//!
//! Conv geometry (dilations, strides, pads, groups) is READ from
//! `vits-graph.json` — extracted from the ONNX graph, never assumed — and
//! two export quirks are reproduced deliberately: the duration predictor's
//! first ConvFlow (`dp.flows.1`) is skipped at inference (the VITS
//! `flows[:-2] + [flows[-1]]` reversal, confirmed present in the graph), and
//! only the TOTAL length is clamped, not per-phoneme durations.
//!
//! Determinism: all noise flows through a seeded xorshift/Box–Muller
//! generator; `noise_scale = noise_w = 0` bypasses it entirely, which is the
//! configuration the acoustic oracle pins.

use std::collections::HashMap;
use std::path::Path;

use ffai_core::candle::{DType, Device, IndexOp, Tensor};
use ffai_core::error::{Error, Result};

use super::phoneme_ids::PhonemeIdMap;

const HIDDEN: usize = 192;
const HEADS: usize = 2;
const WINDOW: usize = 4;
const N_LAYERS: usize = 6;
const SPLINE_BINS: usize = 10;
const SPLINE_TAIL: f64 = 5.0;
const MIN_BIN: f64 = 1e-3;
const MIN_DERIV: f64 = 1e-3;

/// Synthesis knobs, defaulting to the VOICE CONFIG's own values at load.
#[derive(Debug, Clone, Copy)]
pub struct SynthesisOptions {
    pub noise_scale: f64,
    pub length_scale: f64,
    pub noise_w: f64,
    /// Seed for all sampled noise. Same seed + same input = same bytes —
    /// the determinism piper itself does not offer (mission plan §3.3).
    pub seed: u64,
}

pub struct Vits {
    w: HashMap<String, Tensor>,
    geom: HashMap<String, ConvGeom>,
    device: Device,
    /// The decoder pre-flattened into Vec domain (M-T3: the decoder was
    /// 82.7 % of synthesis and its cost was half plumbing).
    flat_dec: super::decoder_kernels::FlatDecoder,
    /// The flat flow — THE DEFAULT since the six-whys descent: memcpy
    /// im2col + cached GEMM weight tensors + fast-exp gates won 42/45
    /// paired rounds at 1.65–2.48× over candle. Stores weight TENSORS only
    /// (the double-copy version flipped the footprint gate once).
    /// `FFAI_CANDLE_FLOW=1` switches back for A/B.
    flat_flow: Option<super::decoder_kernels::FlatFlow>,
    /// Per-layer fused QKV projection (weights concatenated at load): one
    /// conv dispatch instead of three, identical arithmetic.
    fused_qkv: Vec<(Tensor, Tensor)>,
    pub sample_rate: u32,
    pub id_map: PhonemeIdMap,
    pub defaults: SynthesisOptions,
}

#[derive(Debug, Clone, Copy, Default)]
struct ConvGeom {
    transpose: bool,
    stride: usize,
    pad: usize,
    dilation: usize,
}

impl Vits {
    /// Load a voice **straight from the files piper ships**: the `.onnx` and
    /// its `.onnx.json`, with no conversion step, no Python, and no ONNX
    /// runtime — [`super::onnx`] lifts the weights and the convolution
    /// geometry out of the graph.
    ///
    /// This is the path a crates.io consumer takes: the manifest fetches both
    /// files from `rhasspy/piper-voices` and this reads them. Byte-compared
    /// against the Python converter's output by
    /// `examples/onnx_vs_safetensors.rs`.
    pub fn load_onnx(onnx: &Path, config_json: &Path) -> Result<Self> {
        let device = Device::Cpu;
        // Scoped so the 63 MB of file bytes are freed before the tensors are
        // built — the peak, not the steady state, is what the footprint gate
        // measures, and holding both doubles it for no reason.
        let recovered = {
            let bytes = std::fs::read(onnx)?;
            super::onnx::recover(super::onnx::parse(&bytes)?)?
        };

        let mut w = HashMap::new();
        for (name, init) in recovered.tensors {
            let t = Tensor::from_vec(init.data, init.dims, &device).e()?;
            w.insert(name, t);
        }
        let geom = recovered
            .geometry
            .into_iter()
            .map(|(k, g)| {
                (
                    k,
                    ConvGeom {
                        transpose: g.transpose,
                        stride: g.stride,
                        pad: g.pad,
                        dilation: g.dilation,
                    },
                )
            })
            .collect();
        let config_text = std::fs::read_to_string(config_json)?;
        Self::from_parts(w, geom, config_text, device)
    }

    /// Load from a converted-voice directory: `vits.safetensors`,
    /// `vits-graph.json`, `voice-config.json` (see dump_piper_weights.py).
    /// Kept as the in-repo path and as the oracle the ONNX loader is gated
    /// against.
    pub fn load(dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let w = ffai_core::candle::safetensors::load(dir.join("vits.safetensors"), &device)
            .map_err(|e| Error::Model(format!("vits weights: {e}")))?;

        let graph: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vits-graph.json"))?)
                .map_err(|e| Error::Model(format!("vits-graph.json: {e}")))?;
        let mut geom = HashMap::new();
        if let Some(map) = graph.get("conv_attrs").and_then(|v| v.as_object()) {
            for (name, a) in map {
                let get = |k: &str, d: usize| {
                    a.get(k)
                        .and_then(|v| v.as_array())
                        .and_then(|v| v.first())
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .unwrap_or(d)
                };
                geom.insert(
                    name.clone(),
                    ConvGeom {
                        transpose: a.get("op").and_then(|v| v.as_str()) == Some("ConvTranspose"),
                        stride: get("strides", 1),
                        pad: get("pads", 0),
                        dilation: get("dilations", 1),
                    },
                );
            }
        }

        let config_text = std::fs::read_to_string(dir.join("voice-config.json"))?;
        Self::from_parts(w, geom, config_text, device)
    }

    /// The shared tail of both loaders: everything downstream of "we have
    /// named weights, geometry, and the voice config".
    fn from_parts(
        w: HashMap<String, Tensor>,
        geom: HashMap<String, ConvGeom>,
        config_text: String,
        device: Device,
    ) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_str(&config_text)
            .map_err(|e| Error::Model(format!("voice-config.json: {e}")))?;
        let sample_rate = config
            .pointer("/audio/sample_rate")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Model("voice config has no audio.sample_rate".into()))?
            as u32;
        let knob = |k: &str, d: f64| config.pointer(&format!("/inference/{k}")).and_then(|v| v.as_f64()).unwrap_or(d);
        let defaults = SynthesisOptions {
            noise_scale: knob("noise_scale", 0.667),
            length_scale: knob("length_scale", 1.0),
            noise_w: knob("noise_w", 0.8),
            seed: 0,
        };
        let id_map = PhonemeIdMap::from_json(&config_text)?;

        let flat_dec = {
            let get = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
                let t = w
                    .get(name)
                    .ok_or_else(|| Error::Model(format!("vits weight `{name}` missing")))?;
                Ok((t.flatten_all().e()?.to_vec1().e()?, t.dims().to_vec()))
            };
            let has_bias = |name: &str| w.contains_key(&format!("{name}.bias"));
            let geom_of = |name: &str| -> (usize, usize, usize) {
                let g = geom.get(name).copied().unwrap_or_default();
                (g.pad, g.stride.max(1), g.dilation.max(1))
            };
            super::decoder_kernels::FlatDecoder::from_weights(&get, &has_bias, &geom_of)?
        };
        let flat_flow = {
            let get = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
                let t = w
                    .get(name)
                    .ok_or_else(|| Error::Model(format!("vits weight `{name}` missing")))?;
                Ok((t.flatten_all().e()?.to_vec1().e()?, t.dims().to_vec()))
            };
            let geom_of = |name: &str| -> (usize, usize, usize) {
                let g = geom.get(name).copied().unwrap_or_default();
                (g.pad, g.stride.max(1), g.dilation.max(1))
            };
            Some(super::decoder_kernels::FlatFlow::from_weights(&get, &geom_of, HIDDEN)?)
        };
        let fused_qkv = {
            let mut v = Vec::new();
            for i in 0..N_LAYERS {
                let base = format!("enc_p.encoder.attn_layers.{i}");
                let t = |n: &str| {
                    w.get(&format!("{base}.{n}"))
                        .ok_or_else(|| Error::Model(format!("missing {base}.{n}")))
                };
                let wq = Tensor::cat(
                    &[t("conv_q.weight")?, t("conv_k.weight")?, t("conv_v.weight")?],
                    0,
                )
                .e()?;
                let bq = Tensor::cat(
                    &[t("conv_q.bias")?, t("conv_k.bias")?, t("conv_v.bias")?],
                    0,
                )
                .e()?;
                v.push((wq, bq));
            }
            v
        };

        Ok(Vits { w, geom, device, flat_dec, flat_flow, fused_qkv, sample_rate, id_map, defaults })
    }

    /// Full synthesis: interleaved phoneme ids → mono f32 samples.
    pub fn synthesize_ids(&self, ids: &[i64], opts: &SynthesisOptions) -> Result<Vec<f32>> {
        let mut rng = GaussRng::new(opts.seed);
        let (m_p, logs_p, hidden) = self.text_encoder(ids)?;
        let w_ceil = self.durations(&hidden, opts.noise_w, opts.length_scale, &mut rng)?;
        let (m_exp, logs_exp) = self.expand_prior(&m_p, &logs_p, &w_ceil)?;
        let z_p = if opts.noise_scale > 0.0 {
            let noise = rng.tensor(m_exp.shape().dims(), &self.device)?;
            (&m_exp + noise.mul(&logs_exp.exp()?)?.affine(opts.noise_scale, 0.0)?)?
        } else {
            m_exp
        };
        let z = self.flow_reverse(&z_p)?;
        self.decode(&z)
    }

    // ---------------------------------------------------------------- stage 1

    /// Text encoder: ids → (m_p, logs_p, hidden), each `[1, 192, T]`.
    pub fn text_encoder(&self, ids: &[i64]) -> Result<(Tensor, Tensor, Tensor)> {
        let t_len = ids.len();
        let idx = Tensor::from_slice(ids, (t_len,), &self.device).e()?;
        let emb = self.t("enc_p.emb.weight")?;
        // [T, 192] * sqrt(hidden) -> [1, 192, T]
        let mut x = emb
            .index_select(&idx.to_dtype(DType::U32).e()?, 0)
            .e()?
            .affine((HIDDEN as f64).sqrt(), 0.0)
            .e()?
            .transpose(0, 1)
            .e()?
            .unsqueeze(0)
            .e()?;

        for i in 0..N_LAYERS {
            let base = format!("enc_p.encoder.attn_layers.{i}");
            let y = self.rel_attention(&base, i, &x, t_len)?;
            x = self.layer_norm_named(&format!("enc_p.encoder.norm_layers_1.{i}"), &(&x + y).e()?)?;
            let y = self.ffn(&format!("enc_p.encoder.ffn_layers.{i}"), &x)?;
            x = self.layer_norm_named(&format!("enc_p.encoder.norm_layers_2.{i}"), &(&x + y).e()?)?;
        }
        let stats = self.conv("enc_p.proj", &x)?;
        let m_p = stats.narrow(1, 0, HIDDEN).e()?;
        let logs_p = stats.narrow(1, HIDDEN, HIDDEN).e()?;
        Ok((m_p, logs_p, x))
    }

    /// Multi-head attention with windowed relative position embeddings —
    /// the exact arithmetic of VITS `attentions.MultiHeadAttention`
    /// (window 4, both key and value relative embeddings).
    fn rel_attention(&self, base: &str, layer: usize, x: &Tensor, t_len: usize) -> Result<Tensor> {
        let head_dim = HIDDEN / HEADS;
        let scale = 1.0 / (head_dim as f64).sqrt();
        // One fused QKV projection (weights concatenated at load) — the
        // narrows are views into its output, arithmetic identical.
        let (wf, bf) = &self.fused_qkv[layer];
        let qkv = x
            .conv1d(wf, 0, 1, 1, 1)
            .e()?
            .broadcast_add(&bf.reshape((1, 3 * HIDDEN, 1)).e()?)
            .e()?;

        // FLAT fused relative attention. The tensor formulation needed ~60
        // ops/layer of pad/reshape/cat purely to express two index-shifted
        // sums; in flat form they are the indices themselves:
        //   scores[h,i,j] = (q[h,i]·k[h,j] + q[h,i]·relk[j−i+T−1])·s
        //   out[h,i]     = Σⱼ p[h,i,j]·(v[h,j] + relv[j−i+T−1])
        // Work is ~12 M MACs at T≈90 — the op tax was the stage, not the
        // arithmetic. m_p oracle gates it (reassociation-level deltas only).
        let hd = HIDDEN / HEADS;
        let qkv_v: Vec<f32> = qkv.flatten_all().e()?.to_vec1().e()?; // [3*192][T]
        let relk = self.rel_table(&format!("{base}.emb_rel_k"), t_len)?; // [(2T-1)*hd]
        let relv = self.rel_table(&format!("{base}.emb_rel_v"), t_len)?;

        // Contiguous [h][t][d] copies of q/k/v (one transpose pass each) —
        // the first version used per-row Vec allocations and nested-Vec
        // indexing, and was 2× SLOWER than the tensor path it replaced.
        // Layout is the speed; the arithmetic was never the problem.
        let mut q = vec![0f32; HEADS * t_len * hd];
        let mut kk = vec![0f32; HEADS * t_len * hd];
        let mut vv = vec![0f32; HEADS * t_len * hd];
        for h in 0..HEADS {
            for d in 0..hd {
                let cq = (h * hd + d) * t_len;
                let ck = (HIDDEN + h * hd + d) * t_len;
                let cv = (2 * HIDDEN + h * hd + d) * t_len;
                for t in 0..t_len {
                    q[(h * t_len + t) * hd + d] = (qkv_v[cq + t] as f64 * scale) as f32;
                    kk[(h * t_len + t) * hd + d] = qkv_v[ck + t];
                    vv[(h * t_len + t) * hd + d] = qkv_v[cv + t];
                }
            }
        }

        let mut out = vec![0f32; HIDDEN * t_len];
        let mut p = vec![0f32; t_len];
        let mut orow = vec![0f32; hd];
        for h in 0..HEADS {
            for i in 0..t_len {
                let qr = &q[(h * t_len + i) * hd..(h * t_len + i + 1) * hd];
                for (j, pj) in p.iter_mut().enumerate() {
                    let kr = &kk[(h * t_len + j) * hd..(h * t_len + j + 1) * hd];
                    let rr = &relk[(j + t_len - 1 - i) * hd..(j + t_len - i) * hd];
                    let mut s = 0f32;
                    for d in 0..hd {
                        s += qr[d] * (kr[d] + rr[d]);
                    }
                    *pj = s;
                }
                // Softmax over the row, in place.
                let m = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for pj in p.iter_mut() {
                    *pj = (*pj - m).exp();
                    sum += *pj;
                }
                let inv = 1.0 / sum;
                // out[h,i] = Σⱼ p·(v[j] + relv[j−i+T−1]), accumulated in a
                // local row, scattered once.
                orow.iter_mut().for_each(|o| *o = 0.0);
                for (j, pj) in p.iter().enumerate() {
                    let w = pj * inv;
                    let vr = &vv[(h * t_len + j) * hd..(h * t_len + j + 1) * hd];
                    let rr = &relv[(j + t_len - 1 - i) * hd..(j + t_len - i) * hd];
                    for d in 0..hd {
                        orow[d] += w * (vr[d] + rr[d]);
                    }
                }
                for d in 0..hd {
                    out[(h * hd + d) * t_len + i] = orow[d];
                }
            }
        }
        let out = Tensor::from_vec(out, (1, HIDDEN, t_len), &self.device).e()?;
        self.conv(&format!("{base}.conv_o"), &out)
    }

    /// The `[(2T-1) × head_dim]` relative table, flat: window-4 learned
    /// embeddings centered, zeros outside — an index gather, not a pad/cat.
    fn rel_table(&self, name: &str, t_len: usize) -> Result<Vec<f32>> {
        let emb = self.t(name)?; // [1, 2*WINDOW+1, hd]
        let hd = emb.dim(2).e()?;
        let ev: Vec<f32> = emb.flatten_all().e()?.to_vec1().e()?;
        let span = 2 * t_len - 1;
        let mut rows = vec![0f32; span * hd];
        for r in 0..span {
            let centered = r as isize - (t_len as isize - 1) + WINDOW as isize;
            if (0..(2 * WINDOW + 1) as isize).contains(&centered) {
                let base = centered as usize * hd;
                rows[r * hd..(r + 1) * hd].copy_from_slice(&ev[base..base + hd]);
            }
        }
        Ok(rows)
    }

    fn ffn(&self, base: &str, x: &Tensor) -> Result<Tensor> {
        // The graph pads outside its Conv nodes (ONNX export artifact); the
        // same arithmetic is conv1d's own padding parameter — two fewer
        // pad allocations+copies per layer.
        let conv = |name: &str, x: &Tensor| -> Result<Tensor> {
            let w = self.t(&format!("{name}.weight"))?;
            let b = self.t(&format!("{name}.bias"))?;
            let c = b.dim(0).e()?;
            x.conv1d(w, 1, 1, 1, 1).e()?.broadcast_add(&b.reshape((1, c, 1)).e()?).e()
        };
        let h = conv(&format!("{base}.conv_1"), x)?.relu().e()?;
        conv(&format!("{base}.conv_2"), &h)
    }

    // ---------------------------------------------------------------- stage 2

    /// Stochastic duration predictor, reverse mode → frames per phoneme.
    ///
    /// Two graph-confirmed quirks: `dp.flows.1` never runs (the VITS
    /// reversal drops it), and durations are NOT clamped — only the total
    /// length is, downstream.
    pub fn durations(
        &self,
        hidden: &Tensor,
        noise_w: f64,
        length_scale: f64,
        rng: &mut GaussRng,
    ) -> Result<Vec<u32>> {
        let t_len = hidden.dim(2).e()?;
        // Conditioning: pre -> DDSConv -> proj.
        let g = self.conv("dp.pre", hidden)?;
        let g = self.dds_conv("dp.convs", &g, None)?;
        let g = self.conv("dp.proj", &g)?;

        // z starts as noise * noise_w over 2 channels.
        let mut z = if noise_w > 0.0 {
            rng.tensor(&[1, 2, t_len], &self.device)?.affine(noise_w, 0.0).e()?
        } else {
            Tensor::zeros((1, 2, t_len), DType::F32, &self.device).e()?
        };

        // Reversed flow stack: Flip·CF7·Flip·CF5·Flip·CF3·Flip·EA0.
        for flow in [7usize, 5, 3] {
            z = z.flip(&[1]).e()?;
            z = self.conv_flow_reverse(&format!("dp.flows.{flow}"), &z, &g)?;
        }
        z = z.flip(&[1]).e()?;
        // ElementwiseAffine reverse: (z - m) * exp(-logs). The exporter
        // folded exp(-logs) into a constant; the converter names it.
        let m = self.t("dp.flows.0.m")?.reshape((1, 2, 1)).e()?;
        let exp_neg_logs = self.t("dp.flows.0.exp_neg_logs")?.reshape((1, 2, 1)).e()?;
        z = z.broadcast_sub(&m).e()?.broadcast_mul(&exp_neg_logs).e()?;

        let logw: Vec<f32> = z.i((0, 0)).e()?.to_vec1().e()?;
        Ok(logw
            .iter()
            .map(|lw| ((lw.exp() as f64) * length_scale).ceil().max(0.0) as u32)
            .collect())
    }

    /// DDSConv: depthwise-separable dilated conv stack with channel
    /// LayerNorms and exact-erf GELU (the graph uses Erf, not tanh-GELU).
    fn dds_conv(&self, base: &str, x: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let mut x = match g {
            Some(g) => (x + g).e()?,
            None => x.clone(),
        };
        for i in 0..3 {
            let y = self.conv(&format!("{base}.convs_sep.{i}"), &x)?;
            let y = self.layer_norm_named(&format!("{base}.norms_1.{i}"), &y)?;
            let y = gelu_erf(&y)?;
            let y = self.conv(&format!("{base}.convs_1x1.{i}"), &y)?;
            let y = self.layer_norm_named(&format!("{base}.norms_2.{i}"), &y)?;
            let y = gelu_erf(&y)?;
            x = (x + y).e()?;
        }
        Ok(x)
    }

    fn layer_norm_named(&self, base: &str, x: &Tensor) -> Result<Tensor> {
        let c = self.t(&format!("{base}.gamma"))?.dim(0).e()?;
        let mean = x.mean_keepdim(1).e()?;
        let centered = x.broadcast_sub(&mean).e()?;
        let var = centered.sqr().e()?.mean_keepdim(1).e()?;
        let normed = centered.broadcast_div(&(var + 1e-5).e()?.sqrt().e()?).e()?;
        let gamma = self.t(&format!("{base}.gamma"))?.reshape((1, c, 1)).e()?;
        let beta = self.t(&format!("{base}.beta"))?.reshape((1, c, 1)).e()?;
        normed.broadcast_mul(&gamma).e()?.broadcast_add(&beta).e()
    }

    /// ConvFlow reverse: spline-transform the second channel conditioned on
    /// the first (rational-quadratic, linear tails, 10 bins, bound 5).
    fn conv_flow_reverse(&self, base: &str, z: &Tensor, g: &Tensor) -> Result<Tensor> {
        let t_len = z.dim(2).e()?;
        let z0 = z.narrow(1, 0, 1).e()?;
        let z1 = z.narrow(1, 1, 1).e()?;
        let h = self.conv(&format!("{base}.pre"), &z0)?;
        let h = self.dds_conv(&format!("{base}.convs"), &h, Some(g))?;
        let h = self.conv(&format!("{base}.proj"), &h)?; // [1, 3*bins-1, T]

        let filt_scale = 1.0 / (HIDDEN as f64).sqrt();
        let params: Vec<Vec<f32>> = h.i(0).e()?.to_vec2().e()?; // [29][T]
        let z1v: Vec<f32> = z1.i((0, 0)).e()?.to_vec1().e()?;
        let mut out = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let widths: Vec<f64> =
                (0..SPLINE_BINS).map(|b| params[b][t] as f64 * filt_scale).collect();
            let heights: Vec<f64> = (SPLINE_BINS..2 * SPLINE_BINS)
                .map(|b| params[b][t] as f64 * filt_scale)
                .collect();
            let derivs: Vec<f64> =
                (2 * SPLINE_BINS..3 * SPLINE_BINS - 1).map(|b| params[b][t] as f64).collect();
            out.push(rq_spline_inverse(z1v[t] as f64, &widths, &heights, &derivs) as f32);
        }
        let z1_new = Tensor::from_vec(out, (1, 1, t_len), &self.device).e()?;
        Tensor::cat(&[&z0, &z1_new], 1).e()
    }

    // ---------------------------------------------------------------- stage 3

    /// Repeat each phoneme's prior stats by its duration (the attention-path
    /// expansion of the graph, computed directly).
    pub fn expand_prior(
        &self,
        m_p: &Tensor,
        logs_p: &Tensor,
        w_ceil: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        let mut indices: Vec<u32> = Vec::new();
        for (i, &w) in w_ceil.iter().enumerate() {
            for _ in 0..w {
                indices.push(i as u32);
            }
        }
        if indices.is_empty() {
            // The graph clamps TOTAL length to >= 1 (the lone Clip node).
            indices.push(0);
        }
        let n = indices.len();
        let idx = Tensor::from_vec(indices, (n,), &self.device).e()?;
        let m = m_p.index_select(&idx, 2).e()?;
        let logs = logs_p.index_select(&idx, 2).e()?;
        Ok((m, logs))
    }

    /// Residual coupling flow, reversed: z_p → z.
    ///
    /// The flat fused-gate version exists (`FFAI_FLAT_FLOW=1`) and is
    /// oracle-exact, but the interleaved paired A/B scored it a WASH —
    /// 0.99×, 4/15 rounds — because this stage's cost is its convolutions,
    /// not op dispatch. candle stays the default until a quiet-machine
    /// retest or a faster short-length conv kernel changes the verdict.
    pub fn flow_reverse(&self, z_p: &Tensor) -> Result<Tensor> {
        // Flat is the default (42/45 paired wins, §6.14);
        // FFAI_CANDLE_FLOW=1 restores the candle path for A/B.
        if let Some(flat_flow) = &self.flat_flow {
            if std::env::var("FFAI_CANDLE_FLOW").is_err() {
                let len = z_p.dim(2).e()?;
                let mut zv: Vec<f32> = z_p.flatten_all().e()?.to_vec1().e()?;
                flat_flow.run(&mut zv, len)?;
                return Tensor::from_vec(zv, (1, HIDDEN, len), &self.device).e();
            }
        }
        let half = HIDDEN / 2;
        let mut z = z_p.clone();
        for flow in [6usize, 4, 2, 0] {
            z = z.flip(&[1]).e()?;
            let base = format!("flow.flows.{flow}");
            let x0 = z.narrow(1, 0, half).e()?;
            let x1 = z.narrow(1, half, half).e()?;
            let h = self.conv(&format!("{base}.pre"), &x0)?;
            let h = self.wavenet(&base, &h)?;
            let m = self.conv(&format!("{base}.post"), &h)?;
            let x1 = (x1 - m).e()?; // mean-only coupling, reverse
            z = Tensor::cat(&[&x0, &x1], 1).e()?;
        }
        Ok(z)
    }

    /// The WN block: 4 gated non-dilated k5 layers with residual/skip 1x1s.
    fn wavenet(&self, base: &str, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        let mut skip: Option<Tensor> = None;
        for i in 0..4 {
            let a = self.conv(&format!("{base}.enc.in_layers.{i}"), &x)?; // [1, 384, T]
            let t_half = a.narrow(1, 0, HIDDEN).e()?.tanh().e()?;
            let s_half = candle_nn::ops::sigmoid(&a.narrow(1, HIDDEN, HIDDEN).e()?).e()?;
            let acts = (t_half * s_half).e()?;
            let rs = self.conv(&format!("{base}.enc.res_skip_layers.{i}"), &acts)?;
            if i < 3 {
                x = (x + rs.narrow(1, 0, HIDDEN).e()?).e()?;
                let s = rs.narrow(1, HIDDEN, HIDDEN).e()?;
                skip = Some(match skip {
                    Some(acc) => (acc + s).e()?,
                    None => s,
                });
            } else {
                skip = Some(match skip {
                    Some(acc) => (acc + rs).e()?,
                    None => rs,
                });
            }
        }
        Ok(skip.expect("4 layers"))
    }

    // ---------------------------------------------------------------- stage 4

    /// HiFi-GAN generator (piper's compact variant: 3 ups, 9 two-conv
    /// resblocks, per-conv geometry from the graph JSON), running entirely
    /// in the flat Vec domain — see [`super::decoder_kernels::FlatDecoder`].
    /// `FFAI_CANDLE_DEC=1` switches to the candle-op reference path for A/B.
    pub fn decode(&self, z: &Tensor) -> Result<Vec<f32>> {
        if std::env::var("FFAI_CANDLE_DEC").is_err() {
            let len = z.dim(2).e()?;
            let zv: Vec<f32> = z.flatten_all().e()?.to_vec1().e()?;
            return Ok(self.flat_dec.run(&zv, len));
        }
        let mut x = self.conv("dec.conv_pre", z)?;
        for up in 0..3 {
            x = leaky_relu(&x, 0.1)?;
            x = self.conv(&format!("dec.ups.{up}"), &x)?;
            let mut acc: Option<Tensor> = None;
            for j in 0..3 {
                let rb = format!("dec.resblocks.{}", up * 3 + j);
                let mut y = x.clone();
                for c in 0..2 {
                    let yt = leaky_relu(&y, 0.1)?;
                    let yt = self.conv(&format!("{rb}.convs.{c}"), &yt)?;
                    y = (y + yt).e()?;
                }
                acc = Some(match acc {
                    Some(a) => (a + y).e()?,
                    None => y,
                });
            }
            x = acc.expect("3 resblocks").affine(1.0 / 3.0, 0.0).e()?;
        }
        // Final activation uses torch's default slope (the lone 0.01 in the
        // graph), then a bias-free projection and tanh.
        x = leaky_relu(&x, 0.01)?;
        x = self.conv("dec.conv_post", &x)?;
        let audio = x.tanh().e()?.flatten_all().e()?;
        audio.to_vec1().e()
    }

    // ------------------------------------------------------------- plumbing

    fn t(&self, name: &str) -> Result<&Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| Error::Model(format!("vits weight `{name}` missing from safetensors")))
    }

    /// Conv/ConvTranspose by canonical name, geometry from the graph JSON,
    /// bias applied when present (conv_post has none).
    ///
    /// Decoder-stage convs route through the cache-blocked direct kernels
    /// ([`super::decoder_kernels`]) — profiled 82.7 % of synthesis, and
    /// candle's conv1d degrades with length (99 → 27 GF/s) as its working
    /// set leaves cache. `FFAI_CANDLE_CONV=1` switches back for A/B; the
    /// kernels' scalar twin IS candle's conv (see their tests).
    fn conv(&self, name: &str, x: &Tensor) -> Result<Tensor> {
        let weight = self.t(&format!("{name}.weight"))?;
        let g = self.geom.get(name).copied().unwrap_or(ConvGeom {
            transpose: false,
            stride: 1,
            pad: 0,
            dilation: 1,
        });
        let bias = self.w.get(&format!("{name}.bias"));
        // Routing, settled by measurement per stage: dec.* on the direct
        // kernels (4× there), dp's DEPTHWISE convs on the serial depthwise
        // kernel (395 → 254 ms per 20 sentences), and enc_p/flow REVERTED to
        // candle — universal routing was tried and enc_p regressed 1.35×
        // (serial 1×1s lose to candle's threaded matmul at 192×192×T) while
        // flow measured flat. `FFAI_CANDLE_CONV=1` restores candle end to end.
        let use_direct = (name.starts_with("dec.") || name.starts_with("dp."))
            && std::env::var("FFAI_CANDLE_CONV").is_err()
            && (g.transpose || g.stride == 1);
        if use_direct {
            let in_per_group = weight.dim(1).e()?;
            let x_ch = x.dim(1).e()?;
            if g.transpose {
                return super::decoder_kernels::conv_transpose1d_direct(
                    x, weight, bias, g.pad, g.stride,
                );
            }
            if in_per_group == 1 && x_ch == weight.dim(0).e()? {
                let (_, c, len) = x.dims3().e()?;
                let xv: Vec<f32> = x.flatten_all().e()?.to_vec1().e()?;
                let wv: Vec<f32> = weight.flatten_all().e()?.to_vec1().e()?;
                let bvv: Option<Vec<f32>> = match bias {
                    Some(b) => Some(b.flatten_all().e()?.to_vec1().e()?),
                    None => None,
                };
                let k = weight.dim(2).e()?;
                let (out, l_out) = super::decoder_kernels::conv1d_depthwise_flat(
                    &xv,
                    c,
                    len,
                    &wv,
                    bvv.as_deref(),
                    k,
                    g.pad,
                    g.dilation,
                );
                return Tensor::from_vec(out, (1, c, l_out), &self.device).e();
            }
            if x_ch / in_per_group == 1 {
                return super::decoder_kernels::conv1d_direct(x, weight, bias, g.pad, g.dilation);
            }
        }
        let groups = if g.transpose {
            1
        } else {
            let in_per_group = weight.dim(1).e()?;
            x.dim(1).e()? / in_per_group
        };
        let y = if g.transpose {
            x.conv_transpose1d(weight, g.pad, 0, g.stride, g.dilation, 1).e()?
        } else {
            x.conv1d(weight, g.pad, g.stride, g.dilation, groups).e()?
        };
        match bias {
            Some(b) => {
                let c = b.dim(0).e()?;
                y.broadcast_add(&b.reshape((1, c, 1)).e()?).e()
            }
            None => Ok(y),
        }
    }
}

// ------------------------------------------------------------------- helpers

/// Bridge candle errors into ffai errors without a From impl in ffai-core.
trait CandleExt<T> {
    fn e(self) -> Result<T>;
}
impl<T> CandleExt<T> for ffai_core::candle::Result<T> {
    fn e(self) -> Result<T> {
        self.map_err(|e| Error::Model(format!("vits: {e}")))
    }
}

fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    let pos = x.relu().e()?;
    let neg = (x - &pos).e()?.affine(slope, 0.0).e()?;
    (pos + neg).e()
}

fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    x.gelu_erf().e()
}

/// `[h, T, 2T-1]` relative logits → `[h, T, T]` absolute (the VITS pad /
/// reshape / slice trick, verified by the hand example in the tests).
fn relative_to_absolute(x: &Tensor) -> Result<Tensor> {
    let (h, t, _span) = x.dims3().e()?;
    let x = x.pad_with_zeros(2, 0, 1).e()?; // [h,T,2T]
    let flat = x.reshape((h, t * 2 * t)).e()?;
    let flat = flat.pad_with_zeros(1, 0, t - 1).e()?;
    let x = flat.reshape((h, t + 1, 2 * t - 1)).e()?;
    x.narrow(1, 0, t).e()?.narrow(2, t - 1, t).e()
}

/// `[h, T, T]` attention weights → `[h, T, 2T-1]` relative positions.
fn absolute_to_relative(x: &Tensor) -> Result<Tensor> {
    let (h, t, _) = x.dims3().e()?;
    let x = x.pad_with_zeros(2, 0, t - 1).e()?; // [h,T,2T-1]
    let flat = x.reshape((h, t * (2 * t - 1))).e()?;
    let flat = flat.pad_with_zeros(1, t, 0).e()?;
    let x = flat.reshape((h, t, 2 * t)).e()?;
    x.narrow(2, 1, 2 * t - 1).e()
}

/// Inverse monotonic rational-quadratic spline with linear tails
/// (Durkan et al.; the transform `dp`'s ConvFlows apply to durations).
fn rq_spline_inverse(y: f64, uw: &[f64], uh: &[f64], ud: &[f64]) -> f64 {
    let nb = uw.len();
    if !(-SPLINE_TAIL..=SPLINE_TAIL).contains(&y) {
        return y; // linear tails: identity outside the bound
    }
    // Normalized bin widths/heights (softmax with a minimum), cumulative
    // over [-B, B].
    let norm = |u: &[f64]| -> Vec<f64> {
        let m = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = u.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        exps.iter().map(|e| MIN_BIN + (1.0 - MIN_BIN * nb as f64) * e / s).collect()
    };
    let widths = norm(uw);
    let heights = norm(uh);
    let span = 2.0 * SPLINE_TAIL;
    let mut cw = vec![-SPLINE_TAIL];
    let mut ch = vec![-SPLINE_TAIL];
    for b in 0..nb {
        cw.push(cw[b] + widths[b] * span);
        ch.push(ch[b] + heights[b] * span);
    }
    cw[nb] = SPLINE_TAIL;
    ch[nb] = SPLINE_TAIL;
    // Derivatives: boundary fixed at exactly 1 (that is what the linear-tail
    // padding constant achieves), interior via softplus.
    let deriv = |k: usize| -> f64 {
        if k == 0 || k == nb {
            1.0
        } else {
            MIN_DERIV + softplus(ud[k - 1])
        }
    };
    let bin = match ch[..=nb].partition_point(|&edge| edge <= y) {
        0 => 0,
        p => (p - 1).min(nb - 1),
    };
    let (x_k, y_k) = (cw[bin], ch[bin]);
    let w_k = cw[bin + 1] - x_k;
    let h_k = ch[bin + 1] - y_k;
    let delta = h_k / w_k;
    let (d0, d1) = (deriv(bin), deriv(bin + 1));
    let dy = y - y_k;
    let a = dy * (d0 + d1 - 2.0 * delta) + h_k * (delta - d0);
    let b = h_k * d0 - dy * (d0 + d1 - 2.0 * delta);
    let c = -delta * dy;
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let theta = 2.0 * c / (-b - disc.sqrt());
    x_k + theta * w_k
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Deterministic gaussian noise: xorshift64* + Box–Muller. Same seed, same
/// stream — the byte-stability contract behind `--seed`.
pub struct GaussRng {
    state: u64,
    spare: Option<f64>,
}

impl GaussRng {
    pub fn new(seed: u64) -> Self {
        GaussRng { state: seed.wrapping_mul(0x9E3779B97F4A7C15) | 1, spare: None }
    }

    fn uniform(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        (bits >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn gauss(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        self.spare = Some(r * s);
        r * c
    }

    fn tensor(&mut self, dims: &[usize], device: &Device) -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.gauss() as f32).collect();
        Tensor::from_vec(data, dims, device).e()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_absolute_conversions_match_hand_example() {
        // T = 2: relative logits [h=1, 2, 3] with distinct values; absolute
        // form picks (worked by hand from the VITS pad/reshape semantics):
        //   row 0 (query 0): rel positions {0:+0, +1} -> [b, c]
        //   row 1 (query 1): rel positions {-1, 0}    -> [d, e]
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 2, 3), &dev).unwrap();
        let abs = relative_to_absolute(&x).unwrap();
        let v: Vec<Vec<f32>> = abs.i(0).unwrap().to_vec2().unwrap();
        assert_eq!(v, vec![vec![2., 3.], vec![4., 5.]]);

        // And back: absolute [h,2,2] -> relative [h,2,3] puts each row's
        // values at its diagonal offsets, zeros elsewhere.
        let a = Tensor::from_vec(vec![7f32, 8., 9., 10.], (1, 2, 2), &dev).unwrap();
        let rel = absolute_to_relative(&a).unwrap();
        let v: Vec<Vec<f32>> = rel.i(0).unwrap().to_vec2().unwrap();
        assert_eq!(v, vec![vec![0., 7., 8.], vec![9., 10., 0.]]);
    }

    #[test]
    fn spline_inverse_is_monotonic_with_identity_tails() {
        let uw = vec![0.3; SPLINE_BINS];
        let uh: Vec<f64> = (0..SPLINE_BINS).map(|i| (i as f64) * 0.1 - 0.4).collect();
        let ud = vec![0.2; SPLINE_BINS - 1];
        // Tails: exactly identity.
        assert_eq!(rq_spline_inverse(7.3, &uw, &uh, &ud), 7.3);
        assert_eq!(rq_spline_inverse(-6.0, &uw, &uh, &ud), -6.0);
        // Interior: strictly monotonic, bounded by the tail interval.
        let mut prev = f64::NEG_INFINITY;
        for i in 0..200 {
            let y = -4.9 + (i as f64) * 0.049;
            let x = rq_spline_inverse(y, &uw, &uh, &ud);
            assert!(x > prev, "not monotonic at y={y}: {x} <= {prev}");
            assert!((-SPLINE_TAIL..=SPLINE_TAIL).contains(&x));
            prev = x;
        }
    }

    #[test]
    fn seeded_rng_is_reproducible_and_roughly_gaussian() {
        let mut a = GaussRng::new(42);
        let mut b = GaussRng::new(42);
        let xs: Vec<f64> = (0..10_000).map(|_| a.gauss()).collect();
        let ys: Vec<f64> = (0..10_000).map(|_| b.gauss()).collect();
        assert_eq!(xs, ys, "same seed must give the same stream");
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }
}
