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
//! first `ConvFlow` (`dp.flows.1`) is skipped at inference (the VITS
//! `flows[:-2] + [flows[-1]]` reversal, confirmed present in the graph), and
//! only the TOTAL length is clamped, not per-phoneme durations.
//!
//! Determinism: all noise flows through a seeded xorshift/Box–Muller
//! generator; `noise_scale = noise_w = 0` bypasses it entirely, which is the
//! configuration the acoustic oracle pins.

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
    clippy::cast_possible_wrap
)]

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
    /// Every conv's weight, bias and geometry under its canonical name, so a
    /// call is ONE map lookup instead of two `format!`s and three lookups
    /// (vein 9).
    conv_cache: HashMap<String, ConvEntry>,
    /// Every layer norm's gamma/beta under its base name (vein 9).
    ln_cache: HashMap<String, (Tensor, Tensor)>,
    /// The text encoder's per-layer weight names, formatted once (vein 9).
    enc_names: Vec<EncLayerNames>,
    /// `DDSConv` stacks' per-layer names by base (vein 9).
    dds_names: HashMap<String, [DdsLayerNames; 3]>,
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

/// One conv's operands, resolved once at load (vein 9).
struct ConvEntry {
    weight: Tensor,
    bias: Option<Tensor>,
    geom: ConvGeom,
}

/// The names one text-encoder layer touches, formatted at load (vein 9).
struct EncLayerNames {
    rel_k: String,
    rel_v: String,
    conv_o: String,
    norm1: String,
    ffn_conv1: String,
    ffn_conv2: String,
    norm2: String,
}

/// The names one `DDSConv` layer touches (vein 9).
struct DdsLayerNames {
    sep: String,
    norm1: String,
    one_by_one: String,
    norm2: String,
}

impl DdsLayerNames {
    fn for_base(base: &str) -> [Self; 3] {
        std::array::from_fn(|i| Self {
            sep: format!("{base}.convs_sep.{i}"),
            norm1: format!("{base}.norms_1.{i}"),
            one_by_one: format!("{base}.convs_1x1.{i}"),
            norm2: format!("{base}.norms_2.{i}"),
        })
    }
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
        crate::pin_gemm_blocking();
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
    /// `vits-graph.json`, `voice-config.json` (see `dump_piper_weights.py`).
    /// Kept as the in-repo path and as the oracle the ONNX loader is gated
    /// against.
    pub fn load(dir: &Path) -> Result<Self> {
        crate::pin_gemm_blocking();
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
                        .and_then(serde_json::Value::as_u64)
                        .map_or(d, |v| v as usize)
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
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Model("voice config has no audio.sample_rate".into()))?
            as u32;
        let knob = |k: &str, d: f64| {
            config
                .pointer(&format!("/inference/{k}"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(d)
        };
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
            Some(super::decoder_kernels::FlatFlow::from_weights(
                &get, &geom_of, HIDDEN,
            )?)
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
                    &[
                        t("conv_q.weight")?,
                        t("conv_k.weight")?,
                        t("conv_v.weight")?,
                    ],
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

        // Vein 9: resolve every conv and layer norm ONCE. Tensor clones are
        // refcounted handles, so the caches hold no second copy of a weight.
        let mut conv_cache = HashMap::new();
        let mut ln_cache = HashMap::new();
        for (key, t) in &w {
            if let Some(name) = key.strip_suffix(".weight")
                && t.rank() == 3
            {
                conv_cache.insert(
                    name.to_string(),
                    ConvEntry {
                        weight: t.clone(),
                        bias: w.get(&format!("{name}.bias")).cloned(),
                        geom: geom.get(name).copied().unwrap_or(ConvGeom {
                            transpose: false,
                            stride: 1,
                            pad: 0,
                            dilation: 1,
                        }),
                    },
                );
            }
            if let Some(base) = key.strip_suffix(".gamma")
                && let Some(beta) = w.get(&format!("{base}.beta"))
            {
                ln_cache.insert(base.to_string(), (t.clone(), beta.clone()));
            }
        }
        let enc_names = (0..N_LAYERS)
            .map(|i| {
                let base = format!("enc_p.encoder.attn_layers.{i}");
                let ffn = format!("enc_p.encoder.ffn_layers.{i}");
                EncLayerNames {
                    rel_k: format!("{base}.emb_rel_k"),
                    rel_v: format!("{base}.emb_rel_v"),
                    conv_o: format!("{base}.conv_o"),
                    norm1: format!("enc_p.encoder.norm_layers_1.{i}"),
                    ffn_conv1: format!("{ffn}.conv_1"),
                    ffn_conv2: format!("{ffn}.conv_2"),
                    norm2: format!("enc_p.encoder.norm_layers_2.{i}"),
                }
            })
            .collect();
        let dds_names = [
            "dp.convs",
            "dp.flows.3.convs",
            "dp.flows.5.convs",
            "dp.flows.7.convs",
        ]
        .into_iter()
        .map(|b| (b.to_string(), DdsLayerNames::for_base(b)))
        .collect();

        Ok(Self {
            w,
            geom,
            device,
            flat_dec,
            flat_flow,
            fused_qkv,
            conv_cache,
            ln_cache,
            enc_names,
            dds_names,
            sample_rate,
            id_map,
            defaults,
        })
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

    /// Text encoder: ids → (`m_p`, `logs_p`, hidden), each `[1, 192, T]`.
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
            // Names formatted once at load (vein 9): zero `format!` per layer.
            let names = &self.enc_names[i];
            let y = self.rel_attention(i, &x, t_len)?;
            x = self.layer_norm_named(&names.norm1, &(&x + y).e()?)?;
            let y = self.ffn_named(&names.ffn_conv1, &names.ffn_conv2, &x)?;
            x = self.layer_norm_named(&names.norm2, &(&x + y).e()?)?;
        }
        let stats = self.conv("enc_p.proj", &x)?;
        let m_p = stats.narrow(1, 0, HIDDEN).e()?;
        let logs_p = stats.narrow(1, HIDDEN, HIDDEN).e()?;
        Ok((m_p, logs_p, x))
    }

    /// Multi-head attention with windowed relative position embeddings —
    /// the exact arithmetic of VITS `attentions.MultiHeadAttention`
    /// (window 4, both key and value relative embeddings).
    fn rel_attention(&self, layer: usize, x: &Tensor, t_len: usize) -> Result<Tensor> {
        let names = &self.enc_names[layer];
        let head_dim = HIDDEN / HEADS;
        let scale = 1.0 / (head_dim as f64).sqrt();
        // One fused QKV projection (weights concatenated at load) — the
        // narrows are views into its output, arithmetic identical.
        // The fused QKV projection is a 1x1 conv, i.e. a plain GEMM — routed
        // through the matmul directly rather than candle's conv1d wrapper,
        // which measured a 1.71x tax on the sibling k=3 shapes.
        let (wf, bf) = &self.fused_qkv[layer];
        let qkv = if crate::asr::knobs::TTS_CANDLE_FFN.get() {
            x.conv1d(wf, 0, 1, 1, 1)
                .e()?
                .broadcast_add(&bf.reshape((1, 3 * HIDDEN, 1)).e()?)
                .e()?
        } else {
            super::decoder_kernels::conv1d_k1_gemm(x, wf, Some(bf), &self.device)?
        };

        // FLAT fused relative attention. The tensor formulation needed ~60
        // ops/layer of pad/reshape/cat purely to express two index-shifted
        // sums; in flat form they are the indices themselves:
        //   scores[h,i,j] = (q[h,i]·k[h,j] + q[h,i]·relk[j−i+T−1])·s
        //   out[h,i]     = Σⱼ p[h,i,j]·(v[h,j] + relv[j−i+T−1])
        // Work is ~12 M MACs at T≈90 — the op tax was the stage, not the
        // arithmetic. m_p oracle gates it (reassociation-level deltas only).
        let hd = HIDDEN / HEADS;
        let relk = self.rel_table(&names.rel_k, t_len)?; // [(2T-1)*hd]
        let relv = self.rel_table(&names.rel_v, t_len)?;

        // Contiguous [h][t][d] copies of q/k/v (one transpose pass each) —
        // the first version used per-row Vec allocations and nested-Vec
        // indexing, and was 2× SLOWER than the tensor path it replaced.
        // Layout is the speed; the arithmetic was never the problem.
        //
        // Read straight out of the projection's storage (vein 7) and written
        // into buffers that get no zero pass first (vein 15): every (h, t, d)
        // is assigned below.
        let qkv = if qkv.is_contiguous() {
            qkv
        } else {
            qkv.contiguous().e()?
        };
        let n_hd = HEADS * t_len * hd;
        let (q, kk, vv) = crate::buf::with_f32(&qkv, |qkv_v| {
            let q = crate::buf::filled_f32(n_hd, |q| {
                for h in 0..HEADS {
                    for d in 0..hd {
                        let cq = (h * hd + d) * t_len;
                        for t in 0..t_len {
                            q[(h * t_len + t) * hd + d] = (f64::from(qkv_v[cq + t]) * scale) as f32;
                        }
                    }
                }
            });
            let kk = crate::buf::filled_f32(n_hd, |kk| {
                for h in 0..HEADS {
                    for d in 0..hd {
                        let ck = (HIDDEN + h * hd + d) * t_len;
                        for t in 0..t_len {
                            kk[(h * t_len + t) * hd + d] = qkv_v[ck + t];
                        }
                    }
                }
            });
            let vv = crate::buf::filled_f32(n_hd, |vv| {
                for h in 0..HEADS {
                    for d in 0..hd {
                        let cv = (2 * HIDDEN + h * hd + d) * t_len;
                        for t in 0..t_len {
                            vv[(h * t_len + t) * hd + d] = qkv_v[cv + t];
                        }
                    }
                }
            });
            (q, kk, vv)
        })
        .e()?;

        // Each output row (h, i) reads all of k/v/relk/relv but writes only its
        // own hd values, so the (h, i) grid is embarrassingly parallel. It ran
        // serial while the convs on either side of it were candle-threaded,
        // which is why the attention half measured 3.94 cores and 16.4 GFLOP/s
        // against the feed-forward's 9.21 and 72.1 in the same stage.
        //
        // Accumulate in [h][i][d] so each task owns one contiguous chunk, then
        // transpose once into the [C][T] the output conv wants. Per-row
        // arithmetic and its summation order are untouched, so this is
        // BIT-IDENTICAL to the serial form.
        // Vein 15: the relative tables are zero outside the ±WINDOW band
        // (`rel_table` gathers window-4 embeddings into a (2T-1)-row table
        // and leaves the rest zero), so for the ~90 % of (i, j) pairs outside
        // it the `+ rr[d]` is an add of +0.0 — which cannot change a value
        // (x + 0.0 == x for every x but -0.0, and a -0.0 operand yields the
        // same product sign-folded into a sum that is never -0.0 itself).
        // Those pairs take the plain dot; only the band pays for the table.
        let in_band = |i: usize, j: usize| i.abs_diff(j) <= WINDOW;
        let row_kernel = |p: &mut Vec<f32>, hi: usize, orow: &mut [f32]| {
            let (h, i) = (hi / t_len, hi % t_len);
            let qr = &q[(h * t_len + i) * hd..(h * t_len + i + 1) * hd];
            for (j, pj) in p.iter_mut().enumerate() {
                let kr = &kk[(h * t_len + j) * hd..(h * t_len + j + 1) * hd];
                let mut s = 0f32;
                if in_band(i, j) {
                    let rr = &relk[(j + t_len - 1 - i) * hd..(j + t_len - i) * hd];
                    for d in 0..hd {
                        s += qr[d] * (kr[d] + rr[d]);
                    }
                } else {
                    for d in 0..hd {
                        s += qr[d] * kr[d];
                    }
                }
                *pj = s;
            }
            // Softmax over the row, in place. The max is the vectorised
            // reduction (vein 15): max is associative on these finite scores,
            // so the value is the fold's to the bit.
            let m = ffai_core::fastmath::max_f32(p);
            let mut sum = 0f32;
            for pj in p.iter_mut() {
                *pj = ffai_core::fastmath::exp(*pj - m);
                sum += *pj;
            }
            let inv = 1.0 / sum;
            // out[h,i] = Σⱼ p·(v[j] + relv[j−i+T−1]).
            orow.iter_mut().for_each(|o| *o = 0.0);
            for (j, pj) in p.iter().enumerate() {
                let w = pj * inv;
                let vr = &vv[(h * t_len + j) * hd..(h * t_len + j + 1) * hd];
                if in_band(i, j) {
                    let rr = &relv[(j + t_len - 1 - i) * hd..(j + t_len - i) * hd];
                    for d in 0..hd {
                        orow[d] += w * (vr[d] + rr[d]);
                    }
                } else {
                    for d in 0..hd {
                        orow[d] += w * vr[d];
                    }
                }
            }
        };
        // Guard MEASURED, not reasoned (`examples/len_sweep.rs`, paired ABBA
        // win-rate at each length). Parallel wins at EVERY length tested and
        // the margin grows without bound, because attention is O(T^2) and only
        // the parallel arm spreads that:
        //   T=16 1.30x z+2.32 | 64 1.68x z+3.87 | 256 4.26x z+3.87 | 1024 9.63x
        // So this floor is a safety net for degenerate inputs, not a crossover
        // -- there isn't one in the range that occurs. It also means the 1.04x
        // this change bought on ~88-phoneme sentences UNDERSTATES it badly for
        // long-form text: the same code is worth ~9.6x at T=1024.
        let parallel = HEADS * t_len >= 32 && !crate::asr::knobs::TTS_SERIAL_ATTN.get();
        // Every row of `outh` is zeroed-then-accumulated by `row_kernel`, and
        // every element of `out` is assigned by the transpose: no zero passes
        // (vein 15).
        let outh = crate::buf::filled_f32(HEADS * t_len * hd, |outh| {
            if parallel {
                use crate::par::prelude::*;
                outh.par_chunks_mut(hd).enumerate().for_each_init(
                    || vec![0f32; t_len],
                    |p, (hi, orow)| row_kernel(p, hi, orow),
                );
            } else {
                let mut p = vec![0f32; t_len];
                for (hi, orow) in outh.chunks_mut(hd).enumerate() {
                    row_kernel(&mut p, hi, orow);
                }
            }
        });

        let out = crate::buf::filled_f32(HIDDEN * t_len, |out| {
            for h in 0..HEADS {
                for i in 0..t_len {
                    let src = &outh[(h * t_len + i) * hd..(h * t_len + i + 1) * hd];
                    for (d, s) in src.iter().enumerate() {
                        out[(h * hd + d) * t_len + i] = *s;
                    }
                }
            }
        });
        let out = Tensor::from_vec(out, (1, HIDDEN, t_len), &self.device).e()?;
        self.conv(&names.conv_o, &out)
    }

    /// The `[(2T-1) × head_dim]` relative table, flat: window-4 learned
    /// embeddings centered, zeros outside — an index gather, not a pad/cat.
    fn rel_table(&self, name: &str, t_len: usize) -> Result<Vec<f32>> {
        let emb = self.t(name)?; // [1, 2*WINDOW+1, hd]
        let hd = emb.dim(2).e()?;
        let ev: Vec<f32> = emb.flatten_all().e()?.to_vec1().e()?;
        {
            let c = crate::asr::profile::counters();
            c.rel_table_builds.hit();
            c.tts_copy_bytes
                .add_n((ev.len() + (2 * t_len - 1) * hd) * 4);
        }
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

    /// One encoder layer's attention half, exposed for `enc_anatomy`.
    #[doc(hidden)]
    pub fn enc_attn_probe(&self, layer: usize, x: &Tensor, t_len: usize) -> Result<Tensor> {
        self.rel_attention(layer, x, t_len)
    }

    /// One encoder layer's feed-forward half, exposed for `enc_anatomy`.
    ///
    #[doc(hidden)]
    pub fn enc_ffn_probe(&self, layer: usize, x: &Tensor) -> Result<Tensor> {
        self.ffn(&format!("enc_p.encoder.ffn_layers.{layer}"), x)
    }

    /// One encoder layer's LayerNorm, exposed for `enc_anatomy`.
    #[doc(hidden)]
    pub fn enc_norm_probe(&self, layer: usize, x: &Tensor) -> Result<Tensor> {
        self.layer_norm_named(&format!("enc_p.encoder.norm_layers_1.{layer}"), x)
    }

    fn ffn(&self, base: &str, x: &Tensor) -> Result<Tensor> {
        // The graph pads outside its Conv nodes (ONNX export artifact); the
        // same arithmetic is conv1d's own padding parameter — two fewer
        // pad allocations+copies per layer.
        //
        // A k=3 conv IS a GEMM, and candle's conv1d wrapper charges 1.71x to
        // reach it: expressing the identical arithmetic as im2col + matmul
        // measured 251 -> 147 ms over 6 layers x 20 sentences
        // (`examples/ffn_ceiling.rs`, 74.9 -> 127.8 GFLOP/s). That matters
        // because the feed-forward half alone was costing more than
        // onnxruntime spends on the ENTIRE text encoder. Ceiling for reference:
        // the same GEMMs with no data movement run 96.3 ms, so the im2col copy
        // is ~51 ms of what remains and the N=88 column count costs a further
        // 2.4x against a batched shape — both still on the table.
        // `FFAI_CANDLE_FFN=1` restores the conv1d path for A/B.
        //
        // A further de-plumbing (one flat Vec through both convs, killing the
        // 768-channel intermediate Tensor and candle's relu) was built and
        // REVERTED. Its first version reused a scratch buffer, which forced
        // `Tensor::from_slice` (copies) in place of `from_vec` (moves) and cost
        // a full copy of the im2col matrix per conv — measured a ~6% CPU-time
        // REGRESSION (11/31, z = -1.62), the opposite of what the wall-clock
        // probe suggested. Fixing that to allocate-and-move left it NEUTRAL
        // (16/31, z = +0.18 against a clean 1.019 null arm), so it was removed
        // as complexity that buys nothing. The GEMM itself is the whole win:
        // 1.456x CPU time (29/31, z = +4.85) and 1.719x wall against candle.
        self.ffn_named(&format!("{base}.conv_1"), &format!("{base}.conv_2"), x)
    }

    /// [`Self::ffn`] with the two conv names already formatted (vein 9).
    fn ffn_named(&self, conv_1: &str, conv_2: &str, x: &Tensor) -> Result<Tensor> {
        let use_gemm = !crate::asr::knobs::TTS_CANDLE_FFN.get();
        let conv = |name: &str, x: &Tensor| -> Result<Tensor> {
            // One lookup for weight + bias (vein 9).
            let (w, b) = match self.conv_cache.get(name) {
                Some(ConvEntry {
                    weight,
                    bias: Some(bias),
                    ..
                }) => {
                    crate::asr::profile::counters().name_lookups.hit();
                    (weight, bias)
                }
                _ => (
                    self.t(&format!("{name}.weight"))?,
                    self.t(&format!("{name}.bias"))?,
                ),
            };
            let c = b.dim(0).e()?;
            if use_gemm {
                return super::decoder_kernels::conv1d_k3_gemm(x, w, b, &self.device);
            }
            x.conv1d(w, 1, 1, 1, 1)
                .e()?
                .broadcast_add(&b.reshape((1, c, 1)).e()?)
                .e()
        };
        let h = conv(conv_1, x)?.relu().e()?;
        conv(conv_2, &h)
    }

    // ---------------------------------------------------------------- stage 2

    /// The duration predictor's conditioning half (pre -> DDSConv -> proj),
    /// exposed so `dp_anatomy` can separate it from the flow half. Nearly all
    /// of this stage's multiply-accumulate lives here: 192 channels against the
    /// flow half's 2.
    #[doc(hidden)]
    pub fn dp_conditioning_probe(&self, hidden: &Tensor) -> Result<Tensor> {
        let g = self.conv("dp.pre", hidden)?;
        let g = self.dds_conv("dp.convs", &g, None)?;
        self.conv("dp.proj", &g)
    }

    /// Stochastic duration predictor, reverse mode → raw (pre-`ceil`) frames
    /// per phoneme.
    ///
    /// Two graph-confirmed quirks: `dp.flows.1` never runs (the VITS
    /// reversal drops it), and durations are NOT clamped — only the total
    /// length is, downstream.
    fn durations_raw(
        &self,
        hidden: &Tensor,
        noise_w: f64,
        length_scale: f64,
        rng: &mut GaussRng,
    ) -> Result<Vec<f64>> {
        let t_len = hidden.dim(2).e()?;
        // Conditioning: pre -> DDSConv -> proj.
        let g = self.conv("dp.pre", hidden)?;
        let g = self.dds_conv("dp.convs", &g, None)?;
        let g = self.conv("dp.proj", &g)?;

        // z starts as noise * noise_w over 2 channels.
        let mut z = if noise_w > 0.0 {
            rng.tensor(&[1, 2, t_len], &self.device)?
                .affine(noise_w, 0.0)
                .e()?
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
            .map(|lw| f64::from(lw.exp()) * length_scale)
            .collect())
    }

    /// Frames per phoneme — the integer contract the graph's `w_ceil` pins.
    pub fn durations(
        &self,
        hidden: &Tensor,
        noise_w: f64,
        length_scale: f64,
        rng: &mut GaussRng,
    ) -> Result<Vec<u32>> {
        Ok(self
            .durations_raw(hidden, noise_w, length_scale, rng)?
            .into_iter()
            .map(|d| d.ceil().max(0.0) as u32)
            .collect())
    }

    /// The same durations BEFORE `.ceil()`. Exposed so `corpus_stability` can
    /// report how close the nearest phoneme sits to an integer boundary: a
    /// routing change that reorders float accumulation is only safe by the
    /// margin between the raw duration and the boundary it would round across,
    /// and "it passed" means little without that margin.
    #[doc(hidden)]
    pub fn durations_raw_probe(
        &self,
        hidden: &Tensor,
        noise_w: f64,
        length_scale: f64,
        rng: &mut GaussRng,
    ) -> Result<Vec<f64>> {
        self.durations_raw(hidden, noise_w, length_scale, rng)
    }

    /// `DDSConv`: depthwise-separable dilated conv stack with channel
    /// `LayerNorms` and exact-erf GELU (the graph uses Erf, not tanh-GELU).
    fn dds_conv(&self, base: &str, x: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let mut x = match g {
            Some(g) => (x + g).e()?,
            None => x.clone(),
        };
        // Names formatted once at load for the four stacks the graph has
        // (vein 9); anything else formats on the spot as before.
        let fallback;
        let names = if let Some(n) = self.dds_names.get(base) {
            n
        } else {
            fallback = DdsLayerNames::for_base(base);
            &fallback
        };
        for n in names {
            let y = self.conv(&n.sep, &x)?;
            let y = self.layer_norm_named(&n.norm1, &y)?;
            let y = gelu_erf(&y)?;
            let y = self.conv(&n.one_by_one, &y)?;
            let y = self.layer_norm_named(&n.norm2, &y)?;
            let y = gelu_erf(&y)?;
            x = (x + y).e()?;
        }
        Ok(x)
    }

    fn layer_norm_named(&self, base: &str, x: &Tensor) -> Result<Tensor> {
        // One lookup for gamma + beta (vein 9), read in place (vein 7).
        let (gamma, beta) = match self.ln_cache.get(base) {
            Some((g, b)) => {
                crate::asr::profile::counters().name_lookups.hit();
                (g, b)
            }
            None => (
                self.t(&format!("{base}.gamma"))?,
                self.t(&format!("{base}.beta"))?,
            ),
        };
        let c = gamma.dim(0).e()?;
        // Nine allocating tensor ops (mean/sub/sqr/mean/add/sqrt/div/mul/add)
        // for what is two passes over 66 KB. The duration predictor runs 24 of
        // these per sentence and the text encoder 12, so the tensor spelling
        // costs ~216 intermediate allocations per utterance -- and dp measured
        // 8.4 GFLOP/s, i.e. almost entirely plumbing rather than arithmetic.
        // `FFAI_CANDLE_LN=1` restores the tensor path for A/B.
        if !crate::asr::knobs::TTS_CANDLE_LN.get() {
            let (_, cc, t) = x.dims3().e()?;
            let xc = if x.is_contiguous() {
                x.clone()
            } else {
                x.contiguous().e()?
            };
            let out = flat3(&xc, gamma, beta, |xv, g, b| {
                super::decoder_kernels::layer_norm_flat(xv, cc, t, g, b, 1e-5)
            })?;
            return Tensor::from_vec(out, (1, cc, t), &self.device).e();
        }
        let mean = x.mean_keepdim(1).e()?;
        let centered = x.broadcast_sub(&mean).e()?;
        let var = centered.sqr().e()?.mean_keepdim(1).e()?;
        let normed = centered.broadcast_div(&(var + 1e-5).e()?.sqrt().e()?).e()?;
        let gamma = gamma.reshape((1, c, 1)).e()?;
        let beta = beta.reshape((1, c, 1)).e()?;
        normed.broadcast_mul(&gamma).e()?.broadcast_add(&beta).e()
    }

    /// `ConvFlow` reverse: spline-transform the second channel conditioned on
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
        // Each column's spline is independent, and its parameter vectors are
        // 10/10/9 f64 — small enough to live on the stack. The original
        // collected three heap Vecs PER COLUMN and walked the columns serially,
        // which is most of why the duration predictor's flow half (65% of the
        // stage) stayed at ~1 core.
        let column = |t: usize| -> f32 {
            let mut widths = [0f64; SPLINE_BINS];
            let mut heights = [0f64; SPLINE_BINS];
            let mut derivs = [0f64; SPLINE_BINS - 1];
            for b in 0..SPLINE_BINS {
                widths[b] = f64::from(params[b][t]) * filt_scale;
                heights[b] = f64::from(params[SPLINE_BINS + b][t]) * filt_scale;
            }
            for (b, d) in derivs.iter_mut().enumerate() {
                *d = f64::from(params[2 * SPLINE_BINS + b][t]);
            }
            rq_spline_inverse(f64::from(z1v[t]), &widths, &heights, &derivs) as f32
        };
        // Parallelising this loop was rejected at T~88, then RE-TESTED across a
        // length sweep, because a mixed win/lose outcome is an unfinished
        // dispatch rather than a verdict (codec-content-adaptive-dispatch). Its
        // two siblings on the same work-per-task axis — dp's dense convs and
        // the attention row grid — both WON, so the spline deserved the same
        // question asked properly: is there a length where it crosses over?
        //
        // There is not, anywhere in range (`examples/len_sweep.rs`, paired):
        //   T=32 0.877x z-2.84 | 64 0.939x z-2.32 | 128 0.994x z-0.26
        //   T=256 1.043x z+1.81 | 512 z+0.26 | 1024 1.038x z+1.81
        // Serial genuinely wins at 32-64; parallel only ever LEANS at >=256 and
        // never reaches z=+2. A column is one ~10-bin spline solve, so rayon's
        // fork/join outweighs the work at every size that occurs. This is the
        // "no cheap signal separates them -> gate it off" endpoint, recorded
        // with what was measured. `FFAI_PAR_SPLINE=1` re-enables it.
        //
        // The de-plumbing above (three heap Vecs per column -> stack arrays) is
        // kept separately, on byte-identity and strictly-less-work.
        let out: Vec<f32> = if t_len >= 32 && crate::asr::knobs::TTS_PAR_SPLINE.get() {
            use crate::par::prelude::*;
            (0..t_len).into_par_iter().map(column).collect()
        } else {
            (0..t_len).map(column).collect()
        };
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

    /// Residual coupling flow, reversed: `z_p` → z.
    ///
    /// The flat fused-gate version exists (`FFAI_FLAT_FLOW=1`) and is
    /// oracle-exact, but the interleaved paired A/B scored it a WASH —
    /// 0.99×, 4/15 rounds — because this stage's cost is its convolutions,
    /// not op dispatch. candle stays the default until a quiet-machine
    /// retest or a faster short-length conv kernel changes the verdict.
    pub fn flow_reverse(&self, z_p: &Tensor) -> Result<Tensor> {
        // Flat is the default (42/45 paired wins, §6.14);
        // FFAI_CANDLE_FLOW=1 restores the candle path for A/B.
        if let Some(flat_flow) = &self.flat_flow
            && !crate::asr::knobs::TTS_CANDLE_FLOW.get()
        {
            let len = z_p.dim(2).e()?;
            let mut zv: Vec<f32> = z_p.flatten_all().e()?.to_vec1().e()?;
            flat_flow.run(&mut zv, len)?;
            return Tensor::from_vec(zv, (1, HIDDEN, len), &self.device).e();
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
            let t_half = ffai_core::fastops::tanh(&a.narrow(1, 0, HIDDEN).e()?).e()?;
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
        if !crate::asr::knobs::TTS_CANDLE_DEC.get() {
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
        let audio = ffai_core::fastops::tanh(&x).e()?.flatten_all().e()?;
        audio.to_vec1().e()
    }

    // ------------------------------------------------------------- plumbing

    fn t(&self, name: &str) -> Result<&Tensor> {
        crate::asr::profile::counters().name_lookups.hit();
        self.w
            .get(name)
            .ok_or_else(|| Error::Model(format!("vits weight `{name}` missing from safetensors")))
    }

    /// Conv/ConvTranspose by canonical name, geometry from the graph JSON,
    /// bias applied when present (`conv_post` has none).
    ///
    /// Decoder-stage convs route through the cache-blocked direct kernels
    /// ([`super::decoder_kernels`]) — profiled 82.7 % of synthesis, and
    /// candle's conv1d degrades with length (99 → 27 GF/s) as its working
    /// set leaves cache. `FFAI_CANDLE_CONV=1` switches back for A/B; the
    /// kernels' scalar twin IS candle's conv (see their tests).
    fn conv(&self, name: &str, x: &Tensor) -> Result<Tensor> {
        // One lookup resolves weight, bias and geometry (vein 9); the
        // formatted-name path stays for a conv the cache never saw.
        let (weight, g, bias) = match self.conv_cache.get(name) {
            Some(entry) => {
                crate::asr::profile::counters().name_lookups.hit();
                (&entry.weight, entry.geom, entry.bias.as_ref())
            }
            None => (
                self.t(&format!("{name}.weight"))?,
                self.geom.get(name).copied().unwrap_or(ConvGeom {
                    transpose: false,
                    stride: 1,
                    pad: 0,
                    dilation: 1,
                }),
                self.w.get(&format!("{name}.bias")),
            ),
        };
        // Routing, settled by measurement per stage: dec.* on the direct
        // kernels (4× there), dp's DEPTHWISE convs on the serial depthwise
        // kernel (395 → 254 ms per 20 sentences), and enc_p/flow REVERTED to
        // candle — universal routing was tried and enc_p regressed 1.35×
        // (serial 1×1s lose to candle's threaded matmul at 192×192×T) while
        // flow measured flat. `FFAI_CANDLE_CONV=1` restores candle end to end.
        let use_direct = (name.starts_with("dec.") || name.starts_with("dp."))
            && !crate::asr::knobs::TTS_CANDLE_CONV.get()
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
                let k = weight.dim(2).e()?;
                // Operands read in place (vein 7): no `to_vec1` copies of the
                // activation, the weight or the bias.
                let xc = if x.is_contiguous() {
                    x.clone()
                } else {
                    x.contiguous().e()?
                };
                let wc = if weight.is_contiguous() {
                    weight.clone()
                } else {
                    weight.contiguous().e()?
                };
                let run = |xv: &[f32], wv: &[f32], bv: Option<&[f32]>| {
                    super::decoder_kernels::conv1d_depthwise_flat(
                        xv, c, len, wv, bv, k, g.pad, g.dilation,
                    )
                };
                let (out, l_out) = match bias {
                    Some(b) => flat3(&xc, &wc, b, |xv, wv, bv| run(xv, wv, Some(bv)))?,
                    None => crate::buf::with_f32(&xc, |xv| {
                        crate::buf::with_f32(&wc, |wv| run(xv, wv, None))
                    })
                    .e()?
                    .e()?,
                };
                return Tensor::from_vec(out, (1, c, l_out), &self.device).e();
            }
            if x_ch / in_per_group == 1 {
                // `conv1d_direct` is serial. That is right for dec.*, whose
                // shapes are large enough to win anyway, but dp's dense convs
                // are 192x192xT — the very shape the note above records as
                // LOSING to candle's threaded matmul in enc_p. Routing dp here
                // is why the duration predictor runs at 0.88 cores while every
                // other stage reaches 11+. Depthwise dp convs still take the
                // flat kernel above (measured 395 -> 254 ms), only the dense
                // ones fall through. `FFAI_DP_DIRECT_1X1=1` restores the old
                // routing for A/B.
                let dp_dense_to_candle =
                    name.starts_with("dp.") && !crate::asr::knobs::TTS_DP_DIRECT_1X1.get();
                if !dp_dense_to_candle {
                    return super::decoder_kernels::conv1d_direct(
                        x, weight, bias, g.pad, g.dilation,
                    );
                }
            }
        }
        let groups = if g.transpose {
            1
        } else {
            let in_per_group = weight.dim(1).e()?;
            x.dim(1).e()? / in_per_group
        };
        // Every 1x1 convolution is a GEMM wearing a costume, and candle's
        // conv1d wrapper charges to reach it (1.71x measured on the sibling
        // k=3 FFN shapes). This catches the text encoder's output projections
        // AND essentially the whole duration predictor, whose dense convs are
        // all k=1. Depthwise and transposed convs are already routed above.
        if !g.transpose
            && groups == 1
            && weight.dim(2).e()? == 1
            && g.pad == 0
            && g.stride == 1
            && !crate::asr::knobs::TTS_CANDLE_FFN.get()
        {
            return super::decoder_kernels::conv1d_k1_gemm(x, weight, bias, &self.device);
        }
        let y = if g.transpose {
            x.conv_transpose1d(weight, g.pad, 0, g.stride, g.dilation, 1)
                .e()?
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

/// Exact-erf GELU. The graph specifies `Erf`, so the function is fixed; the
/// evaluation is ours. `FFAI_CANDLE_GELU=1` restores candle's own.
fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    if crate::asr::knobs::TTS_CANDLE_GELU.get() {
        return x.gelu_erf().e();
    }
    let dims = x.dims().to_vec();
    // Read in place (vein 7); the one copy left is the output itself.
    let xc = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous().e()?
    };
    let out = crate::buf::with_f32(&xc, super::decoder_kernels::gelu_erf_flat).e()?;
    Tensor::from_vec(out, dims, x.device()).e()
}

/// Borrow three contiguous f32 tensors at once (vein 7) — the layer norm's
/// `(x, gamma, beta)` and the depthwise conv's `(x, weight, bias)`.
fn flat3<R>(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    f: impl FnOnce(&[f32], &[f32], &[f32]) -> R,
) -> Result<R> {
    crate::buf::with_f32(a, |av| {
        crate::buf::with_f32(b, |bv| crate::buf::with_f32(c, |cv| f(av, bv, cv)))
    })
    .e()?
    .e()?
    .e()
}

/// `[h, T, 2T-1]` relative logits → `[h, T, T]` absolute (the VITS pad /
/// reshape / slice trick, verified by the hand example in the tests).
#[allow(dead_code)] // relative-attention path: verified by the hand example in tests, kept for when it is wired up
fn relative_to_absolute(x: &Tensor) -> Result<Tensor> {
    let (h, t, _span) = x.dims3().e()?;
    let x = x.pad_with_zeros(2, 0, 1).e()?; // [h,T,2T]
    let flat = x.reshape((h, t * 2 * t)).e()?;
    let flat = flat.pad_with_zeros(1, 0, t - 1).e()?;
    let x = flat.reshape((h, t + 1, 2 * t - 1)).e()?;
    x.narrow(1, 0, t).e()?.narrow(2, t - 1, t).e()
}

/// `[h, T, T]` attention weights → `[h, T, 2T-1]` relative positions.
#[allow(dead_code)] // inverse of relative_to_absolute; same reason
fn absolute_to_relative(x: &Tensor) -> Result<Tensor> {
    let (h, t, _) = x.dims3().e()?;
    let x = x.pad_with_zeros(2, 0, t - 1).e()?; // [h,T,2T-1]
    let flat = x.reshape((h, t * (2 * t - 1))).e()?;
    let flat = flat.pad_with_zeros(1, t, 0).e()?;
    let x = flat.reshape((h, t, 2 * t)).e()?;
    x.narrow(2, 1, 2 * t - 1).e()
}

/// Inverse monotonic rational-quadratic spline with linear tails
/// (Durkan et al.; the transform `dp`'s `ConvFlows` apply to durations).
fn rq_spline_inverse(y: f64, uw: &[f64], uh: &[f64], ud: &[f64]) -> f64 {
    let nb = uw.len();
    if !(-SPLINE_TAIL..=SPLINE_TAIL).contains(&y) {
        return y; // linear tails: identity outside the bound
    }
    // Normalized bin widths/heights (softmax with a minimum), cumulative
    // over [-B, B].
    let norm = |u: &[f64]| -> Vec<f64> {
        let m = u.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = u.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        exps.iter()
            .map(|e| MIN_BIN + MIN_BIN.mul_add(-(nb as f64), 1.0) * e / s)
            .collect()
    };
    let widths = norm(uw);
    let heights = norm(uh);
    let span = 2.0 * SPLINE_TAIL;
    let mut cw = vec![-SPLINE_TAIL];
    let mut ch = vec![-SPLINE_TAIL];
    for b in 0..nb {
        cw.push(widths[b].mul_add(span, cw[b]));
        ch.push(heights[b].mul_add(span, ch[b]));
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
    let a = dy.mul_add(2.0f64.mul_add(-delta, d0 + d1), h_k * (delta - d0));
    let b = h_k.mul_add(d0, -(dy * 2.0f64.mul_add(-delta, d0 + d1)));
    let c = -delta * dy;
    let disc = b.mul_add(b, -(4.0 * a * c)).max(0.0);
    let theta = 2.0 * c / (-b - disc.sqrt());
    x_k + theta * w_k
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Deterministic gaussian noise: xorshift64* + Box–Muller. Same seed, same
/// stream — the byte-stability contract behind `--seed`.
pub struct GaussRng {
    state: u64,
    spare: Option<f64>,
}

impl GaussRng {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E3779B97F4A7C15) | 1,
            spare: None,
        }
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
