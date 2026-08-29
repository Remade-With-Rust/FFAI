//! PP-FormulaNet_plus-M — formula recognition, the last missing pipeline stage.
//!
//! §40 measured 87% of genuinely-missed characters sitting in LaTeX-bearing
//! blocks, and every detection-threshold sweep came back flat because those
//! knobs filter boxes that already exist — a formula region has to be ROUTED to
//! a model that speaks LaTeX before any of it can be read. `doclayout` supplies
//! the routing; this is the model.
//!
//! The graph is 538 top-level nodes ending in a single `Loop` (node 537) that
//! carries 62 KV-cache variables. It emits TOKEN IDS directly —
//! `fetch_name_0` is `[N, T]` of int64 — so there is no logit argmax to do
//! here, only detokenisation.
//!
//! ## Preprocessing
//!
//! Transcribed from `UniMERNetImgDecode` / `UniMERNetTestTransform` /
//! `LatexImageFormat` in PaddleX's `formula_recognition/processors.py`, not
//! guessed: margin-crop at grey < 200, aspect-preserving fit into 384x384,
//! centred WHITE pad, then `(x/255 - 0.7931) / 0.1738` on a single channel.
//! `LatexImageFormat`'s pad-to-multiple-of-16 is a no-op at 384.

use candle_core::{DType, Device, Tensor};
use ffai_core::error::{Error, Result};
use ffai_core::types::ImageBuffer;
use std::collections::HashMap;

const SIDE: usize = 384;
const MEAN: f32 = 0.7931;
const STD: f32 = 0.1738;
/// `UniMERNetImgDecode` builds its bounding rect from `255 * (data < 200)`.
const INK: u8 = 200;

/// GPT-2 `bytes_to_unicode`, inverted: the printable code point a byte was
/// mapped to, back to the byte. The vocabulary is ByteLevel BPE, so a token
/// string is not text — it is bytes wearing a reversible disguise, and decoding
/// it as if it were text mangles every non-ASCII symbol in the LaTeX.
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn byte_decoder() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = (33..127).chain(161..173).chain(174..256).collect();
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter()
        .zip(cs.iter())
        .filter_map(|(&b, &c)| char::from_u32(c).map(|c| (c, b as u8)))
        .collect()
}

pub struct FormulaModel {
    graph: crate::onnx_graph::Graph,
    vocab: Vec<String>,
    bdec: HashMap<char, u8>,
    device: Device,
}

impl FormulaModel {
    pub fn new(
        arch_json: &std::path::Path,
        weights: &std::path::Path,
        vocab_json: &std::path::Path,
    ) -> Result<Self> {
        let arch: crate::onnx_graph::Arch = serde_json::from_str(
            &std::fs::read_to_string(arch_json)
                .map_err(|e| Error::Other(format!("formula arch: {e}")))?,
        )
        .map_err(|e| Error::Other(format!("formula arch parse: {e}")))?;
        let vocab: Vec<String> = serde_json::from_str(
            &std::fs::read_to_string(vocab_json)
                .map_err(|e| Error::Other(format!("formula vocab: {e}")))?,
        )
        .map_err(|e| Error::Other(format!("formula vocab parse: {e}")))?;
        let device = Device::Cpu;
        let weights = candle_core::safetensors::load(weights, &device)
            .map_err(crate::image::candle_err)?;
        Ok(Self {
            graph: crate::onnx_graph::Graph { arch, weights },
            vocab,
            bdec: byte_decoder(),
            device,
        })
    }

    /// Margin-crop, aspect-fit into 384x384, centred white pad, normalise.
    fn input(&self, img: &ImageBuffer) -> Result<Tensor> {
        let (w, h) = (img.width as usize, img.height as usize);
        let bpp = img.format.bytes_per_pixel();
        let grey: Vec<f32> = (0..w * h)
            .map(|i| {
                if bpp == 1 {
                    f32::from(img.data[i])
                } else {
                    // Rec. 601 luma, the same weights `image` uses elsewhere
                    let p = i * bpp;
                    0.299 * f32::from(img.data[p])
                        + 0.587 * f32::from(img.data[p + 1])
                        + 0.114 * f32::from(img.data[p + 2])
                }
            })
            .collect();

        // bounding rect of the ink, so a formula sitting in a wide margin is
        // scaled by its OWN extent rather than the crop the layout stage gave us.
        //
        // `crop_margin` STRETCHES the histogram to full range before thresholding
        // — `data = (data - min) / (max - min) * 255` — and only then takes
        // `data < 200`. Thresholding the raw values instead makes the crop depend
        // on the scan's exposure: a grey-background page has no pixel under 200
        // at all, so the "crop" keeps the whole region including its margins, and
        // the formula is then scaled down by the margin rather than by its own
        // extent.
        let (lo, hi) = grey.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let span = if hi > lo { hi - lo } else { 1.0 };
        let ink = |v: f32| (v - lo) / span * 255.0;

        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        for y in 0..h {
            for x in 0..w {
                if ink(grey[y * w + x]) < f32::from(INK) {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        if x1 <= x0 || y1 <= y0 {
            x0 = 0;
            y0 = 0;
            x1 = w;
            y1 = h;
        }
        let (cw, ch) = (x1 - x0, y1 - y0);
        let crop: Vec<f32> = (0..ch)
            .flat_map(|y| (0..cw).map(move |x| (y, x)))
            .map(|(y, x)| grey[(y0 + y) * w + x0 + x])
            .collect();

        // `thumbnail` semantics: fit INSIDE the box, never upscale past it
        let s = (SIDE as f32 / cw as f32).min(SIDE as f32 / ch as f32);
        let (rw, rh) = (
            ((cw as f32 * s).round() as usize).clamp(1, SIDE),
            ((ch as f32 * s).round() as usize).clamp(1, SIDE),
        );
        let r = crate::image::resize_bilinear(&crop, cw, ch, rw, rh);

        // Centred pad, filled BLACK. `ImageOps.expand(img, padding)` is called
        // with NO `fill` argument, and PIL's default is 0 — black.
        //
        // "The model was trained on white paper, so pad white" is the plausible
        // reading, and it is wrong. Measured, white padding made the decoder
        // hallucinate structure that is not in the image: a single clean line
        // `w(v) ~ 1, |v| <= 1.` came back wrapped in `\left\{ egin{aligned}`,
        // emitted twice, with a stray CJK character between the copies. The
        // same crop on black padding decodes to exactly the ground truth. The
        // huge uniform border is not neutral to this model — it reads as page,
        // and page implies more lines.
        let padv = env_f32("FFAI_FORMULA_PAD", 0.0) / 255.0;
        let mut out = vec![(padv - MEAN) / STD; SIDE * SIDE];
        let (ox, oy) = ((SIDE - rw) / 2, (SIDE - rh) / 2);
        for y in 0..rh {
            for x in 0..rw {
                out[(oy + y) * SIDE + ox + x] = (r[y * rw + x] / 255.0 - MEAN) / STD;
            }
        }
        // Raw f32 for the onnxruntime oracle: comparing against the reference
        // only means anything if BOTH sides get byte-identical input, so the
        // tensor is handed over rather than the preprocessing re-implemented.
        if let Ok(p) = std::env::var("FFAI_FORMULA_BIN") {
            let mut b = Vec::with_capacity(out.len() * 4);
            for v in &out {
                b.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::write(p, b);
        }
        if let Ok(p) = std::env::var("FFAI_FORMULA_DUMP") {
            // exactly what the network sees, back in 0..255 for eyeballing
            let px: Vec<u8> = out
                .iter()
                .map(|&v| ((v * STD + MEAN) * 255.0).clamp(0.0, 255.0) as u8)
                .collect();
            let mut f = format!("P5\n{SIDE} {SIDE}\n255\n").into_bytes();
            f.extend_from_slice(&px);
            let _ = std::fs::write(p, f);
        }
        Tensor::from_vec(out, (1, 1, SIDE, SIDE), &self.device).map_err(crate::image::candle_err)
    }

    /// Turn ByteLevel-BPE ids back into LaTeX.
    fn detokenize(&self, ids: &[i64]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &i in ids {
            let t = match self.vocab.get(i as usize) {
                Some(t) => t.as_str(),
                None => continue,
            };
            // `</s>` ends the sequence; `<s>`/`<pad>` never reach the output
            match t {
                "</s>" => break,
                "<s>" | "<pad>" => continue,
                _ => {}
            }
            for c in t.chars() {
                match self.bdec.get(&c) {
                    Some(&b) => bytes.push(b),
                    // a token outside the byte alphabet is a real added token
                    None => bytes.extend_from_slice(c.to_string().as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).trim().to_string()
    }

    /// Recognise ONE cropped formula region as LaTeX.
    pub fn recognize(&self, img: &ImageBuffer) -> Result<String> {
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert("x".into(), self.input(img)?);
        let n_nodes = self.graph.arch.nodes.len();
        let env = self
            .graph
            .run(inputs, n_nodes, &self.device)
            .map_err(crate::image::candle_err)?;
        let out = env
            .get("fetch_name_0")
            .ok_or_else(|| Error::Other("formula: no token output".into()))?;
        let ids = out
            .to_dtype(DType::I64)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(crate::image::candle_err)?;
        if std::env::var("FFAI_FORMULA_DEBUG").is_ok() {
            eprintln!("  ids {:?} first16 {:?}", out.dims(), &ids[..ids.len().min(16)]);
        }
        Ok(self.detokenize(&ids))
    }
}
