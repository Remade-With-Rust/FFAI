//! PP-DocLayout-S — document layout detection, the pipeline stage Carmenta was
//! missing.
//!
//! ## Why this exists
//!
//! §44: Carmenta implements two stages of the five-stage pipeline the field
//! uses — text detection and text recognition — and none of layout, formula or
//! table. That is the whole of the headline gap, and the evidence is on disk:
//! on `unlimited_holdout` (236 English pages with ZERO LaTeX blocks, ZERO
//! tables, ZERO equations) Carmenta reads **0.0406** text, against **0.0794**
//! for PP-StructureV3 and **0.0326** for PaddleOCR-VL. On content we can
//! represent we already beat the reference pipeline. 0.1278 corpus-wide is a
//! COVERAGE deficit.
//!
//! Layout comes first because nothing can be ROUTED until regions are
//! classified: a formula cannot go to a formula recognizer if nothing ever
//! decides a region is a formula. §40 measured 87% of genuinely-missed
//! characters sitting in LaTeX-bearing blocks, and every detection-threshold
//! sweep came back flat precisely because those knobs filter boxes that already
//! exist — they never create the region class.
//!
//! ## The model
//!
//! PicoDet/GFL, 4.9 MB, 1.20 M parameters, 411 executable nodes. Input is a
//! fixed 480x480 with ImageNet normalisation; output is 23 region classes, of
//! which `formula` (7), `table` (8) and `formula_number` (19) are the ones the
//! downstream stages need.
//!
//! Executed through `onnx_graph`, not hand-transcribed — see that module for
//! why. The graph tail (NMS, top-k, packing: nodes 378..411) is done here in
//! Rust instead, because it is clearer written than interpreted.

use candle_core::{DType, Device, Tensor};
use ffai_core::error::{Error, Result};
use ffai_core::types::ImageBuffer;
use std::collections::HashMap;

pub const LABELS: [&str; 23] = [
    "paragraph_title", "image", "text", "number", "abstract", "content",
    "figure_title", "formula", "table", "table_title", "reference", "doc_title",
    "footnote", "header", "algorithm", "footer", "seal", "chart_title", "chart",
    "formula_number", "header_image", "footer_image", "aside_text",
];
pub const CLS_FORMULA: usize = 7;
pub const CLS_TABLE: usize = 8;
pub const CLS_FORMULA_NUMBER: usize = 19;

const SIDE: usize = 480;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// The node index the network proper ends at: `Div.1` (boxes, node 377) and
/// `Concat.9` (scores, node 365) are both produced by then, and node 378 is
/// `NonMaxSuppression`.
const STOP: usize = 378;
const BOXES: &str = "Div.1";
const SCORES: &str = "Concat.9";

#[derive(Debug, Clone)]
pub struct Region {
    pub class: usize,
    pub score: f32,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Region {
    pub fn label(&self) -> &'static str {
        LABELS.get(self.class).copied().unwrap_or("?")
    }
    pub fn is_formula(&self) -> bool {
        self.class == CLS_FORMULA || self.class == CLS_FORMULA_NUMBER
    }
    /// Whether this region should be READ AS LATEX.
    ///
    /// `formula_number` is deliberately excluded. It is the "(26)" printed in
    /// the margin beside an equation — ordinary text that happens to sit next
    /// to maths. Sending it to the formula decoder replaces a two-character
    /// string we already read correctly with a LaTeX rendering of it, which is
    /// wrong in the output AND removes it from the reading order it belongs to.
    pub fn routes_to_latex(&self) -> bool {
        self.class == CLS_FORMULA
    }
    pub fn is_table(&self) -> bool {
        self.class == CLS_TABLE
    }
}

pub struct DocLayout {
    graph: crate::onnx_graph::Graph,
    device: Device,
}

impl DocLayout {
    pub fn new(arch_json: &std::path::Path, weights: &std::path::Path) -> Result<Self> {
        let arch: crate::onnx_graph::Arch = serde_json::from_str(
            &std::fs::read_to_string(arch_json)
                .map_err(|e| Error::Other(format!("layout arch: {e}")))?,
        )
        .map_err(|e| Error::Other(format!("layout arch parse: {e}")))?;
        let device = Device::Cpu;
        let weights = candle_core::safetensors::load(weights, &device)
            .map_err(crate::image::candle_err)?;
        Ok(Self { graph: crate::onnx_graph::Graph { arch, weights }, device })
    }

    /// Resize to 480x480 (no aspect preservation — `keep_ratio: false` in the
    /// reference config), scale to 0..1, ImageNet-normalise, CHW.
    ///
    /// Uses `image::resize_bilinear`, the same sampler `craft_input`,
    /// `crnn_input` and `mobiledet_input` go through, rather than a private
    /// loop: two resamplers in one pipeline is two sets of numerics to keep in
    /// step, and the difference would show up as a silent coordinate drift
    /// between the layout boxes and the text boxes they are meant to contain.
    fn input(&self, img: &ImageBuffer) -> Result<Tensor> {
        let (w, h) = (img.width as usize, img.height as usize);
        let bpp = img.format.bytes_per_pixel();
        let mut chw = vec![0f32; 3 * SIDE * SIDE];
        for c in 0..3 {
            // channel plane at source resolution, then the house resampler
            let src_c = if bpp == 1 { 0 } else { c };
            let plane: Vec<f32> =
                (0..w * h).map(|i| img.data[i * bpp + src_c] as f32).collect();
            let resized = crate::image::resize_bilinear(&plane, w, h, SIDE, SIDE);
            let dst = &mut chw[c * SIDE * SIDE..(c + 1) * SIDE * SIDE];
            for (d, &s) in dst.iter_mut().zip(&resized) {
                *d = (s / 255.0 - MEAN[c]) / STD[c];
            }
        }
        // oracle hand-off, same contract as `table.rs`/`formula.rs`: the
        // comparison is only worth anything if both sides get identical bytes
        if let Ok(p) = std::env::var("FFAI_LAYOUT_BIN") {
            let mut b = Vec::with_capacity(chw.len() * 4);
            for v in &chw {
                b.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::write(p, b);
        }
        Tensor::from_vec(chw, (1, 3, SIDE, SIDE), &self.device).map_err(crate::image::candle_err)
    }

    pub fn detect(&self, img: &ImageBuffer, score_thr: f32, iou_thr: f32) -> Result<Vec<Region>> {
        let (w, h) = (img.width as f32, img.height as f32);
        let x = self.input(img)?;
        // `scale_factor` is [scale_y, scale_x] = network size / source size; the
        // graph divides its boxes by it to return SOURCE coordinates.
        let sf = Tensor::from_vec(
            vec![SIDE as f32 / h, SIDE as f32 / w],
            (1, 2),
            &self.device,
        )
        .map_err(crate::image::candle_err)?;

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert("image".into(), x);
        inputs.insert("scale_factor".into(), sf);
        let env = self
            .graph
            .run(inputs, STOP, &self.device)
            .map_err(crate::image::candle_err)?;

        let boxes = env
            .get(BOXES)
            .ok_or_else(|| Error::Other(format!("layout: no `{BOXES}`")))?
            .to_dtype(DType::F32)
            .map_err(crate::image::candle_err)?;
        let scores = env
            .get(SCORES)
            .ok_or_else(|| Error::Other(format!("layout: no `{SCORES}`")))?
            .to_dtype(DType::F32)
            .map_err(crate::image::candle_err)?;

        // boxes [1, A, 4]  scores [1, C, A]
        let bd = boxes.dims().to_vec();
        let sd = scores.dims().to_vec();
        let bv = boxes.flatten_all().map_err(crate::image::candle_err)?
            .to_vec1::<f32>().map_err(crate::image::candle_err)?;
        let sv = scores.flatten_all().map_err(crate::image::candle_err)?
            .to_vec1::<f32>().map_err(crate::image::candle_err)?;
        let n_anchor = *bd.get(1).unwrap_or(&0);
        let n_class = *sd.get(1).unwrap_or(&0);

        // MULTI-LABEL, not argmax. PicoDet scores every class independently at
        // every anchor and the graph's own NMS keeps each class that clears the
        // threshold; taking only the top class per anchor silently discards the
        // rest. Measured against onnxruntime on one page that cost a real
        // region — `paragraph_title` 0.451 sitting under `text` 0.490 on the
        // same anchor — and the failure mode that matters here is worse than a
        // missing title: an anchor scoring `table` 0.44 behind `text` 0.46 would
        // never reach the table model at all, so the routing decision would be
        // made by a margin of 0.02 that nothing ever gets to see.
        let mut cand: Vec<Region> = Vec::new();
        for a in 0..n_anchor {
            let o = a * 4;
            for c in 0..n_class {
                let s = sv[c * n_anchor + a];
                if s < score_thr {
                    continue;
                }
                cand.push(Region {
                    class: c,
                    score: s,
                    x0: bv[o].max(0.0),
                    y0: bv[o + 1].max(0.0),
                    x1: bv[o + 2].min(w),
                    y1: bv[o + 3].min(h),
                });
            }
        }
        Ok(nms(cand, iou_thr))
    }
}

/// Class-wise greedy NMS — the graph tail, written rather than interpreted.
fn nms(mut c: Vec<Region>, iou_thr: f32) -> Vec<Region> {
    c.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<Region> = Vec::new();
    'outer: for r in c {
        for k in &keep {
            if k.class != r.class {
                continue;
            }
            let ix = (r.x1.min(k.x1) - r.x0.max(k.x0)).max(0.0);
            let iy = (r.y1.min(k.y1) - r.y0.max(k.y0)).max(0.0);
            let inter = ix * iy;
            let ar = ((r.x1 - r.x0) * (r.y1 - r.y0)).max(1e-6);
            let ak = ((k.x1 - k.x0) * (k.y1 - k.y0)).max(1e-6);
            if inter / (ar + ak - inter) > iou_thr {
                continue 'outer;
            }
        }
        keep.push(r);
    }
    keep
}
