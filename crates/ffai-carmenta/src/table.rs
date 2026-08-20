//! SLANet_plus — table structure recognition, the third missing pipeline stage.
//!
//! §44/§45: the reference pipeline routes a `table` region to a structure model
//! that emits HTML tags plus a cell box per tag. Carmenta had no such stage, so
//! table content arrived as loose text lines in the middle of the reading order
//! — and `configs/unlimited_holdout.yaml` records the consequence: "Carmenta
//! emits neither tables nor formulas, so TEDS and CDM are not merely skipped for
//! convenience — they are not computable for this engine". The benchmark's
//! Overall metric cannot be reported at all without this.
//!
//! Encoder is PP-LCNet; the decoder is an ONNX `Loop` (144 nodes, 11 carried
//! variables, one nested `If`) that emits at most 500 structure tokens
//! autoregressively. Both run through `onnx_graph`.
//!
//! Outputs per the exported graph:
//!   `fetch_name_0`  [1, T, 8]   four corner points per predicted cell
//!   `fetch_name_1`  [1, T, 50]  structure-token logits

use candle_core::{DType, Device, Tensor};
use ffai_core::error::{Error, Result};
use ffai_core::types::ImageBuffer;
use std::collections::HashMap;

/// SLANet's structure vocabulary — EXTRACTED from the shipped `inference.yml`
/// (`character_dict`, 48 entries), not invented. PaddleOCR's
/// `TableLabelDecode` then applies TWO transforms before the list indexes the
/// model's 50 output channels, and BOTH are load-bearing:
///
///  1. `merge_no_span_structure: true` — also in that config — REMOVES `<td>`
///     and APPENDS `<td></td>` at the end. That is what puts the ordinary
///     no-span cell at index 48 rather than 7.
///  2. `add_special_char` wraps the result as `[sos] + dict + [eos]` → 50.
///
/// Skipping (1) is not a cosmetic error. With the raw config order, this table
/// decoded a genuinely correct 7-row x 5-column result as
/// `<tr> rowspan="20" x5 </tr>` — the LAST dict entry — because the cell token
/// had been read off the end. The shape of the output was already right; only
/// the names were wrong, which is the failure mode that looks most like a
/// broken model and is least like one.
pub const TOKENS: [&str; 50] = [
    "sos", "<thead>", "</thead>", "<tbody>", "</tbody>", "<tr>", "</tr>", "<td",
    ">", "</td>", " colspan=\"2\"", " colspan=\"3\"", " colspan=\"4\"", " colspan=\"5\"",
    " colspan=\"6\"", " colspan=\"7\"", " colspan=\"8\"", " colspan=\"9\"",
    " colspan=\"10\"", " colspan=\"11\"", " colspan=\"12\"", " colspan=\"13\"",
    " colspan=\"14\"", " colspan=\"15\"", " colspan=\"16\"", " colspan=\"17\"",
    " colspan=\"18\"", " colspan=\"19\"", " colspan=\"20\"", " rowspan=\"2\"",
    " rowspan=\"3\"", " rowspan=\"4\"", " rowspan=\"5\"", " rowspan=\"6\"",
    " rowspan=\"7\"", " rowspan=\"8\"", " rowspan=\"9\"", " rowspan=\"10\"",
    " rowspan=\"11\"", " rowspan=\"12\"", " rowspan=\"13\"", " rowspan=\"14\"",
    " rowspan=\"15\"", " rowspan=\"16\"", " rowspan=\"17\"", " rowspan=\"18\"",
    " rowspan=\"19\"", " rowspan=\"20\"", "<td></td>", "eos",
];

const SIDE: usize = 488;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone)]
pub struct Cell {
    /// axis-aligned bounds of the predicted cell, in source pixels
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Debug, Clone)]
pub struct TableStructure {
    /// the emitted structure tokens, in order
    pub tokens: Vec<String>,
    /// one box per `<td` / `<td></td>` token
    pub cells: Vec<Cell>,
}

impl TableStructure {
    /// Assemble the tokens into HTML. Cell TEXT is filled by the caller from
    /// the OCR lines whose centre falls inside each cell box — this stage
    /// recovers STRUCTURE, which is what TEDS scores.
    /// A cell is opened by either `<td>` (plain) or `<td` (one that carries a
    /// span attribute and is closed by a later `>`); BOTH consume one box from
    /// the box head, which is why the counter advances on each. `sos`/`eos` are
    /// control tokens and never reach the markup.
    pub fn to_html(&self, cell_text: &[String]) -> String {
        let mut s = String::from("<table>");
        let mut ci = 0usize;
        for t in &self.tokens {
            match t.as_str() {
                "sos" => {}
                "eos" => break,
                "<td></td>" => {
                    s.push_str("<td>");
                    if let Some(txt) = cell_text.get(ci) {
                        s.push_str(txt);
                    }
                    s.push_str("</td>");
                    ci += 1;
                }
                "<td" => {
                    s.push_str("<td");
                    ci += 1;
                }
                // closes the open `<td` tag, then the text for that cell
                ">" => {
                    s.push('>');
                    if let Some(txt) = cell_text.get(ci.saturating_sub(1)) {
                        s.push_str(txt);
                    }
                }
                other => s.push_str(other),
            }
        }
        s.push_str("</table>");
        s
    }
}

pub struct TableModel {
    graph: crate::onnx_graph::Graph,
    device: Device,
}

impl TableModel {
    pub fn new(arch_json: &std::path::Path, weights: &std::path::Path) -> Result<Self> {
        let arch: crate::onnx_graph::Arch = serde_json::from_str(
            &std::fs::read_to_string(arch_json)
                .map_err(|e| Error::Other(format!("table arch: {e}")))?,
        )
        .map_err(|e| Error::Other(format!("table arch parse: {e}")))?;
        let device = Device::Cpu;
        let weights = candle_core::safetensors::load(weights, &device)
            .map_err(crate::image::candle_err)?;
        Ok(Self { graph: crate::onnx_graph::Graph { arch, weights }, device })
    }

    /// `ResizeTableImage(max_len=488)` then `PaddingTableImage(488,488)` — the
    /// reference scales the LONG side to 488 keeping aspect and pads the rest,
    /// it does not stretch.
    ///
    /// Stretching instead was measured: a 497x222 table squashed to a square
    /// decoded to 501 `<pad>` tokens and one stray `<td`. The decoder was
    /// working correctly and reporting, accurately, that it could not see a
    /// table — the input was the defect.
    fn input(&self, img: &ImageBuffer) -> Result<Tensor> {
        let (w, h) = (img.width as usize, img.height as usize);
        let bpp = img.format.bytes_per_pixel();
        let scale = SIDE as f32 / w.max(h) as f32;
        let (rw, rh) = (((w as f32 * scale) as usize).max(1).min(SIDE),
                        ((h as f32 * scale) as usize).max(1).min(SIDE));
        let mut chw = vec![0f32; 3 * SIDE * SIDE];
        for c in 0..3 {
            // BGR: the reference config decodes with `img_mode: BGR`
            let src_c = if bpp == 1 { 0 } else { 2 - c };
            let plane: Vec<f32> =
                (0..w * h).map(|i| img.data[i * bpp + src_c] as f32).collect();
            let r = crate::image::resize_bilinear(&plane, w, h, rw, rh);
            let dst = &mut chw[c * SIDE * SIDE..(c + 1) * SIDE * SIDE];
            // pad region stays at the normalised value of zero-intensity pixels
            let pad = (0.0 - MEAN[c]) / STD[c];
            dst.fill(pad);
            for y in 0..rh {
                for x in 0..rw {
                    dst[y * SIDE + x] = (r[y * rw + x] / 255.0 - MEAN[c]) / STD[c];
                }
            }
        }
        // same oracle hand-off as `formula.rs`: identical bytes to both sides
        if let Ok(p) = std::env::var("FFAI_TABLE_BIN") {
            let mut b = Vec::with_capacity(chw.len() * 4);
            for v in &chw {
                b.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::write(p, b);
        }
        Tensor::from_vec(chw, (1, 3, SIDE, SIDE), &self.device).map_err(crate::image::candle_err)
    }

    /// Recognise the structure of ONE cropped table region.
    pub fn recognize(&self, img: &ImageBuffer) -> Result<TableStructure> {
        // Cell boxes come back normalised to the PADDED SQUARE the model was
        // fed, not to the crop. `ResizeTableImage` scales the long side to 488
        // and pads the short one, so both axes share the long side's scale and
        // both must be denormalised by `max(w, h)`.
        //
        // Using `w` for x and `h` for y is the natural-looking version and it
        // silently compresses every box along the SHORT axis. On a 497x222
        // table that put all seven predicted rows inside the top 45 % (222/497)
        // of the real table, so consecutive rows read the SAME source row twice
        // and the bottom half of the table was never read at all — while the
        // structure, the cell count and every individual crop still looked
        // entirely reasonable.
        let side = (img.width.max(img.height)) as f32;
        let (w, h) = (side, side);
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert("x".into(), self.input(img)?);
        let n_nodes = self.graph.arch.nodes.len();
        let env = self
            .graph
            .run(inputs, n_nodes, &self.device)
            .map_err(crate::image::candle_err)?;

        let logits = env
            .get("fetch_name_1")
            .ok_or_else(|| Error::Other("table: no structure output".into()))?;
        let boxes = env.get("fetch_name_0");

        let ld = logits.dims().to_vec();
        let t_steps = if ld.len() >= 2 { ld[ld.len() - 2] } else { 0 };
        let n_tok = *ld.last().unwrap_or(&0);
        let lv = logits
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(crate::image::candle_err)?;

        if std::env::var("FFAI_TABLE_DEBUG").is_ok() {
            let mut mn = f32::MAX; let mut mx = f32::MIN; let mut s0 = 0f32;
            for v in lv.iter() { mn = mn.min(*v); mx = mx.max(*v); }
            for k in 0..n_tok.min(lv.len()) { s0 += lv[k]; }
            eprintln!("  logits dims {ld:?}  min {mn:.4} max {mx:.4}  step0 sum {s0:.4}");
            eprintln!("  step0 first 12: {:?}", &lv[..lv.len().min(12)]);
            let mid = (t_steps/2)*n_tok;
            if mid + 12 <= lv.len() { eprintln!("  step{} first 12: {:?}", t_steps/2, &lv[mid..mid+12]); }
        }
        let mut tokens = Vec::new();
        let mut cells = Vec::new();
        let bv: Option<Vec<f32>> = boxes.and_then(|b| {
            b.to_dtype(DType::F32).ok().and_then(|t| t.flatten_all().ok()).and_then(|t| t.to_vec1::<f32>().ok())
        });

        for s in 0..t_steps {
            let (mut best, mut bi) = (f32::MIN, 0usize);
            for k in 0..n_tok {
                let v = lv[s * n_tok + k];
                if v > best {
                    best = v;
                    bi = k;
                }
            }
            let tok = TOKENS.get(bi).copied().unwrap_or("?").to_string();
            let stop = tok == "eos";
            if tok == "<td></td>" || tok == "<td" {
                if let Some(b) = &bv {
                    let o = s * 8;
                    if o + 7 < b.len() {
                        let xs = [b[o], b[o + 2], b[o + 4], b[o + 6]];
                        let ys = [b[o + 1], b[o + 3], b[o + 5], b[o + 7]];
                        cells.push(Cell {
                            x0: xs.iter().cloned().fold(f32::MAX, f32::min) * w,
                            y0: ys.iter().cloned().fold(f32::MAX, f32::min) * h,
                            x1: xs.iter().cloned().fold(f32::MIN, f32::max) * w,
                            y1: ys.iter().cloned().fold(f32::MIN, f32::max) * h,
                        });
                    }
                }
            }
            tokens.push(tok);
            if stop {
                break;
            }
        }
        Ok(TableStructure { tokens, cells })
    }
}
