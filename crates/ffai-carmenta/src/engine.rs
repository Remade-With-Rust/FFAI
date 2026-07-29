//! The `craft-crnn` engine: CRAFT detection + english_g2 CRNN recognition,
//! composed exactly as the mission plan's §2 stages — detect → group lines →
//! crop → recognize — with each stage independently oracle-tested.

use std::path::Path;
use std::sync::OnceLock;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use ffai_core::engine::{EngineInfo, EngineStatus, OcrEngine, OcrOptions, Task};
use ffai_core::error::{Error, Result};
use ffai_core::types::{BoundingBox, ImageBuffer, OcrBlock, OcrLine, OcrOutput};

use crate::boxes;
use crate::craft::Craft;
use crate::crnn::{ctc_greedy, Crnn};
use crate::image;

/// Crop padding around a detected line, as fractions of line height. CRAFT
/// region scores hug character cores; without context the recognizer clips
/// ascenders/descenders. Tuned on the TRAIN split of carmenta-render-v1
/// (holdout stays legal for claims).
const PAD_Y: f32 = 0.35;
const PAD_X: f32 = 0.25;

/// Which recognition stage this engine composes over CRAFT detection.
#[derive(Clone, Copy, PartialEq)]
pub enum RecStage {
    /// english_g2 CRNN: LINE-level crops (arbitrary width, CTC).
    Crnn,
    /// PARSeq-tiny: WORD-level 32x128 crops (25-char AR decoder), joined
    /// with spaces within each detected line.
    Parseq,
}

struct Models {
    craft: Craft,
    crnn: Option<Crnn>,
    parseq: Option<crate::parseq::Parseq>,
    device: Device,
}

pub struct CraftCrnn {
    /// Lazy: weights load on first `recognize`, matching Mercury's contract
    /// (the bench harness warms and records this separately).
    models: OnceLock<std::result::Result<Models, String>>,
    manifest_dir: std::path::PathBuf,
    rec: RecStage,
}

impl CraftCrnn {
    pub fn new() -> Self {
        Self::with_manifest_dir(Path::new("models"))
    }

    /// The `craft-parseq` variant: same detection, PARSeq-tiny recognition.
    pub fn new_parseq() -> Self {
        CraftCrnn {
            models: OnceLock::new(),
            manifest_dir: Path::new("models").to_path_buf(),
            rec: RecStage::Parseq,
        }
    }

    pub fn with_manifest_dir(dir: &Path) -> Self {
        CraftCrnn { models: OnceLock::new(), manifest_dir: dir.to_path_buf(), rec: RecStage::Crnn }
    }

    fn models(&self) -> Result<&Models> {
        let rec = self.rec;
        let loaded = self
            .models
            .get_or_init(|| load_models(&self.manifest_dir, rec).map_err(|e| e.to_string()));
        match loaded {
            Ok(m) => Ok(m),
            Err(e) => Err(Error::Model(e.clone())),
        }
    }
}

impl Default for CraftCrnn {
    fn default() -> Self {
        Self::new()
    }
}

fn load_models(dir: &Path, rec: RecStage) -> Result<Models> {
    let device = Device::Cpu;
    let manifests = ffai_models::load_dir(dir)?;
    let find = |name: &str| {
        manifests
            .iter()
            .find(|m| m.name == name)
            .ok_or_else(|| Error::Model(format!("no model manifest named `{name}` in {}", dir.display())))
    };
    let load_vb = |file: std::path::PathBuf| {
        unsafe { VarBuilder::from_mmaped_safetensors(&[file], DType::F32, &device) }
            .map_err(image::candle_err)
    };
    let craft_file = find("craft-mlt")?.fetch()?;
    let craft = Craft::new(load_vb(craft_file.file("craft.safetensors")?.to_path_buf())?)
        .map_err(image::candle_err)?;
    let (crnn, parseq) = match rec {
        RecStage::Crnn => {
            let f = find("crnn-english-g2")?.fetch()?;
            let m = Crnn::new(load_vb(f.file("crnn.safetensors")?.to_path_buf())?)
                .map_err(image::candle_err)?;
            (Some(m), None)
        }
        RecStage::Parseq => {
            let f = find("parseq-tiny")?.fetch()?;
            let m = crate::parseq::Parseq::new(load_vb(f.file("parseq-tiny.safetensors")?.to_path_buf())?)
                .map_err(image::candle_err)?;
            (None, Some(m))
        }
    };
    Ok(Models { craft, crnn, parseq, device })
}

impl OcrEngine for CraftCrnn {
    fn info(&self) -> EngineInfo {
        let (name, description) = match self.rec {
            RecStage::Crnn => (
                "craft-crnn",
                "CRAFT detection + english_g2 CRNN recognition (EasyOCR lineage), pure Rust on candle",
            ),
            RecStage::Parseq => (
                "craft-parseq",
                "CRAFT detection + PARSeq-tiny recognition (word-level AR decoder), pure Rust on candle",
            ),
        };
        EngineInfo {
            name: name.into(),
            task: Task::Ocr,
            status: EngineStatus::Experimental,
            description: description.into(),
        }
    }

    fn recognize(&self, img: &ImageBuffer, opts: &OcrOptions) -> Result<OcrOutput> {
        let m = self.models()?;

        let (w, h) = (img.width as usize, img.height as usize);

        // --psm-7 analog: the caller vouches the image is ONE text line
        // (LIVE dirty bands with known single-line geometry). Detection is
        // skipped entirely — the CRNN reads the whole frame as a line crop.
        // Falls through to the full path for the word-level (parseq) stage,
        // which needs the maps for word splitting.
        if opts.single_line && self.rec == RecStage::Crnn {
            let gray = crate::profile::timed(|p| &p.det_pre, || image::to_gray_f32(img))?;
            let crnn = m.crnn.as_ref().expect("crnn loaded for RecStage::Crnn");
            // Ink-extent x-trim: CRNN cost is linear in strip WIDTH (the
            // LSTM walks the columns), and a full-width band is mostly
            // margin. The extent is measured on the CURRENT pixels (column
            // deviation from the strip's median background), never on cached
            // geometry — cached extents clip text that just grew longer.
            let (x0, x1) = if std::env::var("FFAI_NO_TRIM").is_ok() {
                (0, w)
            } else {
                image::ink_extent(&gray, w, h)
            };
            let crop = crate::profile::timed(|p| &p.rec_pre, || {
                image::crnn_input(&gray, w, h, x0, 0, x1, h, &m.device)
            })?;
            let logits = crate::profile::timed(|p| &p.rec_fwd, || crnn.forward(&crop))
                .map_err(image::candle_err)?;
            let (text, confidence) = crate::profile::timed(|p| &p.decode, || ctc_greedy(&logits))
                .map_err(image::candle_err)?;
            let mut lines = Vec::new();
            if !text.is_empty() {
                lines.push(OcrLine {
                    text,
                    words: Vec::new(),
                    bbox: Some(BoundingBox { x: 0.0, y: 0.0, width: w as f32, height: h as f32 }),
                    confidence,
                });
            }
            return Ok(OcrOutput { blocks: vec![OcrBlock { lines, bbox: None }] });
        }

        // ---- detect ----
        let (gray, input, scale) = crate::profile::timed(
            |p| &p.det_pre,
            || -> Result<_> {
                let gray = image::to_gray_f32(img)?;
                let (input, scale) = image::craft_input(&gray, w, h, &m.device)?;
                Ok((gray, input, scale))
            },
        )?;
        let maps = crate::profile::timed(|p| &p.det_fwd, || m.craft.forward(&input))
            .map_err(image::candle_err)?;
        let (lines, region, affinity, mw) = crate::profile::timed(
            |p| &p.boxes,
            || -> Result<_> {
                let (mh, mw, _) = image::ok(maps.dims3())?;
                let flat = image::ok(maps.flatten_all()?.to_vec1::<f32>())?;
                // maps are (h, w, 2) interleaved: region even, affinity odd.
                let region: Vec<f32> = flat.iter().step_by(2).copied().collect();
                let affinity: Vec<f32> = flat.iter().skip(1).step_by(2).copied().collect();
                let word_boxes = boxes::extract_boxes(&region, &affinity, mw, mh);
                Ok((boxes::group_lines(word_boxes), region, affinity, mw))
            },
        )?;

        // ---- recognize, line by line, ACROSS CORES ----
        // Lines are independent; recognition was the serial half of the
        // frame after detection banding. Models hold only candle tensors
        // (Send+Sync), so rayon fans lines out with zero locking; profile
        // stages accumulate atomically.
        // Map coords are at (input/2); input coords are original * scale.
        use rayon::prelude::*;
        let to_img = |v: usize| (v as f32 * 2.0 / scale).round();
        // Parallel across lines ONLY when there are enough to amortize the
        // pool contention with candle's internal rayon: measured, a 1-line
        // band strip pays 177 ms/line under par_iter vs 82 ms serial, while
        // a 7-line frame wins 2.07s -> 1.65s. Threshold 3 splits the cases.
        let recognize_line = |line: &Vec<boxes::DetBox>| -> Result<Option<OcrLine>> {
                let lb = boxes::line_bbox(line);
                let line_h = to_img(lb.y1) - to_img(lb.y0);
                let (px, py) = (line_h * PAD_X, line_h * PAD_Y);
                let x0 = (to_img(lb.x0) - px).max(0.0) as usize;
                let y0 = (to_img(lb.y0) - py).max(0.0) as usize;
                let x1 = (to_img(lb.x1) + px) as usize;
                let y1 = (to_img(lb.y1) + py) as usize;

                let (text, confidence) = match self.rec {
                    RecStage::Crnn => {
                        let crnn = m.crnn.as_ref().expect("crnn loaded for RecStage::Crnn");
                        let crop = match crate::profile::timed(|p| &p.rec_pre, || {
                            image::crnn_input(&gray, w, h, x0, y0, x1, y1, &m.device)
                        }) {
                            Ok(c) => c,
                            Err(_) => return Ok(None), // degenerate box: skip, not fail
                        };
                        let logits = crate::profile::timed(|p| &p.rec_fwd, || crnn.forward(&crop))
                            .map_err(image::candle_err)?;
                        crate::profile::timed(|p| &p.decode, || ctc_greedy(&logits))
                            .map_err(image::candle_err)?
                    }
                    RecStage::Parseq => {
                        // Word-level: recognize each split box, join with spaces.
                        let parseq = m.parseq.as_ref().expect("parseq loaded for RecStage::Parseq");
                        let mut words = Vec::new();
                        let mut confs = Vec::new();
                        let lb_map = boxes::line_bbox(line);
                        let word_boxes = boxes::split_words(&region, &affinity, mw, &lb_map);
                        for b in word_boxes.iter() {
                            let bh = to_img(b.y1) - to_img(b.y0);
                            let (wpx, wpy) = (bh * 0.08, bh * 0.12);
                            let bx0 = (to_img(b.x0) - wpx).max(0.0) as usize;
                            let by0 = (to_img(b.y0) - wpy).max(0.0) as usize;
                            let bx1 = ((to_img(b.x1) + wpx) as usize).min(w);
                            let by1 = ((to_img(b.y1) + wpy) as usize).min(h);
                            if bx1 <= bx0 + 1 || by1 <= by0 + 1 {
                                continue;
                            }
                            let (cw, ch) = (bx1 - bx0, by1 - by0);
                            let mut crop = vec![0f32; cw * ch];
                            for y in 0..ch {
                                crop[y * cw..(y + 1) * cw]
                                    .copy_from_slice(&gray[(by0 + y) * w + bx0..(by0 + y) * w + bx1]);
                            }
                            let x = crate::profile::timed(|p| &p.rec_pre, || {
                                crate::parseq::parseq_input(&crop, cw, ch, &m.device)
                            })
                            .map_err(image::candle_err)?;
                            let (wtext, wconf) =
                                crate::profile::timed(|p| &p.rec_fwd, || parseq.recognize(&x))
                                    .map_err(image::candle_err)?;
                            if !wtext.is_empty() {
                                words.push(wtext);
                                if let Some(c) = wconf {
                                    confs.push(c);
                                }
                            }
                        }
                        let conf = if confs.is_empty() {
                            None
                        } else {
                            Some(confs.iter().sum::<f32>() / confs.len() as f32)
                        };
                        (words.join(" "), conf)
                    }
                };
                if text.is_empty() {
                    return Ok(None);
                }
                Ok(Some(OcrLine {
                    text,
                    words: Vec::new(), // line-level recognition; word detail is open work
                    bbox: Some(BoundingBox {
                        x: x0 as f32,
                        y: y0 as f32,
                        width: (x1 - x0) as f32,
                        height: (y1 - y0) as f32,
                    }),
                    confidence,
                }))
        };
        let results: Vec<Option<OcrLine>> = if lines.len() >= 3 {
            lines.par_iter().map(&recognize_line).collect::<Result<Vec<_>>>()?
        } else {
            lines.iter().map(&recognize_line).collect::<Result<Vec<_>>>()?
        };
        let out_lines: Vec<OcrLine> = results.into_iter().flatten().collect();

        // v1: one block per page — paragraph segmentation is the DOCUMENT
        // milestone's work, and inventing it early would be unearned.
        Ok(OcrOutput { blocks: vec![OcrBlock { lines: out_lines, bbox: None }] })
    }
}
