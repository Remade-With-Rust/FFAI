//! The `craft-crnn` engine: CRAFT detection + english_g2 CRNN recognition,
//! composed exactly as the mission plan's §2 stages — detect → group lines →
//! crop → recognize — with each stage independently oracle-tested.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

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

struct Models {
    craft: Craft,
    crnn: Crnn,
    device: Device,
}

pub struct CraftCrnn {
    /// Lazy: weights load on first `recognize`, matching Mercury's contract
    /// (the bench harness warms and records this separately).
    models: OnceLock<std::result::Result<Mutex<Models>, String>>,
    manifest_dir: std::path::PathBuf,
}

impl CraftCrnn {
    pub fn new() -> Self {
        Self::with_manifest_dir(Path::new("models"))
    }

    pub fn with_manifest_dir(dir: &Path) -> Self {
        CraftCrnn { models: OnceLock::new(), manifest_dir: dir.to_path_buf() }
    }

    fn models(&self) -> Result<&Mutex<Models>> {
        let loaded = self.models.get_or_init(|| load_models(&self.manifest_dir).map(Mutex::new).map_err(|e| e.to_string()));
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

fn load_models(dir: &Path) -> Result<Models> {
    let device = Device::Cpu;
    let manifests = ffai_models::load_dir(dir)?;
    let find = |name: &str| {
        manifests
            .iter()
            .find(|m| m.name == name)
            .ok_or_else(|| Error::Model(format!("no model manifest named `{name}` in {}", dir.display())))
    };
    let craft_file = find("craft-mlt")?.fetch()?;
    let crnn_file = find("crnn-english-g2")?.fetch()?;

    let craft = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[craft_file.file("craft.safetensors")?], DType::F32, &device)
        }
        .map_err(image::candle_err)?;
        Craft::new(vb).map_err(image::candle_err)?
    };
    let crnn = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[crnn_file.file("crnn.safetensors")?], DType::F32, &device)
        }
        .map_err(image::candle_err)?;
        Crnn::new(vb).map_err(image::candle_err)?
    };
    Ok(Models { craft, crnn, device })
}

impl OcrEngine for CraftCrnn {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "craft-crnn".into(),
            task: Task::Ocr,
            status: EngineStatus::Experimental,
            description: "CRAFT detection + english_g2 CRNN recognition (EasyOCR lineage), pure Rust on candle"
                .into(),
        }
    }

    fn recognize(&self, img: &ImageBuffer, _opts: &OcrOptions) -> Result<OcrOutput> {
        let models = self.models()?;
        let m = models.lock().map_err(|_| Error::Other("model lock poisoned".into()))?;

        let (w, h) = (img.width as usize, img.height as usize);

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
        let lines = crate::profile::timed(
            |p| &p.boxes,
            || -> Result<_> {
                let (mh, mw, _) = image::ok(maps.dims3())?;
                let flat = image::ok(maps.flatten_all()?.to_vec1::<f32>())?;
                // maps are (h, w, 2) interleaved: region even, affinity odd.
                let region: Vec<f32> = flat.iter().step_by(2).copied().collect();
                let affinity: Vec<f32> = flat.iter().skip(1).step_by(2).copied().collect();
                let word_boxes = boxes::extract_boxes(&region, &affinity, mw, mh);
                Ok(boxes::group_lines(word_boxes))
            },
        )?;

        // ---- recognize, line by line ----
        // Map coords are at (input/2); input coords are original * scale.
        let to_img = |v: usize| (v as f32 * 2.0 / scale).round();
        let mut out_lines = Vec::new();
        for line in &lines {
            let lb = boxes::line_bbox(line);
            let line_h = to_img(lb.y1) - to_img(lb.y0);
            let (px, py) = (line_h * PAD_X, line_h * PAD_Y);
            let x0 = (to_img(lb.x0) - px).max(0.0) as usize;
            let y0 = (to_img(lb.y0) - py).max(0.0) as usize;
            let x1 = (to_img(lb.x1) + px) as usize;
            let y1 = (to_img(lb.y1) + py) as usize;

            let crop = match crate::profile::timed(|p| &p.rec_pre, || {
                image::crnn_input(&gray, w, h, x0, y0, x1, y1, &m.device)
            }) {
                Ok(c) => c,
                Err(_) => continue, // degenerate box: skip, don't fail the page
            };
            let logits = crate::profile::timed(|p| &p.rec_fwd, || m.crnn.forward(&crop))
                .map_err(image::candle_err)?;
            let (text, confidence) =
                crate::profile::timed(|p| &p.decode, || ctc_greedy(&logits)).map_err(image::candle_err)?;
            if text.is_empty() {
                continue;
            }
            out_lines.push(OcrLine {
                text,
                words: Vec::new(), // line-level recognition; word detail is open work
                bbox: Some(BoundingBox {
                    x: x0 as f32,
                    y: y0 as f32,
                    width: (x1 - x0) as f32,
                    height: (y1 - y0) as f32,
                }),
                confidence,
            });
        }

        // v1: one block per page — paragraph segmentation is the DOCUMENT
        // milestone's work, and inventing it early would be unearned.
        Ok(OcrOutput { blocks: vec![OcrBlock { lines: out_lines, bbox: None }] })
    }
}
