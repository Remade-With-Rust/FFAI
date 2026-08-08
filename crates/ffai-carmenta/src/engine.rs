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
    /// PP-OCRv5 mobile rec (SVTR): the matched half of the mobile DETECTOR we
    /// already ship. LINE crops at 3x48xW BGR, 18,385-class CTC. Opt-in by
    /// engine name until an A/B says otherwise (§8.165-168).
    Svtr,
}

/// Which detector feeds the recognition stage.
#[derive(Clone, Copy, PartialEq)]
pub enum DetStage {
    /// CRAFT: character heatmaps, reassembled into words by the component walk.
    Craft,
    /// PP-OCRv5 mobile-det (DBNet/PP-LCNetV3): emits text-LINE regions
    /// directly, from 4.7 MB against CRAFT's VGG16.
    MobileDet,
    /// Both, composed: DBNet supplies the LINE GROUPING, CRAFT supplies the
    /// WORD BOXES. Deliberately the expensive option — it runs both detectors —
    /// because it isolates a question neither alone can answer: when CRAFT
    /// loses, is it the box geometry or is it `group_lines`' heuristic
    /// stitching those boxes into the wrong lines?
    Composed,
}

/// Minimum short side the mobile-det input is scaled UP to; larger images pass
/// through untouched (capped at `FFAI_DET_MAX_SIDE`). See
/// `image::mobiledet_input` for why this is a floor and not a ceiling — the
/// reading that made it a ceiling cost a factor of 17 in detector input.
/// Swept per corpus because detection dominates frame time.
fn mobiledet_min_side() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("FFAI_DET_MIN_SIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(736)
    })
}

/// Crop pads for the mobile-det path, as fractions of line height. Zero by
/// default because DB's unclip has already expanded the box; sweepable because
/// "already expanded" is a claim a gate should get to test.
fn mobiledet_pads() -> (f32, f32) {
    static V: OnceLock<(f32, f32)> = OnceLock::new();
    *V.get_or_init(|| {
        let get = |k: &str| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        (get("FFAI_MDET_PAD_X"), get("FFAI_MDET_PAD_Y"))
    })
}

/// Minimum recognition confidence for a line to be emitted at all.
///
/// Zero by default — off until a gate says otherwise, and the sweep that sets
/// it lives on the CORD train split. Raising it trades deletions for
/// insertions, so the optimum is wherever that exchange stops paying.
fn reject_threshold() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("FFAI_REJECT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)
    })
}

struct Models {
    craft: Option<Craft>,
    mobiledet: Option<crate::mobiledet::MobileDet>,
    crnn: Option<Crnn>,
    parseq: Option<crate::parseq::Parseq>,
    svtr: Option<(crate::svtr::Svtr, Vec<String>)>,
    device: Device,
}

pub struct CraftCrnn {
    /// Lazy: weights load on first `recognize`, matching Mercury's contract
    /// (the bench harness warms and records this separately).
    models: OnceLock<std::result::Result<Models, String>>,
    manifest_dir: std::path::PathBuf,
    rec: RecStage,
    det: DetStage,
}

impl CraftCrnn {
    pub fn new() -> Self {
        Self::with_manifest_dir(Path::new("models"))
    }

    /// The `craft-parseq` variant: same detection, PARSeq-tiny recognition.
    pub fn new_parseq() -> Self {
        Self::variant(RecStage::Parseq, DetStage::Craft)
    }

    /// The mobile-det variants: DBNet line regions instead of CRAFT's
    /// reassembled character components.
    pub fn new_mobiledet(rec: RecStage) -> Self {
        Self::variant(rec, DetStage::MobileDet)
    }

    /// The composed variant: both detectors, DBNet grouping CRAFT's words.
    pub fn new_composed(rec: RecStage) -> Self {
        Self::variant(rec, DetStage::Composed)
    }

    fn variant(rec: RecStage, det: DetStage) -> Self {
        CraftCrnn {
            models: OnceLock::new(),
            manifest_dir: Path::new("models").to_path_buf(),
            rec,
            det,
        }
    }

    pub fn with_manifest_dir(dir: &Path) -> Self {
        CraftCrnn {
            models: OnceLock::new(),
            manifest_dir: dir.to_path_buf(),
            rec: RecStage::Crnn,
            det: DetStage::Craft,
        }
    }

    fn models(&self) -> Result<&Models> {
        let (rec, det) = (self.rec, self.det);
        let loaded = self
            .models
            .get_or_init(|| load_models(&self.manifest_dir, rec, det).map_err(|e| e.to_string()));
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

fn load_models(dir: &Path, rec: RecStage, det: DetStage) -> Result<Models> {
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
    let (craft, mobiledet) = match det {
        DetStage::Craft => {
            let f = find("craft-mlt")?.fetch()?;
            let m = Craft::new(load_vb(f.file("craft.safetensors")?.to_path_buf())?)
                .map_err(image::candle_err)?;
            (Some(m), None)
        }
        DetStage::MobileDet => {
            let f = find("ppocrv5-mobile-det")?.fetch()?;
            let m = crate::mobiledet::MobileDet::new(load_vb(
                f.file("det-fused.safetensors")?.to_path_buf(),
            )?)
            .map_err(image::candle_err)?;
            (None, Some(m))
        }
        DetStage::Composed => {
            let cf = find("craft-mlt")?.fetch()?;
            let c = Craft::new(load_vb(cf.file("craft.safetensors")?.to_path_buf())?)
                .map_err(image::candle_err)?;
            let df = find("ppocrv5-mobile-det")?.fetch()?;
            let d = crate::mobiledet::MobileDet::new(load_vb(
                df.file("det-fused.safetensors")?.to_path_buf(),
            )?)
            .map_err(image::candle_err)?;
            (Some(c), Some(d))
        }
    };
    let mut svtr = None;
    let (crnn, parseq) = match rec {
        RecStage::Crnn => {
            // §8.143: `FFAI_REC_LANG=zh` swaps in `zh_sim_g2`, whose head is
            // 6 719 classes against English's 97 and whose charset covers the
            // same ASCII plus 6 614 CJK characters. Architecturally identical
            // otherwise, so the same code path runs both. English is the
            // default and remains the oracle.
            let lang = crate::crnn::RecLang::from_env();
            let f = find(lang.model_name())?.fetch()?;
            let weights = f.file("crnn.safetensors")?.to_path_buf();
            let charset = crate::crnn::charset_for(lang, weights.parent())
                .map_err(image::candle_err)?;
            let m = Crnn::new_with_charset(load_vb(weights)?, charset)
                .map_err(image::candle_err)?;
            (Some(m), None)
        }
        RecStage::Parseq => {
            let f = find("parseq-tiny")?.fetch()?;
            let m = crate::parseq::Parseq::new(load_vb(f.file("parseq-tiny.safetensors")?.to_path_buf())?)
                .map_err(image::candle_err)?;
            (None, Some(m))
        }
        RecStage::Svtr => {
            let f = find("ppocrv5-mobile-rec")?.fetch()?;
            let weights = f.file("rec.safetensors")?.to_path_buf();
            let charset = crate::svtr::load_charset(&f.file("charset.txt")?.to_path_buf())
                .map_err(image::candle_err)?;
            // Head classes = charset + 1 blank. Asserted at load rather than
            // trusted: an off-by-one shifts every decoded character (§8.168).
            let m = crate::svtr::Svtr::new(load_vb(weights)?, charset.len() + 1)
                .map_err(image::candle_err)?;
            svtr = Some((m, charset));
            (None, None)
        }
    };
    Ok(Models { craft, mobiledet, crnn, parseq, svtr, device })
}

impl OcrEngine for CraftCrnn {
    fn info(&self) -> EngineInfo {
        let (name, description) = match (self.det, self.rec) {
            (DetStage::Craft, RecStage::Crnn) => (
                "craft-crnn",
                "CRAFT detection + english_g2 CRNN recognition (EasyOCR lineage), pure Rust on candle",
            ),
            (DetStage::MobileDet, RecStage::Svtr) => (
                "mobiledet-svtr",
                "PP-OCRv5 mobile det + PP-OCRv5 mobile rec (SVTR) — the matched pair, pure Rust on candle",
            ),
            (_, RecStage::Svtr) => (
                "svtr",
                "PP-OCRv5 mobile rec (SVTR) recognition",
            ),
            (DetStage::Craft, RecStage::Parseq) => (
                "craft-parseq",
                "CRAFT detection + PARSeq-tiny recognition (word-level AR decoder), pure Rust on candle",
            ),
            (DetStage::MobileDet, RecStage::Crnn) => (
                "mobiledet-crnn",
                "PP-OCRv5 mobile-det (DBNet/PP-LCNetV3) + english_g2 CRNN, pure Rust on candle",
            ),
            (DetStage::MobileDet, RecStage::Parseq) => (
                "mobiledet-parseq",
                "PP-OCRv5 mobile-det (DBNet/PP-LCNetV3) + PARSeq-tiny, pure Rust on candle",
            ),
            (DetStage::Composed, RecStage::Crnn) => (
                "composed-crnn",
                "DBNet line grouping over CRAFT word boxes + english_g2 CRNN (both detectors)",
            ),
            (DetStage::Composed, RecStage::Parseq) => (
                "composed-parseq",
                "DBNet line grouping over CRAFT word boxes + PARSeq-tiny (both detectors)",
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
            let (text, confidence) = crate::profile::timed(|p| &p.decode, || crnn.decode(&logits))
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
        let content_kind = crate::content::classify(img);
        let gray = crate::profile::timed(|p| &p.det_pre, || image::to_gray_f32(img))?;
        // `scale` is carried so the crop stage can map detector coordinates
        // back to image pixels via `to_img` below; mobile-det already emits
        // image coordinates, which is why its scale is the identity 2.0.
        let (lines, region, affinity, mw, scale) = match self.det {
            DetStage::Craft => {
                // Detection sees COLOR (chroma contrast is real signal on
                // photographs); recognition crops stay grayscale (both rec
                // lineages take gray, and their gates were earned on it).
                let (input, scale) = crate::profile::timed(|p| &p.det_pre, || {
                    image::craft_input_color(img, &m.device)
                })?;
                let craft = m.craft.as_ref().expect("craft loaded for DetStage::Craft");
                let maps = crate::profile::timed(|p| &p.det_fwd, || craft.forward(&input))
                    .map_err(image::candle_err)?;
                crate::profile::timed(
                    |p| &p.boxes,
                    || -> Result<_> {
                        let (mh, mw, _) = image::ok(maps.dims3())?;
                        let flat = image::ok(maps.flatten_all()?.to_vec1::<f32>())?;
                        // maps are (h, w, 2) interleaved: region even, affinity odd.
                        let region: Vec<f32> = flat.iter().step_by(2).copied().collect();
                        let affinity: Vec<f32> = flat.iter().skip(1).step_by(2).copied().collect();
                        let word_boxes = boxes::extract_boxes(&region, &affinity, mw, mh);
                        Ok((
                            boxes::order_reading(boxes::group_lines(word_boxes), mw),
                            region, affinity, mw, scale,
                        ))
                    },
                )?
            }
            DetStage::MobileDet => {
                let (input, sx, sy) = crate::profile::timed(|p| &p.det_pre, || {
                    image::mobiledet_input(img, mobiledet_min_side(), &m.device)
                })?;
                let det = m.mobiledet.as_ref().expect("mobiledet loaded for DetStage::MobileDet");
                let prob = crate::profile::timed(|p| &p.det_fwd, || det.forward(&input))
                    .map_err(image::candle_err)?;
                crate::profile::timed(
                    |p| &p.boxes,
                    || -> Result<_> {
                        let (_, _, ph, pw) = image::ok(prob.dims4())?;
                        let flat = image::ok(prob.flatten_all()?.to_vec1::<f32>())?;
                        // Unclip is a property of the RECOGNIZER's crop, not
                        // of the detector — see `UNCLIP_LINE` / `UNCLIP_WORD`.
                        let unclip = match self.rec {
                            // SVTR is a LINE recognizer like CRNN, so it takes
                            // the line unclip. Holding this constant is also
                            // what keeps the A/B honest: detection identical,
                            // only the recognizer differs.
                            RecStage::Crnn | RecStage::Svtr => crate::mobiledet::UNCLIP_LINE,
                            RecStage::Parseq => crate::mobiledet::UNCLIP_WORD,
                        };
                        // FFAI_DUMP_PROB writes the raw text-probability map so
                        // a probe can ask whether it separates figures better
                        // than "absence of boxes" does. Boxes are thresholded
                        // FROM this map, so the marginal information is exactly
                        // what a continuous value adds over a binary one — and
                        // ink alone was refuted at 1.8x on both axes (§8.71).
                        if let Ok(dir) = std::env::var("FFAI_DUMP_PROB") {
                            let tag = std::env::var("FFAI_TRACE_TAG").unwrap_or_default();
                            let _ = std::fs::create_dir_all(&dir);
                            let mut out = format!("{pw} {ph}
");
                            for v in &flat {
                                out.push_str(&format!("{v:.3} "));
                            }
                            let _ = std::fs::write(format!("{dir}/{tag}.prob"), out);
                        }
                        let mut b = crate::mobiledet::boxes_from_probability(
                            &flat, pw, ph, sx, sy, unclip,
                        );
                        // DBNet emits text LINES, so there is nothing to group
                        // — each region IS a line. Sorting is still needed:
                        // the component walk returns raster order, and reading
                        // order is what the output contract promises.
                        b.sort_by_key(|r| (r.y0, r.x0));
                        // §8.88: split any box carrying a white corridor wide
                        // enough to be a column gutter, read from the SOURCE
                        // PIXELS. The probability map cannot arbitrate this —
                        // it reads 1.000 at the very gutter that was bridged,
                        // and that hallucination is the defect.
                        let mut b = boxes::split_at_white_corridor(b, &gray, w, h);
                        if std::env::var("FFAI_DET_DEBUG").is_ok() {
                            eprintln!(
                                "mobiledet: {} boxes; image {w}x{h} -> map {pw}x{ph} (sx {sx:.3} sy {sy:.3})",
                                b.len()
                            );
                            for r in &b {
                                eprintln!(
                                    "   [{},{},{},{}] {}x{} score {:.3}",
                                    r.x0, r.y0, r.x1, r.y1, r.x1 - r.x0, r.y1 - r.y0, r.score
                                );
                            }
                        }
                        let lines: Vec<Vec<boxes::DetBox>> =
                            b.into_iter().map(|r| vec![r]).collect();
                        Ok((boxes::order_reading(lines, w), Vec::new(), Vec::new(), 0, 2.0))
                    },
                )?
            }
            DetStage::Composed => {
                // Both detectors. CRAFT's components are the word boxes; DBNet's
                // regions decide which words share a line. Everything downstream
                // therefore sees CRAFT geometry in image coordinates, which is
                // why the scale is the identity 2.0.
                let (cinput, cscale) = crate::profile::timed(|p| &p.det_pre, || {
                    image::craft_input_color(img, &m.device)
                })?;
                let craft = m.craft.as_ref().expect("craft loaded for DetStage::Composed");
                let det = m.mobiledet.as_ref().expect("mobiledet loaded for DetStage::Composed");
                let maps = crate::profile::timed(|p| &p.det_fwd, || craft.forward(&cinput))
                    .map_err(image::candle_err)?;
                let (dinput, sx, sy) = crate::profile::timed(|p| &p.det_pre, || {
                    image::mobiledet_input(img, mobiledet_min_side(), &m.device)
                })?;
                let prob = crate::profile::timed(|p| &p.det_fwd, || det.forward(&dinput))
                    .map_err(image::candle_err)?;
                crate::profile::timed(
                    |p| &p.boxes,
                    || -> Result<_> {
                        let (mh, mwid, _) = image::ok(maps.dims3())?;
                        let flat = image::ok(maps.flatten_all()?.to_vec1::<f32>())?;
                        let region: Vec<f32> = flat.iter().step_by(2).copied().collect();
                        let affinity: Vec<f32> = flat.iter().skip(1).step_by(2).copied().collect();
                        let k = 2.0 / cscale;
                        let words: Vec<boxes::DetBox> =
                            boxes::extract_boxes(&region, &affinity, mwid, mh)
                                .into_iter()
                                .map(|b| boxes::DetBox {
                                    x0: (b.x0 as f32 * k).round() as usize,
                                    y0: (b.y0 as f32 * k).round() as usize,
                                    x1: (b.x1 as f32 * k).round() as usize,
                                    y1: (b.y1 as f32 * k).round() as usize,
                                    score: b.score,
                                })
                                .collect();

                        let (_, _, ph, pw) = image::ok(prob.dims4())?;
                        let pflat = image::ok(prob.flatten_all()?.to_vec1::<f32>())?;
                        let regions = crate::mobiledet::boxes_from_probability(
                            &pflat, pw, ph, sx, sy, crate::mobiledet::UNCLIP_LINE,
                        );

                        // A word joins the region containing its centre. Words
                        // landing in no region are NOT dropped — they fall back
                        // to `group_lines`, so composing can change grouping but
                        // can never lose text.
                        let mut groups: Vec<Vec<boxes::DetBox>> = vec![Vec::new(); regions.len()];
                        let mut orphans = Vec::new();
                        for wb in words {
                            let cx = (wb.x0 + wb.x1) / 2;
                            let cy = (wb.y0 + wb.y1) / 2;
                            match regions
                                .iter()
                                .position(|r| cx >= r.x0 && cx < r.x1 && cy >= r.y0 && cy < r.y1)
                            {
                                Some(i) => groups[i].push(wb),
                                None => orphans.push(wb),
                            }
                        }
                        let mut lines: Vec<Vec<boxes::DetBox>> = groups
                            .into_iter()
                            .filter(|g| !g.is_empty())
                            .map(|mut g| {
                                g.sort_by_key(|b| b.x0);
                                g
                            })
                            .collect();
                        lines.extend(boxes::group_lines(orphans));
                        lines.sort_by_key(|l| boxes::line_bbox(l).y0);
                        Ok((lines, region, affinity, mwid, 2.0))
                    },
                )?
            }
        };

        // ---- recognize, line by line, ACROSS CORES ----
        // Lines are independent; recognition was the serial half of the
        // frame after detection banding. Models hold only candle tensors
        // (Send+Sync), so rayon fans lines out with zero locking; profile
        // stages accumulate atomically.
        // Map coords are at (input/2); input coords are original * scale.
        use rayon::prelude::*;
        let to_img = |v: usize| (v as f32 * 2.0 / scale).round();
        // Mobile-det gives one line box per region rather than per-word
        // components, so the photographic per-word path has nothing to iterate;
        // the ink-gap splitter supplies word boundaries inside each line.
        let det_is_line_level = self.det == DetStage::MobileDet;
        // Parallel across lines ONLY when there are enough to amortize the
        // pool contention with candle's internal rayon: measured, a 1-line
        // band strip pays 177 ms/line under par_iter vs 82 ms serial, while
        // a 7-line frame wins 2.07s -> 1.65s. Threshold 3 splits the cases.
        let recognize_line = |line: &Vec<boxes::DetBox>| -> Result<Option<OcrLine>> {
                let lb = boxes::line_bbox(line);
                let line_h = to_img(lb.y1) - to_img(lb.y0);
                // CRAFT's boxes hug character cores and need padding; DBNet's
                // arrive ALREADY unclipped, and the two expansions compound
                // badly. DB's offset is `area * 1.5 / perimeter`, which on a
                // wide line is enormous vertically: a 1900x90 line grows 64 px
                // on every side — 72 % of its own height — before the crop pad
                // adds another 35 %. Padding a second time hands the recognizer
                // mostly background and the neighbouring lines.
                let (pad_x, pad_y) = match self.det {
                    // Composed crops CRAFT's word boxes, so it inherits CRAFT's
                    // pads — the detector that produced the geometry is the one
                    // whose padding was tuned for it.
                    DetStage::Craft | DetStage::Composed => (PAD_X, PAD_Y),
                    DetStage::MobileDet => mobiledet_pads(),
                };
                let (px, py) = (line_h * pad_x, line_h * pad_y);
                let x0 = (to_img(lb.x0) - px).max(0.0) as usize;
                let y0 = (to_img(lb.y0) - py).max(0.0) as usize;
                let x1 = (to_img(lb.x1) + px) as usize;
                let y1 = (to_img(lb.y1) + py) as usize;

                let (text, confidence) = match self.rec {
                    RecStage::Svtr => {
                        let (svtr, charset) =
                            m.svtr.as_ref().expect("svtr loaded for RecStage::Svtr");
                        let crop = match crate::profile::timed(|p| &p.rec_pre, || {
                            crate::svtr::svtr_input(img, x0, y0, x1, y1, &m.device)
                        }) {
                            Ok(c) => c,
                            Err(_) => return Ok(None),
                        };
                        let probs = crate::profile::timed(|p| &p.rec_fwd, || svtr.forward(&crop))
                            .map_err(image::candle_err)?;
                        crate::profile::timed(|p| &p.decode, || {
                            crate::svtr::ctc_greedy(&probs, charset)
                        })
                        .map_err(image::candle_err)?
                    }
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
                        crate::profile::timed(|p| &p.decode, || crnn.decode(&logits))
                            .map_err(image::candle_err)?
                    }
                    RecStage::Parseq => {
                        // Word-level: recognize each split box, join with spaces.
                        let parseq = m.parseq.as_ref().expect("parseq loaded for RecStage::Parseq");
                        let mut words = Vec::new();
                        let mut confs = Vec::new();
                        // WORD BOXES COME FROM CRAFT DIRECTLY.
                        //
                        // The ink-gap splitter used to re-derive word
                        // boundaries from the line's pixels, and it is the
                        // measured reason parseq read 33.4% in-pipeline
                        // while reading 1.5% on true crops. The census says
                        // the re-derivation was never needed: 1230 CRAFT
                        // boxes against 1027 ground-truth words on the CORD
                        // holdout — CRAFT's connected components ARE words.
                        // FFAI_PARSEQ_SPLIT=1 restores the old path for A/B.
                        // DISPATCH (mission plan §4). A measured, TWO-SIDED
                        // sign-flip: the ink-gap splitter is 4.5x better on
                        // rendered text (0.149 % vs 0.673 %) and 1.6x worse
                        // on photographs (34.16 % vs 21.70 %). Clean rendered
                        // gaps suit a projection; camera noise, tilt and
                        // variable spacing suit CRAFT's learned affinity.
                        // Neither wins everywhere, so the strategy is chosen
                        // per image — see `content` for the signal and its
                        // empty-band margin.
                        let word_ranges: Vec<(usize, usize, usize, usize)> = match (det_is_line_level, content_kind) {
                            // Rendered text takes the ink-gap projection, and
                            // so does ANY line-level detector: DBNet hands us
                            // one box per line, so there are no per-word
                            // components to iterate and the gaps are all we
                            // have to go on.
                            (_, crate::content::ContentKind::Rendered) | (true, _) => {
                                let lb_map = boxes::line_bbox(line);
                                let bh = (to_img(lb_map.y1) - to_img(lb_map.y0)).max(1.0);
                                let sy0 = (to_img(lb_map.y0) - bh * 0.12).max(0.0) as usize;
                                let sy1 = ((to_img(lb_map.y1) + bh * 0.12) as usize).min(h);
                                let sx0 = (to_img(lb_map.x0) - bh * 0.5).max(0.0) as usize;
                                let sx1 = ((to_img(lb_map.x1) + bh * 0.5) as usize).min(w);
                                image::split_ink_words(&gray, w, sy0, sy1, sx0, sx1)
                                    .into_iter()
                                    .map(|(a, b)| (a, sy0, b, sy1))
                                    .collect()
                            }
                            // PER-WORD geometry, expanded by the MEASURED
                            // bias. Matched against CORD's ground-truth word
                            // quads our boxes run 0.69x their height and
                            // 0.82x their width, shifted +0.17/+0.18 — and
                            // reading the same words from our boxes instead
                            // of theirs costs PARSeq 1.30 % -> 17.41 % CER.
                            //
                            // The expansion happens HERE, at crop time, and
                            // NOT in `extract_boxes`: widening the detector's
                            // boxes changes line GROUPING, and that cascade
                            // measured far worse than the crop win (CORD
                            // 21.70 -> 32.5 %). Tight boxes for structure,
                            // expanded boxes for reading.
                            (false, crate::content::ContentKind::Photographic) => line
                                .iter()
                                .map(|b| {
                                    let (bx0, by0) = (to_img(b.x0), to_img(b.y0));
                                    let (bx1, by1) = (to_img(b.x1), to_img(b.y1));
                                    let bh = (by1 - by0).max(1.0);
                                    (
                                        (bx0 - bh * 0.27).max(0.0) as usize,
                                        (by0 - bh * 0.24).max(0.0) as usize,
                                        ((bx1 + bh * 0.20) as usize).min(w),
                                        ((by1 + bh * 0.21) as usize).min(h),
                                    )
                                })
                                .collect(),
                        };
                        let lb_map = boxes::line_bbox(line);
                        let bh_img = (to_img(lb_map.y1) - to_img(lb_map.y0)).max(1.0);
                        // Word-crop pads, fractions of line height. Swept on
                        // CORD's TRAIN split and left at their synthetic-render
                        // values, which measured optimal on photographs too;
                        // FFAI_PARSEQ_PAD_X/_Y keep the sweep repeatable.
                        let (pad_x, pad_y) = crate::parseq_pads();
                        let ly0 = (to_img(lb_map.y0) - bh_img * pad_y).max(0.0) as usize;
                        let ly1 = ((to_img(lb_map.y1) + bh_img * pad_y) as usize).min(h);
                        let _ = (&region, &affinity, mw);
                        let _ = (ly0, ly1);
                        for &(wx0, wy0, wx1, wy1) in word_ranges.iter() {
                            let pad = (bh_img * pad_x) as usize;
                            let bx0 = wx0.saturating_sub(pad);
                            let by0 = wy0.saturating_sub((bh_img * pad_y * 0.0) as usize);
                            let bx1 = (wx1 + pad).min(w);
                            let by1 = wy1.min(h);
                            if bx1 <= bx0 + 1 || by1 <= by0 + 1 {
                                continue;
                            }
                            let (cw, ch) = (bx1 - bx0, by1 - by0);
                            let mut crop = vec![0f32; cw * ch];
                            for y in 0..ch {
                                crop[y * cw..(y + 1) * cw]
                                    .copy_from_slice(&gray[(by0 + y) * w + bx0..(by0 + y) * w + bx1]);
                            }
                            if let Ok(dir) = std::env::var("FFAI_DUMP_CROPS") {
                                dump_crop(&crop, cw, ch, &dir);
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
                // REJECTION. §8.22 decomposed the CORD gap and found 97 % of it
                // is INSERTIONS — text emitted for regions the ground truth does
                // not contain, because CORD's receipts are privacy-blurred and
                // we read the smear. Our substitutions are within 0.46 pp of
                // PaddleOCR's; what it does that we do not is refuse to answer.
                // A recognizer asked to read an illegible crop still returns its
                // best guess, and the confidence it returns alongside is the
                // signal that the guess is worthless.
                if let Some(c) = confidence {
                    if c < reject_threshold() {
                        return Ok(None);
                    }
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
        // FFAI_REC_SERIAL forces the serial path so the OUTER fan-out can be
        // isolated from candle's INTERNAL tile parallelism (§8.100 D5). Both
        // are rayon; measuring them apart is the only way to tell which one is
        // actually delivering the speedup.
        let serial = std::env::var("FFAI_REC_SERIAL").is_ok();
        let results: Vec<Option<OcrLine>> = if lines.len() >= 3 && !serial {
            // Three levels of rayon nest here — ours over lines, candle's over
            // im2col tiles, and `gemm`, which candle hands
            // `Parallelism::Rayon(num_cpus::get())` on EVERY matmul
            // (`cpu_backend/mod.rs:1394`). Flattening it to ONE level via a
            // dedicated pool + RAYON_NUM_THREADS=1 was built and measured:
            // z = +1.50 over 16 ABBA-paired rounds, INSIDE a 17-27 % noise
            // floor. Reverted per revert-if-unproven; §8.100 carries the
            // numbers so it is not rebuilt.
            lines.par_iter().map(&recognize_line).collect::<Result<Vec<_>>>()?
        } else {
            lines.iter().map(&recognize_line).collect::<Result<Vec<_>>>()?
        };
        let out_lines: Vec<OcrLine> = results.into_iter().flatten().collect();
        // §8.106: opt-in body-text scope. Off by default — a filter that
        // silently deletes output must be asked for, and a document-to-text
        // user usually wants the table cells this drops.
        // §8.157: the guard's geometry comes from the FULL population, before
        // suppression deletes any of it — gutters over 111 lines are not gutters
        // over 80, and that mistake voided a whole harness in §8.153.
        let probe_stats = crate::suppress::probe_stats(&out_lines, w as f32, h as f32);
        let out_lines = crate::suppress::body_only(out_lines, w as f32, h as f32);
        let out_lines =
            crate::suppress::probe_reorder(out_lines, &probe_stats, w as f32, h as f32);

        // v1: one block per page — paragraph segmentation is the DOCUMENT
        // milestone's work, and inventing it early would be unearned.
        Ok(OcrOutput { blocks: vec![OcrBlock { lines: out_lines, bbox: None }] })
    }
}

/// Six-whys instrument for the parseq word-crop defect: write the exact
/// crops the recognizer receives so the failure can be SEEN, not guessed.
fn dump_crop(gray: &[f32], w: usize, h: usize, dir: &str) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n >= 40 {
        return;
    }
    let _ = std::fs::create_dir_all(dir);
    let path = std::path::Path::new(dir).join(format!("crop-{n:02}.png"));
    if let Ok(f) = std::fs::File::create(&path) {
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        if let Ok(mut wr) = enc.write_header() {
            let buf: Vec<u8> = gray.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
            let _ = wr.write_image_data(&buf);
        }
    }
}
