//! `Yolo26` — the [`DetectEngine`] composing preprocess → backbone → neck →
//! head → decode.
//!
//! Weights load lazily on first use, from the model cache the audited
//! conversion writes to (mission plan §7). They are AGPL-3.0 and carry no
//! `hf_repo`, so there is nothing to download: a missing model is a message
//! naming the conversion command, not a network error.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rayon::prelude::*;
use ffai_core::engine::{DetectEngine, DetectOptions, EngineInfo, EngineStatus, Task};
use ffai_core::error::{Error, Result};
use ffai_core::types::{Detection, DetectOutput, ImageBuffer};

use crate::backbone::Backbone;
use crate::config::ModelConfig;
use crate::head::Head;
use crate::image::{letterbox_with, Geometry};
use crate::neck::Neck;

/// The tiers this build can load. Every one uses the SAME graph code — the
/// widths and repeat counts come from the checkpoint's scale via
/// [`crate::config::Dims`], so adding a tier is a table entry, not a port.
pub const TIERS: [&str; 5] = ["n", "s", "m", "l", "x"];

struct Model {
    backbone: Backbone,
    neck: Neck,
    head: Head,
    cfg: ModelConfig,
    device: Device,
}

pub struct Yolo26 {
    manifest_dir: PathBuf,
    geometry: Geometry,
    /// The size tier: "n", "s", "m", "l" or "x".
    tier: String,
    model: OnceLock<std::result::Result<Model, String>>,
}

impl Yolo26 {
    /// A specific tier and geometry — the general constructor the others
    /// delegate to.
    pub fn build(tier: &str, geometry: Geometry, manifest_dir: impl Into<PathBuf>) -> Self {
        Yolo26 {
            manifest_dir: manifest_dir.into(),
            geometry,
            tier: tier.to_string(),
            model: OnceLock::new(),
        }
    }

    pub fn tier(&self) -> &str {
        &self.tier
    }

    /// Build from weights and manifest already in memory — **no filesystem**.
    ///
    /// [`Self::build`] resolves a manifest directory and mmaps a file, which
    /// two real targets cannot do:
    ///
    /// * a **browser**, where `wasm32` has no filesystem at all — this is the
    ///   API that stands between Diana compiling for wasm (it does) and
    ///   running there (it cannot, yet);
    /// * an **embedded** device that ships weights in flash or a blob
    ///   appended to the binary, where there is a filesystem but nothing on
    ///   it worth reading.
    ///
    /// `safetensors` is the converted weight file's bytes and `manifest_json`
    /// the `-diana.json` beside it. Both are validated exactly as the
    /// filesystem path validates them — a manifest whose scale disagrees with
    /// `tier`, or that was produced by a different converter version, is
    /// refused here too. The bytes path is not the lenient path.
    pub fn from_bytes(
        tier: &str,
        geometry: Geometry,
        safetensors: Vec<u8>,
        manifest_json: &str,
    ) -> Result<Self> {
        let model = load_from_bytes(tier, safetensors, manifest_json)?;
        let cell = OnceLock::new();
        let _ = cell.set(Ok(model));
        Ok(Yolo26 {
            manifest_dir: PathBuf::new(),
            geometry,
            tier: tier.to_string(),
            model: cell,
        })
    }

    /// The default engine: **rectangular** letterboxing, matching
    /// Ultralytics' own default.
    ///
    /// Square padding is not merely slower — at this tier it is *dominated*.
    /// M-D0's reference board measured the rect variant at **70.14 mAP50 vs
    /// 68.65** and **18.26 img/s vs 15.30**, and our own forward is
    /// 1.35-1.48x faster on the corpus's aspect ratios. Square remains
    /// available as [`Self::square`] because the M-D0 parity gate is pinned
    /// to it; pinning the GATE and choosing the DEFAULT are two decisions.
    pub fn new() -> Self {
        Self::with_geometry(Geometry::Rect)
    }

    /// The square-padded engine — the configuration the parity gate pins.
    pub fn square() -> Self {
        Self::with_geometry(Geometry::Square)
    }

    pub fn with_geometry(geometry: Geometry) -> Self {
        Self::build("n", geometry, "models")
    }

    /// Point at a different manifest directory (tests, embedders).
    pub fn with_manifest_dir(dir: impl Into<PathBuf>) -> Self {
        Self::build("n", Geometry::Rect, dir)
    }

    /// Manifest directory AND geometry — what the bench harness needs to
    /// measure the pinned parity configuration.
    pub fn with_manifest_dir_and_geometry(dir: impl Into<PathBuf>, geometry: Geometry) -> Self {
        Self::build("n", geometry, dir)
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    fn model(&self) -> Result<&Model> {
        match self
            .model
            .get_or_init(|| load(&self.manifest_dir, &self.tier).map_err(|e| e.to_string()))
        {
            Ok(m) => Ok(m),
            Err(e) => Err(Error::Model(e.clone())),
        }
    }
}

impl Default for Yolo26 {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve a converted checkpoint: manifest plus weights path.
///
/// Shared by the detect and depth engines. Both need the same three things —
/// find the manifest, refuse to fetch AGPL weights, check the declared scale
/// against the tier asked for — and a second copy of that would be a second
/// place for the AGPL refusal to rot.
pub(crate) fn resolve_model(
    dir: &Path,
    model_name: &str,
    tier: &str,
) -> Result<(ModelConfig, PathBuf)> {
    let manifests = ffai_models::load_dir(dir)?;
    let manifest = manifests.iter().find(|m| m.name == model_name).ok_or_else(|| {
        Error::Model(format!("no model manifest named `{model_name}` in {}", dir.display()))
    })?;
    let resolved = manifest.fetch().map_err(|e| {
        Error::Model(format!(
            "{e}
  YOLO26 weights are AGPL-3.0 and are never fetched by FFai. Convert your              own checkpoint:
    .venv-diana/Scripts/python.exe tools/diana_convert.py              --model {model_name}"
        ))
    })?;
    let cfg = ModelConfig::load(resolved.file(&format!("{model_name}.json"))?)?;
    let weights = resolved.file(&format!("{model_name}.safetensors"))?.to_path_buf();
    // The manifest's own scale must match the tier we were asked for, or the
    // graph is built to different widths than the weights carry — which the
    // strict load would catch, but with a shape error rather than the reason.
    if cfg.scale != tier {
        return Err(Error::Model(format!(
            "manifest `{model_name}` declares scale `{}` but this engine is tier `{tier}` —              re-run tools/diana_convert.py",
            cfg.scale
        )));
    }
    Ok((cfg, weights))
}

/// The filesystem-free half of [`load`]. Both funnel into the same graph
/// construction so the two entry points cannot drift apart in what they
/// build — only in where the bytes came from.
fn load_from_bytes(tier: &str, safetensors: Vec<u8>, manifest_json: &str) -> Result<Model> {
    let device = Device::Cpu;
    let cfg: ModelConfig = serde_json::from_str(manifest_json)
        .map_err(|e| Error::Model(format!("depth/detect manifest json: {e}")))?;
    cfg.validate()?;
    if cfg.scale != tier {
        return Err(Error::Model(format!(
            "manifest declares scale `{}` but this engine is tier `{tier}`",
            cfg.scale
        )));
    }
    let vb = VarBuilder::from_buffered_safetensors(safetensors, DType::F32, &device)
        .map_err(candle_err)?;
    build_model(vb, cfg, device)
}

fn load(dir: &Path, tier: &str) -> Result<Model> {
    let device = Device::Cpu;
    let model_name = format!("yolo26{tier}-diana");
    let (cfg, weights) = resolve_model(dir, &model_name, tier)?;
    // SAFETY: the mapped file is owned by the model cache and is not
    // mutated while this process holds it.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)
    }
    .map_err(candle_err)?;

    build_model(vb, cfg, device)
}

fn build_model(vb: VarBuilder, cfg: ModelConfig, device: Device) -> Result<Model> {
    let dims = cfg.dims()?;
    let backbone = Backbone::new(vb.clone(), dims).map_err(candle_err)?;
    let neck = Neck::new(vb.clone(), dims).map_err(candle_err)?;
    let head =
        Head::new(vb, dims, cfg.nc, cfg.reg_max, cfg.strides.clone()).map_err(candle_err)?;
    Ok(Model { backbone, neck, head, cfg, device })
}

pub(crate) fn candle_err(e: candle_core::Error) -> Error {
    Error::Model(e.to_string())
}

impl DetectEngine for Yolo26 {
    fn info(&self) -> EngineInfo {
        // Distinct names because the two geometries are distinct measurement
        // configurations — the bench's matched-tier gate compares against
        // whichever reference declares the same geometry, and conflating
        // them is the M-D0 defect that started this whole thread.
        let (suffix, description) = match self.geometry {
            Geometry::Rect => (
                "",
                "on candle — NMS-free one2one head, rectangular letterbox (the \
                 reference's default), pure Rust",
            ),
            Geometry::Square => (
                "-square",
                "on candle, square 640x640 letterbox — the configuration the M-D0 \
                 parity gate pins",
            ),
        };
        EngineInfo {
            name: format!("yolo26{}{suffix}", self.tier),
            task: Task::Detect,
            // Per-layer oracle-gated against the official model; the corpus
            // gates (mission plan §8.1's baselines) are what would make it
            // `Stable`, and they have not been cleared yet.
            status: EngineStatus::Experimental,
            description: format!("YOLO26{} (Ultralytics lineage) {description}", self.tier),
        }
    }

    fn detect(&self, image: &ImageBuffer, opts: &DetectOptions) -> Result<DetectOutput> {
        // A batch has already spent the machine's cores across images, and
        // `serial_scope` has turned the per-layer fan-out off — installing a
        // second pool inside it would nest one fan-out in another, the exact
        // arrangement `crate::parallel` measured at a 2.32x CPU tax.
        if crate::parallel::serial_kernels() || crate::parallel::no_pool() {
            return self.detect_inner(image, opts);
        }
        crate::parallel::latency_pool().install(|| self.detect_inner(image, opts))
    }
    fn detect_batch(
        &self,
        images: &[ImageBuffer],
        opts: &DetectOptions,
    ) -> Result<Vec<DetectOutput>> {
        // Ensure the weights are loaded BEFORE the fan-out, so N threads do
        // not race into the same lazy init and serialize on it.
        self.model()?;
        images
            .par_iter()
            // Parallelism goes HERE and nowhere below it. Nesting the
            // per-layer fan-out inside a batch that already fills every core
            // buys nothing and costs a barrier per layer: measured 844 ms of
            // CPU per image at 24 threads against 363 ms serial, a 2.32x
            // tax on work that is identical either way. `crate::parallel`
            // carries the measurements and the sign-flip that makes this a
            // per-call decision rather than a constant.
            .map(|image| crate::parallel::serial_scope(|| self.detect(image, opts)))
            .collect()
    }

    fn class_names(&self) -> &[String] {
        match self.model() {
            Ok(m) => &m.cfg.class_names,
            Err(_) => &[],
        }
    }
}

impl Yolo26 {
    fn detect_inner(&self, image: &ImageBuffer, opts: &DetectOptions) -> Result<DetectOutput> {
        let m = self.model()?;
        crate::profile::timed(|p| &p.total, || {
            let (input, lb) = crate::profile::timed(|p| &p.pre, || {
                letterbox_with(image, m.cfg.imgsz, self.geometry, &m.device).map_err(candle_err)
            })?;
            let out = forward(m, &input, opts)?;
        let mut detections: Vec<Detection> = out
            .into_iter()
            .filter(|d| d.confidence >= opts.confidence)
            .filter(|d| opts.classes.is_empty() || opts.classes.contains(&d.class_id))
            .map(|d| {
                let (x0, y0) = lb.invert(d.x0, d.y0);
                let (x1, y1) = lb.invert(d.x1, d.y1);
                Detection {
                    x0,
                    y0,
                    x1,
                    y1,
                    class_id: d.class_id,
                    confidence: d.confidence,
                    track_id: None,
                }
            })
            .collect();
        detections.truncate(opts.max_detections);

            let mut output = DetectOutput { detections, letterbox: Some(lb) };
            // The one2one head is NMS-free by construction, so PLAIN
            // suppression stays opt-in — running it costs IDF1, because it
            // drops legitimately overlapping objects along with the duplicates.
            if let Some(iou) = opts.iou {
                output.suppress_overlaps(iou);
            } else {
                // But the construction does not hold in practice: measured
                // against the reference on identical MOT17 frames we emit 1285
                // intra-frame pairs above IoU 0.7 to their 855. The three-way
                // gate drops the duplicates while sparing genuine overlap —
                // IDF1 32.82 -> 33.28, MOTA 18.92 -> 19.24, ID switches
                // 375 -> 239. Thresholds and the one sequence that pays for it
                // are documented on `suppress_duplicates`.
                output.suppress_duplicates(0.80, 0.88, 0.95);
            }
            Ok(output)
        })
    }
}

fn forward(m: &Model, input: &Tensor, opts: &DetectOptions) -> Result<Vec<crate::head::DecodedBox>> {
    use crate::profile::timed;
    let b = timed(|p| &p.backbone, || m.backbone.forward(input).map_err(candle_err))?;
    let n = timed(|p| &p.neck, || m.neck.forward(&b).map_err(candle_err))?;
    let (per_level, boxes, scores, anchors) = timed(|p| &p.head, || {
        let per_level = m.head.forward(&n).map_err(candle_err)?;
        let (boxes, scores) = m.head.concat_levels(&per_level).map_err(candle_err)?;
        let anchors = m.head.anchors(&per_level).map_err(candle_err)?;
        Ok::<_, Error>((per_level, boxes, scores, anchors))
    })?;
    let _ = per_level;
    // Decode exactly what the caller asked for, capped by the checkpoint's
    // declared `max_det`.
    //
    // This was `.max()`, deliberately: decoding the model's full 300 and
    // trimming afterwards meant a small caller limit could not change WHICH
    // detections the two-stage selection produced. Stable, and WRONG against
    // the reference — because Ultralytics feeds `max_det` into BOTH stages:
    //
    //   idx = scores.amax(-1).topk(max_det)      <- stage 1 pool is max_det
    //   scores, index = scores.flatten(1).topk(max_det)
    //
    // So at the bench's `--max-dets 100` the reference selects from a
    // 100-anchor pool (8 000 candidates) while we selected from 300
    // (24 000) and discarded two thirds. That is a different ALGORITHM, not
    // just extra work, and "bit-for-bit parity with the reference" has to
    // mean the selection too.
    //
    // The speed saving is small and honest about it: decode is 2.1 % of the
    // pipeline and stage 1's per-anchor max over 8400x80 is k-INDEPENDENT,
    // so this is worth ~0.5 % — under this box's resolution on its own.
    // Taken anyway. Fifty such gains compound to 1.005^50 = 1.28x, and none
    // of them are individually measurable; refusing every gain that cannot
    // be seen in isolation is how a codebase stays 2x behind forever.
    let k = if crate::smallgains::disabled() {
        m.cfg.max_det.max(opts.max_detections) // the pre-parity behaviour
    } else {
        opts.max_detections.min(m.cfg.max_det).max(1)
    };
    timed(|p| &p.decode, || m.head.decode(&boxes, &scores, &anchors, k).map_err(candle_err))
}
