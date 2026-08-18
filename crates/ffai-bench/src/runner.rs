//! The one-call bench: our engine vs world standards on a pinned corpus.
//!
//! Design point ported from Prometheus trials: references and engine run over
//! the SAME holdout clips, timed the same way, scored by the same metric
//! code — and the run is recorded even when our engine can't compete yet.
//! Baselining the world standards *before* our engine exists is the point:
//! the targets are on the board from day one.
//!
//! Timing follows the two-number rule in [`crate::reference`]: warm
//! (steady-state, model loaded) and end-to-end (including one amortized model
//! load) are both measured and both recorded.

use std::path::{Path, PathBuf};

use ffai_core::engine::{AsrOptions, OcrOptions};
use ffai_core::error::{Error, Result};
use ffai_core::registry::EngineRegistry;

use crate::corpus::{ClipEntry, Manifest};
use crate::gate::{GateKind, GateOutcome, GateReport, GateResult};
use crate::ledger::{BenchRecord, Environment, LEDGER_SCHEMA, RunSummary, append};
use crate::metrics::{cer_with, wer_with};
use crate::normalize::Mode;
use crate::reference::ReferenceSpec;
use crate::speed::{best_of_n, real_time_factor};

/// Which metric the quality gate verdicts on. Both are always recorded; this
/// only chooses the headline. ASR gates on WER (the field's standard); OCR
/// gates on CER — character accuracy is what OCR benchmarks rank by, and word
/// boundaries in OCR output are partly a segmentation artifact. Detection
/// gates on `1 − mAP@0.5` — the gate machinery is written lower-is-better, so
/// the mAP is folded into an error-style "miss" metric rather than growing a
/// second comparison direction; both raw mAP fields ride in the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMetric {
    Wer,
    Cer,
    Map50,
}

/// Parity band for the quality gate: engine WER may exceed the best
/// reference's WER by at most this factor (1.05 = within 5% relative).
pub const QUALITY_PARITY_BAND: f64 = 1.05;

/// Absolute floor for the parity band, in error-rate units (0.25 pp).
///
/// A pure relative band degenerates when the best reference sits at ~0 %
/// error — 1.05 × 0.00 is 0.00, and the gate would demand perfection to the
/// character. Tesseract scores 0.00 % CER on the synthetic render corpus
/// (M-C0), which is exactly this case. The gate therefore passes on
/// `engine <= best * band` OR `engine <= best + floor`, and the detail line
/// says which bound applied. The floor is deliberately small: a quarter
/// point of CER is roughly one wrong character per 400 — parity for
/// practical purposes, not a loophole.
pub const QUALITY_ABS_FLOOR: f64 = 0.0025;

pub struct BenchConfig {
    /// Engine to bench; `None` = registry default. Set `skip_engine` to
    /// baseline references only.
    pub engine: Option<String>,
    pub skip_engine: bool,
    /// Skip every REFERENCE and bench only our engine. The mirror of
    /// [`Self::skip_engine`], and the one an A/B of our own flags needs.
    pub skip_references: bool,
    pub corpus: PathBuf,
    pub references: Vec<ReferenceSpec>,
    /// Timed repetitions of the whole corpus run (best-of-N).
    pub runs: usize,
    /// Ledger file to append to.
    pub ledger: PathBuf,
}

/// Run an ASR bench end-to-end and append the record to the ledger.
/// (TTS/OCR/VLM benches follow the same shape as their engines go live.)
pub fn run_asr(reg: &EngineRegistry, cfg: &BenchConfig) -> Result<BenchRecord> {
    let manifest = Manifest::load(&cfg.corpus)?;
    if manifest.task != "asr" {
        return Err(Error::Other(format!(
            "corpus `{}` is a {} corpus, not asr",
            manifest.name, manifest.task
        )));
    }
    let bad = manifest.verify()?;
    if !bad.is_empty() {
        return Err(Error::Other(format!(
            "corpus verification failed for clips {bad:?} — bytes on disk don't match the \
             manifest's pinned SHA-256; results on drifted data are not valid claims"
        )));
    }
    let holdout: Vec<&ClipEntry> = manifest.holdout().collect();
    if holdout.is_empty() {
        return Err(Error::Other(
            "corpus has no holdout clips — nothing to measure".into(),
        ));
    }
    let paths: Vec<PathBuf> = holdout.iter().map(|c| manifest.clip_path(c)).collect();
    let media_secs: f64 = paths
        .iter()
        .filter_map(|p| ffai_media::load_audio(p).ok())
        .map(|a| a.duration_secs())
        .sum();

    let mut references = Vec::new();
    for spec in cfg.references.iter().filter(|_| !cfg.skip_references) {
        eprintln!(
            "running reference `{}` over {} clips ...",
            spec.name,
            holdout.len()
        );
        match run_reference(
            spec,
            &manifest,
            &holdout,
            &paths,
            media_secs,
            cfg.runs,
            Mode::English,
            None,
        ) {
            Ok(summary) => references.push(summary),
            Err(e) => {
                eprintln!("  reference `{}` failed: {e}", spec.name);
                references.push(RunSummary {
                    name: spec.name.clone(),
                    version: spec.version(),
                    command: Some(spec.command_line()),
                    config: decode_config(spec.config.as_deref()),
                    wer: None,
                    cer: None,
                    rtf_warm: None,
                    rtf_e2e: None,
                    load_secs: None,
                    wall_secs: None,
                    media_secs: Some(media_secs),
                    clips_ok: 0,
                    clips_total: holdout.len(),
                    notes: vec![e.to_string()],
                    peak_bytes: None,
                    steady_bytes: None,
                    map50: None,
                    map5095: None,
                });
            }
        }
    }

    let engine_summary = if cfg.skip_engine {
        None
    } else {
        Some(run_engine(
            reg,
            cfg.engine.as_deref(),
            &manifest,
            &holdout,
            media_secs,
            cfg.runs,
        )?)
    };

    let mut gates = GateReport::new();
    fill_gates(
        &mut gates,
        engine_summary.as_ref(),
        &references,
        QualityMetric::Wer,
    );

    let (id, appended_at) = BenchRecord::now_id("asr");
    let record = BenchRecord {
        schema: LEDGER_SCHEMA,
        id,
        task: "asr".into(),
        corpus: manifest.name.clone(),
        corpus_manifest_hash: manifest.manifest_hash(),
        engine: engine_summary,
        references,
        gates,
        environment: Environment::capture(),
        notes: String::new(),
        appended_at,
    };
    append(&cfg.ledger, &record)?;
    Ok(record)
}

/// Run an OCR bench end-to-end and append the record to the ledger.
///
/// Differences from ASR, all deliberate:
/// - **The media unit is the page.** `media_secs` carries the page count, so
///   the RTF fields read as pages/second (warm and end-to-end). Same schema,
///   task-scoped meaning, labeled per task by [`render`].
/// - **Scoring uses [`Mode::Ocr`]** — whitespace-collapsed, case- and
///   punctuation-preserving. The ASR normalizer would score OCR's hardest
///   parts (case, digits, punctuation) as free.
/// - **The quality gate verdicts on CER** (see [`QualityMetric`]).
/// - **Per-page latency percentiles are recorded per reference** — the view a
///   LIVE streaming loop experiences, kept beside the corpus mean from day
///   one so the M-C2 latency gate has baselines waiting.
pub fn run_ocr(reg: &EngineRegistry, cfg: &BenchConfig) -> Result<BenchRecord> {
    let manifest = Manifest::load(&cfg.corpus)?;
    if manifest.task != "ocr" {
        return Err(Error::Other(format!(
            "corpus `{}` is a {} corpus, not ocr",
            manifest.name, manifest.task
        )));
    }
    let bad = manifest.verify()?;
    if !bad.is_empty() {
        return Err(Error::Other(format!(
            "corpus verification failed for clips {bad:?} — bytes on disk don't match the \
             manifest's pinned SHA-256; results on drifted data are not valid claims"
        )));
    }
    let holdout: Vec<&ClipEntry> = manifest.holdout().collect();
    if holdout.is_empty() {
        return Err(Error::Other(
            "corpus has no holdout clips — nothing to measure".into(),
        ));
    }
    let paths: Vec<PathBuf> = holdout.iter().map(|c| manifest.clip_path(c)).collect();
    let pages = holdout.len() as f64;

    let mut references = Vec::new();
    for spec in cfg.references.iter().filter(|_| !cfg.skip_references) {
        eprintln!(
            "running reference `{}` over {} pages ...",
            spec.name,
            holdout.len()
        );
        match run_reference(
            spec,
            &manifest,
            &holdout,
            &paths,
            pages,
            cfg.runs,
            Mode::Ocr,
            Some("page"),
        ) {
            Ok(summary) => references.push(summary),
            Err(e) => {
                eprintln!("  reference `{}` failed: {e}", spec.name);
                references.push(RunSummary {
                    name: spec.name.clone(),
                    version: spec.version(),
                    command: Some(spec.command_line()),
                    config: decode_config(spec.config.as_deref()),
                    wer: None,
                    cer: None,
                    rtf_warm: None,
                    rtf_e2e: None,
                    load_secs: None,
                    wall_secs: None,
                    media_secs: Some(pages),
                    clips_ok: 0,
                    clips_total: holdout.len(),
                    notes: vec![e.to_string()],
                    peak_bytes: None,
                    steady_bytes: None,
                    map50: None,
                    map5095: None,
                });
            }
        }
    }

    let engine_summary = if cfg.skip_engine {
        None
    } else {
        Some(run_ocr_engine(
            reg,
            cfg.engine.as_deref(),
            &manifest,
            &holdout,
            pages,
            cfg.runs,
        )?)
    };

    let mut gates = GateReport::new();
    fill_gates(
        &mut gates,
        engine_summary.as_ref(),
        &references,
        QualityMetric::Cer,
    );

    let (id, appended_at) = BenchRecord::now_id("ocr");
    let record = BenchRecord {
        schema: LEDGER_SCHEMA,
        id,
        task: "ocr".into(),
        corpus: manifest.name.clone(),
        corpus_manifest_hash: manifest.manifest_hash(),
        engine: engine_summary,
        references,
        gates,
        environment: Environment::capture(),
        notes: String::new(),
        appended_at,
    };
    append(&cfg.ledger, &record)?;
    Ok(record)
}

/// Run a detection bench end-to-end and append the record to the ledger.
///
/// Differences from OCR, all deliberate:
/// - **The media unit is the image.** `media_secs` carries the image count,
///   so the RTF fields read as images/second. Same schema, task-scoped
///   meaning, labeled per task by [`render`].
/// - **Scoring is the mAP proxy in [`crate::detect`]** — ground truth is a
///   JSON boxes file per image, the hypothesis is the adapter's `text` field
///   carrying a JSON detections payload, and the whole batch/timing/memory
///   contract of [`run_reference`] is reused unchanged around it.
/// - **The quality gate verdicts on `1 − mAP@0.5`** (see [`QualityMetric`]);
///   `mAP@0.5:0.95` is recorded beside it so neither is quoted alone.
/// - **There is no in-process engine path yet.** Diana's first engine lands
///   at M-D1; until then `--baseline-only` is the only honest invocation and
///   an engine request fails loudly rather than pretending.
pub fn run_detect(reg: &EngineRegistry, cfg: &BenchConfig) -> Result<BenchRecord> {
    let manifest = Manifest::load(&cfg.corpus)?;
    if manifest.task != "detect" {
        return Err(Error::Other(format!(
            "corpus `{}` is a {} corpus, not detect",
            manifest.name, manifest.task
        )));
    }
    let bad = manifest.verify()?;
    if !bad.is_empty() {
        return Err(Error::Other(format!(
            "corpus verification failed for clips {bad:?} — bytes on disk don't match the \
             manifest's pinned SHA-256; results on drifted data are not valid claims"
        )));
    }
    let holdout: Vec<&ClipEntry> = manifest.holdout().collect();
    if holdout.is_empty() {
        return Err(Error::Other(
            "corpus has no holdout clips — nothing to measure".into(),
        ));
    }
    let paths: Vec<PathBuf> = holdout.iter().map(|c| manifest.clip_path(c)).collect();
    let images = holdout.len() as f64;

    let mut references = Vec::new();
    for spec in cfg.references.iter().filter(|_| !cfg.skip_references) {
        eprintln!(
            "running reference `{}` over {} images ...",
            spec.name,
            holdout.len()
        );
        match run_detect_reference(spec, &manifest, &holdout, &paths, images, cfg.runs) {
            Ok(summary) => references.push(summary),
            Err(e) => {
                eprintln!("  reference `{}` failed: {e}", spec.name);
                references.push(RunSummary {
                    name: spec.name.clone(),
                    version: spec.version(),
                    command: Some(spec.command_line()),
                    config: decode_config(spec.config.as_deref()),
                    wer: None,
                    cer: None,
                    map50: None,
                    map5095: None,
                    rtf_warm: None,
                    rtf_e2e: None,
                    load_secs: None,
                    wall_secs: None,
                    media_secs: Some(images),
                    clips_ok: 0,
                    clips_total: holdout.len(),
                    notes: vec![e.to_string()],
                    peak_bytes: None,
                    steady_bytes: None,
                });
            }
        }
    }

    let engine_summary = if cfg.skip_engine {
        None
    } else {
        Some(run_detect_engine(
            reg,
            cfg.engine.as_deref(),
            &manifest,
            &holdout,
            images,
            cfg.runs,
        )?)
    };

    let mut gates = GateReport::new();
    fill_gates(
        &mut gates,
        engine_summary.as_ref(),
        &references,
        QualityMetric::Map50,
    );

    let (id, appended_at) = BenchRecord::now_id("detect");
    let record = BenchRecord {
        schema: LEDGER_SCHEMA,
        id,
        task: "detect".into(),
        corpus: manifest.name.clone(),
        corpus_manifest_hash: manifest.manifest_hash(),
        engine: engine_summary,
        references,
        gates,
        environment: Environment::capture(),
        notes: String::new(),
        appended_at,
    };
    append(&cfg.ledger, &record)?;
    Ok(record)
}

/// [`run_reference`] with detection scoring in place of text scoring: same
/// batch contract, same fastest-run-keeps-its-own-timings rule, same memory
/// sampling — the adapter's `text` field carries a JSON detections payload
/// scored by [`crate::detect`] instead of by edit distance.
fn run_detect_reference(
    spec: &ReferenceSpec,
    manifest: &Manifest,
    holdout: &[&ClipEntry],
    paths: &[PathBuf],
    images: f64,
    runs: usize,
) -> Result<RunSummary> {
    let mut summary = RunSummary {
        name: spec.name.clone(),
        version: spec.version(),
        command: Some(spec.command_line()),
        config: decode_config(spec.config.as_deref()),
        wer: None,
        cer: None,
        map50: None,
        map5095: None,
        rtf_warm: None,
        rtf_e2e: None,
        load_secs: None,
        wall_secs: None,
        media_secs: Some(images),
        clips_ok: 0,
        clips_total: holdout.len(),
        notes: Vec::new(),
        peak_bytes: None,
        steady_bytes: None,
    };

    if !spec.supports_batch() {
        return Err(Error::Other(format!(
            "reference `{}` declares no batch_command — per-clip invocation would put \
             interpreter startup and model load inside every timed run, which is not a \
             comparable measurement (see crates/ffai-bench/src/reference.rs)",
            spec.name
        )));
    }

    let mut best: Option<(f64, crate::reference::BatchResult)> = None;
    for _ in 0..runs.max(1) {
        let started = std::time::Instant::now();
        let batch = spec.run_batch(paths)?;
        let wall = started.elapsed().as_secs_f64();
        if best.as_ref().is_none_or(|(prev, _)| wall < *prev) {
            best = Some((wall, batch));
        }
    }
    let (wall_secs, batch) = best.expect("at least one run");

    let mut acc = crate::detect::MapAccumulator::new();
    for (clip, path) in holdout.iter().zip(paths) {
        let Some(truth_text) = manifest.ground_truth(clip)? else {
            return Err(Error::Other(format!(
                "clip `{}` has no ground_truth file — a detect corpus without boxes cannot \
                 be scored",
                clip.id
            )));
        };
        let truths = crate::detect::parse_ground_truth(&truth_text)
            .map_err(|e| Error::Other(format!("ground truth for `{}`: {e}", clip.id)))?;
        match batch.text_for(path) {
            Some(payload) => match crate::detect::parse_detections(payload) {
                Ok(dets) => {
                    summary.clips_ok += 1;
                    acc.add_image(&truths, &dets);
                }
                Err(e) => summary.notes.push(format!("{}: bad payload: {e}", clip.id)),
            },
            None => summary
                .notes
                .push(format!("{}: no result returned", clip.id)),
        }
    }
    summary.map50 = acc.map50();
    summary.map5095 = acc.map5095();

    let mut times: Vec<f64> = batch
        .clips
        .iter()
        .filter_map(|c| c.transcribe_secs)
        .collect();
    if !times.is_empty() {
        times.sort_by(|a, b| a.total_cmp(b));
        let pct = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
        summary.notes.push(format!(
            "per-image latency p50 {:.0} ms / p95 {:.0} ms over {} images (adapter-timed, warm)",
            pct(0.50) * 1000.0,
            pct(0.95) * 1000.0,
            times.len()
        ));
    }
    summary.load_secs = batch.load_secs;
    summary.wall_secs = Some(wall_secs);
    summary.rtf_e2e = Some(real_time_factor(images, wall_secs));
    summary.rtf_warm = batch
        .transcribe_secs()
        .map(|t| real_time_factor(images, t))
        .or(summary.rtf_e2e);
    summary.peak_bytes = batch.peak_bytes;
    summary.steady_bytes = batch.steady_bytes;
    Ok(summary)
}

/// The in-process detection engine's run.
///
/// `load_image`, but a panicking decoder costs ONE clip instead of the corpus.
///
/// This is not defensive programming for its own sake. `rusty_jpeg` 0.1.5
/// unwraps a `None` on progressive JPEGs, and 49 of `OmniDocBench`'s 316 pages
/// are progressive — so one upstream defect turns a whole benchmark run into
/// no data at all, and the process dies before the harness can say which clip
/// did it. Catching it converts "the run produced nothing" into "267 pages
/// scored, 49 clips failed, here is the first one", which is the difference
/// between a mystery and a bug report.
///
/// Deliberately NOT applied around the engine itself: a panic in Carmenta is
/// our defect and must abort loudly. This only shields the harness from its
/// decode dependencies.
fn load_image_resilient(path: &Path) -> std::result::Result<ffai_core::types::ImageBuffer, String> {
    let taken = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ffai_media::load_image(path)
    }));
    match taken {
        Ok(Ok(image)) => Ok(image),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("decoder PANICKED — upstream defect, clip skipped".to_string()),
    }
}

/// Same contract as [`run_ocr_engine`]: images decoded once outside the
/// timed region, one untimed warm-up whose cost becomes `load_secs`, memory
/// sampled on the same 20 ms cadence as the reference tree, best-of-N over
/// the whole corpus.
///
/// **The confidence threshold is pinned to the references', not the
/// engine's default.** `DetectOptions::default()` uses 0.25 — the value a
/// person looking at boxes wants — but mAP needs the low-confidence tail,
/// and the reference adapters run at `--conf 0.001 --max-dets 100`. Scoring
/// our engine at 0.25 against references at 0.001 would report a recall
/// collapse that is purely a configuration difference, which is the M0
/// unpinned-decode defect in its detection costume.
fn run_detect_engine(
    reg: &EngineRegistry,
    engine: Option<&str>,
    manifest: &Manifest,
    holdout: &[&ClipEntry],
    images: f64,
    runs: usize,
) -> Result<RunSummary> {
    use ffai_core::engine::DetectOptions;

    let det = reg.detect(engine)?;
    let opts = DetectOptions {
        confidence: 0.001,
        max_detections: crate::detect::MAX_DETS,
        iou: None,
        classes: Vec::new(),
    };
    let mut summary = RunSummary {
        name: det.info().name,
        version: Some(env!("CARGO_PKG_VERSION").to_string()),
        command: None,
        config: {
            let mut c = std::collections::BTreeMap::new();
            // BOTH the tier and the geometry are part of the comparison key,
            // so both are DERIVED from the engine's own name rather than
            // hardcoded. `yolo26s` must be judged against the s references,
            // not the n ones, and a rectangular engine against rectangular
            // references — hardcoding either while the engine ran something
            // else is precisely the M-D0 defect: two rows that look matched
            // and are not.
            let name = det.info().name;
            let geometry = if name.ends_with("-square") {
                "sq"
            } else {
                "rect"
            };
            let tier = name.trim_end_matches("-square").to_string();
            c.insert(DECODE_KEY.to_string(), format!("{tier}/e2e-640{geometry}"));
            c
        },
        wer: None,
        cer: None,
        map50: None,
        map5095: None,
        rtf_warm: None,
        rtf_e2e: None,
        load_secs: None,
        wall_secs: None,
        media_secs: Some(images),
        clips_ok: 0,
        clips_total: holdout.len(),
        notes: Vec::new(),
        peak_bytes: None,
        steady_bytes: None,
    };

    let mut decoded = Vec::new();
    for clip in holdout {
        match load_image_resilient(&manifest.clip_path(clip)) {
            Ok(image) => decoded.push((*clip, image)),
            Err(e) => summary.notes.push(format!("{}: {e}", clip.id)),
        }
    }
    if decoded.is_empty() {
        summary
            .notes
            .push("no holdout image could be decoded — engine run not attempted".to_string());
        return Ok(summary);
    }

    if let Some((_, image)) = decoded.first() {
        let t0 = std::time::Instant::now();
        if let Err(e) = det.detect(image, &opts) {
            summary.notes.push(format!("warm-up failed: {e}"));
        }
        summary.load_secs = Some(t0.elapsed().as_secs_f64());
    }

    let sampling = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let our_samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    let sampler = {
        let (sampling, out) = (sampling.clone(), our_samples.clone());
        std::thread::spawn(move || {
            while sampling.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(b) = crate::footprint::current_self()
                    && let Ok(mut v) = out.lock()
                {
                    v.push(b.0);
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        })
    };

    let mut results: Vec<(String, Vec<crate::detect::Detection>)> = Vec::new();
    let mut per_image_secs: Vec<f64> = Vec::new();
    let mut first_error = None;
    let stats = best_of_n(runs, || {
        results.clear();
        per_image_secs.clear();
        for (clip, image) in &decoded {
            // Decode each image inside the timed loop, as the reference does.
            //
            // `FFAI_BENCH_PREDECODE=1` restores the old behaviour for A/B.
            //
            // The pre-decode looks like it FAVOURS us — it keeps our PNG
            // reader out of the timed region while `predict(path)` pays for
            // its own decode. But it also holds the WHOLE corpus resident:
            // 45 frames at 640x640x3 is 53 MiB against this box's 32 MiB L3,
            // so the model's weights and activations are evicted between
            // every image. The reference's working set is one frame.
            //
            // MEASURED, ABBA, three reps: just-in-time is 11-17 % FASTER than
            // pre-decoding, despite ADDING the decode to the timed region.
            //
            //   rep 1   pre 87.0 ms   jit 72.5 ms
            //   rep 2   pre 102.0     jit 88.0
            //   rep 3   pre 118.5     jit 105.5
            //
            // The mechanism is recency, not residency — and the distinction
            // matters because a working-set sweep (examples/workingset.rs)
            // showed holding 1 image versus 45 is FLAT. Total resident bytes
            // do not matter; whether the ONE buffer being read was written
            // recently does. Under JIT the decoder has just written it, so
            // `detect` reads it out of L1/L2; under pre-decode it was written
            // 45 images ago and must be fetched.
            //
            // Pre-decoding was chosen to keep our PNG reader out of the timed
            // region while `predict(path)` pays for its own decode — which
            // reads as generous to the reference. It was the opposite: the
            // reference decoded each image just before using it and always
            // got a hot buffer, while we always got a cold one. The harness
            // was handicapping us by ~14 %.
            let jit;
            let image = if !std::env::var("FFAI_BENCH_PREDECODE").is_ok_and(|v| v == "1") {
                match load_image_resilient(&manifest.clip_path(clip)) {
                    Ok(i) => {
                        jit = i;
                        &jit
                    }
                    Err(_) => image,
                }
            } else {
                image
            };
            let t = std::time::Instant::now();
            match det.detect(image, &opts) {
                Ok(out) => {
                    per_image_secs.push(t.elapsed().as_secs_f64());
                    results.push((
                        clip.id.clone(),
                        out.detections
                            .iter()
                            .map(|d| crate::detect::Detection {
                                bbox: [
                                    f64::from(d.x0),
                                    f64::from(d.y0),
                                    f64::from(d.x1),
                                    f64::from(d.y1),
                                ],
                                class: i64::from(d.class_id),
                                confidence: f64::from(d.confidence),
                            })
                            .collect(),
                    ));
                }
                Err(e) => {
                    // Record and CONTINUE. Aborting here threw away every other
                    // clip: a 43-page run reported 0/43 because page 23 failed,
                    // which made a real crash — PARSeq overflowing on a long
                    // word — look like a harness quirk and hid it for a whole
                    // benchmark cycle. The correctness gate already expresses
                    // partial success as `clips_ok < clips_total`, so aborting
                    // discards exactly the information the gate exists to
                    // report, and costs 42 good measurements to learn one bad
                    // one.
                    first_error.get_or_insert(format!("{}: {e}", clip.id));
                }
            }
        }
        Ok(())
    });

    sampling.store(false, std::sync::atomic::Ordering::Relaxed);
    sampler.join().ok();
    summary.steady_bytes = our_samples.lock().ok().and_then(|mut v| {
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    });
    summary.peak_bytes = crate::footprint::peak_self().map(|p| p.0);
    let decoded_bytes: u64 = decoded.iter().map(|(_, i)| i.data.len() as u64).sum();
    // Charge the harness's own image cache to the harness, not to the engine.
    //
    // We pre-decode every clip up front so the SPEED measurement excludes
    // image decoding; the reference reads each file inside its own timed
    // region. That is a deliberate choice and it is defensible — but the
    // buffers then sit in OUR process for the whole run, and the reference,
    // running as a subprocess, never holds them. Left uncorrected, the
    // harness's decision about one gate silently decides another.
    //
    // It was invisible while the corpus was small: 45 images is 37 MiB of
    // cache against ~100 MiB of engine. At 450 images it is 355 MiB against
    // ~58 MiB, and the footprint gate INVERTED — Diana read 413 MiB against
    // the reference's 396 and would have been recorded as a regression while
    // actually using a seventh of the memory. A gate whose verdict depends
    // on the corpus SIZE is measuring the harness.
    //
    // Saturating, and the raw figure stays in the note, so the correction is
    // auditable rather than a number that appeared.
    if decoded_bytes > 0 {
        let raw_peak = summary.peak_bytes;
        summary.peak_bytes = summary.peak_bytes.map(|p| p.saturating_sub(decoded_bytes));
        summary.steady_bytes = summary
            .steady_bytes
            .map(|s| s.saturating_sub(decoded_bytes));
        if let Some(raw) = raw_peak {
            summary.notes.push(format!(
                "footprint EXCLUDES {:.1} MiB of pre-decoded images held by the harness, \
                 which the reference does not hold (raw peak {:.1} MiB)",
                decoded_bytes as f64 / (1024.0 * 1024.0),
                raw as f64 / (1024.0 * 1024.0)
            ));
        }
    }

    match stats {
        Ok(stats) => {
            summary.clips_ok = results.len();
            summary.wall_secs = Some(stats.best_secs);
            summary.rtf_warm = Some(real_time_factor(images, stats.best_secs));
            summary.rtf_e2e = Some(real_time_factor(
                images,
                stats.best_secs + summary.load_secs.unwrap_or(0.0),
            ));
            let mut acc = crate::detect::MapAccumulator::new();
            for (id, dets) in &results {
                if let Some(clip) = holdout.iter().find(|c| &c.id == id)
                    && let Some(truth) = manifest.ground_truth(clip)?
                {
                    acc.add_image(&crate::detect::parse_ground_truth(&truth)?, dets);
                }
            }
            summary.map50 = acc.map50();
            summary.map5095 = acc.map5095();
            if !per_image_secs.is_empty() {
                let mut t = per_image_secs.clone();
                t.sort_by(|a, b| a.total_cmp(b));
                let pct = |p: f64| t[((t.len() - 1) as f64 * p).round() as usize];
                summary.notes.push(format!(
                    "per-image latency p50 {:.0} ms / p95 {:.0} ms over {} images (warm)",
                    pct(0.50) * 1000.0,
                    pct(0.95) * 1000.0,
                    t.len()
                ));
            }
        }
        // FRONT, not back. The correctness gate reports `notes.first()` as the
        // "first failure", and by this point `notes` already holds
        // informational entries — the footprint-exclusion note among them. A
        // failing run therefore reported a memory-accounting sentence as its
        // cause, which read as a harness quirk rather than a real error and
        // left two engines wrongly described as "unmeasured rather than
        // broken". The error goes where the gate actually looks.
        Err(e) => summary
            .notes
            .insert(0, first_error.unwrap_or_else(|| e.to_string())),
    }
    Ok(summary)
}

/// The in-process OCR engine's run. Compact by design: the memory sampler and
/// pre-decode asymmetry accounting from [`run_engine`] land at M-C1 when a
/// real engine exists to measure — at M-C0 every registered OCR engine is an
/// honest stub, and this exists so `ffai bench ocr` (without
/// `--baseline-only`) records that failure rather than hiding it.
fn run_ocr_engine(
    reg: &EngineRegistry,
    engine: Option<&str>,
    manifest: &Manifest,
    holdout: &[&ClipEntry],
    pages: f64,
    runs: usize,
) -> Result<RunSummary> {
    let ocr = reg.ocr(engine)?;
    let opts = OcrOptions::default();
    let mut summary = RunSummary {
        name: ocr.info().name,
        version: Some(env!("CARGO_PKG_VERSION").to_string()),
        command: None,
        config: {
            let mut c = std::collections::BTreeMap::new();
            // Carmenta engines share the OCR references' comparison key: all
            // are English printed-text stacks at default configuration. This
            // is a declaration (the same contract references make in
            // references.toml), not an inference from the name.
            c.insert(DECODE_KEY.to_string(), "en/printed".to_string());
            c
        },
        wer: None,
        cer: None,
        rtf_warm: None,
        rtf_e2e: None,
        load_secs: None,
        wall_secs: None,
        media_secs: Some(pages),
        clips_ok: 0,
        clips_total: holdout.len(),
        notes: Vec::new(),
        peak_bytes: None,
        steady_bytes: None,
        map50: None,
        map5095: None,
    };

    // Decode images ONCE, outside the timed region — same rationale as ASR:
    // the comparison measures inference, not our PNG reader.
    let mut decoded = Vec::new();
    for clip in holdout {
        match load_image_resilient(&manifest.clip_path(clip)) {
            Ok(image) => decoded.push((*clip, image)),
            Err(e) => summary.notes.push(format!("{}: {e}", clip.id)),
        }
    }
    if decoded.is_empty() {
        summary
            .notes
            .push("no holdout image could be decoded — engine run not attempted".to_string());
        return Ok(summary);
    }

    // Warm-up pass, untimed, load recorded — identical contract to ASR.
    if let Some((_, image)) = decoded.first() {
        let t0 = std::time::Instant::now();
        if let Err(e) = ocr.recognize(image, &opts) {
            summary.notes.push(format!("warm-up failed: {e}"));
        }
        summary.load_secs = Some(t0.elapsed().as_secs_f64());
    }

    // Sample OUR resident memory the same way the reference tree is sampled
    // (same discipline as `run_engine`): steady = median, peak beside it.
    let sampling = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let our_samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    let sampler = {
        let (sampling, out) = (sampling.clone(), our_samples.clone());
        std::thread::spawn(move || {
            while sampling.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(b) = crate::footprint::current_self()
                    && let Ok(mut v) = out.lock()
                {
                    v.push(b.0);
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        })
    };

    let mut texts: Vec<(String, String)> = Vec::new();
    let mut first_error = None;
    let stats = best_of_n(runs, || {
        texts.clear();
        for (clip, image) in &decoded {
            match ocr.recognize(image, &opts) {
                Ok(out) => texts.push((clip.id.clone(), out.text())),
                Err(e) => {
                    first_error.get_or_insert(format!("{}: {e}", clip.id));
                    return Err(e);
                }
            }
        }
        Ok(())
    });

    sampling.store(false, std::sync::atomic::Ordering::Relaxed);
    sampler.join().ok();
    summary.steady_bytes = our_samples.lock().ok().and_then(|mut v| {
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    });
    summary.peak_bytes = crate::footprint::peak_self().map(|p| p.0);
    // The pre-decoded image buffers are the harness's choice, not the
    // engine's, so they are SUBTRACTED rather than merely noted — see the
    // long comment on the same correction in the single-image path above for
    // what it cost to find that a note is not enough.
    let decoded_bytes: u64 = decoded.iter().map(|(_, i)| i.data.len() as u64).sum();
    if decoded_bytes > 0 {
        let raw_peak = summary.peak_bytes;
        summary.peak_bytes = summary.peak_bytes.map(|p| p.saturating_sub(decoded_bytes));
        summary.steady_bytes = summary
            .steady_bytes
            .map(|s| s.saturating_sub(decoded_bytes));
        if let Some(raw) = raw_peak {
            summary.notes.push(format!(
                "footprint EXCLUDES {:.1} MiB of pre-decoded images held by the harness, \
                 which the reference does not hold (raw peak {:.1} MiB)",
                decoded_bytes as f64 / (1024.0 * 1024.0),
                raw as f64 / (1024.0 * 1024.0)
            ));
        }
    }

    match stats {
        Ok(stats) => {
            summary.clips_ok = texts.len();
            summary.wall_secs = Some(stats.best_secs);
            summary.rtf_warm = Some(real_time_factor(pages, stats.best_secs));
            summary.rtf_e2e = Some(real_time_factor(
                pages,
                stats.best_secs + summary.load_secs.unwrap_or(0.0),
            ));
            let (mut wers, mut cers) = (Vec::new(), Vec::new());
            for (id, text) in &texts {
                if let Some(clip) = holdout.iter().find(|c| &c.id == id)
                    && let Some(truth) = manifest.ground_truth(clip)?
                {
                    wers.push(wer_with(&truth, text, Mode::Ocr));
                    cers.push(cer_with(&truth, text, Mode::Ocr));
                }
            }
            summary.wer = mean(&wers);
            summary.cer = mean(&cers);
        }
        // FRONT, not back. The correctness gate reports `notes.first()` as the
        // "first failure", and by this point `notes` already holds
        // informational entries — the footprint-exclusion note among them. A
        // failing run therefore reported a memory-accounting sentence as its
        // cause, which read as a harness quirk rather than a real error and
        // left two engines wrongly described as "unmeasured rather than
        // broken". The error goes where the gate actually looks.
        Err(e) => summary
            .notes
            .insert(0, first_error.clone().unwrap_or_else(|| e.to_string())),
    }
    // A run that completed with failures still has to say so: the loop no
    // longer aborts, so the error would otherwise never reach `notes`.
    if summary.clips_ok < summary.clips_total
        && let Some(e) = first_error
        && summary.notes.first() != Some(&e)
    {
        summary.notes.insert(0, e);
    }
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn run_reference(
    spec: &ReferenceSpec,
    manifest: &Manifest,
    holdout: &[&ClipEntry],
    paths: &[PathBuf],
    media_secs: f64,
    runs: usize,
    mode: Mode,
    // When set (e.g. `Some("page")`), per-item latency percentiles from the
    // adapter's own timings are recorded as a note — the LIVE-relevant view
    // of the same run, since a streaming loop experiences the per-frame
    // distribution, not the corpus mean.
    per_item_unit: Option<&str>,
) -> Result<RunSummary> {
    let mut summary = RunSummary {
        name: spec.name.clone(),
        version: spec.version(),
        command: Some(spec.command_line()),
        config: decode_config(spec.config.as_deref()),
        wer: None,
        cer: None,
        rtf_warm: None,
        rtf_e2e: None,
        load_secs: None,
        wall_secs: None,
        media_secs: Some(media_secs),
        clips_ok: 0,
        clips_total: holdout.len(),
        notes: Vec::new(),
        peak_bytes: None,
        steady_bytes: None,
        map50: None,
        map5095: None,
    };

    if !spec.supports_batch() {
        return Err(Error::Other(format!(
            "reference `{}` declares no batch_command — per-clip invocation would put \
             interpreter startup and model load inside every timed run, which is not a \
             comparable measurement (see crates/ffai-bench/src/reference.rs)",
            spec.name
        )));
    }

    // One invocation over the whole corpus, repeated; keep the FASTEST run's
    // wall clock together with THAT RUN'S per-clip timings.
    //
    // Taking the best wall time from one run and the adapter timings from
    // another produces impossible pairs — warm throughput reading slower than
    // end-to-end, which cannot happen for a single run. Timing numbers only
    // compose if they come from the same execution.
    let mut best: Option<(f64, crate::reference::BatchResult)> = None;
    for _ in 0..runs.max(1) {
        let started = std::time::Instant::now();
        let batch = spec.run_batch(paths)?;
        let wall = started.elapsed().as_secs_f64();
        if best.as_ref().is_none_or(|(prev, _)| wall < *prev) {
            best = Some((wall, batch));
        }
    }
    let (wall_secs, batch) = best.expect("at least one run");
    let stats = crate::speed::SpeedStats {
        best_secs: wall_secs,
        median_secs: wall_secs,
        runs: runs.max(1),
    };

    let (mut wers, mut cers) = (Vec::new(), Vec::new());
    for (clip, path) in holdout.iter().zip(paths) {
        match batch.text_for(path) {
            Some(hypothesis) => {
                summary.clips_ok += 1;
                if let Some(truth) = manifest.ground_truth(clip)? {
                    wers.push(wer_with(&truth, hypothesis, mode));
                    cers.push(cer_with(&truth, hypothesis, mode));
                }
            }
            None => summary
                .notes
                .push(format!("{}: no result returned", clip.id)),
        }
    }

    summary.wer = mean(&wers);
    summary.cer = mean(&cers);

    if let Some(unit) = per_item_unit {
        let mut times: Vec<f64> = batch
            .clips
            .iter()
            .filter_map(|c| c.transcribe_secs)
            .collect();
        if !times.is_empty() {
            times.sort_by(|a, b| a.total_cmp(b));
            let pct = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
            summary.notes.push(format!(
                "per-{unit} latency p50 {:.0} ms / p95 {:.0} ms over {} {unit}s (adapter-timed, warm)",
                pct(0.50) * 1000.0,
                pct(0.95) * 1000.0,
                times.len()
            ));
        }
    }
    summary.load_secs = batch.load_secs;
    summary.wall_secs = Some(stats.best_secs);
    summary.rtf_e2e = Some(real_time_factor(media_secs, stats.best_secs));
    summary.rtf_warm = batch
        .transcribe_secs()
        .map(|t| real_time_factor(media_secs, t))
        .or(summary.rtf_e2e);
    // From the same execution the timings came from — `best` keeps one run's
    // wall clock and that run's batch together, and the peak rides along.
    summary.peak_bytes = batch.peak_bytes;
    summary.steady_bytes = batch.steady_bytes;
    Ok(summary)
}

fn run_engine(
    reg: &EngineRegistry,
    engine: Option<&str>,
    manifest: &Manifest,
    holdout: &[&ClipEntry],
    media_secs: f64,
    runs: usize,
) -> Result<RunSummary> {
    let asr = reg.asr(engine)?;
    // Named rather than constructed inline at each call site, so what the
    // ledger records is provably the same options object the engine ran with.
    let opts = AsrOptions::default();
    let mut summary = RunSummary {
        name: asr.info().name,
        version: Some(env!("CARGO_PKG_VERSION").to_string()),
        command: None,
        config: {
            let mut c = engine_config(&opts);
            if let Some(key) = engine_comparison_key(&asr.info().name) {
                c.insert(DECODE_KEY.to_string(), key);
            }
            c
        },
        wer: None,
        cer: None,
        rtf_warm: None,
        rtf_e2e: None,
        load_secs: None,
        wall_secs: None,
        media_secs: Some(media_secs),
        clips_ok: 0,
        clips_total: holdout.len(),
        notes: Vec::new(),
        peak_bytes: None,
        steady_bytes: None,
        map50: None,
        map5095: None,
    };

    // Decode audio ONCE, outside the timed region: every implementation gets
    // the same 16 kHz mono samples, so the comparison measures inference, not
    // our WAV reader.
    let mut decoded = Vec::new();
    for clip in holdout {
        match ffai_media::load_audio(&manifest.clip_path(clip)) {
            Ok(audio) => decoded.push((*clip, audio)),
            Err(e) => summary.notes.push(format!("{}: {e}", clip.id)),
        }
    }

    // Warm-up pass, untimed: engines load weights lazily on first use, and
    // references report their load separately. Without this the engine's
    // first timed run would carry model load that no reference's does —
    // the same class of unfairness as timing Python's startup, pointed the
    // other way. Its cost is recorded so it isn't invisible.
    if let Some((_, audio)) = decoded.first() {
        let t0 = std::time::Instant::now();
        if let Err(e) = asr.transcribe(audio, &opts) {
            summary.notes.push(format!("warm-up failed: {e}"));
        }
        summary.load_secs = Some(t0.elapsed().as_secs_f64());
    }

    let (mut wers, mut cers) = (Vec::new(), Vec::new());
    let mut texts: Vec<(String, String)> = Vec::new();
    let mut first_error = None;

    // Sample OUR resident memory the same way the reference tree is sampled,
    // so the two steady-state figures are the same measurement. Peak alone
    // compares our load spike against theirs and calls the result footprint.
    let sampling = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let our_samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    let sampler = {
        let (sampling, out) = (sampling.clone(), our_samples.clone());
        std::thread::spawn(move || {
            while sampling.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(b) = crate::footprint::current_self()
                    && let Ok(mut v) = out.lock()
                {
                    v.push(b.0);
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        })
    };

    let stats = best_of_n(runs, || {
        texts.clear();
        for (clip, audio) in &decoded {
            match asr.transcribe(audio, &opts) {
                Ok(t) => texts.push((clip.id.clone(), t.text())),
                Err(e) => {
                    first_error.get_or_insert(format!("{}: {e}", clip.id));
                    return Err(e);
                }
            }
        }
        Ok(())
    });

    match stats {
        Ok(stats) => {
            summary.clips_ok = texts.len();
            summary.wall_secs = Some(stats.best_secs);
            // Timed runs are all warm (see the warm-up pass above), so the
            // harness-timed figure IS steady-state throughput. End-to-end
            // adds the recorded load, matching how references are scored.
            summary.rtf_warm = Some(real_time_factor(media_secs, stats.best_secs));
            summary.rtf_e2e = Some(real_time_factor(
                media_secs,
                stats.best_secs + summary.load_secs.unwrap_or(0.0),
            ));
            for (id, text) in &texts {
                let clip = holdout.iter().find(|c| &c.id == id);
                if let Some(clip) = clip
                    && let Some(truth) = manifest.ground_truth(clip)?
                {
                    wers.push(wer_with(&truth, text, Mode::English));
                    cers.push(cer_with(&truth, text, Mode::English));
                }
            }
            summary.wer = mean(&wers);
            summary.cer = mean(&cers);
        }
        // FRONT, not back. The correctness gate reports `notes.first()` as the
        // "first failure", and by this point `notes` already holds
        // informational entries — the footprint-exclusion note among them. A
        // failing run therefore reported a memory-accounting sentence as its
        // cause, which read as a harness quirk rather than a real error and
        // left two engines wrongly described as "unmeasured rather than
        // broken". The error goes where the gate actually looks.
        Err(e) => summary
            .notes
            .insert(0, first_error.unwrap_or_else(|| e.to_string())),
    }

    // Peak resident memory for our process. Taken after the timed loop so it
    // covers weight loading, the KV caches and every intermediate.
    //
    // ONE ASYMMETRY, STATED RATHER THAN BURIED: this process holds every clip
    // pre-decoded (above, deliberately, so the speed comparison measures
    // inference and not our WAV reader), while a reference reads its audio one
    // clip at a time. That buffer is the harness's choice, not the engine's,
    // and it inflates OUR side only. It is recorded next to the peak so the
    // number can be read honestly — a footprint claim that quietly counted the
    // harness would be the memory version of the `-nt` flag.
    let decoded_bytes: u64 = decoded
        .iter()
        .map(|(_, a)| (a.samples.len() * size_of::<f32>()) as u64)
        .sum();
    sampling.store(false, std::sync::atomic::Ordering::Relaxed);
    sampler.join().ok();
    summary.steady_bytes = our_samples.lock().ok().and_then(|mut v| {
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    });

    // Peak WORKING SET — resident memory, the same quantity sampled across the
    // reference's process tree. Not commit: calibration showed commit counts
    // address space that is reserved and never faulted in.
    summary.peak_bytes = crate::footprint::peak_self().map(|p| p.0);
    if summary.peak_bytes.is_some() && decoded_bytes > 0 {
        summary.notes.push(format!(
            "peak includes {:.1} MiB of pre-decoded audio held by the harness, \
             which the reference does not hold",
            decoded_bytes as f64 / (1024.0 * 1024.0)
        ));
    }
    Ok(summary)
}

pub(crate) fn fill_gates(
    gates: &mut GateReport,
    engine: Option<&RunSummary>,
    references: &[RunSummary],
    quality: QualityMetric,
) {
    let metric_of = |r: &RunSummary| match quality {
        QualityMetric::Wer => r.wer,
        QualityMetric::Cer => r.cer,
        QualityMetric::Map50 => r.map50.map(|m| 1.0 - m),
    };
    let metric_label = match quality {
        QualityMetric::Wer => "WER",
        QualityMetric::Cer => "CER",
        QualityMetric::Map50 => "1-mAP50",
    };
    let Some(eng) = engine else {
        for kind in GateKind::ALL {
            gates.set(GateResult::skipped(kind, "baseline-only run (no engine)"));
        }
        return;
    };

    gates.set(if eng.clips_ok == eng.clips_total && eng.clips_total > 0 {
        GateResult {
            kind: GateKind::Correctness,
            outcome: GateOutcome::Pass,
            metric: None,
            detail: format!(
                "{}/{} holdout clips processed",
                eng.clips_ok, eng.clips_total
            ),
        }
    } else {
        GateResult {
            kind: GateKind::Correctness,
            outcome: GateOutcome::Fail,
            metric: None,
            detail: format!(
                "{}/{} holdout clips processed; first failure: {}",
                eng.clips_ok,
                eng.clips_total,
                eng.notes.first().map_or("-", String::as_str)
            ),
        }
    });

    // The quality gate asks ONE question: is our implementation as good as
    // other implementations of the same thing? It therefore compares only
    // against references declaring the engine's own configuration.
    //
    // It used to take the lowest WER of everything that ran, which for a 39M
    // greedy engine meant being judged against openai-whisper-base — a 74M
    // beam-search model. That gate read FAIL for months for reasons having
    // nothing to do with implementation quality, and "are we as good as the
    // best ASR available?" was being reported under the same label. That is a
    // real and harder question; it belongs in the record as context, not as
    // this gate's verdict.
    let engine_key = eng.config.get(DECODE_KEY);
    let matched: Vec<(&str, f64)> = references
        .iter()
        .filter(|r| engine_key.is_some() && r.config.get(DECODE_KEY) == engine_key)
        .filter_map(|r| metric_of(r).map(|w| (r.name.as_str(), w)))
        .collect();
    let best_matched = matched.iter().copied().min_by(|a, b| a.1.total_cmp(&b.1));
    let best_any = references
        .iter()
        .filter_map(|r| metric_of(r).map(|w| (r.name.as_str(), w)))
        .min_by(|a, b| a.1.total_cmp(&b.1));

    // Context, never a verdict: where the engine sits against the strongest
    // thing that ran, whatever size or decoding it used.
    let open = match best_any {
        Some((n, w)) => format!(" [open field: best is {n} {:.2}%]", w * 100.0),
        None => String::new(),
    };

    gates.set(match (metric_of(eng), best_matched) {
        (Some(ew), Some((rname, rw))) => GateResult {
            kind: GateKind::Quality,
            outcome: if ew <= rw * QUALITY_PARITY_BAND || ew <= rw + QUALITY_ABS_FLOOR {
                GateOutcome::Pass
            } else {
                GateOutcome::Fail
            },
            metric: Some(ew),
            detail: format!(
                "engine {metric_label} {:.2}% vs best matched reference {rname} {:.2}% \
                 ({} config, band +{:.0}% relative or +{:.2} pp absolute){open}",
                ew * 100.0,
                rw * 100.0,
                engine_key.map_or("?", String::as_str),
                QUALITY_PARITY_BAND * 100.0 - 100.0,
                QUALITY_ABS_FLOOR * 100.0
            ),
        },
        // No comparable reference is a SKIP, never a pass. Falling back to the
        // open field here would quietly restore the defect this split exists
        // to remove.
        (Some(ew), None) => GateResult::skipped(
            GateKind::Quality,
            format!(
                "engine {metric_label} {:.2}% but no reference declares its configuration ({}) — \
                 declare one in corpora/references.toml{open}",
                ew * 100.0,
                engine_key.map_or("unknown", String::as_str)
            ),
        ),
        _ => GateResult::skipped(GateKind::Quality, "engine produced no scored output"),
    });

    // Speed compares WARM throughput: steady-state inference, the fair
    // engineering comparison. End-to-end appears in the detail line so the
    // startup story is never hidden.
    let best_ref_rtf = references
        .iter()
        .filter_map(|r| r.rtf_warm.map(|x| (r.name.as_str(), x)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    gates.set(match (eng.rtf_warm, best_ref_rtf) {
        (Some(er), Some((rname, rr))) => GateResult {
            kind: GateKind::Speed,
            outcome: if er >= rr {
                GateOutcome::Pass
            } else {
                GateOutcome::Fail
            },
            metric: Some(er),
            detail: format!(
                "engine {er:.1}x realtime (warm) vs fastest reference {rname} {rr:.1}x; \
                 engine e2e {:.1}x",
                eng.rtf_e2e.unwrap_or(f64::NAN)
            ),
        },
        (Some(er), None) => GateResult::skipped(
            GateKind::Speed,
            format!("engine {er:.1}x realtime but no reference ran to compare against"),
        ),
        _ => GateResult::skipped(GateKind::Speed, "engine produced no timed runs"),
    });

    // Footprint: peak resident memory against the leanest reference.
    //
    // This gate printed SKIP from Phase 0 until it was built, which was never
    // neutral — `all_passed` counts a skipped gate as not passed, so no run
    // could ever clear a verdict, and two shipped optimizations whose entire
    // value is memory (the int8 decoder variant, the f16 cross-attention
    // cache) had nothing to be recorded against.
    //
    // Compared against the SMALLEST reference rather than the fastest: the
    // question this gate answers is "does the pure-Rust build cost less to
    // run", and the honest bar is the leanest thing on the board.
    // Footprint judged on STEADY resident memory, with peak reported beside it.
    //
    // Peak is dominated by model load — a spike that is over in half a second
    // and never recurs. Measured directly: our process holds 345 MiB after a
    // run, but trimming and repeating the SAME work re-settles at 102 MiB, so
    // ~240 MiB of that peak was load transients and allocator retention, not
    // memory the work needs. Judging on peak would compare our load spike
    // against theirs and call the result footprint.
    //
    // Both are recorded because they answer different questions: peak decides
    // whether it FITS, steady decides what it COSTS to keep running.
    let leanest = references
        .iter()
        .filter_map(|r| {
            r.steady_bytes
                .or(r.peak_bytes)
                .map(|b| (r.name.as_str(), b))
        })
        .min_by_key(|(_, b)| *b);
    const MIB: f64 = 1024.0 * 1024.0;
    let ours = eng.steady_bytes.or(eng.peak_bytes);
    gates.set(match (ours, leanest) {
        (Some(ep), Some((rname, rp))) => GateResult {
            kind: GateKind::Footprint,
            outcome: if ep <= rp { GateOutcome::Pass } else { GateOutcome::Fail },
            metric: Some(ep as f64 / MIB),
            detail: format!(
                "engine steady {:.0} MiB (peak {:.0}) vs leanest reference {rname} steady                  {:.0} MiB (peak {:.0}) — {:.2}x",
                ep as f64 / MIB,
                eng.peak_bytes.unwrap_or(0) as f64 / MIB,
                rp as f64 / MIB,
                references
                    .iter()
                    .find(|r| r.name == rname)
                    .and_then(|r| r.peak_bytes)
                    .unwrap_or(0) as f64
                    / MIB,
                ep as f64 / rp as f64
            ),
        },
        (Some(ep), None) => GateResult::skipped(
            GateKind::Footprint,
            format!(
                "engine steady {:.0} MiB but no reference reported memory to compare against",
                ep as f64 / MIB
            ),
        ),
        _ => GateResult::skipped(
            GateKind::Footprint,
            if crate::footprint::supported() {
                "memory was not captured for this run".to_string()
            } else {
                format!(
                    "memory measurement is not implemented on {} —                      see crates/ffai-bench/src/footprint.rs",
                    std::env::consts::OS
                )
            },
        ),
    });
}

fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        None
    } else {
        Some(xs.iter().sum::<f64>() / xs.len() as f64)
    }
}

/// Render a bench record as a human table (the CLI's output).
#[must_use]
pub fn render(record: &BenchRecord) -> String {
    // The media unit and rate labels are task-scoped: ASR media is seconds of
    // audio and the rate is ×realtime; OCR media is pages and the rate is
    // pages/second; detect media is images and the rate is images/second, with
    // the two quality columns carrying mAP instead of error rates (higher is
    // better — the column heads say which). Same ledger fields either way
    // (see `run_ocr` / `run_detect`).
    let ocr = record.task == "ocr";
    let detect = record.task == "detect";
    let media = record
        .engine
        .as_ref()
        .and_then(|e| e.media_secs)
        .or_else(|| record.references.first().and_then(|r| r.media_secs))
        .unwrap_or(0.0);
    let mut out = String::new();
    out.push_str(&format!(
        "bench {} · corpus {} ({}) · {}\n\n",
        record.id,
        record.corpus,
        &record.corpus_manifest_hash[..12.min(record.corpus_manifest_hash.len())],
        if ocr {
            format!("{media:.0} pages")
        } else if detect {
            format!("{media:.0} images")
        } else {
            format!("{media:.1}s audio")
        },
    ));
    out.push_str(&format!(
        "{:<24} {:>7} {:>7} {:>10} {:>10} {:>8} {:>9} {:>7}\n",
        "IMPLEMENTATION",
        if detect { "mAP50" } else { "WER%" },
        if detect { "mAP5095" } else { "CER%" },
        if ocr {
            "PG/S_WARM"
        } else if detect {
            "IMG/S_WARM"
        } else {
            "xRT_WARM"
        },
        if ocr {
            "PG/S_E2E"
        } else if detect {
            "IMG/S_E2E"
        } else {
            "xRT_E2E"
        },
        "LOAD_S",
        "PEAK_MiB",
        "CLIPS"
    ));
    let rate = |r: f64| {
        if ocr || detect {
            format!("{r:.2}")
        } else {
            format!("{r:.1}x")
        }
    };
    let mut row = |s: &RunSummary, marker: &str| {
        let (q1, q2) = if detect {
            (s.map50, s.map5095)
        } else {
            (s.wer, s.cer)
        };
        out.push_str(&format!(
            "{:<24} {:>7} {:>7} {:>10} {:>10} {:>8} {:>9} {:>7}\n",
            format!("{marker}{}", s.name),
            q1.map_or_else(|| "-".into(), |w| format!("{:.2}", w * 100.0)),
            q2.map_or_else(|| "-".into(), |c| format!("{:.2}", c * 100.0)),
            s.rtf_warm.map_or_else(|| "-".into(), &rate),
            s.rtf_e2e.map_or_else(|| "-".into(), &rate),
            s.load_secs
                .map_or_else(|| "-".into(), |l| format!("{l:.2}")),
            s.peak_bytes.map_or_else(
                || "-".into(),
                |b| format!("{:.0}", b as f64 / (1024.0 * 1024.0))
            ),
            format!("{}/{}", s.clips_ok, s.clips_total),
        ));
    };
    if let Some(eng) = &record.engine {
        row(eng, "> ");
    }
    for r in &record.references {
        row(r, "  ");
    }
    for s in record.engine.iter().chain(record.references.iter()) {
        for note in &s.notes {
            out.push_str(&format!("  ! {}: {note}\n", s.name));
        }
    }
    out.push('\n');
    for g in &record.gates.results {
        out.push_str(&format!(
            "{:<12} {}  {}\n",
            g.kind.label(),
            g.outcome,
            g.detail
        ));
    }
    out.push_str(&format!(
        "\nverdict: {}\n",
        if record.gates.all_passed() {
            "ALL GATES PASS — claimable"
        } else {
            "not claimable yet"
        }
    ));
    out
}

/// Everything that can change this engine's output, captured for the ledger.
///
/// References record their argv; an in-process engine has none, and that
/// asymmetry cost a real defect. When speech segmentation became a default,
/// `whisper-candle` produced 7.99 % WER one day and 6.79 % the next with
/// nothing in the record to distinguish the two runs — and two READMEs and a
/// crates.io release were cut from those numbers.
///
/// Two classes go in. The `AsrOptions` the harness actually passed, and every
/// `FFAI_*` environment override that is set: those exist to A/B behaviour
/// from outside the process, which makes them precisely the thing a reader
/// cannot reconstruct afterwards. Compile-time defaults are pinned by the
/// crate version already recorded alongside.
///
/// Absent knobs are absent, not `"unset"` — a key that appears only when it
/// is doing something keeps the common line short and makes an override
/// impossible to miss.
fn engine_config(opts: &AsrOptions) -> std::collections::BTreeMap<String, String> {
    let mut config = std::collections::BTreeMap::new();
    config.insert("vad".to_string(), opts.vad.to_string());
    config.insert(
        "vad_threshold".to_string(),
        format!("{:.3}", opts.vad_threshold),
    );
    config.insert(
        "vad_chunk_secs".to_string(),
        format!("{:.1}", opts.vad_chunk_secs),
    );
    config.insert(
        "word_timestamps".to_string(),
        opts.word_timestamps.to_string(),
    );
    config.insert("diarize".to_string(), opts.diarize.to_string());
    config.insert("translate".to_string(), opts.translate.to_string());
    config.insert(
        "language".to_string(),
        opts.language.clone().unwrap_or_else(|| "auto".to_string()),
    );

    // Behaviour-changing overrides only. FFAI_CACHE relocates the weight
    // cache and FFAI_DEBUG_TOKENS / FFAI_PROFILE only print, so none of them
    // alter a transcript; listing them would be noise in every record.
    const OVERRIDES: &[&str] = &[
        "FFAI_VAD",
        "FFAI_ALLOW_NONSPEECH",
        "FFAI_PRECISION",
        "FFAI_CANDLE_DECODER",
        "FFAI_CANDLE_ENCODER",
        "FFAI_KV_F16",
        "FFAI_DEC_F16",
        "FFAI_MLP_INT8",
        "FFAI_PAR_HEADS",
        "FFAI_DECODE_MIN_KEYS",
        "FFAI_VOCAB_BLK",
        "FFAI_GEMV_PAD",
        "FFAI_ENC_KT",
    ];
    for key in OVERRIDES {
        if let Ok(value) = std::env::var(key) {
            config.insert(key.to_ascii_lowercase(), value);
        }
    }
    config
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn records_the_setting_that_was_missing() {
        // The specific regression this field exists to prevent: two runs whose
        // only difference is segmentation must not serialise identically.
        let on = engine_config(&AsrOptions {
            vad: true,
            ..Default::default()
        });
        let off = engine_config(&AsrOptions {
            vad: false,
            ..Default::default()
        });
        assert_eq!(on["vad"], "true");
        assert_eq!(off["vad"], "false");
        assert_ne!(on, off);
    }

    #[test]
    fn unset_overrides_do_not_appear() {
        let config = engine_config(&AsrOptions::default());
        assert!(!config.contains_key("ffai_vad"), "{config:?}");
        assert!(!config.contains_key("ffai_allow_nonspeech"), "{config:?}");
    }

    #[test]
    fn thresholds_are_recorded_not_just_the_on_off() {
        // "vad: true" is not reproducible on its own — the threshold changes
        // which audio survives.
        let config = engine_config(&AsrOptions {
            vad_threshold: 0.25,
            ..Default::default()
        });
        assert_eq!(config["vad_threshold"], "0.250");
    }
}

/// Key under which both sides record the decode configuration they represent.
pub const DECODE_KEY: &str = "decode";

/// A reference's declared configuration, as a one-entry config map.
fn decode_config(config: Option<&str>) -> std::collections::BTreeMap<String, String> {
    let mut map = std::collections::BTreeMap::new();
    if let Some(c) = config {
        map.insert(DECODE_KEY.to_string(), c.to_string());
    }
    map
}

/// The engine's comparison key, in the same vocabulary references use.
///
/// This reads the model size out of the engine's own name, which is a
/// convention rather than a declaration — but it is *our* convention, defined
/// and documented in `WhisperCandle::info` (bare `whisper-candle` is tiny.en,
/// larger sizes carry a suffix), not a guess about somebody else's naming.
/// An unrecognised engine returns `None` and its quality gate SKIPS rather
/// than silently comparing against something it does not match.
///
/// Precision suffixes (`-q8_0`, `-f16`) are deliberately ignored: they change
/// memory and speed, not the decoding strategy, so they do not change which
/// references the output is comparable to.
fn engine_comparison_key(name: &str) -> Option<String> {
    let size = if name.starts_with("whisper-candle-base") {
        "base.en"
    } else if name.starts_with("whisper-candle-small") {
        "small.en"
    } else if name.starts_with("whisper-candle") {
        "tiny.en"
    } else {
        return None;
    };
    // Mercury decodes greedily: the temperature ladder starts at 0.0 and only
    // rises on a low-confidence retry. There is no beam search.
    Some(format!("{size}/greedy"))
}

#[cfg(test)]
mod gate_split_tests {
    use super::*;

    fn summary(name: &str, wer: f64, decode: Option<&str>) -> RunSummary {
        RunSummary {
            name: name.into(),
            version: None,
            command: None,
            config: decode_config(decode),
            wer: Some(wer),
            cer: None,
            rtf_warm: Some(30.0),
            rtf_e2e: Some(30.0),
            load_secs: None,
            wall_secs: None,
            media_secs: Some(100.0),
            clips_ok: 10,
            clips_total: 10,
            notes: Vec::new(),
            peak_bytes: None,
            steady_bytes: None,
            map50: None,
            map5095: None,
        }
    }

    fn engine(wer: f64) -> RunSummary {
        let mut e = summary("whisper-candle", wer, Some("tiny.en/greedy"));
        e.config.insert("vad".into(), "true".into());
        e
    }

    #[test]
    fn a_bigger_model_no_longer_decides_the_gate() {
        // The exact historical defect: 6.79% engine failed against
        // openai-whisper-base's 5.96%, a 74M beam-search model.
        let refs = vec![
            summary("openai-whisper-base", 0.0596, Some("base.en/beam5")),
            summary("whisper-cpp-tiny-greedy", 0.0758, Some("tiny.en/greedy")),
        ];
        let mut gates = GateReport::new();
        fill_gates(&mut gates, Some(&engine(0.0679)), &refs, QualityMetric::Wer);
        let q = gates.get(GateKind::Quality).expect("quality gate set");
        assert_eq!(q.outcome, GateOutcome::Pass, "{}", q.detail);
        assert!(q.detail.contains("whisper-cpp-tiny-greedy"), "{}", q.detail);
    }

    #[test]
    fn the_open_field_is_still_reported_as_context() {
        let refs = vec![
            summary("openai-whisper-base", 0.0596, Some("base.en/beam5")),
            summary("whisper-cpp-tiny-greedy", 0.0758, Some("tiny.en/greedy")),
        ];
        let mut gates = GateReport::new();
        fill_gates(&mut gates, Some(&engine(0.0679)), &refs, QualityMetric::Wer);
        let q = gates.get(GateKind::Quality).expect("set");
        assert!(q.detail.contains("open field"), "{}", q.detail);
        assert!(q.detail.contains("openai-whisper-base"), "{}", q.detail);
    }

    #[test]
    fn a_matched_reference_can_still_fail_us() {
        // The gate must retain its teeth against comparable implementations.
        let refs = vec![summary(
            "whisper-cpp-tiny-greedy",
            0.0600,
            Some("tiny.en/greedy"),
        )];
        let mut gates = GateReport::new();
        fill_gates(&mut gates, Some(&engine(0.0900)), &refs, QualityMetric::Wer);
        assert_eq!(
            gates.get(GateKind::Quality).expect("set").outcome,
            GateOutcome::Fail
        );
    }

    #[test]
    fn no_matched_reference_skips_rather_than_passes() {
        // A skipped gate is never a pass — and must not fall back to the open
        // field, which would restore the defect.
        let refs = vec![summary(
            "openai-whisper-base",
            0.0596,
            Some("base.en/beam5"),
        )];
        let mut gates = GateReport::new();
        fill_gates(&mut gates, Some(&engine(0.0679)), &refs, QualityMetric::Wer);
        let q = gates.get(GateKind::Quality).expect("set");
        assert_eq!(q.outcome, GateOutcome::Skipped, "{}", q.detail);
        assert!(!gates.all_passed());
    }

    #[test]
    fn precision_variants_compare_against_the_same_references() {
        assert_eq!(
            engine_comparison_key("whisper-candle-q8_0").as_deref(),
            Some("tiny.en/greedy")
        );
        assert_eq!(
            engine_comparison_key("whisper-candle-base").as_deref(),
            Some("base.en/greedy")
        );
        assert_eq!(engine_comparison_key("oxi-whisper"), None);
    }
}
