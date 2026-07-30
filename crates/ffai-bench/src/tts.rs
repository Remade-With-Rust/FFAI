//! The TTS bench vertical: synthesize a pinned text corpus, judge every
//! implementation's audio with the same frozen third-party ASR, score
//! round-trip WER/CER against the input text, and append the record to the
//! ledger. Mission plan: docs/mercury-tts-mission.md §5.
//!
//! Differences from ASR, all deliberate:
//!
//! - **The corpus pins TEXT, not audio.** A clip's `path` is the input
//!   sentence and its `ground_truth` is the same file — the round-trip
//!   target.
//! - **`media_secs` is GENERATED audio**, measured by the harness from the
//!   WAVs of the kept run (never trusted from the adapter), so `rtf_warm`
//!   reads as seconds of speech synthesized per second of processing.
//! - **Quality is round-trip intelligibility** through a single frozen judge
//!   (`task = "tts-judge"` in references.toml) applied identically to every
//!   implementation's audio — including ours, later. No self-grading: the
//!   judge is external, pinned, and never a Mercury engine.
//! - **Judge audio is normalized by the harness** ([`crate::resample`]):
//!   every WAV becomes 16 kHz mono through the same code path, so no
//!   implementation's score depends on its native sample rate.
//! - **The judge pass is untimed.** It runs after the synthesis batch, so
//!   judging cost never lands in any implementation's speed numbers.
//! - **Time-to-first-audio percentiles** ride in the notes (the per-item
//!   latency convention `run_ocr` established), adapter-timed, warm.
//!
//! Piper caveat, recorded here because it shapes how to read the ledger:
//! piper samples noise inside its ONNX graph with no seed control, so its
//! audio — and therefore its round-trip WER — varies run to run. The harness
//! scores the audio of the run whose wall clock it kept. Treat single-run
//! WER deltas smaller than a point as noise until measured across seeds
//! (mission plan §8).

use std::path::{Path, PathBuf};

use ffai_core::engine::TtsOptions;
use ffai_core::error::{Error, Result};
use ffai_core::registry::EngineRegistry;

use crate::corpus::{ClipEntry, Manifest};
use crate::gate::GateReport;
use crate::ledger::{append, BenchRecord, Environment, RunSummary, LEDGER_SCHEMA};
use crate::metrics::{cer_with, wer_with};
use crate::normalize::Mode;
use crate::reference::{ReferenceSpec, TtsBatchResult};
use crate::runner::{fill_gates, BenchConfig, QualityMetric};
use crate::speed::{best_of_n, real_time_factor};

/// The judge's required input format (whisper-family ASR).
const JUDGE_RATE: u32 = 16_000;

/// Where generated audio lands, one subdirectory per implementation. Kept on
/// disk (not temp) so a spot-listen of what was actually scored is one click
/// away; gitignored.
const OUT_ROOT: &str = "bench/tts-out";

/// Run a TTS bench end-to-end and append the record to the ledger.
///
/// `cfg.references` carries both the TTS implementations (`task = "tts"`)
/// and the round-trip judge (`task = "tts-judge"`); exactly one judge must be
/// present — zero would mean unscored quality, two would mean two different
/// error floors in one table.
pub fn run_tts(reg: &EngineRegistry, cfg: &BenchConfig) -> Result<BenchRecord> {
    let manifest = Manifest::load(&cfg.corpus)?;
    if manifest.task != "tts" {
        return Err(Error::Other(format!(
            "corpus `{}` is a {} corpus, not tts",
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
        return Err(Error::Other("corpus has no holdout clips — nothing to measure".into()));
    }
    let paths: Vec<PathBuf> = holdout.iter().map(|c| manifest.clip_path(c)).collect();

    let (tts_refs, judges): (Vec<&ReferenceSpec>, Vec<&ReferenceSpec>) =
        cfg.references.iter().partition(|r| r.task == "tts");
    let judges: Vec<&ReferenceSpec> = judges.into_iter().filter(|r| r.task == "tts-judge").collect();
    let judge = match judges.as_slice() {
        [one] => *one,
        [] => {
            return Err(Error::Other(
                "no round-trip judge configured — declare exactly one `task = \"tts-judge\"` \
                 reference in corpora/references.toml (a frozen third-party ASR; never a \
                 Mercury engine)"
                    .into(),
            ))
        }
        many => {
            return Err(Error::Other(format!(
                "{} tts-judge references configured — exactly one, or scores in the same table \
                 would carry different judge error floors",
                many.len()
            )))
        }
    };

    let mut references = Vec::new();
    for spec in &tts_refs {
        eprintln!("running reference `{}` over {} utterances ...", spec.name, holdout.len());
        match run_tts_reference(spec, judge, &manifest, &holdout, &paths, cfg.runs) {
            Ok(summary) => references.push(summary),
            Err(e) => {
                eprintln!("  reference `{}` failed: {e}", spec.name);
                references.push(empty_summary(
                    spec.name.clone(),
                    spec.version(),
                    Some(spec.command_line()),
                    voice_config(spec.config.as_deref()),
                    holdout.len(),
                    vec![e.to_string()],
                ));
            }
        }
    }

    let engine_summary = if cfg.skip_engine {
        None
    } else {
        Some(run_tts_engine(reg, cfg.engine.as_deref(), judge, &manifest, &holdout, cfg.runs)?)
    };

    let mut gates = GateReport::new();
    fill_gates(&mut gates, engine_summary.as_ref(), &references, QualityMetric::Wer);

    let (id, appended_at) = BenchRecord::now_id("tts");
    let record = BenchRecord {
        schema: LEDGER_SCHEMA,
        id,
        task: "tts".into(),
        corpus: manifest.name.clone(),
        corpus_manifest_hash: manifest.manifest_hash(),
        engine: engine_summary,
        references,
        gates,
        environment: Environment::capture(),
        notes: format!(
            "round-trip judge: {} ({}) — WER/CER are input-text vs judge-transcript through \
             the same frozen ASR for every implementation, not a transcription benchmark",
            judge.name,
            judge.version().unwrap_or_else(|| "version unknown".into()),
        ),
        appended_at,
    };
    append(&cfg.ledger, &record)?;
    Ok(record)
}

fn run_tts_reference(
    spec: &ReferenceSpec,
    judge: &ReferenceSpec,
    manifest: &Manifest,
    holdout: &[&ClipEntry],
    paths: &[PathBuf],
    runs: usize,
) -> Result<RunSummary> {
    if !spec.supports_batch() {
        return Err(Error::Other(format!(
            "reference `{}` declares no batch_command — per-utterance invocation would put \
             interpreter startup and model load inside every timed run (see \
             crates/ffai-bench/src/reference.rs)",
            spec.name
        )));
    }
    let outdir = PathBuf::from(OUT_ROOT).join(&spec.name);

    // One invocation over the whole corpus, repeated. The FASTEST run's wall
    // clock and adapter timings stay together (timings only compose from one
    // execution) — but QUALITY is scored on EVERY run's audio and averaged:
    // a reference that samples noise in-graph produces a different WER draw
    // per run (observed spread 5.41–6.45 % across seven draws), and gating
    // on one draw is gating on luck. The per-run draws land in the notes.
    let mut best: Option<(f64, usize)> = None;
    let mut batches = Vec::new();
    for run in 0..runs.max(1) {
        let run_dir = outdir.join(format!("run{run}"));
        let started = std::time::Instant::now();
        let batch = spec.run_batch_tts(paths, &run_dir)?;
        let wall = started.elapsed().as_secs_f64();
        if best.as_ref().is_none_or(|(prev, _)| wall < *prev) {
            best = Some((wall, run));
        }
        batches.push((run_dir, batch));
    }
    let (wall_secs, best_run) = best.expect("at least one run");

    let mut summary = empty_summary(
        spec.name.clone(),
        spec.version(),
        Some(spec.command_line()),
        voice_config(spec.config.as_deref()),
        holdout.len(),
        Vec::new(),
    );
    summary.notes.extend(batches[best_run].1.meta.iter().cloned());

    // Judge every run's audio; the kept run also fills clips_ok/media_secs.
    let mut draws: Vec<(f64, f64)> = Vec::new();
    for (run, (run_dir, batch)) in batches.iter().enumerate() {
        let mut scratch = empty_summary(String::new(), None, None, Default::default(), 0, vec![]);
        let target = if run == best_run { &mut summary } else { &mut scratch };
        let judged = prepare_judge_wavs(batch, holdout, paths, run_dir, target)?;
        score_with_judge(judge, &judged, manifest, Mode::English, target)?;
        if let (Some(w), Some(c)) = (target.wer, target.cer) {
            draws.push((w, c));
        }
    }
    if !draws.is_empty() {
        summary.wer = Some(draws.iter().map(|(w, _)| w).sum::<f64>() / draws.len() as f64);
        summary.cer = Some(draws.iter().map(|(_, c)| c).sum::<f64>() / draws.len() as f64);
        if draws.len() > 1 {
            let lo = draws.iter().map(|(w, _)| *w).fold(f64::MAX, f64::min);
            let hi = draws.iter().map(|(w, _)| *w).fold(f64::MIN, f64::max);
            summary.notes.push(format!(
                "quality is the MEAN of {} independent draws (WER range {:.2}–{:.2} %); \
                 the reference samples noise in-graph, so one draw is one sample",
                draws.len(),
                lo * 100.0,
                hi * 100.0
            ));
        }
    }
    let batch = &batches[best_run].1;

    // Speed: warm = adapter-reported synthesis seconds; e2e = our wall clock
    // (one load + synthesis + WAV writing) — both against the audio actually
    // produced.
    let media_secs = summary.media_secs.unwrap_or(0.0);
    summary.wall_secs = Some(wall_secs);
    summary.load_secs = batch.load_secs;
    summary.rtf_e2e = Some(real_time_factor(media_secs, wall_secs));
    summary.rtf_warm =
        batch.synth_secs().map(|t| real_time_factor(media_secs, t)).or(summary.rtf_e2e);
    summary.peak_bytes = batch.peak_bytes;
    summary.steady_bytes = batch.steady_bytes;

    // Time-to-first-audio percentiles: the latency a streaming caller
    // experiences, kept beside the corpus totals from day one so the M-T4
    // gate (< 500 ms) has its baseline waiting.
    let mut ttfa: Vec<f64> = batch.clips.iter().filter_map(|c| c.ttfa_secs).collect();
    if !ttfa.is_empty() {
        ttfa.sort_by(|a, b| a.total_cmp(b));
        let pct = |p: f64| ttfa[((ttfa.len() - 1) as f64 * p).round() as usize];
        summary.notes.push(format!(
            "ttfa p50 {:.0} ms / p95 {:.0} ms over {} utterances (adapter-timed, warm)",
            pct(0.50) * 1000.0,
            pct(0.95) * 1000.0,
            ttfa.len()
        ));
    }
    Ok(summary)
}

/// The in-process TTS engine's run. Compact by design (the `run_ocr_engine`
/// precedent): at M-T0 every registered TTS engine is an honest stub, and
/// this exists so `ffai bench tts` without `--baseline-only` records that
/// failure rather than hiding it. The memory sampler lands with piper-candle
/// at M-T2, when there is an engine worth measuring.
fn run_tts_engine(
    reg: &EngineRegistry,
    engine: Option<&str>,
    judge: &ReferenceSpec,
    manifest: &Manifest,
    holdout: &[&ClipEntry],
    runs: usize,
) -> Result<RunSummary> {
    let tts = reg.tts(engine)?;
    let opts = TtsOptions::default();
    // The engine's comparison key, in the same vocabulary the piper
    // reference declares (`config = "<voice>/defaults"`). The default-voice
    // name is Mercury's documented convention (piper_candle::DEFAULT_VOICE
    // is en_US-lessac-medium) — the same our-convention-not-a-guess
    // reasoning as ASR's engine_comparison_key.
    let voice = opts.voice.clone().unwrap_or_else(|| "en_US-lessac-medium".to_string());
    let mut config = std::collections::BTreeMap::new();
    config.insert(crate::runner::DECODE_KEY.to_string(), format!("{voice}/defaults"));
    let mut summary = empty_summary(
        tts.info().name,
        Some(env!("CARGO_PKG_VERSION").to_string()),
        None,
        config,
        holdout.len(),
        Vec::new(),
    );

    let mut texts = Vec::new();
    for clip in holdout {
        texts.push((*clip, std::fs::read_to_string(manifest.clip_path(clip))?));
    }

    // Warm-up pass, untimed, load recorded — identical contract to ASR.
    if let Some((_, text)) = texts.first() {
        let t0 = std::time::Instant::now();
        if let Err(e) = tts.synthesize(text, &opts) {
            summary.notes.push(format!("warm-up failed: {e}"));
        }
        summary.load_secs = Some(t0.elapsed().as_secs_f64());
    }

    // Sample OUR resident memory the same way the reference tree is sampled
    // (the ASR harness's pattern): steady = median while working, peak
    // recorded beside it. This was the M-T2 footprint-gate SKIP.
    let sampling = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let our_samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    let sampler = {
        let (sampling, out) = (sampling.clone(), our_samples.clone());
        std::thread::spawn(move || {
            while sampling.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(b) = crate::footprint::current_self() {
                    if let Ok(mut v) = out.lock() {
                        v.push(b.0);
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        })
    };

    let mut audio: Vec<(String, ffai_core::types::AudioBuffer)> = Vec::new();
    let mut first_error = None;
    let stats = best_of_n(runs, || {
        audio.clear();
        for (clip, text) in &texts {
            match tts.synthesize(text, &opts) {
                Ok(buf) => audio.push((clip.id.clone(), buf)),
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
            summary.clips_ok = audio.len();
            summary.wall_secs = Some(stats.best_secs);
            let media_secs: f64 = audio.iter().map(|(_, a)| a.duration_secs()).sum();
            summary.media_secs = Some(media_secs);
            summary.rtf_warm = Some(real_time_factor(media_secs, stats.best_secs));
            summary.rtf_e2e = Some(real_time_factor(
                media_secs,
                stats.best_secs + summary.load_secs.unwrap_or(0.0),
            ));

            // WAV writing and judging happen OUTSIDE the timed loop, exactly
            // as the reference contract excludes them from synth_secs.
            let outdir = PathBuf::from(OUT_ROOT).join(&summary.name);
            let judge_dir = outdir.join("judge16k");
            std::fs::create_dir_all(&judge_dir)?;
            let mut judged = Vec::new();
            for (id, buf) in &audio {
                let wav = outdir.join(format!("{id}.wav"));
                ffai_media::save_wav(&wav, buf)?;
                let jwav = judge_dir.join(format!("{id}.wav"));
                ffai_media::save_wav(&jwav, &crate::resample::to_judge_format(buf, JUDGE_RATE))?;
                judged.push((id.clone(), jwav));
            }
            score_with_judge(judge, &judged, manifest, Mode::English, &mut summary)?;
        }
        Err(e) => summary.notes.push(first_error.unwrap_or_else(|| e.to_string())),
    }

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
    Ok(summary)
}

/// Load each generated WAV, accumulate its measured duration into
/// `summary.media_secs`, and write the judge-format (16 kHz mono) copy.
/// Returns `(clip_id, judge_wav_path)` for every utterance that produced
/// loadable audio; the rest become notes and count against `clips_ok`.
fn prepare_judge_wavs(
    batch: &TtsBatchResult,
    holdout: &[&ClipEntry],
    paths: &[PathBuf],
    outdir: &Path,
    summary: &mut RunSummary,
) -> Result<Vec<(String, PathBuf)>> {
    let judge_dir = outdir.join("judge16k");
    std::fs::create_dir_all(&judge_dir)?;
    let mut judged = Vec::new();
    let mut media_secs = 0.0f64;
    for (clip, path) in holdout.iter().zip(paths) {
        let Some(result) = batch.clip_for(path) else {
            summary.notes.push(format!("{}: no result returned", clip.id));
            continue;
        };
        match ffai_media::load_audio(&result.wav) {
            Ok(buf) => {
                media_secs += buf.duration_secs();
                let jwav = judge_dir.join(format!("{}.wav", clip.id));
                ffai_media::save_wav(&jwav, &crate::resample::to_judge_format(&buf, JUDGE_RATE))?;
                judged.push((clip.id.clone(), jwav));
                summary.clips_ok += 1;
            }
            Err(e) => summary.notes.push(format!("{}: generated wav unreadable: {e}", clip.id)),
        }
    }
    summary.media_secs = Some(media_secs);
    Ok(judged)
}

/// Transcribe every judge WAV with the frozen judge and score round-trip
/// WER/CER against the corpus text. Untimed: judging cost is the harness's,
/// never the implementation's.
fn score_with_judge(
    judge: &ReferenceSpec,
    judged: &[(String, PathBuf)],
    manifest: &Manifest,
    mode: Mode,
    summary: &mut RunSummary,
) -> Result<()> {
    if judged.is_empty() {
        return Ok(());
    }
    let judge_paths: Vec<PathBuf> = judged.iter().map(|(_, p)| p.clone()).collect();
    let transcripts = judge.run_batch(&judge_paths)?;
    let (mut wers, mut cers) = (Vec::new(), Vec::new());
    for (id, jwav) in judged {
        let clip = manifest.clips.iter().find(|c| &c.id == id);
        let (Some(clip), Some(hypothesis)) = (clip, transcripts.text_for(jwav)) else {
            summary.notes.push(format!("{id}: judge returned no transcript"));
            continue;
        };
        if let Some(truth) = manifest.ground_truth(clip)? {
            wers.push(wer_with(&truth, hypothesis, mode));
            cers.push(cer_with(&truth, hypothesis, mode));
        }
    }
    if !wers.is_empty() {
        summary.wer = Some(wers.iter().sum::<f64>() / wers.len() as f64);
        summary.cer = Some(cers.iter().sum::<f64>() / cers.len() as f64);
    }
    Ok(())
}

/// A reference's declared voice configuration under the shared comparison
/// key, so the quality gate matches implementations synthesizing with the
/// same voice + knobs (the `decode` key ASR uses, same machinery).
fn voice_config(config: Option<&str>) -> std::collections::BTreeMap<String, String> {
    let mut map = std::collections::BTreeMap::new();
    if let Some(c) = config {
        map.insert(crate::runner::DECODE_KEY.to_string(), c.to_string());
    }
    map
}

fn empty_summary(
    name: String,
    version: Option<String>,
    command: Option<String>,
    config: std::collections::BTreeMap<String, String>,
    clips_total: usize,
    notes: Vec<String>,
) -> RunSummary {
    RunSummary {
        name,
        version,
        command,
        config,
        wer: None,
        cer: None,
        rtf_warm: None,
        rtf_e2e: None,
        load_secs: None,
        wall_secs: None,
        media_secs: None,
        clips_ok: 0,
        clips_total,
        notes,
        peak_bytes: None,
        steady_bytes: None,
    }
}
