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

use std::path::PathBuf;

use ffai_core::engine::AsrOptions;
use ffai_core::error::{Error, Result};
use ffai_core::registry::EngineRegistry;

use crate::corpus::{ClipEntry, Manifest};
use crate::gate::{GateKind, GateOutcome, GateReport, GateResult};
use crate::ledger::{append, BenchRecord, Environment, RunSummary, LEDGER_SCHEMA};
use crate::metrics::{cer, wer};
use crate::reference::ReferenceSpec;
use crate::speed::{best_of_n, real_time_factor};

/// Parity band for the quality gate: engine WER may exceed the best
/// reference's WER by at most this factor (1.05 = within 5% relative).
pub const QUALITY_PARITY_BAND: f64 = 1.05;

pub struct BenchConfig {
    /// Engine to bench; `None` = registry default. Set `skip_engine` to
    /// baseline references only.
    pub engine: Option<String>,
    pub skip_engine: bool,
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
        return Err(Error::Other("corpus has no holdout clips — nothing to measure".into()));
    }
    let paths: Vec<PathBuf> = holdout.iter().map(|c| manifest.clip_path(c)).collect();
    let media_secs: f64 = paths
        .iter()
        .filter_map(|p| ffai_media::load_audio(p).ok())
        .map(|a| a.duration_secs())
        .sum();

    let mut references = Vec::new();
    for spec in &cfg.references {
        eprintln!("running reference `{}` over {} clips ...", spec.name, holdout.len());
        match run_reference(spec, &manifest, &holdout, &paths, media_secs, cfg.runs) {
            Ok(summary) => references.push(summary),
            Err(e) => {
                eprintln!("  reference `{}` failed: {e}", spec.name);
                references.push(RunSummary {
                    name: spec.name.clone(),
                    version: spec.version(),
        command: Some(spec.command_line()),
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
                });
            }
        }
    }

    let engine_summary = if cfg.skip_engine {
        None
    } else {
        Some(run_engine(reg, cfg.engine.as_deref(), &manifest, &holdout, media_secs, cfg.runs)?)
    };

    let mut gates = GateReport::new();
    fill_gates(&mut gates, engine_summary.as_ref(), &references);

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

fn run_reference(
    spec: &ReferenceSpec,
    manifest: &Manifest,
    holdout: &[&ClipEntry],
    paths: &[PathBuf],
    media_secs: f64,
    runs: usize,
) -> Result<RunSummary> {
    let mut summary = RunSummary {
        name: spec.name.clone(),
        version: spec.version(),
        command: Some(spec.command_line()),
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
                    wers.push(wer(&truth, hypothesis));
                    cers.push(cer(&truth, hypothesis));
                }
            }
            None => summary.notes.push(format!("{}: no result returned", clip.id)),
        }
    }

    summary.wer = mean(&wers);
    summary.cer = mean(&cers);
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
    let mut summary = RunSummary {
        name: asr.info().name,
        version: Some(env!("CARGO_PKG_VERSION").to_string()),
        command: None,
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
        if let Err(e) = asr.transcribe(audio, &AsrOptions::default()) {
            summary.notes.push(format!("warm-up failed: {e}"));
        }
        summary.load_secs = Some(t0.elapsed().as_secs_f64());
    }

    let (mut wers, mut cers) = (Vec::new(), Vec::new());
    let mut texts: Vec<(String, String)> = Vec::new();
    let mut first_error = None;

    let stats = best_of_n(runs, || {
        texts.clear();
        for (clip, audio) in &decoded {
            match asr.transcribe(audio, &AsrOptions::default()) {
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
                if let Some(clip) = clip {
                    if let Some(truth) = manifest.ground_truth(clip)? {
                        wers.push(wer(&truth, text));
                        cers.push(cer(&truth, text));
                    }
                }
            }
            summary.wer = mean(&wers);
            summary.cer = mean(&cers);
        }
        Err(e) => summary.notes.push(first_error.unwrap_or_else(|| e.to_string())),
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
    let decoded_bytes: u64 =
        decoded.iter().map(|(_, a)| (a.samples.len() * size_of::<f32>()) as u64).sum();
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

fn fill_gates(gates: &mut GateReport, engine: Option<&RunSummary>, references: &[RunSummary]) {
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
            detail: format!("{}/{} holdout clips processed", eng.clips_ok, eng.clips_total),
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
                eng.notes.first().map(String::as_str).unwrap_or("-")
            ),
        }
    });

    let best_ref = references
        .iter()
        .filter_map(|r| r.wer.map(|w| (r.name.as_str(), w)))
        .min_by(|a, b| a.1.total_cmp(&b.1));
    gates.set(match (eng.wer, best_ref) {
        (Some(ew), Some((rname, rw))) => GateResult {
            kind: GateKind::Quality,
            outcome: if ew <= rw * QUALITY_PARITY_BAND { GateOutcome::Pass } else { GateOutcome::Fail },
            metric: Some(ew),
            detail: format!(
                "engine WER {:.2}% vs best reference {rname} {:.2}% (band +{:.0}% relative)",
                ew * 100.0,
                rw * 100.0,
                QUALITY_PARITY_BAND * 100.0 - 100.0
            ),
        },
        (Some(ew), None) => GateResult::skipped(
            GateKind::Quality,
            format!("engine WER {:.2}% but no reference ran to compare against", ew * 100.0),
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
            outcome: if er >= rr { GateOutcome::Pass } else { GateOutcome::Fail },
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
    let leanest = references
        .iter()
        .filter_map(|r| r.peak_bytes.map(|b| (r.name.as_str(), b)))
        .min_by_key(|(_, b)| *b);
    const MIB: f64 = 1024.0 * 1024.0;
    gates.set(match (eng.peak_bytes, leanest) {
        (Some(ep), Some((rname, rp))) => GateResult {
            kind: GateKind::Footprint,
            outcome: if ep <= rp { GateOutcome::Pass } else { GateOutcome::Fail },
            metric: Some(ep as f64 / MIB),
            detail: format!(
                "engine peak {:.0} MiB vs leanest reference {rname} {:.0} MiB ({:.2}x)",
                ep as f64 / MIB,
                rp as f64 / MIB,
                ep as f64 / rp as f64
            ),
        },
        (Some(ep), None) => GateResult::skipped(
            GateKind::Footprint,
            format!(
                "engine peak {:.0} MiB but no reference reported one to compare against",
                ep as f64 / MIB
            ),
        ),
        _ => GateResult::skipped(
            GateKind::Footprint,
            if crate::footprint::supported() {
                "peak memory was not captured for this run".to_string()
            } else {
                format!(
                    "peak-memory measurement is not implemented on {} — \
                     see crates/ffai-bench/src/footprint.rs",
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
pub fn render(record: &BenchRecord) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "bench {} · corpus {} ({}) · {:.1}s audio\n\n",
        record.id,
        record.corpus,
        &record.corpus_manifest_hash[..12.min(record.corpus_manifest_hash.len())],
        record
            .engine
            .as_ref()
            .and_then(|e| e.media_secs)
            .or_else(|| record.references.first().and_then(|r| r.media_secs))
            .unwrap_or(0.0),
    ));
    out.push_str(&format!(
        "{:<24} {:>7} {:>7} {:>10} {:>10} {:>8} {:>7}\n",
        "IMPLEMENTATION", "WER%", "CER%", "xRT_WARM", "xRT_E2E", "LOAD_S", "CLIPS"
    ));
    let mut row = |s: &RunSummary, marker: &str| {
        out.push_str(&format!(
            "{:<24} {:>7} {:>7} {:>10} {:>10} {:>8} {:>9} {:>7}\n",
            format!("{marker}{}", s.name),
            s.wer.map(|w| format!("{:.2}", w * 100.0)).unwrap_or_else(|| "-".into()),
            s.cer.map(|c| format!("{:.2}", c * 100.0)).unwrap_or_else(|| "-".into()),
            s.rtf_warm.map(|r| format!("{r:.1}x")).unwrap_or_else(|| "-".into()),
            s.rtf_e2e.map(|r| format!("{r:.1}x")).unwrap_or_else(|| "-".into()),
            s.load_secs.map(|l| format!("{l:.2}")).unwrap_or_else(|| "-".into()),
            s.peak_bytes
                .map(|b| format!("{:.0}", b as f64 / (1024.0 * 1024.0)))
                .unwrap_or_else(|| "-".into()),
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
        out.push_str(&format!("{:<12} {}  {}\n", g.kind.label(), g.outcome, g.detail));
    }
    out.push_str(&format!(
        "\nverdict: {}\n",
        if record.gates.all_passed() { "ALL GATES PASS — claimable" } else { "not claimable yet" }
    ));
    out
}
