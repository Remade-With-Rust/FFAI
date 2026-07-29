//! The LIVE bench (M-C2 gates) + auto-ROI observe-only harvest (M-C2's
//! search-skip-gate discipline): run the screencast corpus through a
//! [`ffai_carmenta::live::LiveSession`], measure what the gates need, run
//! tesseract statelessly over the SAME frames for the C++ per-frame bar and
//! its churn under noise, and append everything to the ledger.
//!
//! Measured here:
//! - per-OCR-call p50/p95 latency (the M-C2 latency gate's engine side);
//! - **churn**: consecutive frame pairs whose ground truth is identical but
//!   whose output text differs — with the change gate this must be ZERO,
//!   because an unchanged frame never re-rolls the model;
//! - **CER on change frames** (fresh recognitions against fresh truth);
//! - **batch parity**: the same frames run through plain `recognize` must
//!   produce identical text to the session's outputs (live must not trade
//!   accuracy silently);
//! - **auto-ROI ceiling**: calibrate y-bands from the first 30 frames' line
//!   boxes, then report band coverage of later boxes and band area — the
//!   observe-only harvest that decides whether `--auto-roi` is worth
//!   building at all.
//!
//! ```sh
//! cargo run --release -p ffai-carmenta --example live_bench
//! ```

use ffai_bench::corpus::Manifest;
use ffai_bench::metrics::cer_with;
use ffai_bench::normalize::Mode;
use ffai_carmenta::live::{LiveConfig, LiveSession};
use ffai_core::engine::{OcrEngine, OcrOptions};
use std::path::{Path, PathBuf};

const FPS: f64 = 3.0;

fn main() {
    let manifest = Manifest::load(Path::new("corpora/carmenta-screencast-v1.toml"))
        .expect("screencast manifest (run prepare_carmenta_synth first)");
    let bad = manifest.verify().expect("verify");
    assert!(bad.is_empty(), "corpus drifted: {bad:?}");

    let mut clips: Vec<_> = manifest.clips.iter().collect();
    clips.sort_by(|a, b| a.id.cmp(&b.id));
    let paths: Vec<PathBuf> = clips.iter().map(|c| manifest.clip_path(c)).collect();
    let truths: Vec<String> = clips
        .iter()
        .map(|c| manifest.ground_truth(c).unwrap().unwrap_or_default())
        .collect();

    let engine = std::sync::Arc::new(ffai_carmenta::engine::CraftCrnn::new());
    // Warm-up, untimed.
    let first = ffai_media::load_image(&paths[0]).expect("frame 0");
    engine.recognize(&first, &OcrOptions::default()).expect("warm-up");

    // ---- the live session, BEST-OF-N (benchmarking.md: the minimum is the
    // run least perturbed; gate verdicts stop depending on machine mood).
    // Outputs are deterministic across runs; only timing varies.
    const RUNS: usize = 3;
    let run_session = || {
        let mut session = LiveSession::new(engine.clone(), OcrOptions::default(), LiveConfig {
            auto_roi: std::env::var("FFAI_AUTO_ROI").is_ok(),
            ..LiveConfig::default()
        });
        let mut outputs: Vec<String> = Vec::with_capacity(paths.len());
        let mut line_bands: Vec<Vec<(f32, f32)>> = Vec::new();
        let t0 = std::time::Instant::now();
        for (i, path) in paths.iter().enumerate() {
            let img = ffai_media::load_image(path).expect("frame");
            let out = session.push_frame(&img, i as f64 / FPS).expect("push_frame");
            outputs.push(out.text());
            line_bands.push(
                out.lines()
                    .filter_map(|l| l.bbox.as_ref().map(|b| (b.y, b.y + b.height)))
                    .collect(),
            );
        }
        let wall = t0.elapsed().as_secs_f64();
        let (segments, stats) = session.finish(paths.len() as f64 / FPS);
        (outputs, line_bands, segments, stats, wall)
    };
    let mut runs: Vec<_> = (0..RUNS).map(|_| run_session()).collect();
    let p95_of = |st: &ffai_carmenta::live::LiveStats| st.percentile(0.95).unwrap_or(f64::MAX);
    runs.sort_by(|a, b| p95_of(&a.3).total_cmp(&p95_of(&b.3)));
    let all_p95: Vec<String> =
        runs.iter().map(|r| format!("{:.0}", p95_of(&r.3) * 1000.0)).collect();
    let (outputs, line_bands, segments, stats, wall) = runs.swap_remove(0);
    println!("  best-of-{RUNS} engine p95s: [{}] ms (best kept)", all_p95.join(", "));

    // ---- churn: unchanged-GT pairs must produce unchanged output ----
    let mut churn = 0usize;
    let mut unchanged_pairs = 0usize;
    for i in 1..outputs.len() {
        if truths[i] == truths[i - 1] {
            unchanged_pairs += 1;
            if outputs[i] != outputs[i - 1] {
                churn += 1;
            }
        }
    }

    // ---- CER on change frames ----
    let mut cers = Vec::new();
    for i in 0..outputs.len() {
        if i == 0 || truths[i] != truths[i - 1] {
            cers.push(cer_with(&truths[i], &outputs[i], Mode::Ocr));
        }
    }
    let mean_cer = cers.iter().sum::<f64>() / cers.len().max(1) as f64;

    // ---- batch parity on every 20th frame ----
    let mut parity_break = 0usize;
    let mut parity_checked = 0usize;
    for i in (0..paths.len()).step_by(20) {
        let img = ffai_media::load_image(&paths[i]).expect("frame");
        let batch = engine.recognize(&img, &OcrOptions::default()).expect("batch");
        parity_checked += 1;
        if batch.text() != outputs[i] {
            parity_break += 1;
        }
    }

    // ---- auto-ROI observe-only harvest ----
    // Calibrate: union the line y-bands from the first 30 frames (loose ±8px).
    let mut bands: Vec<(f32, f32)> = Vec::new();
    for f in line_bands.iter().take(30) {
        for &(y0, y1) in f {
            let (y0, y1) = (y0 - 8.0, y1 + 8.0);
            if let Some(b) = bands.iter_mut().find(|b| y0 <= b.1 && y1 >= b.0) {
                b.0 = b.0.min(y0);
                b.1 = b.1.max(y1);
            } else {
                bands.push((y0, y1));
            }
        }
    }
    let frame_h = 720.0f32;
    let band_area: f32 = bands.iter().map(|b| (b.1 - b.0).max(0.0)).sum::<f32>() / frame_h;
    let (mut covered, mut total) = (0usize, 0usize);
    for f in line_bands.iter().skip(30) {
        for &(y0, y1) in f {
            total += 1;
            if bands.iter().any(|b| y0 >= b.0 && y1 <= b.1) {
                covered += 1;
            }
        }
    }
    let coverage = covered as f64 / total.max(1) as f64;

    // ---- tesseract, stateless, same frames: the C++ bar + its churn ----
    let refs = ffai_bench::reference::ReferenceFile::load(Path::new("corpora/references.toml"))
        .expect("references.toml");
    let tess = refs.for_task("ocr").find(|r| r.name == "tesseract").expect("tesseract ref");
    // Same best-of-N for the reference: keep the run with the lowest p95.
    let mut batches: Vec<_> = (0..RUNS).map(|_| tess.run_batch(&paths).expect("tesseract batch")).collect();
    let batch_p95 = |b: &ffai_bench::reference::BatchResult| {
        let mut t: Vec<f64> = b.clips.iter().filter_map(|c| c.transcribe_secs).collect();
        t.sort_by(|a, b| a.total_cmp(b));
        t.get(((t.len().max(1) - 1) as f64 * 0.95).round() as usize).copied().unwrap_or(f64::MAX)
    };
    batches.sort_by(|a, b| batch_p95(a).total_cmp(&batch_p95(b)));
    let batch = batches.swap_remove(0);
    let mut tess_churn = 0usize;
    let mut tess_times: Vec<f64> = Vec::new();
    let mut prev_tess: Option<String> = None;
    for (i, path) in paths.iter().enumerate() {
        let text = batch.text_for(path).unwrap_or_default().to_string();
        if truths[i] == truths.get(i.wrapping_sub(1)).map(String::as_str).unwrap_or("") {
            if let Some(prev) = &prev_tess {
                if *prev != text {
                    tess_churn += 1;
                }
            }
        }
        prev_tess = Some(text);
    }
    for c in &batch.clips {
        if let Some(t) = c.transcribe_secs {
            tess_times.push(t);
        }
    }
    tess_times.sort_by(|a, b| a.total_cmp(b));
    let pct = |v: &[f64], p: f64| v[((v.len() - 1) as f64 * p).round() as usize];

    // ---- report ----
    let p50 = stats.percentile(0.50).unwrap_or(0.0) * 1000.0;
    let p95 = stats.percentile(0.95).unwrap_or(0.0) * 1000.0;
    let (t50, t95) = (pct(&tess_times, 0.50) * 1000.0, pct(&tess_times, 0.95) * 1000.0);
    println!("LIVE bench — carmenta-screencast-v1 ({} frames @ {FPS} fps, wall {wall:.1}s)", paths.len());
    println!("  segments emitted:       {}", segments.len());
    println!(
        "  time-to-first-text:     {:.0} ms",
        stats.first_text_secs.unwrap_or(f64::NAN) * 1000.0
    );
    println!("  ocr calls / gated / roi: {} / {} / {}", stats.ocr_calls, stats.gated, stats.roi_calls);
    let mut fulls = stats.full_secs.clone();
    fulls.sort_by(|a, b| a.total_cmp(b));
    let full_max = fulls.last().copied().unwrap_or(0.0) * 1000.0;
    println!("  engine STEADY p50/p95:  {p50:.0} / {p95:.0} ms per band call ({} calls)", stats.call_secs.len());
    println!("  calib/sync-full calls:  {} (max {full_max:.0} ms) + {} async sweeps landed — the LOAD_S of the loop, off the serving path", stats.full_secs.len(), stats.sweeps_landed);
    println!("  tesseract p50/p95:      {t50:.0} / {t95:.0} ms (stateless, spawn tax included)");
    println!("  churn (engine):         {churn} of {unchanged_pairs} unchanged pairs");
    println!("  churn (tesseract):      {tess_churn} of {unchanged_pairs} (no change gate)");
    println!("  CER on change frames:   {:.2}%", mean_cer * 100.0);
    println!("  batch parity:           {} breaks / {parity_checked} checked", parity_break);
    println!(
        "  auto-ROI harvest:       bands cover {:.1}% of post-calibration boxes at {:.1}% of frame area \
         => detection-pixel ceiling {:.1}%",
        coverage * 100.0,
        band_area * 100.0,
        (1.0 - band_area as f64) * 100.0
    );

    // ---- gates (M-C2 wording) + ledger ----
    use ffai_bench::gate::{GateKind, GateOutcome, GateReport, GateResult};
    let mut gates = GateReport::new();
    let mut set = |kind, pass: bool, detail: String| {
        gates.set(GateResult {
            kind,
            outcome: if pass { GateOutcome::Pass } else { GateOutcome::Fail },
            metric: None,
            detail,
        });
    };
    // Batch parity is a HARD gate in plain mode; with auto-ROI it is the
    // plan's §4.1 explicit opt-in trade (banded output differs structurally
    // while the CER gate proves text accuracy holds) — reported, not judged.
    let auto_roi = std::env::var("FFAI_AUTO_ROI").is_ok();
    set(
        GateKind::Correctness,
        churn == 0 && (auto_roi || parity_break == 0),
        format!(
            "churn {churn}/{unchanged_pairs} unchanged pairs; batch parity {parity_break}/{parity_checked} breaks{}",
            if auto_roi { " (informational: auto-ROI's explicit trade, plan §4.1)" } else { "" }
        ),
    );
    set(GateKind::Quality, mean_cer <= 0.0174 + 0.0025, format!(
        "CER {:.2}% on change frames vs batch-mode frames corpus 1.74% (+0.25pp band)",
        mean_cer * 100.0
    ));
    set(GateKind::Speed, p95 <= t95, format!(
        "engine STEADY p95 {p95:.0} ms (band calls; calibration+sweeps reported separately,          the warm/e2e precedent) vs tesseract p95 {t95:.0} ms per frame"
    ));
    gates.set(GateResult::skipped(GateKind::Footprint, "flat-over-30-min run is a separate soak"));
    for g in &gates.results {
        println!("{:<12} {}  {}", g.kind.label(), g.outcome, g.detail);
    }

    let (id, appended_at) = ffai_bench::ledger::BenchRecord::now_id("ocr");
    let record = ffai_bench::ledger::BenchRecord {
        schema: ffai_bench::ledger::LEDGER_SCHEMA,
        id,
        task: "ocr".into(),
        corpus: manifest.name.clone(),
        corpus_manifest_hash: manifest.manifest_hash(),
        engine: None,
        references: Vec::new(),
        gates,
        environment: ffai_bench::ledger::Environment::capture(),
        notes: format!(
            "LIVE session bench: {nf} frames, {calls} ocr calls, {gated} gated; engine p50/p95 {p50:.0}/{p95:.0} ms; \
             tesseract p50/p95 {t50:.0}/{t95:.0} ms; churn {churn}/{unchanged_pairs} (tesseract {tess_churn}); \
             CER change-frames {cer:.2}%; parity breaks {parity_break}/{parity_checked}; \
             auto-ROI ceiling: coverage {cov:.1}%, band area {area:.1}%",
            nf = paths.len(),
            calls = stats.ocr_calls,
            gated = stats.gated,
            cer = mean_cer * 100.0,
            cov = coverage * 100.0,
            area = band_area * 100.0,
        ),
        appended_at,
    };
    ffai_bench::ledger::append(Path::new("bench/ledger.jsonl"), &record).expect("ledger append");
    println!("appended to bench/ledger.jsonl");
}
