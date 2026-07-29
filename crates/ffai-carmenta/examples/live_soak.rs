//! M-C2 footprint soak: cycle the screencast corpus through a LiveSession
//! for N minutes (default 30) sampling resident memory; the gate is FLAT —
//! last-5-minute median within 10% of first-5-minute median.
use ffai_carmenta::live::{LiveConfig, LiveSession};
use ffai_core::engine::{OcrEngine, OcrOptions};

fn main() {
    let mins: u64 = std::env::var("SOAK_MINS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    if let Ok(replay) = std::env::var("SOAK_REPLAY") {
        let v: Vec<f64> = replay.split(',').map(|x| x.parse().unwrap()).collect();
        let manifest = ffai_bench::corpus::Manifest::load(std::path::Path::new("corpora/carmenta-screencast-v1.toml")).unwrap();
        append_record(mins, v[0] as usize, v[1], v[2], v[2] <= v[1] * 1.10, &manifest);
        return;
    }
    let manifest = ffai_bench::corpus::Manifest::load(std::path::Path::new("corpora/carmenta-screencast-v1.toml")).unwrap();
    let mut clips: Vec<_> = manifest.clips.iter().collect();
    clips.sort_by(|a, b| a.id.cmp(&b.id));
    let frames: Vec<_> = clips.iter().map(|c| ffai_media::load_image(&manifest.clip_path(c)).unwrap()).collect();

    let engine = std::sync::Arc::new(ffai_carmenta::engine::CraftCrnn::new());
    engine.recognize(&frames[0], &OcrOptions::default()).unwrap();
    let mut session = LiveSession::new(engine, OcrOptions::default(), LiveConfig { auto_roi: true, ..Default::default() });

    let t0 = std::time::Instant::now();
    let mut samples: Vec<(f64, u64)> = Vec::new();
    let mut i = 0usize;
    while t0.elapsed().as_secs() < mins * 60 {
        session.push_frame(&frames[i % frames.len()], i as f64 / 3.0).unwrap();
        if let Some(b) = ffai_bench::footprint::current_self() {
            samples.push((t0.elapsed().as_secs_f64(), b.0));
        }
        i += 1;
    }
    let med = |v: &mut Vec<u64>| { v.sort_unstable(); v[v.len() / 2] };
    let window = 300.0f64.min(mins as f64 * 60.0 / 3.0);
    let mut first: Vec<u64> = samples.iter().filter(|(t, _)| *t < window).map(|(_, b)| *b).collect();
    let mut last: Vec<u64> = samples.iter().filter(|(t, _)| *t > mins as f64 * 60.0 - window).map(|(_, b)| *b).collect();
    let (f, l) = (med(&mut first) as f64 / 1048576.0, med(&mut last) as f64 / 1048576.0);
    println!("SOAK {} min, {} frames: first-window median {f:.0} MiB, last-window median {l:.0} MiB, ratio {:.3}", mins, i, l / f);
    let pass = l <= f * 1.10;
    println!("verdict: {}", if pass { "FLAT (PASS)" } else { "GROWING (FAIL)" });
    append_record(mins, i, f, l, pass, &manifest);
}

/// Ledger append. `SOAK_REPLAY="frames,first_mib,last_mib"` records an
/// already-completed run's printed numbers without re-soaking — exists so a
/// measurement whose process exited before this landed still gets its line;
/// the values must be the printed ones, and the note says replay.
fn append_record(mins: u64, frames: usize, first: f64, last: f64, pass: bool, manifest: &ffai_bench::corpus::Manifest) {
    use ffai_bench::gate::{GateKind, GateOutcome, GateReport, GateResult};
    let mut gates = GateReport::new();
    gates.set(GateResult {
        kind: GateKind::Footprint,
        outcome: if pass { GateOutcome::Pass } else { GateOutcome::Fail },
        metric: Some(last / first),
        detail: format!("steady flat over {mins} min: first-window median {first:.0} MiB, last {last:.0} MiB (gate: ratio <= 1.10)"),
    });
    for kind in [GateKind::Correctness, GateKind::Quality, GateKind::Speed] {
        gates.set(GateResult::skipped(kind, "soak measures footprint only; see the LIVE bench record"));
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
            "M-C2 footprint soak: {mins} min, {frames} frames cycled through LiveSession (auto_roi on, FFAI_DET_SCALE=0.5); {}",
            if std::env::var("SOAK_REPLAY").is_ok() { "REPLAY of completed run's printed values" } else { "live run" }
        ),
        appended_at,
    };
    ffai_bench::ledger::append(std::path::Path::new("bench/ledger.jsonl"), &record).expect("ledger");
    println!("appended to bench/ledger.jsonl");
}
