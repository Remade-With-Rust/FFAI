//! M-C2 footprint soak: cycle the screencast corpus through a LiveSession
//! for N minutes (default 30) sampling resident memory; the gate is FLAT —
//! last-5-minute median within 10% of first-5-minute median.
use ffai_carmenta::live::{LiveConfig, LiveSession};
use ffai_core::engine::{OcrEngine, OcrOptions};

fn main() {
    let mins: u64 = std::env::var("SOAK_MINS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let manifest = ffai_bench::corpus::Manifest::load(std::path::Path::new("corpora/carmenta-screencast-v1.toml")).unwrap();
    let mut clips: Vec<_> = manifest.clips.iter().collect();
    clips.sort_by(|a, b| a.id.cmp(&b.id));
    let frames: Vec<_> = clips.iter().map(|c| ffai_media::load_image(&manifest.clip_path(c)).unwrap()).collect();

    let engine = ffai_carmenta::engine::CraftCrnn::new();
    engine.recognize(&frames[0], &OcrOptions::default()).unwrap();
    let mut session = LiveSession::new(&engine, OcrOptions::default(), LiveConfig { auto_roi: true, ..Default::default() });

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
    println!("verdict: {}", if l <= f * 1.10 { "FLAT (PASS)" } else { "GROWING (FAIL)" });
}
