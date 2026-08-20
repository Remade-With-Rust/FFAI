//! Price the diarization hop: DER against forwards.
//!
//! Embedding is ~100 % of diarization's cost and the forward count for a
//! region is `span / hop`, so the hop trades quality for compute *linearly* —
//! it is the only knob that does. 0.75 s is inherited convention (half the
//! window) and has never been measured on this corpus, which makes it a guess
//! sitting in the hot path.
//!
//! Reports both columns because either alone is useless: a hop that halves
//! the forwards at +0.1 pp DER is a clear win, and one that halves them at
//! +4 pp is a clear loss. The number that decides it is the pair.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example hop_sweep
//! ```
use std::path::PathBuf;
use std::time::Instant;

use ffai_bench::corpus::Manifest;
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;
use ffai_mercury::asr::diarizer::cache_stats;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or("corpora/librispeech-diarization.toml".to_string());
    let hops: Vec<f64> = std::env::args()
        .nth(2)
        .unwrap_or("0.75,1.0,1.5,2.0,3.0".into())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    let manifest = Manifest::load(&PathBuf::from(&corpus))?;
    let engine = WhisperCandle::new();

    // The cache would make a later hop look fast on windows an earlier hop
    // already embedded. Off: this measures the GEOMETRY, not the cache.
    unsafe { std::env::set_var("FFAI_DIARIZE_CACHE", "off") };

    println!("corpus {corpus} · cache OFF (measuring geometry, not reuse)\n");
    println!(
        "{:>6} {:>10} {:>12} {:>10}",
        "hop s", "forwards", "diarize s", "vs 0.75"
    );
    let mut base = None;
    for hop in &hops {
        unsafe { std::env::set_var("FFAI_DIARIZE_HOP", hop.to_string()) };
        let (h0, m0) = cache_stats();
        let before = h0 + m0;
        let opts = AsrOptions {
            diarize: true,
            ..AsrOptions::default()
        };

        let t = Instant::now();
        for clip in &manifest.clips {
            let Ok(audio) = ffai_media::load_audio(&manifest.clip_path(clip)) else {
                continue;
            };
            let _ = engine.transcribe(&audio, &opts);
        }
        let secs = t.elapsed().as_secs_f64();
        let (h1, m1) = cache_stats();
        let forwards = (h1 + m1) - before;
        let b = *base.get_or_insert(forwards as f64);
        println!(
            "{hop:>6.2} {forwards:>10} {secs:>12.1} {:>9.2}x",
            b / forwards as f64
        );
    }
    println!(
        "\nDER per hop comes from diarize_gate under the same FFAI_DIARIZE_HOP;\n\
         forwards here are the cost side of that trade."
    );
    Ok(())
}
