//! Interleaved paired A/B: flat fused-gate flow vs candle-op flow, both arms
//! sampling the same machine drift (the §6.16 discipline — sequential
//! comparisons under ±15 % load noise are worthless).
//!
//! ```text
//! cargo run --release -p ffai-mercury --example ab_flow
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    // Pre-compute z_p inputs for 10 sentences once.
    let mut inputs = Vec::new();
    for clip in manifest.holdout().take(10) {
        let text = std::fs::read_to_string(manifest.clip_path(clip))?;
        let ids = vits.id_map.sentence_to_ids(&phonemizer.phonemize(text.trim())?).0;
        let (m_p, logs_p, hidden) = vits.text_encoder(&ids)?;
        let mut rng = GaussRng::new(0);
        let w = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;
        let (m_e, _) = vits.expand_prior(&m_p, &logs_p, &w)?;
        inputs.push(m_e);
    }

    // Warm both arms, then alternate rounds.
    let arm = |candle: bool| -> f64 {
        // SAFETY: this example is single-threaded at the point of mutation
        // (rayon workers only read env inside vits calls, after the flag is
        // set); edition-2024 marks env mutation unsafe for exactly the
        // multi-threaded case this avoids.
        unsafe {
            if candle {
                std::env::set_var("FFAI_CANDLE_FLOW", "1");
            } else {
                std::env::remove_var("FFAI_CANDLE_FLOW");
            }
        }
        let t0 = Instant::now();
        for z_p in &inputs {
            vits.flow_reverse(z_p).unwrap();
        }
        t0.elapsed().as_secs_f64() * 1000.0
    };
    arm(false);
    arm(true);

    let rounds = 15;
    let (mut flat_wins, mut flat_total, mut candle_total) = (0usize, 0f64, 0f64);
    for _ in 0..rounds {
        let f = arm(false);
        let c = arm(true);
        if f < c {
            flat_wins += 1;
        }
        flat_total += f;
        candle_total += c;
    }
    println!(
        "flat {:.1} ms vs candle {:.1} ms per round (10 sentences) — flat wins {flat_wins}/{rounds}, ratio {:.2}x",
        flat_total / rounds as f64,
        candle_total / rounds as f64,
        candle_total / flat_total
    );
    Ok(())
}
