//! Six-whys instrument for the flow (docs/mercury-tts-mission.md §6.13):
//! run the FLAT flow with `FFAI_PROFILE=1` internal timers over real z_p
//! inputs, best-of-3, so the stage's cost decomposes into named ops.
//!
//! ```text
//! FFAI_PROFILE=1 cargo run --release -p ffai-mercury --example flow_whys
//! ```

use std::path::{Path, PathBuf};

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe { std::env::set_var("FFAI_FLAT_FLOW", "1") };
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
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
    // Warm, then 3 passes; the [flow] lines are per flow_reverse call.
    for z_p in &inputs {
        vits.flow_reverse(z_p)?;
    }
    for _ in 0..2 {
        for z_p in &inputs {
            vits.flow_reverse(z_p)?;
        }
    }
    Ok(())
}
