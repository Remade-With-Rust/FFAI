//! The consumer's-eye test: run Mercury from a directory that has no
//! `models/` in it, the way `cargo add ffai-mercury` leaves you.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example foreign_cwd
//! ```
//!
//! This is the failure the embedded manifests exist to fix, so it is worth an
//! instrument rather than an assurance. The example changes its own working
//! directory to a temp dir before touching any engine; if a relative
//! `models/` path survives anywhere in the load paths, this is where it
//! surfaces.
//!
//! Weights still come from the shared cache — that is `ffai-models`' job and
//! is absolute. What must not be required is a manifest *directory*.

use ffai_core::engine::{AsrEngine, AsrOptions, TtsEngine, TtsOptions};
use ffai_mercury::asr::WhisperCandle;
use ffai_mercury::tts::PiperCandle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scratch = std::env::temp_dir().join("ffai-foreign-cwd");
    std::fs::create_dir_all(&scratch)?;
    std::env::set_current_dir(&scratch)?;
    println!("cwd: {}", std::env::current_dir()?.display());
    println!("models/ here: {}\n", scratch.join("models").exists());

    // ---- manifests resolve with no directory at all ----
    for name in [
        "whisper-tiny-en",
        "wav2vec2-base-960h",
        "ecapa-tdnn-voxceleb",
        "cmudict",
    ] {
        let m = ffai_mercury::manifests::resolve(None, name)?;
        println!("manifest {name:<24} ok  (licence: {})", m.license);
    }

    // ---- ASR ----
    let audio = ffai_core::types::AudioBuffer {
        samples: (0..16_000 * 2)
            .map(|i| ((i as f32) * 0.01).sin() * 0.05)
            .collect(),
        sample_rate: 16_000,
        channels: 1,
    };
    let asr = WhisperCandle::new();
    match asr.transcribe(&audio, &AsrOptions::default()) {
        Ok(t) => println!(
            "\nASR ok — {} segment(s), text {:?}",
            t.segments.len(),
            t.text().trim()
        ),
        Err(e) => {
            println!("\nASR FAILED: {e}");
            std::process::exit(1);
        }
    }

    // ---- TTS ----
    let tts = PiperCandle::new();
    match tts.synthesize(
        "The birch canoe slid on the smooth planks.",
        &TtsOptions::default(),
    ) {
        Ok(a) => println!(
            "TTS ok — {:.2} s of audio at {} Hz",
            a.duration_secs(),
            a.sample_rate
        ),
        Err(e) => {
            println!("TTS FAILED: {e}");
            std::process::exit(1);
        }
    }

    println!("\nboth engines ran with no models/ directory present");
    Ok(())
}
