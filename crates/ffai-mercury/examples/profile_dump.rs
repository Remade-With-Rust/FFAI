//! Throwaway: transcribe N clips and dump the full stage/op profile.
//!
//! ```sh
//! FFAI_PROFILE=1 cargo run --release -p ffai-mercury --example profile_dump -- \
//!     --dir corpora/clips/librispeech-test-clean/audio --clips 10 --vad on
//! ```

use std::path::PathBuf;

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::profile;
use ffai_mercury::asr::WhisperCandle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let count: usize = arg("--clips", "10").parse()?;
    let dir = PathBuf::from(arg("--dir", "corpora/clips/librispeech-test-clean/audio"));
    let vad = arg("--vad", "on") == "on";

    if !profile::is_enabled() {
        return Err("set FFAI_PROFILE=1".into());
    }

    let mut clips: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    clips.sort();
    clips.truncate(count);

    let opts = AsrOptions { vad, ..AsrOptions::default() };
    let engine = WhisperCandle::new();

    // Warm-up outside the measured region: model load + precision calibration.
    let first = ffai_media::load_audio(&clips[0])?;
    engine.transcribe(&first, &opts)?;
    profile::reset();

    let t0 = std::time::Instant::now();
    let mut media = 0.0f64;
    for clip in &clips {
        let audio = ffai_media::load_audio(clip)?;
        media += audio.duration_secs();
        engine.transcribe(&audio, &opts)?;
    }
    let wall = t0.elapsed().as_secs_f64();

    println!(
        "{} clips, {media:.1}s audio, vad={vad}, wall {wall:.2}s ({:.1}x RT)",
        clips.len(),
        media / wall
    );
    println!("{}", profile::profile().report());
    Ok(())
}
