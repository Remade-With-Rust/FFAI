//! Where does the 608 MiB go? Peak resident memory, phase by phase.
//!
//! The footprint gate now measures honestly and we fail it 3.03x (608 MiB
//! against whisper.cpp's 201 MiB). Before cutting anything, find out what
//! holds it — the memory version of "decompose the residue until every line is
//! named", because the obvious suspect has already turned out to be wrong once:
//! toggling the f16 decoder projections moved peak only 543 -> 527 MiB.
//!
//! Peak is a high-water mark and never falls, so each phase reports the peak
//! AFTER it, and the interesting quantity is the RISE between phases.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example mem_anatomy
//! ```

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

fn peak_mib() -> f64 {
    ffai_bench::footprint::peak_self()
        .map(|p| p.mib())
        .unwrap_or(f64::NAN)
}

fn current_mib() -> f64 {
    ffai_bench::footprint::current_self()
        .map(|p| p.mib())
        .unwrap_or(f64::NAN)
}

fn phase(label: &str, prev: &mut f64) {
    let now = peak_mib();
    println!(
        "{label:<44} peak {now:8.1}  held {:8.1} MiB   (+{:.1} peak)",
        current_mib(),
        now - *prev
    );
    *prev = now;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut prev = 0.0;
    println!("peak resident memory, phase by phase\n");
    phase("process start", &mut prev);

    // Audio first, so its cost is separated from the model's.
    let dir = std::path::Path::new("corpora/clips/librispeech-test-clean/audio");
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    paths.sort();
    paths.truncate(16);
    let clips: Vec<_> = paths
        .iter()
        .map(|p| ffai_media::load_audio(p))
        .collect::<Result<Vec<_>, _>>()?;
    let audio_mib: f64 = clips
        .iter()
        .map(|a| (a.samples.len() * 4) as f64)
        .sum::<f64>()
        / (1024.0 * 1024.0);
    phase(
        &format!(
            "{} clips decoded ({audio_mib:.1} MiB of samples)",
            clips.len()
        ),
        &mut prev,
    );

    let engine = WhisperCandle::new();
    phase("engine constructed (no weights yet)", &mut prev);

    // First transcribe loads weights and runs the precision calibration.
    engine.transcribe(&clips[0], &AsrOptions::default())?;
    phase("first transcribe (weights + calibration)", &mut prev);

    engine.transcribe(&clips[0], &AsrOptions::default())?;
    phase("second transcribe (same clip, warm)", &mut prev);

    for c in clips.iter().skip(1) {
        engine.transcribe(c, &AsrOptions::default())?;
    }
    phase("remaining clips", &mut prev);

    println!(
        "\nThe f32 safetensors on disk are 144 MB. A weight set held ONCE at f32\n\
         should show roughly that at the load phase; materially more means copies.\n\
         whisper.cpp's whole tree peaks at 201 MiB doing the same work."
    );
    Ok(())
}
