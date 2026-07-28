//! The three footprint claims, measured instead of argued.
//!
//! The footprint gate now exists and we fail it (608 MiB against whisper.cpp's
//! 201 MiB). But three specific memory claims were being carried on reasoning
//! rather than numbers, and two of them were about a variant nobody had
//! actually put on a scale:
//!
//! 1. **the int8 variant's ~4x smaller weights** — §6.5 kept `q8_0` on the
//!    argument that it "buys memory, not speed". Never measured.
//! 2. **the f16 cross-attention cache halving** (18.4 -> 9.2 MB/token).
//! 3. **one binary, no runtime** — measured elsewhere: 19 MB against 69 MB of
//!    exe + DLLs, 49 MB of which is libopenblas.
//!
//! `mem_anatomy` showed the whole cost arrives at load (+525 MiB) and that
//! **386 MiB of it is HELD, not transient** — so `held` is the quantity these
//! claims live or die on, and peak is reported beside it only for context.
//!
//! Each arm is a fresh process, because peak never falls and a variant
//! measured after another variant in the same process inherits its high-water
//! mark. Run without arguments to see the list.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example mem_claims -- f32
//! cargo run --release -p ffai-mercury --example mem_claims -- q8_0
//! cargo run --release -p ffai-mercury --example mem_claims -- kv_f32
//! ```

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::text_decoder::Precision;
use ffai_mercury::asr::{knobs, WhisperCandle};

fn held_mib() -> f64 {
    ffai_bench::footprint::current_self().map(|p| p.mib()).unwrap_or(f64::NAN)
}
fn peak_mib() -> f64 {
    ffai_bench::footprint::peak_self().map(|p| p.mib()).unwrap_or(f64::NAN)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = std::env::args().nth(1).unwrap_or_else(|| "f32".into());

    let engine = match arm.as_str() {
        // The shipped default: f32 weights, f16 single-row projections, int8
        // vocabulary projection, f16 cross-attention cache.
        "f32" => WhisperCandle::with_model("models", "whisper-tiny-en", Precision::F32),
        // The claim under test: q8_0 quantizes the decoder at LOAD from the
        // same f32 safetensors.
        "q8_0" => WhisperCandle::with_model("models", "whisper-tiny-en", Precision::Q8_0),
        "q4k" => WhisperCandle::with_model("models", "whisper-tiny-en", Precision::Q4K),
        // The f16 cross-attention cache, forced off.
        "kv_f32" => {
            knobs::KV_F16_DISABLED.set(true);
            WhisperCandle::with_model("models", "whisper-tiny-en", Precision::F32)
        }
        // Every quantized QLinear also retains `full: Linear` — the f32
        // original — as a multi-row fallback. This arm turns the f16 single-row
        // path off so the duplication is not paid.
        "no_f16_dup" => {
            knobs::DEC_F16_DISABLED.set(true);
            WhisperCandle::with_model("models", "whisper-tiny-en", Precision::F32)
        }
        other => {
            eprintln!("unknown arm `{other}`; try: f32 q8_0 q4k kv_f32 no_f16_dup");
            std::process::exit(2);
        }
    };

    let dir = std::path::Path::new("corpora/clips/librispeech-test-clean/audio");
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    paths.sort();
    paths.truncate(4);
    let clips: Vec<_> =
        paths.iter().map(|p| ffai_media::load_audio(p)).collect::<Result<Vec<_>, _>>()?;

    let before = held_mib();
    for c in &clips {
        engine.transcribe(c, &AsrOptions::default())?;
    }
    println!(
        "{arm:<12} held {:8.1} MiB   peak {:8.1} MiB   (audio+harness before load: {before:.1})",
        held_mib(),
        peak_mib()
    );
    Ok(())
}
