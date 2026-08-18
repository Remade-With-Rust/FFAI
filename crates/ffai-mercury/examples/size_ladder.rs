//! The model-size accuracy ladder: WER/CER and throughput per Whisper size,
//! on the same pinned holdout, same decode settings.
//!
//! Route B of the quality question. tiny.en is line-ball with whisper.cpp
//! because it *is* the same model; the only way to make an accuracy claim the
//! current numbers cannot support is to run a bigger one. This measures what
//! each size actually buys, and what it costs in throughput — the pair, not
//! the accuracy alone, because "more accurate if you accept 8x slower" is a
//! different product decision than "more accurate".
//!
//! First run per size downloads weights (small.en ~1 GB, medium.en ~3 GB).
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example size_ladder -- \
//!     corpora/librispeech-test-clean-v2.toml whisper-tiny-en,whisper-small-en
//! ```
use std::path::PathBuf;
use std::time::Instant;

use ffai_bench::corpus::Manifest;
use ffai_bench::metrics::{cer, wer};
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;
use ffai_mercury::asr::text_decoder::Precision;

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpora/librispeech-test-clean-v2.toml".to_string());
    let models: Vec<String> = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "whisper-tiny-en,whisper-base-en,whisper-small-en".to_string())
        .split(',')
        .map(|s| s.to_string())
        .collect();
    let limit: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let manifest = Manifest::load(&PathBuf::from(&corpus)).expect("corpus loads");
    let opts = AsrOptions::default();

    println!("corpus: {corpus}");
    println!(
        "{:<22} {:>8} {:>8} {:>9} {:>10}",
        "MODEL", "WER%", "CER%", "xRT", "CLIPS"
    );

    for model in &models {
        let engine = WhisperCandle::with_model("models", model.as_str(), Precision::F32);

        // Warm-up: weight load + any first-call calibration, outside the
        // timed region — and the download, on a cold cache.
        let Some(first) = manifest.clips.first() else {
            continue;
        };
        let Ok(audio) = ffai_media::load_audio(&manifest.clip_path(first)) else {
            continue;
        };
        if let Err(e) = engine.transcribe(&audio, &opts) {
            println!("{model:<22}  load/transcribe failed: {e}");
            continue;
        }

        let (mut wnum, mut wden, mut cnum, mut cden) = (0.0, 0.0, 0.0, 0.0);
        let (mut secs, mut audio_secs, mut n) = (0.0f64, 0.0f64, 0usize);
        for clip in manifest.clips.iter().take(limit) {
            let Ok(audio) = ffai_media::load_audio(&manifest.clip_path(clip)) else {
                continue;
            };
            let Ok(Some(truth)) = manifest.ground_truth(clip) else {
                continue;
            };
            audio_secs += audio.duration_secs();
            let t = Instant::now();
            let Ok(out) = engine.transcribe(&audio, &opts) else {
                continue;
            };
            secs += t.elapsed().as_secs_f64();
            let text = out.text();
            // Length-weighted, matching how the bench harness aggregates:
            // a corpus WER is total errors over total reference words, not
            // the mean of per-clip rates.
            let words = truth.split_whitespace().count() as f64;
            let chars = truth.chars().count() as f64;
            wnum += wer(&truth, &text) * words;
            wden += words;
            cnum += cer(&truth, &text) * chars;
            cden += chars;
            n += 1;
            if n % 25 == 0 {
                eprintln!("  {model} {n} clips ...");
            }
        }
        if wden > 0.0 {
            println!(
                "{model:<22} {:>8.2} {:>8.2} {:>9.1} {:>10}",
                wnum / wden * 100.0,
                cnum / cden * 100.0,
                audio_secs / secs,
                n
            );
        }
    }
}
