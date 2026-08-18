//! Spend the adaptive-context speed surplus on COVERAGE: does a narrower
//! packing window reduce long-form WER?
//!
//! Long-form windows are always ~30 s, so adaptive context never fires on
//! them — but it makes a 15 s window cost ~half a 30 s one, which changes
//! what we can afford. Fewer utterances per window means fewer chances for
//! one to be absorbed between two contiguous, individually-plausible spans
//! (the failure class span-based coverage repair cannot see).
use ffai_bench::corpus::Manifest;
use ffai_bench::metrics::{cer, wer};
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or("corpora/librispeech-longform.toml".into());
    let manifest = Manifest::load(&PathBuf::from(&corpus)).expect("corpus");
    let engine = WhisperCandle::new();
    let widths: Vec<f32> = std::env::args()
        .nth(2)
        .unwrap_or("30,20,15,10".into())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    // Warm-up so the first width is not charged for model load.
    if let Some(c) = manifest.clips.first() {
        if let Ok(a) = ffai_media::load_audio(&manifest.clip_path(c)) {
            let _ = engine.transcribe(&a, &AsrOptions::default());
        }
    }

    println!(
        "{:>7} {:>8} {:>8} {:>9} {:>8}",
        "WIDTH", "WER%", "CER%", "WALL_S", "xRT"
    );
    for w in widths {
        let opts = AsrOptions {
            vad_chunk_secs: w,
            ..AsrOptions::default()
        };
        let (mut num, mut den, mut cnum, mut cden, mut secs, mut audio_s) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for clip in &manifest.clips {
            let Ok(audio) = ffai_media::load_audio(&manifest.clip_path(clip)) else {
                continue;
            };
            let Ok(Some(truth)) = manifest.ground_truth(clip) else {
                continue;
            };
            audio_s += audio.duration_secs();
            let t = Instant::now();
            let Ok(out) = engine.transcribe(&audio, &opts) else {
                continue;
            };
            secs += t.elapsed().as_secs_f64();
            let text = out.text();
            // Weight per clip by reference length, as the bench harness does.
            let words = truth.split_whitespace().count() as f64;
            let chars = truth.chars().count() as f64;
            num += wer(&truth, &text) * words;
            den += words;
            cnum += cer(&truth, &text) * chars;
            cden += chars;
        }
        println!(
            "{w:>7.0} {:>8.2} {:>8.2} {secs:>9.2} {:>8.1}",
            num / den * 100.0,
            cnum / cden * 100.0,
            audio_s / secs
        );
    }
}
