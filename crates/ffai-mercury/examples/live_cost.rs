//! What does a LIVE chunk actually cost, and where does it go?
//!
//! The demo runs Mercury with `diarize + persist_speakers` and whisper.cpp
//! with neither, then prints a per-chunk average for each. That average
//! compares different work: an extra speaker-embedding network on our side,
//! and — because VAD drops silent chunks before the encoder — a *speech-only*
//! population on ours against a mostly-silence population on theirs.
//!
//! This prices both, interleaved in one process so thermal drift hits both
//! arms equally.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example live_cost -- clip.wav
//! ```
use std::time::Instant;

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("corpora/clips/librispeech-test-clean/audio/1089-134686-0000.wav".into());
    let rounds: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let audio = ffai_media::load_audio(std::path::Path::new(&path))?;

    // A live tick is a few seconds, not a whole utterance: take the first 3 s
    // so the numbers are comparable to the demo's chunks.
    let mono = audio.to_mono();
    let n = (3.0 * mono.sample_rate as f64) as usize;
    let chunk = ffai_core::types::AudioBuffer {
        samples: mono.samples[..n.min(mono.samples.len())].to_vec(),
        sample_rate: mono.sample_rate,
        channels: 1,
    };
    let silence = ffai_core::types::AudioBuffer {
        samples: vec![0.0; chunk.samples.len()],
        sample_rate: chunk.sample_rate,
        channels: 1,
    };

    let engine = WhisperCandle::new();
    let asr_only = AsrOptions::default();
    let diarized = AsrOptions {
        diarize: true,
        persist_speakers: true,
        ..Default::default()
    };

    // Warm both paths: model load and the speaker model's first fetch must
    // not land inside a measured round.
    engine.transcribe(&chunk, &asr_only)?;
    engine.transcribe(&chunk, &diarized)?;

    let (mut a, mut b, mut s) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..rounds {
        // Alternate the order each round so "second one runs warmer" cancels
        // rather than accumulating into one arm.
        if r % 2 == 0 {
            let t = Instant::now();
            engine.transcribe(&chunk, &asr_only)?;
            a.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            engine.transcribe(&chunk, &diarized)?;
            b.push(t.elapsed().as_secs_f64() * 1e3);
        } else {
            let t = Instant::now();
            engine.transcribe(&chunk, &diarized)?;
            b.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            engine.transcribe(&chunk, &asr_only)?;
            a.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let t = Instant::now();
        engine.transcribe(&silence, &diarized)?;
        s.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let (ma, mb, ms) = (median(a), median(b), median(s));
    println!("{} · 3 s chunk · {rounds} interleaved rounds\n", path);
    println!("{:<34} {:>10}", "CONFIGURATION", "ms/chunk");
    println!("{:<34} {ma:>10.0}", "speech: ASR only (what cpp does)");
    println!("{:<34} {mb:>10.0}", "speech: ASR + diarize + persist");
    println!("{:<34} {ms:>10.0}", "SILENCE: full demo config");
    println!(
        "\ndiarization costs {:+.0} ms/chunk ({:.2}x the ASR-only path)",
        mb - ma,
        mb / ma
    );
    println!(
        "silence costs {ms:.0} ms — VAD drops it before the encoder, where whisper.cpp\n\
         pays a full pass to print [BLANK_AUDIO]"
    );
    Ok(())
}
