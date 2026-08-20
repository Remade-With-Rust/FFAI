//! The live diarization pattern: a sliding window, ticked.
//!
//! `live_cost.rs` prices ONE chunk. This prices the pattern the demo
//! actually produces — a trailing N-second buffer re-sent every tick — which
//! is where the waste lived: consecutive ticks share all but one window of
//! audio, and every one of them was being re-embedded.
//!
//! Reports wall per tick and the embedding cache's hit rate, because a
//! speedup without a hit rate is a number you cannot attribute.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example live_diarize -- clip.wav 10 1
//! ```
use std::time::Instant;

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("corpora/clips/librispeech-longform/audio/longform-01.wav".into());
    let window: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);
    let tick: f64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let ticks: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let audio = ffai_media::load_audio(std::path::Path::new(&path))?;
    let mono = audio.to_mono();
    let sr = mono.sample_rate as f64;

    let engine = WhisperCandle::new();
    let opts = AsrOptions {
        diarize: true,
        persist_speakers: true,
        ..Default::default()
    };

    // Warm: model loads (Whisper AND the speaker net) outside the timed loop.
    let head = ffai_core::types::AudioBuffer {
        samples: mono.samples[..(window * sr) as usize].to_vec(),
        sample_rate: mono.sample_rate,
        channels: 1,
    };
    engine.transcribe(&head, &opts)?;

    println!("{path}\n{window} s window, {tick} s tick, {ticks} ticks\n");
    // PAIRED: the same tick's audio through both arms, back to back. Cost
    // varies with how much speech VAD finds in that particular tick, so
    // arm-by-arm over separate runs compares different work — the exact
    // error that made the demo read "2x slower".
    println!(
        "{:>5} {:>10} {:>10} {:>9}",
        "tick", "cache ms", "no-cache", "speedup"
    );
    let (mut on, mut off) = (Vec::new(), Vec::new());
    for i in 0..ticks {
        // The trailing `window` seconds as of this tick — exactly what the
        // browser sends.
        let end = ((window + i as f64 * tick) * sr) as usize;
        if end > mono.samples.len() {
            break;
        }
        let start = end.saturating_sub((window * sr) as usize);
        let buf = ffai_core::types::AudioBuffer {
            samples: mono.samples[start..end].to_vec(),
            sample_rate: mono.sample_rate,
            channels: 1,
        };
        // Tell the engine where this buffer sits in the stream, which is what
        // lets the window grid stay put while the buffer slides. Omitting it
        // would measure the feature with the feature turned off.
        let opts = AsrOptions {
            stream_offset_secs: start as f64 / sr,
            ..opts.clone()
        };
        // Cache arm first so it faces a cache warmed only by EARLIER ticks —
        // the real live condition. Running no-cache first would let it warm
        // the cache for its own comparison.
        unsafe { std::env::set_var("FFAI_DIARIZE_CACHE", "on") };
        let t = Instant::now();
        engine.transcribe(&buf, &opts)?;
        let a = t.elapsed().as_secs_f64() * 1e3;

        unsafe { std::env::set_var("FFAI_DIARIZE_CACHE", "off") };
        let t = Instant::now();
        engine.transcribe(&buf, &opts)?;
        let b = t.elapsed().as_secs_f64() * 1e3;
        unsafe { std::env::set_var("FFAI_DIARIZE_CACHE", "on") };

        on.push(a);
        off.push(b);
        println!("{:>5} {a:>10.0} {b:>10.0} {:>8.2}x", i + 1, b / a);
    }

    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (mo, mf) = (med(on.clone()), med(off.clone()));
    let wins = on.iter().zip(off.iter()).filter(|(a, b)| a < b).count();
    let (hits, misses) = ffai_mercury::asr::diarizer::cache_stats();
    println!(
        "\nmedian: cache {mo:.0} ms · no-cache {mf:.0} ms  ->  {:.2}x",
        mf / mo
    );
    println!("cache won {wins}/{} paired ticks", on.len());
    println!(
        "embed calls: {hits} hits / {misses} misses ({:.0}% hit rate)",
        hits as f64 / (hits + misses).max(1) as f64 * 100.0
    );
    Ok(())
}
