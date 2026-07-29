//! The streaming-diarization gate: does a speaker keep its label across calls?
//!
//! Each corpus conversation is fed in as **sequential chunks**, the way a live
//! stream arrives, instead of one whole-file call. The whole-file DER already
//! passes at 4.21 %; that measures nothing about identity across calls,
//! because a single call has no across to fail at.
//!
//! Two arms, and the comparison is the point:
//!
//! - **batch** — `persist_speakers: false`, today's default. Every chunk
//!   starts from nothing, so labels are per-chunk names for clusters.
//! - **streaming** — `persist_speakers: true`. The registry carries identity
//!   between chunks.
//!
//! Both are scored with DER against the same reference, with the labels
//! concatenated across chunks into one timeline. That is the honest way to
//! measure this: DER's optimal label mapping is computed **once over the whole
//! conversation**, so a system that renames a speaker halfway through cannot
//! hide behind a per-chunk remapping.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example streaming_gate
//! ```

use std::path::PathBuf;

use ffai_bench::corpus::Manifest;
use ffai_bench::der::{diarization_error_rate, parse_rttm, Turn};
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_core::types::AudioBuffer;
use ffai_mercury::asr::{diarize, WhisperCandle};

const COLLAR: f64 = 0.25;
/// How much audio each call sees. Short enough to be a stream, long enough to
/// hold a speaker turn.
const CHUNK_SECS: f64 = 8.0;
const SR: usize = 16_000;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpora/librispeech-diarization.toml".to_string());
    let manifest = match Manifest::load(&PathBuf::from(&path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot load {path}: {e}");
            std::process::exit(1);
        }
    };
    let engine = WhisperCandle::new();

    // `sweep` calibrates ENROL_MARGIN, the one knob the streaming arm turns
    // and the one that has never been measured.
    if std::env::args().any(|a| a == "sweep") {
        println!("{:>7} {:>6} {:>10} {:>28}", "margin", "match", "DER", "labels emitted / true");
        println!("{}", "-".repeat(56));
        let mut margin = 0.00f32;
        while margin <= 0.401 {
            // SAFETY: single-threaded here, set before the engine is used.
            unsafe { std::env::set_var("FFAI_ENROL_MARGIN", format!("{margin}")) };
            let (mut err, mut refs) = (0.0, 0.0);
            let mut counts = Vec::new();
            for clip in &manifest.clips {
                let Ok(audio) = ffai_media::load_audio(&manifest.clip_path(clip)) else { continue };
                let Ok(Some(truth)) = manifest.ground_truth(clip) else { continue };
                let reference = parse_rttm(&truth);
                let true_n = reference
                    .iter()
                    .map(|t| t.speaker.as_str())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                let mono = audio.to_mono();
                let per_chunk = (CHUNK_SECS * SR as f64) as usize;
                engine.reset_speakers();
                let mut turns: Vec<Turn> = Vec::new();
                for c in 0..mono.samples.len().div_ceil(per_chunk) {
                    let a = c * per_chunk;
                    let b = ((c + 1) * per_chunk).min(mono.samples.len());
                    if b <= a {
                        continue;
                    }
                    let chunk = AudioBuffer {
                        samples: mono.samples[a..b].to_vec(),
                        sample_rate: SR as u32,
                        channels: 1,
                    };
                    let opts = AsrOptions {
                        diarize: true,
                        persist_speakers: true,
                        max_speakers: None,
                        diarize_threshold: diarize::DEFAULT_THRESHOLD,
                        ..Default::default()
                    };
                    let Ok(t) = engine.transcribe(&chunk, &opts) else { continue };
                    let offset = a as f64 / SR as f64;
                    for s in t.speakers.iter().flatten() {
                        turns.push(Turn::new(s.start + offset, s.end + offset, s.value.clone()));
                    }
                }
                let labels = turns
                    .iter()
                    .map(|t| t.speaker.as_str())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                counts.push(format!("{labels}/{true_n}"));
                let (b, _) = diarization_error_rate(&reference, &turns, COLLAR);
                err += b.missed_secs + b.false_alarm_secs + b.confusion_secs;
                refs += b.reference_secs;
            }
            println!(
                "{margin:>7.2} {:>6.2} {:>9.2}% {:>28}",
                diarize::DEFAULT_THRESHOLD - margin,
                100.0 * err / refs.max(1e-9),
                counts.join(" ")
            );
            margin += 0.05;
        }
        return;
    }

    println!(
        "{:<10} {:>6} {:>8} {:>10} {:>8} {:>10}",
        "clip", "chunks", "batch", "batch lbls", "stream", "stream lbls"
    );
    println!("{}", "-".repeat(56));

    let (mut b_err, mut b_ref, mut s_err, mut s_ref) = (0.0, 0.0, 0.0, 0.0);

    for clip in &manifest.clips {
        let Ok(audio) = ffai_media::load_audio(&manifest.clip_path(clip)) else { continue };
        let Ok(Some(truth)) = manifest.ground_truth(clip) else { continue };
        let reference = parse_rttm(&truth);
        let true_n = reference
            .iter()
            .map(|t| t.speaker.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();

        let mono = audio.to_mono();
        let per_chunk = (CHUNK_SECS * SR as f64) as usize;
        let n_chunks = mono.samples.len().div_ceil(per_chunk);

        let mut results = Vec::new();
        for persist in [false, true] {
            // A new arm is a new session: whatever the previous arm taught the
            // registry must not leak into this one.
            engine.reset_speakers();
            let mut turns: Vec<Turn> = Vec::new();

            for c in 0..n_chunks {
                let a = c * per_chunk;
                let b = ((c + 1) * per_chunk).min(mono.samples.len());
                if b <= a {
                    continue;
                }
                let chunk = AudioBuffer {
                    samples: mono.samples[a..b].to_vec(),
                    sample_rate: SR as u32,
                    channels: 1,
                };
                let opts = AsrOptions {
                    diarize: true,
                    persist_speakers: persist,
                    max_speakers: None,
                    diarize_threshold: diarize::DEFAULT_THRESHOLD,
                    ..Default::default()
                };
                let Ok(t) = engine.transcribe(&chunk, &opts) else { continue };
                let offset = a as f64 / SR as f64;
                for s in t.speakers.iter().flatten() {
                    // Chunk-local times shift into the conversation's clock;
                    // labels are taken AS-IS, which is exactly what is under
                    // test — a batch arm's SPEAKER_00 means something
                    // different in every chunk.
                    turns.push(Turn::new(s.start + offset, s.end + offset, s.value.clone()));
                }
            }

            let labels = turns
                .iter()
                .map(|t| t.speaker.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let (br, _) = diarization_error_rate(&reference, &turns, COLLAR);
            results.push((br, labels));
        }

        let (batch, batch_labels) = &results[0];
        let (stream, stream_labels) = &results[1];
        b_err += batch.missed_secs + batch.false_alarm_secs + batch.confusion_secs;
        b_ref += batch.reference_secs;
        s_err += stream.missed_secs + stream.false_alarm_secs + stream.confusion_secs;
        s_ref += stream.reference_secs;

        println!(
            "{:<10} {:>6} {:>7.1}% {:>7}/{:<2} {:>7.1}% {:>7}/{:<2}",
            clip.id,
            n_chunks,
            100.0 * batch.der().unwrap_or(f64::NAN),
            batch_labels,
            true_n,
            100.0 * stream.der().unwrap_or(f64::NAN),
            stream_labels,
            true_n
        );
    }

    let batch_der = 100.0 * b_err / b_ref.max(1e-9);
    let stream_der = 100.0 * s_err / s_ref.max(1e-9);
    println!("\n{}", "=".repeat(56));
    println!("  batch     (persist_speakers: false)  DER {batch_der:>6.2}%");
    println!("  streaming (persist_speakers: true)   DER {stream_der:>6.2}%");
    println!(
        "\n  GATE: {}",
        if stream_der < batch_der {
            "PASS — identity survives across calls"
        } else {
            "FAIL — the registry is not buying anything"
        }
    );
    println!(
        "\nThe batch arm is not broken code; it is the correct BATCH answer\n\
         measured on a streaming workload, which is the gap this milestone\n\
         exists to close. `lbls` is distinct labels emitted vs true speakers —\n\
         a batch arm inflates it because every chunk names its clusters afresh."
    );
}
