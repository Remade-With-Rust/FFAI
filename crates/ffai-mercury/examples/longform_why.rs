//! Why is long-form WER 10.55 % when the same speech scores 6.79 % in clips?
//!
//! Phase E measured the gap and then guessed at it: "most likely an artifact
//! of concatenating unrelated utterances rather than a long-form weakness".
//! That was a hypothesis stated as a conclusion, which is the habit this
//! campaign keeps having to correct. This tests it.
//!
//! The long-form corpus is built from known source clips with exact spans, so
//! the same audio can be scored two ways:
//!
//! - **individually** — each source clip transcribed on its own, the way the
//!   benchmark corpus scores it.
//! - **in context** — the concatenated file transcribed once, then each
//!   utterance's words recovered by their timestamps.
//!
//! Identical audio, identical ground truth. Any difference is what putting an
//! utterance next to unrelated neighbours does to it.
//!
//! Two hypotheses, and the probe separates them:
//!
//! 1. **Context confusion** — Whisper conditions on preceding text, and the
//!    preceding text here is a different book by a different speaker. Then the
//!    damage should be spread across utterances, worst at the START of each
//!    (where the wrong context bites hardest).
//! 2. **Window boundaries** — errors concentrate where a 30 s encoder window
//!    cuts. Then damage should cluster at multiples of 30 s and be absent
//!    elsewhere.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example longform_why
//! ```

use std::path::PathBuf;

use ffai_bench::corpus::Manifest;
use ffai_bench::der::parse_rttm;
use ffai_bench::metrics::wer;
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_core::types::AudioBuffer;
use ffai_mercury::asr::WhisperCandle;

const SR: usize = 16_000;
const WINDOW: f64 = 30.0;

fn main() {
    let manifest = Manifest::load(&PathBuf::from("corpora/librispeech-longform.toml"))
        .expect("long-form corpus loads");
    let clip_truth = PathBuf::from("corpora/clips/librispeech-test-clean/truth");
    let engine = WhisperCandle::new();
    let opts = AsrOptions { word_timestamps: true, ..Default::default() };

    struct Utt {
        id: String,
        wer_alone: f64,
        wer_in_context: f64,
        start: f64,
        /// Distance from this utterance's start to the nearest 30 s boundary.
        to_boundary: f64,
        straddles: bool,
    }
    let mut utts: Vec<Utt> = Vec::new();

    for clip in &manifest.clips {
        let path = manifest.clip_path(clip);
        let Ok(audio) = ffai_media::load_audio(&path) else { continue };
        let spans_path = path
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("truth").join(format!("{}.rttm", clip.id)));
        let Some(spans) = spans_path
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|t| parse_rttm(&t))
        else {
            continue;
        };

        // One pass over the whole file, with word timings so each utterance's
        // share of the output can be recovered.
        let Ok(whole) = engine.transcribe(&audio, &opts) else { continue };
        let words = whole.words.clone().unwrap_or_default();
        let mono = audio.to_mono();

        for span in &spans {
            // `span.speaker` carries the source clip's stem.
            let Ok(truth) = std::fs::read_to_string(clip_truth.join(format!("{}.txt", span.speaker)))
            else {
                continue;
            };
            let truth = truth.trim().to_string();
            if truth.is_empty() {
                continue;
            }

            // Alone: exactly the audio of this utterance, transcribed by itself.
            let a = ((span.start * SR as f64) as usize).min(mono.samples.len());
            let b = ((span.end * SR as f64).ceil() as usize).clamp(a, mono.samples.len());
            let slice = AudioBuffer {
                samples: mono.samples[a..b].to_vec(),
                sample_rate: SR as u32,
                channels: 1,
            };
            let Ok(alone) = engine.transcribe(&slice, &AsrOptions::default()) else { continue };

            // In context: the words the whole-file pass placed inside this span.
            let in_ctx: String = words
                .iter()
                .filter(|w| w.start >= span.start - 0.1 && w.start < span.end + 0.1)
                .map(|w| w.value.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            let to_boundary = {
                let k = (span.start / WINDOW).round() * WINDOW;
                (span.start - k).abs()
            };
            utts.push(Utt {
                id: span.speaker.clone(),
                wer_alone: wer(&truth, &alone.text()),
                wer_in_context: wer(&truth, &in_ctx),
                start: span.start,
                to_boundary,
                straddles: (span.start / WINDOW).floor() != (span.end / WINDOW).floor(),
            });
        }
        eprintln!("  {} done ({} utterances)", clip.id, spans.len());
    }

    if utts.is_empty() {
        eprintln!("nothing scored");
        return;
    }

    let mean = |f: &dyn Fn(&Utt) -> f64| utts.iter().map(f).sum::<f64>() / utts.len() as f64;
    let alone = mean(&|u: &Utt| u.wer_alone);
    let ctx = mean(&|u: &Utt| u.wer_in_context);

    println!("\n=== same audio, same truth, {} utterances ===", utts.len());
    println!("  transcribed ALONE       mean WER {:.4}", alone);
    println!("  transcribed IN CONTEXT  mean WER {:.4}", ctx);
    println!("  difference              {:+.4}", ctx - alone);

    let worse = utts.iter().filter(|u| u.wer_in_context > u.wer_alone + 1e-9).count();
    let better = utts.iter().filter(|u| u.wer_in_context < u.wer_alone - 1e-9).count();
    let n = (worse + better) as f64;
    let z = if n > 0.0 { (worse as f64 - n / 2.0) / (n * 0.25).sqrt() } else { 0.0 };
    println!(
        "  worse in context {worse}   better {better}   tied {}   sign z = {z:+.2} {}",
        utts.len() - worse - better,
        if z.abs() > 2.0 { "SIGNIFICANT" } else { "(bar |z| > 2)" }
    );

    println!("\n=== hypothesis 2: is it the 30 s window boundary? ===");
    let straddling: Vec<&Utt> = utts.iter().filter(|u| u.straddles).collect();
    let clean: Vec<&Utt> = utts.iter().filter(|u| !u.straddles).collect();
    let m = |v: &[&Utt]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().map(|u| u.wer_in_context).sum::<f64>() / v.len() as f64
        }
    };
    println!(
        "  utterances STRADDLING a window boundary: {:>3}   mean WER {:.4}",
        straddling.len(),
        m(&straddling)
    );
    println!(
        "  utterances wholly inside one window    : {:>3}   mean WER {:.4}",
        clean.len(),
        m(&clean)
    );

    // If boundaries were the cause, WER should rise with proximity to one.
    let near: Vec<&Utt> = utts.iter().filter(|u| u.to_boundary < 5.0).collect();
    let far: Vec<&Utt> = utts.iter().filter(|u| u.to_boundary >= 5.0).collect();
    println!(
        "  starting within 5 s of a boundary      : {:>3}   mean WER {:.4}",
        near.len(),
        m(&near)
    );
    println!(
        "  starting far from a boundary           : {:>3}   mean WER {:.4}",
        far.len(),
        m(&far)
    );

    println!("\n=== worst degradations in context ===");
    let mut by_delta: Vec<&Utt> = utts.iter().collect();
    by_delta.sort_by(|a, b| {
        (b.wer_in_context - b.wer_alone).total_cmp(&(a.wer_in_context - a.wer_alone))
    });
    for u in by_delta.iter().take(5) {
        println!(
            "  {:<22} alone {:.3} -> context {:.3}   at {:>6.1}s, {:.1}s from a boundary{}",
            u.id,
            u.wer_alone,
            u.wer_in_context,
            u.start,
            u.to_boundary,
            if u.straddles { ", straddles" } else { "" }
        );
    }
}
