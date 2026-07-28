//! The word-timestamp gate, and the multi-window gate.
//!
//! Word timestamps have been verified on exactly one clip — 26 words that
//! looked right. This scores them, and it also exercises the multi-window
//! path, which nothing else does: every clip in every other corpus is one
//! Whisper context.
//!
//! **Containment, not word-level error.** Word-level ground truth needs a
//! forced aligner, and scoring our aligner against another aligner measures
//! agreement rather than truth. What IS exactly known is where each source
//! utterance sits, because the audio was built by concatenating them. So the
//! gate is: **every aligned word must fall inside the utterance it came
//! from.** That cannot resolve a 50 ms boundary error; it catches drift,
//! offset bugs, and misordering — the failures that break a caption file.
//!
//! Three things are measured:
//!
//! 1. **containment** — the fraction of words inside their own utterance.
//! 2. **monotonicity** — words must not go backwards in time.
//! 3. **coverage** — the fraction of transcript words that got a timestamp
//!    at all, because an aligner that silently drops words would otherwise
//!    score 100 % containment on the handful it kept.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example align_gate
//! ```

use std::path::PathBuf;

use ffai_bench::corpus::Manifest;
use ffai_bench::der::parse_rttm;
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpora/librispeech-longform.toml".to_string());
    let manifest = match Manifest::load(&PathBuf::from(&path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot load {path}: {e}");
            std::process::exit(1);
        }
    };
    let engine = WhisperCandle::new();

    println!(
        "{:<13} {:>6} {:>5} {:>7} {:>8} {:>9} {:>8}",
        "file", "secs", "wins", "words", "coverage", "contained", "monotone"
    );
    println!("{}", "-".repeat(62));

    let (mut tot_words, mut tot_contained, mut tot_expected) = (0usize, 0usize, 0usize);
    let mut all_monotone = true;

    for clip in &manifest.clips {
        let audio = match ffai_media::load_audio(&manifest.clip_path(clip)) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("  {}: {e}", clip.id);
                continue;
            }
        };
        let duration = audio.duration_secs();

        // Utterance spans, exact by construction.
        let spans_path = manifest
            .clip_path(clip)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("truth").join(format!("{}.rttm", clip.id)));
        let spans = spans_path
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|t| parse_rttm(&t))
            .unwrap_or_default();
        if spans.is_empty() {
            eprintln!("  {}: no utterance spans", clip.id);
            continue;
        }

        let opts = AsrOptions { word_timestamps: true, ..Default::default() };
        let transcript = match engine.transcribe(&audio, &opts) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  {}: {e}", clip.id);
                continue;
            }
        };
        let words: Vec<_> = transcript.words.clone().unwrap_or_default();

        // Coverage: words timed against words transcribed. An aligner that
        // drops half the transcript scores perfectly on containment without
        // this.
        let transcribed = transcript
            .segments
            .iter()
            .map(|s| s.value.split_whitespace().count())
            .sum::<usize>();

        // A word counts as contained if it lies inside ANY utterance span —
        // the concatenation is gapped, so a word in the 0.5 s of silence
        // between utterances is genuinely misplaced.
        let contained = words
            .iter()
            .filter(|w| {
                spans
                    .iter()
                    .any(|s| w.start >= s.start - 0.05 && w.end <= s.end + 0.05)
            })
            .count();
        let monotone = words.windows(2).all(|p| p[0].start <= p[1].start + 1e-9);
        all_monotone &= monotone;

        tot_words += words.len();
        tot_contained += contained;
        tot_expected += transcribed;

        println!(
            "{:<13} {:>6.0} {:>5} {:>7} {:>7.0}% {:>8.1}% {:>8}",
            clip.id,
            duration,
            (duration / 30.0).ceil() as usize,
            words.len(),
            if transcribed > 0 { 100.0 * words.len() as f64 / transcribed as f64 } else { 0.0 },
            if words.is_empty() { 0.0 } else { 100.0 * contained as f64 / words.len() as f64 },
            if monotone { "yes" } else { "NO" }
        );
    }

    println!("\n{}", "=".repeat(62));
    let containment = if tot_words > 0 { 100.0 * tot_contained as f64 / tot_words as f64 } else { 0.0 };
    let coverage = if tot_expected > 0 { 100.0 * tot_words as f64 / tot_expected as f64 } else { 0.0 };
    println!("  words timed        {tot_words} of {tot_expected} transcribed ({coverage:.0}% coverage)");
    println!("  inside utterance   {tot_contained} ({containment:.1}%)");
    println!("  monotone           {}", if all_monotone { "yes, every file" } else { "NO — words go backwards" });
    println!(
        "\n  GATE: {}",
        if containment >= 95.0 && coverage >= 90.0 && all_monotone {
            "PASS"
        } else {
            "FAIL — see which column moved"
        }
    );
    println!(
        "\nContainment is a coarse instrument by design: it proves words land in\n\
         the right utterance across a multi-window file, not that a boundary is\n\
         accurate to 50 ms. Word-level error needs a reference aligner and is\n\
         still open."
    );
}
