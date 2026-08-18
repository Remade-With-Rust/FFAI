//! The diarization gate: DER against the Phase E corpus.
//!
//! Until this existed, `--diarize` had no measurement at all — it was
//! verified only by looking at one hand-built fixture and agreeing that the
//! labels seemed right. This scores it.
//!
//! Two conditions are reported per conversation, and the difference between
//! them is the interesting part:
//!
//! - **oracle count** — the true number of speakers is supplied. Measures the
//!   embeddings and the clustering, with the threshold taken out of play.
//! - **blind** — the system must infer the count from
//!   [`diarize::DEFAULT_THRESHOLD`]. Measures the threshold too, which is the
//!   knob documented as untuned.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example diarize_gate
//! ```

use std::path::PathBuf;

use ffai_bench::corpus::Manifest;
use ffai_bench::der::{Turn, diarization_error_rate, parse_rttm};
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::{WhisperCandle, diarize};

/// NIST convention, and what pyannote reports by default.
const COLLAR: f64 = 0.25;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpora/librispeech-diarization.toml".to_string());
    let manifest = match Manifest::load(&PathBuf::from(&path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "cannot load {path}: {e}\nrun: cargo run --release -p ffai-bench --example prepare_phase_e"
            );
            std::process::exit(1);
        }
    };

    let engine = WhisperCandle::new();

    // `sweep` mode: the threshold is the one knob the blind condition
    // measures, and it has never been tuned. Sweeping it here is what turns
    // DEFAULT_THRESHOLD from a guess into a measurement.
    if std::env::args().any(|a| a == "sweep") {
        println!(
            "{:>6}  {:>10}  {:>26}",
            "thresh", "corpus DER", "speakers found / true"
        );
        println!("{}", "-".repeat(48));
        let mut t = 0.5f32;
        while t <= 1.45 {
            let mut sum_err = 0.0;
            let mut sum_ref = 0.0;
            let mut counts = Vec::new();
            for clip in &manifest.clips {
                let Ok(audio) = ffai_media::load_audio(&manifest.clip_path(clip)) else {
                    continue;
                };
                let Ok(Some(truth)) = manifest.ground_truth(clip) else {
                    continue;
                };
                let reference = parse_rttm(&truth);
                let true_n = reference
                    .iter()
                    .map(|x| x.speaker.as_str())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                let opts = AsrOptions {
                    diarize: true,
                    max_speakers: None,
                    diarize_threshold: t,
                    ..Default::default()
                };
                let Ok(tr) = engine.transcribe(&audio, &opts) else {
                    continue;
                };
                let hyp: Vec<Turn> = tr
                    .speakers
                    .iter()
                    .flatten()
                    .map(|s| Turn::new(s.start, s.end, s.value.clone()))
                    .collect();
                let found = hyp
                    .iter()
                    .map(|x| x.speaker.as_str())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                counts.push(format!("{found}/{true_n}"));
                let (b, _) = diarization_error_rate(&reference, &hyp, COLLAR);
                sum_err += b.missed_secs + b.false_alarm_secs + b.confusion_secs;
                sum_ref += b.reference_secs;
            }
            println!(
                "{t:>6.2}  {:>9.2}%  {:>26}",
                100.0 * sum_err / sum_ref.max(1e-9),
                counts.join(" ")
            );
            t += 0.05;
        }
        return;
    }

    println!(
        "{:<10} {:>6} {:>9} {:>9} {:>9} {:>9}",
        "clip", "spk", "DER@.25", "DER@0", "missed", "confus"
    );
    println!("{}", "-".repeat(58));

    let mut rows: Vec<(String, f64, f64, usize, usize)> = Vec::new();
    for mode in ["oracle", "blind"] {
        let mut sum_err = 0.0;
        let mut sum_ref = 0.0;
        for clip in &manifest.clips {
            let audio = match ffai_media::load_audio(&manifest.clip_path(clip)) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("  {}: {e}", clip.id);
                    continue;
                }
            };
            let truth_text = match manifest.ground_truth(clip) {
                Ok(Some(t)) => t,
                _ => continue,
            };
            let reference = parse_rttm(&truth_text);
            let true_speakers = reference
                .iter()
                .map(|t| t.speaker.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len();

            let opts = AsrOptions {
                diarize: true,
                max_speakers: if mode == "oracle" {
                    Some(true_speakers)
                } else {
                    None
                },
                diarize_threshold: diarize::DEFAULT_THRESHOLD,
                ..Default::default()
            };
            let transcript = match engine.transcribe(&audio, &opts) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("  {}: {e}", clip.id);
                    continue;
                }
            };
            let hypothesis: Vec<Turn> = transcript
                .speakers
                .iter()
                .flatten()
                .map(|s| Turn::new(s.start, s.end, s.value.clone()))
                .collect();
            let found = hypothesis
                .iter()
                .map(|t| t.speaker.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len();

            let (collared, _) = diarization_error_rate(&reference, &hypothesis, COLLAR);
            let (full, _) = diarization_error_rate(&reference, &hypothesis, 0.0);
            sum_err += collared.missed_secs + collared.false_alarm_secs + collared.confusion_secs;
            sum_ref += collared.reference_secs;

            if mode == "oracle" {
                rows.push((
                    clip.id.clone(),
                    collared.der().unwrap_or(f64::NAN),
                    full.der().unwrap_or(f64::NAN),
                    true_speakers,
                    found,
                ));
            }
            println!(
                "{:<10} {:>2}/{:<3} {:>8.1}% {:>8.1}% {:>8.2}s {:>8.2}s",
                format!(
                    "{}:{}",
                    &clip.id[..clip.id.len().min(7)],
                    if mode == "oracle" { "o" } else { "b" }
                ),
                found,
                true_speakers,
                100.0 * collared.der().unwrap_or(f64::NAN),
                100.0 * full.der().unwrap_or(f64::NAN),
                collared.missed_secs,
                collared.confusion_secs,
            );
        }
        let corpus_der = if sum_ref > 0.0 {
            sum_err / sum_ref
        } else {
            f64::NAN
        };
        println!(
            "  {mode:<8} CORPUS DER @{COLLAR} collar: {:.2}%\n",
            100.0 * corpus_der
        );
    }

    println!(
        "Corpus DER is the pooled ratio, not the mean of per-clip rates — a\n\
         long conversation should weigh more than a short one, and averaging\n\
         the rates would give them equal say.\n\n\
         `spk` is found/true. Getting the COUNT right is most of diarization;\n\
         a system that finds one speaker in a four-way conversation can still\n\
         post a modest DER by covering the speech, which is why the count is\n\
         printed next to the rate rather than folded into it."
    );
    let _ = rows;
}
