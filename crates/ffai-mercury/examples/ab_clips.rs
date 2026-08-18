//! Per-clip A/B decomposition between two engine configurations.
//!
//! **Why this is a tool and not another one-off probe.** Three times now a
//! corpus metric has moved and the aggregate has been believed before the
//! distribution was looked at — VAD's "15 % quality win" that decomposed to
//! 38 improved / 38 worsened at z = 0.00 being the worst case. Each time the
//! decomposition was rebuilt from scratch. This is that decomposition, once.
//!
//! Arm A is the shipped default. Arm B sets one `FFAI_*` override, so any
//! knob the engine already exposes for A/B can be decomposed without new code.
//!
//! ```sh
//! # Q1: does conditioning annotations on P(<|nospeech|>) actually help?
//! cargo run --release -p ffai-mercury --example ab_clips -- \
//!     corpora/librispeech-test-clean-v2.toml FFAI_ANNOT_THRESHOLD=0.10
//!
//! # any other knob
//! ... FFAI_VAD=0
//! ```
//!
//! Reports what a corpus delta cannot: how many clips moved each way, the
//! sign test on that, and what share of the total a handful of clips carry.
//! A metric that improves while the sign test sits at zero is a re-roll, not
//! a mechanism.

use std::path::PathBuf;

use ffai_bench::corpus::Manifest;
use ffai_bench::metrics::{cer, wer};
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

struct Row {
    id: String,
    wer_a: f64,
    wer_b: f64,
    cer_a: f64,
    cer_b: f64,
    text_a: String,
    text_b: String,
}

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpora/librispeech-test-clean-v2.toml".to_string());
    let Some(override_spec) = std::env::args().nth(2) else {
        eprintln!("usage: ab_clips <corpus.toml> KEY=VALUE  (KEY is an FFAI_* knob)");
        std::process::exit(1);
    };
    let (key, value) = match override_spec.split_once('=') {
        Some(kv) => kv,
        None => {
            eprintln!("arm B must be KEY=VALUE, got `{override_spec}`");
            std::process::exit(1);
        }
    };
    let limit: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let manifest = Manifest::load(&PathBuf::from(&corpus)).expect("corpus loads");
    let engine = WhisperCandle::new();
    let opts = AsrOptions::default();

    println!("corpus : {corpus}");
    println!("arm A  : shipped defaults");
    println!("arm B  : {key}={value}\n");

    let mut rows = Vec::new();
    for clip in manifest.clips.iter().take(limit) {
        let Ok(audio) = ffai_media::load_audio(&manifest.clip_path(clip)) else {
            continue;
        };
        let Ok(Some(truth)) = manifest.ground_truth(clip) else {
            continue;
        };

        // SAFETY: single-threaded, and set between calls rather than during
        // one. The engine reads these when it builds its DecodeConfig, so the
        // two arms genuinely differ by this one knob and nothing else.
        unsafe { std::env::remove_var(key) };
        let Ok(a) = engine.transcribe(&audio, &opts) else {
            continue;
        };
        unsafe { std::env::set_var(key, value) };
        let Ok(b) = engine.transcribe(&audio, &opts) else {
            continue;
        };
        unsafe { std::env::remove_var(key) };

        let (ta, tb) = (a.text(), b.text());
        rows.push(Row {
            id: clip.id.clone(),
            wer_a: wer(&truth, &ta),
            wer_b: wer(&truth, &tb),
            cer_a: cer(&truth, &ta),
            cer_b: cer(&truth, &tb),
            text_a: ta,
            text_b: tb,
        });
        if rows.len() % 25 == 0 {
            eprintln!("  {} clips ...", rows.len());
        }
    }

    if rows.is_empty() {
        eprintln!("no clips scored");
        return;
    }

    for (name, get_a, get_b) in [
        (
            "WER",
            (|r: &Row| r.wer_a) as fn(&Row) -> f64,
            (|r: &Row| r.wer_b) as fn(&Row) -> f64,
        ),
        (
            "CER",
            (|r: &Row| r.cer_a) as fn(&Row) -> f64,
            (|r: &Row| r.cer_b) as fn(&Row) -> f64,
        ),
    ] {
        let improved = rows.iter().filter(|r| get_b(r) < get_a(r) - 1e-9).count();
        let worsened = rows.iter().filter(|r| get_b(r) > get_a(r) + 1e-9).count();
        let unchanged = rows.len() - improved - worsened;

        // Sign test over the clips that moved. Ties carry no directional
        // information, so they are excluded rather than counted as agreement.
        let n = (improved + worsened) as f64;
        let z = if n > 0.0 {
            (improved as f64 - n / 2.0) / (n * 0.25).sqrt()
        } else {
            0.0
        };

        let mut deltas: Vec<f64> = rows.iter().map(|r| get_a(r) - get_b(r)).collect();
        deltas.sort_by(|a, b| b.total_cmp(a));
        let total: f64 = deltas.iter().sum();
        let top5: f64 = deltas.iter().take(5).sum();

        println!("=== {name} ===");
        println!("  improved {improved}   worsened {worsened}   unchanged {unchanged}");
        println!(
            "  sign test  z = {z:+.2}   {}",
            if z.abs() > 2.0 {
                "SIGNIFICANT"
            } else {
                "not significant (bar is |z| > 2)"
            }
        );
        println!(
            "  net delta {total:+.3}; top 5 clips carry {top5:+.3} ({:.0}% of net)",
            if total.abs() > 1e-9 {
                100.0 * top5 / total
            } else {
                0.0
            }
        );
        if total.abs() > 1e-9 && (100.0 * top5 / total) > 100.0 {
            println!("  NOTE: top-5 share exceeds 100% — the remaining clips are net NEGATIVE");
        }
        println!();
    }

    // Corpus aggregate, for contrast with the distribution above.
    let truths: Vec<String> = manifest
        .clips
        .iter()
        .take(rows.len())
        .filter_map(|c| manifest.ground_truth(c).ok().flatten())
        .collect();
    let all_truth = truths.join(" ");
    let all_a: Vec<&str> = rows.iter().map(|r| r.text_a.as_str()).collect();
    let all_b: Vec<&str> = rows.iter().map(|r| r.text_b.as_str()).collect();
    println!("=== corpus aggregate (what a bench line reports) ===");
    println!(
        "  A: WER {:.4}  CER {:.4}\n  B: WER {:.4}  CER {:.4}",
        wer(&all_truth, &all_a.join(" ")),
        cer(&all_truth, &all_a.join(" ")),
        wer(&all_truth, &all_b.join(" ")),
        cer(&all_truth, &all_b.join(" "))
    );

    println!("\n=== biggest movers, both directions ===");
    let mut by_delta: Vec<&Row> = rows.iter().collect();
    by_delta.sort_by(|x, y| (y.wer_a - y.wer_b).total_cmp(&(x.wer_a - x.wer_b)));
    for r in by_delta.iter().take(3).chain(by_delta.iter().rev().take(3)) {
        if (r.wer_a - r.wer_b).abs() < 1e-9 {
            continue;
        }
        println!("\n[{}] WER {:.3} -> {:.3}", r.id, r.wer_a, r.wer_b);
        println!("  A: {}", one_line(&r.text_a));
        println!("  B: {}", one_line(&r.text_b));
    }
}

fn one_line(s: &str) -> String {
    let joined = s.replace('\n', " / ");
    if joined.chars().count() > 130 {
        joined.chars().take(130).collect::<String>() + " ..."
    } else {
        joined
    }
}
