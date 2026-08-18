//! Q2: is the CER difference against whisper.cpp systematic, or a tail?
//!
//! A test-clean CER deficit has been open since mission-plan §6.7 — 3.27 %
//! against whisper.cpp's 2.87 %, ~14 % relative, absent on test-other. It has
//! survived int8, the f16 cache, and every kernel change since. It has also
//! never been **looked at**: the campaign has re-measured the gap repeatedly
//! and never once inspected which characters were wrong.
//!
//! It is now a *lead* (2.74 % vs 2.87 %) — but it moved there via speech
//! segmentation, whose per-clip effect is 38 improved / 38 worsened at
//! z = 0.00. A gap closed by something with no directional effect is
//! displaced, not explained, and that makes the original question sharper
//! rather than moot: **was it ever systematic?**
//!
//! This decomposes it per clip against whisper.cpp's own output, and prints
//! the character edits rather than the rate.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example cer_anatomy
//! ```

use std::path::PathBuf;

use ffai_bench::corpus::Manifest;
use ffai_bench::metrics::cer;
use ffai_bench::normalize::{Mode, normalize};
use ffai_bench::reference::ReferenceFile;
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

const REFERENCE: &str = "whisper-cpp-tiny-greedy-t24";

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpora/librispeech-test-clean-v2.toml".to_string());
    let manifest = Manifest::load(&PathBuf::from(&corpus)).expect("corpus loads");
    let refs = ReferenceFile::load(&PathBuf::from("corpora/references.toml")).expect("refs load");
    let spec = refs
        .for_task("asr")
        .find(|r| r.name == REFERENCE)
        .expect("reference declared");

    let clips: Vec<_> = manifest.holdout().collect();
    let paths: Vec<PathBuf> = clips.iter().map(|c| manifest.clip_path(c)).collect();

    eprintln!("running {REFERENCE} over {} clips ...", paths.len());
    let batch = spec.run_batch(&paths).expect("reference runs");

    let engine = WhisperCandle::new();
    let opts = AsrOptions::default();

    struct Row {
        id: String,
        ours: f64,
        theirs: f64,
        our_text: String,
        their_text: String,
        truth: String,
    }
    let mut rows = Vec::new();

    for (clip, path) in clips.iter().zip(&paths) {
        let Some(their_text) = batch.text_for(path) else {
            continue;
        };
        let Ok(audio) = ffai_media::load_audio(path) else {
            continue;
        };
        let Ok(Some(truth)) = manifest.ground_truth(clip) else {
            continue;
        };
        let Ok(t) = engine.transcribe(&audio, &opts) else {
            continue;
        };
        let our_text = t.text();
        rows.push(Row {
            id: clip.id.clone(),
            ours: cer(&truth, &our_text),
            theirs: cer(&truth, their_text),
            our_text,
            their_text: their_text.to_string(),
            truth,
        });
    }

    // ---- distribution, not aggregate ----
    let better = rows.iter().filter(|r| r.ours < r.theirs - 1e-9).count();
    let worse = rows.iter().filter(|r| r.ours > r.theirs + 1e-9).count();
    let same = rows.len() - better - worse;
    let n = (better + worse) as f64;
    let z = if n > 0.0 {
        (better as f64 - n / 2.0) / (n * 0.25).sqrt()
    } else {
        0.0
    };

    let mut deltas: Vec<f64> = rows.iter().map(|r| r.theirs - r.ours).collect();
    deltas.sort_by(|a, b| b.total_cmp(a));
    let total: f64 = deltas.iter().sum();
    let top5: f64 = deltas.iter().take(5).sum();

    println!(
        "\n=== per-clip CER, us vs {REFERENCE} ({} clips) ===",
        rows.len()
    );
    println!("  we are better on {better}   worse on {worse}   tied {same}");
    println!(
        "  sign test  z = {z:+.2}   {}",
        if z.abs() > 2.0 {
            "SIGNIFICANT"
        } else {
            "not significant (bar is |z| > 2)"
        }
    );
    println!(
        "  net {total:+.3}; top 5 clips carry {top5:+.3} ({:.0}% of net)",
        if total.abs() > 1e-9 {
            100.0 * top5 / total
        } else {
            0.0
        }
    );

    // ---- what the characters actually are ----
    println!("\n=== the clips we lose most, with the actual edits ===");
    let mut by_gap: Vec<&Row> = rows.iter().collect();
    by_gap.sort_by(|a, b| (b.ours - b.theirs).total_cmp(&(a.ours - a.theirs)));
    for r in by_gap.iter().take(6) {
        if r.ours <= r.theirs + 1e-9 {
            break;
        }
        println!(
            "\n[{}] CER ours {:.3} vs theirs {:.3}",
            r.id, r.ours, r.theirs
        );
        let truth = normalize(&r.truth, Mode::default());
        let ours = normalize(&r.our_text, Mode::default());
        let theirs = normalize(&r.their_text, Mode::default());
        for (label, hyp) in [("ours  ", &ours), ("theirs", &theirs)] {
            let diffs = word_diffs(&truth, hyp);
            println!(
                "  {label}: {}",
                if diffs.is_empty() {
                    "(exact)".to_string()
                } else {
                    diffs.join("  ")
                }
            );
        }
    }

    println!("\n=== aggregate, for contrast ===");
    let joined_truth: String = rows
        .iter()
        .map(|r| r.truth.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "  ours   {:.4}\n  theirs {:.4}",
        cer(
            &joined_truth,
            &rows
                .iter()
                .map(|r| r.our_text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        ),
        cer(
            &joined_truth,
            &rows
                .iter()
                .map(|r| r.their_text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        )
    );
    println!(
        "\nIf the sign test is flat and the top clips carry the net, the CER\n\
         difference is a handful of clips rather than a systematic character\n\
         weakness — and 'we are behind on CER' was never the right description."
    );
}

/// Word-level substitutions, as `truth->hyp` pairs. Coarser than a character
/// edit script and far more readable: a character diff of a whole utterance
/// is noise, while `SUPERFLUOUS->SUPER VOLUS` names the failure.
fn word_diffs(truth: &str, hyp: &str) -> Vec<String> {
    let t: Vec<&str> = truth.split_whitespace().collect();
    let h: Vec<&str> = hyp.split_whitespace().collect();
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < t.len() && j < h.len() {
        if t[i] == h[j] {
            i += 1;
            j += 1;
            continue;
        }
        // Look ahead a little to resynchronise after an insertion or deletion,
        // so one slip does not report the rest of the sentence as wrong.
        let resync = (1..4).find_map(|k| {
            if i + k < t.len() && t[i + k] == h[j] {
                Some((k, 0))
            } else if j + k < h.len() && t[i] == h[j + k] {
                Some((0, k))
            } else {
                None
            }
        });
        match resync {
            Some((di, 0)) if di > 0 => {
                out.push(format!("[-{}]", t[i..i + di].join(" ")));
                i += di;
            }
            Some((0, dj)) if dj > 0 => {
                out.push(format!("[+{}]", h[j..j + dj].join(" ")));
                j += dj;
            }
            _ => {
                out.push(format!("{}->{}", t[i], h[j]));
                i += 1;
                j += 1;
            }
        }
        if out.len() >= 6 {
            out.push("...".into());
            break;
        }
    }
    out
}
