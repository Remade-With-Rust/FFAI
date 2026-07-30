//! M-T1 phoneme oracle: Mercury's pure-Rust G2P against espeak-ng's pinned
//! output on the Harvard corpus (docs/mercury-tts-mission.md §6.2).
//!
//! ```text
//! cargo run -p ffai-mercury --example phoneme_oracle -- --split train --show 15
//! cargo run -p ffai-mercury --example phoneme_oracle -- --split holdout   # END of tuning only
//! ```
//!
//! Reports per split: sentence exact-match rate and mean phoneme error rate
//! (character-level Levenshtein against the fixture IPA, no normalization —
//! stress marks, length marks and spacing all count). Tuning iterates on
//! `--split train`; holdout is read once when the mapping is frozen, because
//! a holdout read per iteration is how test-clean became a tuning set
//! (finished plan §6.23).

use std::collections::HashMap;
use std::path::Path;

use ffai_bench::metrics::cer_with;
use ffai_bench::normalize::Mode;
use ffai_mercury::tts::phonemize::Phonemizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let split = get("--split", "train");
    let show: usize = get("--show", "10").parse()?;
    // `--fixtures` selects an alternate fixture file (e.g. the TUNE set from
    // Harvard lists 21-40, disjoint from the corpus); pair it with
    // `--split all`, since tune sentences are not in the manifest.
    let fixtures_path = get("--fixtures", "corpora/fixtures/harvard-espeak-phonemes-v1.jsonl");

    let manifest = ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    let want_split = |id: &str| -> bool {
        let clip = manifest.clips.iter().find(|c| c.id == id);
        match (split.as_str(), clip) {
            ("all", _) => true,
            ("train", Some(c)) => c.split == ffai_bench::corpus::Split::Train,
            ("holdout", Some(c)) => c.split == ffai_bench::corpus::Split::Holdout,
            _ => false,
        }
    };

    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

    let fixtures = std::fs::read_to_string(&fixtures_path)?;
    let mut exact = 0usize;
    let mut total = 0usize;
    let mut pers: Vec<f64> = Vec::new();
    let mut mismatches: Vec<(String, f64, String, String, String)> = Vec::new();

    for line in fixtures.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let id = v["id"].as_str().unwrap_or_default().to_string();
        if !want_split(&id) {
            continue;
        }
        let text = v["text"].as_str().unwrap_or_default().to_string();
        let espeak: String = v["phonemes"][0]
            .as_array()
            .map(|a| a.iter().filter_map(|p| p.as_str()).collect())
            .unwrap_or_default();

        let ours = phonemizer.phonemize(&text)?;
        total += 1;
        // Raw characters, no normalizer: stress and length marks are the
        // contract here, not noise.
        let per = cer_with(&espeak, &ours, Mode::None);
        pers.push(per);
        if ours == espeak {
            exact += 1;
        } else {
            mismatches.push((id, per, text, espeak, ours));
        }
    }

    mismatches.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mean_per = pers.iter().sum::<f64>() / pers.len().max(1) as f64;

    println!("split: {split}  ({total} sentences)");
    println!("sentence exact-match: {exact}/{total} ({:.1} %)", 100.0 * exact as f64 / total.max(1) as f64);
    println!("mean phoneme error rate (char-level): {:.2} %", 100.0 * mean_per);

    // Divergence census: which character substitutions dominate. This names
    // the next rule to fix, instead of eyeballing diffs.
    let mut subs: HashMap<(char, char), usize> = HashMap::new();
    for (_, _, _, espeak, ours) in &mismatches {
        for (e, o) in align_chars(espeak, ours) {
            if e != o {
                *subs.entry((e, o)).or_default() += 1;
            }
        }
    }
    let mut census: Vec<_> = subs.into_iter().collect();
    census.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    if !census.is_empty() {
        println!("\ntop divergences (espeak -> ours, aligned):");
        for ((e, o), n) in census.iter().take(12) {
            println!("  {} -> {}   x{n}", printable(*e), printable(*o));
        }
    }

    println!("\nworst {} mismatches:", show.min(mismatches.len()));
    for (id, per, text, espeak, ours) in mismatches.iter().take(show) {
        println!("  {id}  PER {:.1} %\n    txt: {text}\n    esp: {espeak}\n    our: {ours}", per * 100.0);
    }
    Ok(())
}

fn printable(c: char) -> String {
    match c {
        ' ' => "␣".to_string(),
        '\u{0}' => "∅".to_string(),
        c => c.to_string(),
    }
}

/// Minimal alignment for the census: pair characters via LCS-free greedy
/// diagonal (good enough to count systematic substitutions; insertions and
/// deletions pair against ∅).
fn align_chars(a: &str, b: &str) -> Vec<(char, char)> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    // Full Levenshtein backtrace, small strings — exact, not greedy.
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = (dp[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]))
                .min(dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1);
        }
    }
    let mut out = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]) {
            out.push((a[i - 1], b[j - 1]));
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            out.push((a[i - 1], '\u{0}'));
            i -= 1;
        } else {
            out.push(('\u{0}', b[j - 1]));
            j -= 1;
        }
    }
    out.reverse();
    out
}
