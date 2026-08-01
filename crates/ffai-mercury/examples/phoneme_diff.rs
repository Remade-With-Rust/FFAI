//! How far is our clean-room G2P from espeak, and on WHICH words?
//!
//! Float noise in the kernels is worth ~1 word over the corpus, so it is not
//! where quality lives. The phonemizer is: it is a clean-room CMUdict
//! implementation standing in for espeak (which piper embeds, and which is why
//! piper is GPL), it is essentially free at runtime, and any systematic
//! mismatch is a systematic quality cost.
//!
//! The substitution gate already priced the phonemizer END TO END by feeding
//! our phonemes through piper's runtime. This asks the cheaper upstream
//! question first: how often do the two disagree at all, and on what? A
//! symbol-level diff costs nothing and decides whether a judged experiment is
//! worth building.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example phoneme_diff
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::Vits;

const PRIMARY: char = 'ˈ';
const SECONDARY: char = 'ˌ';

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

    // espeak's own output, dumped once by corpora/refs/dump_espeak_phonemes.py
    // and pinned in the repo — the oracle, used out of process, never linked.
    let fixture = std::fs::read_to_string("corpora/fixtures/harvard-espeak-phonemes-v1.jsonl")?;
    let mut rows: Vec<(String, String, Vec<String>)> = Vec::new();
    for line in fixture.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let id = v["id"].as_str().unwrap_or("?").to_string();
        let text = v["text"].as_str().unwrap_or("").to_string();
        let ph: Vec<String> = v["phonemes"][0]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        rows.push((id, text, ph));
    }
    println!("{} sentences from the espeak fixture\n", rows.len());

    let (mut exact, mut total_sym, mut diff_sym) = (0usize, 0usize, 0usize);
    // Which SYMBOL substitutions dominate, and which words carry them.
    let mut subs: HashMap<(String, String), usize> = HashMap::new();
    let mut bad_words: HashMap<String, usize> = HashMap::new();

    for (_id, text, esp) in &rows {
        let ours_str = phonemizer.phonemize(text)?;
        let ours: Vec<String> = ours_str.chars().map(|c| c.to_string()).collect();
        let _ = vits.id_map.sentence_to_ids(&ours_str);

        if ours == *esp {
            exact += 1;
        }
        // Word-level: split both on the space symbol so a mismatch can be
        // attributed to the word that produced it rather than to an offset.
        let ow: Vec<Vec<String>> =
            ours.split(|s| s == " ").map(|w| w.to_vec()).filter(|w| !w.is_empty()).collect();
        let ew: Vec<Vec<String>> =
            esp.split(|s| s == " ").map(|w| w.to_vec()).filter(|w| !w.is_empty()).collect();
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, (a, b)) in ow.iter().zip(&ew).enumerate() {
            total_sym += b.len();
            if a != b {
                diff_sym += a.len().max(b.len()) - a.iter().zip(b).filter(|(x, y)| x == y).count();
                if let Some(w) = words.get(i) {
                    *bad_words.entry(w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                        .or_insert(0) += 1;
                }
                for (x, y) in a.iter().zip(b) {
                    if x != y {
                        *subs.entry((y.clone(), x.clone())).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    println!(
        "  sentences matching espeak EXACTLY: {exact}/{}  ({:.1} %)",
        rows.len(),
        100.0 * exact as f64 / rows.len() as f64
    );
    println!(
        "  symbol disagreement: {diff_sym}/{total_sym}  ({:.2} %)",
        100.0 * diff_sym as f64 / total_sym.max(1) as f64
    );

    // The substitution table below ZIPS positionally, so a symbol we DROP
    // (espeak's secondary stress, per phonemize.rs's documented rule) shifts
    // every later position and is reported as a substitution it is not.
    // Print the offending words verbatim instead -- no alignment guessing.
    println!("
  mismatched words, espeak vs ours:");
    let mut shown = 0usize;
    for (_id, text, esp) in &rows {
        if shown >= 24 { break; }
        let ours_str = phonemizer.phonemize(text)?;
        let ours: Vec<String> = ours_str.chars().map(|c| c.to_string()).collect();
        if ours == *esp { continue; }
        let ow: Vec<String> =
            ours.split(|s| s == " ").map(|w| w.concat()).filter(|w| !w.is_empty()).collect();
        let ew: Vec<String> =
            esp.split(|s| s == " ").map(|w| w.concat()).filter(|w| !w.is_empty()).collect();
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, (a, b)) in ow.iter().zip(&ew).enumerate() {
            if a != b && shown < 24 {
                let w = words.get(i).copied().unwrap_or("?");
                let strip = |z: &String| {
                    z.chars().filter(|c| *c != PRIMARY && *c != SECONDARY).collect::<String>()
                };
                let tag = if strip(a) == strip(b) { "[STRESS ONLY]" } else { "" };
                println!("    {w:<12} espeak {b:<24} ours {a:<24} {tag}");
                shown += 1;
            }
        }
    }

    let mut sv: Vec<_> = subs.into_iter().collect();
    sv.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("\n  top substitutions (espeak -> ours):");
    for ((e, o), n) in sv.iter().take(14) {
        println!("    {e:<4} -> {o:<4}  {n}");
    }

    let mut wv: Vec<_> = bad_words.into_iter().collect();
    wv.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("\n  words most often mismatched:");
    for (w, n) in wv.iter().take(20) {
        println!("    {w:<16} {n}");
    }
    Ok(())
}
