//! Census for the full-width fold (`ffai_carmenta::fold`): on real
//! documents, how many lines would it change, what are they, and does page
//! CER against ground truth move?
//!
//! ```text
//! FFAI_NO_FOLD=1 cargo run --release -p ffai-carmenta --example fold_census -- [--pages 316]
//! ```
//!
//! The engine runs with the fold OFF (the env var), and the fold is applied
//! here, offline, to every line — so one pass yields both arms, exactly
//! paired. CER is our whitespace-collapsed edit distance over the page's
//! ground-truth text: a direction signal, not the official evaluator.

use std::path::PathBuf;

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage};
use ffai_carmenta::fold::fold_latin_fullwidth;
use ffai_core::engine::{OcrEngine, OcrOptions};

fn edit(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

fn norm(s: &str) -> Vec<char> {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .collect()
}

fn main() {
    assert!(
        std::env::var_os("FFAI_NO_FOLD").is_some(),
        "run with FFAI_NO_FOLD=1 so the engine's own fold is off and this census sees raw output"
    );
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = args
        .iter()
        .position(|a| a == "--pages")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(316);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let e = CraftCrnn::variant_in(RecStage::Svtr, DetStage::MobileDet, &root.join("models"));
    let dir = root.join("corpora/clips/carmenta-omnidoc");
    let mut pages: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    pages.sort();
    pages.truncate(n);

    let (mut lines_total, mut lines_changed, mut pages_changed) = (0usize, 0usize, 0usize);
    let (mut better, mut worse, mut same) = (0usize, 0usize, 0usize);
    let (mut edits_off, mut edits_on, mut chars) = (0usize, 0usize, 0usize);
    for p in &pages {
        let Ok(gt) = std::fs::read_to_string(p.with_extension("txt")) else {
            continue;
        };
        let img = ffai_media::load_image(p).unwrap();
        let out = e.recognize(&img, &OcrOptions::default()).unwrap();
        let raw: Vec<String> = out
            .blocks
            .iter()
            .flat_map(|b| &b.lines)
            .map(|l| l.text.clone())
            .collect();
        let mut changed_here = 0;
        let folded: Vec<String> = raw
            .iter()
            .map(|l| {
                lines_total += 1;
                fold_latin_fullwidth(l).map_or_else(
                    || l.clone(),
                    |f| {
                        changed_here += 1;
                        println!(
                            "{}: {l:?} -> {f:?}",
                            p.file_name().unwrap().to_string_lossy()
                        );
                        f
                    },
                )
            })
            .collect();
        lines_changed += changed_here;
        let g = norm(&gt);
        let off = edit(&norm(&raw.join("\n")), &g);
        let on = if changed_here > 0 {
            edit(&norm(&folded.join("\n")), &g)
        } else {
            off
        };
        edits_off += off;
        edits_on += on;
        chars += g.len();
        if changed_here > 0 {
            pages_changed += 1;
            match on.cmp(&off) {
                std::cmp::Ordering::Less => better += 1,
                std::cmp::Ordering::Greater => worse += 1,
                std::cmp::Ordering::Equal => same += 1,
            }
        }
    }
    println!(
        "\n{} pages, {lines_total} lines: fold fires on {lines_changed} lines across {pages_changed} pages",
        pages.len()
    );
    println!("changed pages: {better} better, {worse} worse, {same} unchanged in CER");
    println!(
        "corpus CER: fold off {:.4} %  fold on {:.4} %",
        100.0 * edits_off as f64 / chars.max(1) as f64,
        100.0 * edits_on as f64 / chars.max(1) as f64
    );
}
