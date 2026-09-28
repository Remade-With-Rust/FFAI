//! Orientation bench (docs/plans/commercial-gaps.md, gap 5): turn upright
//! pages by a KNOWN quarter turn and check `detect_orientation` returns the
//! turn that undoes it. Ground truth by construction, no labelling.
//!
//! ```text
//! cargo run --release -p ffai-carmenta --example orient_bench -- \
//!     [--engine mobiledet-svtr|mobiledet-crnn] [--pages 24] [--forms 12]
//! ```
//!
//! Sources: OmniDocBench pages (dense documents) and `carmenta-form-v1`
//! control clips (a single sparse form line, the hard case for an axis vote).
//! Also prints the cost of a decision against one full-page read.

use std::path::PathBuf;
use std::time::Instant;

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage};
use ffai_carmenta::orient::rotate;
use ffai_core::engine::{OcrEngine, OcrOptions};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let engine = get("--engine").unwrap_or_else(|| "mobiledet-svtr".into());
    let n_pages: usize = get("--pages").and_then(|v| v.parse().ok()).unwrap_or(24);
    let n_forms: usize = get("--forms").and_then(|v| v.parse().ok()).unwrap_or(12);
    let rec = match engine.as_str() {
        "mobiledet-svtr" => RecStage::Svtr,
        "mobiledet-crnn" => RecStage::Crnn,
        other => panic!("unsupported engine {other}"),
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let e = CraftCrnn::variant_in(rec, DetStage::MobileDet, &root.join("models"));
    e.preload().expect("weights");

    let list = |dir: &str, filter: &dyn Fn(&str) -> bool, n: usize| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(root.join(dir))
            .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        v.retain(|p| {
            p.extension().is_some_and(|x| x == "png")
                && filter(p.file_name().unwrap().to_str().unwrap())
        });
        v.sort();
        v.truncate(n);
        v
    };
    let sources = [
        (
            "omnidoc",
            list("corpora/clips/carmenta-omnidoc", &|_| true, n_pages),
        ),
        (
            "form",
            list(
                "corpora/clips/carmenta-form",
                &|f| f.contains("-none-gray"),
                n_forms,
            ),
        ),
    ];

    println!("engine {engine}");
    println!(
        "{:<8} {:>5} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "source", "n", "t=0", "t=90", "t=180", "t=270", "margin"
    );
    for (name, pages) in &sources {
        let mut correct = [0usize; 4];
        let mut margins = Vec::new();
        let (mut t_orient, mut t_full, mut reads) = (0f64, 0f64, 0usize);
        for p in pages {
            let upright = ffai_media::load_image(p).unwrap();
            let t0 = Instant::now();
            let _ = e.recognize(&upright, &OcrOptions::default()).unwrap();
            t_full += t0.elapsed().as_secs_f64();
            for t in 0..4u8 {
                let turned = rotate(&upright, t);
                let t1 = Instant::now();
                let o = e.detect_orientation(&turned).unwrap();
                t_orient += t1.elapsed().as_secs_f64();
                reads += 1;
                let want = (4 - t) % 4;
                if o.turns == want {
                    correct[t as usize] += 1;
                } else {
                    eprintln!(
                        "MISS {} turned {}: got {} (h {:.0} v {:.0} conf {:.3} vs {:.3}, n {})",
                        p.file_name().unwrap().to_string_lossy(),
                        u32::from(t) * 90,
                        u32::from(o.turns) * 90,
                        o.horizontal_area,
                        o.vertical_area,
                        o.chosen_confidence,
                        o.other_confidence,
                        o.sampled
                    );
                }
                margins.push(o.chosen_confidence - o.other_confidence);
            }
        }
        let n = pages.len();
        margins.sort_by(f32::total_cmp);
        println!(
            "{name:<8} {n:>5} {:>7}/{n} {:>6}/{n} {:>6}/{n} {:>6}/{n} {:>9.3}",
            correct[0],
            correct[1],
            correct[2],
            correct[3],
            margins.first().copied().unwrap_or(0.0)
        );
        println!(
            "         decision {:.2} s vs full read {:.2} s ({:.0} % of a read)",
            t_orient / reads.max(1) as f64,
            t_full / n.max(1) as f64,
            100.0 * (t_orient / reads.max(1) as f64) / (t_full / n.max(1) as f64).max(1e-9)
        );
    }
    println!("\nmargin = smallest (chosen - other) confidence gap across all decisions");
}
