//! What does a SHORT-SIDE CEILING on the detector input cost, per corpus?
//!
//! `image::mobiledet_input` floors the short side at `min_side` (736) and caps
//! only the LONG side, at 4000. Nothing brings a large image DOWN to the
//! resolution the network actually works at, so a 3000x4000 phone photo reaches
//! the detector at 12 MP. `FFAI_DET_MAX_SHORT` / `image::set_det_max_short` is
//! the missing ceiling, and it is OFF by default because a default that changes
//! what a page reads is set by a corpus, not by a page.
//!
//! This is that corpus. The prior on this knob is genuinely two-sided:
//!
//! * A document page measured word-for-word identical at short-side 736 and
//!   ~16 % faster — the observation that opened the question.
//! * The aggressive form of the same idea (cap the LONG side at 960, which is
//!   `PaddleOCR`'s own `limit_type='max'` default) merged an entire RECEIPT
//!   into one 1903x1781 blob. That is why `mobiledet_input` reads `min_side`
//!   as a floor at all.
//!
//! Receipts are the risk case, so `carmenta-cord` is the corpus that decides
//! this, not `carmenta-doc`. Run both.
//!
//! Reported per cap: mean CER, its delta against the uncapped arm, the WORST
//! per-clip regression (a corpus mean hides one destroyed page — that is the
//! failure mode this knob has already produced once), and wall time. Splits
//! are kept apart: a default is chosen on TRAIN and confirmed on HOLDOUT.
//!
//! ```sh
//! cargo run --release -p ffai-carmenta --example det_scale_sweep -- \
//!     mobiledet-crnn corpora/carmenta-cord-v1.toml 0 736 960 1280
//! ```
//!
//! `0` means "ceiling off" and must be present: it is the arm every delta is
//! measured against.

#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use ffai_bench::corpus::{Manifest, Split};
use ffai_bench::metrics::cer;
use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage};
use ffai_core::engine::{OcrEngine, OcrOptions};

/// One clip's result under one cap.
struct Cell {
    cer: f64,
    ms: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let engine_name = args.next().expect("usage: det_scale_sweep <engine> <corpus.toml> <cap>...");
    let corpus = args.next().expect("usage: det_scale_sweep <engine> <corpus.toml> <cap>...");
    let caps: Vec<usize> = args.map(|a| a.parse().expect("cap must be a number")).collect();
    assert!(!caps.is_empty(), "give at least one cap; 0 (off) is the baseline arm");
    assert!(caps.contains(&0), "cap 0 (off) must be present — it is the baseline every delta uses");

    let (det, rec) = match engine_name.as_str() {
        "mobiledet-crnn" => (DetStage::MobileDet, RecStage::Crnn),
        "mobiledet-svtr" => (DetStage::MobileDet, RecStage::Svtr),
        "mobiledet-parseq" => (DetStage::MobileDet, RecStage::Parseq),
        // CRAFT does not read this knob — it has its own scaling policy — so
        // offering it here would silently produce four identical arms.
        other => panic!("`{other}` is not a mobile-det engine; this knob is `mobiledet_input`'s"),
    };
    let engine = CraftCrnn::new_mobiledet(rec);
    let _ = det;

    let manifest = Manifest::load(Path::new(&corpus)).expect("load corpus");
    assert_eq!(manifest.task, "ocr", "this sweep scores OCR text");
    let opts = OcrOptions::default();

    // Model load is not part of any arm (§6.1): warm it before the first timer.
    if let Some(first) = manifest.clips.first()
        && let Ok(img) = ffai_media::load_image(&manifest.clip_path(first))
    {
        let _ = engine.recognize(&img, &opts);
    }

    // clip-major, cap-minor: the arms for one clip run back to back, so a
    // machine that slows down halfway through slows every arm equally rather
    // than handing the tail arm a worse number.
    let mut cells: HashMap<usize, Vec<(String, Split, Cell)>> =
        caps.iter().map(|&c| (c, Vec::new())).collect();
    let total = manifest.clips.len();
    for (i, clip) in manifest.clips.iter().enumerate() {
        let Ok(img) = ffai_media::load_image(&manifest.clip_path(clip)) else {
            eprintln!("  skip {} — cannot decode", clip.id);
            continue;
        };
        let Ok(Some(truth)) = manifest.ground_truth(clip) else {
            eprintln!("  skip {} — no ground truth", clip.id);
            continue;
        };
        for &cap in &caps {
            ffai_carmenta::image::set_det_max_short(cap);
            let t0 = Instant::now();
            let out = engine.recognize(&img, &opts);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let text = out.map(|o| o.text()).unwrap_or_default();
            cells.get_mut(&cap).unwrap().push((
                clip.id.clone(),
                clip.split,
                Cell { cer: cer(&truth, &text), ms },
            ));
        }
        eprint!("\r  {}/{total} clips", i + 1);
    }
    eprintln!();
    ffai_carmenta::image::set_det_max_short(0);

    let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
    println!("\ncorpus: {} ({} clips)   engine: {engine_name}", manifest.name, total);

    for split in [Split::Train, Split::Holdout] {
        let label = if split == Split::Train { "TRAIN" } else { "HOLDOUT" };
        let base: Vec<(String, f64)> = cells[&0]
            .iter()
            .filter(|(_, s, _)| *s == split)
            .map(|(id, _, c)| (id.clone(), c.cer))
            .collect();
        if base.is_empty() {
            continue;
        }
        println!(
            "\n  {label} ({} clips)\n  {:>8} {:>10} {:>10} {:>12} {:>26} {:>10}",
            base.len(), "cap", "mean CER", "delta", "worse clips", "worst regression", "ms/page",
        );
        for &cap in &caps {
            let arm: HashMap<&str, &Cell> = cells[&cap]
                .iter()
                .filter(|(_, s, _)| *s == split)
                .map(|(id, _, c)| (id.as_str(), c))
                .collect();
            let cers: Vec<f64> = base.iter().map(|(id, _)| arm[id.as_str()].cer).collect();
            let times: Vec<f64> = base.iter().map(|(id, _)| arm[id.as_str()].ms).collect();
            let base_mean = mean(&base.iter().map(|(_, c)| *c).collect::<Vec<_>>());
            // The corpus mean is not the gate. ONE destroyed page is what this
            // knob has produced before, and a 60-clip mean absorbs it.
            let mut worst = ("", 0.0f64);
            let mut worse = 0usize;
            for (id, b) in &base {
                let d = arm[id.as_str()].cer - b;
                if d > 1e-9 {
                    worse += 1;
                }
                if d > worst.1 {
                    worst = (id.as_str(), d);
                }
            }
            let name = if cap == 0 { "off".to_string() } else { cap.to_string() };
            println!(
                "  {name:>8} {:>10.4} {:>+10.4} {:>12} {:>26} {:>10.0}",
                mean(&cers),
                mean(&cers) - base_mean,
                format!("{worse}/{}", base.len()),
                if worst.0.is_empty() { "none".to_string() } else { format!("{} {:+.4}", worst.0, worst.1) },
                mean(&times),
            );
        }
    }
}
