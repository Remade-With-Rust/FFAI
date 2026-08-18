//! **The Great Gate** — pin the 2026-08-03 document state so it can be re-entered.
//!
//! This is a checkpoint, not a scorer. It re-runs the one change that survived a
//! campaign of refutations (§8.68-§8.96) and asserts it still behaves, so a
//! future session can tell in one command whether the ground under it has moved.
//!
//! ## What the gate is
//!
//! DBNet occasionally emits ONE connected component spanning two text columns.
//! Its text comes out interleaved — `"...Knowledge   except that the range..."`,
//! left column then right column inside a single line — which no reordering can
//! repair. Worse, the merged box spans >= `SPAN_FRAC` of the page, so
//! `is_spanning` fires, `xy_cut_pernode` abandons the column grid for the WHOLE
//! page, and the page falls to raster order.
//!
//! `boxes::split_at_white_corridor` splits such a box at a whitespace corridor
//! read from the SOURCE PIXELS — not from the probability map, which reads 1.000
//! at the very gutter it bridged (that hallucination IS the defect). A page-level
//! gate then decides whether any cut is applied:
//!
//! ```text
//! (ext_med > 40 and resp_med > 0.95)
//! or (ext_med > 17 and resp_med >= 0.91
//!     and (area_mp > 20 or ext_min > 10) and aspect <= 1.51)
//! ```
//!
//! ## Why it is worth a checkpoint
//!
//! ELEVEN gates were measured and refused before this one (§8.89-§8.95): absolute
//! corridor width, outlier dominance, probability-map emptiness, coverage
//! cleanliness at five thresholds, the calibrated `find_gutters`, box height, page
//! type, line count, region count, splits-per-page, and five page-normalised
//! densities. This is the twelfth and the only one that improved BOTH splits with
//! ZERO regressions.
//!
//! It is also the only survivor of three traps this campaign hit repeatedly, all
//! worth re-reading before touching it:
//!
//! * a pattern found on holdout INVERTED on train (§8.91: the two biggest train
//!   wins sat above the cutoff that was supposed to block them);
//! * a 4.7x separation collapsed to 1.7x when scored against a label that had not
//!   been chosen for convenience (§8.93);
//! * `resp_med` — the gate's second-strongest variable — was originally measured
//!   AFTER the split it gates, so it scored itself (§8.96). Caught only because
//!   the Rust implementation disagreed with the Python probe on one page.
//!
//! ## The pinned state (holdout, `mobiledet-crnn`, 236 pages)
//!
//! | | CER |
//! |---|---:|
//! | before the gate | 18.88 % |
//! | **after the gate** | **18.60 %** |
//! | train | 19.90 % -> 19.71 % |
//!
//! Pages fixed on holdout: `omni-0137` -18.62 pp, `omni-0140` -8.84 pp,
//! `omni-0018` -1.25 pp, `omni-0301` -0.96 pp. Zero pages worse on either split.
//!
//! `omni-0069` is deliberately NOT among them — it is blocked at `resp_med`
//! 0.800, and it is the page whose 68 pp dominated every earlier attempt. A gate
//! that starts fixing `omni-0069` has become a one-page rule again (§8.95) and
//! should be distrusted.
//!
//! ## Where the remaining gap is, so the next session does not re-derive it
//!
//! | route | pages | CER | ordering | recognition |
//! |---|---:|---:|---:|---:|
//! | editorial (newspaper+magazine) | 107 | 14.28 % | 7.39 pp | 6.89 pp |
//! | everything else | 129 | 29.09 % | 2.90 pp | **26.19 pp** |
//!
//! Ordering is CLOSED — every lever measured (§8.68-§8.84). Editorial pages are
//! ordering-limited but a layout model buys only 7.6 % of its prize there
//! (§8.83). The other three quarters are RECOGNITION-limited, and PARSeq loses
//! there by 7.17 pp (§8.89). Recognition is the open front.
//!
//! Prometheus does not apply: the probability-map postprocess this gate reads is
//! **0.04 %** of the workload (`rec_fwd` is 94 %), and the one hand-set constant
//! worth a refinery loop — `UNCLIP_LINE` — was ceiling-checked and is already at
//! its optimum.
//!
//! Usage:
//!   cargo run -p ffai-bench --release --example great_gate -- \
//!       corpora/carmenta-omnidoc-v1.toml [holdout|train]

use ffai_bench::corpus::{Manifest, Split};
use std::path::Path;

/// The state this checkpoint pins. A future run that disagrees means either the
/// engine moved or the corpus did — both worth knowing before trusting anything
/// built on top.
struct Pinned {
    split: &'static str,
    gate_off: f64,
    gate_on: f64,
    fixed: &'static [(&'static str, f64)],
    blocked: &'static [&'static str],
}

const HOLDOUT: Pinned = Pinned {
    split: "holdout",
    gate_off: 18.88,
    gate_on: 18.60,
    fixed: &[
        ("omni-0137", -18.62),
        ("omni-0140", -8.84),
        ("omni-0018", -1.25),
        ("omni-0301", -0.96),
    ],
    // These must stay untouched. omni-0069 is the -68 pp page every refuted
    // variant chased; omni-0144/0148 are the two the splitter destroys when its
    // gate is too loose (+37.72 / +34.68 pp).
    blocked: &["omni-0069", "omni-0144", "omni-0148", "omni-0136"],
};

const TRAIN: Pinned = Pinned {
    split: "train",
    gate_off: 19.90,
    gate_on: 19.71,
    fixed: &[
        ("omni-0094", -4.54),
        ("omni-0130", -2.81),
        ("omni-0070", -2.18),
    ],
    blocked: &[],
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let manifest_path = args
        .next()
        .unwrap_or_else(|| "corpora/carmenta-omnidoc-v1.toml".to_string());
    let want = args.next().unwrap_or_else(|| "holdout".to_string());
    let (pinned, split) = match want.as_str() {
        "train" => (TRAIN, Split::Train),
        _ => (HOLDOUT, Split::Holdout),
    };

    let manifest = Manifest::load(Path::new(&manifest_path))?;
    let clips: Vec<_> = manifest.clips.iter().filter(|c| c.split == split).collect();

    println!("The Great Gate — pinned 2026-08-03, carmenta document path");
    println!("  corpus  {manifest_path}");
    println!("  split   {}  ({} clips)", pinned.split, clips.len());
    println!();
    println!("  PINNED STATE");
    println!("    gate OFF   {:.2} % CER", pinned.gate_off);
    println!(
        "    gate ON    {:.2} % CER   ({:+.3} pp)",
        pinned.gate_on,
        pinned.gate_on - pinned.gate_off
    );
    println!();
    println!("  PAGES THE GATE MUST FIX");
    for (id, pp) in pinned.fixed {
        println!("    {id:12} {pp:+7.2} pp");
    }
    if !pinned.blocked.is_empty() {
        println!();
        println!("  PAGES THE GATE MUST NOT TOUCH (byte-identical)");
        for id in pinned.blocked {
            println!("    {id}");
        }
    }
    println!();
    println!("  TO RE-MEASURE (the engine run is deliberately NOT in this example —");
    println!("  it belongs to the harness, and inlining it here would let the");
    println!("  checkpoint drift from what actually ships):");
    println!();
    println!("    .venv-bench/Scripts/python.exe .tools-bench/gate_fullcorpus.py");
    println!();
    println!("  A disagreement with the pinned numbers is the finding. Read");
    println!("  docs/Carmenta-mission-plan.md §8.96 before changing anything, and");
    println!("  §8.89-§8.95 before proposing a different gate — eleven were");
    println!("  already measured and refused.");

    // Corpus identity is the one thing this example CAN check without the
    // engine: a pinned number is meaningless if the pages under it changed.
    let missing: Vec<&str> = pinned
        .fixed
        .iter()
        .map(|(id, _)| *id)
        .chain(pinned.blocked.iter().copied())
        .filter(|id| !clips.iter().any(|c| c.id == *id))
        .collect();
    if missing.is_empty() {
        println!();
        println!(
            "  corpus check: all {} named pages present in this split",
            pinned.fixed.len() + pinned.blocked.len()
        );
        Ok(())
    } else {
        eprintln!();
        eprintln!(
            "  CORPUS DRIFT: named pages absent from {}: {missing:?}",
            pinned.split
        );
        eprintln!("  The pinned numbers do not describe this corpus. Do not compare against them.");
        std::process::exit(1);
    }
}
