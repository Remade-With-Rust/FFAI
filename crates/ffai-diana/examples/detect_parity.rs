//! Corpus-wide DETECTION parity against the PyTorch reference.
//!
//! # What "parity with YOLO" can and cannot mean
//!
//! Not bit-identical floats. Float addition is not associative, every GEMM
//! reorders accumulation, and PyTorch's own oneDNN kernels change with
//! shape, thread count and ISA — PyTorch does not reproduce its own bits
//! across `OMP_NUM_THREADS`. Bit-exactness would require replicating the
//! reference's exact scalar op order with no FMA and no GEMM, at roughly a
//! 20x speed cost.
//!
//! What IS meaningful, and what this measures: **does the engine emit the
//! same DETECTIONS as PyTorch** — the same count, the same classes, and
//! boxes agreeing to sub-pixel — on every holdout image.
//!
//! # Why this exists when mAP already matches to six decimals
//!
//! mAP is an aggregate, and "a metric can be blind to the change you are
//! actively making". Two different detection sets can produce the same mAP
//! by compensating errors — a box that drifts one way on one image and the
//! other way elsewhere nets out. This compares detection to detection, so
//! there is nowhere for a compensating error to hide.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example detect_parity -- \
//!     corpora/diana-coco-v2.toml ref_pt.jsonl [conf]
//! ```

use std::collections::HashMap;

use ffai_bench::corpus::Manifest;
use ffai_core::engine::{DetectEngine, DetectOptions};

#[derive(Debug, Clone, Copy)]
struct Det {
    b: [f32; 4],
    class: u32,
    conf: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let manifest_path = args.next().unwrap_or_else(|| "corpora/diana-coco-v2.toml".into());
    let ref_path = args.next().expect("usage: detect_parity <corpus.toml> <ref.jsonl> [conf]");
    // Compare at a confidence where detections are STABLE. Below ~0.1 the
    // reference's own ordering of near-tied junk is arbitrary, so a
    // mismatch there measures tie-breaking, not correctness.
    let conf: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.25);
    // Which tier. This was hardcoded to n, so the strongest claim this repo
    // makes about Diana — that every DETECTION matches, not merely the mAP —
    // had only ever been checked on the smallest of five models. mAP and the
    // per-layer oracles gate the others; box-for-box did not.
    let tier = args.next().unwrap_or_else(|| "n".into());

    let manifest = Manifest::load(std::path::Path::new(&manifest_path))?;

    // The reference dump, keyed by file stem (paths differ by absoluteness).
    let mut reference: HashMap<String, Vec<Det>> = HashMap::new();
    for line in std::fs::read_to_string(&ref_path)?.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let (Some(p), Some(t)) =
            (v.get("path").and_then(|p| p.as_str()), v.get("text").and_then(|t| t.as_str()))
        else {
            continue;
        };
        let stem = std::path::Path::new(p)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let rows: Vec<Vec<f64>> = serde_json::from_str(t)?;
        reference.insert(
            stem,
            rows.iter()
                .map(|r| Det {
                    b: [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32],
                    class: r[4] as u32,
                    conf: r[5] as f32,
                })
                .collect(),
        );
    }

    let engine = ffai_diana::engine::Yolo26::build(
        &tier,
        ffai_diana::image::Geometry::Rect,
        "models",
    );
    println!("tier {tier}, conf {conf}");
    let opts = DetectOptions { confidence: 0.001, max_detections: 100, ..Default::default() };

    let (mut images, mut total, mut count_mismatch, mut class_mismatch) = (0, 0usize, 0, 0);
    let (mut worst_box, mut worst_conf) = (0f32, 0f32);
    let mut worst_image = String::new();

    for clip in manifest.holdout() {
        let path = manifest.clip_path(clip);
        let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let Some(want_all) = reference.get(&stem) else {
            eprintln!("no reference row for {stem}");
            continue;
        };
        let image = ffai_media::load_image(&path)?;
        let got_all = engine.detect(&image, &opts)?;

        let want: Vec<&Det> = want_all.iter().filter(|d| d.conf >= conf).collect();
        let got: Vec<_> = got_all.detections.iter().filter(|d| d.confidence >= conf).collect();
        images += 1;

        if want.len() != got.len() {
            count_mismatch += 1;
            println!("{stem}: COUNT {} ours vs {} reference", got.len(), want.len());
            continue;
        }
        // Both sides are confidence-ordered, so compare in order.
        for (g, w) in got.iter().zip(&want) {
            total += 1;
            if g.class_id != w.class {
                class_mismatch += 1;
                println!("{stem}: CLASS {} ours vs {} reference", g.class_id, w.class);
            }
            let d = [
                (g.x0 - w.b[0]).abs(),
                (g.y0 - w.b[1]).abs(),
                (g.x1 - w.b[2]).abs(),
                (g.y1 - w.b[3]).abs(),
            ]
            .into_iter()
            .fold(0.0f32, f32::max);
            if d > worst_box {
                worst_box = d;
                worst_image = stem.clone();
            }
            worst_conf = worst_conf.max((g.confidence - w.conf).abs());
        }
    }

    println!("\ncorpus detection parity vs PyTorch (Ultralytics), conf >= {conf}");
    println!("{:<18} {}", "images", images);
    println!("{:<18} {total}", "detections");
    println!(
        "{:<18} {}",
        "count agreement",
        if count_mismatch == 0 {
            "EXACT on every image".to_string()
        } else {
            format!("{count_mismatch} image(s) DIFFER")
        }
    );
    println!(
        "{:<18} {}",
        "class agreement",
        if class_mismatch == 0 {
            format!("{total}/{total} EXACT")
        } else {
            format!("{class_mismatch} MISMATCH")
        }
    );
    println!("{:<18} {worst_box:.4} px  (on {worst_image})", "max box delta");
    println!("{:<18} {worst_conf:.3e}", "max conf delta");

    // Two DIFFERENT claims, reported separately because they have different
    // strengths and different failure meanings.
    //
    // STRUCTURAL parity — same detections, same classes, same order — is an
    // exact, discrete property. Any mismatch is a real behavioural
    // divergence and there is no tolerance to hide behind.
    //
    // GEOMETRIC parity is a float tolerance and can never be exact, for the
    // reason in the module docs. The bound is 1 px on a 640 px letterbox
    // (~0.15%), which is far below the ~1 px quantization of any box a
    // detector is asked to draw, and an order of magnitude below the
    // ~5 px scale at which IoU-based scoring starts to care.
    const BOX_TOL_PX: f32 = 1.0;
    let structural = count_mismatch == 0 && class_mismatch == 0;
    let geometric = worst_box < BOX_TOL_PX;
    println!(
        "\nstructural parity  {}  (count + class + order — exact, no tolerance)",
        if structural { "EXACT" } else { "FAILED" }
    );
    println!(
        "geometric parity   {}  (max {worst_box:.4} px vs {BOX_TOL_PX:.1} px bound)",
        if geometric { "PASS" } else { "FAILED" }
    );
    if !(structural && geometric) {
        std::process::exit(1);
    }
    Ok(())
}
