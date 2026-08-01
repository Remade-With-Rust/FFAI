//! Score a detections JSONL dump against a detect corpus with the proxy mAP
//! scorer — the standalone entry point `tools/diana_validate_scorer.py` uses
//! to cross-validate `ffai_bench::detect` against pycocotools on identical
//! inputs before any number the scorer produces goes on the board.
//!
//! Usage:
//!   cargo run -p ffai-bench --example score_detect -- \
//!       corpora/diana-coco-v1.toml dets.jsonl
//!
//! The JSONL is the adapter wire format: one `{"path", "text"}` object per
//! image, `text` carrying `[[x0,y0,x1,y1,cls,conf], ...]`. Only holdout
//! clips are scored, matching `run_detect`.

use ffai_bench::corpus::Manifest;
use ffai_bench::detect::{parse_detections, parse_ground_truth, MapAccumulator};
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let manifest_path = args.next().expect("args: <corpus.toml> <dets.jsonl>");
    let jsonl_path = args.next().expect("args: <corpus.toml> <dets.jsonl>");

    let manifest = Manifest::load(Path::new(&manifest_path))?;
    let mut by_path: HashMap<String, String> = HashMap::new();
    for line in std::fs::read_to_string(&jsonl_path)?.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        if let (Some(path), Some(text)) =
            (v.get("path").and_then(|p| p.as_str()), v.get("text").and_then(|t| t.as_str()))
        {
            by_path.insert(path.replace('\\', "/").to_lowercase(), text.to_string());
        }
    }

    let mut acc = MapAccumulator::new();
    let mut scored = 0usize;
    for clip in manifest.holdout() {
        let clip_path = manifest.clip_path(clip);
        let key = clip_path.to_string_lossy().replace('\\', "/").to_lowercase();
        // Adapter paths may be absolute and may differ in drive-letter case
        // (Windows), so match exact-then-suffix over case-folded keys. A clip
        // that matches nothing is reported, never silently skipped — a scorer
        // that quietly scores 0 images is the failure mode this whole file
        // exists to catch.
        let Some(payload) =
            by_path.get(&key).or_else(|| {
                by_path.iter().find(|(k, _)| k.ends_with(&key)).map(|(_, v)| v)
            })
        else {
            eprintln!("no detections for {key}");
            continue;
        };
        let truth = manifest
            .ground_truth(clip)?
            .ok_or_else(|| format!("clip {} has no ground truth", clip.id))?;
        acc.add_image(&parse_ground_truth(&truth)?, &parse_detections(payload)?);
        scored += 1;
    }

    if scored == 0 {
        return Err("scored 0 images — no detection payload matched any holdout clip".into());
    }
    println!(
        "scored {scored} holdout images: mAP50 {:.4}  mAP50-95 {:.4}",
        acc.map50().unwrap_or(f64::NAN),
        acc.map5095().unwrap_or(f64::NAN)
    );
    Ok(())
}
