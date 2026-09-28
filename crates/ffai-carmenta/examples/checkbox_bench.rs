//! Checkbox bench over `carmenta-checkbox-v1` (docs/plans/commercial-gaps.md
//! gaps 4 and 14): are the boxes found, is anything else taken for a box, and
//! is each box's marked/empty state right?
//!
//! ```text
//! python tools/carmenta_checkbox_corpus.py
//! cargo run --release -p ffai-carmenta --example checkbox_bench -- [--misses]
//! ```
//!
//! A found box matches a ground-truth box at IoU >= 0.5. Precision counts
//! found boxes that match nothing (letters, ideographs, table cells) as false
//! positives; recall counts ground-truth boxes nothing matched.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ffai_carmenta::checkbox::{CheckboxParams, find};
use ffai_core::types::BoundingBox;

fn iou(a: &BoundingBox, b: [f32; 4]) -> f32 {
    let (bx, by, bw, bh) = (b[0], b[1], b[2], b[3]);
    let ix = (a.x + a.width).min(bx + bw) - a.x.max(bx);
    let iy = (a.y + a.height).min(by + bh) - a.y.max(by);
    if ix <= 0.0 || iy <= 0.0 {
        return 0.0;
    }
    let inter = ix * iy;
    inter / (a.width * a.height + bw * bh - inter)
}

#[derive(Default)]
struct Tally {
    gt: usize,
    matched: usize,
    state_right: usize,
}

fn main() {
    let misses = std::env::args().any(|a| a == "--misses");
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpora/clips/carmenta-checkbox");
    let mut metas: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("corpus: run tools/carmenta_checkbox_corpus.py")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    metas.sort();

    let mut by_ink: BTreeMap<String, Tally> = BTreeMap::new();
    let mut by_mark: BTreeMap<String, Tally> = BTreeMap::new();
    let (mut found_total, mut false_pos) = (0usize, 0usize);
    for m in &metas {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(m).unwrap()).unwrap();
        let ink = meta["ink"].as_str().unwrap().to_string();
        let img = ffai_media::load_image(&m.with_extension("png")).unwrap();
        let found = find(&img, CheckboxParams::default());
        found_total += found.len();
        let gts: Vec<(BoundingBox, [f32; 4], bool, String)> = meta["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                let v: Vec<f32> = b["bbox"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect();
                let bb = BoundingBox {
                    x: v[0],
                    y: v[1],
                    width: v[2],
                    height: v[3],
                };
                let mark = b["mark"].as_str().unwrap().to_string();
                let glyph = b["glyph"].as_bool().unwrap_or(false);
                (
                    bb,
                    [v[0], v[1], v[2], v[3]],
                    b["checked"].as_bool().unwrap(),
                    format!("{mark}{}", if glyph { "/glyph" } else { "" }),
                )
            })
            .collect();
        let mut used = vec![false; found.len()];
        for (_, frame, checked, mark) in &gts {
            let best = found
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .map(|(i, f)| (i, iou(&f.bbox, *frame)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            for t in [
                by_ink.entry(ink.clone()).or_default(),
                by_mark.entry(mark.clone()).or_default(),
            ] {
                t.gt += 1;
            }
            match best {
                Some((i, v)) if v >= 0.5 => {
                    used[i] = true;
                    let right = found[i].checked == *checked;
                    for t in [
                        by_ink.get_mut(&ink).unwrap(),
                        by_mark.get_mut(mark).unwrap(),
                    ] {
                        t.matched += 1;
                        t.state_right += usize::from(right);
                    }
                    if misses && !right {
                        eprintln!(
                            "STATE {} {mark}: truth {checked}, read {} (fill {:.3})",
                            m.file_stem().unwrap().to_string_lossy(),
                            found[i].checked,
                            found[i].fill
                        );
                    }
                }
                _ => {
                    if misses {
                        eprintln!(
                            "MISSED {} {mark} at {frame:?}",
                            m.file_stem().unwrap().to_string_lossy()
                        );
                    }
                }
            }
        }
        for (i, f) in found.iter().enumerate() {
            if !used[i] {
                false_pos += 1;
                if misses {
                    eprintln!(
                        "FALSE {} at ({:.0},{:.0}) {:.0}x{:.0} checked {}",
                        m.file_stem().unwrap().to_string_lossy(),
                        f.bbox.x,
                        f.bbox.y,
                        f.bbox.width,
                        f.bbox.height,
                        f.checked
                    );
                }
            }
        }
    }
    let print = |title: &str, t: &BTreeMap<String, Tally>| {
        println!(
            "\n{title:<14} {:>5} {:>8} {:>12}",
            "boxes", "recall", "state right"
        );
        for (k, v) in t {
            println!(
                "{k:<14} {:>5} {:>7.1}% {:>11.1}%",
                v.gt,
                100.0 * v.matched as f32 / v.gt.max(1) as f32,
                100.0 * v.state_right as f32 / v.matched.max(1) as f32
            );
        }
    };
    print("ink", &by_ink);
    print("mark", &by_mark);
    println!(
        "\nfound {found_total}, false positives {false_pos} (precision {:.1}%)",
        100.0 * (found_total - false_pos) as f32 / found_total.max(1) as f32
    );
}
