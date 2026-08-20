//! Run PP-DocLayout-S over a LIST of pages and print regions as JSONL.
//!
//! The §50 routing-guard sweep needs per-region (class, score, bbox) for every
//! routed page so absorption-based guards can be swept OFFLINE against the
//! banked per-page evaluator deltas — without re-running OCR. `layout_probe`
//! loads the model per invocation; this loads it once.
//!
//! usage: layout_batch <image-dir> <stems-file> [score_thr]
//! stdout: one JSON object per page: {"page":..., "regions":[[label,score,x0,y0,x1,y1],...]}
use ffai_carmenta::doclayout::DocLayout;
use std::io::Write;

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("usage: layout_batch <image-dir> <stems-file> [thr]");
    let stems = a.next().expect("stems file");
    let thr: f32 = a.next().and_then(|v| v.parse().ok()).unwrap_or(0.45);

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap();
    let m = DocLayout::new(
        &root.join("corpora/refs/fixtures/doclayout_s_arch.json"),
        &root.join("corpora/refs/fixtures/doclayout_s.safetensors"),
    ).expect("load layout model");

    let out = std::io::stdout();
    let mut out = out.lock();
    for stem in std::fs::read_to_string(&stems).expect("read stems").lines() {
        let stem = stem.trim();
        if stem.is_empty() { continue; }
        // OmniDocBench images carry mixed extensions.
        let img = ["png", "jpg", "jpeg"].iter().find_map(|e| {
            let p = std::path::Path::new(&dir).join(format!("{stem}.{e}"));
            p.exists().then(|| ffai_media::load_image(&p).ok()).flatten()
        });
        let Some(img) = img else {
            eprintln!("MISSING {stem}");
            continue;
        };
        match m.detect(&img, thr, 0.40) {
            Ok(regions) => {
                let rs: Vec<String> = regions.iter().map(|r| format!(
                    "[\"{}\",{:.4},{:.1},{:.1},{:.1},{:.1}]",
                    r.label(), r.score, r.x0, r.y0, r.x1, r.y1)).collect();
                writeln!(out, "{{\"page\":\"{stem}\",\"w\":{},\"h\":{},\"regions\":[{}]}}",
                    img.width, img.height, rs.join(",")).unwrap();
            }
            Err(e) => eprintln!("FAIL {stem}: {e}"),
        }
    }
}
