//! Run PP-DocLayout-S on one page and print the regions it finds.
//!
//! The first validation gate for the layout stage: does a ported detector find
//! the regions OmniDocBench annotates? Bit-exactness against paddle is not
//! available on this box (no paddle/torch/onnx installed), so the oracle is the
//! benchmark's own layout annotations — a functional gate, stated as such.
use ffai_carmenta::doclayout::DocLayout;

fn main() {
    let mut a = std::env::args().skip(1);
    let img_path = a.next().expect("usage: layout_probe <image> [score_thr]");
    let thr: f32 = a.next().and_then(|v| v.parse().ok()).unwrap_or(0.4);

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap();
    let arch = root.join("corpora/refs/fixtures/doclayout_s_arch.json");
    let w = root.join("corpora/refs/fixtures/doclayout_s.safetensors");

    let t0 = std::time::Instant::now();
    let m = DocLayout::new(&arch, &w).expect("load layout model");
    let load = t0.elapsed();

    let img = ffai_media::load_image(std::path::Path::new(&img_path)).expect("load image");
    let t1 = std::time::Instant::now();
    let regions = m.detect(&img, thr, 0.5).expect("detect");
    println!("load {:?}  infer {:?}  {}x{}", load, t1.elapsed(), img.width, img.height);
    println!("{} regions at score>={thr}", regions.len());
    for r in &regions {
        println!(
            "  {:<16} {:.3}  [{:.0},{:.0} {:.0}x{:.0}]",
            r.label(), r.score, r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0
        );
    }
}
