//! Is the im2col buffer DRAM-resident or L2-resident?
//!
//! Ceiling probe for implicit GEMM, run BEFORE building it — because the two
//! preceding refutations in this campaign were both the same mistake: an
//! L2-resident buffer priced at DRAM bandwidth, producing a prize that did
//! not exist. Eliminating im2col's materialisation is only worth something if
//! its bytes actually go to DRAM and come back.
//!
//! The mean buffer is 380 KiB (216.9 MiB over 585 calls), which would sit in
//! L2 — but the distribution is skewed, so the mean is the wrong statistic.
//! This bins the BYTES by the size of the buffer carrying them.
use ffai_core::engine::{DetectEngine, DetectOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let engine = ffai_diana::engine::Yolo26::build(&tier, ffai_diana::image::Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    engine.detect(&img, &opts)?;
    let _ = ffai_diana::conv3x3::take_size_hist();
    engine.detect(&img, &opts)?;
    let hist = ffai_diana::conv3x3::take_size_hist();

    let total: u64 = hist.iter().map(|(_, v)| v).sum();
    println!("tier {tier}: im2col bytes by BUFFER SIZE (one image)");
    // This box: 2 MiB L2 per core, 32 MiB shared L3. A buffer under L2 never
    // reaches DRAM on the GEMM's read-back; one over L3 certainly does.
    const L2: u64 = 2 * 1024 * 1024;
    const L3: u64 = 32 * 1024 * 1024;
    let (mut in_l2, mut in_l3, mut over) = (0u64, 0u64, 0u64);
    for (b, v) in &hist {
        let lo = 1u64 << b;
        let pct = *v as f64 / total as f64 * 100.0;
        println!("  {:>7} .. {:>7}: {:>7.1} MiB  {pct:5.1}%",
                 human(lo), human(lo * 2), *v as f64 / 1048576.0);
        if lo < L2 { in_l2 += v } else if lo < L3 { in_l3 += v } else { over += v }
    }
    println!("  ---");
    println!("  under L2 (2 MiB):  {:>7.1} MiB  {:5.1}%  — never reaches DRAM", in_l2 as f64 / 1048576.0, in_l2 as f64 / total as f64 * 100.0);
    println!("  L2..L3 (32 MiB):   {:>7.1} MiB  {:5.1}%  — L3, ~3x DRAM bandwidth", in_l3 as f64 / 1048576.0, in_l3 as f64 / total as f64 * 100.0);
    println!("  over L3:           {:>7.1} MiB  {:5.1}%  — real DRAM traffic", over as f64 / 1048576.0, over as f64 / total as f64 * 100.0);
    println!("  PRIZE for implicit GEMM is bounded by the last two rows, not the total.");
    Ok(())
}

fn human(b: u64) -> String {
    if b >= 1048576 { format!("{} MiB", b / 1048576) } else { format!("{} KiB", b / 1024) }
}
