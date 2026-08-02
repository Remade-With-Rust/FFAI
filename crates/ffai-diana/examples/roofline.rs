//! Does the reference beat us with a better KERNEL, or with a smaller
//! ALGORITHM?
//!
//! Every latency experiment this session assumed the first: same arithmetic,
//! theirs executed faster. That assumption was never tested, and it bounds
//! everything downstream — because if the reference issues FEWER MULTIPLIES
//! than we do, no kernel tuning can reach parity, and the entire search has
//! been in the wrong space.
//!
//! The test needs no timer. Count our multiply-accumulates, divide by the
//! reference's published per-image latency, and compare the implied
//! throughput against what this machine can physically retire. If the
//! reference would have to exceed its own hardware peak to do our arithmetic
//! in its measured time, it is not doing our arithmetic.
use ffai_core::engine::{DetectEngine, DetectOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let engine = ffai_diana::engine::Yolo26::build(&tier, ffai_diana::image::Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    engine.detect(&img, &opts)?;
    let _ = ffai_diana::conv3x3::take_macs();
    engine.detect(&img, &opts)?;
    let macs = ffai_diana::conv3x3::take_macs();
    let acts = ffai_diana::conv3x3::take_acts();

    // A MAC is two flops. This counts the 3x3 path only — the subset Winograd
    // would apply to, and the one that dominates.
    let gflop = macs as f64 * 2.0 / 1e9;
    println!("tier {tier}: 3x3 convolution issues {:.3} G MACs = {:.3} GFLOP per image", macs as f64 / 1e9, gflop);

    // The fusion prize. Each activation element is written by the GEMM, then
    // read+written by the bias broadcast, then read+written by SiLU: five
    // touches where a fused epilogue does one write.
    let ab = acts as f64 * 4.0;
    println!("  activations {:.1} M elements = {:.1} MiB", acts as f64 / 1e6, ab / 1048576.0);
    println!("  bias pass + silu pass = 4 extra touches = {:.1} MiB of traffic", ab * 4.0 / 1048576.0);
    // The number that refuted the fusion built on it. Per CONVOLUTION the
    // activation is ~430 KiB, which lives in L2 — so these touches are not
    // DRAM traffic and pricing them at DRAM bandwidth invents a prize.
    println!("  BUT per-conv that is ~{:.0} KiB, which is L2-resident:", ab / 1024.0 / 120.0);
    println!("  the DRAM figure below is an UPPER BOUND that does not apply. See docs/whys/diana-latency.md.");
    for bw in [24.0f64] {
        println!("  at {bw} GB/s (DRAM, NOT the operative level) that would be {:.1} ms/image", ab * 4.0 / (bw * 1e9) * 1e3);
    }

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    // AVX2: 8 lanes x 2 flops/FMA x 2 FMA ports.
    for gh in [2.5f64, 3.5] {
        println!("  peak at {gh} GHz x {cores} cores: {:.0} GFLOP/s", 8.0 * 2.0 * 2.0 * gh * cores as f64);
    }
    for (who, ms) in [("ultralytics p50", 59.0f64)] {
        println!("  {who} {ms} ms => our arithmetic would need {:.0} GFLOP/s", gflop / (ms / 1e3));
    }
    Ok(())
}
