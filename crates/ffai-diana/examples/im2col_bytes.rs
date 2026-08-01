//! How many BYTES does im2col move per image, and is that near this
//! machine's memory bandwidth?
//!
//! A deterministic counter, because wall time on this box drifts further
//! than the effect under test. If the traffic is close to the bandwidth
//! roofline then im2col is memory-bound, more threads cannot help it, and
//! the only lever is moving FEWER bytes.
use ffai_core::engine::{DetectEngine, DetectOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let engine = ffai_diana::engine::Yolo26::build(&tier, ffai_diana::image::Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    engine.detect(&img, &opts)?;                     // warm; also clears
    let _ = ffai_diana::conv3x3::take_im2col_elems();
    engine.detect(&img, &opts)?;
    let elems = ffai_diana::conv3x3::take_im2col_elems();
    let bytes = elems as f64 * 4.0;
    // Written once by im2col, then read once by the GEMM: 2x traffic.
    let traffic = bytes * 2.0;
    println!("tier {tier}: im2col writes {:.1} M elements = {:.1} MiB per image", elems as f64 / 1e6, bytes / 1048576.0);
    println!("  with the GEMM's read-back that is {:.1} MiB of traffic", traffic / 1048576.0);
    for bw in [17.6f64, 24.0] {
        println!("  at {bw} GB/s that is {:.1} ms of pure memory time", traffic / (bw * 1e9) * 1e3);
    }
    Ok(())
}
