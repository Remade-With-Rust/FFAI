//! The p50 lever hiding in the reference table: RECTANGULAR inference.
//!
//! We letterbox every image to a 640x640 square. A 428x640 photo therefore
//! spends **33% of its compute on grey padding**. Ultralytics' own DEFAULT
//! is rectangular — pad only to the next multiple of 32 — and the M-D0
//! bench measured its rect variant as BOTH faster AND more accurate at the
//! n tier:
//!
//! | ultralytics yolo26n | mAP50 | img/s warm |
//! |---|---:|---:|
//! | square 640x640 (what we implement) | 68.65 | 15.30 |
//! | **rect (their default)** | **70.14** | **18.26** |
//!
//! So square is *dominated* at this tier: +1.49 pp mAP and +19% speed are
//! both on the table. We pinned square in M-D0 for a good reason — an
//! unpinned geometry made the .pt and ORT rows disagree — but pinning it for
//! the PARITY GATE and shipping it as the DEFAULT are different decisions,
//! and conflating them cost us both axes.
//!
//! This measures what rect is worth to OUR latency, which is the question
//! the reference table cannot answer for us.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example rect_probe
//! ```

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use ffai_diana::{backbone::Backbone, head::Head, neck::Neck};

fn best_ms<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = std::time::Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf();
    let weights = root.join("corpora/cache/yolo26n-diana.safetensors");
    if !weights.exists() {
        eprintln!("SKIP: run tools/diana_convert.py first");
        return Ok(());
    }
    let dev = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &dev)?
    };
    let dims = ffai_diana::config::Dims::for_scale("n").expect("n dims");
    let backbone = Backbone::new(vb.clone(), dims)?;
    let neck = Neck::new(vb.clone(), dims)?;
    let head = Head::new(vb, dims, 80, 1, vec![8.0, 16.0, 32.0])?;

    // (label, H, W). The rect sizes are what Ultralytics' LetterBox would
    // choose for the corpus's real aspect ratios — the next multiple of 32.
    let shapes: &[(&str, usize, usize)] = &[
        ("square 640x640 (ours today)", 640, 640),
        ("rect   640x448  (428x640 img)", 640, 448),
        ("rect   640x608  (586x640 img)", 640, 608),
        ("rect   384x640  (640x359 img)", 384, 640),
    ];

    println!("{:<32} {:>10} {:>10} {:>9}", "INPUT", "fwd ms", "px vs sq", "speedup");
    let mut base = 0.0;
    for &(label, h, w) in shapes {
        let x = Tensor::randn(0f32, 1.0, (1, 3, h, w), &dev)?;
        let ms = best_ms(
            || {
                let b = backbone.forward(&x).unwrap();
                let n = neck.forward(&b).unwrap();
                let per = head.forward(&n).unwrap();
                let (bx, sc) = head.concat_levels(&per).unwrap();
                std::hint::black_box((bx, sc));
            },
            10,
        );
        if base == 0.0 {
            base = ms;
        }
        println!(
            "{label:<32} {ms:>10.1} {:>9.0}% {:>8.2}x",
            (h * w) as f64 / (640.0 * 640.0) * 100.0,
            base / ms
        );
    }
    println!(
        "\nOur kernels already accept any H,W — conv3x3/dwconv/pointwise are all\n\
         shape-generic. Only `image::letterbox` hard-codes the square, so this is\n\
         a preprocessing decision, not a port."
    );
    Ok(())
}
