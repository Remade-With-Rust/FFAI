//! Depth 4/5: is candle's CPU `conv2d` reaching GEMM speed at OUR shapes?
//!
//! The profile says convolution is 84.6% of detect. A convolution IS a GEMM
//! after im2col, and a tuned GEMM beats a direct loop by orders of magnitude
//! ("look for hot loops that are secretly matmuls"). This probe prices, for
//! each real layer shape in the model:
//!
//!   - `conv2d` as we call it today,
//!   - the SAME arithmetic as an explicit matmul of the im2col'd operands,
//!   - the machine's GEMM ceiling at that exact shape.
//!
//! A roofline is only valid for the SHAPE it was measured at, so the ceiling
//! is measured per shape rather than borrowed from a big square matmul.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example conv_roofline
//! ```

use candle_core::{Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module};

/// (label, in_ch, out_ch, spatial, kernel, stride, groups) — the shapes the
/// model actually runs, taken from the probed architecture.
const SHAPES: &[(&str, usize, usize, usize, usize, usize, usize)] = &[
    ("l0 stem      ", 3, 16, 640, 3, 2, 1),
    ("l1 down      ", 16, 32, 320, 3, 2, 1),
    ("l2 c3k2 1x1  ", 32, 32, 160, 1, 1, 1),
    ("l2 bneck 3x3 ", 16, 8, 160, 3, 1, 1),
    ("l4 bneck 3x3 ", 32, 16, 80, 3, 1, 1),
    ("l6 c3k 3x3   ", 32, 32, 40, 3, 1, 1),
    ("l8 c3k 3x3   ", 64, 64, 20, 3, 1, 1),
    ("head box 3x3 ", 64, 16, 80, 3, 1, 1),
    ("head dw 3x3  ", 64, 64, 80, 3, 1, 64),
    ("head pw 1x1  ", 64, 80, 80, 1, 1, 1),
    ("head cls 1x1 ", 80, 80, 80, 1, 1, 1),
];

fn bench<F: FnMut() -> candle_core::Result<()>>(mut f: F, iters: usize) -> f64 {
    let _ = f();
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = std::time::Instant::now();
        let _ = f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    println!(
        "{:<14} {:>7} {:>9} {:>10} {:>10} {:>10} {:>8}",
        "SHAPE", "MFLOP", "conv ms", "conv GF/s", "gemm ms", "gemm GF/s", "RATIO"
    );

    let mut conv_total = 0.0;
    let mut gemm_total = 0.0;
    for &(label, cin, cout, sp, k, stride, groups) in SHAPES {
        let out_sp = (sp + 2 * (k / 2) - k) / stride + 1;
        // MACs = out_pixels * cout * (cin/groups) * k*k ; 2 FLOP per MAC
        let macs = (out_sp * out_sp * cout * (cin / groups) * k * k) as f64;
        let mflop = macs * 2.0 / 1e6;

        let x = Tensor::randn(0f32, 1.0, (1, cin, sp, sp), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (cout, cin / groups, k, k), &dev)?;
        let b = Tensor::randn(0f32, 1.0, cout, &dev)?;
        let cfg = Conv2dConfig { padding: k / 2, stride, groups, ..Default::default() };
        let conv = Conv2d::new(w.clone(), Some(b), cfg);
        let conv_s = bench(
            || {
                conv.forward(&x)?;
                Ok(())
            },
            20,
        );

        // The same arithmetic as one GEMM: (out_pixels x cin*k*k) @ (cin*k*k x cout).
        // Operands are synthesized at the right SHAPE — this prices the
        // multiply, which is what a GEMM-backed conv would spend its time on.
        let m = out_sp * out_sp;
        let kk = (cin / groups) * k * k;
        let a = Tensor::randn(0f32, 1.0, (m, kk), &dev)?;
        let bm = Tensor::randn(0f32, 1.0, (kk, cout), &dev)?;
        let gemm_s = bench(
            || {
                a.matmul(&bm)?;
                Ok(())
            },
            20,
        );

        conv_total += conv_s;
        gemm_total += gemm_s;
        println!(
            "{label} {:>7.1} {:>9.3} {:>10.1} {:>10.3} {:>10.1} {:>8.1}x",
            mflop,
            conv_s * 1e3,
            mflop / 1e3 / conv_s,
            gemm_s * 1e3,
            mflop / 1e3 / gemm_s,
            conv_s / gemm_s
        );
    }
    println!(
        "\nsummed over these shapes: conv {:.1} ms · gemm {:.1} ms · {:.1}x",
        conv_total * 1e3,
        gemm_total * 1e3,
        conv_total / gemm_total
    );
    println!(
        "NOTE: the gemm column EXCLUDES im2col, which a real GEMM-backed conv must pay.\n\
         It is the ceiling of the multiply, not a drop-in replacement's cost."
    );
    Ok(())
}
