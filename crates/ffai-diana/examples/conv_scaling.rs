//! Depth 5: WHY is conv 8.2x off the GEMM ceiling, when candle already
//! im2col's it?
//!
//! Two facts are established and they constrain the answer: 24 cores buy the
//! whole pipeline only 1.42x, and removing 10.6% of CPU work moved the wall
//! 0%. So the wall is not gated by CPU throughput. This probe discriminates
//! between the remaining candidates by asking one question per shape:
//!
//!   **does `conv2d` scale with threads, and does `matmul` scale on the same
//!   machine, in the same process, at the same arithmetic?**
//!
//! - conv scales like matmul  -> both are bandwidth-limited; the fix is
//!   traffic (precision, fusion), not parallelism.
//! - matmul scales, conv does NOT -> candle's conv is leaving parallelism on
//!   the floor; the fix is to route conv through the matmul that already
//!   scales (our own im2col + `Tensor::matmul`).
//!
//! Run it twice, once per thread count — rayon's global pool is fixed at
//! startup, so the honest way to vary it is the process boundary:
//!
//! ```sh
//! RAYON_NUM_THREADS=1  cargo run --release -p ffai-diana --example conv_scaling
//! RAYON_NUM_THREADS=24 cargo run --release -p ffai-diana --example conv_scaling
//! ```

use candle_core::{Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module};

/// (label, cin, cout, spatial, k, stride) — the model's real shapes, dense
/// only (depthwise now has its own kernel).
const SHAPES: &[(&str, usize, usize, usize, usize, usize)] = &[
    ("l0 stem  3x3", 3, 16, 640, 3, 2),
    ("l1 down  3x3", 16, 32, 320, 3, 2),
    ("l2 c3k2  1x1", 32, 32, 160, 1, 1),
    ("l2 bneck 3x3", 16, 8, 160, 3, 1),
    ("l4 bneck 3x3", 32, 16, 80, 3, 1),
    ("l8 c3k   3x3", 64, 64, 20, 3, 1),
    ("head box 3x3", 64, 16, 80, 3, 1),
    ("head cls 1x1", 80, 80, 80, 1, 1),
];

fn best_s<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = std::time::Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    println!("RAYON_NUM_THREADS={threads}");

    // The machine's streaming ceiling, measured here rather than borrowed —
    // a roofline is only valid for the shape it was measured at.
    //
    // Two bugs lived in this block and both produced an IMPOSSIBLE number
    // (a "ceiling" BELOW what the convolutions below it achieved, which is
    // the instrument asking for help): the size was `64 << 20 / 4`, which
    // parses as `64 << 5` = 2048 elements and measured an L1-resident
    // buffer; and a serial `f32` sum will not vectorize, because float
    // addition is not associative. Fixed: a genuinely L3-exceeding buffer,
    // and eight independent accumulators so the reduction can vectorize.
    let n = 16 << 20; // 16 M f32 = 64 MB, far past L3
    let big = vec![1.0f32; n];
    let copy_s = best_s(
        || {
            let mut acc = [0f32; 8];
            for c in big.chunks_exact(8) {
                for (a, v) in acc.iter_mut().zip(c) {
                    *a += *v;
                }
            }
            std::hint::black_box(acc.iter().sum::<f32>());
        },
        5,
    );
    println!(
        "streaming read ceiling: {:.1} GB/s ({} MB pass)\n",
        (n * 4) as f64 / copy_s / 1e9,
        n * 4 / (1 << 20)
    );

    println!(
        "{:<14} {:>9} {:>10} {:>9} {:>10} {:>9}",
        "SHAPE", "conv ms", "conv GF/s", "gemm ms", "gemm GF/s", "conv GB/s"
    );
    for &(label, cin, cout, sp, k, stride) in SHAPES {
        let out_sp = (sp + 2 * (k / 2) - k) / stride + 1;
        let mflop = (out_sp * out_sp * cout * cin * k * k) as f64 * 2.0 / 1e6;

        let x = Tensor::randn(0f32, 1.0, (1, cin, sp, sp), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (cout, cin, k, k), &dev)?;
        let cfg = Conv2dConfig { padding: k / 2, stride, ..Default::default() };
        let conv = Conv2d::new(w, None, cfg);
        let cs = best_s(|| { let _ = conv.forward(&x); }, 15);

        let m = out_sp * out_sp;
        let kk = cin * k * k;
        let a = Tensor::randn(0f32, 1.0, (m, kk), &dev)?;
        let b = Tensor::randn(0f32, 1.0, (kk, cout), &dev)?;
        let gs = best_s(|| { let _ = a.matmul(&b); }, 15);

        // Bytes an im2col conv must touch at minimum: read input, write and
        // re-read the im2col matrix, write output.
        let bytes = ((cin * sp * sp) + 2 * (m * kk) + (cout * m)) as f64 * 4.0;
        println!(
            "{label:<14} {:>9.3} {:>10.1} {:>9.3} {:>10.1} {:>9.1}",
            cs * 1e3,
            mflop / 1e3 / cs,
            gs * 1e3,
            mflop / 1e3 / gs,
            bytes / cs / 1e9
        );
    }
    println!(
        "\nRun at 1 and at N threads and compare the two GF/s columns:\n  \
         both scale -> bandwidth-bound, fix traffic.\n  \
         gemm scales, conv does not -> candle's conv is the problem; route conv\n  \
         through the matmul that already scales."
    );
    Ok(())
}
