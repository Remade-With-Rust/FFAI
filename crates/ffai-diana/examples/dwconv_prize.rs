//! Price the depthwise-convolution brick BEFORE building it.
//!
//! `expected pipeline gain = stage share x speedup` — if the ceiling lands
//! under the measured noise floor, the experiment cannot succeed however
//! good the kernel is, and two minutes of arithmetic saves a day.
//!
//! candle has no grouped-conv kernel: `Tensor::conv2d` with `groups > 1`
//! does `chunk(groups, 1)` -> one `conv2d_single_group` per group -> `cat`
//! (candle-core/src/conv.rs). A depthwise conv therefore runs `c_in` separate
//! single-channel convolutions. This measures what that costs across the six
//! depthwise convolutions YOLO26n's head actually runs.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example dwconv_prize
//! ```

use candle_core::{Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module};

/// (label, channels, spatial) for every DWConv in the one2one class head:
/// two per level, at strides 8 / 16 / 32.
const DW: &[(&str, usize, usize)] = &[
    ("L0 dw c=64  s=80", 64, 80),
    ("L0 dw c=80  s=80", 80, 80),
    ("L1 dw c=128 s=40", 128, 40),
    ("L1 dw c=80  s=40", 80, 40),
    ("L2 dw c=256 s=20", 256, 20),
    ("L2 dw c=80  s=20", 80, 20),
];

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

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    println!(
        "{:<18} {:>7} {:>9} {:>9} {:>10}",
        "DEPTHWISE", "MFLOP", "ms", "GFLOP/s", "us/group"
    );
    let mut total = 0.0;
    for &(label, c, sp) in DW {
        let macs = (sp * sp * c * 9) as f64;
        let mflop = macs * 2.0 / 1e6;
        let x = Tensor::randn(0f32, 1.0, (1, c, sp, sp), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (c, 1, 3, 3), &dev)?;
        let b = Tensor::randn(0f32, 1.0, c, &dev)?;
        let cfg = Conv2dConfig { padding: 1, stride: 1, groups: c, ..Default::default() };
        let conv = Conv2d::new(w, Some(b), cfg);
        let ms = best_ms(|| { let _ = conv.forward(&x); }, 20);
        total += ms;
        println!(
            "{label:<18} {:>7.2} {ms:>9.3} {:>9.2} {:>10.1}",
            mflop,
            mflop / 1e3 / (ms / 1e3),
            ms * 1e3 / c as f64
        );
    }

    // A same-shape STANDARD (groups=1) conv, for scale: it does c times MORE
    // arithmetic, so if it is faster the group splitting is the whole cost.
    let x = Tensor::randn(0f32, 1.0, (1, 64, 80, 80), &dev)?;
    let w = Tensor::randn(0f32, 1.0, (64, 64, 3, 3), &dev)?;
    let dense = Conv2d::new(w, None, Conv2dConfig { padding: 1, ..Default::default() });
    let dense_ms = best_ms(|| { let _ = dense.forward(&x); }, 20);

    println!("\nsix depthwise convs: {total:.2} ms/image");
    println!(
        "for scale, a DENSE 64->64 3x3 at 80x80 (64x MORE arithmetic): {dense_ms:.2} ms\n\
         => depthwise is {:.1}x SLOWER per FLOP than the dense conv beside it",
        (total / 6.0) / (dense_ms / 64.0)
    );

    // The prune: what is this worth on the whole pipeline?
    let detect_ms = 264.5; // measured, profiled run, coco-032
    println!(
        "\nPRUNE: detect is ~{detect_ms:.0} ms; these are {:.1}% of it.",
        total / detect_ms * 100.0
    );
    for speedup in [4.0, 10.0, 40.0] {
        let saved = total - total / speedup;
        println!(
            "  at {speedup:>4.0}x on depthwise: save {saved:>6.1} ms -> pipeline {:.3}x",
            detect_ms / (detect_ms - saved)
        );
    }
    Ok(())
}
