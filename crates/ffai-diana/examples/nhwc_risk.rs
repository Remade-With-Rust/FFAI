//! The three pieces the NHWC GEMM probe did NOT cover.
//!
//! ```text
//! cargo run --release -p ffai-diana --example nhwc_risk
//! ```
//!
//! Global NHWC measured 1.140x on the GEMM, ~3.67 % of detect. That number
//! covers convolution matmuls and nothing else. Three things change layout with
//! it, and each could give the win back:
//!
//! 1. **im2col** - produces `[HW, 9cin]` instead of `[9cin, HW]`.
//! 2. **the epilogue** - bias+SiLU over `[HW, cout]` instead of `[cout, HW]`.
//!    In NCHW one channel bias is a scalar over a contiguous run; in NHWC the
//!    bias VECTOR repeats along the fast axis.
//! 3. **channel plumbing** - counted in the real graph at **15 cat + 9 narrow
//!    per image over 16.3 MiB**. In NCHW a channel is a contiguous plane, so
//!    `narrow(dim=1)` is a free VIEW and `cat(dim=1)` is a block copy. In NHWC
//!    the channel is the fastest axis, so both become strided.
//!
//! The third is the real risk and the reason this probe exists: a free
//! operation becoming a copy cannot show up in any GEMM benchmark.
//!
//! Both plumbing arms use the SAME candle op on the same element count and
//! differ only in which axis the channel sits on. Timed min-of-N, interleaved.

use candle_core::{Device, Result, Tensor};
use rayon::prelude::*;

fn im2col_nchw(x: &[f32], cin: usize, h: usize, w: usize) -> Vec<f32> {
    let hw = h * w;
    let mut col = vec![0.0f32; 9 * cin * hw];
    col.par_chunks_mut(hw).enumerate().for_each(|(row, dst)| {
        let (c, tap) = (row / 9, row % 9);
        let (dy, dx) = ((tap / 3) as isize - 1, (tap % 3) as isize - 1);
        for oy in 0..h {
            let sy = oy as isize + dy;
            if sy < 0 || sy >= h as isize {
                continue;
            }
            let src = c * hw + sy as usize * w;
            let d = oy * w;
            for ox in 0..w {
                let sx = ox as isize + dx;
                if sx >= 0 && sx < w as isize {
                    dst[d + ox] = x[src + sx as usize];
                }
            }
        }
    });
    col
}

fn im2col_nhwc(x: &[f32], cin: usize, h: usize, w: usize) -> Vec<f32> {
    let k = 9 * cin;
    let mut col = vec![0.0f32; h * w * k];
    col.par_chunks_mut(w * k).enumerate().for_each(|(oy, dst)| {
        for tap in 0..9 {
            let (dy, dx) = ((tap / 3) as isize - 1, (tap % 3) as isize - 1);
            let sy = oy as isize + dy;
            if sy < 0 || sy >= h as isize {
                continue;
            }
            for ox in 0..w {
                let sx = ox as isize + dx;
                if sx < 0 || sx >= w as isize {
                    continue;
                }
                let s = (sy as usize * w + sx as usize) * cin;
                let d = ox * k + tap * cin;
                dst[d..d + cin].copy_from_slice(&x[s..s + cin]);
            }
        }
    });
    col
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// bias + SiLU with the channel as the SLOW axis: one scalar per run.
fn epi_nchw(y: &mut [f32], bias: &[f32], hw: usize) {
    y.par_chunks_mut(hw).enumerate().for_each(|(c, run)| {
        let b = bias[c];
        for e in run.iter_mut() {
            *e = silu(*e + b);
        }
    });
}

/// bias + SiLU with the channel as the FAST axis: the bias VECTOR repeats.
fn epi_nhwc(y: &mut [f32], bias: &[f32], cout: usize) {
    y.par_chunks_mut(cout).for_each(|px| {
        for (e, b) in px.iter_mut().zip(bias) {
            *e = silu(*e + b);
        }
    });
}

fn min_of<F: FnMut() -> Result<()>>(n: usize, mut f: F) -> Result<f64> {
    let mut best = f64::MAX;
    for _ in 0..n {
        let t = std::time::Instant::now();
        f()?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    Ok(best)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    const N: usize = 21;
    // (cin, cout, h, w, calls per image) - the heaviest real shapes.
    let shapes = [
        (32usize, 32usize, 48usize, 80usize, 12usize),
        (16, 16, 48, 80, 5),
        (64, 64, 24, 40, 4),
        (32, 64, 96, 160, 1),
    ];

    println!("=== 1. im2col ===");
    println!("{:>14} {:>6} {:>11} {:>11} {:>10}", "shape", "calls", "NCHW ms", "NHWC ms", "NHWC/NCHW");
    let (mut a1, mut b1) = (0.0f64, 0.0f64);
    for &(cin, _, h, w, c) in &shapes {
        let x: Vec<f32> = (0..cin * h * w).map(|i| (i % 13) as f32 * 0.03).collect();
        let (mut a, mut b) = (f64::MAX, f64::MAX);
        for _ in 0..N {
            let t = std::time::Instant::now();
            std::hint::black_box(im2col_nchw(&x, cin, h, w));
            a = a.min(t.elapsed().as_secs_f64());
            let t = std::time::Instant::now();
            std::hint::black_box(im2col_nhwc(&x, cin, h, w));
            b = b.min(t.elapsed().as_secs_f64());
        }
        a1 += a * c as f64;
        b1 += b * c as f64;
        println!("{:>7}c {:>3}x{:<3} {:>6} {:>11.4} {:>11.4} {:>9.2}x", cin, h, w, c, a * 1e3, b * 1e3, b / a);
    }
    println!("  weighted/img: NCHW {:.3} ms   NHWC {:.3} ms   -> {:.2}x\n", a1 * 1e3, b1 * 1e3, b1 / a1);

    println!("=== 2. epilogue (bias + SiLU) ===");
    println!("{:>14} {:>6} {:>11} {:>11} {:>10}", "shape", "calls", "NCHW ms", "NHWC ms", "NHWC/NCHW");
    let (mut a2, mut b2) = (0.0f64, 0.0f64);
    for &(_, cout, h, w, c) in &shapes {
        let hw = h * w;
        let bias: Vec<f32> = (0..cout).map(|i| i as f32 * 0.01 - 0.1).collect();
        let mut y1 = vec![0.3f32; cout * hw];
        let mut y2 = vec![0.3f32; cout * hw];
        let (mut a, mut b) = (f64::MAX, f64::MAX);
        for _ in 0..N {
            let t = std::time::Instant::now();
            epi_nchw(&mut y1, &bias, hw);
            a = a.min(t.elapsed().as_secs_f64());
            let t = std::time::Instant::now();
            epi_nhwc(&mut y2, &bias, cout);
            b = b.min(t.elapsed().as_secs_f64());
        }
        a2 += a * c as f64;
        b2 += b * c as f64;
        println!("{:>7}c {:>3}x{:<3} {:>6} {:>11.4} {:>11.4} {:>9.2}x", cout, h, w, c, a * 1e3, b * 1e3, b / a);
    }
    println!("  weighted/img: NCHW {:.3} ms   NHWC {:.3} ms   -> {:.2}x\n", a2 * 1e3, b2 * 1e3, b2 / a2);

    println!("=== 3. channel plumbing (15 cat + 9 narrow per image, 16.3 MiB) ===");
    println!("{:>18} {:>11} {:>11} {:>10}", "op / shape", "NCHW ms", "NHWC ms", "NHWC/NCHW");
    let (mut a3, mut b3) = (0.0f64, 0.0f64);
    let total_calls: usize = shapes.iter().map(|s| s.4).sum();
    for &(_, cout, h, w, c) in &shapes {
        let cc = cout.max(2);
        let half = cc / 2;
        let n1 = Tensor::rand(-1f32, 1f32, (1, half, h, w), &dev)?;
        let n2 = Tensor::rand(-1f32, 1f32, (1, half, h, w), &dev)?;
        let w1 = Tensor::rand(-1f32, 1f32, (1, h, w, half), &dev)?;
        let w2 = Tensor::rand(-1f32, 1f32, (1, h, w, half), &dev)?;
        let bign = Tensor::rand(-1f32, 1f32, (1, cc, h, w), &dev)?;
        let bigw = Tensor::rand(-1f32, 1f32, (1, h, w, cc), &dev)?;

        let ca = min_of(N, || {
            let t = Tensor::cat(&[&n1, &n2], 1)?;
            let _ = t.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            Ok(())
        })?;
        let cb = min_of(N, || {
            let t = Tensor::cat(&[&w1, &w2], 3)?;
            let _ = t.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            Ok(())
        })?;
        // narrow, then materialise it. In NCHW the slice is already contiguous
        // so `.contiguous()` is free; in NHWC it is strided and must copy.
        let na = min_of(N, || {
            let t = bign.narrow(1, 0, half)?.contiguous()?;
            let _ = t.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            Ok(())
        })?;
        let nb = min_of(N, || {
            let t = bigw.narrow(3, 0, half)?.contiguous()?;
            let _ = t.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            Ok(())
        })?;

        // Spread the counted 15 cat + 9 narrow across shapes by call weight.
        let share = c as f64 / total_calls as f64;
        a3 += (ca * 15.0 + na * 9.0) * share;
        b3 += (cb * 15.0 + nb * 9.0) * share;
        println!("{:>9}c {:>3}x{:<3} cat {:>11.4} {:>11.4} {:>9.2}x", cc, h, w, ca * 1e3, cb * 1e3, cb / ca);
        println!("{:>9}c {:>3}x{:<3} nar {:>11.4} {:>11.4} {:>9.2}x", cc, h, w, na * 1e3, nb * 1e3, nb / na);
    }
    println!("  weighted/img: NCHW {:.3} ms   NHWC {:.3} ms   -> {:.2}x\n", a3 * 1e3, b3 * 1e3, b3 / a3);

    // READ THE RATIOS, NOT THESE ABSOLUTES. The im2col arms here are written
    // for the probe and are ~7.6x slower than the shipped kernel (21.997 ms
    // against a measured in-context 2.887), so weighting the total by them
    // would inflate im2col's share by that factor. The RATIO per shape is the
    // durable output; the campaign applies it to the in-context share measured
    // by FFAI_PROFILE / FFAI_DIANA_ROOFLINE.
    //
    // In-context (ms/img): detect 26.5, conv 20.77, gemm 16.9, im2col 2.887,
    // epilogue+rest 0.98, plumbing 0.91. Applying the ratios below gives
    // gemm +2.075, im2col +0.549, epilogue -0.364, plumbing -0.893 =
    // NET +1.367 ms/img, 5.2 % of detect.
    let gemm_saving = 16.9 - 16.9 / 1.140;
    let extra = (b1 - a1 + b2 - a2 + b3 - a3) * 1e3;
    println!("=== VERDICT ===");
    println!("  GEMM saving from NHWC (measured over the real sequence): {gemm_saving:.3} ms/img");
    println!("  im2col       {:+.3} ms/img", (b1 - a1) * 1e3);
    println!("  epilogue     {:+.3} ms/img", (b2 - a2) * 1e3);
    println!("  plumbing     {:+.3} ms/img", (b3 - a3) * 1e3);
    println!("  ---------------------------");
    println!("  extra cost   {extra:+.3} ms/img");
    println!(
        "  NET {:+.3} ms/img   {}",
        gemm_saving - extra,
        if gemm_saving > extra { "<= NHWC still pays" } else { "<= NHWC GIVES THE WIN BACK" }
    );
    Ok(())
}
