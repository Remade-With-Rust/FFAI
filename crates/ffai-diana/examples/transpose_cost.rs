//! Is the transpose tax fundamental, or is it candle's generic path?
//!
//! Pricing the transposed-GEMM idea put the tax at 16.55 ms against a
//! 2.66 ms saving and pruned it. But 16.55 ms to move ~2.5 M elements is
//! ~20 MB of traffic, which at this machine's ~20-30 GB/s should cost
//! about 1 ms. A 16x discrepancy is the instrument asking for help, so
//! before the prune stands: compare candle's `t().contiguous()` against a
//! plain BLOCKED transpose, which is the standard cache-friendly form.
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

/// Blocked transpose: [rows, cols] -> [cols, rows], 32x32 tiles so each
/// tile's source and destination both stay in L1 while it is copied.
fn blocked_transpose(src: &[f32], rows: usize, cols: usize, dst: &mut [f32]) {
    const B: usize = 32;
    for r0 in (0..rows).step_by(B) {
        for c0 in (0..cols).step_by(B) {
            let (r1, c1) = ((r0 + B).min(rows), (c0 + B).min(cols));
            // Inner loop over R, not C. With `c` innermost the destination
            // writes stride by `rows` and touch a fresh cache line every
            // iteration; with `r` innermost they are contiguous. Same tile,
            // same work, and the first version of this probe had it the
            // wrong way round.
            for c in c0..c1 {
                for r in r0..r1 {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}

fn best(mut f: impl FnMut(), n: usize) -> f64 {
    for _ in 0..2 { f(); }
    let mut b = f64::MAX;
    for _ in 0..n {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    println!("{:>8} {:>8}   {:>12} {:>12} {:>9}", "ohw", "c_out", "candle t() ms", "blocked ms", "speedup");
    let (mut tc, mut tb) = (0.0f64, 0.0f64);
    for (ohw, c_out) in [(102400usize, 16usize), (25600, 32), (25600, 64), (6400, 128), (1600, 256), (400, 256)] {
        let t = Tensor::zeros((ohw, c_out), DType::F32, &dev)?;
        let a = best(|| { let y = t.t().unwrap().contiguous().unwrap(); std::hint::black_box(&y); }, 7);
        let src = vec![0f32; ohw * c_out];
        let mut dst = vec![0f32; ohw * c_out];
        let b = best(|| { blocked_transpose(&src, ohw, c_out, &mut dst); std::hint::black_box(&dst[0]); }, 7);
        tc += a; tb += b;
        println!("{ohw:>8} {c_out:>8}   {:>12.3} {:>12.3} {:>8.2}x", a * 1e3, b * 1e3, a / b);
    }
    println!();
    println!("TOTAL candle {:.3} ms   blocked {:.3} ms   -> {:.1}x", tc * 1e3, tb * 1e3, tc / tb);
    // Re-run the transposed-GEMM arithmetic with the cheaper transpose.
    let (gemm_asis, gemm_t) = (10.627e-3, 7.972e-3);
    println!();
    println!("Transposed-GEMM idea, re-priced with a blocked transpose:");
    println!("  as-is {:.3} ms   vs   transposed {:.3} + tax {:.3} = {:.3} ms",
             gemm_asis * 1e3, gemm_t * 1e3, tb * 1e3, (gemm_t + tb) * 1e3);
    let net = gemm_asis / (gemm_t + tb);
    println!("  NET {net:.3}x  ({})", if net > 1.0 { "REOPENED" } else { "still dead" });
    Ok(())
}
