//! Does the conv GEMM's ORIENTATION matter? The axis never tested here.
//!
//! We compute `w[c_out, K] x col[K, ohw]`: M = c_out is SMALL (16-256 at the
//! n tier), N = ohw is large. The skill's learnings carry a case where the
//! same arithmetic ran 54 vs 539 GFLOP/s purely on which operand was M — so
//! the transposed form `col^T[ohw, K] x w^T[K, c_out]` is worth pricing
//! before concluding the GEMM is someone else's problem.
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn best(f: impl Fn() -> candle_core::Result<Tensor>, n: usize) -> f64 {
    for _ in 0..2 { let _ = f().unwrap(); }
    let mut b = f64::MAX;
    for _ in 0..n {
        let t = Instant::now();
        let y = f().unwrap();
        std::hint::black_box(&y);
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    println!("{:>6} {:>7} {:>8}   {:>11} {:>11} {:>8} {:>9} {:>9}",
             "c_out", "K", "ohw", "as-is ms", "transposed", "speedup", "tax ms", "verdict");
    let (mut tot_a, mut tot_b, mut tot_tax) = (0.0f64, 0.0f64, 0.0f64);
    // Real yolo26n 3x3 shapes: (c_out, c_in, ohw)
    for (c_out, c_in, ohw) in [
        (16usize, 3usize, 102400usize),   // stem 640->320
        (32, 16, 25600),                  // 160x160
        (64, 32, 25600),
        (128, 64, 6400),                  // 80x80
        (256, 128, 1600),                 // 40x40
        (256, 256, 400),                  // 20x20
    ] {
        let k = c_in * 9;
        let w = Tensor::zeros((c_out, k), DType::F32, &dev)?;
        let col = Tensor::zeros((k, ohw), DType::F32, &dev)?;
        let wt = Tensor::zeros((k, c_out), DType::F32, &dev)?;
        let colt = Tensor::zeros((ohw, k), DType::F32, &dev)?;
        let a = best(|| w.matmul(&col), 7);
        let b = best(|| colt.matmul(&wt), 7);
        // THE TAX: the transposed form produces [ohw, c_out] and the graph
        // needs [c_out, ohw]. Price it, because a win that pays for its own
        // undoing is not a win.
        let out_t = Tensor::zeros((ohw, c_out), DType::F32, &dev)?;
        let tax = best(|| out_t.t()?.contiguous(), 7);
        println!("{c_out:>6} {k:>7} {ohw:>8}   {:>11.3} {:>11.3} {:>7.2}x {:>9.3} {:>9}",
                 a * 1e3, b * 1e3, a / b, tax * 1e3,
                 if b + tax < a { "NET WIN" } else { "tax eats it" });
        tot_a += a; tot_b += b; tot_tax += tax;
    }
    println!();
    println!("TOTALS  as-is {:.3} ms   transposed {:.3} ms   + transpose tax {:.3} ms = {:.3}",
             tot_a * 1e3, tot_b * 1e3, tot_tax * 1e3, (tot_b + tot_tax) * 1e3);
    let net = tot_a / (tot_b + tot_tax);
    println!("NET after paying the tax: {net:.3}x  ({})",
             if net > 1.0 { "still a win" } else { "the tax eats the win" });
    println!("GEMM is 17.2% of serial detect, so that is {:.1}% of the pipeline.",
             17.2 * (1.0 - 1.0 / net.max(1e-9)));
    Ok(())
}
