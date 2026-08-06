//! Does candle's GEMM lose throughput at small M, outside our graph?
//!
//! ```text
//! cargo run --release -p ffai-diana --example gemm_m_sweep
//! ```
//!
//! The per-layer roofline says effective GFLOP/s across 47 convolution shapes
//! correlates **+0.823 with log2(cout)** and **-0.048 with the streamed
//! operand's size**. Since `cout` is the GEMM's M dimension, and since 1x1
//! convolutions — which build no im2col operand at all — show the same
//! deficit as 3x3, the constraint looks like the matmul rather than anything
//! we wrap around it.
//!
//! "Looks like" is not a measurement, and `codec-measurement` §6/§11 both
//! warn that in-context and isolated numbers mislead in BOTH directions. So
//! this reproduces the question with nothing of ours in the loop: the same
//! total FLOPs at each M, K and N adjusted to hold the work constant, straight
//! through `Tensor::matmul`.
//!
//! If the isolated sweep shows the same collapse, the target is a small-M
//! kernel and the three structural ideas already refuted (im2col fusion,
//! direct convolution, cache tiling) stay refuted for a reason we can name.
//! If it does NOT collapse, the deficit is ours and lives in the wrapper —
//! which would be far better news, and is exactly why this is worth 20 lines.

use candle_core::{DType, Device, Tensor};

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    // Hold total work near-constant across M so GFLOP/s is comparable:
    // 2*M*K*N stays put while M varies over the range the graph actually uses.
    const K: usize = 576; // 9 * 64, a real im2col K
    const WORK: usize = 8 * 576 * 400_000; // reference M*K*N product

    println!(
        "candle matmul, K={K}, N chosen to hold M*K*N constant, f32, \
         {} threads",
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0)
    );
    println!("(N shrinks as M grows, so every row does the same arithmetic)\n");
    println!("{:>6} {:>9} {:>12} {:>11} {:>9}", "M", "N", "GFLOP", "ms", "GFLOP/s");

    let mut base = 0.0f64;
    for m in [8usize, 16, 32, 64, 128, 256, 512] {
        let n = (WORK / (m * K)).max(64);
        let a = Tensor::rand(-0.5f32, 0.5f32, (m, K), &dev)?.to_dtype(DType::F32)?;
        let b = Tensor::rand(-0.5f32, 0.5f32, (K, n), &dev)?.to_dtype(DType::F32)?;

        // Warm: first call pays lazy init and first-touch faults.
        let _ = a.matmul(&b)?;

        // Best-of-N, because this is a clock on a box that will not hold
        // still; the minimum is the least contaminated statistic available
        // and §1 shows minima agree even when spreads do not.
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t = std::time::Instant::now();
            let c = a.matmul(&b)?;
            // Force materialisation; candle is eager, but read one element so
            // nothing can be elided.
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            best = best.min(t.elapsed().as_secs_f64());
        }
        let gflop = 2.0 * m as f64 * K as f64 * n as f64 / 1e9;
        let rate = gflop / best;
        if base == 0.0 {
            base = rate;
        }
        println!(
            "{m:>6} {n:>9} {gflop:>12.2} {:>11.2} {rate:>9.1}",
            best * 1000.0
        );
    }
    // ---------------------------------------------------------------
    // THE ORIENTATION TEST. A GEMM microkernel is not symmetric in M and N:
    // M is normally blocked over register ROWS and N over columns, so the
    // same contraction can be fast one way round and slow the other.
    //
    // Our convolution issues W[cout, 9cin] @ col[9cin, HW] — M = cout, the
    // small one. The transpose col^T[HW, 9cin] @ W^T[9cin, cout] does
    // IDENTICAL arithmetic with M = HW, which is huge. That is also exactly
    // what an NHWC layout produces naturally.
    //
    // If the transposed orientation is fast, the fix is a LAYOUT change, not
    // a kernel. codec-measurement §11 names orientation as the first axis to
    // vary — "same arithmetic, 10x apart". This is that test.
    // ---------------------------------------------------------------
    println!("
ORIENTATION: identical contraction, operands swapped");
    println!(
        "{:>6} {:>9} {:>13} {:>13} {:>9}",
        "cout", "HW", "M=cout GF/s", "M=HW GF/s", "speedup"
    );
    for cout in [8usize, 16, 32, 64, 128] {
        let hw = (WORK / (cout * K)).max(64);
        let w = Tensor::rand(-0.5f32, 0.5f32, (cout, K), &dev)?;
        let col = Tensor::rand(-0.5f32, 0.5f32, (K, hw), &dev)?;
        // Pre-transposed and contiguous OUTSIDE the timed region: an NHWC
        // path produces them this way, so charging the transpose here would
        // measure a copy nobody would perform.
        let col_t = col.t()?.contiguous()?;
        let w_t = w.t()?.contiguous()?;

        let (mut a_best, mut b_best) = (f64::MAX, f64::MAX);
        let _ = w.matmul(&col)?;
        let _ = col_t.matmul(&w_t)?;
        for _ in 0..9 {
            let t = std::time::Instant::now();
            let c = w.matmul(&col)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            a_best = a_best.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let c = col_t.matmul(&w_t)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            b_best = b_best.min(t.elapsed().as_secs_f64());
        }
        let gflop = 2.0 * cout as f64 * K as f64 * hw as f64 / 1e9;
        println!(
            "{cout:>6} {hw:>9} {:>13.1} {:>13.1} {:>8.2}x",
            gflop / a_best,
            gflop / b_best,
            a_best / b_best
        );
    }
    // Does a per-layer NHWC ISLAND pay, or does it need the whole graph?
    //
    // The gain above assumes the operands ARRIVE transposed. In an NCHW graph
    // they do not, so an island would pay a transpose in and a transpose out
    // per layer. That is a real cost and it is measured here rather than
    // assumed: if it exceeds the GEMM saving, the win is only available to a
    // whole-graph layout change and an incremental version is a trap.
    println!("
COST OF AN NHWC ISLAND: transpose in + out, per layer");
    println!(
        "{:>6} {:>9} {:>12} {:>12} {:>12} {:>9}",
        "cout", "HW", "gemm save", "transposes", "net", "verdict"
    );
    for cout in [16usize, 32, 64] {
        let hw = (WORK / (cout * K)).max(64);
        let w = Tensor::rand(-0.5f32, 0.5f32, (cout, K), &dev)?;
        let col = Tensor::rand(-0.5f32, 0.5f32, (K, hw), &dev)?;
        let col_t = col.t()?.contiguous()?;
        let w_t = w.t()?.contiguous()?;

        let (mut a, mut b, mut tr) = (f64::MAX, f64::MAX, f64::MAX);
        let _ = w.matmul(&col)?;
        let _ = col_t.matmul(&w_t)?;
        for _ in 0..9 {
            let t = std::time::Instant::now();
            let c = w.matmul(&col)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            a = a.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let c = col_t.matmul(&w_t)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            b = b.min(t.elapsed().as_secs_f64());

            // One transpose of the streamed operand plus one of the result:
            // what an island pays to enter and leave the layout.
            let t = std::time::Instant::now();
            let ct = col.t()?.contiguous()?;
            let back = ct.t()?.contiguous()?;
            let _ = back.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            tr = tr.min(t.elapsed().as_secs_f64());
        }
        let save = (a - b) * 1000.0;
        let cost = tr * 1000.0;
        println!(
            "{cout:>6} {hw:>9} {save:>11.2}ms {cost:>11.2}ms {:>11.2}ms {:>9}",
            save - cost,
            if save > cost { "pays" } else { "TRAP" }
        );
    }
    Ok(())
}
