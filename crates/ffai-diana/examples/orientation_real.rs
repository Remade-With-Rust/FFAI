//! Orientation on the REAL convolution shapes, not synthetic ones.
//!
//! ```text
//! cargo run --release -p ffai-diana --example orientation_real
//! ```
//!
//! The synthetic sweep said transposing the GEMM is worth 1.31-1.68x, and a
//! follow-up said a per-layer NHWC island is 100-200x underwater on transpose
//! cost. **The second measurement was wrong**, and wrong in a way worth
//! naming: it transposed the COL MATRIX, which nobody would ever do.
//!
//! im2col *builds* the col matrix. Its layout is ours to choose — a gather can
//! write `[HW, 9cin]` exactly as easily as `[9cin, HW]`, at the same cost, and
//! the weights transpose once at load time. So the NHWC orientation needs **no
//! transpose of the large operand at all**. The only thing that changes layout
//! is the ACTIVATION, which is `cout*H*W` — 1 MB where the col matrix was 460.
//!
//! This measures three things on the shapes the graph actually runs:
//!
//! 1. `W[cout,9cin] @ col[9cin,HW]` — what we ship.
//! 2. `colT[HW,9cin] @ WT[9cin,cout]` — identical arithmetic, NHWC orientation.
//! 3. The activation transpose, so an island's real cost is priced against the
//!    real saving rather than against a copy nobody performs.
//!
//! Shapes are the top rows of the per-layer roofline at 640x384, with their
//! per-image call counts, so the totals are weighted the way the graph is.

use candle_core::{Device, Result, Tensor};

/// (cin, cout, h, w, calls per image) — from `FFAI_DIANA_ROOFLINE=1`.
const SHAPES: &[(usize, usize, usize, usize, usize)] = &[
    (32, 32, 24, 40, 12),
    (16, 16, 48, 80, 5),
    (64, 16, 48, 80, 1),
    (16, 8, 96, 160, 1),
    (128, 16, 24, 40, 1),
    (8, 16, 96, 160, 1),
    (32, 16, 48, 80, 1),
    (16, 32, 48, 80, 1),
    (256, 16, 12, 20, 1),
    (64, 64, 12, 20, 4),
    (128, 64, 12, 20, 1),
    (64, 128, 12, 20, 1),
];

fn best_of<F: FnMut() -> Result<()>>(n: usize, mut f: F) -> Result<f64> {
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
    println!("real conv shapes at 640x384, best-of-11, weighted by calls/image\n");
    println!(
        "{:>10} {:>9} {:>6} {:>10} {:>10} {:>9} {:>11}",
        "cin->cout", "HxW", "calls", "NCHW ms", "NHWC ms", "speedup", "transpose"
    );

    let (mut tot_a, mut tot_b, mut tot_t) = (0.0f64, 0.0f64, 0.0f64);

    for &(cin, cout, h, w, calls) in SHAPES {
        let hw = h * w;
        let k = 9 * cin;

        // The two operand pairs. Both are built OUTSIDE the timed region
        // because both are things a real path would already have: im2col
        // produces whichever col layout it was written to produce, and the
        // weights are transposed once when the model loads.
        let wt_nchw = Tensor::rand(-0.5f32, 0.5f32, (cout, k), &dev)?;
        let col_nchw = Tensor::rand(-0.5f32, 0.5f32, (k, hw), &dev)?;
        let col_nhwc = Tensor::rand(-0.5f32, 0.5f32, (hw, k), &dev)?;
        let wt_nhwc = Tensor::rand(-0.5f32, 0.5f32, (k, cout), &dev)?;

        // The activation, which IS what an island would have to transpose.
        let act = Tensor::rand(-0.5f32, 0.5f32, (cout, hw), &dev)?;

        let _ = wt_nchw.matmul(&col_nchw)?;
        let _ = col_nhwc.matmul(&wt_nhwc)?;

        let a = best_of(11, || {
            let c = wt_nchw.matmul(&col_nchw)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            Ok(())
        })?;
        let b = best_of(11, || {
            let c = col_nhwc.matmul(&wt_nhwc)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            Ok(())
        })?;
        let t = best_of(11, || {
            let c = act.t()?.contiguous()?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            Ok(())
        })?;

        tot_a += a * calls as f64;
        tot_b += b * calls as f64;
        tot_t += t * calls as f64;

        println!(
            "{:>4}->{:<5} {:>4}x{:<4} {:>6} {:>10.3} {:>10.3} {:>8.2}x {:>10.3}",
            cin,
            cout,
            h,
            w,
            calls,
            a * 1000.0,
            b * 1000.0,
            a / b,
            t * 1000.0
        );
    }

    println!(
        "\nweighted per image: NCHW {:.3} ms   NHWC {:.3} ms   -> {:.2}x",
        tot_a * 1000.0,
        tot_b * 1000.0,
        tot_a / tot_b
    );
    println!(
        "saving {:.3} ms/img; one activation transpose per layer would cost {:.3} ms/img",
        (tot_a - tot_b) * 1000.0,
        tot_t * 1000.0
    );
    if tot_a - tot_b > tot_t {
        println!("=> even an ISLAND pays: the saving exceeds a full transpose per layer.");
    } else {
        println!(
            "=> an island does NOT pay ({:.2}x underwater); the win needs consecutive\n\
             \x20  convs to stay in NHWC so no transpose is paid between them.",
            tot_t / (tot_a - tot_b).max(1e-9)
        );
    }
    Ok(())
}
