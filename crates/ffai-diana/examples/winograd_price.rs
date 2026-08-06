//! Winograd F(2x2,3x3), priced on arithmetic BEFORE anything is built.
//!
//! ```text
//! cargo run --release -p ffai-diana --example winograd_price
//! ```
//!
//! Tonight's NHWC campaign ended on a break-even calculation whose every input
//! was measurable before a line of the conversion was written, and which would
//! have priced it out in an afternoon. Winograd gets that calculation first.
//!
//! # What Winograd actually changes
//!
//! F(2x2,3x3) produces a 2x2 output tile from a 4x4 input tile with **16
//! element-wise multiplies instead of 36** (4 outputs x 9 taps) - a 2.25x
//! reduction in multiplies. The multiplies become 16 independent GEMMs, one per
//! position in the 4x4 tile:
//!
//! | | current | Winograd |
//! |---|---|---|
//! | GEMMs per conv | 1 | **16** |
//! | M | cout | cout |
//! | K | **9 * cin** | **cin** |
//! | N | H*W | **H*W / 4** |
//!
//! **M does not change.** The per-layer roofline found effective GFLOP/s
//! correlates +0.823 with log2(cout) and -0.048 with operand size, so Winograd
//! does not touch the axis that governs this graph's GEMM efficiency - and it
//! shrinks BOTH of the other two dimensions by 9x and 4x. Smaller GEMMs run
//! worse. That is the risk, and it is measurable now.
//!
//! # The break-even
//!
//! Winograd wins only if
//!
//! ```text
//!   t(16 batched small GEMMs) + t(input transform) + t(output transform)
//!     <  t(1 large GEMM) + t(im2col)
//! ```
//!
//! The GEMM terms are measured here directly. The transform terms are counted:
//! the input transform is ~32 adds per (tile, cin) and the output ~24 per
//! (tile, cout), against the direct convolution's 9*cin*cout*HW multiplies. The
//! filter transform is once at load and free.
//!
//! Transforms are counted rather than timed because they are elementwise and
//! memory-bound, so an op count is a LOWER bound on their cost - which makes
//! any verdict against Winograd conservative and any verdict for it provisional.

use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let dev = Device::Cpu;
    const REPS: usize = 21;

    // (cin, cout, h, w, calls/img) - the 3x3 stride-1 shapes Winograd applies
    // to, from FFAI_DIANA_ROOFLINE=1 at 640x384.
    let shapes = [
        (32usize, 32usize, 24usize, 40usize, 12usize),
        (16, 16, 48, 80, 5),
        (64, 64, 12, 20, 4),
        (64, 16, 48, 80, 1),
        (16, 8, 96, 160, 1),
        (128, 16, 24, 40, 1),
        (32, 16, 48, 80, 1),
        (16, 32, 48, 80, 1),
    ];

    println!("Winograd F(2x2,3x3) priced against the shipped im2col+GEMM\n");
    println!(
        "{:>10} {:>9} {:>6} {:>10} {:>12} {:>9} {:>9}",
        "cin->cout", "HxW", "calls", "1 big GEMM", "16 small GEMM", "gemm x", "xform %"
    );

    let (mut tot_big, mut tot_wino, mut tot_xform) = (0.0f64, 0.0f64, 0.0f64);

    for &(cin, cout, h, w, calls) in &shapes {
        let (hw, k) = (h * w, 9 * cin);
        let ntiles = hw / 4;

        // Current: one GEMM, [cout, 9cin] @ [9cin, HW].
        let wn = Tensor::rand(-0.5f32, 0.5f32, (cout, k), &dev)?;
        let col = Tensor::rand(-0.5f32, 0.5f32, (k, hw), &dev)?;

        // Winograd: 16 GEMMs of [cout, cin] @ [cin, ntiles], as one batched
        // matmul, which is how any real implementation issues them.
        let gw = Tensor::rand(-0.5f32, 0.5f32, (16, cout, cin), &dev)?;
        let gv = Tensor::rand(-0.5f32, 0.5f32, (16, cin, ntiles), &dev)?;

        let _ = wn.matmul(&col)?;
        let _ = gw.matmul(&gv)?;

        let (mut a, mut b) = (f64::MAX, f64::MAX);
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let c = wn.matmul(&col)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            a = a.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let c = gw.matmul(&gv)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            b = b.min(t.elapsed().as_secs_f64());
        }

        // Transform op counts against the direct multiply count, as a fraction.
        // input:  ~32 adds per (tile, cin);  output: ~24 per (tile, cout)
        let xform_ops = ntiles as f64 * (32.0 * cin as f64 + 24.0 * cout as f64);
        let direct_mults = 9.0 * cin as f64 * cout as f64 * hw as f64;
        let xform_pct = 100.0 * xform_ops / direct_mults;

        tot_big += a * calls as f64;
        tot_wino += b * calls as f64;
        // Price the transform at the big GEMM's own throughput - generous to
        // Winograd, since elementwise passes do not reach GEMM throughput.
        tot_xform += a * (xform_ops / (2.0 * cout as f64 * k as f64 * hw as f64)) * calls as f64;

        println!(
            "{:>4}->{:<5} {:>4}x{:<4} {:>6} {:>10.4} {:>12.4} {:>8.2}x {:>8.1}%",
            cin,
            cout,
            h,
            w,
            calls,
            a * 1e3,
            b * 1e3,
            a / b,
            xform_pct
        );
    }

    println!(
        "\nweighted per image:  1 big GEMM {:.3} ms   16 small {:.3} ms   -> {:.2}x",
        tot_big * 1e3,
        tot_wino * 1e3,
        tot_big / tot_wino
    );
    println!("transform cost, priced at GEMM throughput: {:.3} ms", tot_xform * 1e3);
    let net = tot_big - tot_wino - tot_xform;
    println!(
        "\nGEMM saving {:+.3} ms - transforms {:.3} ms = NET {:+.3} ms/img  {}",
        (tot_big - tot_wino) * 1e3,
        tot_xform * 1e3,
        net * 1e3,
        if net > 0.0 { "<= worth building" } else { "<= DEAD before it is built" }
    );
    println!(
        "\nIdeal arithmetic says 2.25x on the multiplies. Anything far below that\n\
         is the 16 small GEMMs losing what the multiply count saved."
    );
    Ok(())
}
