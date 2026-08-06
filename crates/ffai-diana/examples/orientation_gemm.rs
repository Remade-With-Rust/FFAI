//! GEMM orientation on REAL conv shapes, at enough N to beat the noise floor.
//!
//! ```text
//! cargo run --release -p ffai-diana --example orientation_gemm
//! ```
//!
//! Three probes narrowed to this one. The synthetic sweep said transposing the
//! GEMM is worth 1.31-1.68x, but its N was 25,000-400,000 — real feature maps
//! are H*W = 3,840. A chain probe then read 1.03x, but its im2col was a
//! reimplementation rather than the shipped one and dominated the total, so its
//! GEMM signal was buried.
//!
//! So this times the GEMM and NOTHING else, on the shapes the graph runs, at
//! min-of-101 with the arms ABBA-interleaved.
//!
//! It carries its own noise floor: `DUPLICATES` runs the SAME shape twice under
//! different labels. Whatever separation appears between two identical shapes is
//! this box's floor, and any real effect must clear it. That is cheaper and more
//! honest than asserting a floor measured at some other time.

use candle_core::{Device, Result, Tensor};

/// (cin, cout, h, w, calls/img) from FFAI_DIANA_ROOFLINE=1 at 640x384.
const SHAPES: &[(usize, usize, usize, usize, usize)] = &[
    (32, 32, 24, 40, 12),
    (16, 16, 48, 80, 5),
    (64, 64, 12, 20, 4),
    (64, 16, 48, 80, 1),
    (16, 8, 96, 160, 1),
    (128, 16, 24, 40, 1),
    (32, 16, 48, 80, 1),
    (16, 32, 48, 80, 1),
    // duplicates of two rows above: the floor
    (32, 32, 24, 40, 0),
    (16, 16, 48, 80, 0),
];

fn main() -> Result<()> {
    let dev = Device::Cpu;
    const REPS: usize = 101;
    println!("GEMM only, real shapes, min-of-{REPS}, ABBA-interleaved");
    println!("last two rows are DUPLICATES of rows 1-2 => this box's floor\n");
    println!(
        "{:>10} {:>9} {:>6} {:>10} {:>10} {:>9}",
        "cin->cout", "HxW", "calls", "M=cout", "M=HW", "speedup"
    );

    let (mut wa, mut wb) = (0.0f64, 0.0f64);
    for &(cin, cout, h, w, calls) in SHAPES {
        let (hw, k) = (h * w, 9 * cin);
        let wn = Tensor::rand(-0.5f32, 0.5f32, (cout, k), &dev)?;
        let col_n = Tensor::rand(-0.5f32, 0.5f32, (k, hw), &dev)?;
        let col_t = Tensor::rand(-0.5f32, 0.5f32, (hw, k), &dev)?;
        let wt = Tensor::rand(-0.5f32, 0.5f32, (k, cout), &dev)?;
        let _ = wn.matmul(&col_n)?;
        let _ = col_t.matmul(&wt)?;

        let (mut a, mut b) = (f64::MAX, f64::MAX);
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let c = wn.matmul(&col_n)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            a = a.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let c = col_t.matmul(&wt)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            b = b.min(t.elapsed().as_secs_f64());
        }
        wa += a * calls as f64;
        wb += b * calls as f64;
        println!(
            "{:>4}->{:<5} {:>4}x{:<4} {:>6} {:>10.4} {:>10.4} {:>8.2}x",
            cin, cout, h, w, calls, a * 1000.0, b * 1000.0, a / b
        );
    }
    println!(
        "\nweighted by calls/image: M=cout {:.3} ms   M=HW {:.3} ms   -> {:.3}x",
        wa * 1000.0,
        wb * 1000.0,
        wa / wb
    );
    println!(
        "GEMM is ~29.8% of detect, so a whole-graph NHWC conversion is worth\n\
         about {:.1}% of detect BEFORE boundary transposes and any change in\n\
         im2col cost — neither of which is in this number.",
        100.0 * 0.298 * (1.0 - wb / wa)
    );
    Ok(())
}
