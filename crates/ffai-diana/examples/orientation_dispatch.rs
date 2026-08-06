//! Per-shape orientation dispatch: is the layout preference REPRODUCIBLE?
//!
//! The whole-probe orientation ratio is 1.047x, inside a +-25% floor. But an
//! aggregate hides a dispatch: if some shapes reliably prefer one layout and
//! others reliably prefer the other, averaging them reports neither.
//!
//! And the shape is known when the model LOADS - every dimension is static.
//! So this is not content-adaptive dispatch with a runtime probe; it is a
//! per-layer constant chosen offline, at zero runtime cost. That makes the bar
//! lower than usual: a reproducible 1.2x on some layers is bankable.
//!
//! The discipline that makes it honest is REPRODUCIBILITY, not size. Each shape
//! is measured in R independent passes with the arms ABBA-interleaved. A shape
//! counts as dispatchable only if every pass agrees on the SIGN. Anything that
//! sign-flips is noise, however large it looks in one pass.

use candle_core::{Device, Result, Tensor};

/// (cin, cout, hout, wout, calls/img, taps) from FFAI_DIANA_ROOFLINE=1.
const SHAPES: &[(usize, usize, usize, usize, usize, usize)] = &[
(3, 16, 192, 320, 1, 9),
(16, 32, 96, 160, 1, 9),
(64, 64, 48, 80, 1, 9),
(32, 32, 24, 40, 12, 9),
(128, 128, 24, 40, 1, 9),
(48, 64, 96, 160, 1, 1),
(16, 8, 96, 160, 1, 9),
(192, 128, 24, 40, 4, 1),
(16, 16, 48, 80, 5, 9),
(64, 16, 48, 80, 1, 9),
(128, 256, 12, 20, 4, 1),
(256, 64, 48, 80, 1, 1),
(384, 256, 12, 20, 3, 1),
(32, 32, 96, 160, 1, 1),
(128, 256, 12, 20, 1, 9),
(384, 128, 24, 40, 1, 1),
(256, 256, 12, 20, 3, 1),
(8, 16, 96, 160, 1, 9),
(96, 128, 48, 80, 1, 1),
(64, 64, 24, 40, 1, 9),
(128, 128, 12, 20, 1, 9),
(64, 64, 12, 20, 4, 9),
(32, 16, 48, 80, 1, 9),
(512, 256, 12, 20, 1, 1),
(128, 16, 24, 40, 1, 9),
(128, 80, 24, 40, 1, 1),
(96, 64, 48, 80, 1, 1),
(64, 64, 24, 40, 3, 1),
(64, 32, 24, 40, 6, 1),
(64, 80, 48, 80, 1, 1),
(16, 32, 48, 80, 1, 9),
(80, 80, 48, 80, 1, 1),
(256, 128, 12, 20, 3, 1),
(256, 16, 12, 20, 1, 9),
(64, 64, 48, 80, 1, 1),
(128, 64, 12, 20, 1, 9),
(128, 128, 12, 20, 3, 1),
(128, 128, 24, 40, 1, 1),
(64, 128, 12, 20, 1, 9),
(32, 16, 48, 80, 2, 1),
(32, 32, 48, 80, 1, 1),
(80, 80, 24, 40, 1, 1),
(256, 80, 12, 20, 1, 1),
(128, 64, 12, 20, 2, 1),
(16, 16, 24, 40, 1, 9),
(80, 80, 12, 20, 1, 1),
(16, 16, 12, 20, 1, 9),
];

fn main() -> Result<()> {
    let dev = Device::Cpu;
    const R: usize = 4;
    const REPS: usize = 25;
    println!("{} shapes, {} independent passes, min-of-{} each, ABBA
", SHAPES.len(), R, REPS);

    let mut wins = Vec::new();
    let (mut base_ms, mut disp_ms) = (0.0f64, 0.0f64);
    for &(cin, cout, h, w, calls, taps) in SHAPES {
        let (hw, k) = (h * w, taps * cin);
        let wn = Tensor::rand(-0.5f32, 0.5f32, (cout, k), &dev)?;
        let cn = Tensor::rand(-0.5f32, 0.5f32, (k, hw), &dev)?;
        let ct = Tensor::rand(-0.5f32, 0.5f32, (hw, k), &dev)?;
        let wt = Tensor::rand(-0.5f32, 0.5f32, (k, cout), &dev)?;
        let _ = wn.matmul(&cn)?;
        let _ = ct.matmul(&wt)?;

        let mut sp = Vec::new();
        let (mut aa, mut bb) = (f64::MAX, f64::MAX);
        for _ in 0..R {
            let (mut a, mut b) = (f64::MAX, f64::MAX);
            for _ in 0..REPS {
                let t = std::time::Instant::now();
                let c = wn.matmul(&cn)?;
                let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
                a = a.min(t.elapsed().as_secs_f64());
                let t = std::time::Instant::now();
                let c = ct.matmul(&wt)?;
                let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
                b = b.min(t.elapsed().as_secs_f64());
            }
            sp.push(a / b);
            aa = aa.min(a);
            bb = bb.min(b);
        }
        // Dispatchable only if EVERY pass agrees on the sign.
        let all_nhwc = sp.iter().all(|&s| s > 1.0);
        let all_nchw = sp.iter().all(|&s| s < 1.0);
        let c = calls as f64;
        base_ms += aa * c;
        disp_ms += if all_nhwc { bb * c } else { aa * c };
        if all_nhwc || all_nchw {
            wins.push((cin, cout, h, w, calls, sp.clone(), all_nhwc, aa, bb));
        }
    }

    println!("REPRODUCIBLE shapes (every pass agrees on the sign):");
    println!("{:>10} {:>9} {:>6} {:>8} {:>26}", "cin->cout", "HxW", "calls", "prefers", "passes");
    for (cin, cout, h, w, calls, sp, nhwc, _, _) in &wins {
        let s: Vec<String> = sp.iter().map(|v| format!("{v:.2}")).collect();
        println!(
            "{:>4}->{:<5} {:>4}x{:<4} {:>6} {:>8} {:>26}",
            cin, cout, h, w, calls,
            if *nhwc { "NHWC" } else { "NCHW" },
            s.join(" ")
        );
    }
    let n_nhwc = wins.iter().filter(|w| w.6).count();
    println!(
        "
{} of {} shapes reproducible; {} prefer NHWC, {} prefer NCHW; {} sign-flip (noise)",
        wins.len(), SHAPES.len(), n_nhwc, wins.len() - n_nhwc, SHAPES.len() - wins.len()
    );
    println!(
        "
PRIZE of a perfect per-shape dispatch, weighted by calls/image:
           all-NCHW {:.3} ms   dispatched {:.3} ms   -> {:.3}x  ({:.2}% of GEMM)",
        base_ms * 1000.0, disp_ms * 1000.0, base_ms / disp_ms,
        100.0 * (1.0 - disp_ms / base_ms)
    );
    println!(
        "  GEMM is ~29.8% of detect, so this dispatch is worth {:.2}% of detect.",
        29.8 * (1.0 - disp_ms / base_ms)
    );
    Ok(())
}
