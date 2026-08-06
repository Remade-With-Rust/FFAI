//! Layout assignment over the REAL layer sequence, solved rather than bounded.
//!
//! The per-shape dispatch measured a 1.173x ceiling on the GEMM assuming every
//! layer freely picks its best layout. That is only reachable if neighbours
//! agree: when layer i emits NHWC and layer i+1 wants NCHW, something has to
//! transpose the activation between them.
//!
//! So this emits, per layer in EXECUTION ORDER, the three numbers a solver
//! needs - GEMM cost in each orientation, and the cost of transposing that
//! layers output activation. The assignment is then a shortest path over
//! 2 states per layer, which is exact rather than an upper bound.

use candle_core::{Device, Result, Tensor};

/// (cin, cout, hout, wout, taps) in execution order, from FFAI_DIANA_ROOFLINE=1.
const SEQ: &[(usize, usize, usize, usize, usize)] = &[
    (3, 16, 192, 320, 9),
    (16, 32, 96, 160, 9),
    (32, 32, 96, 160, 1),
    (16, 8, 96, 160, 9),
    (8, 16, 96, 160, 9),
    (48, 64, 96, 160, 1),
    (64, 64, 48, 80, 9),
    (64, 64, 48, 80, 1),
    (32, 16, 48, 80, 9),
    (16, 32, 48, 80, 9),
    (96, 128, 48, 80, 1),
    (128, 128, 24, 40, 9),
    (128, 128, 24, 40, 1),
    (64, 32, 24, 40, 1),
    (32, 32, 24, 40, 9),
    (32, 32, 24, 40, 9),
    (32, 32, 24, 40, 9),
    (32, 32, 24, 40, 9),
    (64, 32, 24, 40, 1),
    (64, 64, 24, 40, 1),
    (192, 128, 24, 40, 1),
    (128, 256, 12, 20, 9),
    (256, 256, 12, 20, 1),
    (128, 64, 12, 20, 1),
    (64, 64, 12, 20, 9),
    (64, 64, 12, 20, 9),
    (64, 64, 12, 20, 9),
    (64, 64, 12, 20, 9),
    (128, 64, 12, 20, 1),
    (128, 128, 12, 20, 1),
    (384, 256, 12, 20, 1),
    (256, 128, 12, 20, 1),
    (512, 256, 12, 20, 1),
    (256, 256, 12, 20, 1),
    (128, 256, 12, 20, 1),
    (128, 128, 12, 20, 1),
    (128, 256, 12, 20, 1),
    (256, 128, 12, 20, 1),
    (256, 256, 12, 20, 1),
    (384, 128, 24, 40, 1),
    (64, 32, 24, 40, 1),
    (32, 32, 24, 40, 9),
    (32, 32, 24, 40, 9),
    (32, 32, 24, 40, 9),
    (32, 32, 24, 40, 9),
    (64, 32, 24, 40, 1),
    (64, 64, 24, 40, 1),
    (192, 128, 24, 40, 1),
    (256, 64, 48, 80, 1),
    (32, 16, 48, 80, 1),
    (16, 16, 48, 80, 9),
    (16, 16, 48, 80, 9),
    (16, 16, 48, 80, 9),
    (16, 16, 48, 80, 9),
    (32, 16, 48, 80, 1),
    (32, 32, 48, 80, 1),
    (96, 64, 48, 80, 1),
    (64, 64, 24, 40, 9),
    (192, 128, 24, 40, 1),
    (64, 32, 24, 40, 1),
    (32, 32, 24, 40, 9),
    (32, 32, 24, 40, 9),
    (32, 32, 24, 40, 9),
    (32, 32, 24, 40, 9),
    (64, 32, 24, 40, 1),
    (64, 64, 24, 40, 1),
    (192, 128, 24, 40, 1),
    (128, 128, 12, 20, 9),
    (384, 256, 12, 20, 1),
    (128, 64, 12, 20, 9),
    (64, 128, 12, 20, 9),
    (128, 256, 12, 20, 1),
    (128, 128, 12, 20, 1),
    (128, 256, 12, 20, 1),
    (256, 128, 12, 20, 1),
    (384, 256, 12, 20, 1),
    (64, 16, 48, 80, 9),
    (16, 16, 48, 80, 9),
    (64, 80, 48, 80, 1),
    (80, 80, 48, 80, 1),
    (128, 16, 24, 40, 9),
    (16, 16, 24, 40, 9),
    (128, 80, 24, 40, 1),
    (80, 80, 24, 40, 1),
    (256, 16, 12, 20, 9),
    (16, 16, 12, 20, 9),
    (256, 80, 12, 20, 1),
    (80, 80, 12, 20, 1)
];

fn main() -> Result<()> {
    let dev = Device::Cpu;
    const REPS: usize = 15;
    println!("idx cin cout hout wout nchw_ms nhwc_ms transpose_ms");
    for (i, &(cin, cout, h, w, taps)) in SEQ.iter().enumerate() {
        let (hw, k) = (h * w, taps * cin);
        let wn = Tensor::rand(-0.5f32, 0.5f32, (cout, k), &dev)?;
        let cn = Tensor::rand(-0.5f32, 0.5f32, (k, hw), &dev)?;
        let ct = Tensor::rand(-0.5f32, 0.5f32, (hw, k), &dev)?;
        let wt = Tensor::rand(-0.5f32, 0.5f32, (k, cout), &dev)?;
        let act = Tensor::rand(-0.5f32, 0.5f32, (cout, hw), &dev)?;
        let _ = wn.matmul(&cn)?;
        let _ = ct.matmul(&wt)?;
        let (mut a, mut b, mut t2) = (f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..REPS {
            let s = std::time::Instant::now();
            let c = wn.matmul(&cn)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            a = a.min(s.elapsed().as_secs_f64());
            let s = std::time::Instant::now();
            let c = ct.matmul(&wt)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            b = b.min(s.elapsed().as_secs_f64());
            let s = std::time::Instant::now();
            let c = act.t()?.contiguous()?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            t2 = t2.min(s.elapsed().as_secs_f64());
        }
        println!("{i} {cin} {cout} {h} {w} {:.5} {:.5} {:.5}", a*1000.0, b*1000.0, t2*1000.0);
    }
    Ok(())
}
