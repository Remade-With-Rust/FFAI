//! Can candle matmul a TRANSPOSED VIEW without materialising it?
//!
//! If yes, the NHWC GEMM orientation is available with the SHIPPED NCHW
//! im2col and no transpose anywhere: build `col[K, OHW]` the fast way, hand
//! `col.t()` to matmul, and get `[OHW, Cout]`. That would deliver the entire
//! orientation win with no layout conversion, no NHWC im2col, and no run.
//!
//! candle is backed by the `gemm` crate, which takes arbitrary row/column
//! strides, so this is not obviously impossible. It is also exactly the kind of
//! thing that silently calls `.contiguous()` underneath and costs more than it
//! saves, which is why it is measured rather than assumed.

use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let dev = Device::Cpu;
    const N: usize = 25;
    // (cin, cout, hout, wout) - real shapes.
    for &(cin, cout, h, w) in &[(32usize, 32usize, 48usize, 80usize),
                                (16, 16, 48, 80), (64, 64, 24, 40), (32, 64, 96, 160)] {
        let (hw, k) = (h * w, 9 * cin);
        let wn = Tensor::rand(-0.5f32, 0.5f32, (cout, k), &dev)?;
        let col = Tensor::rand(-0.5f32, 0.5f32, (k, hw), &dev)?;   // shipped layout
        let w_t = wn.t()?.contiguous()?;                            // hoisted to load
        let col_t_mat = col.t()?.contiguous()?;                     // materialised

        let mut base = f64::MAX;
        let mut view = f64::MAX;
        let mut mat = f64::MAX;
        let _ = wn.matmul(&col)?;
        for _ in 0..N {
            let t = std::time::Instant::now();
            let c = wn.matmul(&col)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            base = base.min(t.elapsed().as_secs_f64());

            // The question: a transposed VIEW straight into matmul.
            let t = std::time::Instant::now();
            let c = col.t()?.matmul(&w_t)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            view = view.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let c = col_t_mat.matmul(&w_t)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            mat = mat.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{cin:>4}->{cout:<4} {h}x{w:<4}  NCHW {:>8.4} ms   view-t {:>8.4} ms ({:.2}x)   materialised {:>8.4} ms ({:.2}x)",
            base * 1e3, view * 1e3, base / view, mat * 1e3, base / mat
        );
    }
    println!("\nview-t > 1.00x means the transposed VIEW is faster than the shipped");
    println!("orientation - the whole win with no layout conversion at all.");
    Ok(())
}
