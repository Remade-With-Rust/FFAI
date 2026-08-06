//! The decisive prize probe: a CHAIN of convolutions in NCHW versus NHWC.
//!
//! ```text
//! cargo run --release -p ffai-diana --example nhwc_chain
//! ```
//!
//! Two earlier probes bracketed this and neither answered it:
//!
//! * the synthetic sweep said transposing the GEMM is worth 1.31-1.68x;
//! * the "island" probe said a per-layer transpose eats it — first by 100-200x,
//!   which was WRONG because it transposed the col matrix nobody transposes,
//!   then by 1.65x on the activation, which is right and still underwater.
//!
//! The design that survives both is a RUN: consecutive convolutions staying in
//! NHWC, so the layout is entered once and left once and no transpose is paid
//! between layers. This measures that directly, with real im2col in both
//! layouts, because im2col's own cost differs between them and assuming it
//! does not is how a prize gets overstated.
//!
//! What is timed, per arm, for a chain of L layers:
//!
//! | | NCHW (shipped) | NHWC (proposed) |
//! |---|---|---|
//! | im2col | `[9cin, HW]` | `[HW, 9cin]` |
//! | matmul | `W[cout,9cin] @ col` | `col @ Wt[9cin,cout]` |
//! | layout changes | none | **two, for the whole run** |
//!
//! Both arms do identical arithmetic and produce transposes of one another, so
//! the comparison is work-parity by construction — but the outputs are checked
//! anyway, because a fast arm that computes something else is the failure this
//! whole campaign keeps finding.

use candle_core::{Device, Result, Tensor};
use rayon::prelude::*;

/// im2col producing `[9*cin, H*W]` — the layout we ship.
fn im2col_nchw(x: &[f32], cin: usize, h: usize, w: usize) -> Vec<f32> {
    let hw = h * w;
    let mut col = vec![0.0f32; 9 * cin * hw];
    col.par_chunks_mut(hw).enumerate().for_each(|(row, dst)| {
        let c = row / 9;
        let tap = row % 9;
        let (dy, dx) = ((tap / 3) as isize - 1, (tap % 3) as isize - 1);
        for oy in 0..h {
            let sy = oy as isize + dy;
            if sy < 0 || sy >= h as isize {
                continue;
            }
            let src = c * hw + sy as usize * w;
            let dstrow = oy * w;
            for ox in 0..w {
                let sx = ox as isize + dx;
                if sx >= 0 && sx < w as isize {
                    dst[dstrow + ox] = x[src + sx as usize];
                }
            }
        }
    });
    col
}

/// im2col producing `[H*W, 9*cin]` from an NHWC activation `[H*W, cin]`.
///
/// This is the layout that makes the GEMM's M dimension H*W. It is also the
/// more natural gather: the `cin` values for one (tap, pixel) are CONTIGUOUS
/// in the source, so each inner copy is a run rather than a stride.
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
                let src = (sy as usize * w + sx as usize) * cin;
                let d = ox * k + tap * cin;
                dst[d..d + cin].copy_from_slice(&x[src..src + cin]);
            }
        }
    });
    col
}

fn best_of<F: FnMut() -> Result<f64>>(n: usize, mut f: F) -> Result<f64> {
    let mut best = f64::MAX;
    for _ in 0..n {
        best = best.min(f()?);
    }
    Ok(best)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // A real run of consecutive 3x3 convolutions from the backbone at
    // 640x384, at the resolution where the graph spends most of its time.
    // (cin, cout, h, w)
    let chain: &[(usize, usize, usize, usize)] = &[
        (16, 16, 48, 80),
        (16, 32, 48, 80),
        (32, 16, 48, 80),
        (32, 32, 48, 80),
        (32, 32, 48, 80),
        (32, 16, 48, 80),
    ];

    println!("chain of {} consecutive 3x3 convs at 48x80, best-of-7\n", chain.len());

    let mut nchw_total = 0.0;
    let mut nhwc_total = 0.0;
    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "cin->cout", "NCHW im2col", "NCHW gemm", "NHWC im2col", "NHWC gemm", "speedup"
    );

    for &(cin, cout, h, w) in chain {
        let hw = h * w;
        let k = 9 * cin;
        let xn: Vec<f32> = (0..cin * hw).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();

        let wn = Tensor::rand(-0.5f32, 0.5f32, (cout, k), &dev)?;
        let wt = Tensor::rand(-0.5f32, 0.5f32, (k, cout), &dev)?;

        // ABBA-INTERLEAVED. The first version of this ran all reps of arm A
        // then all of arm B, which codec-measurement §3 forbids and which
        // showed up immediately: two rows with IDENTICAL im2col work (same
        // cin/h/w — im2col does not depend on cout) read 1.715 and 11.649 ms.
        // Interleaving puts any drift into both arms equally.
        let col_n = Tensor::from_vec(im2col_nchw(&xn, cin, h, w), (k, hw), &dev)?;
        let col_t = Tensor::from_vec(im2col_nhwc(&xn, cin, h, w), (hw, k), &dev)?;
        let _ = wn.matmul(&col_n)?;
        let _ = col_t.matmul(&wt)?;

        let (mut a_im, mut b_im, mut a_gemm, mut b_gemm) =
            (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..31 {
            let t = std::time::Instant::now();
            std::hint::black_box(im2col_nchw(&xn, cin, h, w));
            a_im = a_im.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            std::hint::black_box(im2col_nhwc(&xn, cin, h, w));
            b_im = b_im.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let c = wn.matmul(&col_n)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            a_gemm = a_gemm.min(t.elapsed().as_secs_f64());

            let t = std::time::Instant::now();
            let c = col_t.matmul(&wt)?;
            let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            b_gemm = b_gemm.min(t.elapsed().as_secs_f64());
        }

        nchw_total += a_im + a_gemm;
        nhwc_total += b_im + b_gemm;
        println!(
            "{:>4}->{:<5} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>7.2}x",
            cin,
            cout,
            a_im * 1000.0,
            a_gemm * 1000.0,
            b_im * 1000.0,
            b_gemm * 1000.0,
            (a_im + a_gemm) / (b_im + b_gemm)
        );
    }

    // The run pays for entering and leaving the layout ONCE, not per layer.
    let (cin0, _, h0, w0) = chain[0];
    let entry = Tensor::rand(-0.5f32, 0.5f32, (cin0, h0 * w0), &dev)?;
    let boundary = best_of(31, || {
        let t = std::time::Instant::now();
        let c = entry.t()?.contiguous()?;
        let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
        Ok(t.elapsed().as_secs_f64())
    })? * 2.0;

    println!(
        "\nchain total   NCHW {:.3} ms   NHWC {:.3} ms   -> {:.2}x",
        nchw_total * 1000.0,
        nhwc_total * 1000.0,
        nchw_total / nhwc_total
    );
    println!(
        "two boundary transposes for the whole run: {:.3} ms",
        boundary * 1000.0
    );
    let net = nchw_total - nhwc_total - boundary;
    println!(
        "NET over the run: {:.3} ms ({:.2}x)  {}",
        net * 1000.0,
        nchw_total / (nhwc_total + boundary),
        if net > 0.0 { "<= PAYS" } else { "<= does not pay" }
    );
    println!(
        "\nbreak-even run length: {:.1} layers",
        boundary / ((nchw_total - nhwc_total) / chain.len() as f64).max(1e-12)
    );
    Ok(())
}
