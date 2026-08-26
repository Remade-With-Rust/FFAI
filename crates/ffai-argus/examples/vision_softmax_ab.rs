//! Vision's softmax allocates 50 MB, 204 times a caption. Can we stop?
//!
//! `siglip::forward` calls candle's `softmax_last_dim` on `(1,12,1024,1024)`.
//! That is **50.3 MB allocated, written and dropped, 12 layers x 17 tiles =
//! 10.2 GB of churn** per caption.
//!
//! Our own softmax was refuted FOUR times (0.09x-0.40x), but every one of those
//! predates `ffai_core::fastmath` and the in-place kernel `text.rs` now has.
//! The same "re-measure when the premise changed" check flipped Diana's AVX2
//! verdict and closed blocked attention for good, so it is worth one probe.
//!
//! Vision attention is NOT causal; `CausalSoftmaxInplace` with an offset of
//! `k_len - 1` degenerates to a full-row softmax, which is exactly what is
//! wanted here.
use candle_core::{Device, Tensor};
use std::time::Instant;

fn t(f: &mut dyn FnMut() -> candle_core::Result<()>) -> f64 {
    f().expect("warm");
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let s = Instant::now();
        f().expect("op");
        b = b.min(s.elapsed().as_secs_f64());
    }
    b * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    const H: usize = 12;
    const S: usize = 1024;
    let mb = (H * S * S) as f64 * 4.0 / 1e6;
    println!("vision scores (1,{H},{S},{S}) = {mb:.1} MB, 12 layers x 17 tiles\n");

    let src = Tensor::rand(-4.0f32, 4.0, (1, H, S, S), &d)?;

    // candle's: allocates a second 50 MB tensor every call.
    let a = t(&mut || {
        let _ = candle_nn::ops::softmax_last_dim(&src)?;
        Ok(())
    });
    println!("  {:<44} {a:>8.2} ms", "candle softmax_last_dim (allocates)");

    // ours, in place: needs an owned buffer, so clone first and time BOTH the
    // clone and the kernel — that is the honest comparison, because the real
    // call site already owns a fresh matmul output and pays no clone.
    let owned = src.clone();
    let b = t(&mut || {
        owned.inplace_op1(&ffai_argus::text::CausalSoftmaxProbe { offset: S - 1 })?;
        Ok(())
    });
    println!("  {:<44} {b:>8.2} ms   {:>5.2}x", "ours, IN PLACE (no allocation)", a / b);

    // correctness: full-row softmax must match candle's
    let x = Tensor::rand(-4.0f32, 4.0, (1, 2, 64, 64), &d)?;
    let want = candle_nn::ops::softmax_last_dim(&x)?.flatten_all()?.to_vec1::<f32>()?;
    let got = x.clone();
    got.inplace_op1(&ffai_argus::text::CausalSoftmaxProbe { offset: 63 })?;
    let got = got.flatten_all()?.to_vec1::<f32>()?;
    let worst = want.iter().zip(&got).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
    // THE COLUMN THAT SHIPS. The tower runs 6 tiles concurrently and tells its
    // kernels to stand down, so the SERIAL path is what production takes.
    // §20 records this exact mistake being made once already: round 1 measured
    // the parallel column and shipped the serial one, putting an 11x
    // regression on the path a 17-tile image actually runs.
    ffai_argus::siglip::set_kernels_parallel(false);
    let c = t(&mut || {
        owned.inplace_op1(&ffai_argus::text::CausalSoftmaxProbe { offset: S - 1 })?;
        Ok(())
    });
    println!("  {:<44} {c:>8.2} ms   {:>5.2}x", "ours, IN PLACE, SERIAL (what ships)", a / c);
    ffai_argus::siglip::set_kernels_parallel(true);

    println!("\n  max|diff| vs candle on a full-row softmax: {worst:.3e}");
    println!("  per caption this would remove {:.1} GB of allocation churn.", mb * 204.0 / 1e3);
    Ok(())
}
