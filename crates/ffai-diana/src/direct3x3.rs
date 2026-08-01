//! Direct 3x3 convolution — no im2col, no expanded operand.
//!
//! # Why this exists
//!
//! `D4c` in `docs/whys/diana-latency.md`: every convolution GEMM we issue is
//! MEMORY-bound, and im2col is the reason. The GEMM's B operand *is* the
//! ninefold-expanded activation, so it reads nine times the bytes for the
//! same arithmetic. Measured intensity collapses from ~92 flops/byte on a
//! balanced GEMM to **5-8** on ours, and nothing reaches peak at 5:
//!
//! | shape | flops/byte | GFLOP/s (of ~128 peak) |
//! |---|---:|---:|
//! | 256x1152x1600 (balanced) | 92 | 78.0 |
//! | 16x27x102400 (our stem) | 5 | 27.5 |
//!
//! Tiling was tried and could not fix this: it changes where the buffer
//! lives, not how many bytes the GEMM reads. The only lever that moves
//! intensity is not building the operand at all.
//!
//! # The structure, and why it is this one
//!
//! The contraction `(c_in, ky, kx)` is the OUTER loop and the output columns
//! are innermost. That is deliberate — the skill's rule is that dot products
//! ending in a horizontal reduction are slow, so the accumulator is held
//! across the contraction and each step is a broadcast-and-AXPY over a
//! contiguous row.
//!
//! * `acc[OX]` for one output-channel and a block of `OX` columns stays in
//!   registers across the whole `(c_in, ky, kx)` loop.
//! * At stride 1 the source row for a given `kx` is CONTIGUOUS and merely
//!   offset, so the inner loop is `acc[j] += wv * row[j + off]` — an FMA per
//!   load, which is what AVX2 wants.
//! * The activation is read once per `(output row block, c_in)` and stays in
//!   L1/L2 across the nine taps and every output channel, so DRAM traffic is
//!   the input size rather than nine times it.
//!
//! Interior columns are separated from the two border columns so the hot
//! loop carries no bounds test — a branch there would stop it vectorising,
//! which is the whole point.
//!
//! Stride 2 still goes through im2col: it is 105 of 585 convolution calls
//! and its gather cannot use the contiguous-row trick, so it is a separate
//! question rather than a copy-paste.

use candle_core::{Result, Tensor};
use rayon::prelude::*;

/// Output columns accumulated at once. 32 f32 = 4 AVX2 registers, which
/// leaves room for the broadcast weight and the loaded row.
const OX: usize = 32;

/// **OFF by default: measured 1.73x SLOWER, and the loop order is why.**
///
/// Serial CPU over 24 images, best of 3:
///
/// | path | total CPU | per image |
/// |---|---:|---:|
/// | im2col + candle GEMM | 6234 ms | 260 ms |
/// | this direct kernel | 10766 ms | **448 ms** |
///
/// The premise stands — im2col collapses arithmetic intensity and a direct
/// convolution is the only lever that restores it. **This implementation
/// gets the blocking backwards.** It parallelises over OUTPUT CHANNELS
/// (`chunks_mut(oh * ow)` hands each task one channel), so every task walks
/// the entire activation, and the activation is therefore read `c_out`
/// times: `c_out * c_in * 9` row reads against im2col's `c_in * 9`. That is
/// *worse* intensity than the thing it was written to beat, which is why it
/// loses by more than the GEMM's tuning advantage alone would explain.
///
/// The correct structure is the mirror image: block over output ROWS, hold
/// accumulators for ALL output channels across one row (`c_out * ow` floats
/// — 328 KB at the widest n-tier layer, so L2-resident), and stream the
/// activation ONCE, issuing `c_out` broadcast-AXPYs per `(ic, ky, kx)` from
/// a row that is already in L1.
///
/// Kept behind `FFAI_DIANA_DIRECT=1` rather than deleted: the correctness
/// gate below already passes against candle's own `Conv2d`, so the next
/// attempt inherits a proven-correct kernel and only has to re-block it.
pub fn enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_DIRECT").is_ok_and(|v| v == "1");
            C.store(on as u8, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

/// Stride-1, padding-1, 3x3 convolution computed directly.
///
/// Returns `None` for shapes this kernel does not handle, so the caller can
/// fall back to im2col rather than this silently producing something else.
pub fn conv3x3_direct(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    let (n, c_in, h, w) = x.dims4()?;
    let c_out = weight.dim(0)?;
    if n != 1 || w < 2 {
        return Ok(None);
    }
    let (oh, ow) = (h, w); // stride 1, padding 1, kernel 3

    let ws = weight.flatten_all()?.to_vec1::<f32>()?;
    let bs = match bias {
        Some(b) => b.flatten_all()?.to_vec1::<f32>()?,
        None => vec![0.0; c_out],
    };

    let out = crate::cpuop::SliceOp::new("ffai-direct3x3", move |xs, _| {
        let mut out = vec![0f32; c_out * oh * ow];
        let plane = |c: usize, y: usize| &xs[c * h * w + y * w..c * h * w + (y + 1) * w];

        let row_kernel = |(oc, orow): (usize, &mut [f32])| {
            let b = bs[oc];
            for oy in 0..oh {
                let dst = &mut orow[oy * ow..(oy + 1) * ow];
                dst.fill(b);
                for ic in 0..c_in {
                    let wbase = (oc * c_in + ic) * 9;
                    for ky in 0..3usize {
                        // Source row for this vertical tap; outside the
                        // image the contribution is zero (the padding).
                        let sy = oy + ky;
                        if sy == 0 || sy > h {
                            continue;
                        }
                        let src = plane(ic, sy - 1);
                        for kx in 0..3usize {
                            let wv = ws[wbase + ky * 3 + kx];
                            if wv == 0.0 {
                                continue;
                            }
                            // sx = ox + kx - 1, so the interior is where
                            // both ends stay inside [0, w).
                            let lo = 1usize.saturating_sub(kx);
                            let hi = ow.min(w + 1 - kx);
                            let s0 = lo + kx - 1;
                            // Contiguous, offset, no bounds test: this is
                            // the loop that has to vectorise.
                            let d = &mut dst[lo..hi];
                            let s = &src[s0..s0 + (hi - lo)];
                            let mut j = 0;
                            while j + OX <= d.len() {
                                for t in 0..OX {
                                    d[j + t] += wv * s[j + t];
                                }
                                j += OX;
                            }
                            for t in j..d.len() {
                                d[t] += wv * s[t];
                            }
                        }
                    }
                }
            }
        };

        if crate::parallel::serial_kernels() {
            out.chunks_mut(oh * ow).enumerate().for_each(row_kernel);
        } else {
            out.par_chunks_mut(oh * ow).enumerate().for_each(row_kernel);
        }
        Ok((out, (1, c_out, oh, ow).into()))
    })
    .run(x)?;
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{Conv2d, Conv2dConfig, Module};

    /// candle's own convolution is the oracle, exactly as it is for the
    /// im2col path — a faster kernel that computes something else is not a
    /// faster kernel.
    #[test]
    fn matches_candle_conv2d() {
        let dev = Device::Cpu;
        for (c_in, c_out, h, w) in [(3usize, 16usize, 20usize, 24usize), (8, 8, 13, 13), (16, 4, 7, 9)] {
            let xs: Vec<f32> =
                (0..c_in * h * w).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
            let ws: Vec<f32> =
                (0..c_out * c_in * 9).map(|i| ((i * 17 % 63) as f32 - 31.0) / 40.0).collect();
            let bs: Vec<f32> = (0..c_out).map(|i| (i as f32) * 0.05 - 0.1).collect();
            let x = Tensor::from_vec(xs, (1, c_in, h, w), &dev).unwrap();
            let wt = Tensor::from_vec(ws, (c_out, c_in, 3, 3), &dev).unwrap();
            let b = Tensor::from_vec(bs, c_out, &dev).unwrap();

            let got = conv3x3_direct(&x, &wt, Some(&b)).unwrap().expect("handled");
            let want = Conv2d::new(
                wt.clone(),
                Some(b.clone()),
                Conv2dConfig { padding: 1, stride: 1, ..Default::default() },
            )
            .forward(&x)
            .unwrap();

            let a = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let e = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let scale = e.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let worst =
                a.iter().zip(&e).map(|(x, y)| (x - y).abs() / scale).fold(0f32, f32::max);
            assert!(worst < 1e-6, "c_in={c_in} c_out={c_out} {h}x{w}: worst rel {worst:.3e}");
        }
    }
}
