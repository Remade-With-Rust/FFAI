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
use crate::par::prelude::*;

/// Output columns accumulated at once. 32 f32 = 4 AVX2 registers, which
/// leaves room for the broadcast weight and the loaded row.
// Read only by the OFF-by-default variant documented below.
#[allow(dead_code)]
const OX: usize = 32;

/// **OFF by default: 1.22x slower serial, ~1.10x slower at 24 threads.**
///
/// Two versions, and the difference between them is the whole lesson.
///
/// **v1 blocked over OUTPUT CHANNELS** — `chunks_mut(oh * ow)` handed each
/// task one channel, so every task walked the entire activation and the
/// activation was read `c_out` times: `c_out * c_in * 9` row reads against
/// im2col's `c_in * 9`. WORSE intensity than the thing it was written to
/// beat. Measured **1.73x slower**.
///
/// **v2 blocks over output ROWS** and keeps every output channel live at
/// once, so the activation row is read ONCE and consumed by all of them
/// while it sits in L1. Accumulators are sized to stay L2-resident
/// (`c_out * rows * ow * 4 <= 512 KB`).
///
/// | | serial CPU / image | 24-thread wall (24 imgs) |
/// |---|---:|---:|
/// | im2col + candle GEMM | **260 ms** | **3616 ms** |
/// | direct v1 (channel-blocked) | 448 ms | — |
/// | direct v2 (row-blocked) | 313 ms | 3971 ms |
///
/// **1.73x -> 1.22x. The re-blocking recovered exactly what the intensity
/// argument predicted, and it is still not enough.** What remains is not
/// structure but the microkernel: candle's GEMM does explicit register
/// blocking over several output rows and columns at once, while this leans
/// on LLVM auto-vectorising a single broadcast-AXPY. Closing that needs
/// `#[target_feature(enable = "avx2")]` intrinsics with an 8x8 register tile
/// and a scalar twin as oracle — `codec-vectorize-kernel`, a real brick.
///
/// Kept behind `FFAI_DIANA_DIRECT=1`. The correctness gate below passes
/// against candle's own `Conv2d` to <1e-6, so the next attempt inherits a
/// proven kernel and changes only the inner loop.
pub fn enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_DIRECT").is_ok_and(|v| v == "1");
            C.store(u8::from(on), Ordering::Relaxed);
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

    // Rows per task. One task owns `c_out * rows * ow` accumulators, which
    // must stay L2-resident: at the widest n-tier layer that is
    // 256 * 320 * 4 = 328 KB per row.
    let rows = (1..=8).rev().find(|r| c_out * r * ow * 4 <= (1 << 19)).unwrap_or(1);

    let out = crate::cpuop::SliceOp::new("ffai-direct3x3", move |xs, _| {
        let nblocks = oh.div_ceil(rows);
        let block = |bi: usize| -> Vec<f32> {
            let (r0, r1) = (bi * rows, ((bi + 1) * rows).min(oh));
            let nr = r1 - r0;
            // [c_out][nr][ow] for THIS block. Every output channel is live
            // at once, which is the whole point: the activation row below is
            // read once and consumed by all of them.
            let mut acc = vec![0f32; c_out * nr * ow];
            // REGISTER-TILED inner loop.
            //
            // v2 issued one AXPY per output channel over the full row, so
            // the accumulator it walked was `c_out * ow * 4` bytes — 328 KB
            // at the widest n-tier layer, which misses L1 on every channel.
            // A tile of OCB x OXB accumulators is 4 x 16 = 64 floats = 8 YMM
            // registers, so it lives in registers across the ENTIRE
            // (c_in, ky, kx) contraction and only touches memory twice: once
            // to initialise from the bias, once to store.
            //
            // Per (ic, ky, kx) step that is 2 loads (16 source floats) and 4
            // broadcasts feeding 8 FMAs — the ratio AVX2 wants, and the
            // reason this is register blocking rather than more threads.
            //
            // Interior blocks carry no bounds test: `sx = ox + kx - 1`, so a
            // block is interior when `ox0 >= 1` and `ox0 + OXB <= w - 1`.
            // Edge blocks fall to the scalar path, which is 2 of ~20 blocks
            // per row at the stem and does not deserve its own kernel.
            // AVX2 microkernel where the CPU has it, scalar register
            // tiling otherwise.
            //
            // The scalar form of this convolution lost 9/9 (z = +3.00) to
            // candle's im2col + GEMM, and the conclusion drawn was that a
            // hand-written convolution cannot beat a tuned GEMM. That
            // compared a SCALAR kernel against a VECTORISED one — explicit
            // AVX2 had been written for silu and never for the convolution,
            // which is the larger target. This is the missing arm of that
            // comparison.
            const OCB: usize = crate::direct3x3_avx2::OCB; // 8
            const OXB: usize = crate::direct3x3_avx2::OXB; // 8
            let simd = crate::direct3x3_avx2::available();

            // Bias first, so both paths accumulate into a primed buffer and
            // the microkernel does not need to know about bias at all.
            for oc in 0..c_out {
                for r in 0..nr {
                    acc[(oc * nr + r) * ow..(oc * nr + r + 1) * ow].fill(bs[oc]);
                }
            }

            for r in 0..nr {
                let oy = r0 + r;
                for oc0 in (0..c_out).step_by(OCB) {
                    let ocn = OCB.min(c_out - oc0);
                    let mut ox0 = 0usize;
                    while ox0 < ow {
                        let oxn = OXB.min(ow - ox0);
                        // `sx = ox + kx` for ox in [ox0, ox0+7] and kx in
                        // 0..2, and the load is `src[sx - 1]`, so the top
                        // index is `ox0 + OXB` and it must stay <= w - 1.
                        // Writing `<= w` reads one element past the row and
                        // silently pulls in the next row's data: 4.1e-1
                        // relative error against candle, which is what an
                        // off-by-one in a convolution looks like.
                        let interior = oxn == OXB && ox0 >= 1 && ox0 + OXB <= w - 1;
                        // The call site needs the SAME cfg as the kernel it
                        // calls. Without it this compiled everywhere the
                        // kernel existed and nowhere else — the one thing
                        // standing between this crate and a wasm32 build,
                        // found by compiling for wasm32 rather than by
                        // reading. `simd` is already false off x86_64, so
                        // this is a compile-time guard on dead code, not a
                        // second runtime check.
                        #[cfg(target_arch = "x86_64")]
                        if simd && interior && ocn == OCB {
                            // SAFETY: `available()` checked avx2+fma; the
                            // tile is interior and full by the guard above.
                            unsafe {
                                crate::direct3x3_avx2::tile(
                                    xs, &ws, &mut acc,
                                    (c_in, h, w), (nr, ow),
                                    (oc0, ox0, oy, r),
                                );
                            }
                            ox0 += OXB;
                            continue;
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        let _ = (simd, interior);
                        // Scalar fallback: edges, ragged channel groups, and
                        // any machine without AVX2.
                        for ic in 0..c_in {
                            for ky in 0..3usize {
                                let sy = oy + ky;
                                if sy == 0 || sy > h {
                                    continue;
                                }
                                let src = &xs[ic * h * w + (sy - 1) * w..ic * h * w + sy * w];
                                for kx in 0..3usize {
                                    for o in 0..ocn {
                                        let wv = ws[((oc0 + o) * c_in + ic) * 9 + ky * 3 + kx];
                                        let base = ((oc0 + o) * nr + r) * ow;
                                        for j in 0..oxn {
                                            let sx = ox0 + j + kx;
                                            if sx >= 1 && sx <= w {
                                                acc[base + ox0 + j] += wv * src[sx - 1];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        ox0 += OXB;
                    }
                }
            }
            acc
        };

        let blocks: Vec<Vec<f32>> = if crate::parallel::serial_kernels() {
            (0..nblocks).map(block).collect()
        } else {
            (0..nblocks).into_par_iter().map(block).collect()
        };

        // Assemble [c_out][oh][ow] from per-block [c_out][nr][ow]. One
        // output-sized copy, against the 9x operand it avoids building.
        let mut out = vec![0f32; c_out * oh * ow];
        for (bi, blk) in blocks.iter().enumerate() {
            let (r0, r1) = (bi * rows, ((bi + 1) * rows).min(oh));
            let nr = r1 - r0;
            for oc in 0..c_out {
                out[oc * oh * ow + r0 * ow..oc * oh * ow + r1 * ow]
                    .copy_from_slice(&blk[(oc * nr) * ow..(oc * nr + nr) * ow]);
            }
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
