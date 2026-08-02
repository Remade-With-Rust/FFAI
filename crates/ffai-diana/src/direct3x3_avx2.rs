//! Explicit AVX2 microkernel for the direct 3x3 convolution.
//!
//! # The gap this exists to test
//!
//! The direct convolution — which never materialises im2col's 9x-expanded
//! buffer, and so never pays the 216.9 MiB/image of traffic that buffer
//! costs — lost **9/9 rounds, z = +3.00** to candle's im2col + GEMM. The
//! conclusion drawn was "candle's GEMM is at 69 % of peak and a hand-written
//! one is not, so this route is closed."
//!
//! That conclusion was drawn from a **scalar** direct convolution. Three
//! variants of it, all written in safe Rust with register blocking, hoping
//! LLVM would vectorise. An explicit AVX2 microkernel was written for
//! `silu` and never for the convolution, which is the larger target — so the
//! comparison was a scalar kernel against a vectorised GEMM, and it is not
//! surprising which won.
//!
//! # The shape
//!
//! Eight output channels by eight output columns, held in **eight YMM
//! accumulators** across the entire `(c_in, ky, kx)` contraction. Each step
//! is one source load feeding eight broadcast+FMA pairs — an 8:1 arithmetic
//! ratio, which is what the register file is for.
//!
//! Interior columns only: `sx = ox + kx - 1`, so a block is interior when
//! `ox0 >= 1` and `ox0 + 8 <= w - 1`. Edges fall back to the caller's scalar
//! path, which is two blocks per row and does not deserve its own kernel.
//!
//! # The inversion that looked obvious and measured worse
//!
//! Consecutive output channels are `c_in * 9` floats apart, so at c_in = 256
//! this tile's eight weight broadcasts sit **9 KB apart — every one a
//! different cache line**. "One load feeding eight FMAs is worthless if the
//! eight scalars feeding them each cost a miss" is a good argument, and the
//! obvious fix is to invert: hold ONE output channel across eight
//! accumulators (64 columns), so each `(ic, ky, kx)` step needs a single
//! broadcast reused across every column.
//!
//! Built and measured: **1.52x behind im2col+GEMM, against this shape's
//! 1.11x.** Worse, and clearly so (15/15, z = +3.87).
//!
//! The inversion trades one load:eight-FMAs for eight loads:eight-FMAs — a
//! 1:1 ratio, which is memory-bound no matter how well the weights cache.
//! Source reuse is the scarcer resource here and the strided broadcasts,
//! though real, are the cheaper problem: they hit L2 rather than DRAM and
//! the hardware prefetcher sees a constant stride.
//!
//! Recorded because the reasoning was sound and the conclusion was wrong —
//! which is the only kind of prune worth writing down.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Output channels per register tile.
pub const OCB: usize = 8;
/// Output columns per register tile — one YMM of f32.
pub const OXB: usize = 8;

/// True when this CPU has what the kernel needs.
#[inline]
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Accumulate one `OCB x OXB` output tile for output row `oy`.
///
/// `acc` is the block's `[c_out][nr][ow]` accumulator; `oc0`/`ox0` locate the
/// tile; `r` is the row within the block. Bias must already be in `acc`.
///
/// # Safety
///
/// Caller must have checked [`available`], must guarantee the tile is
/// INTERIOR (`ox0 >= 1 && ox0 + OXB <= w`), and must pass indices in range.
/// All of that is asserted in debug.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tile(
    xs: &[f32],
    ws: &[f32],
    acc: &mut [f32],
    (c_in, h, w): (usize, usize, usize),
    (nr, ow): (usize, usize),
    (oc0, ox0, oy, r): (usize, usize, usize, usize),
) {
    debug_assert!(ox0 >= 1 && ox0 + OXB <= w, "tile must be interior");
    let hw = h * w;

    // Load the eight accumulators for this tile.
    let mut a = [_mm256_setzero_ps(); OCB];
    for (o, av) in a.iter_mut().enumerate() {
        *av = _mm256_loadu_ps(acc.as_ptr().add(((oc0 + o) * nr + r) * ow + ox0));
    }

    for ic in 0..c_in {
        for ky in 0..3usize {
            let sy = oy + ky;
            if sy == 0 || sy > h {
                continue;
            }
            let row = ic * hw + (sy - 1) * w;
            for kx in 0..3usize {
                // One load feeds all eight channels: the whole point.
                let s = _mm256_loadu_ps(xs.as_ptr().add(row + ox0 + kx - 1));
                let wbase = (ic * 9 + ky * 3 + kx) as isize;
                for (o, av) in a.iter_mut().enumerate() {
                    let wv = _mm256_broadcast_ss(
                        &*ws.as_ptr().offset((oc0 + o) as isize * (c_in * 9) as isize + wbase),
                    );
                    *av = _mm256_fmadd_ps(wv, s, *av);
                }
            }
        }
    }

    for (o, av) in a.iter().enumerate() {
        _mm256_storeu_ps(acc.as_mut_ptr().add(((oc0 + o) * nr + r) * ow + ox0), *av);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The microkernel must reproduce a plain scalar accumulation of the same
    /// tile. FMA rounds once where the scalar rounds twice, so this is a
    /// TOLERANCE and not equality — the silu kernel breached a full-graph
    /// oracle by exactly that difference, which is why the bound is stated
    /// here and re-checked downstream.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(unsafe_code)]
    fn tile_matches_scalar_accumulation() {
        if !available() {
            eprintln!("SKIP: no avx2+fma");
            return;
        }
        let (c_in, h, w) = (5usize, 6usize, 24usize);
        let (c_out, nr, ow) = (OCB, 1usize, w);
        let xs: Vec<f32> = (0..c_in * h * w).map(|i| (i % 17) as f32 * 0.37 - 3.0).collect();
        let ws: Vec<f32> = (0..c_out * c_in * 9).map(|i| (i % 11) as f32 * 0.13 - 0.7).collect();
        let (oy, r, oc0, ox0) = (2usize, 0usize, 0usize, 8usize);

        let mut got = vec![0f32; c_out * nr * ow];
        // SAFETY: availability checked; the tile is interior by construction.
        unsafe { tile(&xs, &ws, &mut got, (c_in, h, w), (nr, ow), (oc0, ox0, oy, r)) };

        let mut want = vec![0f32; c_out * nr * ow];
        for o in 0..OCB {
            for j in 0..OXB {
                let mut s = 0f32;
                for ic in 0..c_in {
                    for ky in 0..3 {
                        let sy = oy + ky;
                        if sy == 0 || sy > h {
                            continue;
                        }
                        for kx in 0..3 {
                            let sx = ox0 + j + kx;
                            if sx == 0 || sx > w {
                                continue;
                            }
                            s += ws[o * c_in * 9 + ic * 9 + ky * 3 + kx]
                                * xs[ic * h * w + (sy - 1) * w + sx - 1];
                        }
                    }
                }
                want[(o * nr + r) * ow + ox0 + j] = s;
            }
        }
        let mut worst = 0f32;
        for i in 0..got.len() {
            let scale = want[i].abs().max(1e-3);
            worst = worst.max((got[i] - want[i]).abs() / scale);
        }
        assert!(worst < 1e-5, "avx2 tile diverges from scalar by {worst:.3e}");
    }
}
