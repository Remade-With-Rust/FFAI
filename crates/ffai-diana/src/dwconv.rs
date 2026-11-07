//! A depthwise 3x3 convolution kernel, because candle has none.
//!
//! # Why this exists (measured, not assumed)
//!
//! candle's CPU backend has no grouped-convolution kernel. `Tensor::conv2d`
//! with `groups > 1` does `chunk(groups, 1)` -> one `conv2d_single_group`
//! per group -> `cat` (candle-core/src/conv.rs). A depthwise convolution has
//! `groups == channels`, so a 256-channel depthwise runs **256 separate
//! single-channel convolutions**, each through the full im2col machinery,
//! plus 256 allocations and a 256-way concatenation.
//!
//! Measured on `YOLO26n`'s head (`examples/dwconv_prize.rs`):
//!
//! | depthwise | ms | GFLOP/s |
//! |---|---:|---:|
//! | c=64 @ 80x80 | 7.95 | 0.93 |
//! | c=80 @ 80x80 | 8.88 | 1.04 |
//! | c=128 @ 40x40 | 8.31 | 0.44 |
//! | c=256 @ 20x20 | 3.10 | 0.59 |
//!
//! The six in the head total **34.2 ms/image, 12.9% of detect** — while a
//! *dense* 64->64 3x3 at the same resolution, doing **64x more arithmetic**,
//! takes 5.0 ms. Depthwise was **72.9x slower per FLOP than the dense
//! convolution beside it**. That is the group-splitting overhead, not the
//! arithmetic.
//!
//! # What this does instead
//!
//! One pass, parallel over channels, with the 9 taps held in registers and a
//! branch-free interior loop. There is no im2col: a 3x3 depthwise is a
//! 9-tap stencil, and materializing a 9x-larger matrix to feed a GEMM would
//! cost more traffic than the arithmetic saves.
//!
//! Float, so the gate is a tolerance against candle's own grouped path
//! (which stays the oracle), not byte-identity — the accumulation order
//! differs.

use candle_core::{Result, Tensor};
use crate::par::prelude::*;

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

/// The plane kernel compiled with AVX2 enabled: eight lanes per vector op.
///
/// # Safety
///
/// The host must support AVX2 (`avx2_available()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn plane_avx2<const ADD: bool>(plane: &[f32], k: &[f32; 9], bias: f32, h: usize, w: usize, zero_row: &[f32], o: &mut [f32]) {
    plane_body::<ADD>(plane, k, bias, h, w, zero_row, o);
}

/// Eight outputs of the stencil, fixed width so the loop is one vector op
/// per tap with no remainder.
#[inline(always)]
fn stencil8(u: [&[f32]; 3], m: [&[f32]; 3], d: [&[f32]; 3], k: &[f32; 9], bias: Option<f32>, o: &mut [f32]) {
    let [k0, k1, k2, k3, k4, k5, k6, k7, k8] = *k;
    let (ul, uc, ur): (&[f32; 8], &[f32; 8], &[f32; 8]) =
        (u[0].try_into().unwrap(), u[1].try_into().unwrap(), u[2].try_into().unwrap());
    let (ml, mc, mr): (&[f32; 8], &[f32; 8], &[f32; 8]) =
        (m[0].try_into().unwrap(), m[1].try_into().unwrap(), m[2].try_into().unwrap());
    let (dl, dc, dr): (&[f32; 8], &[f32; 8], &[f32; 8]) =
        (d[0].try_into().unwrap(), d[1].try_into().unwrap(), d[2].try_into().unwrap());
    let o: &mut [f32; 8] = o.try_into().unwrap();
    for x in 0..8 {
        let s = ul[x] * k0 + uc[x] * k1 + ur[x] * k2
            + ml[x] * k3 + mc[x] * k4 + mr[x] * k5
            + dl[x] * k6 + dc[x] * k7 + dr[x] * k8;
        o[x] = match bias {
            Some(b) => s + b,
            None => s,
        };
    }
}

/// One channel of the depthwise 3x3: `o = stencil(plane) + bias`.
///
/// `inline(always)` so each `target_feature` wrapper gets its own copy of
/// the loops, compiled for its own instruction set.
///
/// `ADD`: add the bias in the stencil. When `SiLU` follows, the bias goes
/// into the activation's own `x + bias` (`silu_bias_in_place`) instead —
/// `silu(s + bias)` either way, same expression, one vector add fewer here.
#[inline(always)]
fn plane_body<const ADD: bool>(plane: &[f32], k: &[f32; 9], bias: f32, h: usize, w: usize, zero_row: &[f32], o: &mut [f32]) {
    let [k0, k1, k2, k3, k4, k5, k6, k7, k8] = *k;
    if w == 0 {
        return;
    }
    for (y, o) in o.chunks_exact_mut(w).enumerate() {
        let up: &[f32] = if y > 0 { &plane[(y - 1) * w..y * w] } else { zero_row };
        let mid = &plane[y * w..(y + 1) * w];
        let dn: &[f32] = if y + 1 < h { &plane[(y + 1) * w..(y + 2) * w] } else { zero_row };

        // Interior columns. Every operand is re-sliced to the SAME length
        // `n`, which is the bound relation LLVM needs to drop all ten bounds
        // checks and vectorize over x (`rusty-compiler-leverage`:
        // information, not cleverness).
        if w >= 2 {
            let n = w - 2;
            let (ul, uc, ur) = (&up[..n], &up[1..=n], &up[2..n + 2]);
            let (ml, mc, mr) = (&mid[..n], &mid[1..=n], &mid[2..n + 2]);
            let (dl, dc, dr) = (&dn[..n], &dn[1..=n], &dn[2..n + 2]);
            let oi = &mut o[1..=n];
            if n >= 8 {
                // Whole eight-wide blocks only, the LAST one overlapping the
                // one before it. Rows here are 20 to 112 wide, so the plain
                // `for x in 0..n` left a scalar remainder of up to seven
                // elements on EVERY row — 85 000 rows per image, and the
                // line census put more instructions in that remainder than
                // in the vector body. Recomputing up to seven outputs is the
                // same expression on the same inputs: identical bytes.
                let mut b = 0usize;
                loop {
                    let x0 = if b + 8 <= n { b } else { n - 8 };
                    stencil8(
                        [&ul[x0..x0 + 8], &uc[x0..x0 + 8], &ur[x0..x0 + 8]],
                        [&ml[x0..x0 + 8], &mc[x0..x0 + 8], &mr[x0..x0 + 8]],
                        [&dl[x0..x0 + 8], &dc[x0..x0 + 8], &dr[x0..x0 + 8]],
                        k,
                        if ADD { Some(bias) } else { None },
                        &mut oi[x0..x0 + 8],
                    );
                    if b + 8 >= n {
                        break;
                    }
                    b += 8;
                }
            } else {
                for x in 0..n {
                    let s = ul[x] * k0 + uc[x] * k1 + ur[x] * k2
                        + ml[x] * k3 + mc[x] * k4 + mr[x] * k5
                        + dl[x] * k6 + dc[x] * k7 + dr[x] * k8;
                    oi[x] = if ADD { s + bias } else { s };
                }
            }
        }
        // The two edge columns, where the horizontal taps clamp. Same
        // summation order as before: centre column, then left, then right.
        for &x in &[0usize, w - 1] {
            let mut s = up[x] * k1 + mid[x] * k4 + dn[x] * k7;
            if x > 0 {
                s += up[x - 1] * k0 + mid[x - 1] * k3 + dn[x - 1] * k6;
            }
            if x + 1 < w {
                s += up[x + 1] * k2 + mid[x + 1] * k5 + dn[x + 1] * k8;
            }
            o[x] = if ADD { s + bias } else { s };
        }
    }
}

/// Depthwise 3x3, stride 1, padding 1 — the only shape YOLO26 uses.
///
/// `x` is `(1, c, h, w)`, `weight` is `(c, 1, 3, 3)`, `bias` is `(c,)`.
/// Returns `(1, c, h, w)`.
///
/// `act` applies `SiLU` to each plane as it is finished, inside the kernel,
/// with the same AVX2 kernel the epilogue uses (bit-identical to the scalar
/// `silu`, by `silu_avx2::tests::is_bit_identical_to_the_scalar_kernel_over_a_dense_sweep`).
/// Unfused, the activation was a second custom op: a second allocation, a
/// second traversal, and — because the serial `silu` path is scalar — 15.7
/// instructions per element where the vector kernel spends 3.75.
pub fn depthwise3x3(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, act: bool) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    debug_assert_eq!(n, 1, "batch 1 is the only shape this engine runs");
    // The weights are tiny (c*9) and constant; copying them is noise. The
    // ACTIVATION is what must not be copied, and it is borrowed in place by
    // the SliceOp below.
    let ws = crate::cpuop::bias_vec(weight)?;
    let bs = match bias {
        Some(b) => crate::cpuop::bias_vec(b)?,
        None => vec![0.0; c],
    };

    crate::cpuop::SliceOp::new("ffai-dwconv3x3", move |xs, _| {
    // Uninitialised output: every element is written exactly once below
    // (each row's interior, then its two edge columns — or its single column
    // when `w == 1`), so the zero fill `vec![0.0; n]` paid for was a second
    // write of the whole buffer (`calloc` -> `memset`, 1.7 M Ir per image).
    let n_out = c * h * w;
    let mut out: Vec<f32> = Vec::with_capacity(n_out);
    let spare = &mut out.spare_capacity_mut()[..n_out];
    // SAFETY: `f32` is plain data; the kernel writes all `n_out` elements and
    // reads none of them before writing; `set_len` follows the writes.
    #[allow(unsafe_code)]
    let dst: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n_out) };
    // The zero pad as a ROW, not a branch. `at(row, i)` tested `is_empty()`
    // on six of the nine taps for every output element, and the line census
    // put the kernel at 58 Ir per element for 9 multiplies. A shared row of
    // zeros makes every tap an unconditional load; `0.0 * k` is the same
    // signed zero the branch produced, so the sums are bit-identical.
    let zero_row = vec![0f32; w];
    // One closure, two schedulers. `crate::parallel` explains why the
    // choice is per-call: the fan-out wins for latency and loses for
    // throughput, and the caller is the only one who knows which it wants.
    let plane_kernel = |(ch, o): (usize, &mut [f32])| {
        let k: &[f32; 9] = ws[ch * 9..ch * 9 + 9].try_into().expect("nine taps");
        let plane = &xs[ch * h * w..(ch + 1) * h * w];
        // Same body, two compilations. The crate targets the x86-64 baseline,
        // so the stencil vectorized FOUR lanes wide (9.5 Ir per element on the
        // line census). Under `avx2` it is eight. No `fma` is enabled, and
        // Rust never contracts `a * b + c` on its own, so the arithmetic is
        // the baseline's to the bit; only the lane count changes.
        #[cfg(target_arch = "x86_64")]
        let wide = avx2_available();
        #[cfg(not(target_arch = "x86_64"))]
        let wide = false;
        if wide {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: avx2 verified at runtime, just above.
            #[allow(unsafe_code)]
            unsafe {
                if act {
                    plane_avx2::<false>(plane, k, bs[ch], h, w, &zero_row, o);
                } else {
                    plane_avx2::<true>(plane, k, bs[ch], h, w, &zero_row, o);
                }
            }
        } else if act {
            plane_body::<false>(plane, k, bs[ch], h, w, &zero_row, o);
        } else {
            plane_body::<true>(plane, k, bs[ch], h, w, &zero_row, o);
        }
        if act {
            crate::silu::silu_bias_slice_in_place(o, bs[ch]);
        }
    };
    if crate::parallel::serial_kernels() {
        dst.chunks_mut(h * w).enumerate().for_each(plane_kernel);
    } else {
        dst.par_chunks_mut(h * w).enumerate().for_each(plane_kernel);
    }
    // SAFETY: all `n_out` elements written by the plane kernels above.
    #[allow(unsafe_code)]
    unsafe {
        out.set_len(n_out);
    }
        Ok((out, (1, c, h, w).into()))
    })
    .run(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{Conv2d, Conv2dConfig, Module};

    /// The oracle: candle's own grouped path. It stays in the tree forever —
    /// this kernel is only allowed to be faster, never different.
    fn oracle(x: &Tensor, w: &Tensor, b: &Tensor, c: usize) -> Tensor {
        let cfg = Conv2dConfig { padding: 1, stride: 1, groups: c, ..Default::default() };
        Conv2d::new(w.clone(), Some(b.clone()), cfg).forward(x).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn matches_candles_grouped_conv() {
        let dev = Device::Cpu;
        // Shapes spanning the head's real range, plus awkward small ones
        // where the edge handling is most of the work.
        for &(c, h, w) in &[(64, 80, 80), (80, 40, 40), (256, 20, 20), (3, 5, 7), (1, 1, 1), (2, 1, 4)] {
            let x = Tensor::randn(0f32, 1.0, (1, c, h, w), &dev).unwrap();
            let wt = Tensor::randn(0f32, 1.0, (c, 1, 3, 3), &dev).unwrap();
            let b = Tensor::randn(0f32, 1.0, c, &dev).unwrap();
            let got = depthwise3x3(&x, &wt, Some(&b), false).unwrap();
            let want = oracle(&x, &wt, &b, c);
            assert_eq!(got.dims(), want.dims(), "shape for c={c} {h}x{w}");
            let d = max_abs_diff(&got, &want);
            assert!(d < 1e-4, "c={c} {h}x{w}: max abs diff {d:.3e}");
        }
    }

    /// Fused activation == unfused activation, bit for bit.
    #[test]
    fn fused_silu_matches_the_separate_pass() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 6, 9, 13), &dev).unwrap();
        let wt = Tensor::randn(0f32, 1.0, (6, 1, 3, 3), &dev).unwrap();
        let b = Tensor::randn(0f32, 1.0, 6, &dev).unwrap();
        let fused = depthwise3x3(&x, &wt, Some(&b), true).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let plain = depthwise3x3(&x, &wt, Some(&b), false).unwrap();
        let two = crate::silu::silu(&plain).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (f, t)) in fused.iter().zip(&two).enumerate() {
            assert_eq!(f.to_bits(), t.to_bits(), "element {i}: {f} vs {t}");
        }
    }

    #[test]
    fn handles_a_missing_bias() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 9, 9), &dev).unwrap();
        let wt = Tensor::randn(0f32, 1.0, (4, 1, 3, 3), &dev).unwrap();
        let zero = Tensor::zeros(4, candle_core::DType::F32, &dev).unwrap();
        let got = depthwise3x3(&x, &wt, None, false).unwrap();
        let want = oracle(&x, &wt, &zero, 4);
        assert!(max_abs_diff(&got, &want) < 1e-4);
    }
}
