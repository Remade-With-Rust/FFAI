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
//! Measured on YOLO26n's head (`examples/dwconv_prize.rs`):
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
use rayon::prelude::*;

/// Depthwise 3x3, stride 1, padding 1 — the only shape YOLO26 uses.
///
/// `x` is `(1, c, h, w)`, `weight` is `(c, 1, 3, 3)`, `bias` is `(c,)`.
/// Returns `(1, c, h, w)`.
pub fn depthwise3x3(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    debug_assert_eq!(n, 1, "batch 1 is the only shape this engine runs");
    // The weights are tiny (c*9) and constant; copying them is noise. The
    // ACTIVATION is what must not be copied, and it is borrowed in place by
    // the SliceOp below.
    let ws = weight.flatten_all()?.to_vec1::<f32>()?;
    let bs = match bias {
        Some(b) => b.flatten_all()?.to_vec1::<f32>()?,
        None => vec![0.0; c],
    };

    crate::cpuop::SliceOp::new("ffai-dwconv3x3", move |xs, _| {
    let mut out = vec![0f32; c * h * w];
    out.par_chunks_mut(h * w).enumerate().for_each(|(ch, o)| {
        let k = &ws[ch * 9..ch * 9 + 9];
        let (k0, k1, k2) = (k[0], k[1], k[2]);
        let (k3, k4, k5) = (k[3], k[4], k[5]);
        let (k6, k7, k8) = (k[6], k[7], k[8]);
        let bias = bs[ch];
        let plane = &xs[ch * h * w..(ch + 1) * h * w];

        for y in 0..h {
            let row_o = y * w;
            // Rows above/below clamp to the zero pad, handled by selecting
            // an all-zero source rather than branching inside the x loop.
            let has_up = y > 0;
            let has_dn = y + 1 < h;
            let up = if has_up { &plane[(y - 1) * w..y * w] } else { &[][..] };
            let mid = &plane[row_o..row_o + w];
            let dn = if has_dn { &plane[(y + 1) * w..(y + 2) * w] } else { &[][..] };

            // Interior columns: every tap in bounds, no branches, so the
            // loop auto-vectorizes over x.
            let at = |r: &[f32], i: usize| -> f32 { if r.is_empty() { 0.0 } else { r[i] } };
            if w >= 2 {
                for x in 1..w - 1 {
                    let s = at(up, x - 1) * k0 + at(up, x) * k1 + at(up, x + 1) * k2
                        + mid[x - 1] * k3 + mid[x] * k4 + mid[x + 1] * k5
                        + at(dn, x - 1) * k6 + at(dn, x) * k7 + at(dn, x + 1) * k8;
                    o[row_o + x] = s + bias;
                }
            }
            // The two edge columns, where the horizontal taps clamp.
            for &x in [0usize, w.saturating_sub(1)].iter() {
                if w == 0 {
                    continue;
                }
                let l = x > 0;
                let r = x + 1 < w;
                let mut s = at(up, x) * k1 + mid[x] * k4 + at(dn, x) * k7;
                if l {
                    s += at(up, x - 1) * k0 + mid[x - 1] * k3 + at(dn, x - 1) * k6;
                }
                if r {
                    s += at(up, x + 1) * k2 + mid[x + 1] * k5 + at(dn, x + 1) * k8;
                }
                o[row_o + x] = s + bias;
            }
        }
    });
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
            let got = depthwise3x3(&x, &wt, Some(&b)).unwrap();
            let want = oracle(&x, &wt, &b, c);
            assert_eq!(got.dims(), want.dims(), "shape for c={c} {h}x{w}");
            let d = max_abs_diff(&got, &want);
            assert!(d < 1e-4, "c={c} {h}x{w}: max abs diff {d:.3e}");
        }
    }

    #[test]
    fn handles_a_missing_bias() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 9, 9), &dev).unwrap();
        let wt = Tensor::randn(0f32, 1.0, (4, 1, 3, 3), &dev).unwrap();
        let zero = Tensor::zeros(4, candle_core::DType::F32, &dev).unwrap();
        let got = depthwise3x3(&x, &wt, None).unwrap();
        let want = oracle(&x, &wt, &zero, 4);
        assert!(max_abs_diff(&got, &want) < 1e-4);
    }
}
