//! Softmax over the last axis as ONE pass per row, for the attention scores.
//!
//! # Why (measured)
//!
//! `candle_nn::ops::softmax` is five tensor ops — `max_keepdim`,
//! `broadcast_sub`, `exp`, `sum_keepdim`, `broadcast_div` — each a fresh
//! allocation and a full traversal of the `(heads, n, n)` score matrix, with
//! candle's generic strided index arithmetic on the broadcasts. The
//! exponentials themselves are the irreducible part (`expf`, 15 instructions
//! each); the four extra traversals and four intermediates are not.
//!
//! # Exactly candle's answer, bit for bit
//!
//! Per row: `m = max`, `e_i = exp(x_i - m)`, `s = Σ e_i` in index order,
//! `y_i = e_i / s`. That IS candle's composition — the same `f32::exp`
//! (libm), the same subtraction, the same sequential f32 sum (this crate is
//! built for the x86-64 baseline, where candle's `vec_sum` is the plain
//! in-order loop), the same division — so every output bit matches, and the
//! test says so against candle itself.

use candle_core::{Result, Tensor};

/// `FFAI_DIANA_SOFTMAX_LIBM=1` keeps libm's `expf` in the softmax — the
/// bit-exact-to-candle arm, for A/B against the polynomial.
fn libm_exp() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FFAI_DIANA_SOFTMAX_LIBM").is_ok_and(|v| v == "1"))
}

/// `softmax(x, D::Minus1)` for a contiguous f32 tensor of rank >= 1.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let n = *dims.last().unwrap_or(&0);
    if n == 0 {
        return candle_nn::ops::softmax(x, candle_core::D::Minus1);
    }
    let libm = libm_exp();
    crate::cpuop::SliceOp::new("ffai-softmax-last", move |xs, l| {
        // Four passes per row, like candle's four elementwise ops — because
        // the single-pass `push` form MEASURED SLOWER (+3.8 M Ir): the
        // subtract and the divide vectorize when they are their own loops
        // over a sized buffer, and `exp` is a libm call either way. What this
        // removes is the four intermediates and candle's strided-broadcast
        // index arithmetic, not the arithmetic.
        let mut out = vec![0f32; xs.len()];
        for (row, o) in xs.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            let mut m = row[0];
            for &v in &row[1..] {
                m = if v > m { v } else { m };
            }
            for (d, &v) in o.iter_mut().zip(row) {
                *d = v - m;
            }
            if libm {
                for d in o.iter_mut() {
                    *d = d.exp();
                }
            } else {
                // The same range-reduced polynomial `SiLU` already runs on
                // every activation in this graph (`ffai_core::fastmath::exp`,
                // ~1e-7 relative), on the attention scores. This is the one
                // step in this module that CHANGES bits — priced against the
                // full-graph oracle (`tests/oracles.rs`), not assumed.
                for d in o.iter_mut() {
                    *d = ffai_core::fastmath::exp(*d);
                }
            }
            let mut s = 0f32;
            for &d in o.iter() {
                s += d;
            }
            for d in o.iter_mut() {
                *d /= s;
            }
        }
        Ok((out, l.shape().clone()))
    })
    .run(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    #[test]
    fn matches_candle_bit_for_bit() {
        let dev = Device::Cpu;
        for &(h, r, c) in &[(2usize, 280usize, 280usize), (1, 3, 5), (3, 1, 1), (1, 7, 33)] {
            let x = (Tensor::randn(0f32, 1.0, (1, h, r, c), &dev).unwrap() * 3.0).unwrap();
            let want = candle_nn::ops::softmax(&x, D::Minus1).unwrap();
            let got = softmax_last(&x).unwrap();
            if !libm_exp() {
                // Polynomial arm: tolerance, not bits — the oracle prices it.
                let (g, w) = (
                    got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                );
                let worst = g.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(worst < 2e-6, "h={h} {r}x{c}: worst abs diff {worst:e}");
                continue;
            }
            assert_eq!(got.dims(), want.dims());
            let (g, w) = (
                got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            );
            for (i, (a, b)) in g.iter().zip(&w).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "h={h} {r}x{c} element {i}: {a} vs {b}");
            }
        }
    }
}
