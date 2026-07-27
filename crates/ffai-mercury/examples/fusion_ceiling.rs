//! Ceiling probe for a fused attention kernel — before building one.
//!
//! The attention trio (scores q@kᵀ → softmax → attn@v) is 74.6 ms of a 232 ms
//! encoder pass and is memory-bound: the 54 MB score matrix is written, read,
//! written and read again. The textbook answer is a fused kernel that keeps
//! score blocks in registers/cache and never materializes them.
//!
//! `codec-analyzer` rule: probe the ceiling to decide go/no-go, because a
//! high ceiling is necessary but not sufficient — the *tax* only shows up on
//! the bench. Query-tiling (§6.8) already measured that tax: splitting the
//! GEMM cost +46 ms even at the best block size. So this asks two questions:
//!
//! 1. **The prize** — how much is the score matrix's memory traffic actually
//!    worth? Measured as the trio total minus the two matmuls alone.
//! 2. **The tax** — what GEMM throughput does a hand-written blocked kernel
//!    actually reach, versus the tuned `gemm` candle calls into?
//!
//! If the tax exceeds the prize, the kernel cannot win and should not be
//! built.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example fusion_ceiling
//! ```

use std::time::Instant;

use ffai_core::candle::{Device, Tensor};
use rayon::prelude::*;

fn best_of<T>(n: usize, mut f: impl FnMut() -> T) -> f64 {
    std::hint::black_box(f());
    let mut best = f64::MAX;
    for _ in 0..n {
        let t0 = Instant::now();
        let out = f();
        std::hint::black_box(&out);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

/// One head of fused attention over raw slices: scores, softmax and the value
/// product in a single pass over query blocks, never materializing the full
/// score matrix.
fn fused_head(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], seq: usize, dim: usize) {
    const BLOCK: usize = 64;
    out.par_chunks_mut(BLOCK * dim)
        .enumerate()
        .for_each(|(bi, out_block)| {
            let q0 = bi * BLOCK;
            let rows = (seq - q0).min(BLOCK);
            let mut scores = vec![0f32; rows * seq];
            // scores = q_block @ kᵀ
            for r in 0..rows {
                let qr = &q[(q0 + r) * dim..(q0 + r + 1) * dim];
                for c in 0..seq {
                    let kc = &k[c * dim..(c + 1) * dim];
                    let mut acc = 0f32;
                    for d in 0..dim {
                        acc += qr[d] * kc[d];
                    }
                    scores[r * seq + c] = acc;
                }
            }
            // softmax over each row, in place while it is hot
            for r in 0..rows {
                let row = &mut scores[r * seq..(r + 1) * seq];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for x in row.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                let inv = 1.0 / sum;
                for x in row.iter_mut() {
                    *x *= inv;
                }
            }
            // out_block = weights @ v
            for r in 0..rows {
                let w = &scores[r * seq..(r + 1) * seq];
                let o = &mut out_block[r * dim..(r + 1) * dim];
                o.fill(0.0);
                for c in 0..seq {
                    let wc = w[c];
                    if wc == 0.0 {
                        continue;
                    }
                    let vc = &v[c * dim..(c + 1) * dim];
                    for d in 0..dim {
                        o[d] += wc * vc[d];
                    }
                }
            }
        });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (seq, heads, dim) = (1500usize, 6usize, 64usize);

    let q = Tensor::randn(0f32, 1., (1, heads, seq, dim), &dev)?.contiguous()?;
    let k_t = Tensor::randn(0f32, 1., (1, heads, dim, seq), &dev)?.contiguous()?;
    let v = Tensor::randn(0f32, 1., (1, heads, seq, dim), &dev)?.contiguous()?;
    let scores = Tensor::randn(0f32, 1., (1, heads, seq, seq), &dev)?.contiguous()?;

    let t_qk = best_of(3, || q.matmul(&k_t).expect("qk"));
    let t_sm = best_of(3, || candle_nn::ops::softmax_last_dim(&scores).expect("sm"));
    let t_wv = best_of(3, || scores.matmul(&v).expect("wv"));
    let trio = t_qk + t_sm + t_wv;
    let matmuls = t_qk + t_wv;

    // Both matmuls together: 2 * 2 * heads * seq * seq * dim FLOPs.
    let flops = 2.0 * 2.0 * (heads * seq * seq * dim) as f64;

    println!("attention trio, per layer (tiny.en, seq=1500, 6 heads, dim=64)");
    println!("  scores q@kT   {:7.2} ms", t_qk * 1e3);
    println!("  softmax       {:7.2} ms", t_sm * 1e3);
    println!("  attn@v        {:7.2} ms", t_wv * 1e3);
    println!("  TRIO          {:7.2} ms   ({:.1} GFLOP/s on the two matmuls)", trio * 1e3, flops / matmuls / 1e9);
    println!("\nTHE PRIZE — perfect fusion removes at most the separate softmax pass:");
    println!("  ceiling saving {:7.2} ms/layer  = {:.1}% of the trio", t_sm * 1e3, t_sm / trio * 100.0);

    // The tax: what a hand-written fused kernel actually achieves.
    let qv: Vec<f32> = q.flatten_all()?.to_vec1()?;
    let kv: Vec<f32> = Tensor::randn(0f32, 1., (1, heads, seq, dim), &dev)?
        .contiguous()?
        .flatten_all()?
        .to_vec1()?;
    let vv: Vec<f32> = v.flatten_all()?.to_vec1()?;
    let mut out = vec![0f32; heads * seq * dim];
    let t_fused = best_of(2, || {
        for h in 0..heads {
            let off = h * seq * dim;
            let (qh, kh, vh) = (
                &qv[off..off + seq * dim],
                &kv[off..off + seq * dim],
                &vv[off..off + seq * dim],
            );
            let oh = &mut out[off..off + seq * dim];
            fused_head(qh, kh, vh, oh, seq, dim);
        }
    });

    println!("\nTHE TAX — hand-written fused kernel, same work:");
    println!("  fused kernel  {:7.2} ms   ({:.1} GFLOP/s)", t_fused * 1e3, flops / t_fused / 1e9);
    println!(
        "\nVERDICT: fused {:.2} ms vs trio {:.2} ms  ->  {}",
        t_fused * 1e3,
        trio * 1e3,
        if t_fused < trio { "WIN, build it" } else { "LOSS, do not build" }
    );
    Ok(())
}
