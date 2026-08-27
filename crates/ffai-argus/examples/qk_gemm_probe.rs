//! Is candle's gemm leaving `q.k^T` on the table, and can plain Rust take it?
//!
//! # The gap that motivates this
//!
//! In situ, the four projections run at **500-563 GF/s** while `q.k^T` runs at
//! **160 GF/s** — and 160 is what it measures ISOLATED too, so it is the shape,
//! not our usage. Same library, same box, 3.5x apart. The projections have
//! `K = 768`; `q.k^T` has `K = 64`, and a reduction that short leaves a general
//! gemm's packing and blocking machinery with very little to amortise over.
//!
//! `q.k^T` is 18.3 % of a tile, so closing even half that gap is worth more
//! than everything else still on the table.
//!
//! # Why a SCALAR kernel first
//!
//! The house rule before reaching for intrinsics is to check whether the
//! compiler already vectorises a well-shaped loop — the reformulation usually
//! beats the intrinsics, and costs a fraction as much. So this writes the
//! outer-product form in plain Rust with register blocking and lets LLVM do it:
//!
//! ```text
//! C[i][j] = SUM_k A[i][k] * Bt[k][j]
//! ```
//!
//! With `Bt` laid out `(head_dim, seq)`, the inner loop walks `j` contiguously
//! and every FMA is a broadcast-times-vector — no horizontal sums, which is
//! what makes the naive `dot(A[i], B[j])` form slow. Crucially **we already own
//! that layout**: `siglip::PackedQkvOp` writes the k buffer itself, so storing
//! k pre-transposed is free rather than an extra copy.
//!
//! If this loses, the answer is "candle's gemm is fine, stop" and no unsafe
//! code was written to find out. If it wins, the intrinsics are worth costing.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example qk_gemm_probe
//! ```
use candle_core::{Device, Tensor};
use rayon::prelude::*;
use std::time::Instant;

const SEQ: usize = 1024;
const HD: usize = 64;
const HEADS: usize = 12;
const REPS: usize = 20;

/// `C = A @ Bt` for one head. `a` is `(SEQ, HD)`, `bt` is `(HD, SEQ)`, both
/// row-major contiguous; `c` is `(SEQ, SEQ)`.
///
/// Four rows of A at a time so each loaded `Bt` vector feeds four FMAs — the
/// ratio of arithmetic to loads is what decides whether this is compute-bound.
fn qk_block(a: &[f32], bt: &[f32], c: &mut [f32]) {
    const R: usize = 4;
    for (ib, cblk) in c.chunks_mut(R * SEQ).enumerate() {
        let i0 = ib * R;
        for k in 0..HD {
            let btk = &bt[k * SEQ..(k + 1) * SEQ];
            // Broadcast the four A values for this k once.
            let av = [
                a[i0 * HD + k],
                a[(i0 + 1) * HD + k],
                a[(i0 + 2) * HD + k],
                a[(i0 + 3) * HD + k],
            ];
            for (r, av) in av.iter().enumerate() {
                let row = &mut cblk[r * SEQ..(r + 1) * SEQ];
                if k == 0 {
                    for (o, &b) in row.iter_mut().zip(btk) {
                        *o = av * b;
                    }
                } else {
                    for (o, &b) in row.iter_mut().zip(btk) {
                        *o = av.mul_add(b, *o);
                    }
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let q = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HD), &d)?;
    let k = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HD), &d)?;
    // The layout `PackedQkvOp` could write for free.
    let kt = k.transpose(2, 3)?.contiguous()?;

    let qv = q.flatten_all()?.to_vec1::<f32>()?;
    let ktv = kt.flatten_all()?.to_vec1::<f32>()?;

    let mine = |out: &mut [f32]| {
        out.par_chunks_mut(SEQ * SEQ).enumerate().for_each(|(h, c)| {
            qk_block(&qv[h * SEQ * HD..(h + 1) * SEQ * HD], &ktv[h * HD * SEQ..(h + 1) * HD * SEQ], c);
        });
    };

    // ---- correctness -----------------------------------------------------
    let mut out = vec![0f32; HEADS * SEQ * SEQ];
    mine(&mut out);
    let want = q.matmul(&k.t()?)?.flatten_all()?.to_vec1::<f32>()?;
    let (mut worst, mut scale) = (0f32, 0f32);
    for (a, b) in out.iter().zip(&want) {
        worst = worst.max((a - b).abs());
        scale = scale.max(b.abs());
    }
    println!("  max |ours - candle| = {worst:.3e}  (values up to {scale:.3e})");
    assert!(worst < 1e-3 * scale.max(1.0), "kernel disagrees with candle");

    // ---- ABBA ------------------------------------------------------------
    let (mut tm, mut tc) = (f64::INFINITY, f64::INFINITY);
    for r in 0..REPS {
        for first in if r % 2 == 0 { [true, false] } else { [false, true] } {
            let t = Instant::now();
            if first {
                mine(&mut out);
                std::hint::black_box(out[0]);
            } else {
                std::hint::black_box(q.matmul(&k.t()?)?.dims());
            }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if first {
                tm = tm.min(ms);
            } else {
                tc = tc.min(ms);
            }
        }
    }

    let gflop = 2.0 * (HEADS * SEQ * SEQ * HD) as f64 / 1e9;
    println!("\n  q.k^T, 12 heads, (1024,64)@(64,1024) — best of {REPS}\n");
    println!("  {:<32} {:>9} {:>9}", "impl", "min ms", "GF/s");
    println!("  {:-<32} {:->9} {:->9}", "", "", "");
    println!("  {:<32} {tc:>9.2} {:>9.0}", "candle gemm", gflop / (tc / 1e3));
    println!("  {:<32} {tm:>9.2} {:>9.0}", "ours (scalar, autovec)", gflop / (tm / 1e3));
    println!("  {:-<32} {:->9} {:->9}", "", "", "");
    println!("  {:<32} {:>8.2}x", "speedup", tc / tm);
    println!(
        "\n  x12 layers x17 tiles per caption: {:.1} s -> {:.1} s",
        tc * 12.0 * 17.0 / 1e3,
        tm * 12.0 * 17.0 / 1e3
    );
    println!("  (q.k^T is 18.3 % of a tile, so this scales straight through.)");
    Ok(())
}
