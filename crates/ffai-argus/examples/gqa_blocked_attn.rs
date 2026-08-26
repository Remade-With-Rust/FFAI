//! Blocked attention, re-measured with the GQA layout — the premise CHANGED.
//!
//! Blocked/flash attention has been refuted five times. Every one of those
//! tests re-read k and v as **9 heads** (post-`repeat_kv`, 2.6 MB each), so a
//! 9-block sweep re-read ~47 MB — exactly cancelling the 47 MB score matrix it
//! was trying to avoid. That is why it lost, monotonically.
//!
//! `text.rs` now regroups q for GQA instead of expanding k/v, so k and v stay
//! at **3 heads, 877 KB each**. Re-reading those per block costs ~16 MB against
//! ~141 MB of score-matrix traffic (write + softmax read/write + read). The
//! arithmetic is no longer self-cancelling, so the verdict has to be re-taken
//! rather than inherited.
use candle_core::{Device, Tensor};
use std::time::Instant;

const SEQ: usize = 1142;
const HEADS: usize = 9;
const KV: usize = 3;
const HDIM: usize = 64;
const REPS: usize = HEADS / KV;

fn t(f: &mut dyn FnMut() -> candle_core::Result<Tensor>) -> f64 {
    let _ = f().expect("warm");
    let mut b = f64::INFINITY;
    for _ in 0..5 {
        let s = Instant::now();
        let o = f().expect("op");
        std::hint::black_box(o.dims());
        b = b.min(s.elapsed().as_secs_f64());
    }
    b * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let q = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HDIM), &d)?;
    let k = Tensor::rand(-1.0f32, 1.0, (1, KV, SEQ, HDIM), &d)?;
    let v = Tensor::rand(-1.0f32, 1.0, (1, KV, SEQ, HDIM), &d)?;
    let mb = |n: usize| n as f64 * 4.0 / 1e6;
    println!("GQA attention, seq {SEQ}, {HEADS} q-heads / {KV} kv-heads, head_dim {HDIM}");
    println!("  score matrix {:.1} MB   k,v together {:.1} MB\n", mb(HEADS*SEQ*SEQ), 2.0*mb(KV*SEQ*HDIM));

    // Unblocked: exactly what text::block does today.
    let whole = t(&mut || {
        let qg = q.reshape((1, KV, REPS * SEQ, HDIM))?;
        let att = qg.matmul(&k.t()?)?;
        let att = att.reshape((1, HEADS, SEQ, SEQ))?;
        att.inplace_op1(&ffai_argus::text::CausalSoftmaxProbe { offset: 0 })?;
        att.reshape((1, KV, REPS * SEQ, SEQ))?.matmul(&v)
    });
    println!("  {:<34} {whole:>8.2} ms", "unblocked [what we ship]");

    for blk in [128usize, 256, 512] {
        let ms = t(&mut || {
            let mut out = Vec::new();
            for s0 in (0..SEQ).step_by(blk) {
                let n = blk.min(SEQ - s0);
                // q block, still grouped so k/v are never expanded.
                let qb = q.narrow(2, s0, n)?.contiguous()?
                    .reshape((1, KV, REPS * n, HDIM))?;
                let a = qb.matmul(&k.t()?)?.reshape((1, HEADS, n, SEQ))?;
                a.inplace_op1(&ffai_argus::text::CausalSoftmaxProbe { offset: s0 })?;
                out.push(a.reshape((1, KV, REPS * n, SEQ))?.matmul(&v)?);
            }
            Tensor::cat(&out, 2)
        });
        println!("  {:<34} {ms:>8.2} ms   {:>5.2}x", format!("blocked, q-chunk {blk}"), whole / ms);
    }
    println!("\n  A ratio above 1.00x would overturn a five-times refutation; below it");
    println!("  confirms it under the new layout and closes the question for good.");
    Ok(())
}
