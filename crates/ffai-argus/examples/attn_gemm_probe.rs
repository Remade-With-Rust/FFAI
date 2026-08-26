//! Why do the two attention matmuls run at a THIRD of the linears' throughput?
//!
//! `vision_ops_now` (FLOPs derived from the shapes):
//!
//! | matmul | GF/s |
//! |---|---:|
//! | fc1 / fc2 / qkv / out_proj | 493-607 |
//! | **q.k^T** | **183** |
//! | **attn.v** | **289** |
//!
//! Together they are 25.7 % of a layer. Two candidate causes, both testable:
//! the STRIDED right operand (`k.t()` is a view, not a buffer), and the tiny
//! reduction depth (k = 64), which gives the GEMM little to amortise over.
use candle_core::{Device, Tensor};
use std::time::Instant;

const SEQ: usize = 1024;
const HEADS: usize = 12;
const HDIM: usize = 64;

fn t(f: &mut dyn FnMut() -> candle_core::Result<Tensor>) -> f64 {
    let _ = f().expect("warm");
    let mut b = f64::INFINITY;
    for _ in 0..7 {
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
    let k = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HDIM), &d)?;
    let v = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HDIM), &d)?;
    let probs = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, SEQ), &d)?;
    // k already stored transposed: (1, heads, hdim, seq), contiguous
    let kt = k.transpose(2, 3)?.contiguous()?;

    let gf = 2.0 * (HEADS * SEQ * HDIM * SEQ) as f64 / 1e9;
    let mut row = |n: &str, ms: f64| println!("  {n:<44} {ms:>8.2} ms  {:>7.0} GF/s", gf / (ms / 1e3));

    println!("q.k^T and attn.v, {HEADS} heads, seq {SEQ}, head_dim {HDIM}\n");
    row("q.matmul(k.t())            [what we do]", t(&mut || q.matmul(&k.t()?)));
    row("q.matmul(kt)   pre-transposed, contiguous", t(&mut || q.matmul(&kt)));
    row("probs.matmul(v)            [what we do]", t(&mut || probs.matmul(&v)));

    // Blocked over query rows: the score block stays cache-resident between
    // the two matmuls instead of round-tripping 50 MB through DRAM.
    for blk in [128usize, 256, 512] {
        let ms = t(&mut || {
            let mut acc = Vec::new();
            for s in (0..SEQ).step_by(blk) {
                let n = blk.min(SEQ - s);
                let qb = q.narrow(2, s, n)?.contiguous()?;
                let sc = qb.matmul(&k.t()?)?;
                let pr = candle_nn::ops::softmax_last_dim(&sc)?;
                acc.push(pr.matmul(&v)?);
            }
            Tensor::cat(&acc, 2)
        });
        println!("  {:<44} {ms:>8.2} ms  (q.k^T + softmax + attn.v)", format!("BLOCKED q-chunk {blk}"));
    }
    let fused = t(&mut || {
        let sc = q.matmul(&k.t()?)?;
        let pr = candle_nn::ops::softmax_last_dim(&sc)?;
        pr.matmul(&v)
    });
    println!("  {:<44} {fused:>8.2} ms  (the same three, unblocked)", "UNBLOCKED reference");
    Ok(())
}
