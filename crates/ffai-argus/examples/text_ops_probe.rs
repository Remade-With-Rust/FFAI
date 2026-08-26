//! Where does prefill's non-matmul 83 % actually go?
//!
//! # What is already established
//!
//! * `text_scaling`: the whole text tower runs at **61-90 GF/s**.
//! * `gemm_shapes`: candle's matmul at the text tower's own shapes reaches
//!   **516-591 GF/s** — parity with the vision shapes. The matmuls are fine.
//! * Summing the measured matmuls per layer at seq 1142 gives ~31 ms; times 30
//!   layers that is **~930 ms of a ~5400 ms prefill**.
//!
//! So ~83 % of prefill is spent between the matmuls, and this prices those ops
//! at the real shapes so the next win is aimed rather than guessed. That is the
//! same instrument `vision_ops_probe` was, and it is here because the vision
//! campaign's four refutations all came from optimising before measuring.
//!
//! The rate column is the point: candle's CPU backend calls rayon for `conv2d`
//! and nothing else, so anything far under this box's ~10 GB/s memory
//! bandwidth is single-threaded elementwise work — the exact class that won
//! 14.5x on the vision tower's GELU.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example text_ops_probe
//! ```
use candle_core::{DType, Device, Tensor, D};
use std::time::Instant;

fn time<T>(f: impl Fn() -> candle_core::Result<T>) -> f64 {
    let _ = f().expect("warm");
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let o = f().expect("op");
        std::hint::black_box(&o);
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    // SmolLM2-135M at the caption's prompt length.
    const SEQ: usize = 1142;
    const HID: usize = 576;
    const INTER: usize = 1536;
    const HEADS: usize = 9;
    const KV: usize = 3;
    const HDIM: usize = 64;
    const LAYERS: f64 = 30.0;

    let x = Tensor::rand(-1.0f32, 1.0, (1, SEQ, HID), &d)?;
    let up = Tensor::rand(-1.0f32, 1.0, (1, SEQ, INTER), &d)?;
    let up2 = Tensor::rand(-1.0f32, 1.0, (1, SEQ, INTER), &d)?;
    let scores = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, SEQ), &d)?;
    let kv = Tensor::rand(-1.0f32, 1.0, (1, KV, SEQ, HDIM), &d)?;
    let q = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HDIM), &d)?;
    let cos = Tensor::rand(-1.0f32, 1.0, (SEQ, HDIM / 2), &d)?;
    let sin = Tensor::rand(-1.0f32, 1.0, (SEQ, HDIM / 2), &d)?;

    println!("prefill non-matmul ops, seq {SEQ}, best of 5\n");
    println!("  {:<38} {:>9} {:>9} {:>11} {:>9}", "op (per layer)", "ms", "MB", "GB/s", "x30 ms");
    println!("  {:-<38} {:->9} {:->9} {:->11} {:->9}", "", "", "", "", "");

    let mut rows: Vec<(String, f64)> = Vec::new();
    let mut row = |name: &str, ms: f64, mb: f64, rows: &mut Vec<(String, f64)>| {
        println!(
            "  {name:<38} {ms:>9.2} {mb:>9.1} {:>11.1} {:>9.0}",
            mb / 1e3 / (ms / 1e3),
            ms * LAYERS
        );
        rows.push((name.to_string(), ms * LAYERS));
    };

    // --- softmax over the score matrix ------------------------------------
    let mb = (HEADS * SEQ * SEQ) as f64 * 4.0 / 1e6;
    row("softmax_last_dim (1,9,S,S)", time(|| candle_nn::ops::softmax_last_dim(&scores)), mb * 3.0, &mut rows);

    // --- RMSNorm ----------------------------------------------------------
    let w = Tensor::rand(-1.0f32, 1.0, HID, &d)?;
    let mbn = (SEQ * HID) as f64 * 4.0 / 1e6;
    row("rms_norm (S,576)", time(|| candle_nn::ops::rms_norm(&x, &w, 1e-5)), mbn * 2.0, &mut rows);

    // --- SiLU + gate multiply (the gated MLP) ------------------------------
    let mbi = (SEQ * INTER) as f64 * 4.0 / 1e6;
    row("silu (S,1536)", time(|| up.silu()), mbi * 2.0, &mut rows);
    row("gate * up (S,1536)", time(|| &up * &up2), mbi * 3.0, &mut rows);

    // --- residual adds ----------------------------------------------------
    row("residual add (S,576) x2", time(|| &x + &x), mbn * 3.0 * 2.0, &mut rows);

    // --- repeat_kv: 3 kv heads -> 9 ---------------------------------------
    let mbkv = (HEADS * SEQ * HDIM) as f64 * 4.0 / 1e6;
    row(
        "repeat_kv 3->9 (x2 for k,v)",
        time(|| {
            let a = Tensor::cat(&[&kv, &kv, &kv], 1)?;
            a.contiguous()
        }) * 2.0,
        mbkv * 2.0 * 2.0,
        &mut rows,
    );

    // --- RoPE -------------------------------------------------------------
    row(
        "rope on q and k",
        time(|| candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)) * 2.0,
        (HEADS * SEQ * HDIM) as f64 * 4.0 / 1e6 * 2.0 * 2.0,
        &mut rows,
    );

    // --- the transposes attention needs -----------------------------------
    row(
        "transpose+contiguous (1,9,S,64)",
        time(|| q.transpose(1, 2)?.contiguous()),
        mbkv * 2.0,
        &mut rows,
    );

    // --- masking ----------------------------------------------------------
    let mask = Tensor::zeros((SEQ, SEQ), DType::F32, &d)?;
    row(
        "causal mask broadcast_add",
        time(|| scores.broadcast_add(&mask)),
        mb * 3.0,
        &mut rows,
    );
    let _ = D::Minus1;

    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("cmp"));
    let total: f64 = rows.iter().map(|r| r.1).sum();
    println!("\n  RANKED by cost over 30 layers (the whole prefill):\n");
    for (n, ms) in &rows {
        println!("    {:>8.0} ms  {:>5.1} %  {n}", ms, 100.0 * ms / total);
    }
    println!("\n    {:>8.0} ms  total non-matmul (measured prefill ~5400 ms,", total);
    println!("                       of which matmul is ~930 ms)");
    Ok(())
}
