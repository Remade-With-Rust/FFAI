//! Encoder anatomy bench — every primitive, isolated, with a roofline verdict.
//!
//! The stage profile says attention 47 % / MLP 45 % / conv 9 %. That is not
//! actionable: it does not say *which* operation inside attention is slow, nor
//! whether any of them are slow for a reason we can fix.
//!
//! This replicates each primitive at the exact shapes the encoder uses and
//! measures it in isolation (codec-analyzer's anatomy-bench pattern), then
//! classifies it:
//!
//! - **FLOPs** and achieved **GFLOP/s** — how close to the machine's compute
//!   ceiling an op runs.
//! - **Bytes** moved and achieved **GB/s** — how close to the memory ceiling.
//! - **Arithmetic intensity** (FLOPs/byte) — the roofline coordinate. Low
//!   intensity means the op is memory-bound *by construction* and no amount
//!   of vectorization will help it; high intensity running at low GFLOP/s
//!   means the kernel itself is leaving performance on the table.
//!
//! The machine's ceilings are measured, not assumed: a large square matmul
//! establishes the compute roof and a large copy establishes the memory roof,
//! so every number below is a fraction of what *this* box actually delivers.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example anatomy_encoder
//! ```

use std::time::Instant;

use ffai_core::candle::{Device, Tensor};

fn best_of<T>(n: usize, mut f: impl FnMut() -> T) -> f64 {
    // One warm-up: first touch pays page faults and any lazy init.
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

struct Row {
    name: &'static str,
    secs: f64,
    flops: f64,
    bytes: f64,
    /// How many times one encoder forward pass runs this op.
    per_pass: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    // tiny.en at the deployment config.
    let (seq, d, heads, layers) = (1500usize, 384usize, 6usize, 4usize);
    let head_dim = d / heads;
    let ff = d * 4;
    const F32: f64 = 4.0;

    println!("shapes: seq={seq} d_model={d} heads={heads} head_dim={head_dim} ff={ff} layers={layers}");

    // ---- machine ceilings, measured ----
    //
    // BOTH axes of this roofline were wrong, and the table said so for months
    // without anyone acting on it: ops printed at "220 % of memory peak" and
    // "158 % of memory peak", which are arithmetically impossible.
    //
    // * The MEMORY axis was calibrated with candle's `Tensor::copy`, which is
    //   SINGLE-THREADED. It reports ~9 GB/s on this box while ggml's own
    //   `whisper-bench -w 1` measures **30.8 GB/s** across 11-24 threads. Every
    //   "% of memory peak" was therefore ~3.4x too high, and the ops that
    //   exceeded 100 % were the instrument announcing the error.
    // * The COMPUTE axis was a 2048^3 SQUARE matmul. Attention contracts over
    //   K=64, where candle reaches a fraction of that. Scoring an attention
    //   kernel against the square peak invented 4.4x of headroom once already
    //   (mission plan 6.20); it must be scored at its own contraction depth.
    let big = Tensor::randn(0f32, 1., (2048, 2048), &dev)?;
    let mm_secs = best_of(3, || big.matmul(&big).expect("matmul"));
    let peak_gflops = 2.0 * 2048f64.powi(3) / mm_secs / 1e9;

    // Reference rates at the awkward attention shapes, measured AT THE SAME
    // BATCH the ops use. A single-head probe is not the ceiling for a 6-head
    // batched matmul — the batched call amortizes better and legitimately
    // beats it, which is how the first version of this fix printed "112 % of
    // peak". These are context, not a score: for a candle matmul, candle's own
    // rate at the same shape is circular. What they show is how far the SHAPE
    // sits below the hardware, which is the number 6.20 got wrong.
    let bq = Tensor::randn(0f32, 1., (1, heads, seq, head_dim), &dev)?.contiguous()?;
    let bk = Tensor::randn(0f32, 1., (1, heads, head_dim, seq), &dev)?.contiguous()?;
    let k64_secs = best_of(3, || bq.matmul(&bk).expect("k64"));
    let peak_k64 = 2.0 * (heads * seq * seq * head_dim) as f64 / k64_secs / 1e9;

    // Memory. Calibrate with something other than the system under test:
    // ggml's `whisper-bench -w 1` measures this box at ~30.8 GB/s, so a probe
    // reporting double that is measuring cache, not memory. The buffers below
    // are 128 MB each — far past this CPU's ~30 MB L3 — and the statistic is
    // the MEDIAN, because min-of-N rewards the run that got the luckiest
    // residency.
    let buf = Tensor::randn(0f32, 1., (4096, 4096), &dev)?;
    let copy_secs = best_of(5, || buf.copy().expect("copy"));
    let serial_gbs = 2.0 * 4096.0 * 4096.0 * F32 / copy_secs / 1e9;
    let n = 32 * 1024 * 1024; // 128 MB per buffer
    let src = vec![1f32; n];
    let mut dst = vec![0f32; n];
    let mut samples: Vec<f64> = (0..5)
        .map(|_| {
            use rayon::prelude::*;
            let t0 = Instant::now();
            dst.par_chunks_mut(65536)
                .zip(src.par_chunks(65536))
                .for_each(|(d, s)| d.copy_from_slice(s));
            std::hint::black_box(&dst);
            t0.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    let peak_gbs = 2.0 * n as f64 * F32 / samples[2] / 1e9;
    println!(
        "measured ceilings:\n  compute  {peak_gflops:.0} GFLOP/s (2048^3 square, the hardware limit)  \
         ·  {peak_k64:.0} GFLOP/s (batched K=64 — what the attention SHAPE allows)\n  \
         memory   {peak_gbs:.1} GB/s (threaded memcpy, 128 MB buffers, median)  \
         ·  {serial_gbs:.1} GB/s (candle serial copy)\n  \
         cross-check: ggml `whisper-bench -w 1` reports 30.8 GB/s on this box\n"
    );

    let x = Tensor::randn(0f32, 1., (1, seq, d), &dev)?;
    let x2 = x.reshape((seq, d))?;
    let w_proj = Tensor::randn(0f32, 1., (d, d), &dev)?;
    let w_fc1 = Tensor::randn(0f32, 1., (d, ff), &dev)?;
    let w_fc2 = Tensor::randn(0f32, 1., (ff, d), &dev)?;
    let q = Tensor::randn(0f32, 1., (1, heads, seq, head_dim), &dev)?.contiguous()?;
    let k_t = Tensor::randn(0f32, 1., (1, heads, head_dim, seq), &dev)?.contiguous()?;
    let v = Tensor::randn(0f32, 1., (1, heads, seq, head_dim), &dev)?.contiguous()?;
    let scores = Tensor::randn(0f32, 1., (1, heads, seq, seq), &dev)?.contiguous()?;
    let ff_act = Tensor::randn(0f32, 1., (seq, ff), &dev)?;

    let mut rows = Vec::new();

    // ---- attention ----
    rows.push(Row {
        name: "qkv/out projection",
        secs: best_of(5, || x2.matmul(&w_proj).expect("proj")),
        flops: 2.0 * seq as f64 * d as f64 * d as f64,
        bytes: (seq * d + d * d + seq * d) as f64 * F32,
        per_pass: 4 * layers, // q, k, v, out
    });
    rows.push(Row {
        name: "split_heads+contig",
        secs: best_of(5, || {
            x.reshape((1, seq, heads, head_dim))
                .and_then(|t| t.transpose(1, 2))
                .and_then(|t| t.contiguous())
                .expect("split")
        }),
        flops: 0.0,
        bytes: 2.0 * (seq * d) as f64 * F32,
        per_pass: 3 * layers,
    });
    rows.push(Row {
        name: "scores q@kT",
        secs: best_of(3, || q.matmul(&k_t).expect("qk")),
        flops: 2.0 * (heads * seq * seq * head_dim) as f64,
        bytes: (2 * heads * seq * head_dim + heads * seq * seq) as f64 * F32,
        per_pass: layers,
    });
    rows.push(Row {
        name: "softmax(scores)",
        secs: best_of(3, || {
            candle_nn::ops::softmax_last_dim(&scores).expect("softmax")
        }),
        flops: 5.0 * (heads * seq * seq) as f64, // exp + max + sum + div
        bytes: 2.0 * (heads * seq * seq) as f64 * F32,
        per_pass: layers,
    });
    rows.push(Row {
        name: "attn@v",
        secs: best_of(3, || scores.matmul(&v).expect("wv")),
        flops: 2.0 * (heads * seq * seq * head_dim) as f64,
        bytes: (heads * seq * seq + 2 * heads * seq * head_dim) as f64 * F32,
        per_pass: layers,
    });
    rows.push(Row {
        name: "merge heads",
        secs: best_of(5, || {
            v.transpose(1, 2).and_then(|t| t.flatten_from(2)).expect("merge")
        }),
        flops: 0.0,
        bytes: 2.0 * (seq * d) as f64 * F32,
        per_pass: layers,
    });

    // ---- MLP ----
    rows.push(Row {
        name: "mlp fc1",
        secs: best_of(5, || x2.matmul(&w_fc1).expect("fc1")),
        flops: 2.0 * (seq * d * ff) as f64,
        bytes: (seq * d + d * ff + seq * ff) as f64 * F32,
        per_pass: layers,
    });
    rows.push(Row {
        name: "gelu (candle/tanh)",
        secs: best_of(5, || ff_act.gelu().expect("gelu")),
        flops: 8.0 * (seq * ff) as f64,
        bytes: 2.0 * (seq * ff) as f64 * F32,
        per_pass: 0, // superseded — kept to show what was replaced
    });
    rows.push(Row {
        name: "gelu (ours/pade)",
        secs: best_of(5, || {
            ffai_mercury::asr::text_decoder::fast_gelu(&ff_act).expect("fast gelu")
        }),
        flops: 8.0 * (seq * ff) as f64,
        bytes: 2.0 * (seq * ff) as f64 * F32,
        per_pass: layers,
    });
    rows.push(Row {
        name: "mlp fc2",
        secs: best_of(5, || ff_act.matmul(&w_fc2).expect("fc2")),
        flops: 2.0 * (seq * ff * d) as f64,
        bytes: (seq * ff + ff * d + seq * d) as f64 * F32,
        per_pass: layers,
    });

    // ---- glue ----
    let ln_w = Tensor::randn(0f32, 1., d, &dev)?;
    let ln_b = Tensor::randn(0f32, 1., d, &dev)?;
    let ln = candle_nn::LayerNorm::new(ln_w, ln_b, 1e-5);
    rows.push(Row {
        name: "layer_norm",
        secs: best_of(5, || {
            <candle_nn::LayerNorm as candle_nn::Module>::forward(&ln, &x).expect("ln")
        }),
        flops: 6.0 * (seq * d) as f64,
        bytes: 2.0 * (seq * d) as f64 * F32,
        per_pass: 2 * layers,
    });
    rows.push(Row {
        name: "residual add",
        secs: best_of(5, || (&x + &x).expect("add")),
        flops: (seq * d) as f64,
        bytes: 3.0 * (seq * d) as f64 * F32,
        per_pass: 2 * layers,
    });

    // ---- report ----
    println!(
        "{:<22} {:>9} {:>6} {:>10} {:>9} {:>8} {:>8} {:>7}",
        "OP", "us/call", "x/pass", "TOTAL ms", "GFLOP/s", "GB/s", "FLOP/B", "BOUND"
    );
    let mut total_ms = 0.0;
    let mut ranked = Vec::new();
    for r in &rows {
        let total = r.secs * r.per_pass as f64 * 1000.0;
        total_ms += total;
        let gflops = r.flops / r.secs / 1e9;
        let gbs = r.bytes / r.secs / 1e9;
        let intensity = if r.bytes > 0.0 { r.flops / r.bytes } else { 0.0 };
        // Roofline: an op is memory-bound when its intensity sits below the
        // machine's balance point (peak FLOP/s divided by peak bytes/s).
        let balance = peak_gflops / peak_gbs;
        let bound = if intensity < balance { "MEM" } else { "CPU" };
        println!(
            "{:<22} {:>9.1} {:>6} {:>10.2} {:>9.1} {:>8.1} {:>8.2} {:>7}",
            r.name,
            r.secs * 1e6,
            r.per_pass,
            total,
            gflops,
            gbs,
            intensity,
            bound
        );
        ranked.push((total, r.name, gflops, gbs, bound));
    }
    println!("{:<22} {:>9} {:>6} {:>10.2}", "SUM", "", "", total_ms);
    println!(
        "\nroofline balance point: {:.1} FLOP/byte — below that an op is memory-bound\n",
        peak_gflops / peak_gbs
    );

    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("RANKED BY TOTAL COST PER ENCODER PASS");
    println!(
        "(% is against the HARDWARE peak. For attention the shape itself caps\n \
         the reachable rate at {peak_k64:.0} GFLOP/s — {:.0}% of hardware — so a\n \
         shortfall there is shape, not implementation. The question worth asking\n \
         of these rows is whether a DIFFERENT kernel beats candle at this exact\n \
         shape, which is what the fused kernel answers.)",
        peak_k64 / peak_gflops * 100.0
    );
    for (total, name, gflops, gbs, bound) in ranked.iter().take(8) {
        // Pick the ceiling this op could actually reach. Scoring a K=64
        // contraction against a 2048^3 square matmul is how 4.4x of imaginary
        // headroom got into the mission plan once already.
        let pct = if *bound == "CPU" {
            gflops / peak_gflops * 100.0
        } else {
            gbs / peak_gbs * 100.0
        };
        let of = if *bound == "CPU" { "hardware peak" } else { "memory peak" };
        // An impossible percentage means the ceiling is wrong, not that the op
        // is magic. Say so loudly instead of printing it as a curiosity.
        let flag = if pct > 100.0 { "  <<< IMPOSSIBLE — ceiling is miscalibrated" } else { "" };
        println!("  {total:>7.2} ms  {name:<22} {bound}-bound, {pct:.0}% of {of}{flag}");
    }
    Ok(())
}
