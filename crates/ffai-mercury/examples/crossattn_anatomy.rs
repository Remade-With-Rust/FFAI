//! Cross-attention, cracked open to primitives.
//!
//! Cross-attention is ~30 % of decoder time against roughly 740 µs of real
//! arithmetic per token, so most of it is something other than the maths.
//! This times every single operation in `Attention::forward_cross` at the
//! exact shapes it runs at, one decoder token, tiny.en.
//!
//! Run with `--pass 1|2|3` to descend: pass 1 is the primitive list, pass 2
//! decomposes whatever pass 1 blames, pass 3 tests the fix candidates.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example crossattn_anatomy -- --pass 1
//! ```

use std::time::Instant;

use ffai_core::candle::{DType, Device, Tensor};

fn best<T>(n: usize, mut f: impl FnMut() -> T) -> f64 {
    std::hint::black_box(f());
    let mut b = f64::MAX;
    for _ in 0..n {
        let t = Instant::now();
        let o = f();
        std::hint::black_box(&o);
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn row(name: &str, us: f64, note: &str) {
    println!("  {name:<34} {us:>9.2} us   {note}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let pass: u32 = args
        .iter()
        .position(|a| a == "--pass")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let dev = Device::Cpu;
    let (d, heads, kv_len, layers) = (384usize, 6usize, 1500usize, 4usize);
    let head_dim = d / heads;

    // Exactly what forward_cross sees for one generated token.
    let x = Tensor::zeros((1, 1, d), DType::F32, &dev)?;
    let w = Tensor::zeros((d, d), DType::F32, &dev)?;
    let bias = Tensor::zeros(d, DType::F32, &dev)?;
    let lin = candle_nn::Linear::new(w, Some(bias));
    // The cross-attention cache: prepared once per window, reused per token.
    let k_cached = Tensor::zeros((1, heads, head_dim, kv_len), DType::F32, &dev)?;
    let v_cached = Tensor::zeros((1, heads, kv_len, head_dim), DType::F32, &dev)?;
    let q_heads = Tensor::zeros((1, heads, 1, head_dim), DType::F32, &dev)?;
    let scores = Tensor::zeros((1, heads, 1, kv_len), DType::F32, &dev)?;
    let ctx = Tensor::zeros((1, heads, 1, head_dim), DType::F32, &dev)?;

    println!(
        "cross-attention, ONE decoder token, tiny.en (d={d}, {heads} heads, kv_len={kv_len})\n"
    );

    if pass == 1 {
        println!("PASS 1 — every primitive in forward_cross");
        use candle_nn::Module;
        let t_qproj = best(200, || lin.forward(&x).unwrap());
        row("1. query projection (matmul)", t_qproj * 1e6, "1x384 @ 384x384");

        let t_split = best(200, || {
            x.reshape((1, 1, heads, head_dim)).unwrap().transpose(1, 2).unwrap()
        });
        row("2. split_heads (reshape+transpose)", t_split * 1e6, "view only");

        let q_v = x.reshape((1, 1, heads, head_dim))?.transpose(1, 2)?;
        let t_scale = best(200, || (&q_v * 0.35f64).unwrap());
        row("3. scale multiply", t_scale * 1e6, "384 elements");

        let scaled = (&q_v * 0.35f64)?;
        let t_contig = best(200, || scaled.contiguous().unwrap());
        row("4. contiguous", t_contig * 1e6, "384 elements");

        let t_qk = best(100, || q_heads.matmul(&k_cached).unwrap());
        row("5. scores q@k (batched)", t_qk * 1e6, "6x(1x64 @ 64x1500), reads 2.3 MB");

        let t_sm = best(200, || candle_nn::ops::softmax_last_dim(&scores).unwrap());
        row("6. softmax", t_sm * 1e6, "6x1500 elements");

        let t_wv = best(100, || scores.matmul(&v_cached).unwrap());
        row("7. weights@v (batched)", t_wv * 1e6, "6x(1x1500 @ 1500x64), reads 2.3 MB");

        let t_tr = best(200, || ctx.transpose(1, 2).unwrap());
        row("8. transpose", t_tr * 1e6, "view only");

        let merged = ctx.transpose(1, 2)?;
        let t_flat = best(200, || merged.flatten_from(2).unwrap());
        row("9. flatten_from (forces copy)", t_flat * 1e6, "384 elements");

        let flat = merged.flatten_from(2)?;
        let t_out = best(200, || lin.forward(&flat).unwrap());
        row("10. out projection (matmul)", t_out * 1e6, "1x384 @ 384x384");

        let sum = t_qproj + t_split + t_scale + t_contig + t_qk + t_sm + t_wv + t_tr + t_flat + t_out;
        println!("\n  {:<34} {:>9.2} us  per layer", "SUM of primitives", sum * 1e6);
        println!("  {:<34} {:>9.2} us  x{layers} layers", "", sum * 1e6 * layers as f64);
        println!(
            "  {:<34} {:>9.2} us  (30.4% of the 6810 us/token measured in context)",
            "measured cross-attn stage", 0.304 * 6810.0
        );
        let modelled = sum * 1e6 * layers as f64;
        println!(
            "\n  UNEXPLAINED: {:.0} us/token ({:.0}% of the stage) is NOT in these primitives",
            0.304 * 6810.0 - modelled,
            (0.304 * 6810.0 - modelled) / (0.304 * 6810.0) * 100.0
        );
        println!("\n  the two batched matmuls read {:.1} MB/layer -> {:.1} MB/token over {layers} layers",
            2.0 * (heads * kv_len * head_dim) as f64 * 4.0 / 1e6,
            2.0 * (heads * kv_len * head_dim) as f64 * 4.0 * layers as f64 / 1e6);
    }

    if pass == 2 {
        println!("PASS 2 — the batched matmuls, decomposed");
        // Pass 1 blames the two batched matmuls. Are they slow because of the
        // bytes they read, or because candle's batched path handles m=1 badly?
        let t_qk = best(100, || q_heads.matmul(&k_cached).unwrap());
        let bytes = (heads * kv_len * head_dim) as f64 * 4.0;
        row("batched q@k, 6 heads", t_qk * 1e6, &format!("{:.1} GB/s", bytes / t_qk / 1e9));

        // Same arithmetic and bytes, flattened to ONE matmul instead of 6.
        let q_flat = Tensor::zeros((1, heads * head_dim), DType::F32, &dev)?;
        let k_flat = Tensor::zeros((heads * head_dim, kv_len), DType::F32, &dev)?;
        let t_flat = best(100, || q_flat.matmul(&k_flat).unwrap());
        row("same bytes as ONE matmul", t_flat * 1e6, &format!("{:.1} GB/s", bytes / t_flat / 1e9));

        // A pure read of the same memory, to establish the floor.
        let t_copy = best(100, || k_cached.copy().unwrap());
        row("pure copy of k (floor)", t_copy * 1e6, &format!("{:.1} GB/s", 2.0 * bytes / t_copy / 1e9));

        println!(
            "\n  batched-vs-single penalty: {:.2}x  — if large, candle's batched matmul is the problem,",
            t_qk / t_flat
        );
        println!("  not the memory traffic, and reshaping to a single matmul is the fix.");
    }

    if pass == 3 {
        println!("PASS 3 — candidate fixes for the batched m=1 matmul");
        let bytes = (heads * kv_len * head_dim) as f64 * 4.0;
        let t_base = best(100, || q_heads.matmul(&k_cached).unwrap());
        row("baseline: batched matmul", t_base * 1e6, &format!("{:.1} GB/s", bytes / t_base / 1e9));

        // Candidate A: keep heads contiguous and do one (heads) x (head_dim, kv)
        // matmul per head in a loop — avoids candle's batched dispatch.
        let ks: Vec<Tensor> = (0..heads)
            .map(|h| k_cached.i((0, h)).unwrap().contiguous().unwrap())
            .collect();
        let qs: Vec<Tensor> = (0..heads)
            .map(|h| q_heads.i((0, h)).unwrap().contiguous().unwrap())
            .collect();
        let t_loop = best(100, || {
            let mut out = Vec::with_capacity(heads);
            for h in 0..heads {
                out.push(qs[h].matmul(&ks[h]).unwrap());
            }
            out
        });
        row("A: per-head loop of 2D matmuls", t_loop * 1e6, &format!("{:.1} GB/s", bytes / t_loop / 1e9));

        // Candidate B: one big matmul over a head-major layout.
        let q_flat = Tensor::zeros((1, heads * head_dim), DType::F32, &dev)?;
        let k_flat = Tensor::zeros((heads * head_dim, kv_len), DType::F32, &dev)?;
        let t_single = best(100, || q_flat.matmul(&k_flat).unwrap());
        row("B: single fused matmul", t_single * 1e6, &format!("{:.1} GB/s", bytes / t_single / 1e9));

        println!("\n  best candidate vs baseline: {:.2}x", t_base / t_loop.min(t_single));
    }
    Ok(())
}

use ffai_core::candle::IndexOp;
