//! How many tiles should the vision tower see at once?
//!
//! `describe` currently loops: one `forward` per tile, then `Tensor::stack`.
//! The tower and connector are both batch-aware — `(b, 3, H, W) -> (b, …)` —
//! so any chunk size from 1 to 17 is available.
//!
//! # Why this is a SWEEP and not an A/B
//!
//! The obvious framing is "looped vs batched", and it is the wrong one,
//! because the two ends of the range trade against different gates:
//!
//! * **Speed** wants a big chunk. CPU GEMM amortises packing the weight matrix
//!   across the batch, so seventeen `(1024, 768) x (768, 768)` products are
//!   worse than one `(17408, 768) x (768, 768)`.
//! * **Footprint** wants a small one, and non-linearly. `SigLIP`'s attention
//!   materialises `(b, heads, patches, patches)`; at `b = 17`, 12 heads and
//!   1024 patches that is `17 x 12 x 1024 x 1024 x 4 B` ≈ **855 MiB for one
//!   tensor**, against 50 MiB at `b = 1`. Peak RSS is one of the four gates,
//!   so a speed win bought there is a trade, not an improvement.
//!
//! The answer is therefore a number, not a yes/no — and it is picked from
//! measurement rather than from the two extremes.
//!
//! Both arms run in ONE process, interleaved, over the same tensors: a
//! before/after across two builds would compare two binaries on a box whose
//! state moved in between.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example tile_batching_ab
//! ```

use candle_core::{Device, Tensor};
use std::time::Instant;

/// Run the tower over every tile in chunks of `chunk`, returning the stacked
/// `(tiles, tokens, dim)` block — the same tensor for every chunk size.
fn tower_chunked(
    vision: &ffai_argus::vision::SmolVlmVision,
    pre: &ffai_argus::preprocess::Preprocessed,
    device: &Device,
    chunk: usize,
) -> Result<Tensor, candle_core::Error> {
    let per = 3 * pre.tile * pre.tile;
    let mut blocks: Vec<Tensor> = Vec::with_capacity(pre.tiles);
    let mut t = 0usize;
    while t < pre.tiles {
        let n = chunk.min(pre.tiles - t);
        let v = pre.pixel_values[t * per..(t + n) * per].to_vec();
        let tensor = Tensor::from_vec(v, (n, 3, pre.tile, pre.tile), device)?;
        let out = vision.forward(&tensor)?;
        for i in 0..n {
            blocks.push(out.i(i)?);
        }
        t += n;
    }
    Tensor::stack(&blocks, 0)
}

use candle_core::IndexOp;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("workspace root")?;
    let manifests = ffai_models::load_dir(&root.join("models"))?;
    let manifest = manifests
        .iter()
        .find(|m| m.name == ffai_argus::engine::MODEL)
        .ok_or("no smolvlm manifest")?;
    let resolved = manifest.fetch()?;
    let weights = resolved.file("model.safetensors")?.to_path_buf();
    let config = std::fs::read_to_string(resolved.file("config.json")?)?;

    let device = Device::Cpu;
    let vision = ffai_argus::vision::load(&weights, &config, &device)?;

    let (w, h) = (512usize, 512usize);
    let mut px = vec![0u8; w * h * 3];
    for (i, v) in px.iter_mut().enumerate() {
        *v = (i % 251) as u8;
    }
    let pre = ffai_argus::preprocess::preprocess_rgb8(&px, w, h);
    println!("{} tiles of {}x{}\n", pre.tiles, pre.tile, pre.tile);

    let chunks: Vec<usize> = vec![1, 2, 4, 8, 17];

    // EQUALITY FIRST. A speed comparison between arms computing different
    // tensors is not a speed comparison. Every chunk size must produce the
    // same block, or the fast one is fast for the wrong reason.
    let base = tower_chunked(&vision, &pre, &device, 1)?;
    for &c in &chunks[1..] {
        let other = tower_chunked(&vision, &pre, &device, c)?;
        let d = (&base - &other)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("chunk {c:>2}: max_abs vs chunk 1 = {d:.3e}");
        if d > 1e-4 {
            println!("  ⚠ chunk {c} DISAGREES with chunk 1 — stopping, a timing here is noise");
            return Ok(());
        }
    }
    println!();

    let time = |c: usize| -> f64 {
        let t = Instant::now();
        let _ = tower_chunked(&vision, &pre, &device, c).expect("arm");
        t.elapsed().as_secs_f64() * 1000.0
    };

    // Warm-up, discarded: the first call pays for lazily-built pools and
    // first-touch page faults, and charging that to whichever arm ran first is
    // the classic way to invent a difference.
    let _ = time(1);

    let mut best: Vec<(usize, f64)> = Vec::new();
    let mut runs: std::collections::BTreeMap<usize, Vec<f64>> = std::collections::BTreeMap::new();
    // Interleave the whole sweep three times rather than repeating each chunk
    // in a row, so a slow patch of wall-clock hits every arm equally.
    for _ in 0..3 {
        for &c in &chunks {
            runs.entry(c).or_default().push(time(c));
        }
        for &c in chunks.iter().rev() {
            runs.entry(c).or_default().push(time(c));
        }
    }
    println!("{:>6}  {:>10}  {:>8}   runs (ms)", "chunk", "min (ms)", "vs 1");
    let base_min = runs[&1].iter().copied().fold(f64::INFINITY, f64::min);
    for &c in &chunks {
        let v = &runs[&c];
        let m = v.iter().copied().fold(f64::INFINITY, f64::min);
        best.push((c, m));
        let fmt: Vec<String> = v.iter().map(|x| format!("{x:.0}")).collect();
        println!("{c:>6}  {m:>10.0}  {:>7.3}x   {}", base_min / m, fmt.join(" "));
    }

    let (bc, bm) = best
        .iter()
        .copied()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("a best");
    println!("\nfastest: chunk {bc} at {bm:.0} ms ({:.3}x chunk 1)", base_min / bm);
    println!(
        "attention tensor at that chunk: {} x 12 x 1024 x 1024 x 4 B = {:.0} MiB",
        bc,
        (bc as f64 * 12.0 * 1024.0 * 1024.0 * 4.0) / (1024.0 * 1024.0)
    );
    // --- stage split -------------------------------------------------------
    //
    // Two hypotheses for the 2.4x are already dead: chunking gains 1.07x, and
    // candle's CPU GEMM measures 589-702 GFLOP/s against PyTorch's 680-697 —
    // parity. So the time is neither in the SHAPE of our calls nor in the
    // multiply itself. This split asks the remaining question: is the residual
    // inside candle's SigLIP, or inside OUR connector?
    let one = Tensor::from_vec(
        pre.pixel_values[..3 * pre.tile * pre.tile].to_vec(),
        (1, 3, pre.tile, pre.tile),
        &device,
    )?;
    let hidden = vision.tower(&one)?;
    let best_of = |f: &dyn Fn() -> Result<Tensor, candle_core::Error>| -> f64 {
        let _ = f().expect("warm");
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let _ = f().expect("stage");
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };
    let t_tower = best_of(&|| vision.tower(&one));
    let t_conn = best_of(&|| vision.connect(&hidden));

    // SigLIP-base, per tile: 12 layers x (qkv 3.62 + scores/AV 3.22 +
    // out-proj 1.21 + mlp 9.66) GFLOP = 212 GFLOP.
    const TILE_GFLOP: f64 = 212.0;
    println!("\nper-tile stage split (min of 5):");
    println!(
        "  tower      {:>8.0} ms  -> {:>6.0} GFLOP/s effective",
        t_tower * 1e3,
        TILE_GFLOP / t_tower
    );
    println!(
        "  connector  {:>8.1} ms  ({:.1}% of the tile)",
        t_conn * 1e3,
        100.0 * t_conn / (t_tower + t_conn)
    );
    println!(
        "\n  A tower running far under the ~660 GFLOP/s this same box reaches in a\n\
         \x20 bare matmul means the residual is ELEMENTWISE + LAYOUT work — layernorm,\n\
         \x20 softmax, GELU, transposes, `contiguous()` copies — not the products."
    );

    println!(
        "\nNOTE: this prices the TOWER only. The decode prefill is the other half \n\
         of a caption, and peak RSS — a gate in its own right — must be read from \n\
         `ffai bench vlm`, not from here."
    );
    Ok(())
}
