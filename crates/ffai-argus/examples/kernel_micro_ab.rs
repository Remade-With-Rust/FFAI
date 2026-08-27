//! Op-level ABBA A/B, for wins too small for the whole-tower harness to see.
//!
//! # Why this exists next to `vision_arm_ab`
//!
//! `vision_arm_ab` times the whole tower, which is the right instrument for a
//! change worth several percent of it. It is the WRONG instrument for a kernel
//! that is 1.9 % of the tower: a 4x win there moves the total by 1.4 %, and this
//! box cannot resolve 1.4 % — the fused-LayerNorm arm read 0.977x on the minimum
//! and 1.048x on the median in the same run, which is a measurement reporting
//! its own noise floor, not a result.
//!
//! So measure where the effect is 100 % of what the clock sees. Each op runs
//! alone, on its real shape, with both implementations interleaved ABBA in one
//! process. What this CANNOT tell you is the whole-tower impact — for that,
//! multiply by the op's share from `vision_inline_prof`, and be honest that a
//! 1 % tower effect is unverifiable on this box.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example kernel_micro_ab
//! ```
use candle_core::{Device, Module, Tensor};
use std::time::Instant;

const REPS: usize = 30;

/// Per-caption count for a per-layer op: 12 layers x 17 tiles.
const PER_CAPTION: f64 = 12.0 * 17.0;

fn abba(
    mut a: impl FnMut() -> candle_core::Result<()>,
    mut b: impl FnMut() -> candle_core::Result<()>,
) -> candle_core::Result<(f64, f64)> {
    a()?;
    b()?;
    let (mut ta, mut tb) = (f64::INFINITY, f64::INFINITY);
    for r in 0..REPS {
        // ABBA: flip the order every rep so a monotone drift cannot
        // systematically favour whichever arm runs first.
        for first in if r % 2 == 0 { [true, false] } else { [false, true] } {
            let t = Instant::now();
            if first {
                a()?;
            } else {
                b()?;
            }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if first {
                ta = ta.min(ms);
            } else {
                tb = tb.min(ms);
            }
        }
    }
    Ok((ta, tb))
}

fn report(op: &str, shape: &str, a_name: &str, a: f64, b_name: &str, b: f64, sites: f64) {
    println!("\n  {op}  {shape} — best of {REPS}\n");
    println!("  {:<30} {:>10}", "impl", "min ms");
    println!("  {:-<30} {:->10}", "", "");
    println!("  {b_name:<30} {b:>10.3}");
    println!("  {a_name:<30} {a:>10.3}");
    println!("  {:-<30} {:->10}", "", "");
    println!("  {:<30} {:>9.2}x", "speedup", b / a);
    println!(
        "  x{:.0} per caption: {:.0} ms -> {:.0} ms  ({:+.0} ms)",
        PER_CAPTION * sites,
        b * PER_CAPTION * sites,
        a * PER_CAPTION * sites,
        (a - b) * PER_CAPTION * sites
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let (seq, hidden, heads, hd) = (1024usize, 768usize, 12usize, 64usize);

    // ---- 1. LayerNorm: ours (one fused pass) vs candle's (a chain of six) --
    let x = Tensor::rand(-1.0f32, 1.0, (1, seq, hidden), &d)?;
    let w = Tensor::rand(0.5f32, 1.5, hidden, &d)?;
    let b = Tensor::rand(-0.1f32, 0.1, hidden, &d)?;
    let ln = candle_nn::LayerNorm::new(w, b, 1e-6);

    let mine = ffai_argus::siglip::layer_norm_for_probe(&ln, &x)?;
    let theirs = ln.forward(&x)?;
    let diff = (&mine - &theirs)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("  LayerNorm  max |ours - candle| = {diff:.3e}");
    assert!(diff < 1e-4, "fused LayerNorm disagrees with candle");

    let (ours, candle) = abba(
        || {
            std::hint::black_box(ffai_argus::siglip::layer_norm_for_probe(&ln, &x)?.dims());
            Ok(())
        },
        || {
            std::hint::black_box(ln.forward(&x)?.dims());
            Ok(())
        },
    )?;
    report(
        "LayerNorm",
        "(1, 1024, 768)",
        "ours (1 fused pass)",
        ours,
        "candle (6 chained passes)",
        candle,
        3.0,
    );

    // ---- 2. q.k^T: transposed VIEW vs a pre-transposed contiguous buffer ---
    //
    // `PackedQkvOp` already writes the k buffer, so storing it as
    // (b, heads, head_dim, seq) instead of (b, heads, seq, head_dim) is FREE —
    // the same copy, a different write order. The only question is whether
    // candle's gemm cares. If it does not, this is a refutation and the buffer
    // stays as it is.
    let q = Tensor::rand(-1.0f32, 1.0, (1, heads, seq, hd), &d)?;
    let k = Tensor::rand(-1.0f32, 1.0, (1, heads, seq, hd), &d)?;
    let kt = k.transpose(2, 3)?.contiguous()?; // (1, heads, hd, seq), contiguous

    let via_view = q.matmul(&k.t()?)?;
    let via_buf = q.matmul(&kt)?;
    let diff = (&via_view - &via_buf)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("\n  q.k^T  max |view - buffer| = {diff:.3e}  (expect 0 — same arithmetic)");

    let (buf, view) = abba(
        || {
            std::hint::black_box(q.matmul(&kt)?.dims());
            Ok(())
        },
        || {
            std::hint::black_box(q.matmul(&k.t()?)?.dims());
            Ok(())
        },
    )?;
    report(
        "q.k^T",
        "(1,12,1024,64) @ (1,12,64,1024)",
        "pre-transposed contiguous k",
        buf,
        "transposed view of k",
        view,
        1.0,
    );

    // ---- 3. q.k^T: does candle take a different path for 4D vs 3D? --------
    //
    // The tower hands candle `(1, 12, 1024, 64) @ (1, 12, 64, 1024)`. The
    // leading batch-of-1 is pure ceremony — the arithmetic is 12 independent
    // gemms either way — but a rank-4 input may fall down a more general
    // striding path than a rank-3 one. If it does, dropping the leading 1 is a
    // free win on the tower's LARGEST op (18.3 % of a tile). If it does not,
    // this is a refutation and costs one benchmark to establish.
    let q3 = q.reshape((heads, seq, hd))?;
    let k3 = k.reshape((heads, seq, hd))?;
    let r4 = q.matmul(&k.t()?)?.reshape((heads, seq, seq))?;
    let r3 = q3.matmul(&k3.t()?)?;
    let diff = (&r4 - &r3)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("
  q.k^T rank  max |4D - 3D| = {diff:.3e}  (expect 0 — same arithmetic)");

    let (three, four) = abba(
        || {
            std::hint::black_box(q3.matmul(&k3.t()?)?.dims());
            Ok(())
        },
        || {
            std::hint::black_box(q.matmul(&k.t()?)?.dims());
            Ok(())
        },
    )?;
    report(
        "q.k^T rank",
        "(12,1024,64) vs (1,12,1024,64)",
        "rank 3 (no leading batch)",
        three,
        "rank 4 (as the tower calls it)",
        four,
        1.0,
    );

    // ---- 4. Does batching all 17 tiles into one gemm pay? ----------------
    //
    // transformers runs ONE batch-17 forward where we run 17 passes, and that
    // is the last structural idea not yet tried here. It would raise the
    // projections from M = 1024 to M = 17408, amortising the weight load.
    //
    // But it also DELETES tile-level concurrency, which `engine::tile_workers`
    // documents as worth **2.50x** at 6 workers against 1.03x at one. So
    // batching only pays if a large-M gemm beats a small-M one by more than
    // that — a very high bar. This prices the premise before anything is built.
    let w_up = Tensor::rand(-1.0f32, 1.0, (hidden, 3072), &d)?;
    let x1 = Tensor::rand(-1.0f32, 1.0, (seq, hidden), &d)?;
    let x17 = Tensor::rand(-1.0f32, 1.0, (seq * 17, hidden), &d)?;

    let (big, small) = abba(
        || {
            std::hint::black_box(x17.matmul(&w_up)?.dims());
            Ok(())
        },
        || {
            // 17 separate passes, as the tower does today.
            for _ in 0..17 {
                std::hint::black_box(x1.matmul(&w_up)?.dims());
            }
            Ok(())
        },
    )?;
    println!("
  fc1-shaped gemm: 17 tiles batched vs 17 separate — best of {REPS}
");
    println!("  {:<30} {:>10}", "impl", "min ms");
    println!("  {:-<30} {:->10}", "", "");
    println!("  {:<30} {small:>10.1}", "17 x M=1024 (today)");
    println!("  {:<30} {big:>10.1}", "1 x M=17408 (batched)");
    println!("  {:-<30} {:->10}", "", "");
    println!("  {:<30} {:>9.2}x", "speedup from batching", small / big);
    println!(
        "  Needs to beat 2.50x (the tile concurrency it would replace) to pay."
    );

    // ---- 5. The connector's pixel shuffle: one fused pass vs two copies ---
    //
    // The connector lives inside `tower_ms` but OUTSIDE `vision_inline_prof`,
    // so it had never been measured op by op. Its shuffle is expressed as
    // reshape/transpose/reshape/transpose, and each transpose forces one of
    // candle's generic strided permutes — two full copies of the activation to
    // perform a single permutation.
    let hidden = Tensor::rand(-1.0f32, 1.0, (1, 1024, 768), &d)?;
    let (side, sf) = (32usize, 4usize);

    ffai_argus::vision::set_fused_shuffle(true);
    let fused = ffai_argus::vision::pixel_shuffle_for_probe(&hidden, side, sf)?;
    ffai_argus::vision::set_fused_shuffle(false);
    let chain = ffai_argus::vision::pixel_shuffle_for_probe(&hidden, side, sf)?;
    let diff = (&fused - &chain)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("
  pixel shuffle  max |fused - chain| = {diff:.3e}  (must be EXACTLY 0)");
    assert!(diff == 0.0, "pixel shuffle is pure movement; it must be bit-identical");

    let (one, two) = abba(
        || {
            ffai_argus::vision::set_fused_shuffle(true);
            std::hint::black_box(
                ffai_argus::vision::pixel_shuffle_for_probe(&hidden, side, sf)?.dims(),
            );
            Ok(())
        },
        || {
            ffai_argus::vision::set_fused_shuffle(false);
            std::hint::black_box(
                ffai_argus::vision::pixel_shuffle_for_probe(&hidden, side, sf)?.dims(),
            );
            Ok(())
        },
    )?;
    ffai_argus::vision::set_fused_shuffle(true);
    // Once per TILE, not per layer: 17 tiles a caption.
    println!("
  pixel shuffle  (1, 1024, 768) -> (1, 64, 12288) — best of {REPS}
");
    println!("  {:<30} {:>10}", "impl", "min ms");
    println!("  {:-<30} {:->10}", "", "");
    println!("  {:<30} {two:>10.3}", "chain (2 generic permutes)");
    println!("  {:<30} {one:>10.3}", "fused (1 pass)");
    println!("  {:-<30} {:->10}", "", "");
    println!("  {:<30} {:>9.2}x", "speedup", two / one);
    println!(
        "  x17 tiles per caption: {:.1} ms -> {:.1} ms  ({:+.1} ms)",
        two * 17.0,
        one * 17.0,
        (one - two) * 17.0
    );
    Ok(())
}
