//! Round 2 re-anchor: are MY kernels actually faster than candle's?
//!
//! Round 1 replaced candle's GELU, softmax and `LayerNorm` with rayon versions
//! and measured a win. But the engine runs six tiles concurrently, and in that
//! mode the kernels are told to stand down (`set_kernels_parallel(false)`) —
//! so the code that actually ships for a 17-tile image is my kernels running
//! **serially**, which round 1 never priced on its own.
//!
//! There is a specific reason to suspect it. Mine round-trip through a `Vec`:
//!
//! ```text
//!   xs.flatten_all()?.to_vec1::<f32>()?   // copy OUT   (12.6 MB for GELU)
//!   ...compute...
//!   Tensor::from_vec(v, shape, device)    // copy BACK  (12.6 MB)
//! ```
//!
//! That is three passes over memory where candle's native op does one. Parallel,
//! the fan-out pays for the copies. Serial, it may well not — and if it does
//! not, round 1 shipped a regression on the path that matters most.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example kernel_ab
//! ```

use candle_core::{Device, IndexOp, Module, Tensor, D};
use std::time::Instant;

const SEQ: usize = 1024;
const HID: usize = 768;
const INTER: usize = 3072;
const HEADS: usize = 12;

fn best(iters: usize, mut f: impl FnMut() -> candle_core::Result<Tensor>) -> f64 {
    let _ = f().expect("warm");
    let mut b = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let o = f().expect("op");
        std::hint::black_box(o.dims());
        b = b.min(t.elapsed().as_secs_f64());
    }
    b * 1e3
}

fn row(name: &str, candle_ms: f64, par_ms: f64, ser_ms: f64) {
    println!(
        "{name:<28} {candle_ms:>9.2} {par_ms:>9.2} {ser_ms:>9.2}   {:>6.2}x {:>6.2}x",
        candle_ms / par_ms,
        candle_ms / ser_ms
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let wide = Tensor::rand(-3.0f32, 3.0, (1, SEQ, INTER), &d)?;
    let scores = Tensor::rand(-3.0f32, 3.0, (1, HEADS, SEQ, SEQ), &d)?;
    let narrow = Tensor::rand(-3.0f32, 3.0, (1, SEQ, HID), &d)?;
    let ln = candle_nn::LayerNorm::new(
        Tensor::rand(0.9f32, 1.1, HID, &d)?,
        Tensor::zeros(HID, candle_core::DType::F32, &d)?,
        1e-6,
    );
    let it = 10;

    println!("candle vs ours (parallel) vs ours (serial) — the shapes SigLIP runs\n");
    println!(
        "{:<28} {:>9} {:>9} {:>9}   {:>6} {:>6}",
        "op", "candle", "ours par", "ours ser", "par", "ser"
    );
    println!("{}", "-".repeat(78));

    // ---- GELU on (1,1024,3072) --------------------------------------------
    let c = best(it, || wide.gelu());
    ffai_argus::siglip::set_kernels_parallel(true);
    let p = best(it, || ffai_argus::siglip::gelu_tanh_par(&wide));
    ffai_argus::siglip::set_kernels_parallel(false);
    let s = best(it, || ffai_argus::siglip::gelu_tanh_par(&wide));
    row("gelu (1,1024,3072)", c, p, s);

    // ---- softmax on (1,12,1024,1024) --------------------------------------
    let c = best(it, || candle_nn::ops::softmax_last_dim(&scores));
    ffai_argus::siglip::set_kernels_parallel(true);
    let p = best(it, || ffai_argus::siglip::softmax_last_dim_ours(&scores));
    ffai_argus::siglip::set_kernels_parallel(false);
    let sr = best(it, || ffai_argus::siglip::softmax_last_dim_ours(&scores));
    row("softmax (1,12,1024,1024)", c, p, sr);

    // layer_norm row removed: ours was refuted (0.47x serial, 0.91x parallel)
    // and reverted to candle's, so there are no longer two implementations to
    // compare. The numbers live in the doc comment on the refutation.
    let _ = &ln;

    ffai_argus::siglip::set_kernels_parallel(true);

    // ---- where the matmul time now goes ------------------------------------
    println!("\nmatmul efficiency at SigLIP's shapes (the new floor):");
    let q = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HID / HEADS), &d)?;
    let w = Tensor::rand(-0.05f32, 0.05, (HID, 3 * HID), &d)?;
    let cases: Vec<(&str, f64, f64)> = vec![
        (
            "fused qkv (1024,768)x(768,2304)",
            best(it, || narrow.flatten_to(1)?.matmul(&w)),
            2.0 * SEQ as f64 * HID as f64 * (3 * HID) as f64,
        ),
        (
            "q*k^T -> (1,12,1024,1024)",
            best(it, || q.matmul(&q.t()?)),
            2.0 * (HEADS * SEQ) as f64 * (HID / HEADS) as f64 * SEQ as f64,
        ),
        (
            "attn*v",
            best(it, || scores.matmul(&q)),
            2.0 * (HEADS * SEQ) as f64 * SEQ as f64 * (HID / HEADS) as f64,
        ),
    ];
    for (name, ms, flops) in &cases {
        // The output tensor is what q*k^T has to WRITE; at 50 MB that can be
        // the binding constraint rather than the arithmetic.
        println!(
            "  {name:<34} {ms:>8.2} ms  {:>6.0} GF/s",
            flops / (ms / 1e3) / 1e9
        );
    }
    let bytes = (HEADS * SEQ * SEQ * 4) as f64;
    println!(
        "\n  the (1,12,1024,1024) score matrix is {:.0} MB. q*k^T writes it, softmax\n  \
         reads+writes it, attn*v reads it: ~{:.0} MB of traffic per layer per tile\n  \
         for a tensor that is never needed whole.",
        bytes / 1e6,
        bytes * 4.0 / 1e6
    );
    qkv_shape_probe()?;
    blocked_attention_probe()?;
    qkv_layout_probe()?;
    gelu_simd_probe();
    qkv_layout_signtest()?;
    Ok(())
}

// Appended probe: is the FUSED qkv actually faster than three separate ones?
// Round 1 assumed it would be (one call, one pass over xs). The GF/s column
// above says otherwise, and a fused matmul that is slower per flop than the
// three it replaced is a pessimization wearing an optimization's name.
fn qkv_shape_probe() -> candle_core::Result<()> {
    let d = Device::Cpu;
    let x = Tensor::rand(-1.0f32, 1.0, (SEQ, HID), &d)?;
    let w1 = Tensor::rand(-0.05f32, 0.05, (HID, HID), &d)?;
    let w3 = Tensor::rand(-0.05f32, 0.05, (HID, 3 * HID), &d)?;
    let flops = 3.0 * 2.0 * SEQ as f64 * HID as f64 * HID as f64;

    let sep = best(10, || {
        let a = x.matmul(&w1)?;
        let b = x.matmul(&w1)?;
        let c = x.matmul(&w1)?;
        Tensor::cat(&[&a, &b, &c], 1)
    });
    let sep_nocat = best(10, || {
        let _a = x.matmul(&w1)?;
        let _b = x.matmul(&w1)?;
        x.matmul(&w1)
    });
    let fused = best(10, || x.matmul(&w3));
    println!("\nqkv: three (768x768) vs one (768x2304), same flops:");
    println!("  3 separate (+cat)   {sep:>8.2} ms  {:>6.0} GF/s", flops / (sep / 1e3) / 1e9);
    println!("  3 separate (no cat) {sep_nocat:>8.2} ms  {:>6.0} GF/s", flops / (sep_nocat / 1e3) / 1e9);
    println!("  1 fused             {fused:>8.2} ms  {:>6.0} GF/s", flops / (fused / 1e3) / 1e9);
    println!("  fused is {:.2}x the separate time", fused / sep_nocat);
    Ok(())
}

/// Would blocked attention pay? Measured before it is built.
///
/// Current path materialises `(1,12,1024,1024)` = 50 MB: q·kᵀ writes it,
/// softmax reads+writes it, attn·v reads it — ~200 MB per layer per tile, for
/// a tensor never needed whole.
///
/// Blocked: per head, per 128-row block, the score tile is 128x1024 = 512 KB
/// and stays in L2. Traffic collapses; CALL COUNT explodes (12 heads x 8
/// blocks x 2 matmuls = 192 per layer, against 2). Whether that trade pays is
/// a question about candle's per-call overhead, so measure it rather than
/// guess.
fn blocked_attention_probe() -> candle_core::Result<()> {
    let d = Device::Cpu;
    let hd = HID / HEADS;
    let q = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, hd), &d)?;
    let k = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, hd), &d)?;
    let v = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, hd), &d)?;

    let whole = best(6, || {
        let s = q.matmul(&k.t()?)?;
        let p = candle_nn::ops::softmax_last_dim(&s)?;
        p.matmul(&v)
    });

    let run_blocked = |block: usize| -> candle_core::Result<Tensor> {
        let mut heads = Vec::with_capacity(HEADS);
        for h in 0..HEADS {
            let qh = q.i((0, h))?;
            let kh = k.i((0, h))?.t()?.contiguous()?;
            let vh = v.i((0, h))?;
            let mut rows = Vec::with_capacity(SEQ.div_ceil(block));
            let mut r = 0;
            while r < SEQ {
                let n = block.min(SEQ - r);
                let s = qh.narrow(0, r, n)?.matmul(&kh)?;
                let p = candle_nn::ops::softmax_last_dim(&s)?;
                rows.push(p.matmul(&vh)?);
                r += n;
            }
            heads.push(Tensor::cat(&rows, 0)?);
        }
        Tensor::stack(&heads, 0)?.unsqueeze(0)
    };

    // Equality first: a faster arm computing something else is not an arm.
    let a = {
        let s = q.matmul(&k.t()?)?;
        let p = candle_nn::ops::softmax_last_dim(&s)?;
        p.matmul(&v)?
    };
    let b = run_blocked(128)?;
    let diff = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;

    println!("\nblocked attention (score tile stays in cache):");
    println!("  whole (1,12,1024,1024)   {whole:>8.2} ms   ~200 MB/layer traffic");
    for blk in [64usize, 128, 256, 512] {
        let ms = best(6, || run_blocked(blk));
        println!(
            "  blocked, {blk:>3}-row tiles     {ms:>8.2} ms   {:>5.2}x   ({} matmul calls)",
            whole / ms,
            HEADS * SEQ.div_ceil(blk) * 2
        );
    }
    println!("  max_abs vs whole: {diff:.3e}");
    Ok(())
}

/// Three `contiguous` copies, or one?
///
/// `Layer::forward` narrows q, k and v out of the fused matmul and gives each
/// its own `reshape -> transpose -> contiguous`: three copies of 3 MB. The
/// same layout is reachable with ONE copy of 9 MB — reshape the whole thing to
/// `(b, seq, 3, heads, hd)`, permute once, and narrow the result as free views.
///
/// Same bytes either way. The question is whether one big copy beats three
/// small ones, which is about locality and call overhead, not volume.
fn qkv_layout_probe() -> candle_core::Result<()> {
    let d = Device::Cpu;
    let hd = HID / HEADS;
    let qkv = Tensor::rand(-1.0f32, 1.0, (1, SEQ, 3 * HID), &d)?;

    let three = best(10, || {
        let mut last = None;
        for i in 0..3 {
            last = Some(
                qkv.narrow(D::Minus1, i * HID, HID)?
                    .reshape((1, SEQ, HEADS, hd))?
                    .transpose(1, 2)?
                    .contiguous()?,
            );
        }
        Ok(last.expect("three"))
    });

    let one = best(10, || {
        qkv.reshape((1, SEQ, 3, HEADS, hd))?
            .permute((0, 2, 3, 1, 4))?
            .contiguous()
    });

    // Equality: the one-copy layout must contain the same q as the three-copy one.
    let a = qkv
        .narrow(D::Minus1, 0, HID)?
        .reshape((1, SEQ, HEADS, hd))?
        .transpose(1, 2)?
        .contiguous()?;
    let b = qkv
        .reshape((1, SEQ, 3, HEADS, hd))?
        .permute((0, 2, 3, 1, 4))?
        .contiguous()?
        .i((0, 0))?
        .unsqueeze(0)?;
    let diff = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;

    println!("\nqkv layout: three copies vs one");
    println!("  3x narrow+transpose+contiguous  {three:>8.2} ms");
    println!("  1x reshape+permute+contiguous   {one:>8.2} ms   {:>5.2}x", three / one);
    println!("  max_abs (same q either way): {diff:.3e}");
    Ok(())
}

/// Scalar vs AVX2, same arithmetic, **one binary**.
///
/// Dispatch selecting AVX2 is not evidence that AVX2 is faster. This races the
/// two directly, interleaved, and reports a sign test as well as a ratio —
/// because on a box that swings +-12 % the ordering is trustworthy long before
/// the magnitude is.
///
/// The mechanism to expect: without `target_feature`, `round()` and `clamp()`
/// in the range reduction lower to libm CALLS, and a call cannot vectorise. The
/// AVX2 build turns both into single instructions (`vroundps`, `vmaxps`/
/// `vminps`) and widens the Horner chains to 8 lanes with FMA.
fn gelu_simd_probe() {
    let n = 1024 * 3072;
    let src: Vec<f32> = (0..n).map(|i| ((i % 977) as f32 - 488.0) / 61.0).collect();

    let time = |f: &dyn Fn(&mut [f32])| -> f64 {
        let mut v = src.clone();
        f(&mut v); // warm
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut v = src.clone();
            let t = Instant::now();
            f(&mut v);
            best = best.min(t.elapsed().as_secs_f64());
            std::hint::black_box(v[0]);
        }
        best * 1e3
    };

    let sc = &ffai_argus::siglip::gelu_scalar_for_probe;
    let av = &ffai_argus::siglip::gelu_avx2_for_probe;

    // Equality first, and it is NOT expected to be bit-identical: FMA fuses a
    // multiply-add that the scalar path rounds twice. Bound it instead.
    let (mut a, mut b) = (src.clone(), src.clone());
    sc(&mut a);
    av(&mut b);
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    let (s_ms, a_ms) = (time(sc), time(av));
    let mut wins = 0;
    for _ in 0..20 {
        if time(av) < time(sc) {
            wins += 1;
        }
    }
    println!("\ngelu kernel: scalar vs AVX2+FMA, one binary, {} M elements", n as f64 / 1e6);
    println!("  selected at runtime: {}", ffai_argus::siglip::gelu_kernel_name());
    println!("  scalar    {s_ms:>8.2} ms");
    println!("  avx2+fma  {a_ms:>8.2} ms   {:>5.2}x", s_ms / a_ms);
    println!("  sign test: AVX2 faster in {wins}/20 interleaved rounds");
    println!("  max_abs scalar vs avx2: {worst:.3e}  (FMA rounds once where scalar rounds twice)");
}

/// The qkv layout question, judged the way the AVX2 one was.
///
/// Deterministically there is NO volume difference — both paths copy 2 359 296
/// elements, so `copy_bytes` is identical and only `copies` falls 3 -> 1. Any
/// win is locality, which no counter can see, so the ordering has to carry the
/// claim instead of the magnitude.
fn qkv_layout_signtest() -> candle_core::Result<()> {
    let d = Device::Cpu;
    let hd = HID / HEADS;
    let qkv = Tensor::rand(-1.0f32, 1.0, (1, SEQ, 3 * HID), &d)?;

    let three = || -> candle_core::Result<Tensor> {
        let mut last = None;
        for i in 0..3 {
            last = Some(
                qkv.narrow(D::Minus1, i * HID, HID)?
                    .reshape((1, SEQ, HEADS, hd))?
                    .transpose(1, 2)?
                    .contiguous()?,
            );
        }
        Ok(last.expect("t"))
    };
    let one = || -> candle_core::Result<Tensor> {
        qkv.reshape((1, SEQ, 3, HEADS, hd))?
            .permute((0, 2, 3, 1, 4))?
            .contiguous()
    };

    let mut wins = 0;
    for _ in 0..20 {
        if best(3, one) < best(3, three) {
            wins += 1;
        }
    }
    println!("\nqkv layout: 3 copies vs 1 — sign test");
    println!("  DETERMINISTIC: copy calls 3 -> 1; copy BYTES identical (2 359 296 elements)");
    println!("  one-copy faster in {wins}/20 interleaved rounds");
    if wins >= 19 {
        println!("  -> stable under this box's noise; take it.");
    } else {
        println!("  -> NOT stable; refuse rather than average.");
    }
    Ok(())
}
