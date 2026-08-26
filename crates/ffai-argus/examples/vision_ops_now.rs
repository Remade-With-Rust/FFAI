//! What does OUR vision layer cost, op by op, today?
//!
//! # Why this exists next to `vision_ops_probe`
//!
//! `vision_ops_probe` prices candle's op mix — `wide.gelu()`, `&scores * 0.125`
//! — which is the **baseline that motivated `siglip.rs`**, not what `siglip.rs`
//! runs. Reading it as a current profile says GELU is 18.2 % of a layer at
//! 1.6 GB/s and that a `* scale` pass over 12.6 M elements is still there. Both
//! were deleted rounds ago. It is a historical instrument and still a correct
//! one; it just answers a question about the past.
//!
//! This prices the ops `Layer::forward` actually performs, in the order it
//! performs them.
//!
//! # The ranking is DETERMINISTIC; the milliseconds are not
//!
//! This box has been measured at a 4x spread within a single configuration, so
//! a 13 % effect cannot be resolved by a stopwatch here. The **bytes** column
//! is computed from the shapes and is exact on any machine under any load —
//! that is the column to optimise against. `ms` is shown because a byte that
//! moves at 1.6 GB/s costs twenty times a byte that moves at 32 GB/s, and the
//! product is what matters; but a `ms` verdict on two close rows is noise.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example vision_ops_now
//! ```
use candle_core::{DType, Device, Module, Tensor};
use std::time::Instant;

const SEQ: usize = 1024;
const HID: usize = 768;
const HEADS: usize = 12;
const HDIM: usize = 64;
const INTER: usize = 3072;
const LAYERS: f64 = 12.0;
const TILES: f64 = 17.0;

fn bench(f: &mut dyn FnMut() -> candle_core::Result<Tensor>) -> f64 {
    let _ = f().expect("warm");
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let o = f().expect("op");
        std::hint::black_box(o.dims());
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let x = Tensor::rand(-1.0f32, 1.0, (1, SEQ, HID), &d)?;
    let wide = Tensor::rand(-1.0f32, 1.0, (1, SEQ, INTER), &d)?;
    let qkv = Tensor::rand(-1.0f32, 1.0, (1, SEQ, 3 * HID), &d)?;
    let heads4 = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HDIM), &d)?;
    let scores = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, SEQ), &d)?;
    let w_qkv = Tensor::rand(-1.0f32, 1.0, (HID, 3 * HID), &d)?;
    let w_o = Tensor::rand(-1.0f32, 1.0, (HID, HID), &d)?;
    let w_up = Tensor::rand(-1.0f32, 1.0, (HID, INTER), &d)?;
    let w_dn = Tensor::rand(-1.0f32, 1.0, (INTER, HID), &d)?;
    let ln = candle_nn::LayerNorm::new(
        Tensor::ones(HID, DType::F32, &d)?,
        Tensor::zeros(HID, DType::F32, &d)?,
        1e-6,
    );

    let mb = |n: usize| (n as f64) * 4.0 / 1e6;
    let act = mb(SEQ * HID); // one activation tensor, 3.1 MB
    let big = mb(HEADS * SEQ * SEQ); // the score matrix, 50.3 MB
    let widemb = mb(SEQ * INTER); // 12.6 MB

    // (name, times per layer, bytes moved per call, closure)
    let mut rows: Vec<(String, f64, f64, f64)> = Vec::new();
    let mut add = |name: &str, x: f64, bytes: f64, ms: f64, rows: &mut Vec<_>| {
        rows.push((name.to_string(), x, bytes, ms));
    };

    add("ln1 + ln2  (candle LayerNorm)", 2.0, act * 3.0,
        bench(&mut || ln.forward(&x)), &mut rows);
    add("qkv linear  (1024,768)x(768,2304)", 1.0, act * (1.0 + 3.0) + mb(HID * 3 * HID),
        bench(&mut || x.flatten_to(1)?.matmul(&w_qkv)), &mut rows);
    add("packed reshape+permute+contiguous", 1.0, act * 3.0 * 2.0,
        bench(&mut || {
            qkv.reshape((1, SEQ, 3, HEADS, HDIM))?
                .permute((2, 0, 3, 1, 4))?
                .contiguous()
        }), &mut rows);
    add("q.k^T -> (1,12,1024,1024)", 1.0, act * 2.0 + big,
        bench(&mut || heads4.matmul(&heads4.t()?)), &mut rows);
    add("softmax_last_dim  (candle)", 1.0, big * 2.0,
        bench(&mut || candle_nn::ops::softmax_last_dim(&scores)), &mut rows);
    add("attn.v", 1.0, big + act * 2.0,
        bench(&mut || scores.matmul(&heads4)), &mut rows);
    add("transpose+reshape back (a copy)", 1.0, act * 2.0,
        bench(&mut || {
            heads4.transpose(1, 2)?.reshape((1, SEQ, HID))
        }), &mut rows);
    add("out_proj  (1024,768)x(768,768)", 1.0, act * 2.0 + mb(HID * HID),
        bench(&mut || x.flatten_to(1)?.matmul(&w_o)), &mut rows);
    add("residual add x2", 2.0, act * 3.0,
        bench(&mut || &x + &x), &mut rows);
    add("fc1 up  (1024,768)x(768,3072)", 1.0, act + widemb + mb(HID * INTER),
        bench(&mut || x.flatten_to(1)?.matmul(&w_up)), &mut rows);
    add("GELU  (OURS: zero-copy AVX2)", 1.0, widemb * 2.0,
        bench(&mut || ffai_argus::siglip::gelu_tanh_par(&wide)), &mut rows);
    add("fc2 down  (1024,3072)x(3072,768)", 1.0, widemb + act + mb(INTER * HID),
        bench(&mut || wide.flatten_to(1)?.matmul(&w_dn)), &mut rows);

    // ---- report, ranked by the cost that actually accrues ----------------
    let total_ms: f64 = rows.iter().map(|r| r.1 * r.3).sum();
    let total_mb: f64 = rows.iter().map(|r| r.1 * r.2).sum();
    rows.sort_by(|a, b| (b.1 * b.3).partial_cmp(&(a.1 * a.3)).expect("cmp"));

    println!("OUR SigLIP layer: seq {SEQ} hidden {HID} heads {HEADS} inter {INTER}\n");
    println!(
        "  {:<38} {:>3} {:>9} {:>9} {:>9} {:>7}",
        "op", "x", "MB/layer", "ms/layer", "GB/s", "share"
    );
    println!("  {:-<38} {:->3} {:->9} {:->9} {:->9} {:->7}", "", "", "", "", "", "");
    for (name, x, bytes, ms) in &rows {
        let (b, m) = (x * bytes, x * ms);
        println!(
            "  {name:<38} {x:>3.0} {b:>9.1} {m:>9.2} {:>9.1} {:>6.1}%",
            b / 1e3 / (m / 1e3),
            100.0 * m / total_ms
        );
    }
    println!("  {:-<38} {:->3} {:->9} {:->9} {:->9} {:->7}", "", "", "", "", "", "");
    println!("  {:<38} {:>3} {total_mb:>9.1} {total_ms:>9.2}", "TOTAL", "");
    println!(
        "\n  x{LAYERS} layers = {:.0} MB, {:.0} ms/tile   |   x{TILES} tiles = {:.1} GB, {:.1} s",
        total_mb * LAYERS,
        total_ms * LAYERS,
        total_mb * LAYERS * TILES / 1e3,
        total_ms * LAYERS * TILES / 1e3
    );
    println!("\n  Optimise the MB column — it is exact everywhere. Weight the rows by");
    println!("  GB/s: a slow byte costs more than a fast one, and the two rows moving");
    println!("  the most bytes are not necessarily the two costing the most time.");
    Ok(())
}
