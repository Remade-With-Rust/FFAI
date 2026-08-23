//! Where does vision-tower time ACTUALLY go, op by op?
//!
//! `ffai bench vlm` says the vision stage is 77-82 % of a caption. §16 narrowed
//! that further: `SigLIP`-base is ~212 GFLOP per tile, and at the ~660 GFLOP/s
//! candle demonstrably reaches on this box that should take **0.32 s**, against
//! a measured **~1.3 s**. So roughly three quarters of the stage is not in the
//! multiplies.
//!
//! "Not in the multiplies" is a direction, not a finding. This prices every op
//! in one encoder layer at the real shapes, so the next move is aimed at a
//! measured cost rather than at a plausible one.
//!
//! # Reading it
//!
//! The `GFLOP/s` column is meaningful only for the matmul rows. For the
//! elementwise and layout rows the honest unit is **GB/s of tensor touched**,
//! because they are memory-bound — a transpose does no arithmetic at all, and
//! quoting it in FLOP/s would make the cheapest op look infinitely fast.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example vision_ops_probe
//! ```

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

/// `SigLIP`-base vision, as `SmolVLM`-256M configures it.
const SEQ: usize = 1024; // (512/16)^2 patches
const HID: usize = 768;
const HEADS: usize = 12;
const HEAD_DIM: usize = HID / HEADS; // 64
const INTER: usize = 3072;
const LAYERS: usize = 12;

struct Row {
    name: &'static str,
    per_layer: usize,
    ms: f64,
    /// Bytes read+written by one invocation; `None` for compute-bound rows.
    bytes: Option<usize>,
    /// FLOPs of one invocation; `None` for memory-bound rows.
    flops: Option<f64>,
}

fn bench(iters: usize, mut f: impl FnMut() -> candle_core::Result<Tensor>) -> f64 {
    let _ = f().expect("warm");
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let out = f().expect("op");
        // Touch the result so nothing can be elided; candle is eager, but a
        // read makes that explicit rather than assumed.
        std::hint::black_box(out.dims());
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e3
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let f = DType::F32;
    println!("SigLIP-base vision: seq={SEQ} hidden={HID} heads={HEADS} inter={INTER} layers={LAYERS}\n");

    // The tensors one encoder layer actually handles.
    let x = Tensor::rand(-1.0f32, 1.0, (1, SEQ, HID), &d)?;
    let w_qkv = Tensor::rand(-0.05f32, 0.05, (HID, HID), &d)?;
    let w_up = Tensor::rand(-0.05f32, 0.05, (HID, INTER), &d)?;
    let w_dn = Tensor::rand(-0.05f32, 0.05, (INTER, HID), &d)?;
    let heads4 = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HEAD_DIM), &d)?;
    let scores = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, SEQ), &d)?;
    let wide = Tensor::rand(-1.0f32, 1.0, (1, SEQ, INTER), &d)?;
    let ln_w = Tensor::rand(0.9f32, 1.1, HID, &d)?;
    let ln_b = Tensor::zeros(HID, f, &d)?;
    let ln = candle_nn::LayerNorm::new(ln_w, ln_b, 1e-6);

    let fl = |m: usize, k: usize, n: usize| Some(2.0 * m as f64 * k as f64 * n as f64);
    let by = |n: usize| Some(n * 4 * 2); // read + write, f32

    let mut rows: Vec<Row> = Vec::new();
    let it = 12;

    // ---- projections -------------------------------------------------------
    rows.push(Row {
        name: "linear q/k/v  (1024,768)x(768,768)",
        per_layer: 3,
        ms: bench(it, || x.flatten_to(1)?.matmul(&w_qkv)),
        bytes: None,
        flops: fl(SEQ, HID, HID),
    });
    rows.push(Row {
        name: "linear out    (1024,768)x(768,768)",
        per_layer: 1,
        ms: bench(it, || x.flatten_to(1)?.matmul(&w_qkv)),
        bytes: None,
        flops: fl(SEQ, HID, HID),
    });

    // ---- the layout work attention needs ----------------------------------
    rows.push(Row {
        name: "reshape+transpose+contiguous",
        per_layer: 3,
        ms: bench(it, || {
            x.reshape((1, SEQ, HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()
        }),
        bytes: by(SEQ * HID),
        flops: None,
    });

    // ---- attention ---------------------------------------------------------
    rows.push(Row {
        name: "matmul q*k^T -> (1,12,1024,1024)",
        per_layer: 1,
        ms: bench(it, || heads4.matmul(&heads4.t()?)),
        bytes: None,
        flops: fl(HEADS * SEQ, HEAD_DIM, SEQ),
    });
    rows.push(Row {
        name: "* scale on (1,12,1024,1024)",
        per_layer: 1,
        ms: bench(it, || &scores * 0.125),
        bytes: by(HEADS * SEQ * SEQ),
        flops: None,
    });
    rows.push(Row {
        name: "softmax_last_dim (1,12,1024,1024)",
        per_layer: 1,
        ms: bench(it, || candle_nn::ops::softmax_last_dim(&scores)),
        bytes: by(HEADS * SEQ * SEQ),
        flops: None,
    });
    rows.push(Row {
        name: "matmul attn*v",
        per_layer: 1,
        ms: bench(it, || scores.matmul(&heads4)),
        bytes: None,
        flops: fl(HEADS * SEQ, SEQ, HEAD_DIM),
    });
    rows.push(Row {
        name: "transpose+reshape back",
        per_layer: 1,
        ms: bench(it, || {
            heads4
                .transpose(1, 2)?
                .reshape((1, SEQ, HID))?
                .contiguous()
        }),
        bytes: by(SEQ * HID),
        flops: None,
    });

    // ---- mlp ---------------------------------------------------------------
    rows.push(Row {
        name: "linear up     (1024,768)x(768,3072)",
        per_layer: 1,
        ms: bench(it, || x.flatten_to(1)?.matmul(&w_up)),
        bytes: None,
        flops: fl(SEQ, HID, INTER),
    });
    rows.push(Row {
        name: "gelu on (1,1024,3072)",
        per_layer: 1,
        ms: bench(it, || wide.gelu()),
        bytes: by(SEQ * INTER),
        flops: None,
    });
    rows.push(Row {
        name: "gelu_erf on (1,1024,3072)",
        per_layer: 0, // reported for comparison, not counted
        ms: bench(it, || wide.gelu_erf()),
        bytes: by(SEQ * INTER),
        flops: None,
    });
    rows.push(Row {
        name: "linear down   (1024,3072)x(3072,768)",
        per_layer: 1,
        ms: bench(it, || wide.flatten_to(1)?.matmul(&w_dn)),
        bytes: None,
        flops: fl(SEQ, INTER, HID),
    });

    // ---- norms + residuals -------------------------------------------------
    rows.push(Row {
        name: "layer_norm (1,1024,768)",
        per_layer: 2,
        ms: bench(it, || {
            use candle_core::Module;
            ln.forward(&x)
        }),
        bytes: by(SEQ * HID),
        flops: None,
    });
    rows.push(Row {
        name: "residual add (1,1024,768)",
        per_layer: 2,
        ms: bench(it, || &x + &x),
        bytes: by(SEQ * HID),
        flops: None,
    });

    // ---- report ------------------------------------------------------------
    let counted: f64 = rows.iter().map(|r| r.ms * r.per_layer as f64).sum();
    println!(
        "{:<38} {:>3} {:>9} {:>10} {:>9} {:>7}",
        "op (one call)", "x", "ms/call", "ms/layer", "rate", "share"
    );
    println!("{}", "-".repeat(84));
    let mut matmul_ms = 0.0;
    let mut other_ms = 0.0;
    for r in &rows {
        let per = r.ms * r.per_layer as f64;
        let rate = match (r.flops, r.bytes) {
            (Some(fl), _) => format!("{:.0} GF/s", fl / (r.ms / 1e3) / 1e9),
            (_, Some(b)) => format!("{:.1} GB/s", b as f64 / (r.ms / 1e3) / 1e9),
            _ => String::new(),
        };
        if r.per_layer > 0 {
            if r.flops.is_some() {
                matmul_ms += per;
            } else {
                other_ms += per;
            }
        }
        println!(
            "{:<38} {:>3} {:>9.2} {:>10.2} {:>9} {:>6.1}%",
            r.name,
            r.per_layer,
            r.ms,
            per,
            rate,
            if r.per_layer > 0 { per / counted * 100.0 } else { 0.0 }
        );
    }
    println!("{}", "-".repeat(84));
    println!(
        "one layer: {counted:.1} ms   ->  {LAYERS} layers: {:.0} ms/tile  ->  17 tiles: {:.1} s",
        counted * LAYERS as f64,
        counted * LAYERS as f64 * 17.0 / 1e3
    );
    println!(
        "  matmul {:.1} ms/layer ({:.0}%)   everything else {:.1} ms/layer ({:.0}%)",
        matmul_ms,
        matmul_ms / counted * 100.0,
        other_ms,
        other_ms / counted * 100.0
    );
    println!(
        "\nIf `everything else` dominates, the lever is NOT a faster GEMM — candle's\n\
         already runs at PyTorch parity (§16). It is fewer passes over memory."
    );
    Ok(())
}
