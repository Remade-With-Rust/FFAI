//! What does OUR text layer cost, op by op, at prefill length?
//!
//! PyTorch runs the 332 GFLOP text prefill at ~493 GF/s (675 ms, measured
//! directly by `corpora/refs/smolvlm_hf_text.py`). We run it at ~259 GF/s
//! (1283 ms). But `gemm_shapes` measured candle at **516-591 GF/s** on these
//! exact shapes, so our matmuls alone should be ~600 ms — meaning roughly half
//! our prefill is NOT matmul. This prices what `text::block` actually runs.
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const SEQ: usize = 1142;
const HID: usize = 576;
const HEADS: usize = 9;
const KV: usize = 3;
const HDIM: usize = 64;
const INTER: usize = 1536;
const LAYERS: f64 = 30.0;

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
    let x = Tensor::rand(-1.0f32, 1.0, (1, SEQ, HID), &d)?;
    let w = Tensor::rand(-1.0f32, 1.0, HID, &d)?;
    let wq = Tensor::rand(-1.0f32, 1.0, (HEADS * HDIM, HID), &d)?;
    let wkv = Tensor::rand(-1.0f32, 1.0, (KV * HDIM, HID), &d)?;
    let wi = Tensor::rand(-1.0f32, 1.0, (INTER, HID), &d)?;
    let wd = Tensor::rand(-1.0f32, 1.0, (HID, INTER), &d)?;
    let q = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HDIM), &d)?;
    let k3 = Tensor::rand(-1.0f32, 1.0, (1, KV, SEQ, HDIM), &d)?;
    let att = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, SEQ), &d)?;
    let wide = Tensor::rand(-1.0f32, 1.0, (1, SEQ, INTER), &d)?;
    let cos = Tensor::rand(-1.0f32, 1.0, (SEQ, HDIM / 2), &d)?;
    let sin = Tensor::rand(-1.0f32, 1.0, (SEQ, HDIM / 2), &d)?;

    let lin = |x: &Tensor, w: &Tensor| -> candle_core::Result<Tensor> {
        let (b, s, i) = x.dims3()?;
        let o = w.dim(0)?;
        x.reshape((b * s, i))?.matmul(&w.t()?)?.reshape((b, s, o))
    };

    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    let mut add = |n: &str, x: f64, ms: f64, rows: &mut Vec<(String, f64, f64)>| rows.push((n.into(), x, ms));

    add("rms_norm (ours) x2", 2.0, t(&mut || ffai_argus::text::rms_norm_for_probe(&x, &w, 1e-5)), &mut rows);
    add("q proj", 1.0, t(&mut || lin(&x, &wq)), &mut rows);
    add("k,v proj x2", 2.0, t(&mut || lin(&x, &wkv)), &mut rows);
    add("reshape+transpose+contiguous x3", 3.0, t(&mut || {
        x.reshape((1, SEQ, HEADS, HDIM))?.transpose(1, 2)?.contiguous()
    }), &mut rows);
    add("rope on q,k x2", 2.0, t(&mut || candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)), &mut rows);
    add("q.k^T (GQA regrouped)", 1.0, t(&mut || {
        q.reshape((1, KV, 3 * SEQ, HDIM))?.matmul(&k3.t()?)
    }), &mut rows);
    add("causal softmax (ours)", 1.0, t(&mut || ffai_argus::text::causal_softmax_for_probe(&att, 0)), &mut rows);
    add("attn.v (GQA regrouped)", 1.0, t(&mut || {
        att.reshape((1, KV, 3 * SEQ, SEQ))?.matmul(&k3)
    }), &mut rows);
    add("transpose back + reshape", 1.0, t(&mut || q.transpose(1, 2)?.reshape((1, SEQ, HID))), &mut rows);
    add("o proj", 1.0, t(&mut || lin(&x, &wq)), &mut rows);
    add("gate,up proj x2", 2.0, t(&mut || lin(&x, &wi)), &mut rows);
    add("swiglu (ours)", 1.0, t(&mut || ffai_argus::text::swiglu_for_probe(&wide, &wide)), &mut rows);
    add("down proj", 1.0, t(&mut || lin(&wide, &wd)), &mut rows);
    add("residual add x2", 2.0, t(&mut || &x + &x), &mut rows);

    let total: f64 = rows.iter().map(|r| r.1 * r.2).sum();
    rows.sort_by(|a, b| (b.1 * b.2).partial_cmp(&(a.1 * a.2)).expect("cmp"));
    println!("OUR text layer, seq {SEQ}, hidden {HID}, {HEADS} heads / {KV} kv, inter {INTER}\n");
    println!("  {:<38} {:>3} {:>9} {:>9} {:>7}", "op", "x", "ms/layer", "x30 ms", "share");
    println!("  {:-<38} {:->3} {:->9} {:->9} {:->7}", "", "", "", "", "");
    for (n, x, ms) in &rows {
        let m = x * ms;
        println!("  {n:<38} {x:>3.0} {m:>9.2} {:>9.0} {:>6.1}%", m * LAYERS, 100.0 * m / total);
    }
    println!("  {:-<38} {:->3} {:->9} {:->9} {:->7}", "", "", "", "", "");
    println!("  {:<38} {:>3} {total:>9.2} {:>9.0}", "TOTAL", "", total * LAYERS);
    println!("\n  Measured prefill is ~1283 ms; PyTorch's is 675 ms at the same shapes.");
    let _ = DType::F32;
    Ok(())
}
