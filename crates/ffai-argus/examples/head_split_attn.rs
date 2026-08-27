//! Split attention by HEAD, not by query rows.
//!
//! Six refutations of "blocked attention" all chunked the QUERY axis, which
//! forces every block to re-read the whole of k and v — the cost arrives
//! immediately and the cache win never does.
//!
//! The head axis is different: each head owns its own slice of q, k and v, so
//! **nothing is re-read**. All that changes is the live working set:
//!
//! | granularity | score bytes live at once |
//! |---|---:|
//! | all 9 heads (what we ship) | **46.9 MB** — thrashes a 32 MB L3 |
//! | one kv group (3 q-heads) | 15.6 MB |
//! | one head | **5.2 MB** — comfortably resident |
//!
//! That matters because `text_inline_prof` shows our matmuls running at
//! **435 GF/s in situ** where the same shapes reach **500-594 isolated** — the
//! gap is cache pressure from exactly this tensor.
//!
//! Arms are INTERLEAVED and judged by a sign test: this box has drifted 15 %
//! inside a single session, so ordering is trustworthy long before magnitude.
use candle_core::{Device, Tensor};
use std::time::Instant;

const SEQ: usize = 1142;
const HEADS: usize = 9;
const KV: usize = 3;
const HDIM: usize = 64;
const REPS: usize = HEADS / KV;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let q = Tensor::rand(-1.0f32, 1.0, (1, HEADS, SEQ, HDIM), &d)?;
    let k = Tensor::rand(-1.0f32, 1.0, (1, KV, SEQ, HDIM), &d)?;
    let v = Tensor::rand(-1.0f32, 1.0, (1, KV, SEQ, HDIM), &d)?;
    let sm = |t: &Tensor| -> candle_core::Result<()> {
        t.inplace_op1(&ffai_argus::text::CausalSoftmaxProbe { offset: 0 })
    };

    // ARM A — all heads at once, exactly what `text::block` ships.
    let whole = || -> candle_core::Result<Tensor> {
        let qg = q.reshape((1, KV, REPS * SEQ, HDIM))?;
        let att = qg.matmul(&k.t()?)?;
        let att = att.reshape((1, HEADS, SEQ, SEQ))?;
        sm(&att)?;
        att.reshape((1, KV, REPS * SEQ, SEQ))?
            .matmul(&v)?
            .reshape((1, HEADS, SEQ, HDIM))
    };

    // ARM B — one kv group at a time: 3 q-heads against their own k/v head.
    let by_group = || -> candle_core::Result<Tensor> {
        let mut outs = Vec::with_capacity(KV);
        for g in 0..KV {
            let qg = q.narrow(1, g * REPS, REPS)?.contiguous()?
                .reshape((1, 1, REPS * SEQ, HDIM))?;
            let kg = k.narrow(1, g, 1)?;
            let vg = v.narrow(1, g, 1)?;
            let att = qg.matmul(&kg.t()?)?.reshape((1, REPS, SEQ, SEQ))?;
            sm(&att)?;
            outs.push(att.reshape((1, 1, REPS * SEQ, SEQ))?.matmul(&vg)?
                .reshape((1, REPS, SEQ, HDIM))?);
        }
        Tensor::cat(&outs, 1)
    };

    // ARM C — one head at a time: 5.2 MB live.
    let by_head = || -> candle_core::Result<Tensor> {
        let mut outs = Vec::with_capacity(HEADS);
        for h in 0..HEADS {
            let qh = q.narrow(1, h, 1)?.contiguous()?;
            let kh = k.narrow(1, h / REPS, 1)?;
            let vh = v.narrow(1, h / REPS, 1)?;
            let att = qh.matmul(&kh.t()?)?;
            sm(&att)?;
            outs.push(att.matmul(&vh)?);
        }
        Tensor::cat(&outs, 1)
    };

    // correctness first
    let (a, b, c) = (whole()?, by_group()?, by_head()?);
    let cmp = |x: &Tensor, y: &Tensor| -> candle_core::Result<f32> {
        Ok((x - y)?.abs()?.max_all()?.to_scalar::<f32>()?)
    };
    println!("scores live at once: all-heads 46.9 MB | kv-group 15.6 MB | per-head 5.2 MB\n");
    println!("  max|diff| group vs whole : {:.3e}", cmp(&a, &b)?);
    println!("  max|diff| head  vs whole : {:.3e}\n", cmp(&a, &c)?);

    let once = |f: &dyn Fn() -> candle_core::Result<Tensor>| -> f64 {
        let t = Instant::now();
        let o = f().expect("arm");
        std::hint::black_box(o.dims());
        t.elapsed().as_secs_f64() * 1e3
    };
    for _ in 0..2 {
        let _ = (whole()?, by_group()?, by_head()?);
    }
    let (mut wa, mut wg, mut wh) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let (mut gwin, mut hwin) = (0, 0);
    for i in 0..12 {
        // Rotate which arm goes first so no arm keeps a warm-cache advantage.
        let (ta, tg, th) = if i % 3 == 0 {
            (once(&whole), once(&by_group), once(&by_head))
        } else if i % 3 == 1 {
            let g = once(&by_group);
            let h = once(&by_head);
            (once(&whole), g, h)
        } else {
            let h = once(&by_head);
            (once(&whole), once(&by_group), h)
        };
        wa = wa.min(ta);
        wg = wg.min(tg);
        wh = wh.min(th);
        if tg < ta { gwin += 1; }
        if th < ta { hwin += 1; }
    }
    println!("  {:<34} {wa:>8.2} ms", "all heads at once [ships]");
    println!("  {:<34} {wg:>8.2} ms   {:>5.2}x   faster in {gwin}/12", "by kv group (3 heads)", wa / wg);
    println!("  {:<34} {wh:>8.2} ms   {:>5.2}x   faster in {hwin}/12", "by head (5.2 MB live)", wa / wh);
    println!("\n  A sign test at 11/12 or better is a real ordering on this box.");
    Ok(())
}
