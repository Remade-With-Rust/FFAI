//! Our softmax vs candle's at the ENCODER shape: 6 heads x 1500 x 1500 rows
//! (13.5M elements), f32 — 22% of encoder attention.
use std::time::Instant;
use ffai_core::candle::{Device, Tensor};
use ffai_mercury::asr::text_decoder::fast_softmax;
fn t(mut f: impl FnMut()) -> f64 { let s=Instant::now(); f(); s.elapsed().as_secs_f64()*1e3 }
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev=Device::Cpu;
    let x = Tensor::randn(0f32,1.,(1,6,1500,1500),&dev)?;
    let a: Vec<f32> = fast_softmax(&x)?.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = candle_nn::ops::softmax_last_dim(&x)?.flatten_all()?.to_vec1()?;
    println!("correctness: max |delta| {:.2e}", a.iter().zip(&b).map(|(p,q)|(p-q).abs()).fold(0f32,f32::max));
    let rounds=15; let (mut w,mut va,mut vb)=(0usize,vec![],vec![]);
    t(|| {std::hint::black_box(fast_softmax(&x).unwrap());});
    t(|| {std::hint::black_box(candle_nn::ops::softmax_last_dim(&x).unwrap());});
    for i in 0..rounds {
        let (p,q) = if i%2==0 {
            let p=t(|| {std::hint::black_box(fast_softmax(&x).unwrap());});
            (p, t(|| {std::hint::black_box(candle_nn::ops::softmax_last_dim(&x).unwrap());}))
        } else {
            let q=t(|| {std::hint::black_box(candle_nn::ops::softmax_last_dim(&x).unwrap());});
            (t(|| {std::hint::black_box(fast_softmax(&x).unwrap());}), q)
        };
        if p<q { w+=1; } va.push(p); vb.push(q);
    }
    va.sort_by(f64::total_cmp); vb.sort_by(f64::total_cmp);
    let z=(w as f64-0.5*rounds as f64)/(0.5*(rounds as f64).sqrt());
    println!("encoder-shape f32 softmax (13.5M elems)");
    println!("  ours     med {:7.2} ms", va[rounds/2]);
    println!("  candle   med {:7.2} ms", vb[rounds/2]);
    println!("  paired {w}/{rounds} (z={z:+.1}) ratio {:.2}x", vb[rounds/2]/va[rounds/2]);
    Ok(())
}
