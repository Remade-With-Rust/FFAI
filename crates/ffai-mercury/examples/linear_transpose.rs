//! candle's Linear::forward does `weight.broadcast_left(..).t()` on EVERY
//! call. The vocabulary projection had exactly this bug and hoisting the
//! transpose to load time removed 91.5% of decoder time (mission plan 6.3).
//! Every attention projection and MLP layer in the decoder still goes through
//! Linear::forward. Same bug, smaller shapes?
use std::time::Instant;
use ffai_core::candle::{Device, Tensor};
use candle_nn::Module;
fn t(mut f: impl FnMut()) -> f64 { let s=Instant::now(); f(); s.elapsed().as_secs_f64()*1e3 }
fn probe(name: &str, k: usize, n: usize, dev: &Device) {
    let w = Tensor::randn(0f32,1.,(n,k),dev).unwrap();            // (out,in) as Linear stores it
    let b = Tensor::randn(0f32,1.,n,dev).unwrap();
    let lin = candle_nn::Linear::new(w.clone(), Some(b.clone()));
    let wt = w.t().unwrap().contiguous().unwrap();                 // pre-transposed (in,out)
    let x3 = Tensor::randn(0f32,1.,(1,1,k),dev).unwrap();          // decoder shape: 3-dim
    let rounds=201; let (mut win,mut va,mut vb)=(0usize,vec![],vec![]);
    let ours = || t(|| { std::hint::black_box(x3.reshape((1,k)).unwrap().matmul(&wt).unwrap().broadcast_add(&b).unwrap()); });
    let theirs = || t(|| { std::hint::black_box(lin.forward(&x3).unwrap()); });
    ours(); theirs();
    for i in 0..rounds {
        let (p,q) = if i%2==0 { let p=ours(); (p,theirs()) } else { let q=theirs(); (ours(),q) };
        if p<q { win+=1; } va.push(p); vb.push(q);
    }
    va.sort_by(f64::total_cmp); vb.sort_by(f64::total_cmp);
    let z=(win as f64-0.5*rounds as f64)/(0.5*(rounds as f64).sqrt());
    println!("  {name:<24} pre-transposed {:7.4} ms  Linear::forward {:7.4} ms  {win}/{rounds} z={z:+.1} ratio {:.2}x{}",
        va[rounds/2], vb[rounds/2], vb[rounds/2]/va[rounds/2],
        if z>2.0 {"  <- WIN"} else if z< -2.0 {"  <- lose"} else {"  (tie)"});
}
fn main() {
    let dev=Device::Cpu; let d=384usize;
    println!("decoder m=1 projections, 3-dim input (what the decoder actually passes):");
    probe("attn q/k/v/out 384x384", d, d, &dev);
    probe("mlp fc1 384x1536", d, d*4, &dev);
    probe("mlp fc2 1536x384", d*4, d, &dev);
}
