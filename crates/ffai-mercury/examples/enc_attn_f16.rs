//! Encoder attention chain in f16 vs f32.
//! q@kT is (1500x64)@(64x1500): only k=64 of reuse per output element, and it
//! writes a 54 MB score matrix — so it may be MEMORY-bound even though the
//! MLP at the same layer is compute-bound (where f16 loses 1.46x).
use ffai_core::candle::{DType, Device, Tensor};
use ffai_mercury::asr::text_decoder::fast_softmax;
use std::time::Instant;
fn t(mut f: impl FnMut()) -> f64 {
    let s = Instant::now();
    f();
    s.elapsed().as_secs_f64() * 1e3
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (h, seq, hd) = (6usize, 1500usize, 64usize);
    let q32 = Tensor::randn(0f32, 1., (1, h, seq, hd), &dev)?.contiguous()?;
    let k32 = Tensor::randn(0f32, 1., (1, h, hd, seq), &dev)?.contiguous()?;
    let v32 = Tensor::randn(0f32, 1., (1, h, seq, hd), &dev)?.contiguous()?;
    let (q16, k16, v16) = (
        q32.to_dtype(DType::F16)?,
        k32.to_dtype(DType::F16)?,
        v32.to_dtype(DType::F16)?,
    );
    let f16chain = |_: ()| {
        let qk = q16.matmul(&k16).unwrap();
        let w = fast_softmax(&qk).unwrap();
        std::hint::black_box(w.matmul(&v16).unwrap());
    };
    let f32chain = |_: ()| {
        let qk = q32.matmul(&k32).unwrap();
        let w = fast_softmax(&qk).unwrap();
        std::hint::black_box(w.matmul(&v32).unwrap());
    };
    let rounds = 15;
    let (mut w, mut va, mut vb) = (0usize, vec![], vec![]);
    t(|| f16chain(()));
    t(|| f32chain(()));
    for i in 0..rounds {
        let (p, q) = if i % 2 == 0 {
            let p = t(|| f16chain(()));
            (p, t(|| f32chain(())))
        } else {
            let q = t(|| f32chain(()));
            (t(|| f16chain(())), q)
        };
        if p < q {
            w += 1;
        }
        va.push(p);
        vb.push(q);
    }
    va.sort_by(f64::total_cmp);
    vb.sort_by(f64::total_cmp);
    let z = (w as f64 - 0.5 * rounds as f64) / (0.5 * (rounds as f64).sqrt());
    println!("ENCODER attention chain, one layer");
    println!("  f16  med {:7.2} ms", va[rounds / 2]);
    println!("  f32  med {:7.2} ms", vb[rounds / 2]);
    println!(
        "  paired f16 won {w}/{rounds} (z={z:+.1}) ratio {:.2}x -> {}",
        vb[rounds / 2] / va[rounds / 2],
        if z.abs() < 2.0 {
            "INCONCLUSIVE"
        } else if z > 0.0 {
            "f16 FASTER"
        } else {
            "f32 FASTER"
        }
    );
    Ok(())
}
