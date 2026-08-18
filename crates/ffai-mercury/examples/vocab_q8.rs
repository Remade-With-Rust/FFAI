//! The vocabulary projection is 27% of decode and pure bandwidth: it streams
//! the whole embedding matrix per token. We ship f16 (40 MB). Q8_0 is 20 MB.
//! Interleaved paired A/B at the real shape, plus the accuracy cost.
use ffai_core::candle::Module;
use ffai_core::candle::quantized::{GgmlDType, QMatMul, QTensor};
use ffai_core::candle::{DType, Device, Tensor};
use std::time::Instant;
fn t(mut f: impl FnMut()) -> f64 {
    let s = Instant::now();
    f();
    s.elapsed().as_secs_f64() * 1e3
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (d, vocab) = (384usize, 51864usize);
    let w32 = Tensor::randn(0f32, 1., (vocab, d), &dev)?; // (vocab,d) as stored
    let x = Tensor::randn(0f32, 1., (1, d), &dev)?;
    // shipped path: pre-transposed f16 + pad to m
    let wt16 = w32.t()?.contiguous()?.to_dtype(DType::F16)?;
    let x16 = x.to_dtype(DType::F16)?;
    let pad = 2usize;
    // candidate: Q8_0 QMatMul (no transpose needed, contracts the last dim)
    let q8 = QMatMul::from_qtensor(QTensor::quantize(&w32, GgmlDType::Q8_0)?)?;
    // accuracy vs the f32 reference
    let refv: Vec<f32> = x.matmul(&w32.t()?)?.flatten_all()?.to_vec1()?;
    let f16v: Vec<f32> = Tensor::cat(&vec![&x16; pad], 0)?
        .matmul(&wt16)?
        .narrow(0, 0, 1)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let q8v: Vec<f32> = q8.forward(&x)?.flatten_all()?.to_vec1()?;
    let e = |v: &Vec<f32>| {
        v.iter()
            .zip(&refv)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    };
    // does the ARGMAX agree? that is what actually decides a token
    let am = |v: &Vec<f32>| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    println!(
        "accuracy vs f32: f16 max|d| {:.3e} argmax {}   q8_0 max|d| {:.3e} argmax {}   ref argmax {}",
        e(&f16v),
        am(&f16v),
        e(&q8v),
        am(&q8v),
        am(&refv)
    );
    let rounds = 41;
    let (mut w, mut va, mut vb) = (0usize, vec![], vec![]);
    let a = || {
        t(|| {
            std::hint::black_box(q8.forward(&x).unwrap());
        })
    };
    let b = || {
        t(|| {
            std::hint::black_box(
                Tensor::cat(&vec![&x16; pad], 0)
                    .unwrap()
                    .matmul(&wt16)
                    .unwrap()
                    .narrow(0, 0, 1)
                    .unwrap(),
            );
        })
    };
    a();
    b();
    for i in 0..rounds {
        let (p, q) = if i % 2 == 0 {
            let p = a();
            (p, b())
        } else {
            let q = b();
            (a(), q)
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
    println!("vocabulary projection, {rounds} paired rounds");
    println!("  Q8_0 (20 MB)      med {:7.3} ms", va[rounds / 2]);
    println!("  f16+pad (40 MB)   med {:7.3} ms", vb[rounds / 2]);
    println!(
        "  Q8_0 won {w}/{rounds} (z={z:+.1}) ratio {:.2}x -> {}",
        vb[rounds / 2] / va[rounds / 2],
        if z.abs() < 2.0 {
            "INCONCLUSIVE"
        } else if z > 0.0 {
            "Q8_0 FASTER"
        } else {
            "f16 FASTER"
        }
    );
    Ok(())
}
