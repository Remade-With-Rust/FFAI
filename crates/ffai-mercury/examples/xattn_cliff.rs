//! Does the m=1 GEMV cliff also hit cross-attention's BATCHED matmuls?
//! q@k is (1,6,1,64)@(1,6,64,1500): per head m=1, n=1500 — the cliff shape.
use ffai_core::candle::{DType, Device, Tensor};
use std::time::Instant;
fn best(n: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut b = f64::MAX;
    for _ in 0..n {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (h, kv, hd) = (6usize, 1500usize, 64usize);
    let k = Tensor::zeros((1, h, hd, kv), DType::F32, &dev)?;
    let v = Tensor::zeros((1, h, kv, hd), DType::F32, &dev)?;
    println!("cross-attention batched matmuls, query rows swept:");
    for m in [1usize, 2, 4, 8] {
        let q = Tensor::zeros((1, h, m, hd), DType::F32, &dev)?;
        let s = Tensor::zeros((1, h, m, kv), DType::F32, &dev)?;
        let tqk = best(200, || {
            std::hint::black_box(q.matmul(&k).unwrap());
        });
        let twv = best(200, || {
            std::hint::black_box(s.matmul(&v).unwrap());
        });
        println!(
            "  m={m}: q@k {:7.2} us   w@v {:7.2} us   sum {:7.2} us",
            tqk * 1e6,
            twv * 1e6,
            (tqk + twv) * 1e6
        );
    }
    Ok(())
}
