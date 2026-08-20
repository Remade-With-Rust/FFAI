//! Does our softmax beat candle's, and does it unblock the f16 K/V cache?
use ffai_core::candle::{DType, Device, Tensor};
use ffai_mercury::asr::text_decoder::fast_softmax;
use std::time::Instant;
fn interleave(
    rounds: usize,
    mut a: impl FnMut() -> f64,
    mut b: impl FnMut() -> f64,
    la: &str,
    lb: &str,
    name: &str,
) {
    a();
    b();
    let (mut w, mut va, mut vb) = (0usize, vec![], vec![]);
    for i in 0..rounds {
        let (x, y) = if i % 2 == 0 {
            let x = a();
            (x, b())
        } else {
            let y = b();
            (a(), y)
        };
        if x < y {
            w += 1;
        }
        va.push(x);
        vb.push(y);
    }
    va.sort_by(f64::total_cmp);
    vb.sort_by(f64::total_cmp);
    let z = (w as f64 - 0.5 * rounds as f64) / (0.5 * (rounds as f64).sqrt());
    println!(
        "{name}\n  {la:<22} med {:7.3} ms\n  {lb:<22} med {:7.3} ms\n  paired {w}/{rounds} (z={z:+.1}) ratio {:.2}x -> {}",
        va[rounds / 2],
        vb[rounds / 2],
        vb[rounds / 2] / va[rounds / 2],
        if z.abs() < 2.0 {
            "INCONCLUSIVE"
        } else if z > 0.0 {
            la
        } else {
            lb
        }
    );
}
fn t(mut f: impl FnMut()) -> f64 {
    let s = Instant::now();
    f();
    s.elapsed().as_secs_f64() * 1e3
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (h, kv, hd) = (6usize, 1500usize, 64usize);
    let s32 = Tensor::randn(0f32, 1., (1, h, 1, kv), &dev)?;
    let s16 = s32.to_dtype(DType::F16)?;
    // correctness
    let a: Vec<f32> = fast_softmax(&s32)?.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = candle_nn::ops::softmax_last_dim(&s32)?
        .flatten_all()?
        .to_vec1()?;
    println!(
        "correctness f32: max |delta| {:.2e}",
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    );
    interleave(
        41,
        || {
            t(|| {
                std::hint::black_box(fast_softmax(&s16).unwrap());
            })
        },
        || {
            t(|| {
                std::hint::black_box(candle_nn::ops::softmax_last_dim(&s16).unwrap());
            })
        },
        "ours f16",
        "candle f16",
        "\nf16 softmax",
    );
    // the chain that f16 K/V needs to win
    let q32 = Tensor::randn(0f32, 1., (1, h, 1, hd), &dev)?.contiguous()?;
    let k32 = Tensor::randn(0f32, 1., (1, h, hd, kv), &dev)?.contiguous()?;
    let v32 = Tensor::randn(0f32, 1., (1, h, kv, hd), &dev)?.contiguous()?;
    let (q16, k16, v16) = (
        q32.to_dtype(DType::F16)?,
        k32.to_dtype(DType::F16)?,
        v32.to_dtype(DType::F16)?,
    );
    interleave(
        41,
        || {
            t(|| {
                let qk = q16.matmul(&k16).unwrap();
                let w = fast_softmax(&qk).unwrap();
                std::hint::black_box(w.matmul(&v16).unwrap());
            })
        },
        || {
            t(|| {
                let qk = q32.matmul(&k32).unwrap();
                let w = candle_nn::ops::softmax_last_dim(&qk).unwrap();
                std::hint::black_box(w.matmul(&v32).unwrap());
            })
        },
        "f16 K/V + our softmax",
        "all f32 (shipped)",
        "\nFULL cross-attn chain",
    );
    Ok(())
}
