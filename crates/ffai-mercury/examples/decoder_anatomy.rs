//! Decoder anatomy: is the per-token cost bandwidth, or per-op overhead?
//!
//! The decoder runs ~80-100 candle ops per generated token on TINY tensors
//! (one row of activations). Two very different stories fit "3x slower":
//! either the matmuls are bandwidth-limited and we move more bytes, or the
//! tensors are so small that framework overhead per op dominates the work.
//! Those need opposite fixes, so measure which.
use ffai_core::candle::{DType, Device, Tensor};
use std::time::Instant;
fn best<T>(n: usize, mut f: impl FnMut() -> T) -> f64 {
    std::hint::black_box(f());
    let mut b = f64::MAX;
    for _ in 0..n {
        let t = Instant::now();
        let o = f();
        std::hint::black_box(&o);
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let d = 384usize;
    println!(
        "{:<28} {:>10} {:>12} {:>10} {:>12}",
        "OP (one decoder token)", "us", "weight KB", "GB/s", "% of 24GB/s"
    );
    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    let mut probe = |name: &str, k: usize, n: usize, dt: DType, count: usize| {
        let a = Tensor::zeros((1, k), DType::F32, &dev)
            .unwrap()
            .to_dtype(dt)
            .unwrap();
        let b = Tensor::zeros((k, n), DType::F32, &dev)
            .unwrap()
            .to_dtype(dt)
            .unwrap();
        let s = best(50, || a.matmul(&b).unwrap());
        let kb = (k * n) as f64 * if dt == DType::F32 { 4.0 } else { 2.0 } / 1e3;
        let gbs = kb * 1e3 / s / 1e9;
        println!(
            "{:<28} {:>10.1} {:>12.1} {:>10.1} {:>11.0}%",
            name,
            s * 1e6,
            kb,
            gbs,
            gbs / 24.0 * 100.0
        );
        rows.push((name.to_string(), s * 1e6 * count as f64, kb * count as f64));
    };
    probe("attn proj (384x384) f32", d, d, DType::F32, 32); // 4 layers x 8 (self+cross qkvo)
    probe("mlp fc1 (384x1536) f32", d, d * 4, DType::F32, 4);
    probe("mlp fc2 (1536x384) f32", d * 4, d, DType::F32, 4);
    probe("vocab proj (384x51864) f16", d, 51864, DType::F16, 1);
    // per-op floor: the cheapest possible candle op on a 1x384 tensor
    let t = Tensor::zeros((1, d), DType::F32, &dev)?;
    let add = best(200, || (&t + &t).unwrap());
    let ln_w = Tensor::zeros(d, DType::F32, &dev)?;
    let ln_b = Tensor::zeros(d, DType::F32, &dev)?;
    let ln = candle_nn::LayerNorm::new(ln_w, ln_b, 1e-5);
    let lns = best(200, || {
        <candle_nn::LayerNorm as candle_nn::Module>::forward(&ln, &t).unwrap()
    });
    println!(
        "\nper-op floor on a 1x384 tensor:  add {:.1} us · layer_norm {:.1} us",
        add * 1e6,
        lns * 1e6
    );
    let matmul_us: f64 = rows.iter().map(|r| r.1).sum();
    let bytes_kb: f64 = rows.iter().map(|r| r.2).sum();
    println!(
        "\nSUM of matmuls per token: {:.0} us, streaming {:.1} MB -> {:.1} GB/s effective",
        matmul_us,
        bytes_kb / 1e3,
        bytes_kb * 1e3 / (matmul_us * 1e-6) / 1e9
    );
    println!(
        "measured in-context: 6810 us/token  ->  overhead outside matmuls = {:.0} us ({:.0}%)",
        6810.0 - matmul_us,
        (6810.0 - matmul_us) / 6810.0 * 100.0
    );
    Ok(())
}
