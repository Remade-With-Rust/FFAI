//! Assert the candle SVTR port against the paddle oracle fixture (§8.168).
//!
//! §8.167's numpy reference matched at 1.3e-5 and proved the STRUCTURE; this
//! proves the Rust reproduces it. The bar is 1e-4: `concat([h, h])` scored
//! 100 % argmax agreement while structurally wrong, so agreement alone is not
//! evidence — only the tensor distance is.
use std::path::PathBuf;

use candle_core::{Device, Tensor};

fn read_f32(p: &PathBuf) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fix = PathBuf::from("corpora/refs/fixtures");
    // LOCALAPPDATA is where `carmenta_svtr_prepare.py` writes; no new dep.
    let weights = PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_default())
        .join("ffai/models/ppocrv5-mobile-rec/rec.safetensors");
    let dev = Device::Cpu;
    let m = ffai_carmenta::svtr::load(&weights, &dev, 18385)?;

    let x = read_f32(&fix.join("svtr_input_1x3x48x320_f32.bin"));
    let want = read_f32(&fix.join("svtr_logits_f32.bin"));
    let xt = Tensor::from_vec(x, (1, 3, 48, 320), &dev)?;
    let got = m.forward(&xt)?;
    println!("  output {:?}  (want {} values)", got.dims(), want.len());
    let g = got.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(g.len(), want.len(), "shape mismatch against the oracle");
    let (mut mx, mut sum) = (0f32, 0f64);
    for (a, b) in g.iter().zip(want.iter()) {
        let d = (a - b).abs();
        mx = mx.max(d);
        sum += d as f64;
    }
    let n_t = got.dim(1)?;
    let n_c = got.dim(2)?;
    let mut agree = 0usize;
    for t in 0..n_t {
        let am = |v: &[f32]| (0..n_c).max_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap()).unwrap();
        if am(&g[t * n_c..(t + 1) * n_c]) == am(&want[t * n_c..(t + 1) * n_c]) {
            agree += 1;
        }
    }
    println!("  max abs diff {mx:.3e}   mean {:.3e}", sum / g.len() as f64);
    println!("  argmax agreement {:.1} %", 100.0 * agree as f64 / n_t as f64);
    if mx < 1e-4 {
        println!("  MATCH — candle port reproduces the paddle oracle");
        Ok(())
    } else {
        Err(format!("port does not match the oracle: {mx:.3e} >= 1e-4").into())
    }
}
