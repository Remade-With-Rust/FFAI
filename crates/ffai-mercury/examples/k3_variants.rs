//! How much of the FFN's remaining cost is data movement rather than GEMM?
//!
//! `ffn_ceiling` established that candle's conv1d wrapper costs 1.71x and that
//! im2col+matmul recovers most of it — but also that the pure-GEMM ceiling is
//! 96.3 ms against im2col+matmul's 147.0. That 51 ms residue is OUR plumbing,
//! and this prices the pieces of it.
//!
//! The shipped `conv1d_k3_gemm` makes three copies before the GEMM even starts:
//!   x Tensor -> Vec (to_vec1), Vec -> im2col buffer, buffer -> Tensor.
//! And the FFN calls it twice per layer with a candle `relu` and a Tensor
//! round-trip in between. Variants measured:
//!
//!   1 shipped            — per-conv Tensor round trip, as it ships
//!   2 reused buffers     — same, but col/out buffers allocated once
//!   3 flat FFN           — ONE Vec through both convs + relu, weights cached
//!                          as Tensors, no intermediate Tensor materialisation
//!   C ceiling            — the two GEMMs alone, operands preallocated
//!
//! ```text
//! cargo run --release -p ffai-mercury --example k3_variants
//! ```

use std::time::Instant;

use ffai_core::candle::{Device, Tensor};

const C_IN: usize = 192;
const C_HID: usize = 768;
const K: usize = 3;
const LAYERS: usize = 6;

fn bench(label: &str, gflop: f64, mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "  {label:<36} {:>8.1} ms  {:>7.1} GFLOP/s",
        best * 1000.0,
        gflop / best
    );
    best
}

/// im2col k=3 pad=1 into a caller-provided buffer, channel-major.
fn im2col_into(x: &[f32], c: usize, t: usize, col: &mut [f32]) {
    for ci in 0..c {
        let src = &x[ci * t..(ci + 1) * t];
        col[(ci * K) * t + 1..(ci * K) * t + t].copy_from_slice(&src[..t - 1]);
        col[(ci * K) * t] = 0.0;
        col[(ci * K + 1) * t..(ci * K + 1) * t + t].copy_from_slice(src);
        col[(ci * K + 2) * t..(ci * K + 2) * t + t - 1].copy_from_slice(&src[1..]);
        col[(ci * K + 2) * t + t - 1] = 0.0;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let lens: Vec<usize> = (0..20).map(|i| 70 + (i * 7) % 40).collect();
    let total_t: usize = lens.iter().sum();
    let gflop = LAYERS as f64 * total_t as f64 * (C_IN * C_HID * K * 2 * 2) as f64 / 1e9;
    println!(
        "{} sentences, {total_t} columns; FFN {gflop:.2} GFLOP\n",
        lens.len()
    );

    let w_up = Tensor::randn(0f32, 0.05f32, (C_HID, C_IN * K), &dev)?;
    let w_dn = Tensor::randn(0f32, 0.05f32, (C_IN, C_HID * K), &dev)?;
    let b_up = Tensor::randn(0f32, 0.05f32, (C_HID, 1), &dev)?;
    let b_dn = Tensor::randn(0f32, 0.05f32, (C_IN, 1), &dev)?;
    let xs: Vec<Tensor> = lens
        .iter()
        .map(|&t| Tensor::randn(0f32, 1f32, (1, C_IN, t), &dev).unwrap())
        .collect();

    // ---- 1. shipped: Tensor round trip per conv ----
    bench("1 shipped (Tensor round trip)", gflop, || {
        for (&t, x0) in lens.iter().zip(&xs) {
            let mut h = x0.clone();
            for _ in 0..LAYERS {
                for (w, b, cin) in [(&w_up, &b_up, C_IN), (&w_dn, &b_dn, C_HID)] {
                    let hv: Vec<f32> = h.flatten_all().unwrap().to_vec1().unwrap();
                    let mut col = vec![0f32; cin * K * t];
                    im2col_into(&hv, cin, t, &mut col);
                    let cm = Tensor::from_vec(col, (cin * K, t), &dev).unwrap();
                    let y = w.matmul(&cm).unwrap().broadcast_add(b).unwrap();
                    h = y.reshape((1, y.dim(0).unwrap(), t)).unwrap();
                }
                h = h.relu().unwrap();
            }
            std::hint::black_box(&h);
        }
    });

    // ---- 2. reused buffers ----
    bench("2 reused col buffers", gflop, || {
        let mut colu = vec![0f32; C_IN * K * 200];
        let mut cold = vec![0f32; C_HID * K * 200];
        for (&t, x0) in lens.iter().zip(&xs) {
            let mut h = x0.clone();
            for _ in 0..LAYERS {
                for (w, b, cin, buf) in [
                    (&w_up, &b_up, C_IN, &mut colu),
                    (&w_dn, &b_dn, C_HID, &mut cold),
                ] {
                    let hv: Vec<f32> = h.flatten_all().unwrap().to_vec1().unwrap();
                    let col = &mut buf[..cin * K * t];
                    im2col_into(&hv, cin, t, col);
                    let cm = Tensor::from_slice(col, (cin * K, t), &dev).unwrap();
                    let y = w.matmul(&cm).unwrap().broadcast_add(b).unwrap();
                    h = y.reshape((1, y.dim(0).unwrap(), t)).unwrap();
                }
                h = h.relu().unwrap();
            }
            std::hint::black_box(&h);
        }
    });

    // ---- 3. FLAT FFN: one Vec through both convs, relu in place ----
    // The intermediate never becomes a Tensor: the GEMM output is pulled
    // straight back to a Vec, biased and relu'd in place, and fed to the next
    // im2col. Saves one Tensor materialisation, one candle relu op and one
    // to_vec1 per layer.
    let bu: Vec<f32> = b_up.flatten_all()?.to_vec1()?;
    let bd: Vec<f32> = b_dn.flatten_all()?.to_vec1()?;
    bench("3 flat FFN (no intermediate Tensor)", gflop, || {
        let mut colu = vec![0f32; C_IN * K * 200];
        let mut cold = vec![0f32; C_HID * K * 200];
        for (&t, x0) in lens.iter().zip(&xs) {
            let mut hv: Vec<f32> = x0.flatten_all().unwrap().to_vec1().unwrap();
            for _ in 0..LAYERS {
                // up: 192 -> 768, then relu
                let col = &mut colu[..C_IN * K * t];
                im2col_into(&hv, C_IN, t, col);
                let cm = Tensor::from_slice(col, (C_IN * K, t), &dev).unwrap();
                let mut yv: Vec<f32> = w_up
                    .matmul(&cm)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                for c in 0..C_HID {
                    let row = &mut yv[c * t..(c + 1) * t];
                    let bias = bu[c];
                    for v in row.iter_mut() {
                        *v = (*v + bias).max(0.0);
                    }
                }
                // down: 768 -> 192
                let col = &mut cold[..C_HID * K * t];
                im2col_into(&yv, C_HID, t, col);
                let cm = Tensor::from_slice(col, (C_HID * K, t), &dev).unwrap();
                hv = w_dn
                    .matmul(&cm)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                for c in 0..C_IN {
                    let row = &mut hv[c * t..(c + 1) * t];
                    let bias = bd[c];
                    for v in row.iter_mut() {
                        *v += bias;
                    }
                }
            }
            std::hint::black_box(&hv);
        }
    });

    // ---- 4. ORIENTATION: candle's shape vs ours, GEMM only ----
    // Candle computes col[l_out, K] x kernel[K, c_out]; we compute
    // w[c_out, K] x col[K, T]. Same arithmetic, transposed. The six-whys skill
    // records a case where orientation alone was worth 10x, so this is measured
    // rather than assumed -- and it is measured on the GEMM ALONE, with both
    // operands preallocated, so nothing else can contaminate it.
    let mut ours: Vec<(Tensor, Tensor)> = Vec::new();
    let mut theirs: Vec<(Tensor, Tensor)> = Vec::new();
    for &t in &lens {
        for _ in 0..LAYERS {
            // ours: [c_out, K] x [K, T] -> [c_out, T]
            ours.push((
                w_up.clone(),
                Tensor::randn(0f32, 1f32, (C_IN * K, t), &dev).unwrap(),
            ));
            ours.push((
                w_dn.clone(),
                Tensor::randn(0f32, 1f32, (C_HID * K, t), &dev).unwrap(),
            ));
            // theirs: [T, K] x [K, c_out] -> [T, c_out]
            theirs.push((
                Tensor::randn(0f32, 1f32, (t, C_IN * K), &dev).unwrap(),
                w_up.t().unwrap().contiguous().unwrap(),
            ));
            theirs.push((
                Tensor::randn(0f32, 1f32, (t, C_HID * K), &dev).unwrap(),
                w_dn.t().unwrap().contiguous().unwrap(),
            ));
        }
    }
    let o_ours = bench("4a orientation OURS  [Co,K]x[K,T]", gflop, || {
        for (a, b) in &ours {
            std::hint::black_box(a.matmul(b).unwrap());
        }
    });
    let o_theirs = bench("4b orientation CANDLE [T,K]x[K,Co]", gflop, || {
        for (a, b) in &theirs {
            std::hint::black_box(a.matmul(b).unwrap());
        }
    });
    println!(
        "     -> ours is {:.2}x candle's orientation
",
        o_theirs / o_ours
    );

    // ---- C. ceiling ----
    let mut ops: Vec<(Tensor, Tensor)> = Vec::new();
    for &t in &lens {
        for _ in 0..LAYERS {
            ops.push((
                w_up.clone(),
                Tensor::randn(0f32, 1f32, (C_IN * K, t), &dev).unwrap(),
            ));
            ops.push((
                w_dn.clone(),
                Tensor::randn(0f32, 1f32, (C_HID * K, t), &dev).unwrap(),
            ));
        }
    }
    bench("C pure-GEMM ceiling", gflop, || {
        for (w, x) in &ops {
            std::hint::black_box(w.matmul(x).unwrap());
        }
    });
    Ok(())
}
