//! Where is the text encoder's 2.9x deficit — the wrapper, or the shape?
//!
//! ORT runs the WHOLE text encoder in ~139 ms. Our feed-forward half alone
//! costs ~260 ms at 72 GFLOP/s. Last round I called 72 GFLOP/s "healthy" and
//! moved on; in context it is twice the reference's entire stage, so that
//! judgement was made against the wrong baseline.
//!
//! The FFN is `conv1d(k=3, pad=1)` 192->768, relu, then 768->192, six layers,
//! T~88. As arithmetic that is a GEMM. Three ways to express the SAME
//! arithmetic, so the measurement says which part costs us:
//!
//!   1. candle conv1d                   — what ships today
//!   2. im2col + candle matmul          — same GEMM, conv wrapper removed
//!   3. 3 shifted 1x1 GEMMs accumulated — no im2col buffer materialised
//!
//! plus a ceiling (pure GEMM, no data movement) and a batched shape probe.
//!
//! EVERY arm runs 6 layers x (up conv + relu + down conv) per sentence — the
//! first cut of this probe alternated up/down BY LAYER, so it did half the
//! convolutions and reported a 3.35x that was really ~1.7x. Arms that do
//! different amounts of work do not measure the same question.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example ffn_ceiling
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
    println!("  {label:<40} {:>8.1} ms  {:>7.1} GFLOP/s", best * 1000.0, gflop / best);
    best
}

/// im2col for conv1d k=3 pad=1, channel-major [C][T] -> [C*K][T].
fn im2col(x: &[f32], c: usize, t: usize, out: &mut [f32]) {
    for ci in 0..c {
        let src = &x[ci * t..(ci + 1) * t];
        for j in 0..K {
            let row = &mut out[(ci * K + j) * t..(ci * K + j + 1) * t];
            // tap j reads x[i + j - 1]; zero outside (pad = 1)
            let lo = if j == 0 { 1 } else { 0 };
            let hi = if j == 2 { t - 1 } else { t };
            row[..lo].fill(0.0);
            for i in lo..hi {
                row[i] = src[(i + j - 1).min(t - 1)];
            }
            row[hi..].fill(0.0);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let lens: Vec<usize> = (0..20).map(|i| 70 + (i * 7) % 40).collect();
    let total_t: usize = lens.iter().sum();
    println!("{} sentences, {total_t} columns (mean {:.0})", lens.len(), total_t as f64 / 20.0);

    // per column per layer: up 192*768*3*2 + down 768*192*3*2
    let per_col_layer = (C_IN * C_HID * K * 2 * 2) as f64;
    let gflop = LAYERS as f64 * total_t as f64 * per_col_layer / 1e9;
    println!("FFN total (6 layers x up+down): {gflop:.2} GFLOP\n");

    let w_up = Tensor::randn(0f32, 0.05f32, (C_HID, C_IN, K), &dev)?;
    let w_dn = Tensor::randn(0f32, 0.05f32, (C_IN, C_HID, K), &dev)?;
    let w_up2 = w_up.reshape((C_HID, C_IN * K))?;
    let w_dn2 = w_dn.reshape((C_IN, C_HID * K))?;
    let xs: Vec<Tensor> = lens
        .iter()
        .map(|&t| Tensor::randn(0f32, 1f32, (1, C_IN, t), &dev).unwrap())
        .collect();
    let xv: Vec<Vec<f32>> =
        xs.iter().map(|x| x.flatten_all().unwrap().to_vec1().unwrap()).collect();

    // ---- 1. what ships ----
    let t1 = bench("1 candle conv1d (shipped)", gflop, || {
        for x in &xs {
            let mut h = x.clone();
            for _ in 0..LAYERS {
                let a = h.conv1d(&w_up, 1, 1, 1, 1).unwrap().relu().unwrap();
                h = a.conv1d(&w_dn, 1, 1, 1, 1).unwrap();
            }
            std::hint::black_box(&h);
        }
    });

    // ---- 2. im2col + matmul, identical work ----
    let t2 = bench("2 im2col + matmul", gflop, || {
        for (&t, x0) in lens.iter().zip(&xv) {
            let mut hv = x0.clone();
            let mut colu = vec![0f32; C_IN * K * t];
            let mut cold = vec![0f32; C_HID * K * t];
            for _ in 0..LAYERS {
                im2col(&hv, C_IN, t, &mut colu);
                let cm = Tensor::from_vec(colu.clone(), (C_IN * K, t), &dev).unwrap();
                let y = w_up2.matmul(&cm).unwrap();
                let mut yv: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
                yv.iter_mut().for_each(|v| *v = v.max(0.0));

                im2col(&yv, C_HID, t, &mut cold);
                let cm = Tensor::from_vec(cold.clone(), (C_HID * K, t), &dev).unwrap();
                let y = w_dn2.matmul(&cm).unwrap();
                hv = y.flatten_all().unwrap().to_vec1().unwrap();
            }
            std::hint::black_box(&hv);
        }
    });

    // ---- 3. three shifted 1x1 GEMMs, no im2col buffer ----
    // conv1d(k=3) == sum over taps of a 1x1 GEMM on a shifted view. Avoids
    // materialising a 3x-larger input matrix; costs 3 GEMM calls instead of 1.
    let wu_taps: Vec<Tensor> = (0..K)
        .map(|j| w_up.narrow(2, j, 1).unwrap().squeeze(2).unwrap().contiguous().unwrap())
        .collect();
    let wd_taps: Vec<Tensor> = (0..K)
        .map(|j| w_dn.narrow(2, j, 1).unwrap().squeeze(2).unwrap().contiguous().unwrap())
        .collect();
    let t3 = bench("3 three shifted 1x1 GEMMs", gflop, || {
        for (&t, x0) in lens.iter().zip(&xv) {
            let mut hv = x0.clone();
            for _ in 0..LAYERS {
                let mut acc = vec![0f32; C_HID * t];
                for (j, w) in wu_taps.iter().enumerate() {
                    // shifted view: output i reads input i+j-1
                    let (src_lo, dst_lo, n) = match j {
                        0 => (0usize, 1usize, t - 1),
                        1 => (0, 0, t),
                        _ => (1, 0, t - 1),
                    };
                    let mut sh = vec![0f32; C_IN * n];
                    for c in 0..C_IN {
                        sh[c * n..(c + 1) * n]
                            .copy_from_slice(&hv[c * t + src_lo..c * t + src_lo + n]);
                    }
                    let m = Tensor::from_vec(sh, (C_IN, n), &dev).unwrap();
                    let y: Vec<f32> =
                        w.matmul(&m).unwrap().flatten_all().unwrap().to_vec1().unwrap();
                    for c in 0..C_HID {
                        for i in 0..n {
                            acc[c * t + dst_lo + i] += y[c * n + i];
                        }
                    }
                }
                acc.iter_mut().for_each(|v| *v = v.max(0.0));
                let mut acc2 = vec![0f32; C_IN * t];
                for (j, w) in wd_taps.iter().enumerate() {
                    let (src_lo, dst_lo, n) = match j {
                        0 => (0usize, 1usize, t - 1),
                        1 => (0, 0, t),
                        _ => (1, 0, t - 1),
                    };
                    let mut sh = vec![0f32; C_HID * n];
                    for c in 0..C_HID {
                        sh[c * n..(c + 1) * n]
                            .copy_from_slice(&acc[c * t + src_lo..c * t + src_lo + n]);
                    }
                    let m = Tensor::from_vec(sh, (C_HID, n), &dev).unwrap();
                    let y: Vec<f32> =
                        w.matmul(&m).unwrap().flatten_all().unwrap().to_vec1().unwrap();
                    for c in 0..C_IN {
                        for i in 0..n {
                            acc2[c * t + dst_lo + i] += y[c * n + i];
                        }
                    }
                }
                hv = acc2;
            }
            std::hint::black_box(&hv);
        }
    });

    // ---- ceiling: the GEMMs alone, correct count (6 layers x up+down) ----
    let mut ops: Vec<(Tensor, Tensor)> = Vec::new();
    for &t in &lens {
        for _ in 0..LAYERS {
            ops.push((w_up2.clone(), Tensor::randn(0f32, 1f32, (C_IN * K, t), &dev).unwrap()));
            ops.push((w_dn2.clone(), Tensor::randn(0f32, 1f32, (C_HID * K, t), &dev).unwrap()));
        }
    }
    let tc = bench("C pure-GEMM ceiling (no movement)", gflop, || {
        for (w, x) in &ops {
            std::hint::black_box(w.matmul(x).unwrap());
        }
    });

    // ---- shape probe: same FLOPs, all columns in ONE GEMM per conv ----
    let big_u = Tensor::randn(0f32, 1f32, (C_IN * K, total_t), &dev)?;
    let big_d = Tensor::randn(0f32, 1f32, (C_HID * K, total_t), &dev)?;
    bench("B batched ceiling (N=all columns)", gflop, || {
        for _ in 0..LAYERS {
            std::hint::black_box(w_up2.matmul(&big_u).unwrap());
            std::hint::black_box(w_dn2.matmul(&big_d).unwrap());
        }
    });

    println!(
        "\n  conv1d wrapper tax : {:.2}x over im2col+matmul, {:.2}x over shifted-1x1",
        t1 / t2,
        t1 / t3
    );
    println!("  best expressible   : {:.1} ms vs shipped {:.1} ms", t2.min(t3), t1 * 1000.0 / 1000.0 * 1000.0 / 1000.0 * 1000.0);
    println!("  headroom to ceiling: im2col+matmul is {:.2}x the pure GEMM", t2 / tc);
    Ok(())
}
