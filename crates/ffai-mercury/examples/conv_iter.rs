//! Iteration on the 104 GFLOP/s conv. The earlier refutation assumed im2col
//! needs a STRIDED gather. For channel-major (C, L) input it does not: row
//! (c*3+k) of the im2col matrix is in[c][k .. k+L], a CONTIGUOUS slice.
use candle_nn::{Conv1d, Conv1dConfig, Module};
use ffai_core::candle::{Device, Tensor};
use std::time::Instant;
fn t(n: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut b = f64::MAX;
    for _ in 0..n {
        let s = Instant::now();
        f();
        b = b.min(s.elapsed().as_secs_f64());
    }
    b
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let (cin, cout, l) = (80usize, 384usize, 3000usize);
    let mel = Tensor::randn(0f32, 1., (1, cin, l), &d)?.contiguous()?;
    let w = Tensor::randn(0f32, 1., (cout, cin, 3), &d)?.contiguous()?;
    let c1 = Conv1d::new(
        w.clone(),
        None,
        Conv1dConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let base = t(7, || {
        std::hint::black_box(c1.forward(&mel).unwrap());
    });

    // im2col in CHANNEL-MAJOR: (cin*3, l), each row a contiguous slice.
    let src: Vec<f32> = mel.flatten_all()?.to_vec1()?;
    let build = t(7, || {
        let mut col = vec![0f32; cin * 3 * l];
        for c in 0..cin {
            for k in 0..3 {
                let dst = &mut col[(c * 3 + k) * l..(c * 3 + k + 1) * l];
                let s = &src[c * l..(c + 1) * l];
                // pad=1: tap k reads in[c][t+k-1]
                match k {
                    0 => dst[1..].copy_from_slice(&s[..l - 1]),
                    1 => dst.copy_from_slice(s),
                    _ => dst[..l - 1].copy_from_slice(&s[1..]),
                }
            }
        }
        std::hint::black_box(col);
    });

    let colt = Tensor::randn(0f32, 1., (cin * 3, l), &d)?.contiguous()?;
    let wm = w.reshape((cout, cin * 3))?.contiguous()?;
    let gemm = t(7, || {
        std::hint::black_box(wm.matmul(&colt).unwrap());
    });
    let f = 2.0 * cout as f64 * (cin * 3) as f64 * l as f64;
    println!(
        "candle conv1d          {:6.2} ms   {:5.0} GFLOP/s",
        base * 1e3,
        f / base / 1e9
    );
    println!("im2col build (contig)  {:6.2} ms", build * 1e3);
    println!(
        "GEMM (384,240)@(240,3000) {:6.2} ms {:5.0} GFLOP/s",
        gemm * 1e3,
        f / gemm / 1e9
    );
    println!(
        "  im2col + GEMM total  {:6.2} ms  -> {:.2}x vs candle conv1d",
        (build + gemm) * 1e3,
        base / (build + gemm)
    );

    // ---- conv2: 384->384, k3, STRIDE 2. im2col rows become strided reads. ----
    let (c2i, lo) = (384usize, 1500usize);
    let h = Tensor::randn(0f32, 1., (1, c2i, l), &d)?.contiguous()?;
    let w2 = Tensor::randn(0f32, 1., (c2i, c2i, 3), &d)?.contiguous()?;
    let cv2 = Conv1d::new(
        w2.clone(),
        None,
        Conv1dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        },
    );
    let b2 = t(7, || {
        std::hint::black_box(cv2.forward(&h).unwrap());
    });
    let hs: Vec<f32> = h.flatten_all()?.to_vec1()?;
    let build2 = t(7, || {
        let mut col = vec![0f32; c2i * 3 * lo];
        for c in 0..c2i {
            let s = &hs[c * l..(c + 1) * l];
            for k in 0..3 {
                let dst = &mut col[(c * 3 + k) * lo..(c * 3 + k + 1) * lo];
                for (i, o) in dst.iter_mut().enumerate() {
                    let idx = 2 * i + k; // pad=1 => t*2 + k - 1
                    *o = if idx == 0 { 0.0 } else { s[idx - 1] };
                }
            }
        }
        std::hint::black_box(col);
    });
    let col2 = Tensor::randn(0f32, 1., (c2i * 3, lo), &d)?.contiguous()?;
    let wm2 = w2.reshape((c2i, c2i * 3))?.contiguous()?;
    let g2 = t(7, || {
        std::hint::black_box(wm2.matmul(&col2).unwrap());
    });
    let f2 = 2.0 * c2i as f64 * (c2i * 3) as f64 * lo as f64;
    println!(
        "
conv2 (stride 2):"
    );
    println!(
        "  candle conv1d        {:6.2} ms   {:5.0} GFLOP/s",
        b2 * 1e3,
        f2 / b2 / 1e9
    );
    println!("  im2col build (strided reads) {:6.2} ms", build2 * 1e3);
    println!(
        "  GEMM (384,1152)@(1152,1500)  {:6.2} ms  {:5.0} GFLOP/s",
        g2 * 1e3,
        f2 / g2 / 1e9
    );
    println!(
        "  total {:6.2} ms -> {:.2}x vs candle",
        (build2 + g2) * 1e3,
        b2 / (build2 + g2)
    );
    Ok(())
}
