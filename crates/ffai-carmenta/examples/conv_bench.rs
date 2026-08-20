//! Microbenchmark: candle's tiled im2col conv2d vs a direct, non-materialising
//! convolution, on the EXACT shapes the CRNN backbone runs (§8.100 D4).
//!
//! §8.100 measured the backbone at 29.7 GFLOP/s against ~112 GFLOP/s of
//! single-core AVX2 peak. The mechanism is candle materialising a 9x-expanded
//! im2col buffer per tile — 28.5 MB written per 362 px line against 2.28 GFLOP
//! of arithmetic — with a scalar strided gather, and only 2-3 tiles of internal
//! parallelism on these shapes.
//!
//! A direct convolution never builds that buffer: it reads each input element
//! from its natural position and accumulates into an output tile held in
//! registers. Same FLOPs, ~1/9 the memory traffic.
//!
//! Step 0 of `codec-vectorize-kernel` says prove the RESTRUCTURE before writing
//! intrinsics, so this measures the scalar direct form first. If it already
//! wins, the SIMD twin is a separate brick with its own gate; if it loses, the
//! im2col diagnosis was wrong and no kernel should be written.
//!
//! Single-threaded on purpose (`RAYON_NUM_THREADS=1`): §8.100 measured a
//! 17-27 % noise floor on whole-page wall clock, which cannot resolve a kernel.
//! Best-of-N on a pinned shape can.
//!
//! Usage: RAYON_NUM_THREADS=1 cargo run -p ffai-carmenta --release --example conv_bench

use candle_core::{Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use std::time::Instant;

/// The five 3x3 backbone shapes, as measured on a 362 px line:
/// `(name, c_in, c_out, h, w)`. Stride 1, padding 1 throughout.
const SHAPES: [(&str, usize, usize, usize, usize); 5] = [
    ("conv1 32->64", 32, 64, 32, 181),
    ("conv2 64->128", 64, 128, 16, 90),
    ("conv3 128->128", 128, 128, 16, 90),
    ("conv4 128->256", 128, 256, 8, 90),
    ("conv5 256->256", 256, 256, 8, 90),
];

fn reps_done(_: &mut usize) -> bool { true }

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    println!("  conv2d: candle tiled im2col vs direct (scalar), batch 1, 3x3 s1 p1");
    println!("  single-threaded; best of N\n");
    println!(
        "  {:16} {:>8} {:>11} {:>11} {:>9} {:>10} {:>9}",
        "shape", "MFLOP", "candle ms", "ours ms", "speedup", "candle err", "our err"
    );

    let mut tot_c = 0f64;
    let mut tot_d = 0f64;
    for (name, ci, co, h, wd) in SHAPES {
        let n_in = ci * h * wd;
        let n_w = co * ci * 9;
        let inp: Vec<f32> = (0..n_in).map(|i| ((i * 37 % 251) as f32 / 251.0) - 0.5).collect();
        let wv: Vec<f32> = (0..n_w).map(|i| ((i * 53 % 197) as f32 / 197.0) - 0.5).collect();
        let _bias = vec![0f32; co];

        let t_in = Tensor::from_vec(inp.clone(), (1, ci, h, wd), &dev)?;
        let t_w = Tensor::from_vec(wv.clone(), (co, ci, 3, 3), &dev)?;

        let reps = 5;
        let mut best_c = f64::INFINITY;
        let mut candle_out = vec![];
        for _ in 0..reps {
            let t = Instant::now();
            let o = t_in.conv2d(&t_w, 1, 1, 1, 1)?;
            let e = t.elapsed().as_secs_f64();
            if e < best_c {
                best_c = e;
                candle_out = o.flatten_all()?.to_vec1::<f32>()?;
            }
        }

        // Call the SHIPPED kernel through its real entry point, so the bench
        // measures what ships — including the CustomOp2 plumbing — rather than
        // a private copy that can drift from it.
        let conv = Conv2d::new(t_w.clone(), None, Conv2dConfig { padding: 1, ..Default::default() });
        let mine;
        let mut best_d = f64::INFINITY;
        // Runs once by construction; kept as a loop so the early-exit path
        // below stays visible next to the rep loop that follows.
        #[allow(clippy::never_loop)]
        loop {
            let t = Instant::now();
            let o = ffai_carmenta::conv3x3::apply(&t_in, &conv)?;
            let e = t.elapsed().as_secs_f64();
            mine = o.flatten_all()?.to_vec1::<f32>()?;
            best_d = best_d.min(e);
            if best_d < f64::INFINITY && reps_done(&mut 0) {
                break;
            }
            break;
        }
        for _ in 1..reps {
            let t = Instant::now();
            let o = ffai_carmenta::conv3x3::apply(&t_in, &conv)?;
            best_d = best_d.min(t.elapsed().as_secs_f64());
            drop(o);
        }

        // Correctness is judged against an f64 reference, not against candle:
        // both are f32 reassociations of the same 2304-term contraction, so
        // "differs from candle" says nothing about which is right. The metric
        // is max |err| normalised by the tensor's own scale, which is what a
        // ReLU downstream actually cares about.
        let mut r64 = vec![0f64; co * h * wd];
        for c in 0..co {
            for y in 0..h {
                for x in 0..wd {
                    let mut acc = 0f64;
                    for k in 0..ci {
                        for ky in 0..3usize {
                            for kx in 0..3usize {
                                let (iy, ix) = (y + ky, x + kx);
                                if iy == 0 || iy > h || ix == 0 || ix > wd {
                                    continue;
                                }
                                acc += f64::from(wv[(c * ci + k) * 9 + ky * 3 + kx])
                                    * f64::from(inp[k * h * wd + (iy - 1) * wd + (ix - 1)]);
                            }
                        }
                    }
                    r64[c * h * wd + y * wd + x] = acc;
                }
            }
        }
        let scale = r64.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
        let err = |v: &[f32]| {
            v.iter().zip(&r64).fold(0f64, |m, (a, b)| m.max((f64::from(*a) - b).abs())) / scale
        };
        let (e_candle, e_mine) = (err(&candle_out), err(&mine));
        let max_rel = e_mine as f32;
        let _ = e_candle;
        let mflop = 2.0 * (ci * co * 9 * h * wd) as f64 / 1e6;
        tot_c += best_c;
        tot_d += best_d;
        println!(
            "  {name:16} {mflop:8.1} {:11.2} {:11.2} {:8.2}x {:9.1e} {max_rel:9.1e}",
            best_c * 1e3,
            best_d * 1e3,
            best_c / best_d,
            e_candle
        );
    }
    println!(
        "\n  TOTAL             candle {:.1} ms   direct {:.1} ms   {:.2}x",
        tot_c * 1e3,
        tot_d * 1e3,
        tot_c / tot_d
    );
    Ok(())
}
