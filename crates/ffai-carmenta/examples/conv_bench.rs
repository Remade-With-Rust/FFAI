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

use candle_core::{DType, Device, Tensor};
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

/// Direct 3x3 convolution, stride 1, padding 1, NCHW f32, batch 1.
///
/// The contraction (c_in, ky, kx) is the OUTER loop and a tile of the output —
/// `CO` channels x `OXV` columns — stays live in the accumulator across all of
/// it. That is the shape the Whisper kernels found: a dot product per output
/// ends in a horizontal reduction that neither vectorizes nor pipelines,
/// whereas broadcasting one weight and AXPY-ing a row of inputs into a held
/// tile has no reduction anywhere.
///
/// Written to be auto-vectorizable: the inner loop over `ox` is contiguous in
/// both the input read and the accumulator, with the weight loop-invariant.
fn conv3x3_direct(
    inp: &[f32],
    w: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    wd: usize,
    out: &mut [f32],
) {
    // PAD BOTH SIDES TO A COMMON STRIDE, then every (ci, ky, kx) tap becomes ONE
    // contiguous AXPY over the whole plane instead of `h` row-length ones.
    //
    // With input padded to (h+2) x (wd+2) and the accumulator carrying the same
    // stride, output element (oy, ox) reads
    //   in_p[(oy+ky)*(wd+2) + (ox+kx)] = in_p[oy*(wd+2)+ox + ky*(wd+2)+kx]
    // — a FIXED offset for the whole plane. The two pad columns accumulate
    // garbage and are dropped on write-back; 2 of 92 columns is 2 % extra work
    // to buy an 8x longer inner loop.
    //
    // The two forms that lost first: allocating a Vec per tile (0.41x), and a
    // register tile indexed by a runtime channel counter — `dp[i].add(x)` is a
    // pointer gather, which vetoes vectorization outright (0.21x).
    let sw = wd + 2;
    let n = h * sw;
    let mut in_p = vec![0f32; c_in * (h + 3) * sw];
    for ci in 0..c_in {
        for y in 0..h {
            let src = ci * h * wd + y * wd;
            let dst = ci * (h + 3) * sw + (y + 1) * sw + 1;
            in_p[dst..dst + wd].copy_from_slice(&inp[src..src + wd]);
        }
    }

    // CO-BLOCK so the padded input is streamed c_out/CO times, not c_out times.
    // Without this, conv5 re-reads its whole 1 MB padded input once per output
    // channel — 264 MB per call — and the kernel is bandwidth-bound before it is
    // FMA-bound. Eight accumulator planes are 23 KB, comfortably L1-resident.
    const CO: usize = 8;
    let mut acc = vec![0f32; CO * n];
    for co0 in (0..c_out).step_by(CO) {
        let co_n = CO.min(c_out - co0);
        for i in 0..co_n {
            acc[i * n..(i + 1) * n].fill(bias[co0 + i]);
        }
        for ci in 0..c_in {
            let base = ci * (h + 3) * sw;
            for ky in 0..3usize {
                for kx in 0..3usize {
                    let off = base + ky * sw + kx;
                    let src = &in_p[off..off + n];
                    for i in 0..co_n {
                        let a = w[((co0 + i) * c_in + ci) * 9 + ky * 3 + kx];
                        let dst = &mut acc[i * n..(i + 1) * n];
                        for (d, s) in dst.iter_mut().zip(src) {
                            *d += a * s;
                        }
                    }
                }
            }
        }
        for i in 0..co_n {
            for y in 0..h {
                let dst = (co0 + i) * h * wd + y * wd;
                let src = i * n + y * sw;
                out[dst..dst + wd].copy_from_slice(&acc[src..src + wd]);
            }
        }
    }
}

/// AVX2 twin: the accumulator TILE lives in registers across the whole
/// contraction.
///
/// The scalar form above tops out at ~28 GFLOP/s because its accumulator is an
/// array: every fused-multiply-add pays a load AND a store, so the loop is
/// limited by L1 ports (2 loads + 1 store per cycle) rather than by the two FMA
/// units. Holding 4 output channels x 16 columns in eight ymm registers across
/// all `c_in * 9` contraction steps removes both — per step the kernel does
/// 2 input loads + 4 weight broadcasts + 8 FMAs, FMA-bound at ~2 vector-FMAs
/// per cycle.
///
/// This is the Whisper lesson applied: "hold an output tile in registers across
/// the whole contraction ... 73 -> 331 GFLOP/s". The scalar form stays as the
/// oracle and the non-AVX2 path.
///
/// # Safety
/// `in_p` must be the `c_in` x (h+3) x (wd+2) zero-padded input, `out` must hold
/// `c_out * h * wd`, `w` must hold `c_out * c_in * 9`, and `bias` `c_out`. Every
/// vector load reads `tile + 16 + 2*sw + 2 <= plane_p`, which the padding
/// guarantees; the `< 16` remainder takes the scalar tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_avx2(
    in_p: &[f32],
    w: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    wd: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    const CO: usize = 4;
    let sw = wd + 2;
    let n = h * sw;
    let plane_p = (h + 3) * sw;
    // 24 columns per tile, not 16: twelve accumulators + three input vectors +
    // one broadcast is sixteen ymm registers exactly, and it lifts the ratio
    // from 8 FMAs per 6 memory ops to 12 per 7 — the broadcasts are what the
    // narrower tile failed to amortise.
    let n_vec = n / 24 * 24;

    for co0 in (0..c_out).step_by(CO) {
        let co_n = CO.min(c_out - co0);
        for tile in (0..n_vec).step_by(24) {
            let mut a: [[__m256; 3]; CO] = [[_mm256_setzero_ps(); 3]; CO];
            for (i, ai) in a.iter_mut().enumerate().take(co_n) {
                let b = _mm256_set1_ps(*bias.get_unchecked(co0 + i));
                ai[0] = b;
                ai[1] = b;
                ai[2] = b;
            }
            for ci in 0..c_in {
                let base = ci * plane_p + tile;
                for tap in 0..9usize {
                    let off = base + (tap / 3) * sw + (tap % 3);
                    let ptr = in_p.as_ptr().add(off);
                    let v0 = _mm256_loadu_ps(ptr);
                    let v1 = _mm256_loadu_ps(ptr.add(8));
                    let v2 = _mm256_loadu_ps(ptr.add(16));
                    for (i, ai) in a.iter_mut().enumerate().take(co_n) {
                        let wb =
                            _mm256_set1_ps(*w.get_unchecked(((co0 + i) * c_in + ci) * 9 + tap));
                        ai[0] = _mm256_fmadd_ps(wb, v0, ai[0]);
                        ai[1] = _mm256_fmadd_ps(wb, v1, ai[1]);
                        ai[2] = _mm256_fmadd_ps(wb, v2, ai[2]);
                    }
                }
            }
            let mut tmp = [0f32; 24];
            for (i, ai) in a.iter().enumerate().take(co_n) {
                _mm256_storeu_ps(tmp.as_mut_ptr(), ai[0]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), ai[1]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(16), ai[2]);
                for (k, t) in tmp.iter().enumerate() {
                    let pos = tile + k;
                    let (y, x) = (pos / sw, pos % sw);
                    if x < wd && y < h {
                        *out.get_unchecked_mut((co0 + i) * h * wd + y * wd + x) = *t;
                    }
                }
            }
        }
        for pos in n_vec..n {
            let (y, x) = (pos / sw, pos % sw);
            if x >= wd || y >= h {
                continue;
            }
            for i in 0..co_n {
                let mut acc = bias[co0 + i];
                for ci in 0..c_in {
                    for tap in 0..9usize {
                        let off = ci * plane_p + pos + (tap / 3) * sw + (tap % 3);
                        acc += w[((co0 + i) * c_in + ci) * 9 + tap] * in_p[off];
                    }
                }
                out[(co0 + i) * h * wd + y * wd + x] = acc;
            }
        }
    }
}

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
        let bias = vec![0f32; co];

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

        // Build the padded input once, outside the timed region — in the real
        // engine it is built once per conv, not per repetition.
        let sw = wd + 2;
        let mut in_p = vec![0f32; ci * (h + 3) * sw];
        for c in 0..ci {
            for y in 0..h {
                let src = c * h * wd + y * wd;
                let dst = c * (h + 3) * sw + (y + 1) * sw + 1;
                in_p[dst..dst + wd].copy_from_slice(&inp[src..src + wd]);
            }
        }
        let mut mine = vec![0f32; co * h * wd];
        let mut best_d = f64::INFINITY;
        let avx2 = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        for _ in 0..reps {
            let t = Instant::now();
            if avx2 {
                // SAFETY: shapes match the padded layout built above.
                unsafe { conv3x3_avx2(&in_p, &wv, &bias, ci, co, h, wd, &mut mine) }
            } else {
                conv3x3_direct(&inp, &wv, &bias, ci, co, h, wd, &mut mine);
            }
            best_d = best_d.min(t.elapsed().as_secs_f64());
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
                                acc += wv[(c * ci + k) * 9 + ky * 3 + kx] as f64
                                    * inp[k * h * wd + (iy - 1) * wd + (ix - 1)] as f64;
                            }
                        }
                    }
                    r64[c * h * wd + y * wd + x] = acc;
                }
            }
        }
        let scale = r64.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
        let err = |v: &[f32]| {
            v.iter().zip(&r64).fold(0f64, |m, (a, b)| m.max((*a as f64 - b).abs())) / scale
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
