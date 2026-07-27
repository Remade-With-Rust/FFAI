//! Go/no-go probe for a fused encoder attention kernel.
//!
//! Encoder attention is ~67 % of the encoder and runs at ~141 GFLOP/s, against
//! the ~210 candle reaches on these K=64 shapes. Flash attention removes the
//! score matrix's memory round-trip; the question this file answers is whether
//! a hand-written kernel can bank that WITHOUT losing more to a tuned GEMM
//! than it saves.
//!
//! Three attempts are recorded in the git history of this file:
//!   1. naive dot products                      46 GFLOP/s
//!   2. contraction-outermost AXPY              73 GFLOP/s
//!   3. AVX2 register tiling + vectorized exp   (this one)
//!
//! **Stop condition, fixed in advance: it must beat the candle three-op path
//! in an interleaved paired A/B.**
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example flash_probe
//! ```

use std::time::Instant;

use ffai_core::candle::{Device, Tensor};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Query rows per block.
const BQ: usize = 64;
/// Key columns per block. 64 x 256 x 4 B = 64 KB of scores — L2 resident.
const BK: usize = 256;
/// Whisper's head dimension.
const HD: usize = 64;

// ---------------------------------------------------------------------------
// AVX2 helpers
// ---------------------------------------------------------------------------

/// Vectorized `exp`, ~1e-7 relative.
///
/// The scalar `exp` was not incidental: one 30 s window runs 13.5 M of them
/// through the softmax, at ~3 ns each, which is the same order as the whole
/// kernel's runtime. `exp(x) = 2^(x·log2e)`, split into an integer part folded
/// straight into the f32 exponent field and a fractional part evaluated by a
/// degree-5 minimax polynomial on [-0.5, 0.5].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp256(x: __m256) -> __m256 {
    // Inputs here are (score - row_max) <= 0; clamp the tail so the exponent
    // field cannot underflow into garbage.
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
    let t = _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E));
    let k = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let f = _mm256_sub_ps(t, k);

    let mut p = _mm256_set1_ps(0.001_333_355_8);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.009_618_129));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.055_504_11));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.240_226_51));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.693_147_18));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));

    // 2^k by writing k directly into the biased exponent.
    let ki = _mm256_cvtps_epi32(k);
    let pow2k = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ki, _mm256_set1_epi32(127)),
        23,
    ));
    _mm256_mul_ps(p, pow2k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hmax256(v: __m256) -> f32 {
    let m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    let m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    _mm_cvtss_f32(m)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

// ---------------------------------------------------------------------------
// S = Q_block @ K_block  —  4 query rows x 16 key columns held in registers
// ---------------------------------------------------------------------------

/// `qs` is the pre-scaled query block, (rows, HD) row-major.
/// `kt` is (HD, seq) — head-dim major, so the inner loop reads contiguously.
///
/// Eight accumulators (4 rows x 2 vectors) stay resident across the whole
/// 64-step contraction, so each pair of K loads feeds 8 FMAs instead of the
/// AXPY version's one FMA per load-modify-store of the score tile.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scores_tile(
    qs: &[f32],
    kt: &[f32],
    scores: &mut [f32],
    rows: usize,
    k0: usize,
    cols: usize,
    seq: usize,
) {
    let kbase = kt.as_ptr();
    let mut i0 = 0;
    while i0 + 4 <= rows {
        let mut j0 = 0;
        while j0 + 16 <= cols {
            let (mut a00, mut a01) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let (mut a10, mut a11) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let (mut a20, mut a21) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let (mut a30, mut a31) = (_mm256_setzero_ps(), _mm256_setzero_ps());

            for t in 0..HD {
                let kp = kbase.add(t * seq + k0 + j0);
                let kv0 = _mm256_loadu_ps(kp);
                let kv1 = _mm256_loadu_ps(kp.add(8));

                let q0 = _mm256_set1_ps(*qs.get_unchecked((i0) * HD + t));
                let q1 = _mm256_set1_ps(*qs.get_unchecked((i0 + 1) * HD + t));
                let q2 = _mm256_set1_ps(*qs.get_unchecked((i0 + 2) * HD + t));
                let q3 = _mm256_set1_ps(*qs.get_unchecked((i0 + 3) * HD + t));

                a00 = _mm256_fmadd_ps(q0, kv0, a00);
                a01 = _mm256_fmadd_ps(q0, kv1, a01);
                a10 = _mm256_fmadd_ps(q1, kv0, a10);
                a11 = _mm256_fmadd_ps(q1, kv1, a11);
                a20 = _mm256_fmadd_ps(q2, kv0, a20);
                a21 = _mm256_fmadd_ps(q2, kv1, a21);
                a30 = _mm256_fmadd_ps(q3, kv0, a30);
                a31 = _mm256_fmadd_ps(q3, kv1, a31);
            }

            let sp = scores.as_mut_ptr();
            _mm256_storeu_ps(sp.add((i0) * BK + j0), a00);
            _mm256_storeu_ps(sp.add((i0) * BK + j0 + 8), a01);
            _mm256_storeu_ps(sp.add((i0 + 1) * BK + j0), a10);
            _mm256_storeu_ps(sp.add((i0 + 1) * BK + j0 + 8), a11);
            _mm256_storeu_ps(sp.add((i0 + 2) * BK + j0), a20);
            _mm256_storeu_ps(sp.add((i0 + 2) * BK + j0 + 8), a21);
            _mm256_storeu_ps(sp.add((i0 + 3) * BK + j0), a30);
            _mm256_storeu_ps(sp.add((i0 + 3) * BK + j0 + 8), a31);
            j0 += 16;
        }
        // Column remainder.
        for j in j0..cols {
            for m in 0..4 {
                let mut acc = 0f32;
                for t in 0..HD {
                    acc += qs.get_unchecked((i0 + m) * HD + t) * kt.get_unchecked(t * seq + k0 + j);
                }
                *scores.get_unchecked_mut((i0 + m) * BK + j) = acc;
            }
        }
        i0 += 4;
    }
    // Row remainder.
    for i in i0..rows {
        for j in 0..cols {
            let mut acc = 0f32;
            for t in 0..HD {
                acc += qs.get_unchecked(i * HD + t) * kt.get_unchecked(t * seq + k0 + j);
            }
            *scores.get_unchecked_mut(i * BK + j) = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// acc += P_block @ V_block  —  4 rows x 16 head-dims in registers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accum_pv(
    scores: &[f32],
    v: &[f32],
    acc: &mut [f32],
    rows: usize,
    k0: usize,
    cols: usize,
) {
    let vbase = v.as_ptr();
    let mut i0 = 0;
    while i0 + 4 <= rows {
        let mut d0 = 0;
        while d0 < HD {
            let ap = acc.as_mut_ptr();
            let mut a00 = _mm256_loadu_ps(ap.add((i0) * HD + d0));
            let mut a01 = _mm256_loadu_ps(ap.add((i0) * HD + d0 + 8));
            let mut a10 = _mm256_loadu_ps(ap.add((i0 + 1) * HD + d0));
            let mut a11 = _mm256_loadu_ps(ap.add((i0 + 1) * HD + d0 + 8));
            let mut a20 = _mm256_loadu_ps(ap.add((i0 + 2) * HD + d0));
            let mut a21 = _mm256_loadu_ps(ap.add((i0 + 2) * HD + d0 + 8));
            let mut a30 = _mm256_loadu_ps(ap.add((i0 + 3) * HD + d0));
            let mut a31 = _mm256_loadu_ps(ap.add((i0 + 3) * HD + d0 + 8));

            for j in 0..cols {
                let vp = vbase.add((k0 + j) * HD + d0);
                let vv0 = _mm256_loadu_ps(vp);
                let vv1 = _mm256_loadu_ps(vp.add(8));

                let p0 = _mm256_set1_ps(*scores.get_unchecked((i0) * BK + j));
                let p1 = _mm256_set1_ps(*scores.get_unchecked((i0 + 1) * BK + j));
                let p2 = _mm256_set1_ps(*scores.get_unchecked((i0 + 2) * BK + j));
                let p3 = _mm256_set1_ps(*scores.get_unchecked((i0 + 3) * BK + j));

                a00 = _mm256_fmadd_ps(p0, vv0, a00);
                a01 = _mm256_fmadd_ps(p0, vv1, a01);
                a10 = _mm256_fmadd_ps(p1, vv0, a10);
                a11 = _mm256_fmadd_ps(p1, vv1, a11);
                a20 = _mm256_fmadd_ps(p2, vv0, a20);
                a21 = _mm256_fmadd_ps(p2, vv1, a21);
                a30 = _mm256_fmadd_ps(p3, vv0, a30);
                a31 = _mm256_fmadd_ps(p3, vv1, a31);
            }

            _mm256_storeu_ps(ap.add((i0) * HD + d0), a00);
            _mm256_storeu_ps(ap.add((i0) * HD + d0 + 8), a01);
            _mm256_storeu_ps(ap.add((i0 + 1) * HD + d0), a10);
            _mm256_storeu_ps(ap.add((i0 + 1) * HD + d0 + 8), a11);
            _mm256_storeu_ps(ap.add((i0 + 2) * HD + d0), a20);
            _mm256_storeu_ps(ap.add((i0 + 2) * HD + d0 + 8), a21);
            _mm256_storeu_ps(ap.add((i0 + 3) * HD + d0), a30);
            _mm256_storeu_ps(ap.add((i0 + 3) * HD + d0 + 8), a31);
            d0 += 16;
        }
        i0 += 4;
    }
    for i in i0..rows {
        for j in 0..cols {
            let p = *scores.get_unchecked(i * BK + j);
            for d in 0..HD {
                *acc.get_unchecked_mut(i * HD + d) += p * v.get_unchecked((k0 + j) * HD + d);
            }
        }
    }
}

/// Online softmax over one score row, returning (new running max, correction).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row(row: &mut [f32], cols: usize, run_max: f32) -> (f32, f32, f32) {
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut j = 0;
    while j + 8 <= cols {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row.as_ptr().add(j)));
        j += 8;
    }
    let mut block_max = if cols >= 8 { hmax256(vmax) } else { f32::NEG_INFINITY };
    for &s in row[j..cols].iter() {
        block_max = block_max.max(s);
    }

    let new_max = run_max.max(block_max);
    let correction = if run_max.is_finite() { (run_max - new_max).exp() } else { 0.0 };

    let vnm = _mm256_set1_ps(new_max);
    let mut vsum = _mm256_setzero_ps();
    let mut j = 0;
    while j + 8 <= cols {
        let p = exp256(_mm256_sub_ps(_mm256_loadu_ps(row.as_ptr().add(j)), vnm));
        _mm256_storeu_ps(row.as_mut_ptr().add(j), p);
        vsum = _mm256_add_ps(vsum, p);
        j += 8;
    }
    let mut block_sum = if cols >= 8 { hsum256(vsum) } else { 0.0 };
    for s in row[j..cols].iter_mut() {
        *s = (*s - new_max).exp();
        block_sum += *s;
    }
    (new_max, correction, block_sum)
}

/// Fused attention for one head. `q` is (seq, HD), `kt` is (HD, seq),
/// `v` is (seq, HD), `out` is (seq, HD). The seq x seq score matrix is never
/// materialized — only a BQ x BK tile ever exists.
#[cfg(target_arch = "x86_64")]
fn flash_head(q: &[f32], kt: &[f32], v: &[f32], out: &mut [f32], seq: usize, scale: f32) {
    let mut scores = vec![0f32; BQ * BK];
    let mut acc = vec![0f32; BQ * HD];
    let mut qs = vec![0f32; BQ * HD];
    let mut run_max = vec![0f32; BQ];
    let mut run_sum = vec![0f32; BQ];

    for q0 in (0..seq).step_by(BQ) {
        let rows = BQ.min(seq - q0);
        // Fold the 1/sqrt(d) scale into the query block once, not into every
        // score, and give the kernel a contiguous block to broadcast from.
        for i in 0..rows {
            for t in 0..HD {
                qs[i * HD + t] = q[(q0 + i) * HD + t] * scale;
            }
        }
        acc[..rows * HD].fill(0.0);
        run_max[..rows].fill(f32::NEG_INFINITY);
        run_sum[..rows].fill(0.0);

        for k0 in (0..seq).step_by(BK) {
            let cols = BK.min(seq - k0);
            unsafe {
                scores_tile(&qs, kt, &mut scores, rows, k0, cols, seq);
                for i in 0..rows {
                    let (new_max, correction, block_sum) =
                        softmax_row(&mut scores[i * BK..(i + 1) * BK], cols, run_max[i]);
                    run_sum[i] = run_sum[i] * correction + block_sum;
                    run_max[i] = new_max;
                    if correction != 1.0 {
                        for x in acc[i * HD..(i + 1) * HD].iter_mut() {
                            *x *= correction;
                        }
                    }
                }
                accum_pv(&scores, v, &mut acc, rows, k0, cols);
            }
        }

        for i in 0..rows {
            let inv = 1.0 / run_sum[i];
            for (o, &a) in out[(q0 + i) * HD..(q0 + i + 1) * HD]
                .iter_mut()
                .zip(&acc[i * HD..(i + 1) * HD])
            {
                *o = a * inv;
            }
        }
    }
}

/// `kt` is (heads, HD, seq) — head-dim major.
fn flash_attention(q: &[f32], kt: &[f32], v: &[f32], out: &mut [f32], heads: usize, seq: usize) {
    let scale = 1.0 / (HD as f32).sqrt();
    let per_head = seq * HD;
    out.par_chunks_mut(per_head).enumerate().for_each(|(h, o)| {
        let r = h * per_head..(h + 1) * per_head;
        flash_head(&q[r.clone()], &kt[r.clone()], &v[r], o, seq, scale);
    });
    let _ = heads;
}

fn best<T>(n: usize, mut f: impl FnMut() -> T) -> f64 {
    std::hint::black_box(f());
    let mut b = f64::MAX;
    for _ in 0..n {
        let t = Instant::now();
        std::hint::black_box(f());
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        println!("AVX2/FMA not available — kernel not applicable on this machine");
        return Ok(());
    }

    let dev = Device::Cpu;
    let (heads, seq) = (6usize, 1500usize);
    let scale = 1.0 / (HD as f64).sqrt();

    let q = Tensor::randn(0f32, 1., (1, heads, seq, HD), &dev)?.contiguous()?;
    let k = Tensor::randn(0f32, 1., (1, heads, seq, HD), &dev)?.contiguous()?;
    let v = Tensor::randn(0f32, 1., (1, heads, seq, HD), &dev)?.contiguous()?;
    let kt = k.transpose(2, 3)?.contiguous()?;

    let qv: Vec<f32> = q.flatten_all()?.to_vec1()?;
    let kv: Vec<f32> = kt.flatten_all()?.to_vec1()?;
    let vv: Vec<f32> = v.flatten_all()?.to_vec1()?;
    let mut ours = vec![0f32; heads * seq * HD];

    let candle_path = |q: &Tensor, kt: &Tensor, v: &Tensor| -> Tensor {
        let s = (q.matmul(kt).unwrap() * scale).unwrap();
        let w = candle_nn::ops::softmax_last_dim(&s).unwrap();
        w.matmul(v).unwrap()
    };

    // ---- correctness first ----
    flash_attention(&qv, &kv, &vv, &mut ours, heads, seq);
    let reference: Vec<f32> = candle_path(&q, &kt, &v).flatten_all()?.to_vec1()?;
    let max_abs = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("correctness: max |delta| vs candle three-op path = {max_abs:.3e}");
    if max_abs > 1e-4 {
        println!("  FAIL — kernel is wrong, speed is irrelevant");
        return Ok(());
    }

    // ---- interleaved paired A/B ----
    let rounds = 21;
    let (mut wins, mut ta, mut tb) = (0usize, Vec::new(), Vec::new());
    for i in 0..rounds {
        let (a, b) = if i % 2 == 0 {
            let a = best(1, || flash_attention(&qv, &kv, &vv, &mut ours, heads, seq));
            (a, best(1, || candle_path(&q, &kt, &v)))
        } else {
            let b = best(1, || candle_path(&q, &kt, &v));
            (
                best(1, || flash_attention(&qv, &kv, &vv, &mut ours, heads, seq)),
                b,
            )
        };
        if a < b {
            wins += 1;
        }
        ta.push(a);
        tb.push(b);
    }
    ta.sort_by(f64::total_cmp);
    tb.sort_by(f64::total_cmp);
    let (ma, mb) = (ta[rounds / 2], tb[rounds / 2]);

    let flops = 2.0 * 2.0 * (heads * seq * seq * HD) as f64;
    let z = (wins as f64 - 0.5 * rounds as f64) / (0.5 * (rounds as f64).sqrt());

    println!("\nattention, one encoder layer (6 heads, seq 1500, hd 64)");
    println!(
        "  fused flash kernel   med {:7.2} ms   {:6.0} GFLOP/s",
        ma * 1e3,
        flops / ma / 1e9
    );
    println!(
        "  candle three-op      med {:7.2} ms   {:6.0} GFLOP/s",
        mb * 1e3,
        flops / mb / 1e9
    );
    println!(
        "  paired: fused won {wins}/{rounds} (z={z:+.1}) · ratio {:.2}x",
        mb / ma
    );
    println!(
        "\n  STOP CONDITION (beat the three-op path, z>2): {}",
        if z > 2.0 { "PASSED — wire it in" } else { "FAILED" }
    );
    Ok(())
}
