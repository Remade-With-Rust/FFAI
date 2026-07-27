//! Fused cross-attention for the DECODE shape (one query row).
//!
//! Cross-attention is now the largest single item in the pipeline — 47 % of
//! decode, ~24 % of total, after the encoder kernel landed. The shipped
//! `flash_attn` kernel declines it: that one tiles 64 query rows and keeps a
//! 4x16 register block, and the decoder has exactly ONE query row.
//!
//! Roofline: K and V are 18.4 MB/token across 4 layers, which at ~25 GB/s
//! (what a streaming read reaches here) floors at 0.74 ms/token against a
//! measured 2.49. That is a 3.4x hypothesis — **and this campaign has had two
//! wrong ceilings**, so it is a reason to measure, not a prize to claim.
//!
//! At M=1 there is no reuse to tile, so this is not flash attention; it is a
//! streaming kernel. K is read once as 64 contiguous rows (AXPY into a score
//! buffer, no horizontal reduction), softmaxed with a vectorized exp, then V
//! is read once (AXPY into the output). Both operands touched exactly once,
//! nothing intermediate materialized, no framework dispatch.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example xattn_fused
//! ```

use std::time::Instant;

use ffai_core::candle::{Device, Tensor};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const HD: usize = 64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp256(x: __m256) -> __m256 {
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

/// One head. `q` is HD floats (already scaled); `kt` is (HD, keys) head-dim
/// major; `v` is (keys, HD). `out` is HD floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xattn_head(q: &[f32], kt: &[f32], v: &[f32], out: &mut [f32], keys: usize, s: &mut [f32]) {
    // ---- scores = q @ K, contraction outermost so the inner loop is an AXPY
    // over keys with no horizontal reduction ----
    s[..keys].fill(0.0);
    for t in 0..HD {
        let qv = _mm256_set1_ps(*q.get_unchecked(t));
        let krow = kt.as_ptr().add(t * keys);
        let sp = s.as_mut_ptr();
        let mut j = 0;
        while j + 8 <= keys {
            let acc = _mm256_fmadd_ps(qv, _mm256_loadu_ps(krow.add(j)), _mm256_loadu_ps(sp.add(j)));
            _mm256_storeu_ps(sp.add(j), acc);
            j += 8;
        }
        let qs = *q.get_unchecked(t);
        for jj in j..keys {
            *s.get_unchecked_mut(jj) += qs * *krow.add(jj);
        }
    }

    // ---- softmax over the whole key axis ----
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut j = 0;
    while j + 8 <= keys {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(s.as_ptr().add(j)));
        j += 8;
    }
    let mut mx = if keys >= 8 { hmax256(vmax) } else { f32::NEG_INFINITY };
    for &x in s[j..keys].iter() {
        mx = mx.max(x);
    }
    let vmx = _mm256_set1_ps(mx);
    let mut vsum = _mm256_setzero_ps();
    let mut j = 0;
    while j + 8 <= keys {
        let p = exp256(_mm256_sub_ps(_mm256_loadu_ps(s.as_ptr().add(j)), vmx));
        _mm256_storeu_ps(s.as_mut_ptr().add(j), p);
        vsum = _mm256_add_ps(vsum, p);
        j += 8;
    }
    let mut sum = if keys >= 8 { hsum256(vsum) } else { 0.0 };
    for x in s[j..keys].iter_mut() {
        *x = (*x - mx).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;

    // ---- out = weights @ V, AXPY over HD (8 vectors), V read once ----
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut a4 = _mm256_setzero_ps();
    let mut a5 = _mm256_setzero_ps();
    let mut a6 = _mm256_setzero_ps();
    let mut a7 = _mm256_setzero_ps();
    for jj in 0..keys {
        let w = _mm256_set1_ps(*s.get_unchecked(jj) * inv);
        let vp = v.as_ptr().add(jj * HD);
        a0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp), a0);
        a1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(8)), a1);
        a2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(16)), a2);
        a3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(24)), a3);
        a4 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(32)), a4);
        a5 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(40)), a5);
        a6 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(48)), a6);
        a7 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vp.add(56)), a7);
    }
    let op = out.as_mut_ptr();
    _mm256_storeu_ps(op, a0);
    _mm256_storeu_ps(op.add(8), a1);
    _mm256_storeu_ps(op.add(16), a2);
    _mm256_storeu_ps(op.add(24), a3);
    _mm256_storeu_ps(op.add(32), a4);
    _mm256_storeu_ps(op.add(40), a5);
    _mm256_storeu_ps(op.add(48), a6);
    _mm256_storeu_ps(op.add(56), a7);
}

fn fused(q: &[f32], kt: &[f32], v: &[f32], out: &mut [f32], heads: usize, keys: usize) {
    let mut scratch = vec![0f32; keys];
    for h in 0..heads {
        unsafe {
            xattn_head(
                &q[h * HD..(h + 1) * HD],
                &kt[h * HD * keys..(h + 1) * HD * keys],
                &v[h * keys * HD..(h + 1) * keys * HD],
                &mut out[h * HD..(h + 1) * HD],
                keys,
                &mut scratch,
            );
        }
    }
}

fn t(mut f: impl FnMut()) -> f64 {
    let s = Instant::now();
    f();
    s.elapsed().as_secs_f64() * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        println!("no AVX2/FMA");
        return Ok(());
    }
    let dev = Device::Cpu;
    let (heads, keys) = (6usize, 1500usize);

    // Decode shape: ONE query row against the encoder's 1500 keys.
    let q = Tensor::randn(0f32, 1., (1, heads, 1, HD), &dev)?.contiguous()?;
    let kt = Tensor::randn(0f32, 1., (1, heads, HD, keys), &dev)?.contiguous()?;
    let v = Tensor::randn(0f32, 1., (1, heads, keys, HD), &dev)?.contiguous()?;

    let qv: Vec<f32> = q.flatten_all()?.to_vec1()?;
    let kv: Vec<f32> = kt.flatten_all()?.to_vec1()?;
    let vv: Vec<f32> = v.flatten_all()?.to_vec1()?;
    let mut ours = vec![0f32; heads * HD];

    let candle_path = || -> Tensor {
        let s = q.matmul(&kt).unwrap();
        let w = candle_nn::ops::softmax_last_dim(&s).unwrap();
        w.matmul(&v).unwrap()
    };

    fused(&qv, &kv, &vv, &mut ours, heads, keys);
    let reference: Vec<f32> = candle_path().flatten_all()?.to_vec1()?;
    let max_abs = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("correctness: max |delta| vs candle three-op = {max_abs:.3e}");
    if max_abs > 1e-4 {
        println!("  FAIL — wrong, speed irrelevant");
        return Ok(());
    }

    let rounds = 41;
    let (mut wins, mut va, mut vb) = (0usize, vec![], vec![]);
    for i in 0..rounds {
        let (a, b) = if i % 2 == 0 {
            let a = t(|| fused(&qv, &kv, &vv, &mut ours, heads, keys));
            (a, t(|| {
                std::hint::black_box(candle_path());
            }))
        } else {
            let b = t(|| {
                std::hint::black_box(candle_path());
            });
            (t(|| fused(&qv, &kv, &vv, &mut ours, heads, keys)), b)
        };
        if a < b {
            wins += 1;
        }
        va.push(a);
        vb.push(b);
    }
    va.sort_by(f64::total_cmp);
    vb.sort_by(f64::total_cmp);
    let (ma, mb) = (va[rounds / 2], vb[rounds / 2]);
    let z = (wins as f64 - 0.5 * rounds as f64) / (0.5 * (rounds as f64).sqrt());
    let bytes = (2 * heads * keys * HD * 4) as f64;

    println!("\ncross-attention, one decode step (6 heads, 1 query, 1500 keys)");
    println!("  fused kernel    med {ma:7.3} ms   {:5.1} GB/s", bytes / (ma / 1e3) / 1e9);
    println!("  candle three-op med {mb:7.3} ms   {:5.1} GB/s", bytes / (mb / 1e3) / 1e9);
    println!(
        "  fused won {wins}/{rounds} (z={z:+.1}) ratio {:.2}x -> {}",
        mb / ma,
        if z > 2.0 { "FUSED FASTER" } else if z < -2.0 { "candle FASTER" } else { "INCONCLUSIVE" }
    );
    Ok(())
}
