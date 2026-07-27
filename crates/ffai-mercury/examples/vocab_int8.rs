//! Hand-written AVX2 int8 GEMV for the vocabulary projection.
//!
//! The projection is ~27 % of decode and pure bandwidth: `(1,384) @ (384,51864)`
//! streams the whole embedding matrix per token with no reuse. We ship f16
//! (39.8 MB/token) at ~25.5 GB/s — 76 % of this machine's 33.57 GB/s memcpy
//! ceiling, so the shipped path is already good.
//!
//! candle's Q8_0 halves the bytes to 21.2 MB and measured **no faster**
//! (20/41, z = -0.2): it converts a memory-bound op into a compute-bound one,
//! spending on dequantization what it saves on traffic.
//!
//! That is a verdict on the kernel, not on int8. AVX2 has dedicated int8
//! dot-product instructions — `_mm256_maddubs_epi16` yields 16 i16 products
//! from 32 int8 pairs, `_mm256_madd_epi16` folds them to i32 — giving **32
//! MACs per two instructions against 8 for an f32 FMA**. Done directly, int8
//! should win on compute *and* bytes rather than trading one for the other.
//!
//! **Saturation is the design constraint.** `maddubs` accumulates two u8xi8
//! products into a saturating i16. At full range 255x127x2 = 64770 overflows.
//! Quantizing activations to 7 bits (offset to u8 range [0,127]) caps the pair
//! sum at 127x127x2 = 32258 < 32767. One bit is cheap here: the activation is
//! a single 384-vector whose error is negligible beside the weight matrix's.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example vocab_int8
//! ```

use std::time::Instant;

use ffai_core::candle::{DType, Device, Tensor};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const D: usize = 384;

/// Row-symmetric int8 weights: one scale per vocabulary row, which is the
/// right granularity here because argmax compares *across* rows.
struct Int8Vocab {
    q: Vec<i8>,
    scale: Vec<f32>,
    /// Sum of each row's quantized weights, to undo the activation offset.
    rowsum: Vec<i32>,
    vocab: usize,
}

impl Int8Vocab {
    /// `w` is (vocab, D) row-major.
    fn new(w: &[f32], vocab: usize) -> Self {
        let mut q = vec![0i8; vocab * D];
        let mut scale = vec![0f32; vocab];
        let mut rowsum = vec![0i32; vocab];
        for v in 0..vocab {
            let row = &w[v * D..(v + 1) * D];
            let amax = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            scale[v] = s;
            let mut sum = 0i32;
            for k in 0..D {
                let qi = (row[k] / s).round().clamp(-127.0, 127.0) as i8;
                q[v * D + k] = qi;
                sum += qi as i32;
            }
            rowsum[v] = sum;
        }
        Int8Vocab { q, scale, rowsum, vocab }
    }

    fn bytes(&self) -> usize {
        self.vocab * D + self.vocab * 4 + self.vocab * 4
    }

    fn forward(&self, x: &[f32], out: &mut [f32]) {
        // 7-bit activations, offset into u8 so `maddubs` can take them.
        let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let sx = if amax > 0.0 { amax / 63.0 } else { 1.0 };
        let mut xu = vec![0u8; D];
        for k in 0..D {
            let qi = (x[k] / sx).round().clamp(-64.0, 63.0) as i32;
            xu[k] = (qi + 64) as u8;
        }

        let chunk = 512;
        out.par_chunks_mut(chunk).enumerate().for_each(|(ci, o)| {
            let v0 = ci * chunk;
            for (i, slot) in o.iter_mut().enumerate() {
                let v = v0 + i;
                let dot = unsafe { dot_i8(&self.q[v * D..(v + 1) * D], &xu) };
                // sum((x+64)*w) = sum(x*w) + 64*sum(w)
                let corrected = dot - 64 * self.rowsum[v];
                *slot = corrected as f32 * self.scale[v] * sx;
            }
        });
    }
}

/// int8 dot product over D=384: 12 iterations of 32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8(w: &[i8], xu: &[u8]) -> i32 {
    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let wp = w.as_ptr();
    let xp = xu.as_ptr();
    let mut k = 0;
    // Two accumulator chains so the dependent adds overlap.
    while k + 64 <= D {
        let x0 = _mm256_loadu_si256(xp.add(k) as *const __m256i);
        let w0 = _mm256_loadu_si256(wp.add(k) as *const __m256i);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, w0), ones));
        let x1 = _mm256_loadu_si256(xp.add(k + 32) as *const __m256i);
        let w1 = _mm256_loadu_si256(wp.add(k + 32) as *const __m256i);
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(x1, w1), ones));
        k += 64;
    }
    while k + 32 <= D {
        let x0 = _mm256_loadu_si256(xp.add(k) as *const __m256i);
        let w0 = _mm256_loadu_si256(wp.add(k) as *const __m256i);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, w0), ones));
        k += 32;
    }
    let acc = _mm256_add_epi32(acc0, acc1);
    let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_01_00_01));
    let mut tail = _mm_cvtsi128_si32(s);
    for kk in k..D {
        tail += (*xu.get_unchecked(kk) as i32) * (*w.get_unchecked(kk) as i32);
    }
    tail
}

fn t(mut f: impl FnMut()) -> f64 {
    let s = Instant::now();
    f();
    s.elapsed().as_secs_f64() * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !is_x86_feature_detected!("avx2") {
        println!("no AVX2");
        return Ok(());
    }
    let dev = Device::Cpu;
    let vocab = 51864usize;

    let w32 = Tensor::randn(0f32, 1., (vocab, D), &dev)?;
    let x = Tensor::randn(0f32, 1., (1, D), &dev)?;
    let wv: Vec<f32> = w32.flatten_all()?.to_vec1()?;
    let xv: Vec<f32> = x.flatten_all()?.to_vec1()?;

    // Shipped arm: pre-transposed f16 + the adaptive row padding.
    let wt16 = w32.t()?.contiguous()?.to_dtype(DType::F16)?;
    let x16 = x.to_dtype(DType::F16)?;
    let pad = 2usize;

    let q = Int8Vocab::new(&wv, vocab);
    let mut ours = vec![0f32; vocab];
    q.forward(&xv, &mut ours);

    // ---- accuracy: only the ARGMAX decides a token ----
    let refv: Vec<f32> = x.matmul(&w32.t()?)?.flatten_all()?.to_vec1()?;
    let f16v: Vec<f32> = Tensor::cat(&vec![&x16; pad], 0)?
        .matmul(&wt16)?
        .narrow(0, 0, 1)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let am = |v: &Vec<f32>| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let rel = |v: &Vec<f32>| {
        v.iter()
            .zip(&refv)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / refv.iter().fold(0f32, |m, &x| m.max(x.abs()))
    };
    println!(
        "argmax  f32 {}  f16 {}  int8 {}   |  max rel err  f16 {:.2e}  int8 {:.2e}",
        am(&refv),
        am(&f16v),
        am(&ours),
        rel(&f16v),
        rel(&ours)
    );

    let f16_bytes = vocab * D * 2;
    println!(
        "bytes/token   f16 {:.1} MB   int8 {:.1} MB   ({:.2}x less)",
        f16_bytes as f64 / 1e6,
        q.bytes() as f64 / 1e6,
        f16_bytes as f64 / q.bytes() as f64
    );

    // ---- interleaved paired A/B ----
    let rounds = 41;
    let (mut wins, mut va, mut vb) = (0usize, vec![], vec![]);
    let mut buf = vec![0f32; vocab];
    let mut a = |buf: &mut Vec<f32>| t(|| q.forward(&xv, buf));
    let b = || {
        t(|| {
            std::hint::black_box(
                Tensor::cat(&vec![&x16; pad], 0)
                    .unwrap()
                    .matmul(&wt16)
                    .unwrap()
                    .narrow(0, 0, 1)
                    .unwrap(),
            );
        })
    };
    a(&mut buf);
    b();
    for i in 0..rounds {
        let (p, r) = if i % 2 == 0 {
            let p = a(&mut buf);
            (p, b())
        } else {
            let r = b();
            (a(&mut buf), r)
        };
        if p < r {
            wins += 1;
        }
        va.push(p);
        vb.push(r);
    }
    va.sort_by(f64::total_cmp);
    vb.sort_by(f64::total_cmp);
    let (ma, mb) = (va[rounds / 2], vb[rounds / 2]);
    let z = (wins as f64 - 0.5 * rounds as f64) / (0.5 * (rounds as f64).sqrt());

    println!("\nvocabulary projection, {rounds} paired rounds");
    println!(
        "  AVX2 int8       med {ma:7.3} ms   {:5.1} GB/s",
        q.bytes() as f64 / (ma / 1e3) / 1e9
    );
    println!(
        "  f16 + pad       med {mb:7.3} ms   {:5.1} GB/s",
        f16_bytes as f64 / (mb / 1e3) / 1e9
    );
    println!(
        "  int8 won {wins}/{rounds} (z={z:+.1}) ratio {:.2}x -> {}",
        mb / ma,
        if z.abs() < 2.0 {
            "INCONCLUSIVE"
        } else if z > 0.0 {
            "int8 FASTER"
        } else {
            "f16 FASTER"
        }
    );
    Ok(())
}
