//! A direct kernel for the STEM: 3x3, stride 2, `c_in <= 4` — vein 12.
//!
//! # Why the general refutation does not apply here
//!
//! A hand-written direct 3x3 lost to im2col + GEMM at every real shape in
//! this graph (`direct3x3.rs`), because a tuned GEMM amortises its packing
//! over `K = 9 * c_in` and the im2col over `c_out`. The stem is the one
//! convolution where neither holds: `c_in = 3`, so `K = 27` — too short for
//! the microkernel to reach its throughput — and the im2col materialises
//! 27 copies of a 448x640 input for the privilege. callgrind put the stem at
//! roughly 20 M instructions for 31 M multiply-adds.
//!
//! # Bit-identical to the GEMM, by construction
//!
//! The `gemm` crate's microkernel accumulates each output element as
//! `acc = fma(lhs[k], rhs[k], acc)` for `k = 0..K` in order, from zero, with
//! `K = 27 < kc` so there is no split of the k loop, and `alpha = 0, beta = 1`
//! so nothing is scaled. This kernel accumulates with the same FMA, in the
//! same k order (`c * 9 + ky * 3 + kx`, the im2col row order), from zero.
//! The bias and activation then go through the same in-place epilogue. So
//! the output is the GEMM path's to the bit — asserted by the test and by
//! the full-graph `fnv` anchor. `FFAI_DIANA_NO_STEM=1` restores the GEMM.

#[cfg(target_arch = "x86_64")]
use candle_core::{Result, Tensor};

/// `FFAI_DIANA_NO_STEM=1` restores the im2col + GEMM path for the stem.
pub fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_STEM").is_none())
}

/// Largest `c_in` this kernel serves; above it the GEMM wins.
pub const MAX_C_IN: usize = 4;

#[cfg(target_arch = "x86_64")]
fn avx2_fma_available() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"))
}

/// Whether this convolution takes the direct stem kernel.
#[must_use]
pub fn applies(c_in: usize, stride: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        enabled() && stride == 2 && c_in <= MAX_C_IN && avx2_fma_available()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (c_in, stride);
        false
    }
}

/// 3x3, stride 2, padding 1, direct. `x: (1, Ci, H, W)`, `weight: (Co, Ci, 3, 3)`.
#[cfg(target_arch = "x86_64")]
pub fn conv3x3_s2_direct(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, act: bool) -> Result<Tensor> {
    let (n, c_in, h, w) = x.dims4()?;
    debug_assert_eq!(n, 1, "batch 1 is the only shape this engine runs");
    let c_out = weight.dim(0)?;
    let (oh, ow) = ((h + 2 - 3) / 2 + 1, (w + 2 - 3) / 2 + 1);
    let ws = crate::cpuop::bias_vec(weight)?;
    let y = crate::cpuop::SliceOp::new("ffai-stem-direct", move |xs, _| {
        // SAFETY: avx2+fma verified in `applies()`.
        #[allow(unsafe_code)]
        let out = unsafe { stem_avx2(xs, &ws, c_in, c_out, h, w, oh, ow) };
        Ok((out, (c_out, oh * ow).into()))
    })
    .run(x)?;
    let bias_v = match bias {
        Some(b) => Some(crate::cpuop::bias_vec(b)?),
        None => None,
    };
    let y = crate::epilogue::apply(y, bias_v, oh * ow, act)?;
    y.reshape((n, c_out, oh, ow))
}

/// Even/odd de-interleave of one input row into `[0, odd..., 0]` and
/// `[even..., (0)]` so every tap of a stride-2 window is a contiguous load:
/// output `ox` reads `kx=0 -> oddp[ox]`, `kx=1 -> even[ox]`, `kx=2 -> oddp[ox+1]`.
#[cfg(target_arch = "x86_64")]
fn deinterleave_padded(row: &[f32], even: &mut [f32], oddp: &mut [f32]) {
    let w = row.len();
    let half = w / 2;
    // Equal-length zips so the split vectorizes (no per-element index
    // checks); only the pad cells beyond the data are zeroed.
    {
        let (e, o) = (&mut even[..half], &mut oddp[1..=half]);
        for ((e, o), pair) in e.iter_mut().zip(o.iter_mut()).zip(row.chunks_exact(2)) {
            *e = pair[0];
            *o = pair[1];
        }
    }
    oddp[0] = 0.0;
    if w % 2 == 1 {
        even[half] = row[w - 1];
        even[half + 1..].fill(0.0);
    } else {
        even[half..].fill(0.0);
    }
    oddp[half + 1..].fill(0.0);
}

/// # Safety
///
/// The host must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn stem_avx2(
    xs: &[f32],
    ws: &[f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    let ohw = oh * ow;
    // No constant fills: every (co, oy, ox) is stored by the blocks below,
    // and every cell of the split rows is written by `deinterleave_padded`
    // (data cells from the row, pad cells zeroed). The three `vec![0.0; ..]`
    // were ~2 M floats of zero fill per call, all of it overwritten.
    // SAFETY: `f32` is plain data; all `c_out * ohw` outputs are stored
    // before `out` is returned.
    #[allow(unsafe_code)]
    let mut out = unsafe { crate::cpuop::uninit_f32(c_out * ohw) };
    // De-interleaved rows for every input row and channel, padded so the
    // vector loop can run 8 outputs past `ow` without reading out of bounds.
    let half = w.div_ceil(2);
    let pw = half + 1 + 16;
    // SAFETY: as above — `deinterleave_padded` writes every cell of each
    // `pw`-long row (data, then zeroed pads to the end) before any read.
    #[allow(unsafe_code)]
    let (mut even, mut oddp) = unsafe { (crate::cpuop::uninit_f32(c_in * h * pw), crate::cpuop::uninit_f32(c_in * h * pw)) };
    for c in 0..c_in {
        for y in 0..h {
            let row = &xs[(c * h + y) * w..(c * h + y + 1) * w];
            let o = (c * h + y) * pw;
            deinterleave_padded(row, &mut even[o..o + pw], &mut oddp[o..o + pw]);
        }
    }
    let zero = vec![0f32; pw];
    // Tap source for (c, sy, kx): a padded de-interleaved row or the zero row.
    let tap_row = |c: usize, sy: isize, kx: usize| -> &[f32] {
        if sy < 0 || sy as usize >= h {
            return &zero[..];
        }
        let o = (c * h + sy as usize) * pw;
        match kx {
            0 => &oddp[o..o + pw],
            1 => &even[o..o + pw],
            _ => &oddp[o + 1..o + pw],
        }
    };
    let k_len = c_in * 9;
    // FOUR output channels x SIXTEEN outputs per pass: two input vectors and
    // four weight broadcasts feed eight FMAs, against one vector and eight
    // broadcasts per eight FMAs in the 8x8 blocking — 14 instructions per
    // 64 multiply-adds instead of 17. Eight named accumulators either way.
    let mut taps: Vec<*const f32> = vec![std::ptr::null(); k_len];
    let mut wk = vec![0f32; k_len * 4];
    let mut co0 = 0;
    while co0 < c_out {
        let nco = (c_out - co0).min(4);
        for k in 0..k_len {
            for j in 0..4 {
                wk[k * 4 + j] = if j < nco { ws[(co0 + j) * k_len + k] } else { 0.0 };
            }
        }
        for oy in 0..oh {
            for c in 0..c_in {
                for ky in 0..3usize {
                    let sy = 2 * oy as isize + ky as isize - 1;
                    for kx in 0..3usize {
                        taps[c * 9 + ky * 3 + kx] = tap_row(c, sy, kx).as_ptr();
                    }
                }
            }
            let mut ox = 0;
            while ox < ow {
                let (mut a0, mut a1, mut a2, mut a3) =
                    (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
                let (mut b0, mut b1, mut b2, mut b3) =
                    (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
                for (tap, wrow) in taps.iter().zip(wk.chunks_exact(4)) {
                    // SAFETY: every tap row is `pw` floats long and `ox + 16 <= pw`
                    // by the padding above; `wrow` holds four floats.
                    unsafe {
                        let xv = _mm256_loadu_ps(tap.add(ox));
                        let xw = _mm256_loadu_ps(tap.add(ox + 8));
                        let wp = wrow.as_ptr();
                        let w0 = _mm256_broadcast_ss(&*wp);
                        let w1 = _mm256_broadcast_ss(&*wp.add(1));
                        let w2 = _mm256_broadcast_ss(&*wp.add(2));
                        let w3 = _mm256_broadcast_ss(&*wp.add(3));
                        a0 = _mm256_fmadd_ps(w0, xv, a0);
                        b0 = _mm256_fmadd_ps(w0, xw, b0);
                        a1 = _mm256_fmadd_ps(w1, xv, a1);
                        b1 = _mm256_fmadd_ps(w1, xw, b1);
                        a2 = _mm256_fmadd_ps(w2, xv, a2);
                        b2 = _mm256_fmadd_ps(w2, xw, b2);
                        a3 = _mm256_fmadd_ps(w3, xv, a3);
                        b3 = _mm256_fmadd_ps(w3, xw, b3);
                    }
                }
                let accs = [(a0, b0), (a1, b1), (a2, b2), (a3, b3)];
                let lanes = (ow - ox).min(16);
                for (j, (a, b)) in accs.iter().enumerate().take(nco) {
                    let dst = &mut out[(co0 + j) * ohw + oy * ow + ox..];
                    if lanes == 16 {
                        // SAFETY: sixteen lanes fit in `dst`.
                        unsafe {
                            _mm256_storeu_ps(dst.as_mut_ptr(), *a);
                            _mm256_storeu_ps(dst.as_mut_ptr().add(8), *b);
                        }
                    } else {
                        let mut tmp = [0f32; 16];
                        // SAFETY: `tmp` holds sixteen floats.
                        unsafe {
                            _mm256_storeu_ps(tmp.as_mut_ptr(), *a);
                            _mm256_storeu_ps(tmp.as_mut_ptr().add(8), *b);
                        }
                        dst[..lanes].copy_from_slice(&tmp[..lanes]);
                    }
                }
                ox += 16;
            }
        }
        co0 += 4;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn is_bit_identical_to_the_gemm_path() {
        if !avx2_fma_available() {
            eprintln!("SKIP: no avx2+fma");
            return;
        }
        let dev = Device::Cpu;
        for &(c_in, c_out, h, w) in &[(3, 16, 448, 640), (3, 16, 9, 13), (1, 5, 4, 4), (4, 12, 7, 2)] {
            let x = Tensor::randn(0f32, 1.0, (1, c_in, h, w), &dev).unwrap();
            let wt = Tensor::randn(0f32, 0.3, (c_out, c_in, 3, 3), &dev).unwrap();
            let b = Tensor::randn(0f32, 1.0, c_out, &dev).unwrap();
            for act in [false, true] {
                let want = crate::conv3x3::conv3x3_gemm(&x, &wt, Some(&b), 2, act).unwrap();
                let got = conv3x3_s2_direct(&x, &wt, Some(&b), act).unwrap();
                assert_eq!(got.dims(), want.dims());
                let (g, wv) = (
                    got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                );
                let bad = g.iter().zip(&wv).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                assert_eq!(bad, 0, "ci={c_in} co={c_out} {h}x{w} act={act}: {bad} of {} lanes differ", g.len());
            }
        }
    }
}
