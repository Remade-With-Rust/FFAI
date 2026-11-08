//! Stride-1, "same"-padded max-pool for SPPF, as the separable pass it is.
//!
//! # Why this exists (measured, not assumed)
//!
//! SPPF's three 5x5 pools went through `pad_with_same` twice (two `cat`s of
//! narrow slices per pool, 58 200 small `memcpy`s per image) and then
//! candle's generic `max_pool2d`, which walks all 25 taps with strided index
//! arithmetic per output element. callgrind put the pools at 48.8 M Ir and
//! the padding copies at 14.8 M — 6.6 % of detect — for 153 600 output
//! elements, i.e. **318 instructions per element** to take a maximum of 25
//! numbers.
//!
//! A max over a k×k window is a max over k row-maxima, and the replicate
//! padding is a clamp on the index. So: one horizontal pass per row (k
//! compares), one vertical pass per row (k compares), no padded tensors.
//!
//! # Exactly candle's answer, bit for bit
//!
//! candle seeds with the window's top-left element and replaces on strict
//! `<`, so among equal values the FIRST in row-major order wins — which only
//! matters for the sign of zero. The separable form keeps that rule: the
//! horizontal pass keeps the leftmost of equals in a row, the vertical pass
//! keeps the topmost of equal row maxima, and leftmost-in-topmost IS
//! row-major first. The replicated pad values are copies of an edge element
//! already inside the window, so they never replace anything. `max` is exact
//! arithmetic, so there is no rounding to reorder. The test below asserts bit
//! equality against candle's own path, with repeated values and both zero
//! signs in the input to exercise exactly that rule.

use candle_core::{Result, Tensor};

/// A `Vec<f32>` of length `n` whose contents are UNINITIALISED, for buffers
/// every element of which the caller writes before reading. `vec![0.0; n]`
/// is a `calloc` whose zero fill the pool then overwrote (0.43 M Ir per image).
///
/// # Safety
///
/// The caller must write all `n` elements before reading any.
#[allow(unsafe_code, clippy::uninit_vec)]
unsafe fn uninit_f32(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::with_capacity(n);
    // SAFETY: `f32` has no invalid bit patterns, and the caller's contract
    // (above) is that every element is written before it is read.
    unsafe {
        v.set_len(n);
    }
    v
}

/// `max_pool2d(k, stride 1, padding k/2)` with replicate semantics, on
/// `(1, c, h, w)`. `k` must be odd.
pub fn maxpool_same_s1(x: &Tensor, k: usize) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    debug_assert_eq!(n, 1, "batch 1 is the only shape this engine runs");
    debug_assert_eq!(k % 2, 1, "SPPF kernels are odd");
    let r = k / 2;
    crate::cpuop::SliceOp::new("ffai-maxpool-same", move |xs, _| {
        let hw = h * w;
        // SAFETY: `row_max` fills every element of every `hm` row before the
        // vertical pass reads it; the vertical pass writes every element of
        // `out` (a `copy_from_slice` per row) before anything reads it.
        #[allow(unsafe_code)]
        let (mut out, mut hm) = unsafe { (uninit_f32(c * hw), uninit_f32(hw)) };
        for ch in 0..c {
            let plane = &xs[ch * hw..(ch + 1) * hw];
            let o = &mut out[ch * hw..(ch + 1) * hw];
            // Horizontal: hm[y][x] = max(plane[y][clamp(x-r)..=clamp(x+r)]).
            for (src, dst) in plane.chunks_exact(w).zip(hm.chunks_exact_mut(w)) {
                row_max(src, dst, r);
            }
            // Vertical: o[y] = max over rows clamp(y-r)..=clamp(y+r) of hm.
            for y in 0..h {
                let y0 = y.saturating_sub(r);
                let y1 = (y + r).min(h - 1);
                let o = &mut o[y * w..(y + 1) * w];
                vertical_max(o, &hm, w, y0, y1);
            }
        }
        Ok((out, (1, c, h, w).into()))
    })
    .run(x)
}

/// `m[i] = if m[i] < v[i] { v[i] } else { m[i] }` — strict `<`, so the value
/// already held wins a tie (candle's first-seen rule), written as a SELECT.
///
/// The `if m < v { m = v }` form compiled to a compare, a branch and a
/// conditional store per element (the line census put 390 k Ir on the two
/// store lines for 153 k outputs). A select on a value LLVM vectorizes; the
/// result is the same bit pattern, including for signed zeros and NaN.
#[inline(always)]
fn max_into_body(m: &mut [f32], v: &[f32]) {
    let n = m.len().min(v.len());
    let (m, v) = (&mut m[..n], &v[..n]);
    for i in 0..n {
        let (a, b) = (m[i], v[i]);
        m[i] = if a < b { b } else { a };
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

/// # Safety
///
/// The host must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn max_into_avx2(m: &mut [f32], v: &[f32]) {
    max_into_body(m, v);
}

fn max_into(m: &mut [f32], v: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: avx2 verified at runtime, just above.
        #[allow(unsafe_code)]
        unsafe {
            max_into_avx2(m, v);
        }
        return;
    }
    max_into_body(m, v);
}

/// `o[x] = max(hm[y0..=y1][x])` down the rows, strict-`<` in row order (the
/// topmost of equal values wins). Five rows — the interior of a 5x5 pool —
/// take the fused single pass; the clamped edges take the generic chain.
fn vertical_max(o: &mut [f32], hm: &[f32], w: usize, y0: usize, y1: usize) {
    if y1 - y0 == 4 {
        max5_into(
            o,
            [
                &hm[y0 * w..(y0 + 1) * w],
                &hm[(y0 + 1) * w..(y0 + 2) * w],
                &hm[(y0 + 2) * w..(y0 + 3) * w],
                &hm[(y0 + 3) * w..(y0 + 4) * w],
                &hm[(y0 + 4) * w..(y0 + 5) * w],
            ],
        );
        return;
    }
    o.copy_from_slice(&hm[y0 * w..(y0 + 1) * w]);
    for yy in y0 + 1..=y1 {
        max_into(o, &hm[yy * w..(yy + 1) * w]);
    }
}

/// `o[i] = max(s0[i], s1[i], s2[i], s3[i], s4[i])` with the strict-`<` chain
/// in that order: the copy-then-four-passes result, in one pass.
#[inline(always)]
fn max5_body(o: &mut [f32], s: [&[f32]; 5]) {
    let n = o.len().min(s[0].len()).min(s[1].len()).min(s[2].len()).min(s[3].len()).min(s[4].len());
    let (o, s0, s1, s2, s3, s4) = (&mut o[..n], &s[0][..n], &s[1][..n], &s[2][..n], &s[3][..n], &s[4][..n]);
    for i in 0..n {
        let mut m = s0[i];
        let v = s1[i];
        m = if m < v { v } else { m };
        let v = s2[i];
        m = if m < v { v } else { m };
        let v = s3[i];
        m = if m < v { v } else { m };
        let v = s4[i];
        m = if m < v { v } else { m };
        o[i] = m;
    }
}

/// # Safety
///
/// The host must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn max5_avx2(o: &mut [f32], s: [&[f32]; 5]) {
    max5_body(o, s);
}

fn max5_into(o: &mut [f32], s: [&[f32]; 5]) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: avx2 verified at runtime, just above.
        #[allow(unsafe_code)]
        unsafe {
            max5_avx2(o, s);
        }
        return;
    }
    max5_body(o, s);
}

/// `dst[x] = max(src[clamp(x-r)..=clamp(x+r)])`, strict-`<` replacement so
/// the leftmost of equal values wins.
fn row_max(src: &[f32], dst: &mut [f32], r: usize) {
    let w = src.len();
    let clamped = |x: usize, dst: &mut [f32]| {
        let lo = x.saturating_sub(r);
        let hi = (x + r).min(w - 1);
        let mut m = src[lo];
        for &v in &src[lo + 1..=hi] {
            if m < v {
                m = v;
            }
        }
        dst[x] = m;
    };
    if w <= 2 * r {
        // Window wider than the row: every column clamps.
        for x in 0..w {
            clamped(x, dst);
        }
        return;
    }
    // Interior: x in r..w-r, window x-r..=x+r all in bounds. Written as `k`
    // shifted slices of the same length so the loop vectorizes.
    let n = w - 2 * r;
    if r == 2 {
        max5_into(&mut dst[r..r + n], [&src[..n], &src[1..=n], &src[2..n + 2], &src[3..n + 3], &src[4..n + 4]]);
    } else {
        let interior = &mut dst[r..r + n];
        interior.copy_from_slice(&src[..n]);
        for t in 1..=2 * r {
            max_into(interior, &src[t..t + n]);
        }
    }
    // Edges: the clamp.
    for x in (0..r).chain(w - r..w) {
        clamped(x, dst);
    }
}

/// SPPF's whole pyramid in ONE buffer: `cat([x, p1, p2, ..., p_n], dim 1)`
/// where `p_{i+1} = maxpool_same_s1(p_i, k)`.
///
/// The chain allocated each pooled tensor, then `cat` copied all `n + 1` of
/// them into a fifth buffer. Here each stage pools the previous stage's
/// slice of the output into its own slice of the same output; `x` is one
/// `memcpy` into the head of it. Same per-element maxima, no copies.
pub fn maxpool_chain_cat(x: &Tensor, k: usize, n: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    debug_assert_eq!(b, 1, "batch 1 is the only shape this engine runs");
    let r = k / 2;
    crate::cpuop::SliceOp::new("ffai-sppf-pyramid", move |xs, _| {
        let hw = h * w;
        let stage = c * hw;
        // SAFETY: stage 0 is `copy_from_slice`d from the input and each later
        // stage is written in full by `pool_plane` before the next reads it.
        #[allow(unsafe_code)]
        let (mut out, mut hm) = unsafe { (uninit_f32((n + 1) * stage), uninit_f32(hw)) };
        out[..stage].copy_from_slice(&xs[..stage]);
        for si in 0..n {
            let (done, rest) = out.split_at_mut((si + 1) * stage);
            let src = &done[si * stage..];
            let dst = &mut rest[..stage];
            for ch in 0..c {
                pool_plane(&src[ch * hw..(ch + 1) * hw], &mut dst[ch * hw..(ch + 1) * hw], &mut hm, h, w, r);
            }
        }
        Ok((out, (1, (n + 1) * c, h, w).into()))
    })
    .run(x)
}

/// One plane of the separable pool (shared by both entry points).
fn pool_plane(plane: &[f32], o: &mut [f32], hm: &mut [f32], h: usize, w: usize, r: usize) {
    for (src, dst) in plane.chunks_exact(w).zip(hm.chunks_exact_mut(w)) {
        row_max(src, dst, r);
    }
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        let o = &mut o[y * w..(y + 1) * w];
        vertical_max(o, hm, w, y0, y1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    #[test]
    fn pyramid_matches_the_chain_of_pools_and_cat() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 6, 9, 11), &dev).unwrap();
        let mut ys = vec![x.clone()];
        for _ in 0..3 {
            let prev = ys.last().unwrap();
            ys.push(maxpool_same_s1(prev, 5).unwrap());
        }
        let want = Tensor::cat(&ys, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = maxpool_chain_cat(&x, 5, 3).unwrap();
        assert_eq!(got.dims(), &[1, 24, 9, 11]);
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), wv.to_bits(), "element {i}");
        }
    }

    /// candle's own path stays the oracle — forever.
    fn oracle(x: &Tensor, k: usize) -> Tensor {
        let pad = k / 2;
        x.pad_with_same(D::Minus1, pad, pad)
            .unwrap()
            .pad_with_same(D::Minus2, pad, pad)
            .unwrap()
            .max_pool2d_with_stride(k, 1)
            .unwrap()
    }

    #[test]
    fn matches_candle_bit_for_bit() {
        let dev = Device::Cpu;
        for &(c, h, w, k) in &[
            (128, 20, 20, 5),
            (3, 7, 9, 5),
            (2, 4, 4, 3),
            (1, 2, 3, 5),
            (1, 1, 1, 5),
            (4, 5, 11, 3),
        ] {
            // Few distinct values, so equal neighbours are common, and both
            // zero signs, so the first-seen rule is exercised.
            let vals: Vec<f32> = (0..c * h * w)
                .map(|i| {
                    let j = (i * 7919 + 13) % 11;
                    match j {
                        0 => 0.0,
                        1 => -0.0,
                        _ => (j as f32 - 5.0) * 0.25,
                    }
                })
                .collect();
            let x = Tensor::from_vec(vals, (1, c, h, w), &dev).unwrap();
            let got = maxpool_same_s1(&x, k).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let want = oracle(&x, k).flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(got.len(), want.len(), "c={c} {h}x{w} k={k}");
            for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), wv.to_bits(), "c={c} {h}x{w} k={k} element {i}: {g} vs {wv}");
            }
        }
    }
}
