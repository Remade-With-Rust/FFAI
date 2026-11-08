//! Winograd F(2x2, 3x3) for the stride-1 dense 3x3 convolutions — vein 1.
//!
//! # The arithmetic
//!
//! A 2x2 output tile of a 3x3 convolution needs a 4x4 input tile. Winograd's
//! minimal filtering computes it with 16 multiplies per (c_out, c_in) instead
//! of 36: transform the input tile (`V = Bᵀ d B`), the filter once (`U = G g
//! Gᵀ`, cached), multiply element-wise across channels — which, over all
//! tiles, is 16 ordinary GEMMs `U[ξ] (Co x Ci) @ V[ξ] (Ci x T)` — and
//! transform back (`Y = Aᵀ M A`). The GEMM work drops by 2.25x; the
//! transforms are adds and a handful of halvings.
//!
//! ```text
//! Bᵀ = [ 1  0 -1  0 ]    G = [ 1    0    0  ]    Aᵀ = [ 1  1  1  0 ]
//!      [ 0  1  1  0 ]        [ 1/2  1/2  1/2]         [ 0  1 -1 -1 ]
//!      [ 0 -1  1  0 ]        [ 1/2 -1/2  1/2]
//!      [ 0  1  0 -1 ]        [ 0    0    1  ]
//! ```
//!
//! # What this changes
//!
//! The RESULT. The sums are reassociated and the filter is pre-scaled by one
//! half, so the output differs from the im2col+GEMM path in the last bits —
//! F(2x2, 3x3) has the mildest constants of the family and the error is a few
//! ulp, but it is not zero. This is the one vein in the campaign whose gate is
//! the full-graph oracle's tolerance (`tests/oracles.rs`), not byte identity.
//! `FFAI_DIANA_NO_WINOGRAD=1` restores the GEMM path for A/B.
//!
//! # Where it pays
//!
//! The transforms cost ~10 instructions per input element per channel and
//! ~8 per output element per channel, independent of the other channel count,
//! while the GEMM saving scales with `Ci * Co`. Below `MIN_CHANNELS` the
//! transforms cost more than the GEMM saves (measured on the n tier's
//! 64-channel 80x112 convs: break-even), so those stay on the GEMM path.

use candle_core::{Result, Tensor};

/// `FFAI_DIANA_NO_WINOGRAD=1` restores the im2col + GEMM path.
pub fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_WINOGRAD").is_none())
}

/// Smallest `min(c_in, c_out)` the transform overhead is paid back at.
pub const MIN_CHANNELS: usize = 16;

/// Whether this convolution takes the Winograd path.
#[must_use]
pub fn applies(c_in: usize, c_out: usize, stride: usize) -> bool {
    enabled() && stride == 1 && c_in.min(c_out) >= MIN_CHANNELS
}

/// 3x3, stride 1, padding 1 via F(2x2, 3x3). `x: (1, Ci, H, W)`, `weight:
/// (Co, Ci, 3, 3)`. Returns `(1, Co, H, W)` with bias and `SiLU` applied by
/// the shared in-place epilogue.
pub fn conv3x3_winograd(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, act: bool, res: Option<&Tensor>) -> Result<Tensor> {
    let (n, c_in, h, w) = x.dims4()?;
    debug_assert_eq!(n, 1, "batch 1 is the only shape this engine runs");
    let c_out = weight.dim(0)?;
    let (th, tw) = (h.div_ceil(2), w.div_ceil(2));
    let t = th * tw;

    // V: (16, Ci, T)
    let v = crate::profile::timed(|p| &p.im2col, || {
        crate::cpuop::SliceOp::new("ffai-winograd-in", move |xs, _| {
            Ok((input_transform(xs, c_in, h, w, th, tw), (16, c_in, t).into()))
        })
        .run(x)
    })?;
    // U: (16, Co, Ci), once per weight.
    let u = filter_cached(weight, c_out, c_in)?;
    // M: (16, Co, T) — sixteen GEMMs in one batched call.
    let m = crate::profile::timed(|p| &p.gemm, || crate::cpuop::matmul2_batched(&u, &v))?;
    // Y: (Co, H*W)
    let y = crate::cpuop::SliceOp::new("ffai-winograd-out", move |ms, _| {
        Ok((output_transform(ms, c_out, h, w, th, tw), (c_out, h * w).into()))
    })
    .run(&m)?;
    let bias_v = match bias {
        Some(b) => Some(crate::cpuop::bias_vec(b)?),
        None => None,
    };
    let y = match res {
        Some(r) => crate::epilogue::apply_residual(y, bias_v, h * w, act, &r.reshape((c_out, h * w))?)?,
        None => crate::epilogue::apply(y, bias_v, h * w, act)?,
    };
    y.reshape((n, c_out, h, w))
}

/// `V[ξ][ci][tile] = (Bᵀ d B)[ξ]` for every channel and 4x4 input tile (zero
/// padded by one on every side).
///
/// Structure-of-arrays over the tiles of one tile row: with each source row
/// split into its even and odd samples, the four columns of every tile are
/// four CONTIGUOUS slices over `tx` (`d[r][0] = odd[tx-1]`, `d[r][1] =
/// even[tx]`, `d[r][2] = odd[tx]`, `d[r][3] = even[tx+1]`), so the whole
/// transform is 32 slice-wide adds per tile row and vectorizes. The first
/// version gathered 4x4 scalars per tile through `Option` rows and measured
/// slower than the GEMM it displaced.
fn input_transform(xs: &[f32], c_in: usize, h: usize, w: usize, th: usize, tw: usize) -> Vec<f32> {
    let t = th * tw;
    let hw = h * w;
    // SAFETY: every (k, l, ci, ty, tx) entry is written by the loops below.
    #[allow(unsafe_code)]
    let mut v = unsafe { crate::cpuop::uninit_f32(16 * c_in * t) };
    // Per source row, padded so every tap of every tile is in bounds:
    // odd_p = [0, row[1], row[3], ...], even_p = [row[0], row[2], ..., 0].
    // Both `tw + 1` long; `tw + 1 >= ceil(w/2) + 1` covers `tx + 1 <= tw`.
    let pw = tw + 1;
    let mut odd_p = vec![0f32; h * pw];
    let mut even_p = vec![0f32; h * pw];
    let zero = vec![0f32; pw];
    let mut tmp = vec![0f32; 16 * tw];
    for ci in 0..c_in {
        let plane = &xs[ci * hw..(ci + 1) * hw];
        for (y, row) in plane.chunks_exact(w).enumerate() {
            let (o, e) = (&mut odd_p[y * pw..(y + 1) * pw], &mut even_p[y * pw..(y + 1) * pw]);
            // Only the PAD cells are zeroed; the data cells are written by the
            // split. Both rows were zero-filled in full and then overwritten
            // — two constant fills of the input per convolution.
            let half = w / 2;
            {
                let (ed, od) = (&mut e[..half], &mut o[1..=half]);
                for ((ed, od), pair) in ed.iter_mut().zip(od.iter_mut()).zip(row.chunks_exact(2)) {
                    *ed = pair[0];
                    *od = pair[1];
                }
            }
            o[0] = 0.0;
            o[half + 1..].fill(0.0);
            if w % 2 == 1 {
                e[half] = row[w - 1];
                e[half + 1..].fill(0.0);
            } else {
                e[half..].fill(0.0);
            }
        }
        for ty in 0..th {
            // The four source rows of this tile row, resolved ONCE: the odd
            // and even split of each, or the zero row above/below the image.
            let mut od: [&[f32]; 4] = [&zero; 4];
            let mut ev: [&[f32]; 4] = [&zero; 4];
            for r in 0..4 {
                let sy = 2 * ty as isize + r as isize - 1;
                if sy >= 0 && (sy as usize) < h {
                    let sy = sy as usize;
                    od[r] = &odd_p[sy * pw..(sy + 1) * pw];
                    ev[r] = &even_p[sy * pw..(sy + 1) * pw];
                }
            }
            // d[r][c] over tx: c = 0 -> odd[0..tw], 1 -> even[0..tw],
            // 2 -> odd[1..tw+1], 3 -> even[1..tw+1].
            let d = |r: usize, c: usize| -> &[f32] {
                let (src, off) = (if c % 2 == 0 { od[r] } else { ev[r] }, c / 2);
                &src[off..off + tw]
            };
            // Sixteen disjoint scratch rows, split once.
            let mut tk: [&mut [f32]; 16] = {
                let mut rest: &mut [f32] = &mut tmp[..16 * tw];
                let mut out: [&mut [f32]; 16] = [
                    &mut [], &mut [], &mut [], &mut [], &mut [], &mut [], &mut [], &mut [],
                    &mut [], &mut [], &mut [], &mut [], &mut [], &mut [], &mut [], &mut [],
                ];
                for slot in &mut out {
                    let (head, tail) = rest.split_at_mut(tw);
                    *slot = head;
                    rest = tail;
                }
                out
            };
            // tmp[k][c] = (Bᵀ d)[k][c]: rows combined, per column c.
            for c in 0..4 {
                let (d0, d1, d2, d3) = (d(0, c), d(1, c), d(2, c), d(3, c));
                // tk[k * 4 + c] for k in 0..4 — four disjoint rows of the
                // scratch, reached through one split of the slice array.
                let (lo, hi) = tk.split_at_mut(8);
                let (q0, q1) = lo.split_at_mut(4);
                let (q2, q3) = hi.split_at_mut(4);
                let (t0, t1, t2, t3) = (&mut *q0[c], &mut *q1[c], &mut *q2[c], &mut *q3[c]);
                for (((t0, t1), (t2, t3)), ((d0, d1), (d2, d3))) in t0
                    .iter_mut()
                    .zip(t1.iter_mut())
                    .zip(t2.iter_mut().zip(t3.iter_mut()))
                    .zip(d0.iter().zip(d1).zip(d2.iter().zip(d3)))
                {
                    *t0 = d0 - d2;
                    *t1 = d1 + d2;
                    *t2 = d2 - d1;
                    *t3 = d1 - d3;
                }
            }
            // V[k][l] = (tmp B)[k][l]: columns combined, per row k.
            let s = c_in * t;
            let base = ci * t + ty * tw;
            for k in 0..4 {
                let (c0, c1, c2, c3) = (&*tk[k * 4], &*tk[k * 4 + 1], &*tk[k * 4 + 2], &*tk[k * 4 + 3]);
                let (o0, rest) = v[(k * 4) * s + base..].split_at_mut(s);
                let (o1, rest) = rest.split_at_mut(s);
                let (o2, o3) = rest.split_at_mut(s);
                for (((o0, o1), (o2, o3)), ((c0, c1), (c2, c3))) in o0[..tw]
                    .iter_mut()
                    .zip(o1[..tw].iter_mut())
                    .zip(o2[..tw].iter_mut().zip(o3[..tw].iter_mut()))
                    .zip(c0.iter().zip(c1).zip(c2.iter().zip(c3)))
                {
                    *o0 = c0 - c2;
                    *o1 = c1 + c2;
                    *o2 = c2 - c1;
                    *o3 = c1 - c3;
                }
            }
        }
    }
    v
}

/// `Y = Aᵀ M A` per (co, tile), written into `(Co, H, W)`.
///
/// Same structure-of-arrays: for one (co, ty) the sixteen `M[ξ]` are
/// contiguous slices over `tx`, the 2x2 results are four slices, and each
/// output row is the interleave of two of them.
fn output_transform(ms: &[f32], c_out: usize, h: usize, w: usize, th: usize, tw: usize) -> Vec<f32> {
    let t = th * tw;
    let hw = h * w;
    // SAFETY: every (co, oy, ox) is written by the interleave below (rows
    // `oy < h`, the paired columns and the odd tail).
    #[allow(unsafe_code)]
    let mut out = unsafe { crate::cpuop::uninit_f32(c_out * hw) };
    let s = c_out * t;
    let mut tmp = vec![0f32; 8 * tw];
    let mut y = vec![0f32; 4 * tw];
    for co in 0..c_out {
        let o = &mut out[co * hw..(co + 1) * hw];
        for ty in 0..th {
            let base = co * t + ty * tw;
            let m = |k: usize, c: usize| -> &[f32] { &ms[(k * 4 + c) * s + base..(k * 4 + c) * s + base + tw] };
            // tmp[r][c] = (Aᵀ m)[r][c]
            for c in 0..4 {
                let (m0, m1, m2, m3) = (m(0, c), m(1, c), m(2, c), m(3, c));
                let (a, b) = tmp.split_at_mut(4 * tw);
                let (a, b) = (&mut a[c * tw..(c + 1) * tw], &mut b[c * tw..(c + 1) * tw]);
                for x in 0..tw {
                    a[x] = m0[x] + m1[x] + m2[x];
                    b[x] = m1[x] - m2[x] - m3[x];
                }
            }
            // y[r][l] = (tmp A)[r][l]
            for r in 0..2 {
                let tr = &tmp[r * 4 * tw..(r + 1) * 4 * tw];
                let (c0, c1, c2, c3) = (&tr[..tw], &tr[tw..2 * tw], &tr[2 * tw..3 * tw], &tr[3 * tw..4 * tw]);
                let (y0, y1) = y[r * 2 * tw..(r + 1) * 2 * tw].split_at_mut(tw);
                for x in 0..tw {
                    y0[x] = c0[x] + c1[x] + c2[x];
                    y1[x] = c1[x] - c2[x] - c3[x];
                }
            }
            // Interleave into the two output rows (the second may not exist
            // for odd `h`; the last column may not for odd `w`).
            for r in 0..2 {
                let oy = 2 * ty + r;
                if oy >= h {
                    break;
                }
                let row = &mut o[oy * w..(oy + 1) * w];
                let (y0, y1) = (&y[r * 2 * tw..r * 2 * tw + tw], &y[r * 2 * tw + tw..(r + 1) * 2 * tw]);
                let full = w / 2;
                for (x, pair) in row.chunks_exact_mut(2).enumerate().take(full) {
                    pair[0] = y0[x];
                    pair[1] = y1[x];
                }
                if w % 2 == 1 {
                    row[w - 1] = y0[full];
                }
            }
        }
    }
    out
}

/// `U = G g Gᵀ` for every (co, ci), laid out `(16, Co, Ci)`, cached per weight.
fn filter_cached(weight: &Tensor, c_out: usize, c_in: usize) -> Result<Tensor> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static CACHE: Mutex<Option<HashMap<candle_core::TensorId, Tensor>>> = Mutex::new(None);

    let key = weight.id();
    let mut g = CACHE.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(t) = map.get(&key) {
        return Ok(t.clone());
    }
    let ws = crate::cpuop::bias_vec(weight)?;
    let mut u = vec![0f32; 16 * c_out * c_in];
    let s = c_out * c_in;
    for co in 0..c_out {
        for ci in 0..c_in {
            let g9 = &ws[(co * c_in + ci) * 9..(co * c_in + ci) * 9 + 9];
            let gm = [[g9[0], g9[1], g9[2]], [g9[3], g9[4], g9[5]], [g9[6], g9[7], g9[8]]];
            // tmp = G g (4x3)
            let mut tmp = [[0f32; 3]; 4];
            for c in 0..3 {
                tmp[0][c] = gm[0][c];
                tmp[1][c] = 0.5 * (gm[0][c] + gm[1][c] + gm[2][c]);
                tmp[2][c] = 0.5 * (gm[0][c] - gm[1][c] + gm[2][c]);
                tmp[3][c] = gm[2][c];
            }
            // u = tmp Gᵀ (4x4)
            for r in 0..4 {
                let (a, b, c) = (tmp[r][0], tmp[r][1], tmp[r][2]);
                let row = [a, 0.5 * (a + b + c), 0.5 * (a - b + c), c];
                for (col, &val) in row.iter().enumerate() {
                    u[(r * 4 + col) * s + co * c_in + ci] = val;
                }
            }
        }
    }
    let t = Tensor::from_vec(u, (16, c_out, c_in), weight.device())?;
    map.insert(key, t.clone());
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The GEMM path is the oracle: Winograd must agree to a tolerance that
    /// is tight for f32 reassociation and loose for nothing else.
    #[test]
    fn matches_the_gemm_path_within_tolerance() {
        let dev = Device::Cpu;
        for &(c_in, c_out, h, w) in &[(96, 128, 9, 11), (128, 96, 8, 8), (100, 100, 1, 1), (96, 96, 3, 2)] {
            let x = Tensor::randn(0f32, 1.0, (1, c_in, h, w), &dev).unwrap();
            let wt = (Tensor::randn(0f32, 1.0, (c_out, c_in, 3, 3), &dev).unwrap() * 0.1).unwrap();
            let b = Tensor::randn(0f32, 1.0, c_out, &dev).unwrap();
            for act in [false, true] {
                let want = crate::conv3x3::conv3x3_gemm(&x, &wt, Some(&b), 1, act).unwrap();
                let got = conv3x3_winograd(&x, &wt, Some(&b), act, None).unwrap();
                assert_eq!(got.dims(), want.dims());
                let (g, wv) = (
                    got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                );
                let mut worst = 0f32;
                for (a, b) in g.iter().zip(&wv) {
                    worst = worst.max((a - b).abs() / b.abs().max(1.0));
                }
                assert!(worst < 2e-5, "ci={c_in} co={c_out} {h}x{w} act={act}: worst rel diff {worst:e}");
            }
        }
    }
}
