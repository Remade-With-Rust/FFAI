//! Cache-blocked conv kernels for the HiFi-GAN decoder — the M-T3 speed
//! campaign's first brick.
//!
//! WHY (profiled, not assumed — `examples/profile_tts.rs`): the decoder is
//! 82.7 % of synthesis, and candle's conv1d DEGRADES with length (99 GF/s at
//! L=1.5k → 27 GF/s at L=48.6k) because its working set falls out of cache;
//! im2col+matmul is no better (the gathered matrix is ~40 MB per call at the
//! longest stage). The remedy is locality, not a different GEMM: tile the
//! time axis into blocks whose input slab stays L2-resident, keep a per-
//! output-channel accumulator hot, and let the compiler vectorize the
//! innermost contiguous-time FMA. Rayon parallelizes across time blocks —
//! blocks are independent, so this also restores the multi-core scaling the
//! long tail was losing.
//!
//! Gate: candle's own conv is the scalar twin. `conv1d_direct` must match it
//! to float-reassociation tolerance on every shape the decoder uses
//! (`kernels_match_candle`), and `FFAI_CANDLE_CONV=1` switches the engine
//! back for A/B. `ConvTranspose` is decomposed by output phase — with stride s
//! and kernel k, each output sample touches exactly k/s taps, so the
//! upsamplers cost k/s MACs per output instead of k.

//! Cast policy (gate H-15): `cast_possible_truncation`, `cast_sign_loss` and
//! `cast_possible_wrap` are allowed in this module. Every value converted here
//! is a MODEL-INTERNAL dimension, index or accumulator - bounded by weights the
//! loader has already validated - not a number read from caller input. The lint
//! stays DENIED in the untrusted-surface modules (`mel`, `fbank`, `onnx`,
//! `normalize`, `lexicon`, `chunk`, `phonemize`, `phoneme_ids`), which is where
//! this audit's arithmetic defects were actually found.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use crate::par::prelude::*;
use ffai_core::candle::{Device, Tensor};
use ffai_core::error::{Error, Result};

fn e<T>(r: ffai_core::candle::Result<T>) -> Result<T> {
    r.map_err(|err| Error::Model(format!("decoder kernel: {err}")))
}

/// Direct conv1d, `[1, C_in, L] -> [1, C_out, L_out]`, groups = 1.
/// Same-geometry replacement for `Tensor::conv1d` on the decoder's shapes.
pub fn conv1d_direct(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    pad: usize,
    dilation: usize,
) -> Result<Tensor> {
    let (_, c_in, len) = e(x.dims3())?;
    let (c_out, _, k) = e(w.dims3())?;
    let xv: Vec<f32> = e(e(x.flatten_all())?.to_vec1())?;
    let wv: Vec<f32> = e(e(w.flatten_all())?.to_vec1())?;
    let bv: Option<Vec<f32>> = match bias {
        Some(b) => Some(e(e(b.flatten_all())?.to_vec1())?),
        None => None,
    };
    let (out, l_out) = conv1d_flat(&xv, c_in, len, &wv, bv.as_deref(), c_out, k, pad, dilation);
    e(Tensor::from_vec(out, (1, c_out, l_out), &Device::Cpu))
}

/// The flat core: channel-major `&[f32]` in, channel-major `Vec<f32>` out.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv1d_flat(
    xv: &[f32],
    c_in: usize,
    len: usize,
    wv: &[f32],
    bv: Option<&[f32]>,
    c_out: usize,
    k: usize,
    pad: usize,
    dilation: usize,
) -> (Vec<f32>, usize) {
    let l_out = len + 2 * pad - dilation * (k - 1);
    let mut out = vec![0f32; c_out * l_out];
    conv1d_flat_into(xv, c_in, len, wv, bv, c_out, k, pad, dilation, &mut out);
    (out, l_out)
}

/// The output region task `task` owns: rows `co0..co1`, columns `t0..t1`.
///
/// Extracted so the SAME arithmetic the kernel runs can be PROVED disjoint
/// (`src/proofs.rs`, gate H-30) instead of argued in a comment. `UNSAFE.md`
/// class B's soundness — the `unsafe impl Send`/`Sync` on `SendPtr` — is
/// exactly the claim that this returns non-overlapping regions for distinct
/// tasks, and in-bounds regions for every task.
///
/// Chunks partition `[0, c_out)` and blocks partition `[0, l_out)`, so two
/// tasks differ in at least one axis and therefore never address the same
/// element. Change this and the `unsafe impl` becomes unsound.
#[inline]
pub(crate) fn task_region(
    task: usize,
    n_blocks: usize,
    co_chunk: usize,
    block_t: usize,
    c_out: usize,
    l_out: usize,
) -> (usize, usize, usize, usize) {
    let (chunk, b) = (task / n_blocks, task % n_blocks);
    let co0 = chunk * co_chunk;
    let co1 = (co0 + co_chunk).min(c_out);
    let t0 = b * block_t;
    let t1 = (t0 + block_t).min(l_out);
    (co0, co1, t0, t1)
}

/// As [`conv1d_flat`], but ACCUMULATES into `out` (which already holds the
/// residual): fuses the resblock's `y += conv(...)` add into the conv's
/// write pass — one less allocation and one less full-buffer read/write per
/// conv, six per upsample stage.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_flat_into(
    xv: &[f32],
    c_in: usize,
    len: usize,
    wv: &[f32],
    bv: Option<&[f32]>,
    c_out: usize,
    k: usize,
    pad: usize,
    dilation: usize,
    out: &mut [f32],
) {
    conv1d_flat_prepacked_into(xv, c_in, len, wv, None, bv, c_out, k, pad, dilation, out);
}

/// Quad-major weight repack: [co/4][ci][j][4 lanes].
fn pack_quads(wv: &[f32], c_out: usize, c_in: usize, k: usize) -> Vec<f32> {
    let quads = c_out / 4;
    let mut wp = vec![0f32; quads * c_in * k * 4];
    for q in 0..quads {
        for ci in 0..c_in {
            for j in 0..k {
                for lane in 0..4 {
                    wp[((q * c_in + ci) * k + j) * 4 + lane] =
                        wv[((q * 4 + lane) * c_in + ci) * k + j];
                }
            }
        }
    }
    wp
}

#[allow(clippy::too_many_arguments)]
fn conv1d_flat_prepacked_into(
    xv: &[f32],
    c_in: usize,
    len: usize,
    wv: &[f32],
    prepacked: Option<&[f32]>,
    bv: Option<&[f32]>,
    c_out: usize,
    k: usize,
    pad: usize,
    dilation: usize,
    out: &mut [f32],
) {
    let l_out = len + 2 * pad - dilation * (k - 1);
    debug_assert_eq!(out.len(), c_out * l_out);

    // Size-adaptive: below this work threshold the rayon grid costs more
    // than it buys (routing the flow's L≈190 convs through the parallel
    // path measured 1.4× SLOWER than candle — this serial path is the fix,
    // not an exception list). One thread, same FMA loops, no tiles.
    if c_out * c_in * k * l_out < 4_000_000 {
        for co in 0..c_out {
            let acc = &mut out[co * l_out..(co + 1) * l_out];
            if let Some(bv) = bv {
                acc.iter_mut().for_each(|a| *a += bv[co]);
            }
            for ci in 0..c_in {
                let xrow = &xv[ci * len..(ci + 1) * len];
                for j in 0..k {
                    let off = j * dilation;
                    let lo = pad.saturating_sub(off);
                    let hi = l_out.min((len + pad).saturating_sub(off));
                    if lo >= hi {
                        continue;
                    }
                    let wcoef = wv[(co * c_in + ci) * k + j];
                    let src = &xrow[lo + off - pad..hi + off - pad];
                    let dst = &mut acc[lo..hi];
                    for (a, &s) in dst.iter_mut().zip(src) {
                        *a += wcoef * s;
                    }
                }
            }
        }
        return;
    }

    // 2-D task grid: (output-channel chunk × time block). Time blocks alone
    // starve the cores at short lengths (L=1520 is two blocks — measured 3×
    // SLOWER than candle before this); channel chunks alone lose locality at
    // long lengths. The grid gives ≥16 tasks at every decoder shape.
    // Adaptive channel-chunking: aim for ~8 tasks per time block, floor 16
    // channels. Decoder shapes (c_out ≤ 256) keep their 16-channel chunks;
    // flow-scale shapes (c_out 384 at L≈190) get 48-channel chunks — 8 fat
    // tasks instead of 24 thin ones, which under CPU contention stopped the
    // fork-join from being dominated by preempted stragglers. Chunks stay
    // multiples of 4 so the packed quad kernel always applies.
    let co_chunk = (c_out.div_ceil(8).next_multiple_of(4)).max(16);
    // Adaptive time blocks: size each block's input slab to ~256 KiB so it
    // is L2-resident regardless of channel count. The fixed 1024 block made
    // 32-channel stage-3 slabs only 128 KiB — twice as many task boundaries
    // and strip tails as the cache required.
    let block_t = ((64 * 1024) / c_in).clamp(1024, 8192).next_multiple_of(32);
    let n_blocks = l_out.div_ceil(block_t);
    let n_chunks = c_out.div_ceil(co_chunk);

    // Quad-major packed weights (the MLAS trick; ORT's packed convs are the
    // measured 2.2× existence proof): prefer the caller's once-packed copy;
    // pack per call only on the wrapper path where no cache exists.
    let packed_local;
    let wp: &[f32] = if let Some(p) = prepacked {
        p
    } else {
        packed_local = pack_quads(wv, c_out, c_in, k);
        &packed_local
    };

    // Tasks write DIRECTLY into `out`: each (channel-chunk × time-block)
    // task owns a disjoint region by construction, so the tile buffers and
    // the whole assembly += pass (one full extra read+write of the output
    // per conv) are deleted. SAFETY rests on that disjointness: chunk
    // co-ranges partition [0, c_out) and block t-ranges partition
    // [0, l_out); no two tasks touch the same (co, t).
    struct SendPtr(*mut f32);
    // SAFETY: `SendPtr` carries a raw pointer across rayon tasks. Sending it is
    // sound ONLY because the task decomposition below gives every task a disjoint
    // (co, t) region: chunk co-ranges partition [0, c_out) and block t-ranges
    // partition [0, l_out), so no two tasks ever address the same element. If that
    // index arithmetic changes, this impl becomes unsound - see UNSAFE.md class B.
    unsafe impl Send for SendPtr {}
    unsafe impl Sync for SendPtr {}
    let out_ptr = SendPtr(out.as_mut_ptr());

    (0..n_chunks * n_blocks).into_par_iter().for_each(|task| {
        let out_ptr = &out_ptr;
        let (co0, co1, t0, t1) = task_region(task, n_blocks, co_chunk, block_t, c_out, l_out);
        let bt = t1 - t0;
        // This task's disjoint view: rows co0..co1, columns t0..t1.
        // Reconstructed as raw slices per row.
        let row = |co: usize| -> &mut [f32] {
            // SAFETY: disjoint (co, t) regions per task, see above.
            unsafe { std::slice::from_raw_parts_mut(out_ptr.0.add(co * l_out + t0), bt) }
        };
        // Scratch tile, zero-initialized: bias joins in the merge pass.
        let mut tile = vec![0f32; (co1 - co0) * bt];
        // Interior region where every tap's input index is in-range —
        // eligible for the AVX2 micro-kernel; edges take the safe path.
        let span = dilation * (k - 1);
        let int_lo = t0.max(pad);
        let int_hi = t1.min((len + pad).saturating_sub(span).max(pad));
        #[cfg(target_arch = "x86_64")]
        let use_avx = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && int_hi > int_lo;
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx = false;

        // 4-wide output-channel register block: each src load feeds four
        // FMA streams instead of one (the load was the bottleneck — one
        // load per FMA caps at a quarter of FMA throughput).
        let mut co_i = 0;
        while co_i < co1 - co0 {
            let quad = (co1 - co0 - co_i).min(4);
            #[cfg(target_arch = "x86_64")]
            if use_avx && quad == 4 {
                // SAFETY: dispatched on runtime avx2+fma detection; all
                // slice geometry is checked by the kernel's caller
                // contract (interior region only).
                unsafe {
                    let quad_base = ((co0 + co_i) / 4) * c_in * k * 4;
                    conv_quad_avx2(
                        xv,
                        len,
                        &wp[quad_base..quad_base + c_in * k * 4],
                        &mut tile[co_i * bt..(co_i + 4) * bt],
                        bt,
                        c_in,
                        k,
                        pad,
                        dilation,
                        t0,
                        int_lo,
                        int_hi,
                    );
                }
                // Edges (t < int_lo or t >= int_hi) via the safe path.
                for (edge_lo, edge_hi) in [(t0, int_lo), (int_hi, t1)] {
                    if edge_lo >= edge_hi {
                        continue;
                    }
                    for q in 0..4 {
                        let co = co0 + co_i + q;
                        for ci in 0..c_in {
                            let xrow = &xv[ci * len..(ci + 1) * len];
                            for j in 0..k {
                                let off = j * dilation;
                                let lo = edge_lo.max(pad.saturating_sub(off));
                                let hi = edge_hi.min(len + pad - off);
                                if lo >= hi {
                                    continue;
                                }
                                let wcoef = wv[(co * c_in + ci) * k + j];
                                let src = &xrow[lo + off - pad..hi + off - pad];
                                let dst =
                                    &mut tile[(co_i + q) * bt + lo - t0..(co_i + q) * bt + hi - t0];
                                for (a, &s) in dst.iter_mut().zip(src) {
                                    *a += wcoef * s;
                                }
                            }
                        }
                    }
                }
                co_i += 4;
                continue;
            }
            for ci in 0..c_in {
                let xrow = &xv[ci * len..(ci + 1) * len];
                for j in 0..k {
                    let off = j * dilation;
                    let lo = t0.max(pad.saturating_sub(off));
                    let hi = t1.min(len + pad - off);
                    if lo >= hi {
                        continue;
                    }
                    let src = &xrow[lo + off - pad..hi + off - pad];
                    let n = hi - lo;
                    let base = lo - t0;
                    if quad == 4 {
                        let w0 = wv[((co0 + co_i) * c_in + ci) * k + j];
                        let w1 = wv[((co0 + co_i + 1) * c_in + ci) * k + j];
                        let w2 = wv[((co0 + co_i + 2) * c_in + ci) * k + j];
                        let w3 = wv[((co0 + co_i + 3) * c_in + ci) * k + j];
                        // Split tile into four disjoint rows.
                        let (r0, rest) = tile[co_i * bt..].split_at_mut(bt);
                        let (r1, rest) = rest.split_at_mut(bt);
                        let (r2, r3) = rest.split_at_mut(bt);
                        let (d0, d1) = (&mut r0[base..base + n], &mut r1[base..base + n]);
                        let (d2, d3) = (&mut r2[base..base + n], &mut r3[base..base + n]);
                        for t in 0..n {
                            let s = src[t];
                            d0[t] += w0 * s;
                            d1[t] += w1 * s;
                            d2[t] += w2 * s;
                            d3[t] += w3 * s;
                        }
                    } else {
                        for q in 0..quad {
                            let wcoef = wv[((co0 + co_i + q) * c_in + ci) * k + j];
                            let dst = &mut tile[(co_i + q) * bt + base..(co_i + q) * bt + base + n];
                            for (a, &s) in dst.iter_mut().zip(src) {
                                *a += wcoef * s;
                            }
                        }
                    }
                }
            }
            co_i += quad;
        }
        // Merge into this task's disjoint region of `out`, in-task
        // (parallel, tile still cache-hot), bias folded into the pass.
        for (co_i, co) in (co0..co1).enumerate() {
            let dst = row(co);
            let src = &tile[co_i * bt..(co_i + 1) * bt];
            let bias = bv.map_or(0.0, |b| b[co]);
            for (o, &v) in dst.iter_mut().zip(src) {
                *o += v + bias;
            }
        }
    });
}

/// `ConvTranspose1d` by phase decomposition, `[1, C_in, L] -> [1, C_out, L*s]`
/// (stride s, kernel k, `pad` as exported). Each output phase p uses only
/// the taps `j ≡ (p + pad) mod s`, i.e. k/s taps per output sample.
pub fn conv_transpose1d_direct(
    x: &Tensor,
    w: &Tensor, // [C_in, C_out, K] (PyTorch layout)
    bias: Option<&Tensor>,
    pad: usize,
    stride: usize,
) -> Result<Tensor> {
    let (_, c_in, len) = e(x.dims3())?;
    let (_, c_out, k) = e(w.dims3())?;
    let xv: Vec<f32> = e(e(x.flatten_all())?.to_vec1())?;
    let wv: Vec<f32> = e(e(w.flatten_all())?.to_vec1())?;
    let bv: Option<Vec<f32>> = match bias {
        Some(b) => Some(e(e(b.flatten_all())?.to_vec1())?),
        None => None,
    };
    let (out, l_out) =
        conv_transpose1d_flat(&xv, c_in, len, &wv, bv.as_deref(), c_out, k, pad, stride);
    e(Tensor::from_vec(out, (1, c_out, l_out), &Device::Cpu))
}

/// Flat transpose core, phase-decomposed for CONTIGUOUS inner loops.
///
/// out[co][t] = `Σ_ci` `Σ_j` w[ci][co][j]·x[ci][i] where i·s + j − pad = t.
/// Fix the output phase p = t mod s: the contributing taps are exactly the
/// j with (j − pad) ≡ p (mod s) — k/s of them — and for each such j the map
/// t → i is t = n·s + p, i = n + (p + pad − j)/s: a CONTIGUOUS run over n.
/// Each (phase, tap) pair is one scalar×vector FMA; phases interleave at the
/// end. (The first version iterated t with stride-s writes — unvectorizable
/// and cache-hostile.)
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv_transpose1d_flat(
    xv: &[f32],
    c_in: usize,
    len: usize,
    wv: &[f32],
    bv: Option<&[f32]>,
    c_out: usize,
    k: usize,
    pad: usize,
    stride: usize,
) -> (Vec<f32>, usize) {
    let l_out = (len - 1) * stride + k - 2 * pad;
    let out: Vec<Vec<f32>> = (0..c_out)
        .into_par_iter()
        .map(|co| {
            let mut row = vec![bv.map_or(0.0, |b| b[co]); l_out];
            let mut phase_acc = vec![0f32; l_out / stride + 1];
            for phase in 0..stride {
                // Output positions t = phase, phase+s, ... : n_phase of them.
                let n_phase = (l_out + stride - 1 - phase) / stride;
                phase_acc[..n_phase].iter_mut().for_each(|a| *a = 0.0);
                for ci in 0..c_in {
                    let xrow = &xv[ci * len..(ci + 1) * len];
                    let wrow = &wv[(ci * c_out + co) * k..(ci * c_out + co + 1) * k];
                    for j in 0..k {
                        if (j + stride * k).wrapping_sub(pad) % stride != phase % stride {
                            continue;
                        }
                        let wcoef = wrow[j];
                        // i = n + d with d = (phase + pad − j)/s (integer by
                        // the phase condition; may be negative).
                        let d = (phase as isize + pad as isize - j as isize) / stride as isize;
                        let n_lo = (-d).max(0) as usize;
                        let n_hi = n_phase.min(((len as isize) - d).max(0) as usize);
                        if n_lo >= n_hi {
                            continue;
                        }
                        let src = &xrow[(n_lo as isize + d) as usize..(n_hi as isize + d) as usize];
                        let dst = &mut phase_acc[n_lo..n_hi];
                        for (a, &s) in dst.iter_mut().zip(src) {
                            *a += wcoef * s;
                        }
                    }
                }
                for (n, &v) in phase_acc[..n_phase].iter().enumerate() {
                    row[n * stride + phase] += v;
                }
            }
            row
        })
        .collect();

    let mut flat = Vec::with_capacity(c_out * l_out);
    for row in out {
        flat.extend(row);
    }
    (flat, l_out)
}

// ---------------------------------------------------------------------------
// The flat decoder: the whole HiFi-GAN chain in one Vec<f32> domain.
// ---------------------------------------------------------------------------

/// One conv's weights, flattened once at load. Eliminates the per-call
/// tensor↔Vec copies that the eliminate-redundancy pass found costing more
/// than the arithmetic they wrapped.
struct FlatConv {
    w: Vec<f32>,
    /// Quad-packed weights ([co/4][ci][j][4]), built ONCE here. Per-call
    /// packing was fine for the decoder (weights tiny vs work) and HALF the
    /// runtime at flow shapes (1.5 MB of weights vs 70 M MACs) — the flow
    /// v4 loss was the packing, not the kernel.
    wp: Vec<f32>,
    b: Option<Vec<f32>>,
    c_in: usize,
    c_out: usize,
    k: usize,
    pad: usize,
    stride: usize,
    dilation: usize,
}

impl FlatConv {
    fn new(
        w: Vec<f32>,
        b: Option<Vec<f32>>,
        c_in: usize,
        c_out: usize,
        k: usize,
        pad: usize,
        stride: usize,
        dilation: usize,
    ) -> Self {
        let wp = pack_quads(&w, c_out, c_in, k);
        Self {
            w,
            wp,
            b,
            c_in,
            c_out,
            k,
            pad,
            stride,
            dilation,
        }
    }

    fn conv(&self, x: &[f32], len: usize) -> (Vec<f32>, usize) {
        let l_out = len + 2 * self.pad - self.dilation * (self.k - 1);
        let mut out = vec![0f32; self.c_out * l_out];
        self.conv_into(x, len, &mut out);
        (out, l_out)
    }

    /// Accumulate `out += conv(x)` — same-pad convs only (`l_out` == len).
    fn conv_into(&self, x: &[f32], len: usize, out: &mut [f32]) {
        conv1d_flat_prepacked_into(
            x,
            self.c_in,
            len,
            &self.w,
            Some(&self.wp),
            self.b.as_deref(),
            self.c_out,
            self.k,
            self.pad,
            self.dilation,
            out,
        );
    }

    fn transpose(&self, x: &[f32], len: usize) -> (Vec<f32>, usize) {
        conv_transpose1d_flat(
            x,
            self.c_in,
            len,
            &self.w,
            self.b.as_deref(),
            self.c_out,
            self.k,
            self.pad,
            self.stride,
        )
    }
}

/// piper's compact HiFi-GAN, flat: `conv_pre` → 3×(leaky → up → resblock
/// average) → leaky(0.01) → `conv_post` → tanh. Activations are single passes;
/// tensors exist only at the boundary.
pub struct FlatDecoder {
    conv_pre: FlatConv,
    ups: Vec<FlatConv>,
    /// 9 resblocks × 2 convs, grouped 3 per upsample stage.
    resblocks: Vec<[FlatConv; 2]>,
    conv_post: FlatConv,
}

impl FlatDecoder {
    /// Build from the loaded weight map + geometry (called once at load).
    pub fn from_weights(
        get: &dyn Fn(&str) -> Result<(Vec<f32>, Vec<usize>)>,
        has_bias: &dyn Fn(&str) -> bool,
        geom: &dyn Fn(&str) -> (usize, usize, usize), // (pad, stride, dilation)
    ) -> Result<Self> {
        let load = |name: &str, transpose: bool| -> Result<FlatConv> {
            let (w, dims) = get(&format!("{name}.weight"))?;
            let b = if has_bias(name) {
                Some(get(&format!("{name}.bias"))?.0)
            } else {
                None
            };
            let (pad, stride, dilation) = geom(name);
            let (c_in, c_out, k) = if transpose {
                (dims[0], dims[1], dims[2])
            } else {
                (dims[1], dims[0], dims[2])
            };
            Ok(FlatConv::new(w, b, c_in, c_out, k, pad, stride, dilation))
        };
        let mut ups = Vec::new();
        for i in 0..3 {
            ups.push(load(&format!("dec.ups.{i}"), true)?);
        }
        let mut resblocks = Vec::new();
        for i in 0..9 {
            resblocks.push([
                load(&format!("dec.resblocks.{i}.convs.0"), false)?,
                load(&format!("dec.resblocks.{i}.convs.1"), false)?,
            ]);
        }
        let mut conv_post = load("dec.conv_post", false)?;
        // The resblock average (xs/3 per stage) is FOLDED into the next
        // linear op's weights: leaky is positively homogeneous, so
        // next(leaky(x/3)) == next_scaled(leaky(x)) exactly, and three full
        // passes over multi-MB buffers disappear. Biases stay unscaled (they
        // apply after the matmul).
        // Both weight copies scale together — a stale packed copy after the
        // fold would be a silent wrong-audio bug the oracle would catch, but
        // better never to create it.
        for i in [1usize, 2] {
            ups[i].w.iter_mut().for_each(|w| *w /= 3.0);
            ups[i].wp.iter_mut().for_each(|w| *w /= 3.0);
        }
        conv_post.w.iter_mut().for_each(|w| *w /= 3.0);
        conv_post.wp.iter_mut().for_each(|w| *w /= 3.0);
        Ok(Self {
            conv_pre: load("dec.conv_pre", false)?,
            ups,
            resblocks,
            conv_post,
        })
    }

    /// z (channel-major, 192×len) → waveform. `FFAI_PROFILE=1` prints a
    /// per-op breakdown (the ASR convention).
    #[must_use]
    pub fn run(&self, z: &[f32], len: usize) -> Vec<f32> {
        let profile = std::env::var("FFAI_PROFILE").is_ok();
        let mut t_up = 0f64;
        let mut t_conv = 0f64;
        let mut t_glue = 0f64;
        // Per-upsampling-stage buckets. The decoder upsamples 256x overall, so
        // the three stages operate on wildly different tensor sizes and a
        // single lumped total cannot say which one owns the cost.
        let mut stage_ms = [0f64; 3];
        let mut stage_len = [0usize; 3];
        // Glue is ~22% of the decoder and is three different things wearing one
        // name: a per-resblock buffer clone (alloc + copy of a multi-MB
        // tensor), the activation passes, and the accumulate. They have
        // different fixes, so they need different buckets.
        let mut t_clone = 0f64;
        let mut t_act = 0f64;
        let mut t_add = 0f64;
        let clock = crate::clock::Instant::now;

        let t0 = clock();
        let (mut x, mut len) = self.conv_pre.conv(z, len);
        t_conv += t0.elapsed().as_secs_f64();

        // MEASURED AND REVERTED: hoisting these scratch buffers out of the
        // stage/resblock loops (to kill the 15 multi-MB allocations per call --
        // 9 `x.clone()` residuals + 3 `lx` + 3 `scratch`) did NOT pay. The
        // whole-pipeline paired A/B read 11/21, z = +0.22, and the per-op
        // buckets explained why: `clone` fell 159 -> ~100 ms but `act` ROSE
        // 211 -> ~250 ms, because the cost moved into the `resize` first-touch
        // rather than disappearing. Reverted as unproven, not as impossible --
        // an arena that persists ACROSS decode() calls (rather than within one)
        // would not pay the first-touch cost and is still worth trying.
        for (up_i, up) in self.ups.iter().enumerate() {
            let stage_t0 = clock();
            let t0 = clock();
            leaky_inplace(&mut x, 0.1);
            let d = t0.elapsed().as_secs_f64();
            t_glue += d;
            t_act += d;
            let t0 = clock();
            let (xu, lu) = up.transpose(&x, len);
            t_up += t0.elapsed().as_secs_f64();
            x = xu;
            len = lu;
            let mut acc: Option<Vec<f32>> = None;
            // Scratches reused across the stage — no per-conv allocation.
            // All three resblocks read leaky(x) as their FIRST conv's input,
            // so it is computed once per stage, not three times (two full
            // multi-MB passes eliminated, bit-exact).
            let t0 = clock();
            let mut lx = vec![0f32; x.len()];
            leaky_into(&x, &mut lx, 0.1);
            let mut scratch = vec![0f32; x.len()];
            let d = t0.elapsed().as_secs_f64();
            t_glue += d;
            t_act += d;
            for rb in &self.resblocks[up_i * 3..up_i * 3 + 3] {
                let t0 = clock();
                let mut y = x.clone();
                let d = t0.elapsed().as_secs_f64();
                t_glue += d;
                t_clone += d;
                let t0 = clock();
                // Fused: y += conv0(leaky(x)) in the conv's write pass.
                rb[0].conv_into(&lx, len, &mut y);
                t_conv += t0.elapsed().as_secs_f64();
                let t0 = clock();
                leaky_into(&y, &mut scratch, 0.1);
                let d = t0.elapsed().as_secs_f64();
                t_glue += d;
                t_act += d;
                let t0 = clock();
                rb[1].conv_into(&scratch, len, &mut y);
                t_conv += t0.elapsed().as_secs_f64();
                let t0 = clock();
                match &mut acc {
                    Some(a) => add_inplace(a, &y),
                    None => acc = Some(y),
                }
                let d = t0.elapsed().as_secs_f64();
                t_glue += d;
                t_add += d;
            }
            // No /3 here: the resblock average is folded into the next
            // conv's weights (see from_weights).
            x = acc.expect("3 resblocks");
            stage_ms[up_i] = stage_t0.elapsed().as_secs_f64();
            stage_len[up_i] = len;
        }
        let t0 = clock();
        leaky_inplace(&mut x, 0.01);
        let (mut audio, _) = self.conv_post.conv(&x, len);
        // Scalar `tanh` per sample was a libm call per element; the shared
        // kernel is arithmetic, so the loop can vectorise.
        for a in &mut audio {
            *a = ffai_core::fastmath::tanh(*a);
        }
        t_conv += t0.elapsed().as_secs_f64();
        if profile {
            eprintln!(
                "[dec] conv {:.1} ms  ups {:.1} ms  glue(clone/act/add) {:.1} ms  \
                 | clone {:.1} act {:.1} add {:.1}                  | stage0 {:.1} ms (L={})  stage1 {:.1} ms (L={})  stage2 {:.1} ms (L={})",
                t_conv * 1000.0,
                t_up * 1000.0,
                t_glue * 1000.0,
                t_clone * 1000.0,
                t_act * 1000.0,
                t_add * 1000.0,
                stage_ms[0] * 1000.0,
                stage_len[0],
                stage_ms[1] * 1000.0,
                stage_len[1],
                stage_ms[2] * 1000.0,
                stage_len[2],
            );
        }
        audio
    }
}

/// Depthwise conv1d (`groups == C`): each channel convolves independently
/// with its own k-tap filter — the duration predictor's separable convs.
/// Serial: the whole op is ~C·k·L ≈ 100k MACs.
#[must_use]
pub fn conv1d_depthwise_flat(
    xv: &[f32],
    channels: usize,
    len: usize,
    wv: &[f32], // [C, 1, K]
    bv: Option<&[f32]>,
    k: usize,
    pad: usize,
    dilation: usize,
) -> (Vec<f32>, usize) {
    let l_out = len + 2 * pad - dilation * (k - 1);
    let mut out = vec![0f32; channels * l_out];
    for c in 0..channels {
        let acc = &mut out[c * l_out..(c + 1) * l_out];
        if let Some(bv) = bv {
            acc.iter_mut().for_each(|a| *a = bv[c]);
        }
        let xrow = &xv[c * len..(c + 1) * len];
        for j in 0..k {
            let off = j * dilation;
            let lo = pad.saturating_sub(off);
            let hi = l_out.min((len + pad).saturating_sub(off));
            if lo >= hi {
                continue;
            }
            let wcoef = wv[c * k + j];
            let src = &xrow[lo + off - pad..hi + off - pad];
            for (a, &s) in acc[lo..hi].iter_mut().zip(src) {
                *a += wcoef * s;
            }
        }
    }
    (out, l_out)
}

/// The AVX2+FMA conv micro-kernel: four output channels × 32 output lanes
/// held in 16 ymm accumulators across the whole (`c_in` × k) reduction — one
/// broadcast + four loads feed sixteen FMAs. Only runs on the interior
/// region (every tap in-range) at strip granularity; the tail of the
/// interior falls back to 8-lane strips then scalar.
///
/// SAFETY contract: caller has verified avx2+fma at runtime; `tile` is
/// exactly `4 * bt` long; `int_lo..int_hi` lies inside `[t0, t0+bt)` and
/// for every t there, `t + j*dilation - pad` is a valid index into each
/// channel row of `xv`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
// SAFETY: caller must ensure avx2+fma (dispatched on
// `is_x86_feature_detected!` above) and must pass only the INTERIOR region,
// where every tap's input index is in range - the edge columns are handled by
// the scalar path precisely because this kernel does not bounds-check them.
unsafe fn conv_quad_avx2(
    xv: &[f32],
    len: usize,
    wp: &[f32], // this quad's weights, packed [ci][j][4 lanes]
    tile: &mut [f32],
    bt: usize,
    c_in: usize,
    k: usize,
    pad: usize,
    dilation: usize,
    t0: usize,
    int_lo: usize,
    int_hi: usize,
) {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_ps, _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_loadu_ps,
            _mm256_setzero_ps, _mm256_storeu_ps,
        };
        let x = xv.as_ptr();
        let w = wp.as_ptr();
        let mut t = int_lo;
        while t < int_hi {
            let strip = (int_hi - t).min(32);
            if strip == 32 {
                let mut acc = [[_mm256_setzero_ps(); 4]; 4]; // [quad][lane-group]
                let mut wi = 0usize;
                for ci in 0..c_in {
                    let row = x.add(ci * len + t - pad);
                    for j in 0..k {
                        let off = j * dilation;
                        let s0 = _mm256_loadu_ps(row.add(off));
                        let s1 = _mm256_loadu_ps(row.add(off + 8));
                        let s2 = _mm256_loadu_ps(row.add(off + 16));
                        let s3 = _mm256_loadu_ps(row.add(off + 24));
                        // Four packed weights: one cache line, streamed in order.
                        for q in 0..4 {
                            let wq = _mm256_broadcast_ss(&*w.add(wi + q));
                            acc[q][0] = _mm256_fmadd_ps(wq, s0, acc[q][0]);
                            acc[q][1] = _mm256_fmadd_ps(wq, s1, acc[q][1]);
                            acc[q][2] = _mm256_fmadd_ps(wq, s2, acc[q][2]);
                            acc[q][3] = _mm256_fmadd_ps(wq, s3, acc[q][3]);
                        }
                        wi += 4;
                    }
                }
                for q in 0..4 {
                    let dst = tile.as_mut_ptr().add(q * bt + (t - t0));
                    for (g, a) in acc[q].iter().enumerate() {
                        let cur = _mm256_loadu_ps(dst.add(g * 8));
                        _mm256_storeu_ps(dst.add(g * 8), _mm256_add_ps(cur, *a));
                    }
                }
                t += 32;
            } else if strip >= 8 {
                // 8-lane tail strip: at short lengths (the flow's L≈190) the
                // 32-lane kernel leaves ~26 positions, and running them scalar
                // cost ~half the stage. One ymm per quad channel.
                let mut acc = [_mm256_setzero_ps(); 4];
                let mut wi = 0usize;
                for ci in 0..c_in {
                    let row = x.add(ci * len + t - pad);
                    for j in 0..k {
                        let s0 = _mm256_loadu_ps(row.add(j * dilation));
                        for (q, a) in acc.iter_mut().enumerate() {
                            let wq = _mm256_broadcast_ss(&*w.add(wi + q));
                            *a = _mm256_fmadd_ps(wq, s0, *a);
                        }
                        wi += 4;
                    }
                }
                for (q, a) in acc.iter().enumerate() {
                    let dst = tile.as_mut_ptr().add(q * bt + (t - t0));
                    let cur = _mm256_loadu_ps(dst);
                    _mm256_storeu_ps(dst, _mm256_add_ps(cur, *a));
                }
                t += 8;
            } else {
                // Final < 8 positions: scalar quad.
                for ci in 0..c_in {
                    let xrow = &xv[ci * len..(ci + 1) * len];
                    for j in 0..k {
                        let off = j * dilation;
                        for q in 0..4 {
                            let wcoef = wp[(ci * k + j) * 4 + q];
                            for tt in t..int_hi {
                                tile[q * bt + tt - t0] += wcoef * xrow[tt + off - pad];
                            }
                        }
                    }
                }
                break;
            }
        }
    }
}

/// Vectorizable exp: exponent-field construction + degree-5 minimax for the
/// fraction — the `flash_attn.rs` recipe. Relative error ~2e-7: THREE orders
/// tighter than the Padé tanh that failed the `dec_in` oracle (1.8e-4
/// compounding to 7e-2 over 16 coupled gates). Pure float/int ops, no libm,
/// autovectorizes.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    // Delegates to `ffai_core::fastmath`, and the delegation is a FIX, not a
    // tidy-up.
    //
    // The local version reduced the range with `y.floor()`. `f32::floor` has
    // no SSE2 instruction — it needs SSE4.1's `roundps` — so on a portable
    // x86-64 build it lowers to a **call to `floorf`**, one per element, in the
    // middle of the loop this kernel exists to vectorise. Removing `exp` and
    // leaving `floor` removes the headline libm call and keeps a libm call.
    //
    // `ffai-diana` found the same trap in its own copy (with `round`, which is
    // ties-away-from-zero and worse still) and fixed it with the `+1.5*2^23`
    // trick: pure float arithmetic, no call, no branch, no `target_feature`,
    // measured **4.71x on the rounding step and bit-identical**. That fix now
    // lives in one place instead of being rediscovered a third time.
    ffai_core::fastmath::exp(x)
}

#[inline(always)]
fn fast_tanh_exp(x: f32) -> f32 {
    // tanh(x) = 1 − 2/(e^{2x}+1); error tracks fast_exp's ~2e-7.
    1.0 - 2.0 / (fast_exp(2.0 * x) + 1.0)
}

#[inline(always)]
fn fast_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
}

/// Branchless leaky relu, rayon-chunked: these glue ops are pure memory
/// bandwidth over multi-MB buffers, and single-threaded they were costing
/// as much as an upsample stage.
fn leaky_inplace(x: &mut [f32], slope: f32) {
    x.par_chunks_mut(1 << 16).for_each(|c| {
        for v in c {
            *v = slope.mul_add(v.min(0.0), v.max(0.0));
        }
    });
}

// ---------------------------------------------------------------------------
// The flat flow: all four residual couplings in one Vec<f32> domain.
// ---------------------------------------------------------------------------

/// One mean-only residual coupling (VITS `ResidualCouplingLayer` +
/// preceding channel Flip), flat. The candle version spent more on op
/// dispatch and gate plumbing (narrow/tanh/sigmoid/mul per WN layer) than
/// on its convolutions; here the gate is one fused pass.
struct FlatCoupling {
    /// GEMM-shaped weight tensors ([`C_out`, `C_in·K`]), built ONCE — the
    /// per-call `from_slice` was copying ~25 MB of weights per sentence,
    /// and caching these was the change that took the flow from a wash to
    /// 42/45 paired wins. ONLY the tensors are stored (plus biases): a
    /// second flat/packed copy of the flow's 7 M weights is what flipped
    /// the footprint gate FAIL once already.
    pre_t: Tensor,
    in_t: Vec<Tensor>,
    rs_t: Vec<Tensor>,
    post_t: Tensor,
    pre_b: Option<Vec<f32>>,
    in_b: Vec<Option<Vec<f32>>>,
    rs_b: Vec<Option<Vec<f32>>>,
    post_b: Option<Vec<f32>>,
    k: usize,
    pad: usize,
}

pub struct FlatFlow {
    /// In REVERSE application order: flows 6, 4, 2, 0, each preceded by a
    /// channel flip (the odd-indexed Flip modules of the export).
    couplings: Vec<FlatCoupling>,
    hidden: usize,
}

impl FlatFlow {
    pub fn from_weights(
        get: &dyn Fn(&str) -> Result<(Vec<f32>, Vec<usize>)>,
        geom: &dyn Fn(&str) -> (usize, usize, usize),
        hidden: usize,
    ) -> Result<Self> {
        let as_mm = |name: &str| -> Result<(Tensor, Option<Vec<f32>>, usize)> {
            let (w, dims) = get(&format!("{name}.weight"))?;
            let b = get(&format!("{name}.bias")).ok().map(|(v, _)| v);
            let t = e(Tensor::from_slice(
                &w,
                (dims[0], dims[1] * dims[2]),
                &Device::Cpu,
            ))?;
            Ok((t, b, dims[2]))
        };
        let mut couplings = Vec::new();
        for flow in [6usize, 4, 2, 0] {
            let base = format!("flow.flows.{flow}");
            let (pre_t, pre_b, _) = as_mm(&format!("{base}.pre"))?;
            let (post_t, post_b, _) = as_mm(&format!("{base}.post"))?;
            let mut in_t = Vec::new();
            let mut in_b = Vec::new();
            let mut rs_t = Vec::new();
            let mut rs_b = Vec::new();
            let mut k = 5;
            for i in 0..4 {
                let (t, b, kk) = as_mm(&format!("{base}.enc.in_layers.{i}"))?;
                k = kk;
                in_t.push(t);
                in_b.push(b);
                let (t, b, _) = as_mm(&format!("{base}.enc.res_skip_layers.{i}"))?;
                rs_t.push(t);
                rs_b.push(b);
            }
            let (pad, _, _) = geom(&format!("{base}.enc.in_layers.0"));
            couplings.push(FlatCoupling {
                pre_t,
                in_t,
                rs_t,
                post_t,
                pre_b,
                in_b,
                rs_b,
                post_b,
                k,
                pad,
            });
        }
        Ok(Self { couplings, hidden })
    }

    /// `z_p` (channel-major, hidden×len) → z, in place.
    ///
    /// v4. The verdict trail matters here: v1 (flat de-plumbing) washed at
    /// 0.99×; v2 (candle-GEMM shaping) lost at 0.83×; then ORT's per-node
    /// profile proved the convs themselves had 2.7× headroom over candle,
    /// and the QUAD-PACKED AVX2 kernel closed most of that gap at every
    /// shape. v4 is v1's structure (flat convs + one-pass fused gates) on
    /// the packed kernel.
    pub fn run(&self, z: &mut [f32], len: usize) -> Result<()> {
        let profile = std::env::var("FFAI_PROFILE").is_ok();
        let (mut t_conv_in, mut t_conv_rs, mut t_gate, mut t_other) = (0f64, 0f64, 0f64, 0f64);
        let clock = crate::clock::Instant::now;
        let c = self.hidden;
        let half = c / 2;
        for coupling in &self.couplings {
            // Every conv here is a GEMM through candle's tuned matmul — the
            // probe at the flow's exact shapes ranked it decisively
            // (res_skip 1×1: matmul 96 GF/s vs conv-path 19 vs our direct
            // kernel 12; in_layer k5: im2col+matmul > conv > direct). The
            // earlier v2 loss was its branchy per-element im2col gather and
            // pre-fast-exp gates, not the GEMM idea.
            let dev = Device::Cpu;
            let mm = |wt: &Tensor, c_in: usize, x: &[f32]| -> Result<Vec<f32>> {
                let xt = e(Tensor::from_slice(x, (c_in, len), &dev))?;
                e(e(e(wt.matmul(&xt))?.flatten_all())?.to_vec1())
            };

            let t0 = clock();
            // Channel flip (the export's Slice-with-negative-step).
            for lo in 0..c / 2 {
                let hi = c - 1 - lo;
                let (a, b) = z.split_at_mut(hi * len);
                a[lo * len..lo * len + len].swap_with_slice(&mut b[..len]);
            }
            let (x0, x1) = z.split_at_mut(half * len);
            let mut h = mm(&coupling.pre_t, half, x0)?;
            if let Some(b) = &coupling.pre_b {
                for (row, bias) in h.chunks_mut(len).zip(b) {
                    row.iter_mut().for_each(|v| *v += bias);
                }
            }
            t_other += t0.elapsed().as_secs_f64();

            let k = coupling.k;
            let pad = coupling.pad;
            let mut cols = vec![0f32; c * k * len];
            let mut skip = vec![0f32; c * len];
            let mut acts = vec![0f32; c * len];
            let n_layers = coupling.in_t.len();
            for i in 0..n_layers {
                let t0 = clock();
                // im2col by MEMCPY: row (ci·k + j) is h[ci] shifted by
                // j−pad — one copy_from_slice for the overlap, zeros for
                // the 0–2 edge cells. (The per-element branchy version of
                // this gather was half of v2's loss.)
                for ci in 0..c {
                    let src = &h[ci * len..(ci + 1) * len];
                    for j in 0..k {
                        let row = &mut cols[(ci * k + j) * len..(ci * k + j + 1) * len];
                        let off = j as isize - pad as isize;
                        if off >= 0 {
                            let o = off as usize;
                            row[..len - o].copy_from_slice(&src[o..]);
                            row[len - o..].iter_mut().for_each(|v| *v = 0.0);
                        } else {
                            let o = (-off) as usize;
                            row[..o].iter_mut().for_each(|v| *v = 0.0);
                            row[o..].copy_from_slice(&src[..len - o]);
                        }
                    }
                }
                let mut a = mm(&coupling.in_t[i], c * k, &cols)?;
                if let Some(b) = &coupling.in_b[i] {
                    for (row, bias) in a.chunks_mut(len).zip(b) {
                        row.iter_mut().for_each(|v| *v += bias);
                    }
                }
                t_conv_in += t0.elapsed().as_secs_f64();
                // Fused gate: tanh(a[..c]) · sigmoid(a[c..2c]), rayon over
                // channel rows, via the vectorized-exp forms (~2e-7 rel
                // err). History at this exact line: libm was the stage's
                // scalar bottleneck; the Padé tanh (1.8e-4) was tried and
                // REVERTED when it compounded to 7e-2 at dec_in over 16
                // coupled gates. fast_exp is three orders tighter and the
                // dec_in oracle re-gates it.
                let t0 = clock();
                let (gate_t, gate_s) = a.split_at(c * len);
                acts.par_chunks_mut(len)
                    .zip(gate_t.par_chunks(len).zip(gate_s.par_chunks(len)))
                    .for_each(|(dst, (at, as_))| {
                        for ((d, &t), &s) in dst.iter_mut().zip(at).zip(as_) {
                            *d = fast_tanh_exp(t) * fast_sigmoid(s);
                        }
                    });
                t_gate += t0.elapsed().as_secs_f64();
                let t0 = clock();
                // res_skip is a pure GEMM (k=1, the 96-vs-19 GF/s probe
                // row); its bias fuses into the residual/skip adds.
                let r = mm(&coupling.rs_t[i], c, &acts)?;
                let rb = coupling.rs_b[i].as_deref();
                if i < n_layers - 1 {
                    for ch in 0..c {
                        let bias = rb.map_or(0.0, |b| b[ch]);
                        let hr = &mut h[ch * len..(ch + 1) * len];
                        for (hv, rv) in hr.iter_mut().zip(&r[ch * len..(ch + 1) * len]) {
                            *hv += rv + bias;
                        }
                        let bias2 = rb.map_or(0.0, |b| b[c + ch]);
                        let sk = &mut skip[ch * len..(ch + 1) * len];
                        for (sv, rv) in sk.iter_mut().zip(&r[(c + ch) * len..(c + ch + 1) * len]) {
                            *sv += rv + bias2;
                        }
                    }
                } else {
                    for ch in 0..c {
                        let bias = rb.map_or(0.0, |b| b[ch]);
                        let sk = &mut skip[ch * len..(ch + 1) * len];
                        for (sv, rv) in sk.iter_mut().zip(&r[ch * len..(ch + 1) * len]) {
                            *sv += rv + bias;
                        }
                    }
                }
                t_conv_rs += t0.elapsed().as_secs_f64();
            }
            let t0 = clock();
            // post: GEMM; bias fused into the coupling subtract.
            let m = mm(&coupling.post_t, c, &skip)?;
            let pb = coupling.post_b.as_deref();
            for ch in 0..half {
                let bias = pb.map_or(0.0, |b| b[ch]);
                let xr = &mut x1[ch * len..(ch + 1) * len];
                for (x, mv) in xr.iter_mut().zip(&m[ch * len..(ch + 1) * len]) {
                    *x -= mv + bias;
                }
            }
            t_other += t0.elapsed().as_secs_f64();
        }
        if profile {
            eprintln!(
                "[flow] conv_in {:.1} ms  conv_rs+adds {:.1} ms  gates {:.1} ms  pre/post/flip {:.1} ms",
                t_conv_in * 1000.0,
                t_conv_rs * 1000.0,
                t_gate * 1000.0,
                t_other * 1000.0
            );
        }
        Ok(())
    }
}

fn leaky_into(x: &[f32], out: &mut [f32], slope: f32) {
    out.par_chunks_mut(1 << 16)
        .zip(x.par_chunks(1 << 16))
        .for_each(|(oc, xc)| {
            for (o, &v) in oc.iter_mut().zip(xc) {
                *o = slope.mul_add(v.min(0.0), v.max(0.0));
            }
        });
}

fn add_inplace(x: &mut [f32], y: &[f32]) {
    x.par_chunks_mut(1 << 16)
        .zip(y.par_chunks(1 << 16))
        .for_each(|(xc, yc)| {
            for (a, b) in xc.iter_mut().zip(yc) {
                *a += b;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_delta(a: &Tensor, b: &Tensor) -> f32 {
        let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let bv: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            av.len(),
            bv.len(),
            "shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
        av.iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn kernels_match_candle_on_every_decoder_shape() {
        // candle's conv is the scalar twin: every (channels, kernel,
        // dilation, length) combination the decoder actually uses, random
        // data, content-compared to reassociation tolerance.
        let dev = Device::Cpu;
        for (c_in, c_out, k, dil, len) in [
            (192usize, 256usize, 7usize, 1usize, 189usize), // conv_pre
            (128, 128, 3, 1, 1520),                         // resblock k3 d1
            (128, 128, 3, 2, 1520),                         // resblock k3 d2
            (64, 64, 5, 2, 1000),                           // resblock k5 d2
            (64, 64, 5, 6, 1000),                           // resblock k5 d6
            (32, 32, 7, 3, 2048),                           // resblock k7
            (32, 1, 7, 1, 2048),                            // conv_post (no bias)
        ] {
            let x = Tensor::randn(0f32, 1f32, (1, c_in, len), &dev).unwrap();
            let w = Tensor::randn(0f32, 0.3f32, (c_out, c_in, k), &dev).unwrap();
            let bias = Tensor::randn(0f32, 0.3f32, (c_out,), &dev).unwrap();
            let pad = dil * (k - 1) / 2;

            let reference = x
                .conv1d(&w, pad, 1, dil, 1)
                .unwrap()
                .broadcast_add(&bias.reshape((1, c_out, 1)).unwrap())
                .unwrap();
            let ours = conv1d_direct(&x, &w, Some(&bias), pad, dil).unwrap();
            let d = max_delta(&reference, &ours);
            assert!(
                d < 5e-4,
                "conv c{c_in}->{c_out} k{k} d{dil} L{len}: max delta {d}"
            );
        }
    }

    #[test]
    fn fast_exp_family_tracks_libm_to_gate_grade() {
        // The Padé tanh failed at 1.8e-4 (compounds to 7e-2 over 16 coupled
        // gates); the exp-based forms must be ~three orders tighter across
        // the gates' working range.
        let mut max_tanh = 0f32;
        let mut max_sig = 0f32;
        for i in -2000..=2000 {
            let x = i as f32 * 0.01; // [-20, 20]
            max_tanh = max_tanh.max((fast_tanh_exp(x) - x.tanh()).abs());
            max_sig = max_sig.max((fast_sigmoid(x) - 1.0 / (1.0 + (-x).exp())).abs());
        }
        // Measured 3.6e-5 — five times tighter than the Padé form that
        // failed, and it compounds to 1.2e-3 at dec_in against the stage
        // oracle's 2e-3 tolerance (measured, all fixtures). If either
        // number grows past this bar, the compounding math needs re-checking
        // BEFORE loosening anything.
        assert!(max_tanh < 5e-5, "tanh err {max_tanh}");
        assert!(max_sig < 5e-5, "sigmoid err {max_sig}");
    }

    #[test]
    fn transpose_kernel_matches_candle_on_the_three_upsamplers() {
        let dev = Device::Cpu;
        for (c_in, c_out, k, stride, pad, len) in [
            (256usize, 128usize, 16usize, 8usize, 4usize, 189usize),
            (128, 64, 16, 8, 4, 400),
            (64, 32, 8, 4, 2, 800),
        ] {
            let x = Tensor::randn(0f32, 1f32, (1, c_in, len), &dev).unwrap();
            let w = Tensor::randn(0f32, 0.2f32, (c_in, c_out, k), &dev).unwrap();
            let bias = Tensor::randn(0f32, 0.2f32, (c_out,), &dev).unwrap();

            let reference = x
                .conv_transpose1d(&w, pad, 0, stride, 1, 1)
                .unwrap()
                .broadcast_add(&bias.reshape((1, c_out, 1)).unwrap())
                .unwrap();
            let ours = conv_transpose1d_direct(&x, &w, Some(&bias), pad, stride).unwrap();
            let d = max_delta(&reference, &ours);
            assert!(
                d < 5e-4,
                "convT c{c_in}->{c_out} k{k} s{stride}: max delta {d}"
            );
        }
    }
}

/// `conv1d(k=3, pad=1, stride=1, dilation=1, groups=1)` expressed as the GEMM
/// it actually is: im2col the input to `[C_in*3, T]`, then one matmul against
/// the weights viewed as `[C_out, C_in*3]`.
///
/// Candle's `conv1d` wrapper charges **1.71x** to do the same arithmetic on the
/// text encoder's feed-forward shapes (251 -> 147 ms over 6 layers x 20
/// sentences, 74.9 -> 127.8 GFLOP/s — `examples/ffn_ceiling.rs`). The FFN was
/// costing more on its own than onnxruntime spends on the whole text encoder,
/// which is what sent us looking.
///
/// Not bit-identical to `conv1d`: a GEMM accumulates in a different order, so
/// this is gated on tolerance against the stage oracle rather than on `cmp`.
pub fn conv1d_k3_gemm(x: &Tensor, w: &Tensor, bias: &Tensor, device: &Device) -> Result<Tensor> {
    let (_, c_in, t) = e(x.dims3())?;
    let (c_out, _, k) = e(w.dims3())?;
    debug_assert_eq!(k, 3);
    let xv: Vec<f32> = e(e(x.flatten_all())?.to_vec1())?;

    // [C_in*3, T]: row (ci*3 + j) holds x[ci] shifted so output i reads
    // x[i + j - 1], zero outside the sequence (pad = 1).
    let mut col = vec![0f32; c_in * k * t];
    for ci in 0..c_in {
        let src = &xv[ci * t..(ci + 1) * t];
        // j = 0 -> reads i-1, so output 0 is pad and the copy starts at 1
        col[(ci * k) * t + 1..(ci * k) * t + t].copy_from_slice(&src[..t - 1]);
        // j = 1 -> reads i
        col[(ci * k + 1) * t..(ci * k + 1) * t + t].copy_from_slice(src);
        // j = 2 -> reads i+1, so the last output is pad
        col[(ci * k + 2) * t..(ci * k + 2) * t + t - 1].copy_from_slice(&src[1..]);
    }

    let cm = e(Tensor::from_vec(col, (c_in * k, t), device))?;
    let w2 = e(w.reshape((c_out, c_in * k)))?;
    let y = e(w2.matmul(&cm))?;
    let y = e(y.broadcast_add(&e(bias.reshape((c_out, 1)))?))?;
    e(y.reshape((1, c_out, t)))
}

/// `conv1d(k=1, pad=0, stride=1, groups=1)` as the matmul it is.
///
/// A 1x1 convolution needs no im2col at all: with a contiguous `[1, C_in, T]`
/// input, `[C_out, C_in] x [C_in, T]` is the whole operation. Candle's conv1d
/// wrapper still charges its overhead to get there, and these are everywhere —
/// the text encoder's fused QKV and output projections, and very nearly the
/// entire duration predictor, whose dense convs are all k=1.
///
/// Not bit-identical to `conv1d` (GEMM accumulation order), so it is gated on
/// tolerance against the stage oracle.
pub fn conv1d_k1_gemm(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    device: &Device,
) -> Result<Tensor> {
    let (_, c_in, t) = e(x.dims3())?;
    let (c_out, w_in, k) = e(w.dims3())?;
    debug_assert_eq!(k, 1);
    debug_assert_eq!(w_in, c_in);
    let _ = device;
    let xm = e(e(x.reshape((c_in, t)))?.contiguous())?;
    let wm = e(w.reshape((c_out, c_in)))?;
    let y = e(wm.matmul(&xm))?;
    let y = match bias {
        Some(b) => e(y.broadcast_add(&e(b.reshape((c_out, 1)))?))?,
        None => y,
    };
    e(y.reshape((1, c_out, t)))
}

/// Channel-wise `LayerNorm` over `[1, C, T]`, in one pass instead of nine
/// allocating tensor ops.
///
/// The tensor spelling is `mean_keepdim -> broadcast_sub -> sqr ->
/// mean_keepdim -> add -> sqrt -> broadcast_div -> broadcast_mul ->
/// broadcast_add`: nine intermediates, each a fresh `[1, C, T]` allocation.
/// The duration predictor runs 24 `LayerNorms` per sentence (4 DDS stacks x 3
/// layers x 2) and the text encoder 12 more, so that is ~216 allocations per
/// utterance for an operation that is two passes over 66 KB.
///
/// Normalisation is over the CHANNEL axis, so in channel-major layout each
/// column is a stride-T gather; at C=192, T~88 the whole tensor is L2-resident
/// and the strided walk costs nothing measurable. Parallel over columns.
///
/// Matches the tensor path to float-reassociation tolerance, not bit-exactly:
/// the variance is accumulated in a different order.
#[must_use]
pub fn layer_norm_flat(
    xv: &[f32],
    c: usize,
    t: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; c * t];
    // Each column is independent; write through a raw pointer so the columns
    // can be filled in parallel without slicing a strided view.
    let ptr = out.as_mut_ptr() as usize;
    // Accumulate in f64 and DIVIDE rather than reciprocal-multiply.
    //
    // The first version summed 192 channels in f32 and precomputed
    // `inv = 1/sqrt(var+eps)` to turn the per-element divide into a multiply.
    // Both are the obvious micro-optimisations and both change the result:
    // `x * (1/s)` carries an extra rounding that `x / s` does not. Measured
    // against the candle path (`examples/quality_attrib.rs`), that kernel was
    // the SOLE source of the campaign's waveform deviation — 1.368e-4 max,
    // 2.307e-6 RMS — while the GEMM convs and the A&S GELU were bit-identical.
    // The +0.09 pp WER shift had been attributed to the GELU on the reasoning
    // that it was "the only real approximation"; it was this.
    //
    // f64 accumulation costs 192 adds per column on a stage that is bandwidth-
    // bound, and the divide is one per element. Both are gated on the paired
    // CPU-time A/B rather than assumed free.
    let eps = f64::from(eps);
    (0..t).into_par_iter().for_each(|i| {
        let mut mean = 0f64;
        for ci in 0..c {
            mean += f64::from(xv[ci * t + i]);
        }
        mean /= c as f64;
        let mut var = 0f64;
        for ci in 0..c {
            let d = f64::from(xv[ci * t + i]) - mean;
            var += d * d;
        }
        var /= c as f64;
        let denom = (var + eps).sqrt();
        // SAFETY: column `i` owns exactly the elements {ci*t + i}, disjoint
        // across i, so no two tasks touch the same address.
        let o = ptr as *mut f32;
        for ci in 0..c {
            unsafe {
                *o.add(ci * t + i) = (((f64::from(xv[ci * t + i]) - mean) / denom) as f32)
                    .mul_add(gamma[ci], beta[ci]);
            }
        }
    });
    out
}

/// Exact-erf GELU, flat and vectorisable: `0.5*x*(1 + erf(x/sqrt(2)))`.
///
/// The exported graph uses `Erf`, not the tanh approximation, so the shape of
/// the function is fixed; only how erf is evaluated is ours to choose. Candle's
/// `gelu_erf` costs 6.3 ns/element and the duration predictor calls it 24 times
/// per sentence over 16,896 elements. Abramowitz & Stegun 7.1.26 evaluates the
/// same function at 3.6 ns/element — **1.73x** — with |error| <= 1.5e-7 by
/// construction and a measured max deviation of 2.5e-7 on real activations.
///
/// For scale, `w_ceil`'s nearest rounding boundary across the whole corpus sits
/// 1.03e-4 away, so this perturbation is ~400x inside the margin that decides
/// the integer contract. A hand-rolled libm-style f64 erf was also measured and
/// was SLOWER than candle (10.4 ns/element) — the win is the approximation, not
/// the flattening.
#[must_use]
pub fn gelu_erf_flat(xv: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; xv.len()];
    out.par_chunks_mut(4096)
        .zip(xv.par_chunks(4096))
        .for_each(|(o, x)| {
            for (o, &v) in o.iter_mut().zip(x) {
                *o = 0.5 * v * (1.0 + erf_as(v * std::f32::consts::FRAC_1_SQRT_2));
            }
        });
    out
}

/// Abramowitz & Stegun 7.1.26 — |error| <= 1.5e-7.
#[inline]
fn erf_as(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0f32 } else { 1.0f32 };
    let x = x.abs();
    let t = 1.0 / 0.327_591_1f32.mul_add(x, 1.0);
    let poly = 1.061_405_4f32
        .mul_add(t, -1.453_152_1)
        .mul_add(t, 1.421_413_8)
        .mul_add(t, -0.284_496_72)
        .mul_add(t, 0.254_829_6)
        * t;
    s * (1.0 - poly * (-x * x).exp())
}

/// im2col for `conv1d(k=3, pad=1)` into a caller-owned buffer, channel-major
/// `[C][T]` -> `[C*3][T]`.
///
/// Three `copy_from_slice` memcpys per channel. Candle's equivalent
/// (`Im2Col1D`, `cpu_backend`) walks element by element with the padding test
/// inside the innermost loop — that, plus a full extra output transpose it
/// performs afterwards, is where its 1.71x wrapper tax lives. Measured NOT to
/// be about GEMM orientation: candle's `[T,K]x[K,Co]` against our
/// `[Co,K]x[K,T]` is 1.02x, i.e. nothing.
///
/// `col` must be at least `c * 3 * t` long; only that prefix is written.
pub fn im2col_k3_into(x: &[f32], c: usize, t: usize, col: &mut [f32]) {
    debug_assert!(col.len() >= c * 3 * t);
    for ci in 0..c {
        let src = &x[ci * t..(ci + 1) * t];
        let base = ci * 3 * t;
        // tap 0 reads x[i-1]: output 0 is padding
        col[base] = 0.0;
        col[base + 1..base + t].copy_from_slice(&src[..t - 1]);
        // tap 1 reads x[i]
        col[base + t..base + 2 * t].copy_from_slice(src);
        // tap 2 reads x[i+1]: the last output is padding
        col[base + 2 * t..base + 3 * t - 1].copy_from_slice(&src[1..]);
        col[base + 3 * t - 1] = 0.0;
    }
}
