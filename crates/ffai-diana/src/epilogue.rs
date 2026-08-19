//! Bias + `SiLU` applied IN PLACE on the matmul's own output buffer.
//!
//! # The graph-execution difference this closes
//!
//! Counted per image: **96 convolutions, 221 library operations** — 88
//! matmuls and 133 `SliceOp`s. That is **2.3 ops per convolution**, where a
//! fused runtime issues one. The gap to ONNX Runtime is not kernel quality
//! (Intel MKL lands within noise of candle's GEMM on these shapes) and not
//! arithmetic (both engines sit at a similar fraction of peak). It is that we
//! issue more operations over the same data.
//!
//! The largest single count is the epilogue: **88 per image**, one per
//! convolution. It read the matmul's output and wrote bias+SiLU into a
//! FRESH buffer — an allocation, plus a write to cold memory that the write
//! -allocate policy first has to fetch.
//!
//! candle's [`InplaceOp1`] removes both. The matmul just wrote this buffer, so
//! it is in L1/L2; mutating it there is a read-modify-write on hot lines
//! instead of a read from hot and a write to cold.
//!
//! # What is measured, and what is not claimed
//!
//! Deterministic counters, one run each, no clock involved:
//!
//! | | SliceOps/image | allocations/image | bytes/image |
//! |---|---:|---:|---:|
//! | out-of-place | 134 | 3,725 | 251.0 MiB |
//! | **in-place** | **53** | **3,194** | **211.4 MiB** |
//! | removed | **81** | **531** | **39.6 MiB** |
//!
//! Sized: 39.6 MiB not written, plus roughly the same again not FETCHED,
//! because writing to a fresh buffer makes the write-allocate policy pull
//! cold lines the matmul's own buffer already has. At the ~50 GB/s this box
//! sustains for L2/L3-resident data that is ~1.7 ms of a ~30 ms image, about
//! **5.5 %**.
//!
//! **No speed claim.** Two A/B runs disagreed in DIRECTION — +5.5 % at 8
//! images, -2.8 % at 45 — with null spreads of 34 % and 61 %. An effect this
//! size is below what this box can resolve, so the clock cannot be promoted
//! to the verdict in either direction. `codec-measurement` §15: at this scale
//! the counter is primary evidence and the clock is confirmatory, and
//! reversing that either discards real work-removal because a noisy box hid
//! it, or banks a regression because a noisy box flattered it.
//!
//! Kept on the counted work removal, labelled **below instrument
//! resolution** rather than "a small win", and re-testable via
//! `FFAI_DIANA_NO_INPLACE=1` when the surrounding stages shrink enough to
//! make 5 % resolvable.
//!
//! # What this is NOT
//!
//! It does not fuse the epilogue into the GEMM's register tile, which is what
//! ORT actually does and would make the elementwise work free rather than
//! merely cheaper. That needs our own GEMM, and this campaign has measured a
//! hand-written kernel losing to candle's by 1.111x before packing is even
//! considered. This is the part of the fusion available without owning the
//! matmul.

use candle_core::{CpuStorage, InplaceOp1, Layout, Result, Tensor};

pub struct Epilogue {
    /// One value per output channel, already extracted from candle.
    pub bias: Option<Vec<f32>>,
    /// Elements per channel — the stride between one channel's bias and the
    /// next.
    pub per_channel: usize,
    pub act: bool,
}

impl InplaceOp1 for Epilogue {
    fn name(&self) -> &'static str {
        "ffai-conv-epilogue-inplace"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> Result<()> {
        let CpuStorage::F32(buf) = storage else {
            candle_core::bail!("ffai epilogue: expected f32 storage");
        };
        // A matmul's output is freshly allocated and contiguous. Refusing
        // rather than silently mis-striding is the right failure: a strided
        // view here would apply the wrong channel's bias to every element.
        let Some(range) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai epilogue: non-contiguous storage");
        };
        let data = &mut buf[range.0..range.1];
        let n = self.per_channel;
        if n == 0 || data.len() % n != 0 {
            candle_core::bail!(
                "ffai epilogue: {} elements is not a whole number of {n}-element channels",
                data.len()
            );
        }

        let act = self.act;
        let avx2 = act && crate::silu::avx2_enabled();
        let bias = self.bias.as_deref();
        let apply = |(c, chunk): (usize, &mut [f32])| {
            if let Some(b) = bias {
                let bo = b[c];
                for e in chunk.iter_mut() {
                    *e += bo;
                }
            }
            if avx2 {
                // SAFETY: `avx2_enabled()` verified avx2+fma at runtime.
                #[allow(unsafe_code)]
                unsafe {
                    crate::silu_avx2::silu_in_place(chunk);
                }
            } else if act {
                for e in chunk.iter_mut() {
                    *e = crate::silu::silu_scalar_pub(*e);
                }
            }
        };
        if crate::parallel::serial_kernels() {
            data.chunks_mut(n).enumerate().for_each(apply);
        } else {
            use crate::par::prelude::*;
            data.par_chunks_mut(n).enumerate().for_each(apply);
        }
        Ok(())
    }
}

/// Apply bias and optionally `SiLU` to `y` in place, returning it.
///
/// `y` must be the uniquely-owned output of a matmul shaped `[c_out,
/// per_channel]`. Falls back to nothing when there is no work.
pub fn apply(y: Tensor, bias: Option<Vec<f32>>, per_channel: usize, act: bool) -> Result<Tensor> {
    if bias.is_none() && !act {
        return Ok(y);
    }
    if out_of_place() {
        return apply_out_of_place(y, bias, per_channel, act);
    }
    y.inplace_op1(&Epilogue { bias, per_channel, act })?;
    Ok(y)
}

/// `FFAI_DIANA_NO_INPLACE=1` restores the allocating epilogue for A/B.
///
/// Same arithmetic through the same kernels; the arms differ only in whether
/// the result lands in the matmul's own buffer or a fresh one. That is what
/// makes them a clean comparison — the op COUNT is the variable, not the work.
fn out_of_place() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_NO_INPLACE").is_ok_and(|v| v == "1");
            C.store(u8::from(on), Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

fn apply_out_of_place(
    y: Tensor,
    bias: Option<Vec<f32>>,
    per_channel: usize,
    act: bool,
) -> Result<Tensor> {
    let dims = y.dims().to_vec();
    let out = crate::cpuop::SliceOp::new("ffai-conv-epilogue-alloc", move |ys, _| {
        let n_out = ys.len();
        let mut v: Vec<f32> = Vec::with_capacity(n_out);
        {
            let spare = &mut v.spare_capacity_mut()[..n_out];
            let avx2 = act && crate::silu::avx2_enabled();
            let fill = |(c, dst): (usize, &mut [std::mem::MaybeUninit<f32>])| {
                let src = &ys[c * per_channel..(c + 1) * per_channel];
                let bo = bias.as_ref().map_or(0.0, |b| b[c]);
                for (d, s) in dst.iter_mut().zip(src) {
                    d.write(*s + bo);
                }
                // SAFETY: every element of `dst` written by the loop above.
                #[allow(unsafe_code)]
                let dst: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f32>(), dst.len())
                };
                if avx2 {
                    // SAFETY: avx2+fma verified at runtime.
                    #[allow(unsafe_code)]
                    unsafe {
                        crate::silu_avx2::silu_in_place(dst);
                    }
                } else if act {
                    for e in dst.iter_mut() {
                        *e = crate::silu::silu_scalar_pub(*e);
                    }
                }
            };
            if crate::parallel::serial_kernels() {
                spare.chunks_mut(per_channel).enumerate().for_each(fill);
            } else {
                use crate::par::prelude::*;
                spare.par_chunks_mut(per_channel).enumerate().for_each(fill);
            }
        }
        // SAFETY: chunks of `per_channel` cover exactly `n_out`, all written.
        #[allow(unsafe_code)]
        unsafe {
            v.set_len(n_out);
        };
        Ok((v, (dims[0], dims[1]).into()))
    })
    .run(&y)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The in-place epilogue must equal the out-of-place one it replaces:
    /// bias, then SiLU, per channel.
    #[test]
    fn inplace_matches_bias_then_silu() {
        let dev = Device::Cpu;
        let (c_out, n) = (4usize, 37usize);
        let src: Vec<f32> = (0..c_out * n).map(|i| (i % 29) as f32 * 0.21 - 3.0).collect();
        let bias: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.35 - 0.5).collect();

        let y = Tensor::from_vec(src.clone(), (c_out, n), &dev).unwrap();
        let got = apply(y, Some(bias.clone()), n, true).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut want = src;
        for c in 0..c_out {
            for e in &mut want[c * n..(c + 1) * n] {
                *e = crate::silu::silu_scalar_pub(*e + bias[c]);
            }
        }
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            let d = (a - b).abs();
            assert!(d < 1e-5, "element {i}: {a} vs {b}");
        }
    }

    /// Bias with no activation, and activation with no bias, both have to work
    /// — the nine activation-free Convs take the first path.
    #[test]
    fn bias_only_and_act_only() {
        let dev = Device::Cpu;
        let y = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev).unwrap();
        let got = apply(y, Some(vec![10.0, 20.0]), 2, false).unwrap();
        assert_eq!(got.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![11.0, 12.0, 23.0, 24.0]);

        let y = Tensor::from_vec(vec![0.0f32, 0.0], (1, 2), &dev).unwrap();
        let got = apply(y, None, 2, true).unwrap();
        // silu(0) = 0 / (1 + e^0) = 0
        assert_eq!(got.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![0.0, 0.0]);
    }

    /// No bias and no activation must not touch the buffer at all.
    #[test]
    fn no_work_is_a_no_op() {
        let dev = Device::Cpu;
        let y = Tensor::from_vec(vec![1.5f32, -2.5], (1, 2), &dev).unwrap();
        let got = apply(y, None, 2, false).unwrap();
        assert_eq!(got.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![1.5, -2.5]);
    }
}

/// Bias + `SiLU` on an NHWC activation `[OHW, Cout]`, staying NHWC.
///
/// The bias VECTOR repeats along the fast axis here, where the NCHW form
/// broadcasts one scalar over a contiguous run. Measured at 1.37x the NCHW
/// cost - the price a converted run pays, and far smaller than the transpose
/// it avoids.
pub fn apply_nhwc(y: Tensor, bias: Option<Vec<f32>>, c_out: usize, act: bool) -> Result<Tensor> {
    if bias.is_none() && !act {
        return Ok(y);
    }
    y.inplace_op1(&EpilogueNhwc { bias, c_out, act })?;
    Ok(y)
}

struct EpilogueNhwc {
    bias: Option<Vec<f32>>,
    c_out: usize,
    act: bool,
}

impl InplaceOp1 for EpilogueNhwc {
    fn name(&self) -> &'static str {
        "ffai-conv-epilogue-nhwc"
    }
    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> Result<()> {
        let CpuStorage::F32(buf) = storage else {
            candle_core::bail!("ffai epilogue nhwc: expected f32 storage");
        };
        let Some(range) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai epilogue nhwc: non-contiguous storage");
        };
        let data = &mut buf[range.0..range.1];
        let n = self.c_out;
        if n == 0 || data.len() % n != 0 {
            candle_core::bail!("ffai epilogue nhwc: {} is not a multiple of {n}", data.len());
        }
        let (act, bias) = (self.act, self.bias.as_deref());
        let apply = |px: &mut [f32]| {
            if let Some(b) = bias {
                for (e, bb) in px.iter_mut().zip(b) {
                    *e += *bb;
                }
            }
            if act {
                for e in px.iter_mut() {
                    *e = crate::silu::silu_scalar_pub(*e);
                }
            }
        };
        if crate::parallel::serial_kernels() {
            data.chunks_mut(n).for_each(apply);
        } else {
            use crate::par::prelude::*;
            data.par_chunks_mut(n).for_each(apply);
        }
        Ok(())
    }
}

/// Bias + `SiLU` applied while TRANSPOSING `[OHW, Cout]` into `[Cout, OHW]`.
///
/// The NHWC convolution path produces its result pixel-major and the rest of
/// the graph is channel-major, so something has to transpose. Doing it as its
/// own `t()?.contiguous()?` costs a full pass over the activation that the
/// epilogue is about to make anyway — measured in the engine at 0.166 s against
/// an NCHW baseline of 0.019 s, **13 % of the NHWC arm's whole detect time**.
///
/// Fusing them makes the transpose ride along inside a pass that already reads
/// every element and writes every element. The arithmetic is unchanged and the
/// output is the same tensor the separate path produced.
///
/// # Why blocked rather than a column at a time
///
/// The naive form reads column `c` of `[OHW, Cout]` at stride `Cout`, so every
/// load pulls a cache line to use 4 bytes of it, and the line is gone before
/// the next channel wants it. Blocking over a tile of pixels makes each tile's
/// `TILE * Cout` slab L1-resident, so all `Cout` channels read it while it is
/// hot and the strided access is paid once instead of `Cout` times.
pub fn apply_transposed(
    y_t: Tensor,
    bias: Option<Vec<f32>>,
    c_out: usize,
    ohw: usize,
    act: bool,
) -> Result<Tensor> {
    /// Pixels per tile: `TILE * c_out * 4` bytes stays inside L1 for every
    /// `c_out` this graph uses (256 * 64 * 4 = 64 KB at the widest).
    const TILE: usize = 64;

    crate::cpuop::SliceOp::new("ffai-epilogue-transpose", move |ys, _| {
        use crate::par::prelude::*;
        let n = c_out * ohw;
        let mut v: Vec<f32> = vec![0.0; n];
        let avx2 = act && crate::silu::avx2_enabled();
        let bias = bias.as_deref();

        // Parallel over CHANNEL rows: each task owns whole rows of the output,
        // so the writes never overlap and no unsafe splitting is needed. Each
        // task walks the pixel tiles in order, so the slab it reads is the same
        // one its siblings are reading at the same moment — shared in L2/L3
        // rather than fetched per thread.
        let apply_row = |(c, row): (usize, &mut [f32])| {
            let b = bias.map_or(0.0, |bb| bb[c]);
            let mut p0 = 0usize;
            while p0 < ohw {
                let p1 = (p0 + TILE).min(ohw);
                for p in p0..p1 {
                    row[p] = ys[p * c_out + c] + b;
                }
                p0 = p1;
            }
            if avx2 {
                // SAFETY: `avx2_enabled()` verified avx2+fma at runtime.
                #[allow(unsafe_code)]
                unsafe {
                    crate::silu_avx2::silu_in_place(row);
                }
            } else if act {
                for e in row.iter_mut() {
                    *e = crate::silu::silu_scalar_pub(*e);
                }
            }
        };
        if crate::parallel::serial_kernels() {
            v.chunks_mut(ohw).enumerate().for_each(apply_row);
        } else {
            v.par_chunks_mut(ohw).enumerate().for_each(apply_row);
        }
        Ok((v, (c_out, ohw).into()))
    })
    .run(&y_t)
}

#[cfg(test)]
mod transposed_tests {
    use super::*;
    use candle_core::Device;

    /// The fused path must equal transpose-then-epilogue exactly. It is the
    /// same arithmetic in a different order of traversal, so this is an
    /// equality test, not a tolerance one.
    #[test]
    fn fused_transpose_matches_separate() {
        let dev = Device::Cpu;
        let (c_out, ohw) = (7usize, 53usize);
        let src: Vec<f32> = (0..ohw * c_out).map(|i| (i % 31) as f32 * 0.17 - 2.4).collect();
        let bias: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.31 - 0.8).collect();

        let y_t = Tensor::from_vec(src.clone(), (ohw, c_out), &dev).unwrap();
        let got = apply_transposed(y_t, Some(bias.clone()), c_out, ohw, true).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut want = vec![0.0f32; c_out * ohw];
        for c in 0..c_out {
            for p in 0..ohw {
                want[c * ohw + p] = crate::silu::silu_scalar_pub(src[p * c_out + c] + bias[c]);
            }
        }
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-5, "element {i}: {a} vs {b}");
        }
    }
}
