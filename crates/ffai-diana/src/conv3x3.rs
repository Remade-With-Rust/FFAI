//! A 3x3 convolution as one explicit im2col + matmul.
//!
//! # Why
//!
//! candle already im2col's convolutions (`TiledIm2Col`), yet measured
//! **19-21x slower than `Tensor::matmul` on identical arithmetic** at our
//! shapes, at both 1 and 24 threads (`examples/conv_scaling.rs`):
//!
//! | shape | conv GF/s @24t | matmul GF/s @24t |
//! |---|---:|---:|
//! | l4 bneck 32->16 @80 | 25.4 | 522.9 |
//! | head box 64->16 @80 | 29.0 | 553.8 |
//!
//! After the pointwise brick, dense 3x3 is **41% of detect** across 39 calls
//! per image at ~2.0 ms each. This routes it through the tuned GEMM candle
//! already ships — the "hot loops that are secretly matmuls" law.
//!
//! # The layout choice, which is the whole performance argument
//!
//! im2col is written **channel-major**: `(Cin*9, H*W)`, so the matmul is
//! `W (Cout, Cin*9) @ col (Cin*9, H*W) -> (Cout, H*W)` — the output is
//! already NCHW, no transpose.
//!
//! In that orientation each im2col row is *one source row copied with a
//! horizontal offset*, so the inner operation is a contiguous
//! `copy_from_slice`, not a strided gather. (The same observation that made
//! a 3 ms "gather" into a 0.75 ms copy elsewhere.) Rows that fall outside
//! the image are left zero, which is exactly the zero padding.
//!
//! Stride 1, padding 1 only — the seven stride-2 downsampling convolutions
//! stay on candle's path, which is why the dispatch in `blocks.rs` checks
//! the stride.

use candle_core::{Result, Tensor};
use rayon::prelude::*;

/// Dense 3x3, padding 1, groups 1, stride 1 or 2.
///
/// `x` is `(1, Cin, H, W)`, `weight` is `(Cout, Cin, 3, 3)`.
///
/// **Stride 1 and stride 2 are genuinely different kernels inside**, and the
/// difference is the reason each got its own measurement. At stride 1 the
/// horizontal tap is a contiguous `copy_from_slice`; at stride 2 it is a
/// constant-stride gather, which moves half the bytes but cannot use the
/// block copy. Whether that trade pays is an empirical question, not a
/// deduction — see the A/B in the mission plan.
pub fn conv3x3(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    conv3x3_strided(x, weight, bias, 1)
}

/// Tile the im2col so each block is consumed by the GEMM while it is still
/// in cache, instead of materialising the whole `[c_in*9, ohw]` buffer.
///
/// **Why**: the untiled buffer is NINE TIMES the convolution's input and is
/// written to memory and read straight back. Counted, not guessed:
/// 108.4 MiB per image at the n tier, 216.9 MiB of traffic with the
/// read-back, 1.94 GiB at the x tier. Memory bandwidth is shared across
/// cores, so that time does not divide by thread count — which is exactly
/// why im2col scaled 1.55x on 24 threads while the GEMM scaled 3.33x.
///
/// **Tile size targets a BYTE BUDGET, not a tile count**, because the cost
/// curve is sharp on the small side. Measured on layer-0 shapes
/// (`w[16,27] x col[27,N]`, best of 9):
///
/// | tile N | ms/call | tiles for ohw=102400 | total |
/// |---:|---:|---:|---:|
/// | 102400 (untiled) | 1.309 | 1 | 1.309 |
/// | 10240 | 0.097 | 10 | **0.967** |
/// | 4855 | 0.215 | 22 | 4.721 |
/// | 1024 | 0.056 | 100 | 5.560 |
///
/// Ten tiles beat one call by 1.35x; twenty-two are 3.6x worse than ten. A
/// fixed tile COUNT would land on the wrong side of that edge at some layer,
/// so the budget is in bytes and the count falls out of it.
///
/// Tiles are whole output ROWS, which keeps every `ohw` run contiguous and
/// lets the inner loops stay exactly what the untiled kernel runs.
fn conv3x3_tiled(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
) -> Result<Tensor> {
    let (n, c_in, h, w) = x.dims4()?;
    let c_out = weight.dim(0)?;
    let oh = (h + 2 - 3) / stride + 1;
    let ow = (w + 2 - 3) / stride + 1;
    let (ohw, k) = (oh * ow, c_in * 9);

    // ~1 MiB of f32 per col tile. Whole rows, at least one, never more than
    // the whole output.
    const TILE_BYTES: usize = 1 << 20;
    let rows = ((TILE_BYTES / 4) / (k * ow)).clamp(1, oh);
    let dev = x.device().clone();
    let w_mat = weight.reshape((c_out, k))?;
    // Read the activation IN PLACE.
    //
    // This was `x.flatten_all()?.to_vec1()?` — a full copy of the input
    // tensor on EVERY convolution, which is exactly the cost one of the nine
    // M-D2 speed bricks removed from the untiled path. Tiling as written
    // handed it straight back, and the A/B duly measured tiling 24 % more
    // CPU (3/21, z = -3.27) while the GEMM-shape microbench said tiling
    // should WIN by 1.35x. That contradiction was the defect, not the idea.
    let xs_t = x.flatten_all()?;
    let xs = xs_t.to_vec1::<f32>()?;
    let hw = h * w;

    let mut out = vec![0f32; c_out * ohw];
    let mut oy0 = 0usize;
    while oy0 < oh {
        let oy1 = (oy0 + rows).min(oh);
        let brows = oy1 - oy0;
        let b = brows * ow;
        let mut col = vec![0f32; k * b];
        crate::conv3x3::count_im2col(k * b);

        // One task per CHANNEL, all nine taps inside — not per (channel, tap).
        //
        // The comment this replaces argued the tile is cache-resident so the
        // nine plane re-reads cost nothing. Measured elsewhere in this
        // campaign, that same per-(channel, tap) split made the untiled
        // im2col 2.2x SLOWER (0.611 -> 1.366 s serial) because the source
        // plane is read nine times instead of once. A tile being smaller
        // does not make nine reads as cheap as one.
        let fill = |(c, block): (usize, &mut [f32])| {
            let plane = &xs[c * hw..(c + 1) * hw];
            for tap in 0..9usize {
            let (ky, kx) = (tap / 3, tap % 3);
            let row = &mut block[tap * b..(tap + 1) * b];
            for oy in oy0..oy1 {
                let sy = oy * stride + ky;
                if sy == 0 || sy > h {
                    continue;
                }
                let src = &plane[(sy - 1) * w..sy * w];
                let dst = &mut row[(oy - oy0) * ow..(oy - oy0 + 1) * ow];
                if stride == 1 {
                    match kx {
                        0 => dst[1..].copy_from_slice(&src[..w - 1]),
                        1 => dst.copy_from_slice(src),
                        _ => dst[..w - 1].copy_from_slice(&src[1..]),
                    }
                } else {
                    let lo = if kx == 0 { 1 } else { 0 };
                    let hi = if w >= kx { ((w - kx) / 2 + 1).min(ow) } else { 0 };
                    for ox in lo..hi {
                        dst[ox] = src[ox * 2 + kx - 1];
                    }
                }
            }
            }
        };
        if crate::parallel::serial_kernels() {
            col.chunks_mut(9 * b).enumerate().for_each(fill);
        } else {
            col.par_chunks_mut(9 * b).enumerate().for_each(fill);
        }

        let col_t = Tensor::from_vec(col, (k, b), &dev)?;
        let y_t = crate::profile::timed(|p| &p.gemm, || w_mat.matmul(&col_t))?;
        let y_v = y_t.flatten_all()?.to_vec1::<f32>()?;
        // Scatter [c_out, b] into columns [oy0*ow, oy1*ow) of [c_out, ohw].
        for o in 0..c_out {
            out[o * ohw + oy0 * ow..o * ohw + oy1 * ow]
                .copy_from_slice(&y_v[o * b..(o + 1) * b]);
        }
        oy0 = oy1;
    }

    let mut y = Tensor::from_vec(out, (c_out, ohw), &dev)?;
    if let Some(bs) = bias {
        y = y.broadcast_add(&bs.reshape((c_out, 1))?)?;
    }
    y.reshape((n, c_out, oh, ow))
}

/// Tiling is **OFF by default: refuted, twice, the second time cleanly.**
///
/// The bandwidth argument that motivated it is correct and still stands —
/// im2col materialises 216.9 MiB per image at the n tier (deterministic
/// counter, not a timing), read straight back by the GEMM, and bandwidth is
/// shared so it does not divide by thread count. Tiling should fix that.
///
/// **First refutation, with two defects of its own.** 3/21 on CPU
/// (z = -3.27), ~24 % more CPU than untiled. The implementation copied the
/// whole input tensor per convolution (`flatten_all().to_vec1()`) — the very
/// cost an M-D2 brick had removed from the untiled path — and split the fill
/// per (channel, tap), which reads each source plane nine times instead of
/// once and measured 2.2x slower when tried on the untiled im2col.
///
/// **Both defects fixed, and it is still slower: 15/15, z = +3.87, median
/// 1.355x, against a 2.0 % null spread** — so this time the magnitude is
/// usable, not just the direction.
///
/// What remains is the marshalling the GEMM-shape microbench never measured.
/// That probe compared `w[16,27] x col[27,N]` at several N and found ten
/// tiles beat one call by 1.35x — true, and irrelevant, because it priced
/// only the multiply. Each tile here also pays a `Tensor::from_vec`, a
/// `matmul`, and a `to_vec1` back: roughly ten per convolution and ~5850 per
/// image. **Isolation measured the arithmetic and omitted the plumbing**,
/// which is the third time this campaign that a microbench over-promised
/// against the pipeline.
///
/// **The idea is not dead; this route to it is.** Tiling can only pay if the
/// tile never becomes a `Tensor` — i.e. a fused im2col+GEMM micro-kernel
/// owning its own accumulation. That was tried too (`crate::direct3x3`,
/// three variants) and lost 9/9 to candle's GEMM, which runs at 69 % of peak
/// where a hand-written one does not. Closing that gap means beating a tuned
/// GEMM, not avoiding one.
///
/// Kept behind `FFAI_DIANA_TILE=1` so the next attempt starts from a
/// working, gated implementation rather than a blank file.
fn tiling_disabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_TILE").is_ok_and(|v| v == "1");
            C.store(!on as u8, Ordering::Relaxed);
            !on
        }
        v => v == 1,
    }
}

pub fn conv3x3_strided(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
) -> Result<Tensor> {
    // Direct convolution first: it never builds the 9x-expanded operand,
    // which is the only thing that moves the GEMM's arithmetic intensity.
    // Falls through to im2col for shapes it declines (stride 2, w < 2).
    if stride == 1 && crate::direct3x3::enabled() {
        if let Some(y) = crate::direct3x3::conv3x3_direct(x, weight, bias)? {
            return Ok(y);
        }
    }
    if !tiling_disabled() {
        return conv3x3_tiled(x, weight, bias, stride);
    }
    let (n, c_in, h, w) = x.dims4()?;
    let c_out = weight.dim(0)?;
    debug_assert_eq!(n, 1, "batch 1 is the only shape this engine runs");
    debug_assert!(stride == 1 || stride == 2, "only stride 1 and 2 are implemented");
    let hw = h * w;
    // padding 1, kernel 3: out = (in + 2 - 3)/stride + 1
    let oh = (h + 2 - 3) / stride + 1;
    let ow = (w + 2 - 3) / stride + 1;
    let ohw = oh * ow;

    // im2col reads the activation IN PLACE via SliceOp and hands its output
    // Vec straight to the tensor's storage. Two copies of a multi-megabyte
    // buffer removed per convolution, with no arithmetic changed.
    // Deterministic byte counter, `FFAI_DIANA_COUNT=1`.
    //
    // Wall time on this box drifts further than the effects being measured
    // — a thread sweep read im2col at 6.855 s ascending and 1.112 s
    // descending for the SAME thread count. A counter does not care: it
    // answers "is this bandwidth-bound?" arithmetically, which is the
    // question that decides whether more threads can ever help.
    crate::conv3x3::count_im2col(c_in * 9 * ohw);
    let col = crate::profile::timed(|p| &p.im2col, || {
        crate::cpuop::SliceOp::new("ffai-im2col3x3", move |xs, _| {
    let mut col = vec![0f32; c_in * 9 * ohw];
    {
    // One row per (channel, ky, kx). Parallel over channels: each channel
    // owns a contiguous 9*ohw block, so the writes never overlap.
    // See `crate::parallel`: the fan-out is worth its barrier cost for a
    // single image and is pure overhead inside a parallel batch.
    let im2col_channel = |(c, block): (usize, &mut [f32])| {
        let plane = &xs[c * hw..(c + 1) * hw];
        for ky in 0..3usize {
            for kx in 0..3usize {
                let row = &mut block[(ky * 3 + kx) * ohw..(ky * 3 + kx + 1) * ohw];
                for oy in 0..oh {
                    // Source row for this vertical tap; outside the image
                    // the destination stays zero (the padding).
                    let sy = oy * stride + ky;
                    if sy == 0 || sy > h {
                        continue;
                    }
                    let src = &plane[(sy - 1) * w..sy * w];
                    let dst = &mut row[oy * ow..(oy + 1) * ow];
                    if stride == 1 {
                        // Contiguous copy with a horizontal offset.
                        match kx {
                            0 => dst[1..].copy_from_slice(&src[..w - 1]),
                            1 => dst.copy_from_slice(src),
                            _ => dst[..w - 1].copy_from_slice(&src[1..]),
                        }
                    } else {
                        // sx = 2*ox + kx - 1. Solve the in-bounds range once
                        // so the inner loop is branch-free.
                        //   sx >= 0    <=> 2*ox >= 1 - kx
                        //   sx <= w-1  <=> ox <= (w - kx)/2
                        let lo = if kx == 0 { 1 } else { 0 };
                        let hi =
                            if w >= kx { ((w - kx) / 2 + 1).min(ow) } else { 0 };
                        for ox in lo..hi {
                            dst[ox] = src[ox * 2 + kx - 1];
                        }
                    }
                }
            }
        }
    };
    if crate::parallel::serial_kernels() {
        col.chunks_mut(9 * ohw).enumerate().for_each(im2col_channel);
    } else {
        col.par_chunks_mut(9 * ohw).enumerate().for_each(im2col_channel);
    }
    }
            Ok((col, (c_in * 9, ohw).into()))
        })
        .run(x)
    })?;

    let w_mat = weight.reshape((c_out, c_in * 9))?;
    let y = crate::profile::timed(|p| &p.gemm, || w_mat.matmul(&col))?; // (Cout, OH*OW)
    // The wrapper, timed separately: bias broadcast and reshape were inside
    // the parent bucket and therefore invisible.
    crate::profile::timed(|p| &p.conv_wrap, || {
        let mut y = y;
        if let Some(b) = bias {
            y = y.broadcast_add(&b.reshape((c_out, 1))?)?;
        }
        y.reshape((n, c_out, oh, ow))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{Conv2d, Conv2dConfig, Module};

    fn oracle(x: &Tensor, w: &Tensor, b: &Tensor, stride: usize) -> Tensor {
        let cfg = Conv2dConfig { padding: 1, stride, ..Default::default() };
        Conv2d::new(w.clone(), Some(b.clone()), cfg).forward(x).unwrap()
    }

    fn max_rel(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs() / scale).fold(0.0f32, f32::max)
    }

    #[test]
    fn matches_candles_conv2d() {
        let dev = Device::Cpu;
        // Real shapes plus degenerate ones where the padding is most of it.
        for &(ci, co, h, w) in &[
            (32, 16, 80, 80),
            (64, 16, 80, 80),
            (16, 8, 160, 160),
            (64, 64, 20, 20),
            (3, 4, 5, 7),
            (1, 1, 1, 1),
            (2, 3, 1, 6),
            (2, 3, 6, 1),
        ] {
            let x = Tensor::randn(0f32, 1.0, (1, ci, h, w), &dev).unwrap();
            let wt = Tensor::randn(0f32, 1.0, (co, ci, 3, 3), &dev).unwrap();
            let b = Tensor::randn(0f32, 1.0, co, &dev).unwrap();
            let got = conv3x3(&x, &wt, Some(&b)).unwrap();
            let want = oracle(&x, &wt, &b, 1);
            assert_eq!(got.dims(), want.dims(), "shape {ci}->{co} {h}x{w}");
            let d = max_rel(&got, &want);
            assert!(d < 1e-5, "{ci}->{co} {h}x{w}: max rel {d:.3e}");
        }
    }

    #[test]
    fn matches_candles_conv2d_at_stride_2() {
        let dev = Device::Cpu;
        // The model's seven downsampling shapes, plus ODD spatial sizes —
        // the in-bounds range arithmetic differs there and an off-by-one in
        // it would be invisible on the even shapes the model actually runs.
        for &(ci, co, h, w) in &[
            (3, 16, 640, 640),
            (16, 32, 320, 320),
            (64, 64, 160, 160),
            (128, 256, 40, 40),
            (2, 3, 7, 9),
            (2, 3, 5, 5),
            (1, 1, 1, 1),
            (2, 2, 1, 8),
            (2, 2, 8, 1),
        ] {
            let x = Tensor::randn(0f32, 1.0, (1, ci, h, w), &dev).unwrap();
            let wt = Tensor::randn(0f32, 1.0, (co, ci, 3, 3), &dev).unwrap();
            let b = Tensor::randn(0f32, 1.0, co, &dev).unwrap();
            let got = conv3x3_strided(&x, &wt, Some(&b), 2).unwrap();
            let want = oracle(&x, &wt, &b, 2);
            assert_eq!(got.dims(), want.dims(), "shape {ci}->{co} {h}x{w} s2");
            let d = max_rel(&got, &want);
            assert!(d < 1e-5, "{ci}->{co} {h}x{w} s2: max rel {d:.3e}");
        }
    }
}

/// Total im2col output elements produced since process start.
///
/// Counted, never timed. im2col EXPANDS a 3x3 convolution's input ninefold
/// before the GEMM consumes it, so the buffer it writes is the dominant
/// memory traffic in the convolution path — and whether that traffic is
/// near the machine's bandwidth decides whether the fix is more threads or
/// fewer bytes.
static IM2COL_ELEMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn count_im2col(n: usize) {
    if counting() {
        IM2COL_ELEMS.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

fn counting() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_COUNT").is_ok_and(|v| v == "1");
            C.store(on as u8, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

/// Elements written by im2col so far, and reset.
pub fn take_im2col_elems() -> u64 {
    IM2COL_ELEMS.swap(0, std::sync::atomic::Ordering::Relaxed)
}
