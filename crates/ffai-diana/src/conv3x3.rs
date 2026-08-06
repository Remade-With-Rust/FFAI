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
pub fn conv3x3(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, act: bool) -> Result<Tensor> {
    conv3x3_strided(x, weight, bias, 1, act)
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
/// Tiled im2col: build a col block sized to stay in cache, GEMM it, repeat.
///
/// **Refuted three times, the third time properly.** The first two were on the
/// system allocator, before the epilogue fusion and before the im2col
/// zero-fill came out — and worse, the tiled path DECLINED fused calls, so it
/// silently gave up the 12.5 % the fusion is worth and was never compared
/// like for like.
///
/// It now carries the same fused epilogue and serves fused calls, and the
/// answer did not change: **20/21 rounds, z = +4.15, 30.2 % SLOWER**, against
/// a 6.0 % null spread that the magnitude also clears. Five-tier oracle passes
/// either way, so this is a speed verdict and not a correctness one.
///
/// Kept behind `FFAI_DIANA_TILE=1` because a refutation with a live arm is
/// re-testable when the surrounding stages move again, and this one has now
/// earned the right to be believed.
fn conv3x3_tiled(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    act: bool,
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
        crate::conv3x3::count_macs((k as u64) * (b as u64) * (c_out as u64));

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

    // Same fused epilogue the untiled path uses — bias and SiLU in ONE
    // traversal. Without this the tiled path silently gives up the 12.5 %
    // the fusion is worth, which is enough on its own to make tiling look
    // like a regression when it is not being compared like for like.
    if act && !crate::blocks::fuse_disabled() {
        let bias_v: Option<Vec<f32>> = match bias {
            Some(b) => Some(b.flatten_all()?.to_vec1::<f32>()?),
            None => None,
        };
        let avx2 = crate::silu::avx2_enabled();
        let apply = |(o, chunk): (usize, &mut [f32])| {
            if let Some(bv) = bias_v.as_ref() {
                let bo = bv[o];
                for e in chunk.iter_mut() {
                    *e += bo;
                }
            }
            if avx2 {
                // SAFETY: `avx2_enabled()` verified avx2+fma at runtime.
                #[allow(unsafe_code)]
                unsafe {
                    crate::silu_avx2::silu_in_place(chunk)
                }
            } else {
                for e in chunk.iter_mut() {
                    *e = crate::silu::silu_scalar_pub(*e);
                }
            }
        };
        let mut out = out;
        if crate::parallel::serial_kernels() {
            out.chunks_mut(ohw).enumerate().for_each(apply);
        } else {
            out.par_chunks_mut(ohw).enumerate().for_each(apply);
        }
        return Tensor::from_vec(out, (c_out, ohw), &dev)?.reshape((n, c_out, oh, ow));
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


/// `FFAI_DIANA_NHWC=1` routes 3x3 convolutions through the transposed GEMM.
///
/// Every NHWC number this campaign has is ISOLATED - synthetic tensors inside
/// an example. `codec-measurement` §11 requires one probe at the level above
/// the change, and this campaign's history is that isolated numbers mislead in
/// BOTH directions. This is that probe: the same convolution, in the engine,
/// on the real graph.
///
/// It carries a deliberate handicap. A true NHWC graph never transposes,
/// because the previous layer already left the activation in that layout. Here
/// the surrounding graph is still NCHW, so this arm pays a transpose of its
/// OUTPUT that the real design would not. It is therefore a LOWER BOUND: if
/// this is close to parity while paying a transpose per layer, the transpose-
/// free version wins by that transpose.
pub fn nhwc_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_NHWC").is_ok_and(|v| v == "1");
            C.store(on as u8, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}


/// `Wt[K, Cout]` cached per weight tensor, built once and reused.
///
/// It was transposed on EVERY call at first, which is defensible as an
/// honest accounting of the arm but is not what a real NHWC engine would do -
/// the weights are constant, so the transpose belongs at model load. It also
/// hid: the cost sat outside every profile scope, and showed up only as 0.046 s
/// of the NHWC arm's total that none of `im2col`, `gemm` or `wrap` accounted
/// for. A residue that no scope claims is the profiler asking where you did not
/// look (`codec-measurement` §6).
///
/// Keyed by candle's own tensor id, which is unique per tensor and stable for
/// the model's lifetime. The entry holds a `Tensor`, so the cache keeps the
/// weights alive and an id cannot be recycled underneath it.
fn weight_t_cached(weight: &Tensor, c_out: usize, k: usize) -> Result<Tensor> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static CACHE: Mutex<Option<HashMap<candle_core::TensorId, Tensor>>> = Mutex::new(None);

    let key = weight.id();
    let mut g = CACHE.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(t) = m.get(&key) {
        return Ok(t.clone());
    }
    let t = weight.reshape((c_out, k))?.t()?.contiguous()?;
    m.insert(key, t.clone());
    Ok(t)
}

/// im2col producing `[OH*OW, Cin*9]` - the transpose of the shipped layout.
///
/// The K index is `c*9 + ky*3 + kx`, matching the shipped `im2col` exactly, so
/// the weight matrix transposes directly with no re-ordering. Getting that
/// wrong is a wrong graph rather than a slow one, which is why the gate below
/// compares against the NCHW path element by element.
fn im2col_t(
    xs: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    stride: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let k = c_in * 9;
    let hw = h * w;
    // Pre-zeroed, so only the in-bounds taps are written; the padding is
    // inherited. That is the opposite of the shipped NCHW im2col, which writes
    // its padding explicitly because it allocates uninitialised - here the
    // padding is scattered one element at a time rather than in runs, so
    // inheriting it is the cheaper side of that trade.
    let mut col = vec![0.0f32; oh * ow * k];
    // CHANNEL OUTER, PIXEL INNER, writing 3 contiguous floats per (c, ky).
    //
    // Three shapes measured IN THE ENGINE, all on comparably quiet boxes:
    //
    //   v1  (c,ky,kx) outer, ox inner   im2col 0.176x   every write a lone f32
    //   v2  ox outer, c inner           im2col 0.278x   contiguous writes, but
    //                                                   each pixel touches cin
    //                                                   planes = cin read
    //                                                   streams at once
    //   v3  c outer, ox inner  KEPT     im2col 0.314x   one plane streamed;
    //                                                   3 contiguous floats per
    //                                                   write at stride k, with
    //                                                   the row's ow*k = 92 KB
    //                                                   destination L2-resident
    //
    // v3 was briefly reverted on a comparison between a LOADED-box v2 run and a
    // QUIET-box v3 run - the cross-run error this campaign exists to catch. On
    // like boxes v3 wins on both im2col (0.314x vs 0.278x) and detect total
    // (0.675x vs 0.646x).
    //
    // All three are handicapped identically by the NCHW input; in a converted
    // NHWC run the gather reads `cin` CONTIGUOUS floats per (pixel, tap) and
    // the question disappears.
    col.par_chunks_mut(ow * k).enumerate().for_each(|(oy, dst)| {
        for c in 0..c_in {
            let plane = &xs[c * hw..(c + 1) * hw];
            for ky in 0..3usize {
                let sy = oy * stride + ky;
                if sy == 0 || sy > h {
                    continue;
                }
                let src = &plane[(sy - 1) * w..sy * w];
                let base = c * 9 + ky * 3;
                for ox in 0..ow {
                    let d = ox * k + base;
                    for kx in 0..3usize {
                        let sx = ox * stride + kx;
                        if sx != 0 && sx <= w {
                            dst[d + kx] = src[sx - 1];
                        }
                    }
                }
            }
        }
    });
    col
}

/// One 3x3 convolution with the GEMM in the NHWC orientation.
///
/// `col_t[OHW, K] @ w_t[K, Cout]` gives `[OHW, Cout]`, which is transposed back
/// to `[Cout, OHW]` so the rest of the (still NCHW) graph is unchanged. That
/// final transpose is the handicap described on `nhwc_enabled`.
fn conv3x3_nhwc(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    act: bool,
) -> Result<Tensor> {
    let (n, c_in, h, w) = x.dims4()?;
    let c_out = weight.dim(0)?;
    let oh = (h + 2 - 3) / stride + 1;
    let ow = (w + 2 - 3) / stride + 1;
    let ohw = oh * ow;
    let k = c_in * 9;

    let col = crate::profile::timed(|p| &p.im2col, || {
        crate::cpuop::SliceOp::new("ffai-im2col3x3-t", move |xs, _| {
            Ok((im2col_t(xs, c_in, h, w, oh, ow, stride), (ohw, k).into()))
        })
        .run(x)
    })?;

    let w_t = weight_t_cached(weight, c_out, k)?;
    let y_t = crate::profile::timed(|p| &p.gemm, || col.matmul(&w_t))?;

    let bias_v = match bias {
        Some(b) => Some(b.flatten_all()?.to_vec1::<f32>()?),
        None => None,
    };
    // Transpose FUSED into the epilogue: one pass instead of two.
    let out = crate::profile::timed(|p| &p.conv_wrap, || {
        crate::epilogue::apply_transposed(y_t, bias_v, c_out, ohw, act)
    })?;
    out.reshape((n, c_out, oh, ow))
}

pub fn conv3x3_strided(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    // Apply bias AND SiLU in the epilogue, in one traversal of the GEMM's
    // output. `false` leaves the raw convolution, for the nine
    // activation-free Convs in this graph.
    act: bool,
) -> Result<Tensor> {
    // Direct convolution first: it never builds the 9x-expanded operand,
    // which is the only thing that moves the GEMM's arithmetic intensity.
    // Falls through to im2col for shapes it declines (stride 2, w < 2).
    // The direct and tiled paths have no fused epilogue, so a fused call must
    // not reach them — they would return the convolution WITHOUT its
    // activation, a wrong graph rather than a slow one. Both are env-gated
    // experiments; declining is cheaper than keeping a second fused path in
    // step with this one.
    if stride == 1 && !act && crate::direct3x3::enabled() {
        if let Some(y) = crate::direct3x3::conv3x3_direct(x, weight, bias)? {
            return Ok(y);
        }
    }
    if nhwc_enabled() {
        return conv3x3_nhwc(x, weight, bias, stride, act);
    }
    if !tiling_disabled() {
        return conv3x3_tiled(x, weight, bias, stride, act);
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
    crate::conv3x3::count_macs((c_in as u64) * 9 * (ohw as u64) * (c_out as u64));
    crate::conv3x3::count_acts((c_out as u64) * (ohw as u64));
    let col = crate::profile::timed(|p| &p.im2col, || {
        crate::cpuop::SliceOp::new("ffai-im2col3x3", move |xs, _| {
    // Uninitialised, then FULLY written by the fill below.
    //
    // `vec![0f32; n]` is a complete extra write of a buffer whose every
    // element the fill overwrites, to preserve the ~1% of it that is padding.
    // Measured on the shipped allocator at the real size distribution
    // (examples/zerocost.rs): 585 buffers of 380 KiB cost 3.1 ms per image to
    // zero, roughly 7% of a 41 ms frame — and 9.4 ms at 1 MiB, where mimalloc
    // recycles blocks rather than taking pre-zeroed pages from the OS.
    //
    // The earlier reading of this probe used the SYSTEM allocator and saw the
    // 1 MiB case go NEGATIVE, because a fresh OS mapping is already zero. The
    // allocator changed and the number changed sign; that is why this was
    // re-measured rather than taken from the note.
    //
    // The cost of doing it this way is a real safety obligation: every branch
    // of the fill must write, including the padding it used to inherit.
    let need = c_in * 9 * ohw;
    let mut col: Vec<f32> = Vec::with_capacity(need);
    {
        let col = &mut col.spare_capacity_mut()[..need];
        // `FFAI_DIANA_ZEROFILL=1` puts the redundant zero-write back, so the
        // saving can be A/B'd in one process. ONE fill path either way — the
        // arm adds only the wasted write, which is exactly the quantity under
        // test, rather than a second implementation that could differ.
        if zerofill_enabled() {
            for d in col.iter_mut() {
                d.write(0.0);
            }
        }
    // One row per (channel, ky, kx). Parallel over channels: each channel
    // owns a contiguous 9*ohw block, so the writes never overlap.
    // See `crate::parallel`: the fan-out is worth its barrier cost for a
    // single image and is pure overhead inside a parallel batch.
    let im2col_channel = |(c, block): (usize, &mut [std::mem::MaybeUninit<f32>])| {
        let plane = &xs[c * hw..(c + 1) * hw];
        for ky in 0..3usize {
            for kx in 0..3usize {
                let row: &mut [std::mem::MaybeUninit<f32>] =
                    &mut block[(ky * 3 + kx) * ohw..(ky * 3 + kx + 1) * ohw];
                for oy in 0..oh {
                    // Source row for this vertical tap. Outside the image the
                    // destination is the PADDING, and must now be written
                    // explicitly — the buffer is no longer pre-zeroed.
                    let sy = oy * stride + ky;
                    let dst = &mut row[oy * ow..(oy + 1) * ow];
                    if sy == 0 || sy > h {
                        for d in dst.iter_mut() {
                            d.write(0.0);
                        }
                        continue;
                    }
                    let src = &plane[(sy - 1) * w..sy * w];
                    if stride == 1 {
                        // Contiguous copy with a horizontal offset. The one
                        // column the shift leaves uncovered is padding.
                        match kx {
                            0 => {
                                dst[0].write(0.0);
                                for (d, s) in dst[1..].iter_mut().zip(&src[..w - 1]) {
                                    d.write(*s);
                                }
                            }
                            1 => {
                                for (d, s) in dst.iter_mut().zip(src) {
                                    d.write(*s);
                                }
                            }
                            _ => {
                                for (d, s) in dst[..w - 1].iter_mut().zip(&src[1..]) {
                                    d.write(*s);
                                }
                                dst[w - 1].write(0.0);
                            }
                        }
                    } else {
                        // sx = 2*ox + kx - 1. Solve the in-bounds range once
                        // so the inner loop is branch-free.
                        //   sx >= 0    <=> 2*ox >= 1 - kx
                        //   sx <= w-1  <=> ox <= (w - kx)/2
                        let lo = if kx == 0 { 1 } else { 0 };
                        let hi =
                            if w >= kx { ((w - kx) / 2 + 1).min(ow) } else { 0 };
                        // Everything outside [lo, hi) is padding and is now
                        // written rather than inherited.
                        for d in dst[..lo.min(ow)].iter_mut() {
                            d.write(0.0);
                        }
                        for ox in lo..hi {
                            dst[ox].write(src[ox * 2 + kx - 1]);
                        }
                        for d in dst[hi.min(ow)..].iter_mut() {
                            d.write(0.0);
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
    // SAFETY: the fill covers every channel (chunks of 9*ohw over exactly
    // `need` elements) and, within a channel, every tap and every row — each
    // branch above writes its whole `ow`-wide destination, padding included.
    #[allow(unsafe_code)]
    unsafe {
        col.set_len(need)
    };
            Ok((col, (c_in * 9, ohw).into()))
        })
        .run(x)
    })?;

    let w_mat = weight.reshape((c_out, c_in * 9))?;
    let y = crate::profile::timed(|p| &p.gemm, || w_mat.matmul(&col))?; // (Cout, OH*OW)
    // The wrapper, timed separately: bias broadcast and reshape were inside
    // the parent bucket and therefore invisible.
    crate::profile::timed(|p| &p.conv_wrap, || {
        // ONE traversal for bias and activation, the same shape the 1x1 path
        // uses. Unfused this is a `broadcast_add` — allocate, read and write
        // every element to add one number per channel — plus SiLU downstream
        // as its own candle op with its own traversal.
        //
        // The 3x3 path carries 39 % of the pipeline against 1x1's 17 %, and
        // was left unfused when 1x1 was done because its earlier refutation
        // had not yet been re-tested against the new allocator baseline.
        if !act && bias.is_none() {
            return y.reshape((n, c_out, oh, ow));
        }
        if crate::blocks::fuse_disabled() || !act {
            let mut y = y;
            if let Some(b) = bias {
                y = y.broadcast_add(&b.reshape((c_out, 1))?)?;
            }
            return y.reshape((n, c_out, oh, ow));
        }
        let bias_v: Option<Vec<f32>> = match bias {
            Some(b) => Some(b.flatten_all()?.to_vec1::<f32>()?),
            None => None,
        };
        // IN PLACE — one library op per convolution instead of two. See
        // `crate::epilogue` for the counted reason.
        let out = crate::epilogue::apply(y, bias_v, ohw, act)?;
        out.reshape((n, c_out, oh, ow))
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
            let got = conv3x3(&x, &wt, Some(&b), false).unwrap();
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
            let got = conv3x3_strided(&x, &wt, Some(&b), 2, false).unwrap();
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

/// Multiply-accumulates issued by a convolution. Counted, never timed —
/// the point is to compare our ARITHMETIC against the reference's implied
/// throughput, and a counter cannot drift.
static MACS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn count_macs(n: u64) {
    if counting() {
        MACS.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }
}

/// MACs since the last call, and reset.
pub fn take_macs() -> u64 {
    MACS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Output activation elements produced by convolutions.
///
/// Kept after the epilogue-fusion refutation because it is the number that
/// EXPLAINS it: 13.4 M elements per image is 51.2 MiB, but spread over ~120
/// convolutions that is ~430 KiB each — an L2-resident working set, not a
/// DRAM one. Pricing those touches at DRAM bandwidth is what produced an
/// 8.9 ms prize that did not exist.
static ACTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn count_acts(n: u64) {
    if counting() {
        ACTS.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Activation elements since the last call, and reset.
pub fn take_acts() -> u64 {
    ACTS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Size histogram of the im2col buffer, in power-of-two byte buckets.
///
/// The MEAN buffer (380 KiB) sits inside L2, and the last two refutations in
/// this campaign both came from pricing an L2-resident buffer at DRAM
/// bandwidth. A mean cannot answer that question when the distribution is
/// skewed — early layers are wide and shallow, late layers narrow and deep —
/// so this bins the BYTES by the size of the buffer carrying them.
static SIZE_HIST: [std::sync::atomic::AtomicU64; 24] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    [Z; 24]
};

/// Bytes, bucketed by `log2(buffer bytes)`. Index i holds buffers of
/// `[2^i, 2^(i+1))` bytes.
pub fn take_size_hist() -> Vec<(usize, u64)> {
    SIZE_HIST
        .iter()
        .enumerate()
        .map(|(i, a)| (i, a.swap(0, std::sync::atomic::Ordering::Relaxed)))
        .filter(|(_, v)| *v > 0)
        .collect()
}

pub(crate) fn count_im2col(n: usize) {
    if counting() {
        let bytes = (n * 4) as u64;
        let bucket = (63 - bytes.max(1).leading_zeros() as usize).min(23);
        SIZE_HIST[bucket].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        IM2COL_ELEMS.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// See the `col` allocation: restores the redundant zero-fill for A/B.
fn zerofill_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_ZEROFILL").is_ok_and(|v| v == "1");
            C.store(on as u8, Ordering::Relaxed);
            on
        }
        v => v == 1,
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
