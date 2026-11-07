//! YOLO26 building blocks on candle.
//!
//! Every forward here mirrors the reference in
//! `ultralytics/nn/modules/block.py`, read from the installed source rather
//! than from a diagram, and verified end-to-end by
//! `tools/diana_verify_convert.py` before any of this was written.
//!
//! **Weights are pre-fused.** `tools/diana_convert.py` folds each
//! `conv → bn` pair into one convolution with bias, so a [`ConvAct`] here is
//! a single `Conv2d` plus an optional `SiLU` — there is no `BatchNorm` at
//! runtime. The manifest records `fused: true` and [`crate::config`]
//! refuses a file that says otherwise, because unfused weights would be a
//! different forward pass rather than a slower one.

// Feature-dependent: empty under some `par` configurations, hence the allow.
#[allow(unused_imports)]
use crate::par::prelude::*;
use candle_core::{Result, Tensor, D};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};

/// Escape hatch back to candle's grouped convolution.
fn dwconv_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_DWCONV").is_some())
}

/// Escape hatch back to candle's own `SiLU`.
/// `FFAI_DIANA_NO_FUSE=1` restores the unfused epilogue — bias as a
/// `broadcast_add`, activation as a separate downstream op — so the fusion
/// can be A/B'd in one process and the oracle run against both arms.
pub(crate) fn fuse_disabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let off = std::env::var("FFAI_DIANA_NO_FUSE").is_ok_and(|v| v == "1");
            C.store(u8::from(off), Ordering::Relaxed);
            off
        }
        v => v == 1,
    }
}

fn silu_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_SILU").is_some())
}

/// Escape hatch back to candle's own stride-2 3x3 path.
fn conv_s2_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_S2").is_some())
}

/// Escape hatch back to candle's own 3x3 convolution path.
fn conv3x3_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_CONV3").is_some())
}

/// `FFAI_DIANA_NO_MAXPOOL=1` restores candle's padded max-pool for A/B.
fn maxpool_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_MAXPOOL").is_some())
}

/// Escape hatch back to candle's own 1x1 convolution path.
fn pointwise_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_PW").is_some())
}

/// A 1x1 convolution as what it actually is: one matmul over the channels.
///
/// For `x (1, Ci, H, W)` and `w (Co, Ci, 1, 1)` the convolution is
/// `w_mat (Co, Ci) @ x_mat (Ci, H*W)`. **No im2col is involved** — a 1x1
/// kernel needs no patch gathering, only a reshape — so this moves strictly
/// less data than any convolution path can.
///
/// It exists because candle's dedicated `conv2d_1x1` measured **3.6-3.8x
/// slower than `Tensor::matmul` on identical arithmetic** at our shapes
/// (`examples/conv_scaling.rs`), at both 1 and 24 threads. This is the
/// "look for hot loops that are secretly matmuls" law: a tuned GEMM beats a
/// hand-rolled loop, and candle already ships the tuned GEMM.
pub(crate) fn pointwise_matmul(x: &Tensor, w: &Tensor, b: Option<&Tensor>, act: bool) -> Result<Tensor> {
    let bias_v: Option<Vec<f32>> = match b {
        Some(b) => Some(crate::cpuop::bias_vec(b)?),
        None => None,
    };
    pointwise_matmul_with(x, w, b, bias_v.as_deref(), act)
}

/// [`pointwise_matmul`] with the bias ALREADY extracted to a slice.
///
/// Every call used to run `flatten_all().to_vec1()` on the bias — a tensor
/// walk and an allocation for `c_out` floats that never change — ~100 times
/// per image. `ConvAct` now extracts it once at construction and passes it
/// down; the tensor form is still taken for the `broadcast_add` fallback.
pub(crate) fn pointwise_matmul_with(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    bias_v: Option<&[f32]>,
    act: bool,
) -> Result<Tensor> {
    pointwise_matmul_res(x, w, b, bias_v, act, None)
}

/// [`pointwise_matmul_with`] with an optional residual added in the epilogue's
/// own pass (`epilogue::apply_residual`).
pub(crate) fn pointwise_matmul_res(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    bias_v: Option<&[f32]>,
    act: bool,
    res: Option<&Tensor>,
) -> Result<Tensor> {
    let (n, c_in, h, wd) = x.dims4()?;
    let c_out = w.dim(0)?;
    // (Co, Ci, 1, 1) -> (Co, Ci); (1, Ci, H, W) -> (Ci, H*W)
    let w_mat = crate::conv3x3::weight_mat_cached(w, c_out, c_in)?;
    let x_mat = x.reshape((n * c_in, h * wd))?;
    let y = crate::cpuop::matmul2(&w_mat, &x_mat)?; // (Co, H*W)

    // Bias and activation in ONE traversal of the matmul's output.
    //
    // Unfused this is a `broadcast_add` — allocate a tensor, read and write
    // every element to add one number per channel — followed downstream by
    // SiLU as its own candle op with its own traversal. Two passes to apply
    // two elementwise functions to a buffer that is already right here.
    //
    // This was tried once before on the 3x3 path and measured 18/21 AGAINST.
    // That refutation was taken on the system allocator, and the allocator
    // has since changed the whole pipeline by 1.64x —
    // `codec-measurement` §11: a refutation expires when its baseline moves.
    // Re-tested here rather than assumed dead, and gated so the A/B needs no
    // rebuild.
    if fuse_disabled() {
        let mut y = y;
        if let Some(b) = b {
            y = y.broadcast_add(&b.reshape((c_out, 1))?)?;
        }
        let y = y.reshape((n, c_out, h, wd))?;
        return match res {
            Some(r) => crate::cpuop::add_assign(y, r),
            None => Ok(y),
        };
    }
    let hw = h * wd;
    let bias_v: Option<Vec<f32>> = bias_v.map(<[f32]>::to_vec);
    // The nine activation-free Convs take the SAME in-place path with `act =
    // false`: one bias add on the hot buffer, where `broadcast_add` allocated
    // a second tensor and traversed both. Same `e + bias` per element, so the
    // output is bit-identical.
    // IN PLACE on the matmul's own buffer — see `crate::epilogue`. The
    // previous form allocated a fresh Vec and wrote into it, which is an
    // allocation plus a write to cold memory the write-allocate policy has to
    // fetch first. The matmul just wrote this buffer; it is hot.
    let out = match res {
        Some(r) => crate::epilogue::apply_residual(y, bias_v, hw, act, &r.reshape((c_out, hw))?)?,
        None => crate::epilogue::apply(y, bias_v, hw, act)?,
    };
    out.reshape((n, c_out, h, wd))
}

/// A fused convolution with an optional `SiLU`.
///
/// The activation is a **per-Conv fact, not a rule**: Ultralytics shares one
/// `nn.SiLU()` instance across every `Conv` that takes the default, so
/// `PyTorch`'s module walk lists it once and omits it everywhere else, making
/// most Convs *look* activation-free. The nine genuinely activation-free
/// Convs in this graph (SPPF's `cv1`, and the attention `qkv`/`proj`/`pe`
/// plus the second FFN Conv inside each `PSABlock`) are passed `act = false`
/// at their construction sites below, matching the probed truth in
/// `corpora/refs/fixtures/yolo26n_arch.json`.
pub struct ConvAct {
    conv: Conv2d,
    act: bool,
    /// Take [`crate::dwconv`]'s kernel instead of candle's grouped path.
    ///
    /// candle has no grouped-conv kernel — it splits into `groups` separate
    /// convolutions and concatenates — which measured **72.9x slower per
    /// FLOP than the dense convolution beside it**. Set only for the exact
    /// shape that kernel implements.
    depthwise: bool,
    kind: ConvKind,
    /// The bias as plain floats, extracted ONCE here rather than per call.
    bias_v: Option<Vec<f32>>,
}

/// Which convolution family this is — for the profiler's info tier, and for
/// routing the 1x1 case to a plain matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConvKind {
    /// k=1, stride 1, no groups: exactly a matmul over the channel axis,
    /// needing no im2col at all.
    Pointwise,
    Dense3x3,
    /// The seven 3x3 stride-2 downsamples — still on candle's path.
    Stride2,
    Other,
}

impl ConvAct {
    pub fn new(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        k: usize,
        stride: usize,
        groups: usize,
        act: bool,
    ) -> Result<Self> {
        let cfg = Conv2dConfig { padding: k / 2, stride, groups, ..Default::default() };
        let depthwise = groups == c_in && groups == c_out && k == 3 && stride == 1 && groups > 1;
        let kind = if k == 1 && stride == 1 && groups == 1 {
            ConvKind::Pointwise
        } else if k == 3 && groups == 1 && stride == 1 {
            ConvKind::Dense3x3
        } else if k == 3 && groups == 1 && stride == 2 {
            ConvKind::Stride2
        } else {
            ConvKind::Other
        };
        let conv = conv2d(c_in, c_out, k, cfg, vb)?;
        let bias_v = match conv.bias() {
            Some(b) => Some(crate::cpuop::bias_vec(b)?),
            None => None,
        };
        Ok(Self { conv, act, depthwise, kind, bias_v })
    }

    /// The common case: 1-stride, undivided groups, `SiLU`.
    pub fn plain(vb: VarBuilder, c_in: usize, c_out: usize, k: usize) -> Result<Self> {
        Self::new(vb, c_in, c_out, k, 1, 1, true)
    }
}


impl ConvAct {
    /// A 1x1 conv written straight into channels `[c0, c0 + c_out)` of `dst`
    /// (a `(1, C, h, w)` concatenation buffer): the GEMM lands in its slot and
    /// the epilogue runs on that slot, so the concatenation copies nothing
    /// for this part. Returns the view of the written channels.
    ///
    /// `Some` only when this conv takes the fused pointwise path; the caller
    /// falls back to `forward` + a copy otherwise.
    pub(crate) fn forward_into(&self, x: &Tensor, dst: &Tensor, c0: usize) -> Result<Option<Tensor>> {
        let pw = self.kind == ConvKind::Pointwise && !self.depthwise && !pointwise_disabled() && !fuse_disabled();
        if !pw {
            return Ok(None);
        }
        let (n, c_in, h, wd) = x.dims4()?;
        let (_, c_dst, _, _) = dst.dims4()?;
        let c_out = self.conv.weight().dim(0)?;
        if n != 1 || c0 + c_out > c_dst || !x.is_contiguous() {
            return Ok(None);
        }
        let w_mat = self.conv.weight().reshape((c_out, c_in))?;
        let x_mat = x.reshape((c_in, h * wd))?;
        let dst_mat = dst.reshape((c_dst, h * wd))?;
        crate::cpuop::matmul2_into(&dst_mat, c0, &w_mat, &x_mat)?;
        let slot = dst_mat.narrow(0, c0, c_out)?;
        let act = self.act && !silu_disabled();
        if self.bias_v.is_some() || act {
            slot.inplace_op1(&crate::epilogue::Epilogue { bias: self.bias_v.clone(), per_channel: h * wd, act })?;
        }
        Ok(Some(dst.narrow(1, c0, c_out)?))
    }

    /// `forward(x) + res`, with the add folded into the conv epilogue when
    /// this conv takes the dense 3x3 path with the fused `SiLU` — the only
    /// shape the Bottleneck's residual ever lands on. Anything else gets the
    /// plain forward and an in-place add, so the graph is the same either way.
    pub(crate) fn forward_residual(&self, x: &Tensor, res: &Tensor) -> Result<Tensor> {
        let fusable = self.kind == ConvKind::Dense3x3
            && !self.depthwise
            && self.act
            && !silu_disabled()
            && !fuse_disabled()
            && !conv3x3_disabled();
        if fusable {
            return crate::profile::timed(|p| &p.conv, || {
                crate::conv3x3::conv3x3_residual(x, self.conv.weight(), self.conv.bias(), true, res)
            });
        }
        let pw = self.kind == ConvKind::Pointwise && !self.depthwise && !pointwise_disabled() && !fuse_disabled();
        if pw {
            let act = self.act && !silu_disabled();
            return crate::profile::timed(|p| &p.conv1x1, || {
                pointwise_matmul_res(x, self.conv.weight(), self.conv.bias(), self.bias_v.as_deref(), act, Some(res))
            });
        }
        crate::cpuop::add_assign(self.forward(x)?, res)
    }

    /// Eligible for the paired NHWC run: a dense 3x3, not depthwise, with the
    /// fused epilogue actually in play. Every condition MIRRORS the dispatch in
    /// `forward` rather than approximating it - a mismatch here is a wrong
    /// graph, not a slow one.
    pub(crate) fn nhwc_pairable(&self) -> bool {
        self.kind == ConvKind::Dense3x3
            && !self.depthwise
            && self.act
            && !silu_disabled()
            && !fuse_disabled()
            && !conv3x3_disabled()
    }

    pub(crate) fn w(&self) -> &candle_core::Tensor {
        self.conv.weight()
    }

    pub(crate) fn b(&self) -> Option<&candle_core::Tensor> {
        self.conv.bias()
    }

    pub(crate) fn act_on(&self) -> bool {
        self.act
    }
}

impl Module for ConvAct {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Info-tier scope: NESTED inside the stage buckets, so it is excluded
        // from every sum in the report. High call count, so its own overhead
        // is real — read it, then stop enabling it.
        crate::profile::timed(|p| &p.conv, || {
            // Can the 1x1 path absorb its own activation? Only when it IS the
            // path taken and the activation is ours to apply — the condition
            // mirrors the dispatch below rather than approximating it,
            // because getting it wrong yields a WRONG graph, not a slow one.
            // Same question for the 3x3 kernels, mirroring their dispatch.
            let k3_fused = self.act
                && !silu_disabled()
                && !fuse_disabled()
                && !self.depthwise
                && match self.kind {
                    ConvKind::Stride2 => !conv_s2_disabled(),
                    ConvKind::Dense3x3 => !conv3x3_disabled(),
                    _ => false,
                };
            let pw_fused = self.act
                && !silu_disabled()
                && !fuse_disabled()
                && !self.depthwise
                && self.kind == ConvKind::Pointwise
                && !pointwise_disabled();
            let dw_fused = self.act && !silu_disabled() && !fuse_disabled() && self.depthwise && !dwconv_disabled();
            // `FFAI_DIANA_NO_DWCONV=1` restores candle's grouped path — the
            // A/B arm and the shipped fallback in one knob, so the oracle
            // comparison needs no rebuild.
            let t_kernel = crate::profile::roofline_enabled().then(std::time::Instant::now);
            let y = if self.depthwise {
                crate::profile::timed(|p| &p.conv_dw, || {
                    if dwconv_disabled() {
                        self.conv.forward(x)
                    } else {
                        crate::dwconv::depthwise3x3(x, self.conv.weight(), self.conv.bias(), dw_fused)
                    }
                })?
            } else if self.kind == ConvKind::Stride2 {
                crate::profile::timed(|p| &p.conv_s2, || {
                    if conv_s2_disabled() {
                        self.conv.forward(x)
                    } else {
                        crate::conv3x3::conv3x3_strided(
                            x,
                            self.conv.weight(),
                            self.conv.bias(),
                            2,
                            k3_fused,
                        )
                    }
                })?
            } else if self.kind == ConvKind::Pointwise && !pointwise_disabled() {
                crate::profile::timed(|p| &p.conv1x1, || {
                    pointwise_matmul_with(x, self.conv.weight(), self.conv.bias(), self.bias_v.as_deref(), pw_fused)
                })?
            } else if self.kind == ConvKind::Dense3x3 {
                crate::profile::timed(|p| &p.conv3x3, || {
                    if conv3x3_disabled() {
                        self.conv.forward(x)
                    } else {
                        crate::conv3x3::conv3x3(x, self.conv.weight(), self.conv.bias(), k3_fused)
                    }
                })?
            } else {
                self.conv.forward(x)?
            };
            if crate::profile::roofline_enabled()
                && let Ok(v) = y.flatten_all().and_then(|f| f.to_vec1::<f32>()) {
                    crate::profile::census(&v);
                }
            if let Some(t0) = t_kernel {
                let (xd, yd) = (x.dims(), y.dims());
                if xd.len() == 4 && yd.len() == 4 {
                    crate::profile::record_conv(
                        crate::profile::ConvShape {
                            kind: if self.depthwise {
                                "depthwise"
                            } else {
                                match self.kind {
                                    ConvKind::Stride2 => "3x3 s2",
                                    ConvKind::Pointwise => "1x1",
                                    ConvKind::Dense3x3 => "3x3 s1",
                                    _ => "other",
                                }
                            },
                            cin: xd[1],
                            cout: yd[1],
                            hin: xd[2],
                            win: xd[3],
                            hout: yd[2],
                            wout: yd[3],
                            k: if self.kind == ConvKind::Pointwise { 1 } else { 3 },
                            depthwise: self.depthwise,
                        },
                        t0.elapsed().as_nanos() as u64,
                    );
                    crate::profile::record_order(crate::profile::ConvShape {
                        kind: if self.depthwise {
                            "depthwise"
                        } else {
                            match self.kind {
                                ConvKind::Stride2 => "3x3_s2",
                                ConvKind::Pointwise => "1x1",
                                ConvKind::Dense3x3 => "3x3_s1",
                                _ => "other",
                            }
                        },
                        cin: xd[1], cout: yd[1], hin: xd[2], win: xd[3],
                        hout: yd[2], wout: yd[3],
                        k: if self.kind == ConvKind::Pointwise { 1 } else { 3 },
                        depthwise: self.depthwise,
                    });
                }
            }
            // `!pw_fused` because the 1x1 epilogue has ALREADY applied it.
            // Without this the activation runs twice — silu(silu(x)) — which
            // is not a shape error, not a panic, and not caught by any unit
            // test of either piece: both are individually correct. It surfaced
            // as the determinism test reporting "no detections to compare",
            // sixty layers downstream of the mistake.
            if self.act && !pw_fused && !k3_fused && !dw_fused {
                crate::profile::timed(|p| &p.act, || {
                    if silu_disabled() {
                        candle_nn::ops::silu(&y)
                    } else {
                        crate::silu::silu(&y)
                    }
                })
            } else {
                Ok(y)
            }
        })
    }
}

/// `cv2(cv1(x))`, plus the input when `add`.
///
/// **The expansion is a caller's fact, not a default.** The reference's
/// `Bottleneck` defaults to `e = 0.5`, and `C3k2` takes that default for its
/// direct-Bottleneck and attention forms — but `C3`/`C3k` construct their
/// inner bottlenecks with `e = 1.0` explicitly. Hardcoding 0.5 here built
/// `model.6.m.0.m.0.cv1` as 32→16 where the checkpoint has 32→32; the
/// strict load caught it on the first run, which is the whole argument for
/// failing closed on a shape mismatch instead of loading what fits.
pub struct Bottleneck {
    cv1: ConvAct,
    cv2: ConvAct,
    add: bool,
}

impl Bottleneck {
    /// `hidden` is the expanded width — `c_out / 2` for the `e = 0.5`
    /// callers, `c_out` for the `e = 1.0` ones.
    pub fn new(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        hidden: usize,
        shortcut: bool,
        k: usize,
    ) -> Result<Self> {
        Ok(Self {
            cv1: ConvAct::plain(vb.pp("cv1"), c_in, hidden, k)?,
            cv2: ConvAct::plain(vb.pp("cv2"), hidden, c_out, k)?,
            add: shortcut && c_in == c_out,
        })
    }

    /// The `e = 0.5` form: `C3k2`'s own inner blocks.
    pub fn half(vb: VarBuilder, c_in: usize, c_out: usize, shortcut: bool, k: usize) -> Result<Self> {
        Self::new(vb, c_in, c_out, c_out / 2, shortcut, k)
    }

    /// The `e = 1.0` form: the bottlenecks inside `C3`/`C3k`.
    pub fn full(vb: VarBuilder, c_in: usize, c_out: usize, shortcut: bool, k: usize) -> Result<Self> {
        Self::new(vb, c_in, c_out, c_out, shortcut, k)
    }
}

impl Module for Bottleneck {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // THE NHWC RUN. cv1 and cv2 are consecutive 3x3 convolutions with
        // nothing between them, so this is the one place in the graph where
        // both ends of a layout region are known statically. The activation
        // stays NHWC across the pair: one transpose instead of two, and the
        // second im2col gathers CONTIGUOUSLY instead of striding.
        //
        // Gated on `nhwc_pairable`, which mirrors ConvAct::forward's dispatch
        // exactly - if either conv would have taken a different path, the pair
        // must not claim it.
        let y = if crate::conv3x3::nhwc_pair_enabled()
            && self.cv1.nhwc_pairable()
            && self.cv2.nhwc_pairable()
        {
            crate::conv3x3::conv3x3_pair_nhwc(
                x,
                self.cv1.w(),
                self.cv1.b(),
                self.cv1.act_on(),
                self.cv2.w(),
                self.cv2.b(),
                self.cv2.act_on(),
            )?
        } else if self.add {
            // Residual fused into cv2's epilogue pass (`forward_residual`).
            return self.cv2.forward_residual(&self.cv1.forward(x)?, x);
        } else {
            self.cv2.forward(&self.cv1.forward(x)?)?
        };
        if self.add {
            crate::cpuop::add_assign(y, x)
        } else {
            Ok(y)
        }
    }
}

/// C3k = C3 with 3x3 bottlenecks: `cv3(cat(m(cv1(x)), cv2(x)))`.
///
/// Note the **two** inner bottlenecks: `C3k2` constructs `C3k(c, c, 2, ...)`
/// with the repeat count hardcoded to 2, independent of the model's depth
/// scale. Confirmed against the checkpoint (`model.6.m.0.m.{0,1}`).
pub struct C3k {
    cv1: ConvAct,
    cv2: ConvAct,
    cv3: ConvAct,
    m: Vec<Bottleneck>,
}

impl C3k {
    pub fn new(vb: VarBuilder, c_in: usize, c_out: usize, n: usize) -> Result<Self> {
        let hidden = c_out / 2;
        let mut m = Vec::with_capacity(n);
        for i in 0..n {
            // e = 1.0 inside C3/C3k — see Bottleneck's docs.
            m.push(Bottleneck::full(vb.pp("m").pp(i), hidden, hidden, true, 3)?);
        }
        Ok(Self {
            cv1: ConvAct::plain(vb.pp("cv1"), c_in, hidden, 1)?,
            cv2: ConvAct::plain(vb.pp("cv2"), c_in, hidden, 1)?,
            cv3: ConvAct::plain(vb.pp("cv3"), 2 * hidden, c_out, 1)?,
            m,
        })
    }
}

impl Module for C3k {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = self.cv1.forward(x)?;
        for b in &self.m {
            y = b.forward(&y)?;
        }
        // cv2's output GEMM lands directly in the second half of the
        // concatenation buffer (`forward_into`); only `y` is copied in.
        let (_, cy, h, w) = y.dims4()?;
        let c_skip = self.cv2.conv.weight().dim(0)?;
        // SAFETY: [0, cy) written by `write_block`, [cy, cy + c_skip) by
        // `forward_into`, before cv3 reads the buffer.
        #[allow(unsafe_code)]
        let big = unsafe { crate::cpuop::uninit_chw(cy + c_skip, h, w, x.device())? };
        if self.cv2.forward_into(x, &big, cy)?.is_none() {
            let skip = self.cv2.forward(x)?;
            let t = crate::cpuop::cat_channels(&[&y, &skip])?;
            return self.cv3.forward(&t);
        }
        crate::cpuop::write_block(&big, 0, &y)?;
        crate::profile::record_plumb("cat(channel)", big.elem_count() as u64);
        self.cv3.forward(&big)
    }
}

/// Multi-head attention over a feature map, with a depthwise positional
/// encoding added to the values.
///
/// Layout, from the reference: `qkv` produces `key_dim*2 + head_dim` per
/// head; scores are `(q * key_dim^-0.5)ᵀ @ k` softmaxed over the **last**
/// axis; the output is `v @ attnᵀ` reshaped back to BCHW plus `pe(v)`.
pub struct Attention {
    qkv: ConvAct,
    proj: ConvAct,
    pe: ConvAct,
    num_heads: usize,
    key_dim: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = dim / num_heads;
        let key_dim = head_dim / 2; // attn_ratio 0.5
        let h = dim + key_dim * num_heads * 2;
        Ok(Self {
            qkv: ConvAct::new(vb.pp("qkv"), dim, h, 1, 1, 1, false)?,
            proj: ConvAct::new(vb.pp("proj"), dim, dim, 1, 1, 1, false)?,
            pe: ConvAct::new(vb.pp("pe"), dim, dim, 3, 1, dim, false)?,
            num_heads,
            key_dim,
            head_dim,
        })
    }
}

impl Module for Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::profile::timed(|p| &p.attn, || self.forward_inner(x))
    }
}

impl Attention {
    /// `forward(x) + res`, the add fused into `proj`'s epilogue.
    pub(crate) fn forward_residual(&self, x: &Tensor, res: &Tensor) -> Result<Tensor> {
        crate::profile::timed(|p| &p.attn, || self.forward_inner_res(x, Some(res)))
    }

    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner_res(x, None)
    }

    fn forward_inner_res(&self, x: &Tensor, res: Option<&Tensor>) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let n = h * w;
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((b, self.num_heads, self.key_dim * 2 + self.head_dim, n))?;
        let scale = (self.key_dim as f64).powf(-0.5);
        // One pass over qkv: scaled q (`x * mul + 0.0`, candle's affine to the
        // bit), contiguous k, contiguous v (`cpuop::split_qkv`) — was a strided
        // `affine` and two strided `contiguous()` copies, three ops.
        let (qs, k, vc) = crate::cpuop::split_qkv(&qkv, self.key_dim, self.head_dim, scale as f32)?;
        // `(q * scale)ᵀ @ k  ->  (b, heads, n, n)`, the transpose as a view.
        let qt = qs.t()?;
        let attn = qt.matmul(&k)?;
        let attn = crate::softmax::softmax_last(&attn)?;

        // v @ attnᵀ  ->  (b, heads, head_dim, n)  ->  (b, c, h, w)
        //
        // `v` is a NARROW of qkv, so it is not contiguous, and both consumers
        // below need it to be: the matmul asks explicitly, and `reshape` on a
        // non-contiguous view materialises one of its own. Written the
        // obvious way that is two copies of the same tensor, one of them
        // invisible because `reshape` does not look like an allocation.
        //
        // Made once, used twice. Same arithmetic, one fewer traversal of a
        // buffer that is already hot.
        // `v @ attnᵀ` with the transpose as a VIEW: candle's matmul hands the
        // operand strides to `gemm`, which packs the panel either way, so the
        // products and their order are unchanged — and the 2 x n x n copy the
        // transpose kernel made (3.4 M Ir per image) is not made. "Issue
        // fewer ops": the op was a copy.
        let out = vc.matmul(&attn.t()?)?.reshape((b, c, h, w))?;
        let pe = self.pe.forward(&vc.reshape((b, c, h, w))?)?;
        match res {
            Some(r) => self.proj.forward_residual(&(out + pe)?, r),
            None => self.proj.forward(&(out + pe)?),
        }
    }
}

/// Attention then a 2x-expanding FFN, each residual when `add`.
pub struct PsaBlock {
    attn: Attention,
    ffn0: ConvAct,
    ffn1: ConvAct,
    add: bool,
}

impl PsaBlock {
    pub fn new(vb: VarBuilder, c: usize, num_heads: usize, add: bool) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(vb.pp("attn"), c, num_heads)?,
            ffn0: ConvAct::plain(vb.pp("ffn").pp(0), c, c * 2, 1)?,
            ffn1: ConvAct::new(vb.pp("ffn").pp(1), c * 2, c, 1, 1, 1, false)?,
            add,
        })
    }
}

impl Module for PsaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Both residuals fused into the producing 1x1's epilogue pass
        // (`forward_residual`): no separate add traversal.
        let x = if self.add { self.attn.forward_residual(x, x)? } else { self.attn.forward(x)? };
        let h = self.ffn0.forward(&x)?;
        if self.add {
            self.ffn1.forward_residual(&h, &x)
        } else {
            self.ffn1.forward(&h)
        }
    }
}

/// What `C3k2` chains between its two split halves.
///
/// Boxed because the variants differ several-fold in size — a `C3k` carries
/// three convolutions and two bottlenecks, the attention form carries an
/// entire `PSABlock` — and an unboxed enum would size every element of the
/// `Vec` to the largest.
pub enum Inner {
    Bottleneck(Box<Bottleneck>),
    C3k(Box<C3k>),
    /// The 4-arg `attn` form: a Bottleneck followed by a `PSABlock`.
    Attn(Box<(Bottleneck, PsaBlock)>),
}

impl Module for Inner {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Bottleneck(b) => b.forward(x),
            Self::C3k(c) => c.forward(x),
            Self::Attn(bp) => bp.1.forward(&bp.0.forward(x)?),
        }
    }
}

/// Which inner module a `C3k2` layer carries, from the YAML flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnerKind {
    Bottleneck,
    C3k,
    Attn,
}

/// C2f-shaped: split `cv1`'s output in two, chain the inner module from the
/// second half, concatenate everything, then `cv2`.
///
/// The hidden width `c` is `c_out * e` and is passed in rather than derived,
/// because `e` is 0.25 for the early backbone stages and 0.5 elsewhere — a
/// derivation would silently be right for most layers and wrong for two.
pub struct C3k2 {
    cv1: ConvAct,
    cv2: ConvAct,
    m: Vec<Inner>,
    c: usize,
}

impl C3k2 {
    pub fn new(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        c: usize,
        n: usize,
        kind: InnerKind,
    ) -> Result<Self> {
        let mut m = Vec::with_capacity(n);
        for i in 0..n {
            let vbm = vb.pp("m").pp(i);
            m.push(match kind {
                InnerKind::Bottleneck => {
                    Inner::Bottleneck(Box::new(Bottleneck::half(vbm, c, c, true, 3)?))
                }
                // C3k's repeat count is hardcoded to 2 in the reference,
                // independent of the model's depth scale.
                InnerKind::C3k => Inner::C3k(Box::new(C3k::new(vbm, c, c, 2)?)),
                InnerKind::Attn => Inner::Attn(Box::new((
                    Bottleneck::half(vbm.pp(0), c, c, true, 3)?,
                    PsaBlock::new(vbm.pp(1), c, (c / 64).max(1), true)?,
                ))),
            });
        }
        Ok(Self {
            cv1: ConvAct::plain(vb.pp("cv1"), c_in, 2 * c, 1)?,
            cv2: ConvAct::plain(vb.pp("cv2"), (2 + n) * c, c_out, 1)?,
            m,
            c,
        })
    }
}

impl Module for C3k2 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // The concatenation buffer FIRST: cv1's two halves are written into
        // its head by the GEMM itself (`forward_into`), and each inner
        // block's output is copied once into its slot. The old form built
        // cv1's output, split it into views, and then `cat` copied the two
        // halves AND the inner outputs into a fresh buffer.
        let (_, _, h, w) = x.dims4()?;
        let c_all = (2 + self.m.len()) * self.c;
        // SAFETY: channels [0, 2c) are written by `forward_into`, every later
        // block of `c` channels by `write_block`, before cv2 reads the buffer.
        #[allow(unsafe_code)]
        let big = unsafe { crate::cpuop::uninit_chw(c_all, h, w, x.device())? };
        let Some(y0) = self.cv1.forward_into(x, &big, 0)? else {
            // Fallback (A/B knobs off the fused path): the copying form.
            let y0 = self.cv1.forward(x)?;
            let mut parts = vec![y0.narrow(1, 0, self.c)?, y0.narrow(1, self.c, self.c)?];
            for inner in &self.m {
                let next = inner.forward(parts.last().expect("cv1 gives two halves"))?;
                parts.push(next);
            }
            let t = crate::cpuop::cat_channels(&parts.iter().collect::<Vec<_>>())?;
            return self.cv2.forward(&t);
        };
        crate::profile::record_plumb("narrow(channel)", y0.elem_count() as u64);
        let mut last = y0.narrow(1, self.c, self.c)?;
        for (i, inner) in self.m.iter().enumerate() {
            let next = inner.forward(&last)?;
            let c0 = (2 + i) * self.c;
            crate::cpuop::write_block(&big, c0 * h * w, &next)?;
            last = big.narrow(1, c0, self.c)?;
        }
        crate::profile::record_plumb("cat(channel)", big.elem_count() as u64);
        self.cv2.forward(&big)
    }
}

/// Spatial pyramid pooling — fast. `cv1` has **no activation**, the pools
/// are cumulative (each applies to the previous pooled result), and the
/// residual is added **after** `cv2`, only when `c_in == c_out`.
pub struct Sppf {
    cv1: ConvAct,
    cv2: ConvAct,
    k: usize,
    n: usize,
    add: bool,
}

impl Sppf {
    pub fn new(vb: VarBuilder, c_in: usize, c_out: usize, k: usize, n: usize) -> Result<Self> {
        let hidden = c_in / 2;
        Ok(Self {
            cv1: ConvAct::new(vb.pp("cv1"), c_in, hidden, 1, 1, 1, false)?,
            cv2: ConvAct::plain(vb.pp("cv2"), hidden * (n + 1), c_out, 1)?,
            k,
            n,
            add: c_in == c_out,
        })
    }
}

impl Module for Sppf {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.cv1.forward(x)?;
        if !maxpool_disabled() {
            // The whole pyramid — cv1's output and its n cumulative pools —
            // written straight into the concatenated buffer cv2 reads.
            let t = crate::maxpool::maxpool_chain_cat(&x1, self.k, self.n)?;
            crate::profile::record_plumb("cat(channel)", t.elem_count() as u64);
            let y = self.cv2.forward(&t)?;
            return if self.add { crate::cpuop::add_assign(y, x) } else { Ok(y) };
        }
        let mut ys = vec![x1];
        for _ in 0..self.n {
            let prev = ys.last().expect("seeded with cv1");
            // MaxPool2d(k, stride=1, padding=k/2). candle has no padded
            // max-pool, and replication padding is EXACTLY equivalent here
            // rather than merely close: PyTorch maxes over the valid window
            // only, and every replicated value is a copy of an edge element
            // that is already inside that window, so the maximum is
            // unchanged. Zero padding would NOT be equivalent — these
            // activations are signed, and a zero would win wherever the
            // true window maximum is negative.
            //
            // Done as the separable, index-clamped pass in `crate::maxpool`
            // (bit-identical to the padded candle path, by test); the two
            // `pad_with_same` tensors and the 25-tap generic pool were 6.6 %
            // of detect's instructions for a max of 25 numbers.
            ys.push(if maxpool_disabled() {
                let pad = self.k / 2;
                let padded = prev
                    .pad_with_same(D::Minus1, pad, pad)?
                    .pad_with_same(D::Minus2, pad, pad)?;
                padded.max_pool2d_with_stride(self.k, 1)?
            } else {
                crate::maxpool::maxpool_same_s1(prev, self.k)?
            });
        }
        let y = { let t = crate::cpuop::cat_channels(&ys.iter().collect::<Vec<_>>())?; crate::profile::record_plumb("cat(channel)", t.elem_count() as u64); self.cv2.forward(&t)? };
        if self.add {
            crate::cpuop::add_assign(y, x)
        } else {
            Ok(y)
        }
    }
}

/// Split in two, run PSA blocks on the second half, concatenate, project.
pub struct C2psa {
    cv1: ConvAct,
    cv2: ConvAct,
    m: Vec<PsaBlock>,
    c: usize,
}

impl C2psa {
    pub fn new(vb: VarBuilder, c_in: usize, c_out: usize, n: usize) -> Result<Self> {
        let c = c_in / 2;
        let mut m = Vec::with_capacity(n);
        for i in 0..n {
            m.push(PsaBlock::new(vb.pp("m").pp(i), c, (c / 64).max(1), true)?);
        }
        Ok(Self {
            cv1: ConvAct::plain(vb.pp("cv1"), c_in, 2 * c, 1)?,
            cv2: ConvAct::plain(vb.pp("cv2"), 2 * c, c_out, 1)?,
            m,
            c,
        })
    }
}

impl Module for C2psa {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let a = y.narrow(1, 0, self.c)?;
        crate::profile::record_plumb("narrow(channel)", y.elem_count() as u64);
        let mut b = y.narrow(1, self.c, self.c)?;
        for blk in &self.m {
            b = blk.forward(&b)?;
        }
        { let t = crate::cpuop::cat_channels(&[&a, &b])?; crate::profile::record_plumb("cat(channel)", t.elem_count() as u64); self.cv2.forward(&t) }
    }
}


#[cfg(test)]
mod pw_fusion_tests {
    use super::*;
    use candle_core::Device;

    /// The fused 1x1 epilogue must equal bias-then-SiLU exactly.
    #[test]
    fn fused_pointwise_equals_unfused() {
        let dev = Device::Cpu;
        let (c_in, c_out, h, w) = (4usize, 6usize, 5usize, 7usize);
        let xs: Vec<f32> = (0..c_in * h * w).map(|i| (i % 19) as f32 * 0.3 - 2.0).collect();
        let ws: Vec<f32> = (0..c_out * c_in).map(|i| (i % 11) as f32 * 0.2 - 0.9).collect();
        let bs: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.17 - 0.3).collect();
        let x = Tensor::from_vec(xs, (1, c_in, h, w), &dev).unwrap();
        let wt = Tensor::from_vec(ws, (c_out, c_in, 1, 1), &dev).unwrap();
        let b = Tensor::from_vec(bs, (c_out,), &dev).unwrap();

        let fused = pointwise_matmul(&x, &wt, Some(&b), true).unwrap();
        let unfused =
            crate::silu::silu(&pointwise_matmul(&x, &wt, Some(&b), false).unwrap()).unwrap();

        assert_eq!(fused.dims(), unfused.dims(), "shape diverged");
        let f = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let u = unfused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (a, e)) in f.iter().zip(&u).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "element {i}: fused {a} vs unfused {e}");
        }
    }
}
