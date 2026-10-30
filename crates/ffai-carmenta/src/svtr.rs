//! PP-OCRv5 mobile recognition (SVTR) — the matched half of our detector.
//!
//! We pair a 2024 PP-OCRv5 DETECTOR with EasyOCR's 2019 VGG-CRNN recognizer.
//! §8.165 measured the mismatch: with detection held constant and only the
//! recognizer swapped, SVTR reads our hardest stylized pages at 40.12 % against
//! CRNN's 44.13 % (-4.01 pp) AND the clean controls at 2.66 % against 3.20 %
//! (-0.53 pp). It wins on both halves, so the 18,383-class multilingual head's
//! confusability risk did not materialise.
//!
//! ## Written against the checkpoint, not a paper
//!
//! Structure and constants come from the recorded inference program
//! (`corpora/refs/fixtures/ppocrv5_mobile_rec_graph.json`, 407 ops with their
//! weights, strides, groups and paddings), and the backbone table below is
//! GENERATED from it rather than transcribed — hand-copying 30 layers is the
//! error class that yields fluent-but-wrong text.
//!
//! §8.167 wrote a numpy reference first and matched the paddle oracle at
//! **1.3e-5**. Three hypotheses died there, two of which looked correct:
//! skipping the head convs still matched EVERY SHAPE, and `concat([h, h])`
//! scored 100 % argmax agreement while being structurally wrong. What fixed it
//! was an op's `struct_name` — `/MultiHead/SequenceEncoder/EncoderWithSVTR/` —
//! revealing that the encoder keeps a SHORTCUT concatenated with its own output
//! (960 = 480 shortcut + 480 encoder). This port carries that structure.
//!
//! `examples/svtr_oracle.rs` asserts this module against the same fixture.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{ops::softmax, Module, VarBuilder};

/// One backbone unit: conv -> +bias -> LAB -> hardswish -> LAB.
/// BatchNorm is folded into the weights at export, so only the stem has a real
/// one; every other block's `.b_0` IS the folded bias.
struct Blk(&'static str, usize, (usize, usize), (usize, usize));
/// Squeeze-excite: global avg -> conv -> relu -> conv -> hardsigmoid -> scale.
struct Se(&'static str, &'static str);

enum Stage {
    Blk(Blk),
    Se(Se),
}

/// Generated from the recorded graph; 28 blocks and 2 SE branches, in order.
fn backbone_spec() -> Vec<Stage> {
    use Stage::{Blk as B, Se as S};
    vec![
        B(Blk("conv2d_136", 16, (1, 1), (1, 1))),
        B(Blk("conv2d_137", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_138", 32, (1, 1), (1, 1))),
        B(Blk("conv2d_139", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_140", 64, (1, 1), (1, 1))),
        B(Blk("conv2d_141", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_142", 64, (2, 1), (1, 1))),
        B(Blk("conv2d_143", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_144", 128, (1, 1), (1, 1))),
        B(Blk("conv2d_145", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_146", 128, (1, 2), (1, 1))),
        B(Blk("conv2d_147", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_148", 240, (1, 1), (2, 2))),
        B(Blk("conv2d_149", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_150", 240, (1, 1), (2, 2))),
        B(Blk("conv2d_151", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_152", 240, (1, 1), (2, 2))),
        B(Blk("conv2d_153", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_154", 240, (1, 1), (2, 2))),
        B(Blk("conv2d_155", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_156", 240, (2, 1), (2, 2))),
        S(Se("conv2d_96", "conv2d_97")),
        B(Blk("conv2d_157", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_158", 480, (1, 1), (2, 2))),
        S(Se("conv2d_107", "conv2d_108")),
        B(Blk("conv2d_159", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_160", 480, (2, 1), (2, 2))),
        B(Blk("conv2d_161", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_162", 480, (1, 1), (2, 2))),
        B(Blk("conv2d_163", 1, (1, 1), (0, 0))),
    ]
}

const BN_EPS: f64 = 1e-5;
/// The final encoder norm uses a tighter epsilon than the rest (recorded).
const LN_EPS: f64 = 1e-5;
const LN_EPS_FINAL: f64 = 1e-6;
const HEADS: usize = 8;
/// `HEAD_DIM^-0.5`, precomputed — see `attention`.
///
/// # This constant was WRONG, and the guard on it could not fire
///
/// It was first written as `0.125`. `0.125` is `64^-0.5`; `HEAD_DIM` is 15,
/// whose inverse square root is `0.2581988897…` — a **2.07x** error on every
/// attention score in the model, i.e. a temperature change on every softmax.
/// The paddle oracle caught it at `max abs 6.760e-2` against a `1e-4` bar,
/// while argmax agreement stayed at **100 %**, which is exactly why argmax
/// agreement is not evidence (see `examples/svtr_oracle.rs`).
///
/// The guard was a `debug_assert!` comparing it against
/// `(HEAD_DIM as f64).powf(-0.5)`. That assertion is correct and it never
/// ran: `debug_assert!` compiles out in release, and release is the only
/// profile the oracle, the benches and production ever use. **A guard that
/// cannot fire where the code runs is not a guard.** It is now a real test
/// (`attn_scale_matches_head_dim`), which runs in both profiles.
const ATTN_SCALE: f64 = 0.258_198_889_747_161_15;

const HEAD_DIM: usize = 15;

pub struct Svtr {
    vb: VarBuilder<'static>,
    n_class: usize,
    /// Every tensor this graph fetches, materialised once.
    ///
    /// # Why a cache and not just a lookup
    ///
    /// `VarBuilder::get_unchecked` on a mmaped safetensors backend is not a
    /// map lookup. It calls `MmapedSafetensors::load`, which calls candle's
    /// `convert` — and that **allocates a fresh tensor and copies the weight
    /// out of the mapping**, every call. This graph fetches through
    /// `conv`/`bn`/`lab`/`ln`/`linear` at twelve sites, and `forward` runs the
    /// encoder blocks in a loop, so the whole parameter set was being copied
    /// once per forward, per crop, forever.
    ///
    /// A `RwLock` rather than a `RefCell` because the recognizer is shared
    /// across rayon workers; after the first forward every access is a read
    /// lock over a hit, and `Tensor::clone` is an `Arc` bump.
    cache: RwLock<HashMap<String, Tensor>>,
}

fn hardswish(x: &Tensor) -> Result<Tensor> {
    ((x + 3.0)?.clamp(0.0, 6.0)? * x)? / 6.0
}

fn hardsigmoid(x: &Tensor) -> Result<Tensor> {
    ((x / 6.0)? + 0.5)?.clamp(0.0, 1.0)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    x * candle_nn::ops::sigmoid(x)?
}

/// `FFAI_SVTR_FUSED=0` sends the backbone through the unfused candle chain
/// and candle's conv for 1x1s, so the fast path stays one variable from its
/// oracle.
fn fused() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FFAI_SVTR_FUSED").as_deref() != Ok("0"))
}

/// See [`Svtr::block_epilogue`]. Each parameter is per channel, or one value
/// broadcast to every channel.
struct Epilogue {
    bias: Vec<f32>,
    s1: Vec<f32>,
    b1: Vec<f32>,
    s2: Vec<f32>,
    b2: Vec<f32>,
}

impl candle_core::CustomOp1 for Epilogue {
    fn name(&self) -> &'static str {
        "ffai-svtr-epilogue"
    }

    fn cpu_fwd(
        &self,
        s: &candle_core::CpuStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        use crate::par::prelude::*;
        let candle_core::CpuStorage::F32(inp) = s else {
            candle_core::bail!("ffai-svtr-epilogue: f32 only")
        };
        let Some((o0, o1)) = l.contiguous_offsets() else {
            candle_core::bail!("ffai-svtr-epilogue: non-contiguous input")
        };
        let inp = &inp[o0..o1];
        let (_, c, h, w) = l.shape().dims4()?;
        let ok = |p: &[f32]| p.len() == 1 || p.len() == c;
        if !(ok(&self.bias) && ok(&self.s1) && ok(&self.b1) && ok(&self.s2) && ok(&self.b2)) {
            candle_core::bail!("ffai-svtr-epilogue: parameter length is neither 1 nor {c}")
        }
        let at = |p: &[f32], ch: usize| if p.len() == 1 { p[0] } else { p[ch] };
        // Candle's `/ 6.0` is an affine with this multiplier.
        let sixth = (1.0f64 / 6.0) as f32;
        let mut out = vec![0f32; inp.len()];
        let plane = (h * w).max(1);
        out.par_chunks_mut(plane).zip(inp.par_chunks(plane)).enumerate().for_each(
            |(i, (o, x))| {
                let ch = i % c;
                let (bias, s1, b1) = (at(&self.bias, ch), at(&self.s1, ch), at(&self.b1, ch));
                let (s2, b2) = (at(&self.s2, ch), at(&self.b2, ch));
                for (o, &x) in o.iter_mut().zip(x) {
                    let v = x + bias;
                    let v = v * s1 + b1;
                    let t = v * 1.0 + 3.0;
                    let t = if t < 0.0 { 0.0 } else { t };
                    let t = if t > 6.0 { 6.0 } else { t };
                    let v = (t * v) * sixth + 0.0;
                    *o = v * s2 + b2;
                }
            },
        );
        Ok((candle_core::CpuStorage::F32(out), l.shape().clone()))
    }
}

impl Svtr {
    pub fn new(vb: VarBuilder<'static>, n_class: usize) -> Result<Self> {
        Ok(Self { vb, n_class, cache: RwLock::new(HashMap::new()) })
    }

    /// Fetch a tensor by name, materialising it from the mapping at most once.
    ///
    /// See [`Svtr::cache`] for what this is avoiding.
    fn fetch(&self, name: &str) -> Result<Tensor> {
        // Poison-tolerant on purpose. The map is a pure memo: a panic in
        // another thread cannot leave a broken invariant here, because the
        // worst state reachable is a missing entry and the next miss re-fills
        // it. `unwrap()` on the guard would convert one unrelated panic
        // anywhere in the process into a permanently dead recognizer — a
        // user-reachable panic, which this codebase does not ship.
        if let Some(t) = self.cache.read().unwrap_or_else(PoisonError::into_inner).get(name) {
            return Ok(t.clone());
        }
        let t = self.vb.get_unchecked(name)?;
        // Counted so the win is a DETERMINISTIC number rather than a wall
        // clock: `examples/svtr_weight_traffic.rs` reports elements copied on
        // the first forward against every forward after it.
        ffai_core::cost::copy(t.elem_count() as u64);
        self.cache
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(name.to_string(), t.clone());
        Ok(t)
    }

    // Unreferenced today. Kept rather than deleted because it is the
    // shaped-weight accessor the other loaders are written against;
    // deleting it is a call for whoever owns this loader.
    #[allow(dead_code)]
    fn w(&self, name: &str, shape: (usize, usize, usize, usize)) -> Result<Tensor> {
        self.vb.get(shape, name)
    }

    /// Conv with the recorded stride/pad/groups, then the folded bias.
    fn conv(
        &self,
        x: &Tensor,
        name: &str,
        stride: (usize, usize),
        pad: (usize, usize),
        groups: usize,
        bias: bool,
    ) -> Result<Tensor> {
        let w = self.fetch(&format!("{name}.w_0"))?;
        let y = if crate::dwconv::applies(x, &w, groups) {
            // 14 of the 30 backbone convs. Candle splits a grouped conv into
            // one convolution per channel (see `dwconv`), and the kernel takes
            // the asymmetric stride directly instead of computing a stride-1
            // output and keeping every other row of it.
            crate::dwconv::depthwise(x, &w, stride, pad)?
        } else {
            self.dense(x, w, stride, pad, groups)?
        };
        if bias {
            let b = self.fetch(&format!("{name}.b_0"))?;
            y.broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?)
        } else {
            Ok(y)
        }
    }

    /// Candle's conv for everything that is not depthwise: the stem and the
    /// pointwise and head convs.
    fn dense(
        &self,
        x: &Tensor,
        w: Tensor,
        stride: (usize, usize),
        pad: (usize, usize),
        groups: usize,
    ) -> Result<Tensor> {
        let (oc, icg, kh, kw) = w.dims4()?;
        if fused() && (kh, kw) == (1, 1) && stride == (1, 1) && pad == (0, 0) && groups == 1 {
            // A 1x1 conv in NCHW IS a matmul — `(oc, ic) x (ic, h*w)` — with
            // no im2col copy and no layout permute on either side. Candle's
            // conv path builds both; 16 of the 30 backbone convs are 1x1.
            let (n, _, h, wd) = x.dims4()?;
            let y = w.reshape((oc, icg))?.broadcast_matmul(&x.reshape((n, icg, h * wd))?)?;
            return y.reshape((n, oc, h, wd));
        }
        // candle's conv2d takes ONE stride and ONE padding; the recorded model
        // uses asymmetric values (e.g. (2,1) to collapse height faster than
        // width). Emulate by padding manually and striding per axis. A
        // symmetric stride (the (2,2) stem) is candle's own, so it computes a
        // quarter of the outputs rather than all of them.
        let x = x.pad_with_zeros(2, pad.0, pad.0)?.pad_with_zeros(3, pad.1, pad.1)?;
        let native = if stride.0 == stride.1 { stride.0 } else { 1 };
        let cfg = candle_nn::Conv2dConfig { padding: 0, stride: native, dilation: 1, groups,
            ..Default::default() };
        let y = candle_nn::Conv2d::new(w, None, cfg).forward(&x)?;
        let stride = if native > 1 { (1, 1) } else { stride };
        let y = if stride.0 > 1 {
            let n = y.dim(2)?.div_ceil(stride.0);
            y.index_select(
                &Tensor::from_vec(
                    (0..n).map(|i| (i * stride.0) as u32).collect::<Vec<_>>(),
                    n,
                    y.device(),
                )?,
                2,
            )?
        } else {
            y
        };
        let y = if stride.1 > 1 {
            let n = y.dim(3)?.div_ceil(stride.1);
            y.index_select(
                &Tensor::from_vec(
                    (0..n).map(|i| (i * stride.1) as u32).collect::<Vec<_>>(),
                    n,
                    y.device(),
                )?,
                3,
            )?
        } else {
            y
        };
        Ok(y)
    }

    fn bn(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let mean = self.fetch(&format!("{name}.w_1"))?;
        let var = self.fetch(&format!("{name}.w_2"))?;
        let gamma = self.fetch(&format!("{name}.w_0"))?;
        let beta = self.fetch(&format!("{name}.b_0"))?;
        let s = (gamma / (var + BN_EPS)?.sqrt()?)?;
        let shift = (beta - (mean * &s)?)?;
        let c = s.elem_count();
        x.broadcast_mul(&s.reshape((1, c, 1, 1))?)?
            .broadcast_add(&shift.reshape((1, c, 1, 1))?)
    }

    /// `lab(hardswish(lab(conv + bias)))` in ONE pass over the activation.
    ///
    /// The unfused form is eight candle ops — bias add, two affine blocks of
    /// two ops each, and hardswish's add, clamp, multiply and divide — each a
    /// full read, write and allocation of the block's output. PdfCraft's page
    /// profile put those passes at a third of the backbone. The arithmetic
    /// below is candle's own, operation for operation (`+ 3.0` is
    /// `v * 1 + 3`, `clamp` is maximum-then-minimum, `/ 6.0` is
    /// `v * (1/6) + 0`), and Rust never contracts a multiply-add on its own, so
    /// the result is bit-identical to the unfused chain. `FFAI_SVTR_FUSED=0`
    /// restores that chain for an A/B.
    fn block_epilogue(&self, x: &Tensor, conv: &str, i: &mut usize) -> Result<Tensor> {
        let v = |t: Tensor| t.flatten_all().and_then(|t| t.to_vec1::<f32>());
        let bias = v(self.fetch(&format!("{conv}.b_0"))?)?;
        let s1 = v(self.fetch(&format!("learnable_affine_block_{i}.w_0"))?)?;
        let b1 = v(self.fetch(&format!("learnable_affine_block_{i}.w_1"))?)?;
        let s2 = v(self.fetch(&format!("learnable_affine_block_{}.w_0", *i + 1))?)?;
        let b2 = v(self.fetch(&format!("learnable_affine_block_{}.w_1", *i + 1))?)?;
        *i += 2;
        x.contiguous()?.apply_op1_no_bwd(&Epilogue { bias, s1, b1, s2, b2 })
    }

    /// Learnable affine block: per-channel scale then shift, consumed in order.
    fn lab(&self, x: &Tensor, i: &mut usize) -> Result<Tensor> {
        let w = self.fetch(&format!("learnable_affine_block_{i}.w_0"))?;
        let b = self.fetch(&format!("learnable_affine_block_{i}.w_1"))?;
        *i += 1;
        let c = w.elem_count();
        x.broadcast_mul(&w.reshape((1, c, 1, 1))?)?
            .broadcast_add(&b.reshape((1, c, 1, 1))?)
    }

    fn ln(&self, x: &Tensor, name: &str, eps: f64) -> Result<Tensor> {
        let w = self.fetch(&format!("{name}.w_0"))?;
        let b = self.fetch(&format!("{name}.b_0"))?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let d = x.broadcast_sub(&mean)?;
        let var = d.sqr()?.mean_keepdim(D::Minus1)?;
        d.broadcast_div(&(var + eps)?.sqrt()?)?
            .broadcast_mul(&w)?
            .broadcast_add(&b)
    }

    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.fetch(&format!("{name}.w_0"))?;
        let b = self.fetch(&format!("{name}.b_0"))?;
        // Instrumented so the scale-fold question below can be answered with a
        // deterministic count instead of a stopwatch.
        if let (Ok((bs, m, k)), Ok((_, n))) = (x.dims3(), w.dims2()) {
            ffai_core::cost::matmul(bs as u64, m as u64, k as u64, n as u64);
        }
        x.broadcast_matmul(&w)?.broadcast_add(&b)
    }

    fn attention(&self, x: &Tensor, qkv: &str, proj: &str) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let t = self
            .linear(x, qkv)?
            .reshape((b, n, 3, HEADS, HEAD_DIM))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        // The `powf` this constant replaced was ONE scalar call per attention
        // — not per element — sitting in front of a matmul over `b*n*c`
        // values. Hoisting it was worth approximately nothing, and getting it
        // wrong cost a 2.07x error on every score (see [`ATTN_SCALE`]). Kept
        // only because it is now correct and tested; it is not a win.
        //
        // # The scale FOLD is refuted here, and the cache is not why
        //
        // Argus folds this constant into the q third of the qkv weights at
        // load — `(q·kᵀ)s == (q·s)·kᵀ` — deleting the multiply below for ~15 %
        // of a layer. That was blocked here until the weight cache existed,
        // and it is still not worth doing, for a reason that has nothing to do
        // with the cache: **the geometry is three orders of magnitude
        // different.** Argus scales 1088 tokens x 768 dims per layer; this
        // scales 8 heads x 40 timesteps x 15 dims — 4 800 elements, twice per
        // forward — against 194.93 MFLOP of matmul. Counted by
        // `examples/svtr_weight_traffic.rs` under the workspace cost model:
        // **1.28 %** of the instrumented linear algebra, and that denominator
        // excludes the whole convolutional backbone, so the share of an actual
        // forward is lower still.
        //
        // And it would not be free: `ATTN_SCALE` is not a power of two, so
        // folding it reassociates `(Σ x·w + b)·s` into `Σ x·(w·s) + b·s` and
        // changes the rounding. The paddle oracle sits at 8.857e-5 against a
        // 1e-4 bar — **12 % of headroom left**. Spending an unknown share of
        // that to buy at most 1.28 % of the linear algebra is the wrong trade,
        // so the fold is REFUTED here rather than deferred. If the geometry
        // ever changes — a longer sequence, a wider embed — re-run the probe
        // before re-proposing it, because this verdict is about the shape, not
        // about the idea.
        let qs = t.i(0)?;
        ffai_core::cost::elementwise(qs.elem_count() as u64, 1, 1);
        let q = (qs * ATTN_SCALE)?;
        let k = t.i(1)?;
        let v = t.i(2)?;
        let a = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let a = softmax(&a, D::Minus1)?;
        let o = a.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.linear(&o, proj)
    }

    fn enc_block(
        &self,
        x: &Tensor,
        l1: &str,
        qkv: &str,
        proj: &str,
        l2: &str,
        fc1: &str,
        fc2: &str,
    ) -> Result<Tensor> {
        let x = (x + self.attention(&self.ln(x, l1, LN_EPS)?, qkv, proj)?)?;
        let h = swish(&self.linear(&self.ln(&x, l2, LN_EPS)?, fc1)?)?;
        &x + self.linear(&h, fc2)?
    }

    /// (1, 3, 48, W) normalised to [-1, 1] -> (1, T, n_class) probabilities.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Three stages on the profile's `rec_*` rows, so `FFAI_PROFILE=1`
        // says WHERE a recognizer forward goes rather than one opaque number.
        let x = crate::profile::timed(|p| &p.rec_cnn, || self.backbone(x))?;
        let (x, shortcut) = crate::profile::timed(|p| &p.rec_rnn, || self.encoder(&x))?;
        crate::profile::timed(|p| &p.rec_head, || self.head(&x, &shortcut))
    }

    fn backbone(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.bn(&self.conv(x, "conv2d_0", (2, 2), (1, 1), 1, false)?, "batch_norm2d_0")?;
        let mut lab = 0usize;
        for st in backbone_spec() {
            match st {
                Stage::Blk(Blk(name, groups, stride, pad)) => {
                    if fused() {
                        x = self.conv(&x, name, stride, pad, groups, false)?;
                        x = self.block_epilogue(&x, name, &mut lab)?;
                    } else {
                        x = self.conv(&x, name, stride, pad, groups, true)?;
                        x = self.lab(&x, &mut lab)?;
                        x = hardswish(&x)?;
                        x = self.lab(&x, &mut lab)?;
                    }
                }
                Stage::Se(Se(reduce, expand)) => {
                    let s = x.mean_keepdim(2)?.mean_keepdim(3)?;
                    let s = self.conv(&s, reduce, (1, 1), (0, 0), 1, true)?.relu()?;
                    let s = self.conv(&s, expand, (1, 1), (0, 0), 1, true)?;
                    x = x.broadcast_mul(&hardsigmoid(&s)?)?;
                }
            }
        }
        Ok(x)
    }

    /// Neck pool, the two SVTR blocks and the final norm. Returns the encoder
    /// output and the pooled SHORTCUT the head concatenates with it.
    fn encoder(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Neck pool k=(3,2) s=(3,2): H 3->1, W 80->40 (the timesteps).
        let (_, _, h, w) = x.dims4()?;
        let x = x.avg_pool2d_with_stride((3, 2), (3, 2))?;
        let _ = (h, w);
        // SHORTCUT: EncoderWithSVTR keeps the pre-reduction tensor and
        // concatenates it with the encoder output later. 960 = 480 + 480.
        let shortcut = x.clone();
        let x = swish(&self.bn(&self.conv(&x, "conv2d_131", (1, 1), (0, 1), 1, false)?, "batch_norm2d_146")?)?;
        let x = swish(&self.bn(&self.conv(&x, "conv2d_132", (1, 1), (0, 0), 1, false)?, "batch_norm2d_147")?)?;
        let (b, c, h, w) = x.dims4()?;
        let x = x.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?;
        let x = self.enc_block(&x, "layer_norm_0", "linear_0", "linear_1", "layer_norm_1", "linear_2", "linear_3")?;
        let x = self.enc_block(&x, "layer_norm_2", "linear_4", "linear_5", "layer_norm_3", "linear_6", "linear_7")?;
        let x = self.ln(&x, "layer_norm_4", LN_EPS_FINAL)?;
        Ok((x, shortcut))
    }

    fn head(&self, x: &Tensor, shortcut: &Tensor) -> Result<Tensor> {
        // Head: (B,N,C) -> (B,C,1,N) -> conv133 -> cat(shortcut) -> conv134 -> conv135
        let (b, n, c) = x.dims3()?;
        let hd = x.reshape((b, 1, n, c))?.permute((0, 3, 1, 2))?.contiguous()?;
        let hd = swish(&self.bn(&self.conv(&hd, "conv2d_133", (1, 1), (0, 0), 1, false)?, "batch_norm2d_148")?)?;
        let hd = Tensor::cat(&[shortcut, &hd], 1)?;
        let hd = swish(&self.bn(&self.conv(&hd, "conv2d_134", (1, 1), (0, 1), 1, false)?, "batch_norm2d_149")?)?;
        let hd = swish(&self.bn(&self.conv(&hd, "conv2d_135", (1, 1), (0, 0), 1, false)?, "batch_norm2d_150")?)?;
        let x = hd.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let y = self.linear(&x, "linear_8")?;
        let _ = self.n_class;
        softmax(&y, D::Minus1)
    }
}


/// CRNN-path crops are 1x64xW grayscale; SVTR wants 3x48xW **BGR**, normalised
/// (x/255 - 0.5)/0.5, height fixed at 48 with the width scaled by aspect
/// (PaddleOCR's `RecResizeImg`, image_shape [3,48,320]).
///
/// The model's own input signature is `[-1, 3, 48, -1]` — width is dynamic — so
/// a single crop is sized to its true aspect instead of being padded to 320.
/// That is what PaddleOCR does for a one-image batch (it sets `imgW` from the
/// batch's max width/height ratio), and it is also what the §8.165 ceiling probe
/// measured, so the accuracy result carries over.
///
/// BGR is not cosmetic: the checkpoint was trained on cv2-decoded images, and
/// feeding RGB silently swaps two channels of every crop.
pub fn svtr_input(
    img: &ffai_core::types::ImageBuffer,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    device: &Device,
) -> Result<Tensor> {
    const H: usize = 48;
    let (iw, ih) = (img.width as usize, img.height as usize);
    let (x1, y1) = (x1.min(iw), y1.min(ih));
    if x1 <= x0 + 1 || y1 <= y0 + 1 {
        return Err(candle_core::Error::Msg("degenerate crop".into()));
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let w = (((H as f32) * cw as f32 / ch as f32).ceil() as usize).clamp(8, 2048);
    let stride = match img.format {
        ffai_core::types::PixelFormat::Rgb8 => 3,
        ffai_core::types::PixelFormat::Rgba8 => 4,
        ffai_core::types::PixelFormat::Gray8 => 1,
    };
    // planar BGR, bilinear sample from the crop
    let mut planes = vec![0f32; 3 * H * w];
    for oy in 0..H {
        let sy = ((oy as f32 + 0.5) * ch as f32 / H as f32 - 0.5).max(0.0);
        let (y0u, fy) = ffai_core::fastmath::floor_frac_nonneg(sy);
        // `.min(ch - 1)` is a no-op on the value (`sy <= ch - 1` by
        // construction) but it is the fact the bounds-check elimination needs.
        let y0i = (y0u as usize).min(ch - 1);
        let y1i = (y0i + 1).min(ch - 1);
        // The two source rows, hoisted out of the pixel loop as slices of
        // exactly `iw * stride`. The `at` closure below used to be rebuilt per
        // pixel and indexed the whole image, so every tap carried a bounds
        // check; against a row of known length the index is provably inside.
        let row0 = &img.data[(y0 + y0i) * iw * stride..][..iw * stride];
        let row1 = &img.data[(y0 + y1i) * iw * stride..][..iw * stride];
        for ox in 0..w {
            let sx = ((ox as f32 + 0.5) * cw as f32 / w as f32 - 0.5).max(0.0);
            let (x0u, fx) = ffai_core::fastmath::floor_frac_nonneg(sx);
            let x0i = (x0u as usize).min(cw - 1);
            let x1i = (x0i + 1).min(cw - 1);
            let (i0, i1) = ((x0 + x0i) * stride, (x0 + x1i) * stride);
            for c in 0..3 {
                // source channel: BGR output from RGB input. For `stride == 1`
                // `sc` is 0, so `row[i + sc]` is exactly the old `data[i]`.
                let sc = if stride == 1 { 0 } else { 2 - c };
                let top = f32::from(row0[i0 + sc]) * (1.0 - fx)
                    + f32::from(row0[i1 + sc]) * fx;
                let bot = f32::from(row1[i0 + sc]) * (1.0 - fx)
                    + f32::from(row1[i1 + sc]) * fx;
                let v = top * (1.0 - fy) + bot * fy;
                planes[c * H * w + oy * w + ox] = (v / 255.0 - 0.5) / 0.5;
            }
        }
    }
    // R.1 ORACLE HAND-OFF. `FFAI_SVTR_DUMP=<dir>` writes the exact tensor this
    // recognizer is about to read, so the reference weights can be given
    // BYTE-IDENTICAL input under onnxruntime.
    //
    // Handing the reference our IMAGE instead would re-run its own
    // preprocessing and compare two different things — §48 learned that the
    // hard way, where a token-perfect ORT match validated the executor while a
    // white-padding bug sat untouched in the preprocessing the oracle never saw.
    //
    // Pair with `FFAI_REC_SERIAL=1`: lines are recognised in parallel above a
    // threshold, and a shared counter would not match the emitted line order.
    if let Ok(dir) = std::env::var("FFAI_SVTR_DUMP") {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::SeqCst);
        let _ = std::fs::create_dir_all(&dir);
        let mut b = Vec::with_capacity(8 + planes.len() * 4);
        b.extend_from_slice(&(w as u32).to_le_bytes());
        b.extend_from_slice(&(H as u32).to_le_bytes());
        for v in &planes {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{n:05}.bin")), b);
    }
    Tensor::from_vec(planes, (1, 3, H, w), device)
}

/// Greedy CTC over the head's label order: index 0 is the blank, 1..N are the
/// charset lines. Returns the string and the mean probability of the kept
/// (non-blank, non-repeat) steps, matching what `crnn::decode` reports.
pub fn ctc_greedy(probs: &Tensor, charset: &[String]) -> Result<(String, Option<f32>)> {
    let (_, t, n) = probs.dims3()?;
    let best = probs.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
    let conf = probs.max(D::Minus1)?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = String::new();
    let mut kept = Vec::new();
    let mut prev = u32::MAX;
    for i in 0..t {
        let k = best[i];
        if k != 0 && k != prev {
            // class k -> charset[k - 1]; the blank occupies index 0
            if let Some(s) = charset.get(k as usize - 1) {
                out.push_str(s);
            }
            kept.push(conf[i]);
        }
        prev = k;
    }
    let _ = n;
    Ok((out, crate::crnn::line_confidence(&kept)))
}

/// Per-class allow mask for SVTR's head: index 0 (blank) always true, class
/// `k` true when charset entry `k - 1` is non-empty and every character in it
/// is in `allowed`. Entries can be multi-character, so an entry is allowed
/// only when ALL of it is. A requested character no single-character entry
/// covers is an error naming it.
pub fn class_mask(charset: &[String], allowed: &str) -> Result<Vec<bool>> {
    let missing: String = allowed
        .chars()
        .filter(|c| !charset.iter().any(|e| e.chars().eq(std::iter::once(*c))))
        .collect();
    if !missing.is_empty() {
        return Err(candle_core::Error::Msg(format!(
            "charset constraint: this recognizer cannot emit {missing:?}"
        )));
    }
    let mut mask = Vec::with_capacity(charset.len() + 1);
    mask.push(true);
    mask.extend(
        charset.iter().map(|e| !e.is_empty() && e.chars().all(|c| allowed.contains(c))),
    );
    Ok(mask)
}

/// [`ctc_greedy`] restricted to the classes `mask` allows. As in the CRNN
/// twin, the kept confidence is the class's probability over the full
/// distribution, not renormalised, so a forced character reads as uncertain.
/// With every class allowed it equals `ctc_greedy`.
pub fn ctc_greedy_masked(
    probs: &Tensor,
    charset: &[String],
    mask: &[bool],
) -> Result<(String, Option<f32>)> {
    let (_, t, n) = probs.dims3()?;
    if mask.len() != n || !mask[0] {
        return Err(candle_core::Error::Msg(format!(
            "class mask has {} entries; the head has {n} classes and the blank must be allowed",
            mask.len()
        )));
    }
    let flat = probs.flatten_all()?.to_vec1::<f32>()?;
    let mut out = String::new();
    let mut kept = Vec::new();
    let mut prev = usize::MAX;
    for i in 0..t {
        let row = &flat[i * n..(i + 1) * n];
        let (k, p) = row
            .iter()
            .enumerate()
            .filter(|(j, _)| mask[*j])
            .fold((0usize, f32::NEG_INFINITY), |best, (j, &p)| if p > best.1 { (j, p) } else { best });
        if k != 0 && k != prev {
            if let Some(s) = charset.get(k - 1) {
                out.push_str(s);
            }
            kept.push(p);
        }
        prev = k;
    }
    Ok((out, crate::crnn::line_confidence(&kept)))
}

/// [`ctc_greedy`] (or [`ctc_greedy_masked`] when `mask` is given) that also
/// returns every emission with the frames it spans, for word geometry.
///
/// The per-frame choice is made exactly as in those two — the tensor argmax
/// and max unmasked, the same masked fold otherwise — and the string and
/// confidence are built by the same loop, so both are byte-identical to theirs.
/// Returns `(text, confidence, emissions, timesteps)`.
pub fn ctc_greedy_emits(
    probs: &Tensor,
    charset: &[String],
    mask: Option<&[bool]>,
) -> Result<(String, Option<f32>, Vec<crate::words::Emit>, usize)> {
    let (_, t, n) = probs.dims3()?;
    let (best, conf): (Vec<usize>, Vec<f32>) = match mask {
        None => {
            let best = probs.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
            let conf = probs.max(D::Minus1)?.flatten_all()?.to_vec1::<f32>()?;
            (best.into_iter().map(|k| k as usize).collect(), conf)
        }
        Some(mask) => {
            if mask.len() != n || !mask[0] {
                return Err(candle_core::Error::Msg(format!(
                    "class mask has {} entries; the head has {n} classes and the blank must be allowed",
                    mask.len()
                )));
            }
            let flat = probs.flatten_all()?.to_vec1::<f32>()?;
            (0..t)
                .map(|i| {
                    flat[i * n..(i + 1) * n]
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| mask[*j])
                        .fold((0usize, f32::NEG_INFINITY), |b, (j, &p)| if p > b.1 { (j, p) } else { b })
                })
                .unzip()
        }
    };
    let mut out = String::new();
    let mut kept = Vec::new();
    let mut emits: Vec<crate::words::Emit> = Vec::new();
    let mut prev = usize::MAX;
    // Whether the run `prev` belongs to was pushed as an emission.
    let mut open = false;
    for i in 0..t {
        let k = best[i];
        if k != 0 && k != prev {
            open = false;
            if let Some(s) = charset.get(k - 1) {
                out.push_str(s);
                emits.push(crate::words::Emit { text: s.clone(), t0: i, t1: i, p: conf[i] });
                open = true;
            }
            kept.push(conf[i]);
        } else if k != 0
            && open
            && let Some(e) = emits.last_mut()
        {
            e.t1 = i;
        }
        prev = k;
    }
    Ok((out, crate::crnn::line_confidence(&kept), emits, t))
}

/// One emitted run of characters, with WHERE ON THE LINE it came from.
#[derive(Debug, Clone)]
pub struct CharSpan {
    /// byte offset of this run within the decoded string
    pub byte: usize,
    /// fraction along the crop, 0..1, of the timestep that emitted it
    pub x: f32,
}

/// `ctc_greedy`, plus the x-position of every emitted character run.
///
/// F.2 of the recognition audit. Splicing a formula's LaTeX into a line needs to
/// know WHERE in the decoded string a pixel span falls, and CTC already knows:
/// the decoder walks T timesteps left to right across the resized crop, so a
/// kept timestep IS an x-position. Timestep `i` of `t` sits at `(i + 0.5) / t`
/// along the crop, which the caller maps back to page pixels through the crop
/// rectangle it built.
///
/// No model change and no second forward pass — the information was always in
/// the loop, simply discarded. Kept as a SEPARATE function so `ctc_greedy` stays
/// byte-identical: F.2's gate is that the emitted text does not move, and the
/// cheapest way to pass a byte-identity gate is to not touch the code that
/// produces the bytes.
pub fn ctc_greedy_spans(
    probs: &Tensor,
    charset: &[String],
) -> Result<(String, Option<f32>, Vec<CharSpan>)> {
    let (_, t, _) = probs.dims3()?;
    let best = probs.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
    let conf = probs.max(D::Minus1)?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = String::new();
    let mut spans = Vec::new();
    let (mut sum, mut kept) = (0f32, 0usize);
    let mut prev = u32::MAX;
    for i in 0..t {
        let k = best[i];
        if k != 0 && k != prev {
            if let Some(s) = charset.get(k as usize - 1) {
                spans.push(CharSpan {
                    byte: out.len(),
                    x: (i as f32 + 0.5) / t as f32,
                });
                out.push_str(s);
            }
            sum += conf[i];
            kept += 1;
        }
        prev = k;
    }
    Ok((out, if kept == 0 { None } else { Some(sum / kept as f32) }, spans))
}

/// Charset in head order from the file `carmenta_svtr_prepare.py` writes.
/// Entries can be multi-character (CJK ligatures are single lines), so this is
/// `Vec<String>`, not `Vec<char>` — indexing chars here would desynchronise the
/// table from the head.
pub fn load_charset(path: &std::path::Path) -> Result<Vec<String>> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| candle_core::Error::Msg(format!("charset {}: {e}", path.display())))?;
    Ok(charset_from_str(&txt))
}

/// The same parse, from text already in hand.
///
/// Split out so the bytes constructor reaches the identical charset — one
/// entry per line, in order, index = class. A charset that parsed differently
/// from the file path's would shift every decoded character by the difference
/// and look like a bad model rather than a bad loader (§8.168).
#[must_use]
pub fn charset_from_str(txt: &str) -> Vec<String> {
    txt.lines().map(std::string::ToString::to_string).collect()
}

use candle_core::IndexOp;
use std::collections::HashMap;
use std::sync::{PoisonError, RwLock};

/// Load from a safetensors file produced by `tools/carmenta_svtr_prepare.py`.
pub fn load(path: &std::path::Path, device: &Device, n_class: usize) -> Result<Svtr> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    Svtr::new(vb, n_class)
}

#[cfg(test)]
mod tests {
    use super::{ATTN_SCALE, HEAD_DIM, HEADS};

    /// The check that `debug_assert!` could not perform.
    ///
    /// `ATTN_SCALE` shipped as `0.125` — `64^-0.5` against a `HEAD_DIM` of 15
    /// — putting a 2.07x error on every attention score. A `debug_assert!`
    /// guarded it and compiled out of every profile anyone runs. This is the
    /// same assertion in a form that executes: `cargo test` builds the test
    /// harness in both debug and release, so the constant is checked either
    /// way, and it fails at build-verification time rather than in an oracle
    /// run that needs a 300 MB weight file to be present.
    #[test]
    fn attn_scale_matches_head_dim() {
        let want = (HEAD_DIM as f64).powf(-0.5);
        assert!(
            (ATTN_SCALE - want).abs() < 1e-15,
            "ATTN_SCALE is {ATTN_SCALE} but HEAD_DIM = {HEAD_DIM} wants {want}"
        );
    }

    /// A wrong `ATTN_SCALE` is only ever the visible half of a stale-geometry
    /// bug. `0.125` is `64^-0.5`, and 64 is a plausible head dim for a
    /// different model — so pin the geometry it was derived from too, and the
    /// next person editing one of the three constants is told which others
    /// move with it.
    #[test]
    fn head_geometry_is_the_one_the_weights_were_recorded_with() {
        assert_eq!(HEADS, 8, "HEADS moved; ATTN_SCALE and the qkv reshape follow");
        assert_eq!(HEAD_DIM, 15, "HEAD_DIM moved; ATTN_SCALE must be rederived");
        assert_eq!(HEADS * HEAD_DIM, 120, "embed dim per qkv third");
    }
}
