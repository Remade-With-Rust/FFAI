//! The YOLO26 detection head — the **one-to-one** branch only — and decode.
//!
//! # Which branch, and why it matters
//!
//! The checkpoint carries two parallel heads: `cv2`/`cv3` (one-to-many) and
//! `one2one_cv2`/`one2one_cv3`. `Detect.forward` runs
//! `self._inference(preds["one2one"] if self.end2end else preds)` — so at
//! inference the one2many branch contributes **nothing**. It has identical
//! shapes and would produce plausible detections, which is exactly the
//! failure Mercury hit when hidden states were argmaxed as logits. The
//! conversion drops it (mission plan §8.2) and this module never names it.
//!
//! # The head's two branches
//!
//! Per level, from feature widths 64 / 128 / 256:
//!
//! ```text
//! box:  Conv(c->16, k3) -> Conv(16->16, k3) -> Conv2d(16->4, k1, raw)
//! cls:  DWConv(c->c, k3) -> Conv(c->80, k1)
//!       DWConv(80->80, k3) -> Conv(80->80, k1)
//!       Conv2d(80->80, k1, raw)
//! ```
//!
//! The two final 1x1s are raw `Conv2d` with bias and **no** `BatchNorm` in the
//! checkpoint, so the conversion passes them through unfused; every other
//! conv here is fused and carries `SiLU` — including the depthwise ones,
//! which an earlier probe missed because `DWConv` subclasses `Conv` and a
//! name-equality test skipped all twelve of them.
//!
//! # Decode
//!
//! `reg_max == 1`, so there is no DFL: the four box channels are `(l,t,r,b)`
//! distances from the anchor point. `end2end`, so `xywh` is false and boxes
//! come out xyxy:
//!
//! ```text
//! x1y1 = (anchor - lt) * stride      x2y2 = (anchor + rb) * stride
//! ```
//!
//! Anchors are cell centres — `arange(w) + 0.5` — per level, with strides
//! 8/16/32 over an 80x80 + 40x40 + 20x20 grid = 8400 positions at 640px.

use candle_core::{Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};

use crate::blocks::ConvAct;
use crate::config::Dims;
use crate::neck::NeckOutput;

/// One level's box branch: two 3x3 convs then a raw 1x1 to 4 channels.
struct BoxBranch {
    c0: ConvAct,
    c1: ConvAct,
    out: Conv2d,
}

impl BoxBranch {
    /// `c2` is the branch's internal width, `max(16, ch[0]/4, reg_max*4)` —
    /// 16 at the n tier, 32 at s.
    fn new(vb: VarBuilder, c_in: usize, c2: usize) -> Result<Self> {
        Ok(Self {
            c0: ConvAct::new(vb.pp(0), c_in, c2, 3, 1, 1, true)?,
            c1: ConvAct::new(vb.pp(1), c2, c2, 3, 1, 1, true)?,
            out: conv2d(c2, 4, 1, Conv2dConfig::default(), vb.pp(2))?,
        })
    }
}

impl Module for BoxBranch {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        raw_1x1(&self.out, &self.c1.forward(&self.c0.forward(x)?)?)
    }
}

/// The raw (activation-free) 1x1 that ends each branch, as the matmul it is.
///
/// candle's `Conv2d::forward` reaches the SAME `[c_out, c_in] @ [c_in, h*w]`
/// matmul for a 1x1 kernel — but wraps it in an element-by-element copy of
/// the input into a fresh `Vec` (through a rayon fan-out), a zero-filled
/// output it then copies the result into, and a `broadcast_add` for the
/// bias. callgrind: 31.2 M instructions for six convolutions whose
/// arithmetic is a fraction of that. `pointwise_matmul` runs the identical
/// matmul on a reshape and adds the bias in place: same products, same
/// sums, same bytes.
fn raw_1x1(conv: &Conv2d, x: &Tensor) -> Result<Tensor> {
    crate::blocks::pointwise_matmul(x, conv.weight(), conv.bias(), false)
}

/// One level's class branch: two depthwise-separable pairs then a raw 1x1.
struct ClsBranch {
    dw0: ConvAct,
    pw0: ConvAct,
    dw1: ConvAct,
    pw1: ConvAct,
    out: Conv2d,
}

impl ClsBranch {
    /// `c3` is the branch's internal width — **not `nc`**.
    ///
    /// The reference builds it as `max(ch[0], min(nc, 100))`. At the n tier
    /// that happens to equal `nc` (max(64, 80) = 80 = nc), which hid a real
    /// bug here until the s tier was probed: s wants 128, not 80. A constant
    /// that is correct only because two unrelated numbers coincide at one
    /// configuration is exactly the kind a second configuration exposes.
    fn new(vb: VarBuilder, c_in: usize, c3: usize, nc: usize) -> Result<Self> {
        Ok(Self {
            // groups == channels: depthwise. SiLU on all four, verified per
            // module rather than assumed from the class name.
            dw0: ConvAct::new(vb.pp(0).pp(0), c_in, c_in, 3, 1, c_in, true)?,
            pw0: ConvAct::new(vb.pp(0).pp(1), c_in, c3, 1, 1, 1, true)?,
            dw1: ConvAct::new(vb.pp(1).pp(0), c3, c3, 3, 1, c3, true)?,
            pw1: ConvAct::new(vb.pp(1).pp(1), c3, c3, 1, 1, 1, true)?,
            out: conv2d(c3, nc, 1, Conv2dConfig::default(), vb.pp(2))?,
        })
    }
}

impl Module for ClsBranch {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pw0.forward(&self.dw0.forward(x)?)?;
        let x = self.pw1.forward(&self.dw1.forward(&x)?)?;
        raw_1x1(&self.out, &x)
    }
}

pub struct Head {
    boxes: Vec<BoxBranch>,
    scores: Vec<ClsBranch>,
    nc: usize,
    strides: Vec<f32>,
}

impl Head {
    /// Build for a given tier.
    ///
    /// The two branch widths follow the reference's own formulas, which
    /// depend on the FINEST level's channel count (`ch[0]`) rather than on
    /// the level being built — every level shares one width.
    pub fn new(vb: VarBuilder, d: Dims, nc: usize, reg_max: usize, strides: Vec<f32>) -> Result<Self> {
        let m = vb.pp("model").pp(23);
        let levels = Self::level_channels(d);
        let c2 = (levels[0] / 4).max(16).max(reg_max * 4);
        let c3 = levels[0].max(nc.min(100));
        let mut boxes = Vec::with_capacity(3);
        let mut scores = Vec::with_capacity(3);
        for (i, &c) in levels.iter().enumerate() {
            boxes.push(BoxBranch::new(m.pp("one2one_cv2").pp(i), c, c2)?);
            scores.push(ClsBranch::new(m.pp("one2one_cv3").pp(i), c, c3, nc)?);
        }
        Ok(Self { boxes, scores, nc, strides })
    }

    /// The three feature widths the neck hands over, at this tier.
    #[must_use] 
    pub fn level_channels(d: Dims) -> [usize; 3] {
        [d.ch(256), d.ch(512), d.ch(1024)]
    }

    /// Raw per-level outputs: `(box, score)` per level, pre-decode.
    pub fn forward(&self, n: &NeckOutput) -> Result<Vec<(Tensor, Tensor)>> {
        let feats = [&n.p3, &n.p4, &n.p5];
        let mut out = Vec::with_capacity(3);
        for (i, f) in feats.iter().enumerate() {
            out.push((self.boxes[i].forward(f)?, self.scores[i].forward(f)?));
        }
        Ok(out)
    }

    /// Flatten the per-level outputs into `(boxes [1,4,A], scores [1,nc,A])`
    /// in the level order the anchor grid assumes.
    pub fn concat_levels(&self, per_level: &[(Tensor, Tensor)]) -> Result<(Tensor, Tensor)> {
        let mut bs = Vec::with_capacity(per_level.len());
        let mut ss = Vec::with_capacity(per_level.len());
        for (b, s) in per_level {
            let (n, _, h, w) = b.dims4()?;
            bs.push(b.reshape((n, 4, h * w))?);
            ss.push(s.reshape((n, self.nc, h * w))?);
        }
        // Row-wise `memcpy`s into one buffer (`cpuop::cat_last`): candle's
        // `cat` along the last axis went through `copy2d` with per-row layout
        // work and a zeroed destination for the same bytes.
        let bs: Vec<&Tensor> = bs.iter().collect();
        let ss: Vec<&Tensor> = ss.iter().collect();
        Ok((crate::cpuop::cat_last(&bs)?, crate::cpuop::cat_last(&ss)?))
    }

    /// The anchor grid, as three `(h, w, stride)` levels rather than 8400
    /// materialized tuples.
    ///
    /// It used to build a `Vec<(f32, f32, f32)>` of every grid position on
    /// **every call** — a 100 KB allocation whose contents depend only on
    /// the feature-map sizes, which are fixed for a fixed input geometry.
    /// That is the textbook "does this value depend on the input?"
    /// redundancy.
    ///
    /// The fix is not to cache the vector but to **not build one**: decode
    /// needs an anchor for the `max_det` positions it actually selects
    /// (300), not for all 8400, and position → `(x, y, stride)` is three
    /// integer ops. So this removes an allocation *and* 96% of the
    /// arithmetic, and it is byte-identical by construction.
    pub fn anchors(&self, per_level: &[(Tensor, Tensor)]) -> Result<AnchorGrid> {
        let mut levels = [(0usize, 0usize, 0f32); 3];
        for (i, (b, _)) in per_level.iter().enumerate() {
            let (_, _, h, w) = b.dims4()?;
            levels[i] = (h, w, self.strides[i]);
        }
        Ok(AnchorGrid { levels })
    }

    #[must_use] 
    pub fn nc(&self) -> usize {
        self.nc
    }

    /// Decode to `(xyxy, score, class)` triples, applying the reference's
    /// **two-stage** top-k.
    ///
    /// Stage one takes the `max_det` best anchors by their single best
    /// class score; stage two takes the `max_det` best entries from those
    /// anchors' full `anchor x class` score block. The second stage is why
    /// **one anchor can emit several detections under different classes** —
    /// a "top-k anchors, take each one's argmax class" shortcut produces
    /// different output and no shape check would notice.
    ///
    /// Done on CPU vectors rather than in tensor ops: candle has no `topk`
    /// or `gather` in the shapes this needs, batch is 1, and 8400x80 floats
    /// is a sub-millisecond scan. The reference semantics are the hard
    /// part here, not the arithmetic.
    pub fn decode(
        &self,
        boxes: &Tensor,
        scores: &Tensor,
        anchors: &AnchorGrid,
        max_det: usize,
    ) -> Result<Vec<DecodedBox>> {
        let (_, _, a) = boxes.dims3()?;
        let nc = self.nc;
        // Borrow the head's outputs IN PLACE. `flatten_all().to_vec1()` copied
        // `(4 + nc) * A` floats — half a megabyte per image at the n tier —
        // to read them once. `storage_and_layout` hands out the buffer the
        // tensor already owns; both tensors are fresh `cat` outputs, so the
        // layout is contiguous and the view is the whole buffer.
        let (b_store, b_layout) = boxes.storage_and_layout();
        let b = crate::cpuop::storage_f32(&b_store, b_layout)?;

        // RAW LOGITS. The sigmoid is deferred to the survivors.
        //
        // Sigmoid is strictly monotonic, so `max`, `argmax` and every
        // top-k ORDER are identical whether taken on logits or on
        // probabilities — the selection cannot know the difference. Applying
        // it up front evaluated it 8400 x 80 = 672,000 times to report 300
        // numbers, which is 2,240 wasted evaluations per detection.
        //
        // The transform is exact, not approximate: same survivors, same
        // order, same reported scores, because sigmoid is still applied — to
        // the ones that survive.
        let (s_store, s_layout) = scores.storage_and_layout();
        let s = crate::cpuop::storage_f32(&s_store, s_layout)?;

        // Stage one: best class per anchor, then the top `k` anchors.
        let k = max_det.min(a);
        // Class-OUTER, anchor-inner. The scores are `[nc, A]`, so the per-
        // anchor maximum is a reduction down a column — the anchor-outer
        // form read every element at stride `A` and could not vectorize.
        // Walking each class row against a running-max row is contiguous on
        // both sides and compiles to packed compares. Same strict `>`, so
        // the maximum (and its treatment of NaN) is unchanged; the result
        // only records the value, never which class produced it.
        let mut maxes = vec![f32::NEG_INFINITY; a];
        for row in s[..nc * a].chunks_exact(a) {
            for (m, &v) in maxes.iter_mut().zip(row) {
                if v > *m {
                    *m = v;
                }
            }
        }
        let mut best: Vec<(usize, f32)> = maxes.into_iter().enumerate().collect();
        // Partition, do not sort. Stage one only needs the top-k SET — its
        // order is irrelevant because stage two re-ranks the whole block. A
        // full sort of 8400 to keep 300 is ~13x the comparisons of a
        // selection, for an ordering nothing reads.
        if k < best.len() {
            best.select_nth_unstable_by(k - 1, |x, y| y.1.total_cmp(&x.1));
            best.truncate(k);
        }

        // Stage two: top `k` over those anchors' full class block.
        //
        // With a lower bound first. Every surviving anchor's own maximum is
        // one of the `k * nc` candidates, so the k-th largest of those maxima
        // (`floor`) is at most the k-th largest candidate overall: a candidate
        // strictly below `floor` cannot be in the top `k` and need not be
        // materialised, let alone partitioned. On a real image that is the
        // bulk of the block — most of an anchor's 80 class logits sit far
        // under its best one. Candidates at or above `floor` are kept
        // unconditionally, so the set the selection sees is the same at and
        // above the boundary.
        // Only when stage one actually truncated: with `k >= a` there are
        // fewer maxima than slots and the bound does not hold.
        let floor = if best.len() == k && k < a {
            best.iter().map(|&(_, m)| m).fold(f32::INFINITY, f32::min)
        } else {
            f32::NEG_INFINITY
        };
        let mut cand: Vec<(usize, usize, f32)> = Vec::with_capacity(k * nc);
        for &(i, _) in &best {
            for c in 0..nc {
                let v = s[c * a + i];
                if v >= floor {
                    cand.push((i, c, v));
                }
            }
        }
        // Here the order IS read, so: select the k best, then sort only
        // those. 300 sorted out of 24,000 instead of 24,000 sorted to keep
        // 300.
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |x, y| y.2.total_cmp(&x.2));
            cand.truncate(k);
        }
        cand.sort_by(|x, y| y.2.total_cmp(&x.2));

        Ok(cand
            .into_iter()
            .map(|(i, c, logit)| {
                // Sigmoid HERE, on the survivors only.
                let score = 1.0 / (1.0 + (-logit).exp());
                let (ax, ay, stride) = anchors.at(i);
                let (l, t, r, btm) = (b[i], b[a + i], b[2 * a + i], b[3 * a + i]);
                DecodedBox {
                    x0: (ax - l) * stride,
                    y0: (ay - t) * stride,
                    x1: (ax + r) * stride,
                    y1: (ay + btm) * stride,
                    class_id: c as u32,
                    confidence: score,
                }
            })
            .collect())
    }

    /// The `[1, 4+nc, A]` tensor the reference calls `decoded`, for the
    /// oracle. Not on the serving path — the engine goes straight to
    /// [`Self::decode`].
    pub fn decoded_tensor(
        &self,
        boxes: &Tensor,
        scores: &Tensor,
        anchors: &AnchorGrid,
    ) -> Result<Tensor> {
        let (n, _, a) = boxes.dims3()?;
        let dev = boxes.device();
        let ax: Vec<f32> = (0..a).map(|i| anchors.at(i).0).collect();
        let ay: Vec<f32> = (0..a).map(|i| anchors.at(i).1).collect();
        let st: Vec<f32> = (0..a).map(|i| anchors.at(i).2).collect();
        let ax = Tensor::from_vec(ax, (1, 1, a), dev)?;
        let ay = Tensor::from_vec(ay, (1, 1, a), dev)?;
        let st = Tensor::from_vec(st, (1, 1, a), dev)?;

        let lt = boxes.narrow(1, 0, 2)?;
        let rb = boxes.narrow(1, 2, 2)?;
        let anchor_xy = Tensor::cat(&[&ax, &ay], 1)?;
        let x1y1 = anchor_xy.broadcast_sub(&lt)?;
        let x2y2 = anchor_xy.broadcast_add(&rb)?;
        let dbox = Tensor::cat(&[&x1y1, &x2y2], 1)?.broadcast_mul(&st)?;
        let probs = candle_nn::ops::sigmoid(scores)?;
        let out = Tensor::cat(&[&dbox, &probs], 1)?;
        debug_assert_eq!(out.dims()[0], n);
        Ok(out)
    }
}

/// The anchor grid, described rather than materialized.
///
/// Three `(h, w, stride)` levels, flattened in level order — position `i`
/// resolves to its cell centre and stride by integer arithmetic. This is a
/// `Copy` 40-byte value, so sharing it across threads costs nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorGrid {
    levels: [(usize, usize, f32); 3],
}

impl AnchorGrid {
    /// Total positions across all levels (8400 at 640x640).
    #[must_use] 
    pub fn len(&self) -> usize {
        self.levels.iter().map(|(h, w, _)| h * w).sum()
    }

    #[must_use] 
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `(anchor_x, anchor_y, stride)` at flat position `i`.
    ///
    /// Cell centres are `index + 0.5`, matching `make_anchors`'s
    /// `grid_cell_offset` — the reference's own constant, not a guess.
    #[inline]
    #[must_use] 
    pub fn at(&self, i: usize) -> (f32, f32, f32) {
        let mut base = 0;
        for &(h, w, stride) in &self.levels {
            let n = h * w;
            if i < base + n {
                let local = i - base;
                return ((local % w) as f32 + 0.5, (local / w) as f32 + 0.5, stride);
            }
            base += n;
        }
        (0.0, 0.0, 1.0)
    }
}

/// One decoded detection in **letterboxed input** pixels; the engine maps
/// it back to original-image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodedBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub class_id: u32,
    pub confidence: f32,
}
