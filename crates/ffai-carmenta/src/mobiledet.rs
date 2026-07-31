//! PP-OCRv5 mobile text detection on candle — DBNet head over a PP-LCNetV3
//! backbone and an RSEFPN neck (Apache-2.0, plan §7.1 audit).
//!
//! Why this exists alongside CRAFT: CRAFT is a VGG16 that emits character
//! heatmaps we then have to reconstruct into word boxes, and §8.13 measured
//! that reconstruction as a 16-point CER cause with median IoU 0.537 against
//! ground truth. DBNet emits the regions directly, from 4.7 MB of weights.
//! One brick, both open fronts.
//!
//! ## The weights are pre-fused, and that is load-bearing
//!
//! PP-LCNetV3 trains every convolution as a sum of parallel branches (an
//! identity BatchNorm, an optional 1x1, and four kxk convs, each with its own
//! BN) scaled by a learnable affine. All of it is linear, so
//! `tools/carmenta_mobiledet_fuse.py` collapses it offline — 905 tensors to
//! 168 — and this module only ever sees plain convolutions. Nothing here knows
//! what a rep branch is.
//!
//! ## Three things measurement decided, and reading could not
//!
//! The det variant of PP-LCNetV3 is not vendored in any package on this box,
//! and the two sources that describe its parts disagree. Rather than pick the
//! plausible reading, `tools/carmenta_mobiledet_ref.py` searched the space
//! against paddle's own exported program:
//!
//! - the backbone's squeeze-excitation uses paddle's **1/6** hardsigmoid slope;
//! - the neck's squeeze-excitation uses PaddleOCR's **0.2**;
//! - RSELayer applies its residual shortcut.
//!
//! They genuinely differ between backbone and neck. The uniform reading that
//! the source text suggests ("0.2 everywhere") costs 0.55 of logit error —
//! enough to degrade every box while still looking like a working detector.
//!
//! A fourth trap is handled in the fusion rather than here: `LearnableRepLayer`
//! builds its post-activation unconditionally but applies it only when
//! stride != 2, so the checkpoint carries parameters for stride-2 layers that
//! must never be used. The fused file simply omits them, which is why
//! [`RepConv`] can decide by presence instead of by rule.
//!
//! Oracle: `tests/mobiledet_oracle.rs` checks the probability map against
//! paddle's exported program on a pinned page.

use candle_core::{Result, Tensor, D};
use candle_nn::{
    conv2d, conv2d_no_bias, conv_transpose2d, Conv2d, Conv2dConfig, ConvTranspose2d,
    ConvTranspose2dConfig, Module, VarBuilder,
};

/// Squeeze-excitation hardsigmoid slopes — measured, not read. See module docs.
const BACKBONE_SE_SLOPE: f64 = 1.0 / 6.0;
const NECK_SE_SLOPE: f64 = 0.2;

/// `(name, depthwise kernel, stride, in, out)` for the 14 backbone blocks.
///
/// The strides are not a guess either: a rep layer carries an identity branch
/// iff `in == out && stride == 1`, and a depthwise conv always has `in == out`,
/// so the 19 identity branches in the checkpoint pin every depthwise stride
/// exactly. Stages 3/4/5/6 downsample at index 0, giving taps at 1/4, 1/8,
/// 1/16 and 1/32.
const BLOCKS: [(&str, usize, usize, usize, usize); 14] = [
    ("blocks2.0", 3, 1, 16, 32),
    ("blocks3.0", 3, 2, 32, 48),
    ("blocks3.1", 3, 1, 48, 48),
    ("blocks4.0", 3, 2, 48, 96),
    ("blocks4.1", 3, 1, 96, 96),
    ("blocks5.0", 3, 2, 96, 192),
    ("blocks5.1", 5, 1, 192, 192),
    ("blocks5.2", 5, 1, 192, 192),
    ("blocks5.3", 5, 1, 192, 192),
    ("blocks5.4", 5, 1, 192, 192),
    ("blocks6.0", 5, 2, 192, 384),
    ("blocks6.1", 5, 1, 384, 384),
    ("blocks6.2", 5, 1, 384, 384),
    ("blocks6.3", 5, 1, 384, 384),
];

/// Which block output feeds which neck level, and the tap's output width.
const TAPS: [(&str, usize, usize); 4] =
    [("blocks3.1", 48, 12), ("blocks4.1", 96, 18), ("blocks5.4", 192, 42), ("blocks6.3", 384, 360)];

fn hardswish(x: &Tensor) -> Result<Tensor> {
    x.mul(&x.affine(1.0, 3.0)?.clamp(0.0, 6.0)?.affine(1.0 / 6.0, 0.0)?)
}

fn hardsigmoid(x: &Tensor, slope: f64) -> Result<Tensor> {
    x.affine(slope, 0.5)?.clamp(0.0, 1.0)
}

/// A fused rep layer: one convolution, then hardswish and a scalar affine —
/// but only when the fused file carried that affine, which it does exactly
/// when the original layer had stride 1.
struct RepConv {
    conv: Conv2d,
    act: Option<(f64, f64)>,
}

impl RepConv {
    fn new(vb: &VarBuilder, name: &str, cin: usize, cout: usize, k: usize, stride: usize, groups: usize) -> Result<Self> {
        let vb = vb.pp(name);
        let cfg = Conv2dConfig { padding: k / 2, stride, groups, ..Default::default() };
        let act = if vb.contains_tensor("act.lab.scale") {
            let s = vb.get(1, "act.lab.scale")?.to_vec1::<f32>()?[0] as f64;
            let b = vb.get(1, "act.lab.bias")?.to_vec1::<f32>()?[0] as f64;
            Some((s, b))
        } else {
            None
        };
        Ok(RepConv { conv: conv2d(cin, cout, k, cfg, vb)?, act })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        match self.act {
            Some((s, b)) => hardswish(&x)?.affine(s, b),
            None => Ok(x),
        }
    }
}

/// Squeeze-excitation: global average, two 1x1s with ReLU between, hardsigmoid,
/// then a channel-wise rescale of the input.
struct Se {
    conv1: Conv2d,
    conv2: Conv2d,
    slope: f64,
}

impl Se {
    fn new(vb: &VarBuilder, name: &str, ch: usize, reduced: usize, slope: f64) -> Result<Self> {
        let vb = vb.pp(name);
        let cfg = Conv2dConfig::default();
        Ok(Se {
            conv1: conv2d(ch, reduced, 1, cfg, vb.pp("conv1"))?,
            conv2: conv2d(reduced, ch, 1, cfg, vb.pp("conv2"))?,
            slope,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = x.mean_keepdim(D::Minus1)?.mean_keepdim(D::Minus2)?;
        let w = self.conv2.forward(&self.conv1.forward(&w)?.relu()?)?;
        x.broadcast_mul(&hardsigmoid(&w, self.slope)?)
    }
}

/// One depthwise-separable block: depthwise, optional SE, pointwise.
struct Block {
    dw: RepConv,
    se: Option<Se>,
    pw: RepConv,
}

/// RSELayer — a bias-free convolution followed by `x + se(x)`.
struct RseLayer {
    conv: Conv2d,
    se: Se,
}

impl RseLayer {
    fn new(vb: &VarBuilder, name: &str, cin: usize, cout: usize, k: usize) -> Result<Self> {
        let vb = vb.pp(name);
        let cfg = Conv2dConfig { padding: k / 2, ..Default::default() };
        Ok(RseLayer {
            conv: conv2d_no_bias(cin, cout, k, cfg, vb.pp("in_conv"))?,
            se: Se::new(&vb, "se_block", cout, cout / 4, NECK_SE_SLOPE)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        &x + self.se.forward(&x)?
    }
}

pub struct MobileDet {
    stem: Conv2d,
    blocks: Vec<Block>,
    taps: Vec<Conv2d>,
    ins: Vec<RseLayer>,
    inp: Vec<RseLayer>,
    head1: Conv2d,
    head2: ConvTranspose2d,
    head3: ConvTranspose2d,
}

impl MobileDet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let bb = vb.pp("backbone");
        let stem = conv2d(3, 16, 3, Conv2dConfig { padding: 1, stride: 2, ..Default::default() }, bb.pp("conv1"))?;

        let mut blocks = Vec::with_capacity(BLOCKS.len());
        for &(name, k, stride, cin, cout) in BLOCKS.iter() {
            let vbb = bb.pp(name);
            // The depthwise conv keeps the block's input width; only the
            // pointwise conv changes it.
            let dw = RepConv::new(&vbb, "dw_conv", cin, cin, k, stride, cin)?;
            let se = if vbb.contains_tensor("se.conv1.weight") {
                Some(Se::new(&vbb, "se", cin, cin / 4, BACKBONE_SE_SLOPE)?)
            } else {
                None
            };
            let pw = RepConv::new(&vbb, "pw_conv", cin, cout, 1, 1, 1)?;
            blocks.push(Block { dw, se, pw });
        }

        let mut taps = Vec::with_capacity(4);
        let mut ins = Vec::with_capacity(4);
        let mut inp = Vec::with_capacity(4);
        for (i, &(_, cin, cout)) in TAPS.iter().enumerate() {
            taps.push(conv2d(cin, cout, 1, Conv2dConfig::default(), bb.pp(format!("layer_list.{i}")))?);
            ins.push(RseLayer::new(&vb.pp("neck"), &format!("ins_conv.{i}"), cout, 96, 1)?);
            inp.push(RseLayer::new(&vb.pp("neck"), &format!("inp_conv.{i}"), 96, 24, 3)?);
        }

        // DBHead, `binarize` branch only — `thresh` is training-only and the
        // fused file does not carry it. The two transposed convolutions have
        // kernel == stride == 2, so they tile the 1/4 map back to full
        // resolution without overlap.
        let hb = vb.pp("head").pp("binarize");
        let tcfg = ConvTranspose2dConfig { stride: 2, ..Default::default() };
        Ok(MobileDet {
            stem,
            blocks,
            taps,
            ins,
            inp,
            head1: conv2d(96, 24, 3, Conv2dConfig { padding: 1, ..Default::default() }, hb.pp("conv1"))?,
            head2: conv_transpose2d(24, 24, 2, tcfg, hb.pp("conv2"))?,
            head3: conv_transpose2d(24, 1, 2, tcfg, hb.pp("conv3"))?,
        })
    }

    /// `[1, 3, H, W]` normalised input -> `[1, 1, H, W]` text probability map.
    ///
    /// `H` and `W` must be multiples of 32: the backbone downsamples five times
    /// and the neck's top-down adds require the levels to line up exactly.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        if h % 32 != 0 || w % 32 != 0 {
            candle_core::bail!("mobiledet input must be a multiple of 32, got {h}x{w}");
        }

        let mut cur = self.stem.forward(x)?;
        let mut feats: Vec<Tensor> = Vec::with_capacity(4);
        for (i, blk) in self.blocks.iter().enumerate() {
            cur = blk.dw.forward(&cur)?;
            if let Some(se) = &blk.se {
                cur = se.forward(&cur)?;
            }
            cur = blk.pw.forward(&cur)?;
            if let Some(t) = TAPS.iter().position(|&(n, _, _)| n == BLOCKS[i].0) {
                feats.push(self.taps[t].forward(&cur)?);
            }
        }

        // RSEFPN: lateral 1x1s, top-down nearest-neighbour adds, per-level 3x3,
        // then every level lifted to 1/4 and concatenated into 96 channels.
        let lat = feats
            .iter()
            .enumerate()
            .map(|(i, f)| self.ins[i].forward(f))
            .collect::<Result<Vec<_>>>()?;
        let mut outs = vec![lat[3].clone()];
        for i in (0..3).rev() {
            let (_, _, lh, lw) = lat[i].dims4()?;
            let up = outs.last().unwrap().upsample_nearest2d(lh, lw)?;
            outs.push((&lat[i] + up)?);
        }
        outs.reverse(); // outs[i] is now neck level i

        let (_, _, ph, pw) = outs[0].dims4()?;
        let fused = Tensor::cat(
            &[
                self.inp[3].forward(&outs[3])?.upsample_nearest2d(ph, pw)?,
                self.inp[2].forward(&outs[2])?.upsample_nearest2d(ph, pw)?,
                self.inp[1].forward(&outs[1])?.upsample_nearest2d(ph, pw)?,
                self.inp[0].forward(&outs[0])?,
            ],
            1,
        )?;

        let y = self.head1.forward(&fused)?.relu()?;
        let y = self.head2.forward(&y)?.relu()?;
        candle_nn::ops::sigmoid(&self.head3.forward(&y)?)
    }
}

// ---------------------------------------------------------------------------
// DB postprocess
// ---------------------------------------------------------------------------

/// `inference.yml`, pinned: these are the reference's own values.
pub const BIN_THRESHOLD: f32 = 0.3;
pub const BOX_THRESHOLD: f32 = 0.6;

/// How far to unclip, per RECOGNIZER — which is where a gate did say otherwise.
///
/// The reference's single 1.5 is right for a line reader and badly wrong for a
/// word reader, and the split is not subtle. Swept on the CORD train split:
/// CRNN runs 70.3 / 47.4 / 41.6 / 37.0 / **32.0** / 31.8 / 32.6 / 35.2 % across
/// 0.0 -> 2.8, so its optimum sits at the reference value (1.9 is 0.26 pp
/// better, inside the noise of 15 clips, and not worth moving a default for).
/// PARSeq runs 45.6 / 32.8 / **32.3** / 36.5 / 41.3 / 45.4 % — optimum at
/// **0.8**, and 9 points better than the reference default.
///
/// The mechanism is the crop each recognizer consumes. CRNN reads a whole line
/// and wants generous context around it; PARSeq reads WORDS recovered from an
/// ink-gap projection inside that box, and a loose box fills the gaps with
/// background until the projection can no longer find them.
///
/// Confirmed on a second corpus class rather than assumed to generalise: on
/// frames, PARSeq reads 1.83 % at 0.8 against 5.42 % at 1.5.
pub const UNCLIP_LINE: f32 = 1.5;
pub const UNCLIP_WORD: f32 = 0.8;
const MIN_SIDE: f32 = 3.0;

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Andrew's monotone chain, counter-clockwise, no collinear points.
fn convex_hull(mut p: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    if p.len() < 3 {
        return p;
    }
    p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    p.dedup();
    let cross = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    if p.len() < 3 {
        return p;
    }
    let mut hull: Vec<(f32, f32)> = Vec::with_capacity(p.len() * 2);
    for &pt in p.iter() {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0.0 {
            hull.pop();
        }
        hull.push(pt);
    }
    // The upper chain may not eat into the lower one, so its floor is the
    // lower hull's length plus one.
    let floor = hull.len() + 1;
    for &pt in p.iter().rev().skip(1) {
        while hull.len() >= floor && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0.0 {
            hull.pop();
        }
        hull.push(pt);
    }
    hull.pop(); // the closing point repeats p[0]
    hull
}

/// A rotated rectangle as an origin-projected interval pair on an orthonormal
/// basis — the form rotating calipers produces and the form unclipping wants,
/// since growing the rectangle is just widening both intervals.
#[derive(Clone, Copy)]
struct RotRect {
    u: (f32, f32),
    lo_u: f32,
    hi_u: f32,
    lo_v: f32,
    hi_v: f32,
}

impl RotRect {
    fn dims(&self) -> (f32, f32) {
        (self.hi_u - self.lo_u, self.hi_v - self.lo_v)
    }

    /// The four corners, in the winding order `get_mini_boxes` produces.
    fn corners(&self) -> [(f32, f32); 4] {
        let (ux, uy) = self.u;
        let pt = |su: f32, sv: f32| (su * ux + sv * -uy, su * uy + sv * ux);
        [
            pt(self.lo_u, self.lo_v),
            pt(self.hi_u, self.lo_v),
            pt(self.hi_u, self.hi_v),
            pt(self.lo_u, self.hi_v),
        ]
    }

    /// Axis-aligned bounds of the four corners.
    fn aabb(&self) -> (f32, f32, f32, f32) {
        let (ux, uy) = self.u;
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &su in &[self.lo_u, self.hi_u] {
            for &sv in &[self.lo_v, self.hi_v] {
                let (x, y) = (su * ux + sv * -uy, su * uy + sv * ux);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
        (x0, y0, x1, y1)
    }
}

/// Minimum-area enclosing rectangle by rotating calipers: the optimal rectangle
/// always has a side flush with a hull edge, so testing every edge suffices.
fn min_area_rect(hull: &[(f32, f32)]) -> Option<RotRect> {
    if hull.len() < 2 {
        return None;
    }
    let mut best: Option<(f32, RotRect)> = None;
    for i in 0..hull.len() {
        let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            continue;
        }
        let u = (dx / len, dy / len);
        let (mut lo_u, mut hi_u, mut lo_v, mut hi_v) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &(px, py) in hull {
            let pu = px * u.0 + py * u.1;
            let pv = px * -u.1 + py * u.0;
            lo_u = lo_u.min(pu);
            hi_u = hi_u.max(pu);
            lo_v = lo_v.min(pv);
            hi_v = hi_v.max(pv);
        }
        let r = RotRect { u, lo_u, hi_u, lo_v, hi_v };
        let (w, h) = r.dims();
        if best.is_none() || w * h < best.unwrap().0 {
            best = Some((w * h, r));
        }
    }
    best.map(|(_, r)| r)
}

/// Mean probability inside a quadrilateral — the reference's `box_score_fast`,
/// reproduced including the part that looks like an accident.
///
/// Paddle shifts the corners to the crop's origin and then **truncates them to
/// int32** before `cv2.fillPoly`, so the polygon that gets rasterised is not
/// the polygon that was measured. Testing exact float containment instead is
/// more accurate and measurably wrong for our purpose: it scored two fixture
/// boxes 0.06 HIGH, and since this score gates acceptance at 0.6, running high
/// means admitting boxes the reference rejects. Fidelity to the threshold's
/// calibration beats fidelity to geometry.
///
/// ## Known residual, measured and bounded
///
/// Fed cv2's OWN rectangles, this agrees to 0.0000 on axis-aligned quads —
/// identical pixel counts — and runs up to **+0.047** high on rotated ones.
/// The cause is isolated: `cv2.fillPoly` rasterises edges in 16-bit fixed point
/// via its own edge-collection walk, and on a quad rotated a fraction of a
/// degree that claims one more row than any scanline rule reproduces. Matching
/// it exactly means reimplementing that walk.
///
/// Left alone deliberately. The bias is one-directional (we accept slightly
/// more), it only reaches boxes whose true score lands in [0.553, 0.6], and it
/// vanishes for the axis-aligned text that dominates these corpora — §8.8
/// measured deskew at 0.1 pp. Whether it costs anything is a question for
/// end-to-end CER, not for a scoring microbenchmark.
fn box_score_fast(prob: &[f32], w: usize, h: usize, quad: &[(f32, f32); 4]) -> f32 {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(x, y) in quad {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let xmin = (x0.floor().max(0.0) as usize).min(w - 1);
    let xmax = (x1.ceil().max(0.0) as usize).min(w - 1);
    let ymin = (y0.floor().max(0.0) as usize).min(h - 1);
    let ymax = (y1.ceil().max(0.0) as usize).min(h - 1);

    // Corners relative to the crop, truncated toward zero exactly as
    // `astype(np.int32)` does.
    let p: Vec<(f32, f32)> = quad
        .iter()
        .map(|&(x, y)| ((x - xmin as f32).trunc(), (y - ymin as f32).trunc()))
        .collect();

    // The quad is convex, so each scanline meets it in exactly one interval and
    // min/max over the crossings is the whole answer — no pair matching. The
    // rule is CLOSED in y: a half-open rule drops the extreme rows, which cost
    // ~6% of the covered pixels and scored every box high.
    let (mut sum, mut n) = (0f64, 0usize);
    for row in 0..=(ymax - ymin) {
        let yf = row as f32;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for i in 0..4 {
            let (ax, ay) = p[i];
            let (bx, by) = p[(i + 1) % 4];
            if yf < ay.min(by) || yf > ay.max(by) {
                continue;
            }
            if (ay - by).abs() < f32::EPSILON {
                lo = lo.min(ax).min(bx);
                hi = hi.max(ax).max(bx);
            } else {
                let x = ax + (yf - ay) / (by - ay) * (bx - ax);
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if hi < lo {
            continue;
        }
        let c0 = lo.round().max(0.0) as usize;
        let c1 = (hi.round().max(0.0) as usize).min(xmax - xmin);
        for col in c0..=c1 {
            sum += prob[(ymin + row) * w + xmin + col] as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64) as f32
    }
}

/// Probability map -> word boxes in SOURCE image coordinates.
///
/// This is PaddleOCR's `DBPostProcess`, with one deliberate deviation recorded
/// rather than hidden: the reference emits rotated quadrilaterals and we emit
/// their axis-aligned bounds, because everything downstream — line grouping and
/// crop extraction — is axis-aligned. §8.8 measured deskew at 0.1 pp on these
/// corpora, so the rotation is not currently worth the cascade. The rotated
/// rect is still computed, because its area and perimeter drive the unclip.
///
/// The unclip is not optional. DB is trained against *shrunk* polygons, so a
/// raw thresholded region is systematically tighter than the text it covers —
/// the same tightness §8.13 measured as a 16-point CER cause with CRAFT. The
/// expansion is what makes these boxes crop-ready.
pub fn boxes_from_probability(
    prob: &[f32],
    w: usize,
    h: usize,
    sx: f32,
    sy: f32,
    unclip: f32,
) -> Vec<crate::boxes::DetBox> {
    let bin_thr = env_f32("FFAI_DB_BIN", BIN_THRESHOLD);
    let box_thr = env_f32("FFAI_DB_BOX", BOX_THRESHOLD);
    let ratio = env_f32("FFAI_DB_UNCLIP", unclip);

    let mut mask: Vec<bool> = prob.iter().map(|&p| p > bin_thr).collect();
    let mut out = Vec::new();
    let mut stack = Vec::new();
    let mut pts: Vec<(f32, f32)> = Vec::new();

    for start in 0..w * h {
        if !mask[start] {
            continue;
        }
        // 8-connectivity, matching cv2.findContours' notion of a foreground
        // blob; 4-connectivity splits diagonal strokes into separate words.
        pts.clear();
        stack.push(start);
        mask[start] = false;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            pts.push((x as f32, y as f32));
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] {
                        mask[j] = false;
                        stack.push(j);
                    }
                }
            }
        }

        let Some(rect) = min_area_rect(&convex_hull(pts.clone())) else { continue };
        let (rw, rh) = rect.dims();
        if rw.min(rh) < MIN_SIDE {
            continue;
        }

        // box_score_fast, scored BEFORE unclipping — expanding first would drag
        // background into the mean and reject real text.
        let score = box_score_fast(prob, w, h, &rect.corners());
        if score < box_thr {
            continue;
        }

        // Unclip by the Vatti offset the reference uses: distance =
        // area * ratio / perimeter, applied outward on every side.
        let d = rw * rh * ratio / (2.0 * (rw + rh)).max(1e-6);
        let grown = RotRect {
            lo_u: rect.lo_u - d,
            hi_u: rect.hi_u + d,
            lo_v: rect.lo_v - d,
            hi_v: rect.hi_v + d,
            ..rect
        };
        let (gw, gh) = grown.dims();
        if gw.min(gh) < MIN_SIDE + 2.0 {
            continue;
        }

        let (gx0, gy0, gx1, gy1) = grown.aabb();
        let (x0, y0) = ((gx0 / sx).round().max(0.0) as usize, (gy0 / sy).round().max(0.0) as usize);
        let x1 = ((gx1 / sx).round().max(0.0) as usize).min((w as f32 / sx).round() as usize);
        let y1 = ((gy1 / sy).round().max(0.0) as usize).min((h as f32 / sy).round() as usize);
        if x1 > x0 && y1 > y0 {
            out.push(crate::boxes::DetBox { x0, y0, x1, y1, score });
        }
    }
    out
}
