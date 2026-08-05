//! The `Depth` head — the only layer that differs between `yolo26n.pt` and
//! `yolo26n-depth.pt`.
//!
//! Probed side by side, the two checkpoints share a byte-identical backbone
//! YAML and 12 of 13 neck layers. The final entry is the whole delta:
//!
//! ```text
//! detect: [[16, 19, 22], 1, 'Detect', ['nc']]
//! depth:  [[16, 19, 22], 1, 'Depth',  [256, 'log']]
//! ```
//!
//! Same P3/P4/P5 taps, same everything upstream. So this module is the entire
//! cost of the depth task, and [`crate::backbone`] / [`crate::neck`] are
//! reused unchanged — already oracle-gated across five tiers.
//!
//! # The shape, from the checkpoint
//!
//! | block | layers |
//! |---|---|
//! | `proj` | 3 x Conv1x1 + BN + SiLU, each level -> 256 |
//! | `refine[i]` | 2 x Conv3x3 + BN + SiLU, 256 -> 256 |
//! | `head` | Conv3x3 256->128 · ConvTranspose 128->128 k2 s2 · Conv3x3 128->64 · Conv2d 64->1 k1 |
//!
//! Output is `(1, 1, H/4, W/4)` in **metres**, unbounded: `exp` of a clamped
//! logit rather than a scaled sigmoid, which is what lets one model span an
//! indoor corridor and an outdoor street without a per-scene range.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::blocks::ConvAct;
use crate::depth_ops::{bilinear2x_align_corners, convtranspose2x};

/// Clamp bounds on the log-depth logit, copied from the reference rather than
/// chosen: `exp(-4) = 0.018 m` and `exp(5) = 148 m` are the ends of the range
/// the released weights were trained to cover.
const LOGIT_LO: f64 = -4.0;
const LOGIT_HI: f64 = 5.0;

/// The `head` tail: two convolutions around a 2x transposed convolution, then
/// a bare 1x1 to a single channel.
struct HeadTail {
    c0: ConvAct,
    up_w: Tensor,
    up_b: Option<Tensor>,
    c2: ConvAct,
    out_w: Tensor,
    out_b: Option<Tensor>,
}

pub struct DepthHead {
    proj: Vec<ConvAct>,
    refine: Vec<Vec<ConvAct>>,
    tail: HeadTail,
    /// Calibration, applied at eval only: `depth^cal_a * exp(cal_b)`.
    ///
    /// Scalars in the checkpoint, not hyperparameters — the nano weights carry
    /// `cal_a = 1.0`, `cal_b = -0.19384765625`. Loaded rather than hardcoded
    /// because they differ per tier and a wrong one scales every pixel.
    cal_a: f32,
    cal_b: f32,
}

impl DepthHead {
    /// `ch` are the P3/P4/P5 channel counts the neck emits; `c` is the head's
    /// working width (256 at every released tier).
    pub fn new(vb: VarBuilder, ch: [usize; 3], c: usize) -> Result<Self> {
        let mut proj = Vec::with_capacity(3);
        for (i, &c_in) in ch.iter().enumerate() {
            proj.push(ConvAct::new(vb.pp(format!("proj.{i}")), c_in, c, 1, 1, 1, true)?);
        }

        // THREE refine stages are stored and only TWO ever run.
        //
        // The reference loop is `for i in range(nl - 2, -1, -1)`, i.e. i = 1
        // then 0 — `refine[2]` is never reached. It is 1.2 M of the nano
        // model's 6.36 M parameters, dead in every released checkpoint.
        //
        // It is loaded anyway. The converter emits it, a strict load expects
        // it, and dropping it here would make the manifest stop round-tripping
        // for a saving of nothing at inference. Recorded so the next reader
        // does not "fix" the loop to use it.
        let mut refine = Vec::with_capacity(3);
        for i in 0..3 {
            let mut stage = Vec::with_capacity(2);
            for j in 0..2 {
                stage.push(ConvAct::new(vb.pp(format!("refine.{i}.{j}")), c, c, 3, 1, 1, true)?);
            }
            refine.push(stage);
        }

        let hb = vb.pp("head");
        let c0 = ConvAct::new(hb.pp("0"), c, c / 2, 3, 1, 1, true)?;
        // ConvTranspose2d weights are [in, out, kh, kw] — `in` FIRST,
        // transposed relative to Conv2d. Loading it as [out, in, ..] produces
        // a plausible depth map that is simply wrong.
        let up_w = hb.get((c / 2, c / 2, 2, 2), "1.weight")?;
        let up_b = hb.get(c / 2, "1.bias").ok();
        let c2 = ConvAct::new(hb.pp("2"), c / 2, c / 4, 3, 1, 1, true)?;
        let out_w = hb.get((1, c / 4, 1, 1), "3.weight")?;
        let out_b = hb.get(1, "3.bias").ok();

        let cal_a = vb.get(1, "cal_a").and_then(|t| t.flatten_all()?.to_vec1::<f32>()).map(|v| v[0]).unwrap_or(1.0);
        let cal_b = vb.get(1, "cal_b").and_then(|t| t.flatten_all()?.to_vec1::<f32>()).map(|v| v[0]).unwrap_or(0.0);

        Ok(Self { proj, refine, tail: HeadTail { c0, up_w, up_b, c2, out_w, out_b }, cal_a, cal_b })
    }

    /// `feats` are P3, P4, P5 in that order — the same three tensors the
    /// detect head consumes.
    ///
    /// Returns depth in **metres** at `(1, 1, H/4, W/4)`. Callers wanting the
    /// input resolution resize afterwards; the reference does the same and
    /// only its export path bakes the 4x in.
    pub fn forward(&self, feats: &[Tensor; 3]) -> Result<Tensor> {
        use candle_nn::Module;

        let projected: Vec<Tensor> =
            self.proj.iter().zip(feats).map(|(p, f)| p.forward(f)).collect::<Result<_>>()?;

        // Top-down: start at the coarsest level and walk back up, adding the
        // finer feature at each step. Two iterations, i = 1 then 0.
        let mut out = projected[2].clone();
        for i in (0..2).rev() {
            out = bilinear2x_align_corners(&out)?;
            out = (&out + &projected[i])?;
            for conv in &self.refine[i] {
                out = conv.forward(&out)?;
            }
        }

        let t = &self.tail;
        let mut y = t.c0.forward(&out)?;
        y = convtranspose2x(&y, &t.up_w, t.up_b.as_ref())?;
        y = t.c2.forward(&y)?;
        y = y.conv2d(&t.out_w, 0, 1, 1, 1)?;
        if let Some(b) = &t.out_b {
            y = y.broadcast_add(&b.reshape((1, 1, 1, 1))?)?;
        }

        // exp of a CLAMPED logit. The clamp is not defensive tidying — it is
        // what bounds an otherwise unbounded head, and the reference applies
        // it before the exponential, never after.
        let depth = y.clamp(LOGIT_LO, LOGIT_HI)?.exp()?;

        // Eval-time calibration. `powf` is skipped when cal_a is 1.0, which is
        // the released value at every tier probed so far — not to save the
        // multiply, but because `x^1.0` is not guaranteed bit-identical to `x`
        // and this output is oracle-gated.
        let depth = if (self.cal_a - 1.0).abs() > f32::EPSILON {
            depth.powf(self.cal_a as f64)?
        } else {
            depth
        };
        depth.affine(self.cal_b.exp() as f64, 0.0)
    }
}
