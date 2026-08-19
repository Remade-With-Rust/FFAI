//! The YOLO26 neck: layers 11–22, FPN top-down then PAN bottom-up.
//!
//! Nothing new is invented here — it is the same [`crate::blocks::C3k2`] the
//! backbone uses, plus nearest-neighbour upsampling, channel concatenation
//! and two stride-2 convolutions. The only piece with its own character is
//! layer 22, which takes the 4-argument `C3k2 [1024, True, 0.5, True]` form
//! whose final `True` swaps the inner block for a Bottleneck followed by a
//! `PSABlock` ([`crate::blocks::InnerKind::Attn`]).
//!
//! ```text
//! 11 Upsample  P5      x2 nearest              256 -> 256  20 -> 40
//! 12 Concat    [11, backbone P4]               256+128 = 384
//! 13 C3k2      384 -> 128  c=64   C3k
//! 14 Upsample  13      x2 nearest              128 -> 128  40 -> 80
//! 15 Concat    [14, backbone P3]               128+128 = 256
//! 16 C3k2      256 ->  64  c=32   C3k          -> head level 0 (stride 8)
//! 17 Conv       64 ->  64  k3 s2
//! 18 Concat    [17, 13]                        64+128 = 192
//! 19 C3k2      192 -> 128  c=64   C3k          -> head level 1 (stride 16)
//! 20 Conv      128 -> 128  k3 s2
//! 21 Concat    [20, backbone P5]               128+256 = 384
//! 22 C3k2      384 -> 256  c=128  Attn         -> head level 2 (stride 32)
//! ```
//!
//! The skip sources are worth stating because they are the easiest thing to
//! wire wrongly and the hardest to notice: layer 18 concatenates with **13**
//! (the FPN's own intermediate), not with the backbone, and layer 21
//! concatenates with **10** (the backbone's P5), not with layer 11's
//! upsample. Both are checked by the per-layer oracle.

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::backbone::{inner, BackboneOutput};
use crate::blocks::{C3k2, ConvAct, InnerKind};
use crate::config::Dims;

/// The three feature maps the detection head consumes, finest first.
pub struct NeckOutput {
    /// Stride 8, 64 channels.
    pub p3: Tensor,
    /// Stride 16, 128 channels.
    pub p4: Tensor,
    /// Stride 32, 256 channels.
    pub p5: Tensor,
}

pub struct Neck {
    l13: C3k2,
    l16: C3k2,
    l17: ConvAct,
    l19: C3k2,
    l20: ConvAct,
    l22: C3k2,
}

impl Neck {
    /// Build for a given tier. The concatenation widths are SUMS of upstream
    /// outputs, so they are computed rather than tabulated — writing them as
    /// literals is what made this n-only.
    pub fn new(vb: VarBuilder, d: Dims) -> Result<Self> {
        let m = vb.pp("model");
        let (c256, c512, c1024) = (d.ch(256), d.ch(512), d.ch(1024));
        Ok(Self {
            // 12: cat(upsample(P5 = c1024), backbone P4 = c512)
            l13: C3k2::new(m.pp(13), c1024 + c512, c512, d.hidden(c512, 0.5), d.rep(2), inner(d, true))?,
            // 15: cat(upsample(l13 = c512), backbone P3 = c512)
            l16: C3k2::new(m.pp(16), c512 + c512, c256, d.hidden(c256, 0.5), d.rep(2), inner(d, true))?,
            l17: ConvAct::new(m.pp(17), c256, c256, 3, 2, 1, true)?,
            // 18: cat(l17 = c256, l13 = c512)
            l19: C3k2::new(m.pp(19), c256 + c512, c512, d.hidden(c512, 0.5), d.rep(2), inner(d, true))?,
            l20: ConvAct::new(m.pp(20), c512, c512, 3, 2, 1, true)?,
            // 21: cat(l20 = c512, backbone P5 = c1024). `C3k2 [1024, True,
            // 0.5, True]` — the 4-arg attention form, and its repeat count is
            // a literal 1 in the YAML, so it does NOT deepen on l/x.
            l22: C3k2::new(m.pp(22), c512 + c1024, c1024, d.hidden(c1024, 0.5), 1, InnerKind::Attn)?,
        })
    }

    pub fn forward(&self, b: &BackboneOutput) -> Result<NeckOutput> {
        let t = self.forward_traced(b)?;
        Ok(NeckOutput { p3: t.p3, p4: t.p4, p5: t.p5 })
    }

    /// Forward keeping every layer output, for the oracle. Layer indices
    /// are the graph's own, so a divergence names the layer directly.
    pub fn forward_traced(&self, b: &BackboneOutput) -> Result<NeckTrace> {
        let (_, _, h, w) = b.p5.dims4()?;
        let l11 = b.p5.upsample_nearest2d(h * 2, w * 2)?;
        let l12 = Tensor::cat(&[&l11, &b.p4], 1)?;
        let l13 = self.l13.forward(&l12)?;

        let (_, _, h, w) = l13.dims4()?;
        let l14 = l13.upsample_nearest2d(h * 2, w * 2)?;
        let l15 = Tensor::cat(&[&l14, &b.p3], 1)?;
        let l16 = self.l16.forward(&l15)?;

        let l17 = self.l17.forward(&l16)?;
        // 18 concatenates with 13 — the FPN intermediate, not the backbone.
        let l18 = Tensor::cat(&[&l17, &l13], 1)?;
        let l19 = self.l19.forward(&l18)?;

        let l20 = self.l20.forward(&l19)?;
        // 21 concatenates with the backbone's P5 (layer 10), not with 11.
        let l21 = Tensor::cat(&[&l20, &b.p5], 1)?;
        let l22 = self.l22.forward(&l21)?;

        Ok(NeckTrace {
            l11,
            l12,
            l13: l13.clone(),
            l14,
            l15,
            l16: l16.clone(),
            l17,
            l18,
            l19: l19.clone(),
            l20,
            l21,
            l22: l22.clone(),
            p3: l16,
            p4: l19,
            p5: l22,
        })
    }
}

/// Every neck layer output, keyed by the graph index that produced it.
pub struct NeckTrace {
    pub l11: Tensor,
    pub l12: Tensor,
    pub l13: Tensor,
    pub l14: Tensor,
    pub l15: Tensor,
    pub l16: Tensor,
    pub l17: Tensor,
    pub l18: Tensor,
    pub l19: Tensor,
    pub l20: Tensor,
    pub l21: Tensor,
    pub l22: Tensor,
    pub p3: Tensor,
    pub p4: Tensor,
    pub p5: Tensor,
}

impl NeckTrace {
    /// Layers 11..=22 in graph order, for the oracle's per-layer walk.
    #[must_use] 
    pub fn layers(&self) -> [(&str, &Tensor); 12] {
        [
            ("layer_11", &self.l11),
            ("layer_12", &self.l12),
            ("layer_13", &self.l13),
            ("layer_14", &self.l14),
            ("layer_15", &self.l15),
            ("layer_16", &self.l16),
            ("layer_17", &self.l17),
            ("layer_18", &self.l18),
            ("layer_19", &self.l19),
            ("layer_20", &self.l20),
            ("layer_21", &self.l21),
            ("layer_22", &self.l22),
        ]
    }
}
