//! The YOLO26 backbone: layers 0–10, stem through C2PSA.
//!
//! The layer GRAPH here is fixed; every number in it comes from [`Dims`],
//! so this one code path builds n/s/m/l/x. Widths were once a hand-tabulated
//! n-tier list — correct, and unextendable the moment a second tier existed.
//!
//! A wrong width is not reliably a shape error, which is why the derivation
//! is validated against real checkpoints by `tools/diana_probe_arch.py`
//! rather than trusted: a `C3k2` hidden `e` of 0.25 versus 0.5 changes
//! results without changing any output shape.
//!
//! Layer graph (from the model's own YAML, backbone half; the channel
//! counts shown are the n tier's, as `Dims` resolves them):
//!
//! ```text
//!  0 Conv    3 ->  16  k3 s2
//!  1 Conv   16 ->  32  k3 s2
//!  2 C3k2   32 ->  64  c=16  Bottleneck*  (e=0.25)
//!  3 Conv   64 ->  64  k3 s2
//!  4 C3k2   64 -> 128  c=32  Bottleneck*  (e=0.25)   -> P3 skip
//!  5 Conv  128 -> 128  k3 s2
//!  6 C3k2  128 -> 128  c=64  C3k                     -> P4 skip
//!  7 Conv  128 -> 256  k3 s2
//!  8 C3k2  256 -> 256  c=128 C3k
//!  9 SPPF  256 -> 256  k5 n3 + residual
//! 10 C2PSA 256 -> 256  c=128 1 PSABlock              -> P5
//! ```
//!
//! `*` = the two layers whose inner-block kind is tier-dependent: C3k on
//! m/l/x, Bottleneck on n/s. See [`Dims::c3k`].

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::blocks::{C2psa, C3k2, ConvAct, InnerKind, Sppf};
use crate::config::Dims;

/// A `C3k2` layer's inner-block kind, from its YAML flag and the tier.
///
/// Shared with the neck so both halves of the graph resolve the flag the
/// same way; see [`Dims::c3k`] for why the YAML flag alone is wrong.
pub(crate) fn inner(d: Dims, yaml_c3k: bool) -> InnerKind {
    if d.c3k(yaml_c3k) {
        InnerKind::C3k
    } else {
        InnerKind::Bottleneck
    }
}

/// Outputs the neck consumes, by the layer index that produced them.
pub struct BackboneOutput {
    /// Layer 4 — stride 8 features (the neck's P3 skip).
    pub p3: Tensor,
    /// Layer 6 — stride 16 features (the neck's P4 skip).
    pub p4: Tensor,
    /// Layer 10 — stride 32 features, after SPPF and C2PSA.
    pub p5: Tensor,
}

pub struct Backbone {
    l0: ConvAct,
    l1: ConvAct,
    l2: C3k2,
    l3: ConvAct,
    l4: C3k2,
    l5: ConvAct,
    l6: C3k2,
    l7: ConvAct,
    l8: C3k2,
    l9: Sppf,
    l10: C2psa,
}

impl Backbone {
    /// Build from a `VarBuilder` rooted at the safetensors' top level; the
    /// layer prefixes are the checkpoint's own `model.N` names, kept
    /// verbatim so a tensor-name mismatch is a load error rather than a
    /// silent zero.
    /// Build for a given tier. Every channel count and repeat count is
    /// derived from [`Dims`], so the same code builds n/s/m/l/x — the layer
    /// GRAPH is shared across tiers, only these numbers change.
    pub fn new(vb: VarBuilder, d: Dims) -> Result<Self> {
        let m = vb.pp("model");
        // The YAML's channel arguments, scaled. Named so each line reads as
        // the layer table it mirrors.
        let (c64, c128, c256, c512, c1024) =
            (d.ch(64), d.ch(128), d.ch(256), d.ch(512), d.ch(1024));
        Ok(Self {
            l0: ConvAct::new(m.pp(0), 3, c64, 3, 2, 1, true)?,
            l1: ConvAct::new(m.pp(1), c64, c128, 3, 2, 1, true)?,
            // `C3k2 [256, False, 0.25]` — e = 0.25 on the two shallow
            // stages, and the ONLY two layers whose YAML `c3k` is False.
            // `d.c3k` is what turns that False into True on m/l/x.
            l2: C3k2::new(m.pp(2), c128, c256, d.hidden(c256, 0.25), d.rep(2), inner(d, false))?,
            l3: ConvAct::new(m.pp(3), c256, c256, 3, 2, 1, true)?,
            l4: C3k2::new(m.pp(4), c256, c512, d.hidden(c512, 0.25), d.rep(2), inner(d, false))?,
            l5: ConvAct::new(m.pp(5), c512, c512, 3, 2, 1, true)?,
            // `C3k2 [512, True]` — e defaults to 0.5 from here down.
            l6: C3k2::new(m.pp(6), c512, c512, d.hidden(c512, 0.5), d.rep(2), InnerKind::C3k)?,
            l7: ConvAct::new(m.pp(7), c512, c1024, 3, 2, 1, true)?,
            l8: C3k2::new(m.pp(8), c1024, c1024, d.hidden(c1024, 0.5), d.rep(2), InnerKind::C3k)?,
            l9: Sppf::new(m.pp(9), c1024, c1024, 5, 3)?,
            l10: C2psa::new(m.pp(10), c1024, c1024, d.rep(2))?,
        })
    }

    /// Forward, keeping the two skip taps the neck needs.
    pub fn forward(&self, x: &Tensor) -> Result<BackboneOutput> {
        let x = self.l0.forward(x)?;
        let x = self.l1.forward(&x)?;
        let x = self.l2.forward(&x)?;
        let x = self.l3.forward(&x)?;
        let p3 = self.l4.forward(&x)?;
        let x = self.l5.forward(&p3)?;
        let p4 = self.l6.forward(&x)?;
        let x = self.l7.forward(&p4)?;
        let x = self.l8.forward(&x)?;
        let x = self.l9.forward(&x)?;
        let p5 = self.l10.forward(&x)?;
        Ok(BackboneOutput { p3, p4, p5 })
    }

    /// Every layer's output, indexed by graph position — what the per-layer
    /// oracle compares against the reference dump. Kept separate from
    /// [`Self::forward`] so the shipping path allocates nothing extra.
    pub fn forward_traced(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let l0 = self.l0.forward(x)?;
        let l1 = self.l1.forward(&l0)?;
        let l2 = self.l2.forward(&l1)?;
        let l3 = self.l3.forward(&l2)?;
        let l4 = self.l4.forward(&l3)?;
        let l5 = self.l5.forward(&l4)?;
        let l6 = self.l6.forward(&l5)?;
        let l7 = self.l7.forward(&l6)?;
        let l8 = self.l8.forward(&l7)?;
        let l9 = self.l9.forward(&l8)?;
        let l10 = self.l10.forward(&l9)?;
        Ok(vec![l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])
    }
}
