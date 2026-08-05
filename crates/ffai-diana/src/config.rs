//! The conversion manifest: what the weights on disk claim to be.
//!
//! `tools/diana_convert.py` writes this beside the safetensors. Loading is
//! **fail-closed** (mission plan §7 step 4): an unknown architecture, a
//! conversion-map version this build does not implement, or a head
//! configuration the port does not handle is an error, never a default.
//! Carmenta's CRAFT port paid for the opposite policy — a checkpoint whose
//! shapes disagreed with the code's own constructor calls, loaded
//! permissively, cost a session.

use serde::Deserialize;
use std::path::Path;

use ffai_core::error::{Error, Result};

/// Conversion-map versions this build understands.
///
/// A bump in `tools/diana_convert.py` means the key map or the fusion
/// policy changed; refusing an unknown version is what keeps a stale
/// safetensors from being silently misread.
pub const SUPPORTED_MAP_VERSIONS: &[u32] = &[1];

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub scale: String,
    pub task: String,
    // --- detect-only, defaulted so a depth manifest loads -------------
    //
    // These describe a box head and a depth manifest has none. Defaulting
    // rather than splitting the type keeps ONE manifest shape across tasks;
    // `validate()` enforces that a detect manifest actually carries them, so
    // the default cannot silently stand in for a missing field on the path
    // that needs it.
    /// Number of classes.
    #[serde(default)]
    pub nc: usize,
    /// 1 means DFL was removed and box channels are direct distances.
    #[serde(default)]
    pub reg_max: usize,
    #[serde(default)]
    pub end2end: bool,
    #[serde(default)]
    pub max_det: usize,
    #[serde(default)]
    pub strides: Vec<f32>,
    pub imgsz: usize,
    pub letterbox: String,
    pub inference_branch: String,
    pub fused: bool,
    pub conversion_map_version: u32,
    pub source_file: String,
    pub source_sha256: String,
    pub output_file: String,
    pub output_sha256: String,
    pub tensor_count: usize,
    pub param_count: usize,
    #[serde(default)]
    pub class_names: Vec<String>,

    // --- depth-only ---------------------------------------------------
    /// Working width of the depth head (256 at every released tier).
    #[serde(default)]
    pub head_channels: Option<usize>,
    /// P3/P4/P5 channel counts the neck hands the head. Read from the
    /// checkpoint at conversion rather than derived, so a scale rule that
    /// changes upstream cannot silently mis-shape the head.
    #[serde(default)]
    pub proj_in_channels: Option<Vec<usize>>,
    /// `depth^cal_a * exp(cal_b)` — learned, per-tier, scales every pixel.
    #[serde(default)]
    pub cal_a: Option<f32>,
    #[serde(default)]
    pub cal_b: Option<f32>,
    /// Map stride relative to the letterboxed input (4).
    #[serde(default)]
    pub output_stride: Option<usize>,
}

impl ModelConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: ModelConfig = serde_json::from_str(&text)
            .map_err(|e| Error::Other(format!("manifest {}: {e}", path.display())))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Every check that would otherwise become a wrong number later.
    pub fn validate(&self) -> Result<()> {
        let bail = |m: String| -> Result<()> { Err(Error::Other(m)) };
        if self.architecture != "yolo26" {
            return bail(format!(
                "architecture `{}` is not supported by this build (expected yolo26) — \
                 conversion maps are versioned per architecture on purpose",
                self.architecture
            ));
        }
        if !SUPPORTED_MAP_VERSIONS.contains(&self.conversion_map_version) {
            return bail(format!(
                "conversion map v{} was produced by a different tools/diana_convert.py \
                 than this build implements (supported: {:?}) — re-run the conversion",
                self.conversion_map_version, SUPPORTED_MAP_VERSIONS
            ));
        }
        match self.task.as_str() {
            "detect" => {
                // A detect manifest must actually carry the box-head fields.
                // They are `#[serde(default)]` so a DEPTH manifest can load,
                // and this is what stops that default standing in silently
                // for a field the detect path needs.
                if self.nc == 0 || self.strides.is_empty() || self.class_names.is_empty() {
                    return bail(
                        "detect manifest is missing nc / strides / class_names —                          re-run tools/diana_convert.py"
                            .into(),
                    );
                }
            }
            "depth" => {
                if self.proj_in_channels.as_ref().is_none_or(|v| v.len() != 3) {
                    return bail(
                        "depth manifest needs proj_in_channels for P3/P4/P5 —                          re-run tools/diana_convert.py"
                            .into(),
                    );
                }
            }
            other => return bail(format!("task `{other}` is not detect or depth")),
        }
        if !self.fused {
            return bail(
                "manifest says fused=false, but this port implements the FUSED graph \
                 (conv+bn folded at conversion). An unfused file would need BatchNorm at \
                 runtime, which is a different forward pass, not a slower one."
                    .into(),
            );
        }
        // The rest describes a BOX head. A depth manifest has none of it,
        // and asserting it there would reject a perfectly good file for
        // lacking fields its task never defines.
        if self.task == "detect" {
            if self.reg_max != 1 {
                return bail(format!(
                    "reg_max={} implies a DFL head; this port implements direct (l,t,r,b) \
                     regression, which is what reg_max=1 means",
                    self.reg_max
                ));
            }
            if !self.end2end || self.inference_branch != "one2one" {
                return bail(format!(
                    "expected an end2end one2one head (end2end={}, branch={}) — the one2many \
                     branch is training-only and produces plausible but wrong detections",
                    self.end2end, self.inference_branch
                ));
            }
            if self.strides.len() != 3 {
                return bail(format!("expected 3 stride levels, found {}", self.strides.len()));
            }
            if self.class_names.len() != self.nc {
                return bail(format!(
                    "manifest lists {} class names for nc={}",
                    self.class_names.len(),
                    self.nc
                ));
            }
        }
        Ok(())
    }

    /// The width/depth rule for this checkpoint's scale.
    pub fn dims(&self) -> Result<Dims> {
        Dims::for_scale(&self.scale)
    }
}

/// How a YOLO scale turns the shared YAML graph into concrete channel counts
/// and repeat counts.
///
/// The same layer table describes every tier; `n`/`s`/`m`/`l`/`x` differ only
/// by three numbers. **Verified against the real checkpoints, not derived and
/// hoped** (`tools/diana_probe_arch.py` + the scale check): every layer width
/// and every repeat count this produces matches what `yolo26n.pt` and
/// `yolo26s.pt` actually contain.
///
/// The widths were previously a hand-tabulated n-tier list, which was correct
/// and unextendable. A table is safe when there is one tier and a trap when
/// there are five.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dims {
    pub depth: f64,
    pub width: f64,
    pub max_channels: usize,
    /// m/l/x promote every `C3k2` to C3k inner blocks regardless of the
    /// YAML flag — see [`Dims::c3k`].
    pub c3k_all: bool,
}

impl Dims {
    pub fn for_scale(scale: &str) -> Result<Self> {
        // From the model YAML's own `scales` row.
        let (depth, width, max_channels) = match scale {
            "n" => (0.50, 0.25, 1024),
            "s" => (0.50, 0.50, 1024),
            "m" => (0.50, 1.00, 512),
            "l" => (1.00, 1.00, 512),
            "x" => (1.00, 1.50, 512),
            other => {
                return Err(Error::Other(format!(
                    "unknown YOLO scale `{other}` (expected n, s, m, l or x)"
                )))
            }
        };
        Ok(Dims {
            depth,
            width,
            max_channels,
            c3k_all: matches!(scale, "m" | "l" | "x"),
        })
    }

    /// A YAML channel count → this tier's actual channel count.
    ///
    /// `make_divisible(min(c, max_channels) * width, 8)` — Ultralytics'
    /// `parse_model` rule, including the cap applied BEFORE the scaling
    /// (which is what makes m/l/x differ from a pure width multiple).
    pub fn ch(&self, yaml_c: usize) -> usize {
        let scaled = (yaml_c.min(self.max_channels) as f64) * self.width;
        // make_divisible(x, 8) = ceil(x / 8) * 8
        ((scaled / 8.0).ceil() as usize) * 8
    }

    /// A YAML repeat count → this tier's actual repeat count.
    ///
    /// Only counts above 1 scale; a single block stays single. This is why
    /// `l` and `x` carry TWO inner blocks per C3k2 where n/s/m carry one.
    pub fn rep(&self, yaml_n: usize) -> usize {
        if yaml_n > 1 {
            ((yaml_n as f64 * self.depth).round() as usize).max(1)
        } else {
            yaml_n
        }
    }

    /// A C3k2's hidden width: `int(c_out * e)` for the layer's expansion.
    pub fn hidden(&self, c_out: usize, e: f64) -> usize {
        (c_out as f64 * e) as usize
    }

    /// Whether a `C3k2` layer's inner blocks are `C3k` rather than plain
    /// `Bottleneck`, given the flag its YAML line carries.
    ///
    /// **The YAML flag is not the answer.** `parse_model` overrides it:
    ///
    /// ```python
    /// if m is C3k2:
    ///     if scale in "mlx":
    ///         args[3] = True
    /// ```
    ///
    /// so on m/l/x every C3k2 is a C3k regardless of what the file says.
    /// Only layers 2 and 4 carry `False`, so this is invisible on n and s
    /// and changes two layers on m/l/x — the third scale-dependent branch
    /// after the 512-channel cap and the depth-scaled repeat count, and the
    /// one no amount of testing n and s harder could have surfaced. The m
    /// checkpoint reports 224 tensors against n's 204; that difference is
    /// entirely these two layers.
    pub fn c3k(&self, yaml_c3k: bool) -> bool {
        yaml_c3k || self.c3k_all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> ModelConfig {
        ModelConfig {
            architecture: "yolo26".into(),
            scale: "n".into(),
            task: "detect".into(),
            nc: 2,
            reg_max: 1,
            end2end: true,
            max_det: 300,
            strides: vec![8.0, 16.0, 32.0],
            imgsz: 640,
            letterbox: "square-center-pad-114".into(),
            inference_branch: "one2one".into(),
            fused: true,
            conversion_map_version: 1,
            source_file: "yolo26n.pt".into(),
            source_sha256: "x".into(),
            output_file: "yolo26n-diana.safetensors".into(),
            output_sha256: "y".into(),
            tensor_count: 204,
            param_count: 2_408_932,
            class_names: vec!["a".into(), "b".into()],
            head_channels: None,
            proj_in_channels: None,
            cal_a: None,
            cal_b: None,
            output_stride: None,
        }
    }

    #[test]
    fn accepts_a_well_formed_manifest() {
        assert!(base().validate().is_ok());
    }

    #[test]
    fn rejects_an_unknown_conversion_map_version() {
        let mut c = base();
        c.conversion_map_version = 99;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_a_dfl_head() {
        let mut c = base();
        c.reg_max = 16;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_the_training_only_branch() {
        let mut c = base();
        c.inference_branch = "one2many".into();
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_unfused_weights() {
        let mut c = base();
        c.fused = false;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_a_class_name_count_mismatch() {
        let mut c = base();
        c.nc = 80;
        assert!(c.validate().is_err());
    }

    /// The widths and repeats this produces are checked against what the real
    /// checkpoints contain — these are the values probed out of `yolo26n.pt`
    /// and `yolo26s.pt`, not values re-derived from the same formula.
    #[test]
    fn scale_rule_reproduces_the_probed_checkpoints() {
        let n = Dims::for_scale("n").unwrap();
        // yolo26n: layers 0, 1, 2, 7 and the SPPF cap.
        assert_eq!((n.ch(64), n.ch(128), n.ch(256), n.ch(1024)), (16, 32, 64, 256));
        // yolo26s doubles every width at the same depth.
        let s = Dims::for_scale("s").unwrap();
        assert_eq!((s.ch(64), s.ch(128), s.ch(256), s.ch(1024)), (32, 64, 128, 512));
        // m/l/x cap at 512 BEFORE scaling, which is why they are not simply
        // wider than s at the deep layers.
        let m = Dims::for_scale("m").unwrap();
        assert_eq!((m.ch(64), m.ch(1024)), (64, 512));
        let x = Dims::for_scale("x").unwrap();
        assert_eq!((x.ch(64), x.ch(1024)), (96, 768));
    }

    #[test]
    fn only_l_and_x_deepen_the_blocks() {
        for (scale, want) in [("n", 1), ("s", 1), ("m", 1), ("l", 2), ("x", 2)] {
            let d = Dims::for_scale(scale).unwrap();
            assert_eq!(d.rep(2), want, "scale {scale} repeat of a yaml_n=2 block");
            assert_eq!(d.rep(1), 1, "scale {scale}: a single block stays single");
        }
    }

    #[test]
    fn hidden_width_follows_the_layers_own_expansion() {
        let n = Dims::for_scale("n").unwrap();
        // Layer 2 is `C3k2 [256, False, 0.25]`: out 64, hidden 16 (probed).
        assert_eq!(n.hidden(n.ch(256), 0.25), 16);
        // Layer 6 is `C3k2 [512, True]`, e defaults to 0.5: out 128, hidden 64.
        assert_eq!(n.hidden(n.ch(512), 0.5), 64);
    }

    /// The C3k promotion, against what the checkpoints actually contain.
    ///
    /// A module-tree diff of `yolo26n.pt` against `yolo26m.pt` shows exactly
    /// two differences — `model.2.m.0` and `model.4.m.0` go from
    /// `Bottleneck` to `C3k` — and those are the only two `C3k2` layers
    /// whose YAML `c3k` argument is `False`. Everything else in the graph
    /// already carries `True`, which is why this branch is invisible until a
    /// scale in "mlx" exists.
    #[test]
    fn mlx_promote_every_c3k2_regardless_of_the_yaml_flag() {
        for scale in ["n", "s"] {
            let d = Dims::for_scale(scale).unwrap();
            assert!(!d.c3k(false), "{scale}: layers 2 and 4 stay Bottleneck");
            assert!(d.c3k(true), "{scale}: every other C3k2 is already C3k");
        }
        for scale in ["m", "l", "x"] {
            let d = Dims::for_scale(scale).unwrap();
            assert!(
                d.c3k(false),
                "{scale}: parse_model sets args[3] = True for scales in \"mlx\", so the \
                 YAML's False does not survive"
            );
            assert!(d.c3k(true));
        }
    }

    #[test]
    fn rejects_an_unknown_scale() {
        assert!(Dims::for_scale("xl").is_err());
    }
}
