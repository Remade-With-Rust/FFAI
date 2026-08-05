//! `Yolo26Depth` — the [`DepthEngine`] composing preprocess → backbone → neck
//! → depth head.
//!
//! Deliberately a sibling of [`crate::engine::Yolo26`] rather than a variant
//! of it. The two share a backbone and neck at the WEIGHT level — the YAML is
//! byte-identical — but not at the checkpoint level: `yolo26n.pt` and
//! `yolo26n-depth.pt` are different files with differently-trained weights.
//! One engine holding both would have to load two models and choose, which is
//! the wrong shape for a caller that wants one of them.

use std::path::PathBuf;
use std::sync::OnceLock;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use ffai_core::engine::{DepthEngine, DepthOptions, DepthOutput, EngineInfo, EngineStatus, Task};
use ffai_core::error::{Error, Result};
use ffai_core::types::ImageBuffer;

use crate::backbone::Backbone;
use crate::config::ModelConfig;
use crate::depth_head::DepthHead;
use crate::image::{letterbox_with, Geometry};
use crate::neck::Neck;

struct Model {
    backbone: Backbone,
    neck: Neck,
    head: DepthHead,
    cfg: ModelConfig,
    device: Device,
}

pub struct Yolo26Depth {
    manifest_dir: PathBuf,
    geometry: Geometry,
    tier: String,
    model: OnceLock<std::result::Result<Model, String>>,
}

impl Yolo26Depth {
    pub fn build(tier: &str, geometry: Geometry, manifest_dir: impl Into<PathBuf>) -> Self {
        Yolo26Depth {
            manifest_dir: manifest_dir.into(),
            geometry,
            tier: tier.to_string(),
            model: OnceLock::new(),
        }
    }

    pub fn tier(&self) -> &str {
        &self.tier
    }

    fn model(&self) -> Result<&Model> {
        self.model
            .get_or_init(|| {
                load(&self.manifest_dir, &self.tier).map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| Error::Model(e.clone()))
    }
}

fn candle_err(e: candle_core::Error) -> Error {
    Error::Model(e.to_string())
}

fn load(manifest_dir: &std::path::Path, tier: &str) -> Result<Model> {
    let device = Device::Cpu;
    let model_name = format!("yolo26{tier}-depth-diana");
    let (cfg, weights) = crate::engine::resolve_model(manifest_dir, &model_name, tier)?;
    if cfg.task != "depth" {
        return Err(Error::Model(format!(
            "manifest `{model_name}` declares task `{}` — this engine needs depth weights              (tools/diana_convert.py --model yolo26{tier}-depth)",
            cfg.task
        )));
    }
    // SAFETY: mmap of a file resolved above; candle's loader requires it, and
    // the detect engine loads the same way for the same reason.
    #[allow(unsafe_code)]
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)
            .map_err(candle_err)?
    };
    let dims = cfg.dims()?;
    let backbone = Backbone::new(vb.clone(), dims).map_err(candle_err)?;
    let neck = Neck::new(vb.clone(), dims).map_err(candle_err)?;
    // The head sits at model.23, the slot Detect occupies in the sibling
    // graph — the two checkpoints agree on every index before it.
    // Channel counts come from the MANIFEST, read off the checkpoint at
    // conversion, not derived from the scale rule here. Validated non-empty
    // by `ModelConfig::validate`, so the index is safe.
    let ch = cfg.proj_in_channels.as_ref().expect("validated by ModelConfig");
    let head = DepthHead::new(
        vb.pp("model.23"),
        [ch[0], ch[1], ch[2]],
        cfg.head_channels.unwrap_or(256),
    )
    .map_err(candle_err)?;
    Ok(Model { backbone, neck, head, cfg, device })
}

impl DepthEngine for Yolo26Depth {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: format!(
                "yolo26{}-depth{}",
                self.tier,
                match self.geometry {
                    Geometry::Rect => "",
                    Geometry::Square => "-square",
                }
            ),
            task: Task::Depth,
            status: EngineStatus::Experimental,
            description: "YOLO26 monocular depth on candle — dense metric depth, \
                          unbounded log head, from official Ultralytics weights"
                .into(),
        }
    }

    fn depth(&self, image: &ImageBuffer, opts: &DepthOptions) -> Result<DepthOutput> {
        let m = self.model()?;
        crate::profile::timed(|p| &p.total, || {
            let (input, lb) = crate::profile::timed(|p| &p.pre, || {
                letterbox_with(image, m.cfg.imgsz, self.geometry, &m.device).map_err(candle_err)
            })?;

            let b = crate::profile::timed(|p| &p.backbone, || {
                m.backbone.forward(&input).map_err(candle_err)
            })?;
            let n = crate::profile::timed(|p| &p.neck, || m.neck.forward(&b).map_err(candle_err))?;
            let d = crate::profile::timed(|p| &p.head, || {
                m.head.forward(&[n.p3.clone(), n.p4.clone(), n.p5.clone()]).map_err(candle_err)
            })?;

            let (_, _, h, w) = d.dims4().map_err(candle_err)?;
            let depth = d.flatten_all().map_err(candle_err)?.to_vec1::<f32>().map_err(candle_err)?;
            let out = DepthOutput { depth, width: w, height: h, letterbox: Some(lb) };

            if opts.full_resolution {
                to_source_resolution(&out, image)
            } else {
                Ok(out)
            }
        })
    }
}

/// Resize the stride-4 map back to the source image and undo the letterbox.
///
/// Nearest-neighbour on purpose. The map is already a smooth field and the
/// caller asked for source resolution, not for interpolation they did not
/// choose — anyone wanting bilinear can take the raw map and do it. Depth is
/// also not a quantity to blur across a discontinuity: averaging across an
/// object boundary invents a surface at a distance nothing occupies.
fn to_source_resolution(out: &DepthOutput, image: &ImageBuffer) -> Result<DepthOutput> {
    let (sw, sh) = (image.width as usize, image.height as usize);
    let lb = out.letterbox.as_ref().ok_or_else(|| {
        Error::Model("depth map has no letterbox; cannot map to source".into())
    })?;
    // The head emits at stride 4 of the letterboxed canvas, so one map pixel
    // spans exactly 4 canvas pixels — no canvas dimensions needed.
    let stride = 4.0f32;

    let mut v = vec![f32::NAN; sw * sh];
    for y in 0..sh {
        // source -> canvas -> map
        let cy = y as f32 * lb.scale + lb.pad_y;
        let my = (cy / stride).floor() as isize;
        if my < 0 || my as usize >= out.height {
            continue;
        }
        for x in 0..sw {
            let cx = x as f32 * lb.scale + lb.pad_x;
            let mx = (cx / stride).floor() as isize;
            if mx < 0 || mx as usize >= out.width {
                continue;
            }
            v[y * sw + x] = out.depth[my as usize * out.width + mx as usize];
        }
    }
    Ok(DepthOutput { depth: v, width: sw, height: sh, letterbox: out.letterbox.clone() })
}
