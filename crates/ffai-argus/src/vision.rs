//! `SmolVLM`'s vision tower and connector, on candle.
//!
//! Step 3 of `docs/plans/argus-launch-plan.md`, and the plan is blunt about
//! why it comes before anything that emits text:
//!
//! > "A mismatched tower cannot be debugged later through generated text."
//!
//! A tower that is subtly wrong still produces fluent, plausible captions. §7
//! of that plan measured the same class of failure from a prompt difference
//! alone — **43 of 50 answers changed on identical weights, with nothing
//! raising an error**. Numerical error hides behind plausibility. So this is
//! gated as TENSORS against the reference runtime's own output
//! (`corpora/refs/dump_smolvlm_vision.py`), stage by stage, before a single
//! token is generated.
//!
//! # What is ours and what is candle's
//!
//! Almost none of this is ours, which is the point of the Gate 1.2 pick:
//!
//! | piece | where it comes from |
//! |---|---|
//! | `SigLIP` encoder (patch embed, 12 blocks, post-LN) | **candle** `models::siglip::VisionModel` |
//! | pixel-shuffle + projection connector | here — a reshape and one matmul |
//! | weight loading | `candle_nn::VarBuilder` over the checkpoint's safetensors |
//!
//! `SmolVLM`'s vision weights are named exactly as candle's `SigLIP` expects
//! (`embeddings.patch_embedding`, `encoder.layers.N.{layer_norm1, layer_norm2,
//! mlp.fc1, mlp.fc2, self_attn.{q,k,v,out}_proj}`, `post_layernorm`), so the
//! encoder needs no adapter at all — only the right prefix.
//!
//! The checkpoint carries **no `head.*` tensors**, so the attention-pooling
//! head is off: `VisionModel::new(cfg, /* use_head */ false, vb)`. With the
//! head disabled candle applies `post_layernorm` and returns it, which is
//! exactly what the reference calls `last_hidden_state` — confirmed by the
//! oracle, where `post_layernorm` and `vision_out` share a checksum.

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip;

/// The connector's tensor name in the checkpoint. One tensor, no bias.
const PROJ_WEIGHT: &str = "modality_projection.proj.weight";

/// `SmolVLM`'s vision half: `SigLIP` tower + pixel-shuffle connector.
/// Which `SigLIP` implementation runs.
///
/// `crate::siglip` is the default: same graph, same weights, but the
/// memory-bound half of each layer rewritten to use every core (candle's CPU
/// backend parallelises `conv2d` and nothing else). candle's stays reachable
/// via `FFAI_ARGUS_CANDLE_TOWER=1` — it is what `tests/siglip_parity.rs`
/// compares against, and it is the fallback if ours is ever suspected.
enum Tower {
    Ours(Box<crate::siglip::VisionTower>),
    Candle(Box<siglip::VisionModel>),
}

pub struct SmolVlmVision {
    tower: Tower,
    /// `(hidden * scale^2) -> text_hidden`, e.g. `12288 -> 576`.
    proj: Tensor,
    scale_factor: usize,
}

impl SmolVlmVision {
    /// Build from a checkpoint's `VarBuilder`, rooted at the top level.
    ///
    /// `vision_prefix` is where the tower lives (`model.vision_model` in
    /// `SmolVLM`'s safetensors) and `connector_prefix` where the projection does
    /// (`model.connector`). They are parameters rather than constants because
    /// checkpoints move things: `Idefics3` and `SmolVLM2` differ here, and a
    /// hard-coded prefix fails as "tensor not found" rather than as "this is a
    /// different layout".
    pub fn new(
        cfg: &siglip::VisionConfig,
        scale_factor: usize,
        vision_prefix: &str,
        connector_prefix: &str,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        // use_head = false: the checkpoint has no `head.*` tensors, and with
        // the head off candle returns the post-layernorm hidden state, which
        // is what the reference calls `last_hidden_state`.
        // Ours by default; candle's behind an env switch so the two can be
        // A/B'd in ONE binary. A before/after across two builds compares two
        // binaries on a box whose state moved in between — and it also makes
        // "is the slow one still installed?" a question nobody can answer.
        let tower = if std::env::var_os("FFAI_ARGUS_CANDLE_TOWER").is_some() {
            Tower::Candle(Box::new(siglip::VisionModel::new(
                cfg,
                false,
                vb.pp(vision_prefix),
            )?))
        } else {
            Tower::Ours(Box::new(crate::siglip::VisionTower::new(
                cfg,
                vb.pp(vision_prefix),
            )?))
        };
        // `get_unchecked` rather than `get(shape, ..)`: the projection's
        // out_features is the TEXT hidden size, which is not in the vision
        // config. Reading it from the tensor beats threading a second config
        // through — and the in_features IS checked below, which is the half
        // that catches a wrong scale_factor.
        let proj = vb.pp(connector_prefix).get_unchecked(PROJ_WEIGHT)?;
        let expect_in = cfg.hidden_size * scale_factor * scale_factor;
        let (_out, got_in) = proj.dims2()?;
        if got_in != expect_in {
            candle_core::bail!(
                "connector expects in_features {expect_in} \
                 (hidden {} x scale_factor {scale_factor}^2) but the checkpoint's \
                 {PROJ_WEIGHT} has {got_in} — the config and the weights disagree",
                cfg.hidden_size
            );
        }
        Ok(Self {
            tower,
            proj,
            scale_factor,
        })
    }

    /// The tower alone: `(b, 3, H, W)` -> `(b, patches, hidden)`.
    ///
    /// Gated against the oracle's `vision_out` stage.
    pub fn tower(&self, pixel_values: &Tensor) -> CandleResult<Tensor> {
        match &self.tower {
            Tower::Ours(t) => t.forward(pixel_values),
            Tower::Candle(t) => t.forward(pixel_values),
        }
    }

    /// Tower + connector: `(b, 3, H, W)` -> `(b, patches / scale^2, text_hidden)`.
    ///
    /// Gated against the oracle's `connector` stage.
    pub fn forward(&self, pixel_values: &Tensor) -> CandleResult<Tensor> {
        let hidden = self.tower(pixel_values)?;
        self.connect(&hidden)
    }

    /// Pixel-shuffle then project.
    ///
    /// **This is the whole connector**, and its smallness is why `SmolVLM` won
    /// Gate 1.2 column (b): the alternative candidates use a Q-Former or a
    /// perceiver resampler, which are models in their own right. Here 1024
    /// patches of 768 become 64 tokens of 576 — a 16x token reduction, which
    /// is what makes 17 tiles affordable at all.
    pub fn connect(&self, hidden: &Tensor) -> CandleResult<Tensor> {
        let (b, seq, dim) = hidden.dims3()?;
        let s = self.scale_factor;
        // Patches are a square grid; the shuffle is 2-D, so a non-square count
        // means the caller handed us something other than a tower output.
        let side = (seq as f64).sqrt() as usize;
        if side * side != seq {
            candle_core::bail!("expected a square patch grid, got {seq} patches");
        }
        if !side.is_multiple_of(s) {
            candle_core::bail!("patch grid {side} is not divisible by scale_factor {s}");
        }
        // (b, side, side, dim) -> group each s x s block of patches into one
        // token whose feature vector is their features concatenated.
        let x = hidden.reshape((b, side, side, dim))?;
        let x = x.reshape((b, side, side / s, dim * s))?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = x.reshape((b, side / s, side / s, dim * s * s))?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = x.reshape((b, (side / s) * (side / s), dim * s * s))?;
        // One matmul, no bias — through a FLAT 2D product, not `broadcast_matmul`.
        //
        // `broadcast_matmul` stretches the `(12288, 576)` weight to the batch
        // shape. The batch dim here is always 1 and `x` is contiguous, so the
        // stretch buys nothing and costs a great deal
        // (`examples/connector_probe`, best of 10):
        //
        // | | ms |
        // |---|---:|
        // | `broadcast_matmul` | **64.449** |
        // | flatten to 2D + `matmul` | **4.577** |
        //
        // **14.1x on the projection, 11.1x on the whole connector** — which
        // `tile_batching_ab` had measured at 78.9 ms, 5.8 % of a tile. Across
        // 17 tiles that is ~1.1 s of a caption spent stretching a weight.
        //
        // Identical arithmetic: a contiguous reshape is free, and the products
        // and their summation order are unchanged. `text.rs::linear` records
        // the same finding at the other end of the model (33x at seq 1), so
        // this is a trap of candle's API rather than of either call site.
        let (b, n, k) = x.dims3()?;
        let w = self.proj.to_dtype(x.dtype())?.t()?;
        let out = w.dim(1)?;
        x.reshape((b * n, k))?.matmul(&w)?.reshape((b, n, out))
    }
}

/// Read `vision_config` and `scale_factor` out of a checkpoint's `config.json`.
///
/// Deserialized rather than hard-coded so a different `SmolVLM` size (500M,
/// 2.2B) works without a code change — the tower shape is data, and a constant
/// here would be a second place for it to be wrong.
pub fn vision_config_from_json(config_json: &str) -> Result<(siglip::VisionConfig, usize), String> {
    let v: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| format!("config.json: {e}"))?;
    let scale_factor = v
        .get("scale_factor")
        .and_then(serde_json::Value::as_u64)
        .ok_or("config.json has no scale_factor")? as usize;
    let vc = v.get("vision_config").ok_or("config.json has no vision_config")?;
    let cfg: siglip::VisionConfig = serde_json::from_value(vc.clone())
        .map_err(|e| format!("vision_config does not fit candle's SigLIP shape: {e}"))?;
    Ok((cfg, scale_factor))
}

/// Load the vision half from a safetensors checkpoint on disk.
///
/// # Errors
/// Propagates candle's load errors; a missing tensor names itself, which is
/// the failure mode a wrong prefix produces.
pub fn load(
    weights: &std::path::Path,
    config_json: &str,
    device: &Device,
) -> Result<SmolVlmVision, String> {
    let (cfg, scale_factor) = vision_config_from_json(config_json)?;
    // f32 throughout: step 3 compares against an f32 reference dump, and a
    // dtype difference would read as a tower bug. Quantization is a later,
    // separately-gated decision.
    // SAFETY: the mapped file is owned by the model cache and is not
    // mutated while this process holds it.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, device)
    }
    .map_err(|e| format!("load {}: {e}", weights.display()))?;
    SmolVlmVision::new(&cfg, scale_factor, "model.vision_model", "model.connector", vb)
        .map_err(|e| format!("build vision tower: {e}"))
}
