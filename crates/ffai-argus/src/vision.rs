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
use crate::par::prelude::*;

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
        //
        // One fused pass when enabled; otherwise the reshape/transpose chain,
        // which is kept as the null arm and the oracle. The two are
        // bit-identical — see [`PixelShuffleOp`].
        let x = if fused_shuffle() {
            hidden.apply_op1_no_bwd(&PixelShuffleOp { side, s })?
        } else {
            let x = hidden.reshape((b, side, side, dim))?;
            let x = x.reshape((b, side, side / s, dim * s))?;
            let x = x.transpose(1, 2)?.contiguous()?;
            let x = x.reshape((b, side / s, side / s, dim * s * s))?;
            let x = x.transpose(1, 2)?.contiguous()?;
            x.reshape((b, (side / s) * (side / s), dim * s * s))?
        };
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

/// The pixel shuffle as ONE pass, replacing candle's two transpose-copies.
///
/// # What the chain costs
///
/// The shuffle is expressed as reshape / transpose / reshape / transpose, and
/// each `transpose` needs a `.contiguous()` — candle's generic strided permute,
/// walking element by element. That is **two** full copies of the activation
/// (3.1 MB each at this checkpoint) to perform what is, in total, a single
/// permutation.
///
/// # Why one pass suffices
///
/// Composing the chain's four index maps gives, for `r = r'*s + i` and
/// `c = c'*s + j`:
///
/// ```text
/// out[b][r'][c'][(i*s + j)*dim + d]  =  hidden[b][(r'*s+i)*side + (c'*s+j)][d]
/// ```
///
/// `d` does not participate — it rides along. So for each fixed
/// `(b, r', c', i, j)` this is a straight copy of `dim` **contiguous** floats
/// to a `dim`-aligned **contiguous** destination. One pass, and both streams
/// are sequential runs rather than the strided element walk the generic
/// permute performs.
///
/// The same lesson as `siglip::PackedQkvOp`, which cut its copy 20.4 ms -> 4.0 ms
/// per tile by refusing candle's generic permute for a hand-written one.
///
/// # Numerics
///
/// None. This moves floats without reading them as numbers, so it is
/// **bit-identical** to the chain — its test asserts exact equality, not a
/// tolerance.
struct PixelShuffleOp {
    /// Patch-grid edge, `sqrt(seq)`.
    side: usize,
    /// Shuffle factor; `s*s` patches become one token.
    s: usize,
}

impl candle_core::CustomOp1 for PixelShuffleOp {
    fn name(&self) -> &'static str {
        "ffai-pixel-shuffle"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let candle_core::CpuStorage::F32(x) = storage else {
            candle_core::bail!("ffai-pixel-shuffle expects f32")
        };
        let Some((o, e)) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai-pixel-shuffle expects a contiguous input")
        };
        let x = &x[o..e];
        let (b, seq, dim) = layout.shape().dims3()?;
        let (side, s) = (self.side, self.s);
        if side * side != seq || !side.is_multiple_of(s) || s == 0 {
            candle_core::bail!("ffai-pixel-shuffle: {seq} patches, side {side}, s {s}");
        }
        let out_side = side / s;
        let tokens = out_side * out_side;
        let feat = dim * s * s;
        let n = b * tokens * feat;
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`. The chunks
            // below tile `[0, n)` one token at a time, and each token's loop
            // writes all `s*s` groups of `dim` floats that make up its `feat`
            // features. Every element is therefore initialised before `set_len`.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            crate::cost::copy(n as u64);
            let token = |(t, o): (usize, &mut [f32])| {
                let (bb, rest) = (t / tokens, t % tokens);
                let (rp, cp) = (rest / out_side, rest % out_side);
                for i in 0..s {
                    for j in 0..s {
                        let row = (rp * s + i) * side + (cp * s + j);
                        let src = (bb * seq + row) * dim;
                        let at = (i * s + j) * dim;
                        o[at..at + dim].copy_from_slice(&x[src..src + dim]);
                    }
                }
            };
            if crate::siglip::kernels_parallel_for_probe() {
                dst.par_chunks_mut(feat).enumerate().for_each(token);
            } else {
                dst.chunks_mut(feat).enumerate().for_each(token);
            }
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), (b, tokens, feat).into()))
    }
}

/// Use the fused pixel shuffle? `FFAI_ARGUS_FUSED_SHUFFLE=0` uses the chain.
fn fused_shuffle() -> bool {
    FUSED_SHUFFLE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Toggle the fused pixel shuffle, returning the previous setting.
pub fn set_fused_shuffle(on: bool) -> bool {
    FUSED_SHUFFLE.swap(on, std::sync::atomic::Ordering::Relaxed)
}

static FUSED_SHUFFLE: std::sync::LazyLock<std::sync::atomic::AtomicBool> =
    std::sync::LazyLock::new(|| {
        std::sync::atomic::AtomicBool::new(
            std::env::var("FFAI_ARGUS_FUSED_SHUFFLE").ok().as_deref() != Some("0"),
        )
    });

/// The pixel shuffle, exposed so `examples/kernel_micro_ab` can price the fused
/// kernel against the chain at the op level — the connector is inside
/// `tower_ms` but outside `siglip`'s per-op profile, so it had never been
/// measured on its own.
///
/// # Errors
/// If the patch grid is not square or not divisible by `s`.
pub fn pixel_shuffle_for_probe(hidden: &Tensor, side: usize, s: usize) -> CandleResult<Tensor> {
    if fused_shuffle() {
        return hidden.apply_op1_no_bwd(&PixelShuffleOp { side, s });
    }
    let (b, _seq, dim) = hidden.dims3()?;
    let x = hidden.reshape((b, side, side, dim))?;
    let x = x.reshape((b, side, side / s, dim * s))?;
    let x = x.transpose(1, 2)?.contiguous()?;
    let x = x.reshape((b, side / s, side / s, dim * s * s))?;
    let x = x.transpose(1, 2)?.contiguous()?;
    x.reshape((b, (side / s) * (side / s), dim * s * s))
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
    // f32 throughout: step 3 compares against an f32 reference dump, and a
    // dtype difference would read as a tower bug. Quantization is a later,
    // separately-gated decision.
    // SAFETY: the mapped file is owned by the model cache and is not
    // mutated while this process holds it.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, device)
    }
    .map_err(|e| format!("load {}: {e}", weights.display()))?;
    load_vb(vb, config_json)
}

/// Build the tower from a `VarBuilder` the caller already has.
///
/// The path constructor above is written in terms of this, so a browser and a
/// server build the same tower from the same tensors — the only difference is
/// where the bytes came from. `wasm32-unknown-unknown` has no mmap, so this is
/// the entry point a wasm build reaches (via
/// `VarBuilder::from_buffered_safetensors`).
pub fn load_vb(vb: VarBuilder<'static>, config_json: &str) -> Result<SmolVlmVision, String> {
    let (cfg, scale_factor) = vision_config_from_json(config_json)?;
    SmolVlmVision::new(&cfg, scale_factor, "model.vision_model", "model.connector", vb)
        .map_err(|e| format!("build vision tower: {e}"))
}

#[cfg(test)]
mod tests {
    use super::{PixelShuffleOp, Tensor};
    use candle_core::Device;

    /// The fused pixel shuffle against the reshape/transpose chain it replaces.
    ///
    /// `assert_eq!` on every element, not a tolerance: this kernel moves floats
    /// without reading them as numbers, so there is no reassociation that could
    /// excuse a difference. Anything non-zero is an index bug — and an index bug
    /// here would silently feed the projection the wrong patch, surfacing as a
    /// subtly wrong caption rather than as a crash.
    ///
    /// `arange` rather than random data on purpose: every element is its own
    /// index, so a mis-permutation is visible in the failure message instead of
    /// being one unrecognisable float against another.
    #[test]
    fn fused_pixel_shuffle_matches_the_transpose_chain() {
        let d = Device::Cpu;
        for (side, s, dim) in [(8usize, 4usize, 6usize), (32, 4, 8), (6, 2, 3)] {
            let seq = side * side;
            let hidden = Tensor::arange(0f32, (seq * dim) as f32, &d)
                .expect("arange")
                .reshape((1, seq, dim))
                .expect("reshape");

            let fused =
                hidden.apply_op1_no_bwd(&PixelShuffleOp { side, s }).expect("fused");

            let x = hidden.reshape((1, side, side, dim)).expect("r1");
            let x = x.reshape((1, side, side / s, dim * s)).expect("r2");
            let x = x.transpose(1, 2).expect("t1").contiguous().expect("c1");
            let x = x.reshape((1, side / s, side / s, dim * s * s)).expect("r3");
            let x = x.transpose(1, 2).expect("t2").contiguous().expect("c2");
            let want =
                x.reshape((1, (side / s) * (side / s), dim * s * s)).expect("r4");

            assert_eq!(fused.dims(), want.dims(), "shape at side {side} s {s}");
            let got = fused.flatten_all().expect("f").to_vec1::<f32>().expect("v");
            let want = want.flatten_all().expect("f").to_vec1::<f32>().expect("v");
            assert_eq!(got, want, "pixel shuffle at side {side} s {s} dim {dim}");
        }
    }
}
