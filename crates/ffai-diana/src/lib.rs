//! # Diana — `FFai`'s detection component
//!
//! Named for the Roman goddess of the hunt — fast, precise detection —
//! keeping the pantheon Roman alongside Mercury (voice), Carmenta (the
//! alphabet) and Argus (vision).
//!
//! Mission plan: `docs/diana-mission-plan.md`. YOLO26 reimplemented on the
//! candle spine so official Ultralytics `.pt` checkpoints can be deployed
//! from a pure-Rust binary with no Python at inference time.
//!
//! ## What this crate does NOT do
//!
//! It does not read `.pt`. Conversion is offline, versioned and audited
//! (`tools/diana_convert.py`, mission plan §7): the checkpoint becomes
//! safetensors plus a manifest, and this crate loads only those. Weights
//! are AGPL-3.0 and are never vendored or redistributed — the user fetches
//! their own.
//!
//! ## Stages
//!
//! Each is independently callable and independently oracle-tested
//! (mission plan §2): [`blocks`] (the building blocks), [`backbone`]
//! (stem → C3k2 stages → SPPF → C2PSA), with the neck, head and decode
//! following.
//!
//! ## Architecture facts, read off the checkpoint
//!
//! These are not from a paper figure — they were probed from the artifact
//! (`tools/diana_probe_arch.py`) and every one of them would break the port
//! silently if guessed:
//!
//! - `reg_max = 1`, so **DFL is gone**: the 4 box channels are `(l,t,r,b)`
//!   distances from the anchor, not distribution bins.
//! - `end2end = true`, so boxes decode as **xyxy** and there is no NMS.
//! - The head has both a one-to-many (`cv2`/`cv3`) and a one-to-one branch;
//!   **inference uses one2one only**. The conversion drops the other.
//! - Ultralytics shares one `nn.SiLU()` across every default-activation
//!   `Conv`, so a module walk under-reports activations. 93 Convs have
//!   `SiLU`, 9 have none — recorded per-Conv in the arch fixture.

pub mod backbone;
pub mod blocks;
pub mod config;
pub mod conv3x3;
pub mod cpuop;
pub mod cputime;
pub mod depth_engine;
pub mod depth_head;
pub mod depth_ops;
pub mod direct3x3;
pub mod direct3x3_avx2;
pub mod dwconv;
pub mod engine;
pub mod epilogue;
pub mod head;
pub mod image;
pub mod live;
pub mod neck;
pub mod par;
pub mod parallel;
pub mod profile;
pub mod track;
pub mod silu;
pub mod telemetry;
pub mod transpose;
pub mod silu_avx2;
pub mod smallgains;

pub use config::ModelConfig;

use std::sync::Arc;

use ffai_core::registry::EngineRegistry;

/// Install every Diana engine into a registry.
///
/// Two engines, because the letterbox geometry is a distinct MEASUREMENT
/// configuration, not just a knob: the bench's matched-tier gate compares
/// against whichever reference declares the same geometry, and conflating
/// the two is the exact defect M-D0 was created to prevent.
///
/// `yolo26n` (rectangular) is the default — it is what Ultralytics itself
/// defaults to, and at this tier it is faster AND more accurate.
/// `yolo26n-square` is the configuration the parity oracle pins.
pub fn register(reg: &mut EngineRegistry) {
    // n first, so it stays the registry default (the default is the FIRST
    // engine registered, not the alphabetically first).
    for tier in engine::TIERS {
        reg.register_detect(Arc::new(engine::Yolo26::build(
            tier,
            image::Geometry::Rect,
            "models",
        )));
        reg.register_detect(Arc::new(engine::Yolo26::build(
            tier,
            image::Geometry::Square,
            "models",
        )));
        // Depth shares the backbone and neck but not the checkpoint, so it
        // registers as its own engine per tier. Registration is lazy — the
        // model loads on first use — so listing depth engines costs nothing
        // for a user who only converted detect weights.
        reg.register_depth(Arc::new(depth_engine::Yolo26Depth::build(
            tier,
            image::Geometry::Rect,
            "models",
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_yolo26n_as_default() {
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        assert!(reg.detect(Some("yolo26n")).is_ok());
        assert_eq!(reg.detect(None).unwrap().info().name, "yolo26n");
    }
}
