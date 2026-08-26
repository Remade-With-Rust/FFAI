//! # Argus — `FFai`'s vision-language component
//!
//! Named for Argus Panoptes, the all-seeing hundred-eyed watchman: image
//! captioning, visual Q&A, and video understanding.
//!
//! # What runs
//!
//! **`SmolVLM`-256M-Instruct on candle** — a `SigLIP` tower, a pixel-shuffle
//! connector and a Llama text decoder, each gated tensor-by-tensor against the
//! reference implementation (`docs/plans/argus-launch-plan.md`, steps 3-6).
//! Video understanding composes `ffai-media::stream_frames` (rff-backed
//! sampling) with per-frame captioning into a timed track.
//!
//! **`mistral.rs` is not rejected** — it stays the documented path for the
//! serving concerns it owns (quantized weights, grammar-constrained JSON), and
//! the `mistralrs-backend` feature is reserved for it. It is not the path here
//! because the version proven to serve `SmolVLM` is a git revision, and
//! `cargo publish` refuses a git dependency — a constraint that has already
//! made every downstream `FFai` crate unpublishable once.

pub mod cost;
pub mod decode;
pub mod engine;
pub mod preprocess;
pub mod prompt;
pub mod siglip;
pub mod text;
pub mod vision;

use std::sync::Arc;

use ffai_core::registry::EngineRegistry;

pub use engine::SmolVlm;

/// Install every Argus engine into a registry.
pub fn register(reg: &mut EngineRegistry) {
    reg.register_vlm(Arc::new(SmolVlm::new()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffai_core::engine::EngineStatus;

    #[test]
    fn registers_vlm_engine() {
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        let e = reg.vlm(None).expect("vlm engine");
        assert_eq!(e.info().name, "smolvlm");
        // The status is what `ffai engines` prints. `Stable` is defined as
        // "oracle-gated against a reference implementation" — a claim steps
        // 3-6 earned and that this assert stops anyone quietly re-stubbing.
        assert_eq!(e.info().status, EngineStatus::Stable);
    }
}
