//! # Argus — `FFai`'s vision-language component
//!
//! Named for Argus Panoptes, the all-seeing hundred-eyed watchman: image
//! captioning, visual Q&A, and video understanding.
//!
//! Backend plan (Phase 4): **mistral.rs** — the inference engine built on
//! candle (`FFai`'s tensor spine, so buffers are shared without conversion) —
//! running Qwen-VL / LLaVA-class models with quantization. Video
//! understanding composes `ffai-media::sample_frames` (rff-backed keyframe
//! sampling) with per-frame or windowed captioning into a timed track.

use std::sync::Arc;

use ffai_core::engine::{EngineInfo, EngineStatus, Task, VlmEngine, VlmOptions};
use ffai_core::error::{Error, Result};
use ffai_core::registry::EngineRegistry;
use ffai_core::types::{ImageBuffer, TimedSegment, VideoFrame};

/// mistral.rs-backed VLM engine (Qwen-VL / LLaVA-class, quantized).
pub struct MistralRs;

impl VlmEngine for MistralRs {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "mistralrs".into(),
            task: Task::Vlm,
            status: EngineStatus::Stub,
            description: "mistral.rs VLM engine — Qwen-VL/LLaVA on candle (Phase 4)".into(),
        }
    }

    fn describe_image(&self, _image: &ImageBuffer, _opts: &VlmOptions) -> Result<String> {
        Err(Error::NotImplemented {
            task: Task::Vlm,
            engine: "mistralrs".into(),
        })
    }

    fn describe_video(
        &self,
        _frames: &[VideoFrame],
        _opts: &VlmOptions,
    ) -> Result<Vec<TimedSegment<String>>> {
        Err(Error::NotImplemented {
            task: Task::Vlm,
            engine: "mistralrs".into(),
        })
    }
}

/// Install every Argus engine into a registry.
pub fn register(reg: &mut EngineRegistry) {
    reg.register_vlm(Arc::new(MistralRs));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_vlm_engine() {
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        assert!(reg.vlm(None).is_ok());
    }
}
