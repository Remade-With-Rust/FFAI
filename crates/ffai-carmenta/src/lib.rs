//! # Carmenta — FFai's OCR component
//!
//! Named for the Roman goddess Carmenta, credited in myth with adapting the
//! Greek alphabet into the Latin alphabet — literally the deity of turning
//! foreign marks into readable letters. She keeps the pantheon Roman
//! alongside Mercury (voice) and Argus (vision).
//!
//! Carmenta blends multiple OCR lineages behind one trait so users pick per
//! job (`--engine unlimited-ocr` vs `--engine easy-ocr`), exactly like
//! choosing a codec in ffmpeg:
//!
//! | Engine | Plan |
//! |---|---|
//! | `unlimited-ocr` | Unlimited-OCR lineage — document/layout-strong tier |
//! | `easy-ocr` | EasyOCR lineage (CRAFT detection + CRNN recognition) — broad-language scene-text tier |
//!
//! Both go live in Phase 3 as detection → recognition pipelines on candle,
//! oracle-gated on public OCR ground-truth sets (CER/WER). The pure-Rust
//! `ocrs` crate is tracked as a potential zero-download baseline engine.

use std::sync::Arc;

use ffai_core::engine::{EngineInfo, EngineStatus, OcrEngine, OcrOptions, Task};
use ffai_core::error::{Error, Result};
use ffai_core::registry::EngineRegistry;
use ffai_core::types::{ImageBuffer, OcrOutput};

/// Unlimited-OCR lineage: the document-strong tier.
pub struct UnlimitedOcr;

impl OcrEngine for UnlimitedOcr {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "unlimited-ocr".into(),
            task: Task::Ocr,
            status: EngineStatus::Stub,
            description: "Unlimited-OCR lineage — document/layout tier (Phase 3)".into(),
        }
    }

    fn recognize(&self, _image: &ImageBuffer, _opts: &OcrOptions) -> Result<OcrOutput> {
        Err(Error::NotImplemented { task: Task::Ocr, engine: "unlimited-ocr".into() })
    }
}

/// EasyOCR lineage: CRAFT text detection + CRNN recognition, broad language
/// coverage — the scene-text tier.
pub struct EasyOcr;

impl OcrEngine for EasyOcr {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "easy-ocr".into(),
            task: Task::Ocr,
            status: EngineStatus::Stub,
            description: "EasyOCR lineage (CRAFT+CRNN) — scene-text tier (Phase 3)".into(),
        }
    }

    fn recognize(&self, _image: &ImageBuffer, _opts: &OcrOptions) -> Result<OcrOutput> {
        Err(Error::NotImplemented { task: Task::Ocr, engine: "easy-ocr".into() })
    }
}

/// Install every Carmenta engine into a registry.
pub fn register(reg: &mut EngineRegistry) {
    reg.register_ocr(Arc::new(UnlimitedOcr));
    reg.register_ocr(Arc::new(EasyOcr));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_both_ocr_engines() {
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        assert!(reg.ocr(Some("unlimited-ocr")).is_ok());
        assert!(reg.ocr(Some("easy-ocr")).is_ok());
    }
}
