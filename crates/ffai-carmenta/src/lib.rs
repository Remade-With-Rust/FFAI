//! # Carmenta — FFai's OCR component
//!
//! Named for the Roman goddess Carmenta, credited in myth with adapting the
//! Greek alphabet into the Latin alphabet — literally the deity of turning
//! foreign marks into readable letters. She keeps the pantheon Roman
//! alongside Mercury (voice) and Argus (vision).
//!
//! Mission plan: `docs/carmenta-mission-plan.md` — four functions (LIVE,
//! DOCUMENT, LONG, FORMULA) over one detection → recognition core.
//!
//! ## Engines
//!
//! | Engine | Lineage | Status |
//! |---|---|---|
//! | `craft-crnn` | CRAFT detection + english_g2 CRNN recognition (the EasyOCR stack, pure Rust on candle) | experimental — per-stage oracle-gated, corpus gates in the ledger |
//!
//! Naming note (M-C1 stub reconciliation): Phase 0 registered `easy-ocr` and
//! `unlimited-ocr` as honest stubs. Engines are named by *lineage*, so the
//! shipped engine is `craft-crnn` — it IS the EasyOCR model stack, but it is
//! not EasyOCR, and a name that claims otherwise would be borrowed credit.
//! The document-tier engine (PP-OCRv5-class) lands with its own lineage name;
//! `unlimited-ocr` was retired with it.
//!
//! ## Stages
//!
//! Each stage is independently callable and independently oracle-tested
//! (mission plan §2): [`craft`] (detection net), [`boxes`] (map → boxes →
//! lines), [`crnn`] (recognition net + CTC), [`image`] (preprocessing),
//! composed by [`engine`].

pub mod boxes;
pub mod content;
pub mod craft;
pub mod crnn;
pub mod engine;
pub mod image;
pub mod conv3x3;
pub mod live;
pub mod suppress;
pub mod mobiledet;
pub mod parseq;
pub mod profile;

use std::sync::Arc;

use ffai_core::registry::EngineRegistry;

/// Word-crop pads for the PARSeq path, `(x, y)` as fractions of line
/// height. Defaults are the synthetic-render values; `FFAI_PARSEQ_PAD_X` /
/// `FFAI_PARSEQ_PAD_Y` override for sweeps and for content-specific tuning.
pub fn parseq_pads() -> (f32, f32) {
    use std::sync::OnceLock;
    static P: OnceLock<(f32, f32)> = OnceLock::new();
    *P.get_or_init(|| {
        let get = |k: &str, d: f32| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        (get("FFAI_PARSEQ_PAD_X", 0.10), get("FFAI_PARSEQ_PAD_Y", 0.12))
    })
}

/// Install every Carmenta engine into a registry.
pub fn register(reg: &mut EngineRegistry) {
    reg.register_ocr(Arc::new(engine::CraftCrnn::new()));
    reg.register_ocr(Arc::new(engine::CraftCrnn::new_parseq()));
    reg.register_ocr(Arc::new(engine::CraftCrnn::new_mobiledet(engine::RecStage::Crnn)));
    reg.register_ocr(Arc::new(engine::CraftCrnn::new_mobiledet(engine::RecStage::Parseq)));
    // `composed-*` is deliberately NOT registered. It runs both detectors and
    // was measured dominated on every corpus class (§8.19) — on frames it ties
    // craft-parseq exactly, on receipts it loses to it. The code path stays
    // reachable through `CraftCrnn::new_composed` so the probe is reproducible;
    // shipping a slower engine that never wins would be clutter.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_craft_crnn_as_default() {
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        assert!(reg.ocr(Some("craft-crnn")).is_ok());
        assert_eq!(reg.ocr(None).unwrap().info().name, "craft-crnn");
    }
}
