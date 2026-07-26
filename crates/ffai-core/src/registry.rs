//! The engine registry — FFai's equivalent of ffmpeg's codec registry.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::engine::{AsrEngine, EngineInfo, OcrEngine, Task, TtsEngine, VlmEngine};
use crate::error::{Error, Result};

/// Holds every registered engine, keyed by task and name.
///
/// Feature crates (`ffai-mercury`, `ffai-carmenta`, `ffai-argus`) each expose
/// a `register(&mut EngineRegistry)` that installs their engines; the CLI (or
/// any embedding application) composes the registry it wants. A future plugin
/// is just an engine registered at runtime — the architecture doesn't change.
#[derive(Default)]
pub struct EngineRegistry {
    asr: BTreeMap<String, Arc<dyn AsrEngine>>,
    tts: BTreeMap<String, Arc<dyn TtsEngine>>,
    ocr: BTreeMap<String, Arc<dyn OcrEngine>>,
    vlm: BTreeMap<String, Arc<dyn VlmEngine>>,
    // The default per task is the FIRST engine registered (the reference
    // engine), not the alphabetically first.
    asr_default: Option<String>,
    tts_default: Option<String>,
    ocr_default: Option<String>,
    vlm_default: Option<String>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_asr(&mut self, engine: Arc<dyn AsrEngine>) {
        let name = engine.info().name;
        self.asr_default.get_or_insert_with(|| name.clone());
        self.asr.insert(name, engine);
    }

    pub fn register_tts(&mut self, engine: Arc<dyn TtsEngine>) {
        let name = engine.info().name;
        self.tts_default.get_or_insert_with(|| name.clone());
        self.tts.insert(name, engine);
    }

    pub fn register_ocr(&mut self, engine: Arc<dyn OcrEngine>) {
        let name = engine.info().name;
        self.ocr_default.get_or_insert_with(|| name.clone());
        self.ocr.insert(name, engine);
    }

    pub fn register_vlm(&mut self, engine: Arc<dyn VlmEngine>) {
        let name = engine.info().name;
        self.vlm_default.get_or_insert_with(|| name.clone());
        self.vlm.insert(name, engine);
    }

    /// Resolve an ASR engine; `None` selects the default (first registered).
    pub fn asr(&self, name: Option<&str>) -> Result<Arc<dyn AsrEngine>> {
        resolve(&self.asr, name, self.asr_default.as_deref(), Task::Asr)
    }

    pub fn tts(&self, name: Option<&str>) -> Result<Arc<dyn TtsEngine>> {
        resolve(&self.tts, name, self.tts_default.as_deref(), Task::Tts)
    }

    pub fn ocr(&self, name: Option<&str>) -> Result<Arc<dyn OcrEngine>> {
        resolve(&self.ocr, name, self.ocr_default.as_deref(), Task::Ocr)
    }

    pub fn vlm(&self, name: Option<&str>) -> Result<Arc<dyn VlmEngine>> {
        resolve(&self.vlm, name, self.vlm_default.as_deref(), Task::Vlm)
    }

    /// All engine metadata, ordered by task then name (for `ffai engines`).
    pub fn list(&self) -> Vec<EngineInfo> {
        let mut out: Vec<EngineInfo> = Vec::new();
        out.extend(self.asr.values().map(|e| e.info()));
        out.extend(self.tts.values().map(|e| e.info()));
        out.extend(self.ocr.values().map(|e| e.info()));
        out.extend(self.vlm.values().map(|e| e.info()));
        out.sort_by(|a, b| a.task.cmp(&b.task).then_with(|| a.name.cmp(&b.name)));
        out
    }
}

fn resolve<E: ?Sized>(
    map: &BTreeMap<String, Arc<E>>,
    name: Option<&str>,
    default: Option<&str>,
    task: Task,
) -> Result<Arc<E>> {
    match name.or(default) {
        Some(n) => map.get(n).cloned().ok_or_else(|| Error::UnknownEngine {
            task,
            name: n.to_string(),
        }),
        None => Err(Error::UnknownEngine {
            task,
            name: "<no engines registered>".to_string(),
        }),
    }
}
