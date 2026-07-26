//! # ffai-models
//!
//! Weight management for FFai. Core principle: **weights are data, not
//! code** — the repo holds only TOML *manifests* describing each model (files,
//! source, checksums, and crucially its *license*, which is often different
//! from FFai's). Weights are fetched into a local cache, never vendored.
//!
//! Phase 0 ships manifests + cache resolution; the downloader (Hugging Face
//! hub, resumable, checksum-verified) lands in Phase 1.

use std::path::{Path, PathBuf};

use ffai_core::error::{Error, Result};
use serde::Deserialize;

/// One weight/config file belonging to a model.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelFile {
    /// Filename within the model's cache directory (and its HF repo).
    pub name: String,
    /// Hex SHA-256, verified after download when present.
    pub sha256: Option<String>,
}

/// A model manifest (`models/*.toml`).
#[derive(Debug, Clone, Deserialize)]
pub struct ModelManifest {
    pub name: String,
    /// Task tag: "asr", "tts", "ocr", "vlm".
    pub task: String,
    pub description: Option<String>,
    /// The WEIGHTS' license — surfaced to users because it may be more
    /// restrictive than FFai's MIT/Apache code license.
    pub license: String,
    /// Hugging Face repo id, e.g. "openai/whisper-tiny".
    pub hf_repo: Option<String>,
    #[serde(default)]
    pub files: Vec<ModelFile>,
}

impl ModelManifest {
    pub fn from_toml(text: &str) -> Result<Self> {
        toml::from_str(text).map_err(|e| Error::Model(format!("bad manifest: {e}")))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_toml(&text)
    }

    /// Directory this model's files live in inside the cache.
    pub fn cache_path(&self) -> PathBuf {
        cache_dir().join("models").join(&self.name)
    }

    /// True when every listed file is present in the cache.
    pub fn is_cached(&self) -> bool {
        !self.files.is_empty()
            && self
                .files
                .iter()
                .all(|f| self.cache_path().join(&f.name).exists())
    }

    /// Download missing files. Lands in Phase 1 (HF hub, resumable,
    /// SHA-256-verified); manifests and cache layout are stable now so
    /// engines can already resolve paths.
    pub fn fetch(&self) -> Result<()> {
        Err(Error::Model(format!(
            "downloading `{}` is not implemented in Phase 0 — place files under {} \
             manually, or wait for the Phase 1 fetcher",
            self.name,
            self.cache_path().display()
        )))
    }
}

/// Load every `*.toml` manifest in a directory (typically `models/`).
pub fn load_dir(dir: &Path) -> Result<Vec<ModelManifest>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            out.push(ModelManifest::load(&path)?);
        }
    }
    out.sort_by(|a, b| a.task.cmp(&b.task).then_with(|| a.name.cmp(&b.name)));
    Ok(out)
}

/// The FFai model cache root: `$FFAI_CACHE` or `<os cache dir>/ffai`.
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("FFAI_CACHE") {
        return PathBuf::from(dir);
    }
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("ffai")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parses_and_surfaces_license() {
        let m = ModelManifest::from_toml(
            r#"
            name = "whisper-tiny"
            task = "asr"
            license = "Apache-2.0"
            hf_repo = "openai/whisper-tiny"

            [[files]]
            name = "model.safetensors"
            "#,
        )
        .unwrap();
        assert_eq!(m.name, "whisper-tiny");
        assert_eq!(m.license, "Apache-2.0");
        assert_eq!(m.files.len(), 1);
        assert!(!m.is_cached());
    }
}
