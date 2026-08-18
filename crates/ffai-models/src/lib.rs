//! # ffai-models
//!
//! Weight management for `FFai`. Core principle: **weights are data, not
//! code** — the repo holds only TOML *manifests* describing each model (files,
//! source, checksums, and crucially its *license*, which is often different
//! from `FFai`'s). Weights are fetched into a local cache, never vendored.
//!
//! Phase 0 ships manifests + cache resolution; the downloader (Hugging Face
//! hub, resumable, checksum-verified) lands in Phase 1.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ffai_core::error::{Error, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// A model whose files are all present locally, with their resolved paths.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub name: String,
    /// The WEIGHTS' license — may be more restrictive than `FFai`'s own.
    pub license: String,
    /// Filename → local path.
    pub files: BTreeMap<String, PathBuf>,
}

impl ResolvedModel {
    /// Path of a required file, or a clear error naming what's missing.
    pub fn file(&self, name: &str) -> Result<&Path> {
        self.files
            .get(name)
            .map(PathBuf::as_path)
            .ok_or_else(|| Error::Model(format!("model `{}` has no file `{name}`", self.name)))
    }
}

/// Resolve one file from the shared Hugging Face cache, downloading it when
/// `cache_only` is false. Uses the same cache as `transformers` and
/// `faster-whisper`, so a model either side already has is not fetched twice.
#[cfg(feature = "fetch")]
fn hub_download(repo: &str, filename: &str, cache_only: bool) -> Result<PathBuf> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| Error::Model(format!("hf_repo `{repo}` is not in `owner/name` form")))?;
    let client = hf_hub::HFClientSync::new()
        .map_err(|e| Error::Model(format!("hugging face client init failed: {e}")))?;
    client
        .model(owner, name)
        .download_file()
        .filename(filename)
        .local_files_only(cache_only)
        .send()
        .map_err(|e| Error::Model(format!("{repo}/{filename}: {e}")))
}

/// Without `fetch`, nothing is downloaded — and the error says which file to
/// supply rather than failing as a missing symbol.
#[cfg(not(feature = "fetch"))]
fn hub_download(repo: &str, filename: &str, _cache_only: bool) -> Result<PathBuf> {
    Err(Error::Model(format!(
        "{repo}/{filename} is not present locally and this build has the          `fetch` feature disabled — place the file in the model directory, or          enable `ffai-models/fetch` to download it"
    )))
}

/// Verify a downloaded file against its manifest checksum, when one is
/// declared. A mismatch is an error, never a warning: silently running on
/// unexpected weights would invalidate every measurement taken with them.
fn verify_checksum(path: &Path, file: &ModelFile) -> Result<()> {
    let Some(expected) = &file.sha256 else {
        return Ok(());
    };
    let bytes = std::fs::read(path)?;
    let actual: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if actual != expected.to_ascii_lowercase() {
        return Err(Error::Model(format!(
            "checksum mismatch for {}: manifest says {expected}, file is {actual}",
            path.display()
        )));
    }
    Ok(())
}

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
    /// restrictive than `FFai`'s MIT/Apache code license.
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

    /// Directory this model's files live in when placed manually.
    #[must_use]
    pub fn cache_path(&self) -> PathBuf {
        cache_dir().join("models").join(&self.name)
    }

    /// True when every listed file already resolves locally — no network.
    #[must_use]
    pub fn is_cached(&self) -> bool {
        !self.files.is_empty()
            && self
                .files
                .iter()
                .all(|f| self.local_path(&f.name).is_some())
    }

    /// Resolve one file without touching the network: a manual placement
    /// under [`Self::cache_path`] wins, otherwise the shared Hugging Face
    /// cache (the same one `transformers`/`faster-whisper` use, so a model
    /// downloaded by either side is not downloaded twice).
    #[must_use]
    pub fn local_path(&self, name: &str) -> Option<PathBuf> {
        let manual = self.cache_path().join(name);
        if manual.exists() {
            return Some(manual);
        }
        hub_download(self.hf_repo.as_ref()?, name, true).ok()
    }

    /// Resolve every file, downloading from the Hugging Face hub as needed.
    ///
    /// Downloads are cached, so this is cheap on repeat calls — but it is
    /// still network I/O the first time, which is why benchmarks warm the
    /// cache outside any timed region (see docs/benchmarking.md).
    pub fn fetch(&self) -> Result<ResolvedModel> {
        let mut files = BTreeMap::new();
        for file in &self.files {
            if let Some(path) = self.local_path(&file.name) {
                verify_checksum(&path, file)?;
                files.insert(file.name.clone(), path);
                continue;
            }
            let repo = self.hf_repo.as_ref().ok_or_else(|| {
                Error::Model(format!(
                    "model `{}` declares no hf_repo and `{}` is not present under {}",
                    self.name,
                    file.name,
                    self.cache_path().display()
                ))
            })?;
            let path = hub_download(repo, &file.name, false)?;
            verify_checksum(&path, file)?;
            files.insert(file.name.clone(), path);
        }
        Ok(ResolvedModel {
            name: self.name.clone(),
            license: self.license.clone(),
            files,
        })
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

/// The `FFai` model cache root: `$FFAI_CACHE` or `<os cache dir>/ffai`.
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
