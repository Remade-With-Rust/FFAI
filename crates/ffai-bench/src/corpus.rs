//! Hashed corpus manifests, ported from Prometheus (`prom-corpus`).
//!
//! Corpus rigor is what makes a result repeatable: a [`Manifest`] is a
//! hashed, versioned, content-classified list of test clips with a
//! **clip-level** train/holdout split baked in. Each clip records its SHA-256
//! so [`Manifest::verify`] proves the bytes on disk are exactly the bytes a
//! result was measured on, and the manifest's own hash pins "which clips at
//! which versions" into every ledger record — no result silently drifts onto
//! different data. Per-clip `license` is mandatory: `FFai` is public, and only
//! redistributable corpora belong in the repo.

use ffai_core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Train/holdout split, a property of the clip so it never drifts across runs.
/// Engines may be tuned on `train`; claims are measured on `holdout` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Holdout,
}

/// Content class, for stratified reporting ("we win on clean speech, lose on
/// noisy") — extended from Prometheus's audio/video classes to `FFai`'s tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentClass {
    // ---- speech / audio (Mercury) ----
    CleanSpeech,
    NoisySpeech,
    Accented,
    Multilingual,
    // ---- OCR (Carmenta) ----
    DocumentScan,
    SceneText,
    Handwritten,
    // ---- vision (Argus) ----
    Photo,
    Diagram,
    Video,
    Other,
}

/// One test item: media file + ground truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipEntry {
    /// Stable id (appears in per-clip result tables).
    pub id: String,
    /// Media path, relative to the manifest file's directory.
    pub path: PathBuf,
    /// Ground-truth path (transcript / OCR text / reference captions),
    /// relative to the manifest directory.
    pub ground_truth: Option<PathBuf>,
    pub class: ContentClass,
    pub split: Split,
    /// License of THIS clip (mandatory — public repo, public corpora only).
    pub license: String,
    /// Hex SHA-256 of the media file's bytes.
    pub sha256: String,
}

/// A corpus manifest (`corpora/*.toml`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: u32,
    /// Task this corpus measures: "asr", "tts", "ocr", "vlm".
    pub task: String,
    #[serde(default)]
    pub clips: Vec<ClipEntry>,
    /// Directory the clip paths are relative to (set on load).
    #[serde(skip)]
    pub base_dir: PathBuf,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut m: Self =
            toml::from_str(&text).map_err(|e| Error::Other(format!("bad corpus manifest: {e}")))?;
        m.base_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Ok(m)
    }

    /// Absolute path of a clip's media file.
    #[must_use]
    pub fn clip_path(&self, clip: &ClipEntry) -> PathBuf {
        self.base_dir.join(&clip.path)
    }

    /// Ground-truth text for a clip, if declared.
    pub fn ground_truth(&self, clip: &ClipEntry) -> Result<Option<String>> {
        match &clip.ground_truth {
            Some(rel) => Ok(Some(std::fs::read_to_string(self.base_dir.join(rel))?)),
            None => Ok(None),
        }
    }

    pub fn holdout(&self) -> impl Iterator<Item = &ClipEntry> {
        self.clips.iter().filter(|c| c.split == Split::Holdout)
    }

    /// Deterministic hash over (name, version, clip ids + hashes + splits) —
    /// the fingerprint every ledger record pins its data to.
    #[must_use]
    pub fn manifest_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        hasher.update(self.version.to_le_bytes());
        let mut clips: Vec<&ClipEntry> = self.clips.iter().collect();
        clips.sort_by(|a, b| a.id.cmp(&b.id));
        for c in clips {
            hasher.update(c.id.as_bytes());
            hasher.update(c.sha256.as_bytes());
            hasher.update(match c.split {
                Split::Train => b"t",
                Split::Holdout => b"h",
            });
        }
        hex(&hasher.finalize())
    }

    /// Verify every clip's bytes match its recorded SHA-256. Returns the ids
    /// that failed (missing or mismatched); empty = corpus is exactly as pinned.
    pub fn verify(&self) -> Result<Vec<String>> {
        let mut bad = Vec::new();
        for clip in &self.clips {
            let path = self.clip_path(clip);
            match std::fs::read(&path) {
                Ok(bytes) => {
                    if file_sha256(&bytes) != clip.sha256.to_ascii_lowercase() {
                        bad.push(clip.id.clone());
                    }
                }
                Err(_) => bad.push(clip.id.clone()),
            }
        }
        Ok(bad)
    }
}

/// SHA-256 of a byte slice as lowercase hex (for authoring manifests).
#[must_use]
pub fn file_sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        toml::from_str::<Manifest>(
            r#"
            name = "asr-smoke"
            version = 1
            task = "asr"

            [[clips]]
            id = "clip-a"
            path = "a.wav"
            ground_truth = "a.txt"
            class = "clean_speech"
            split = "holdout"
            license = "CC-BY-4.0"
            sha256 = "aa"

            [[clips]]
            id = "clip-b"
            path = "b.wav"
            class = "noisy_speech"
            split = "train"
            license = "CC-BY-4.0"
            sha256 = "bb"
            "#,
        )
        .unwrap()
    }

    #[test]
    fn hash_is_stable_and_order_independent() {
        let m = manifest();
        let h1 = m.manifest_hash();
        let mut m2 = m.clone();
        m2.clips.reverse();
        assert_eq!(h1, m2.manifest_hash());
        // changing a clip's bytes changes the fingerprint
        let mut m3 = m;
        m3.clips[0].sha256 = "cc".into();
        assert_ne!(h1, m3.manifest_hash());
    }

    #[test]
    fn holdout_filters_split() {
        let m = manifest();
        let ids: Vec<_> = m.holdout().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["clip-a"]);
    }
}
