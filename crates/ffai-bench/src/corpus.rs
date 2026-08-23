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
    /// The question or instruction for this item — VLM corpora only.
    ///
    /// **Inline, not a side file, and that is deliberate.** A VLM benchmark
    /// item is an (image, question) pair: change the wording of the question
    /// and the score changes, so the question is part of what a claim must
    /// pin. Held inline it falls under [`Manifest::manifest_hash`] for free,
    /// the way clip bytes fall under their per-clip SHA-256. A prompt in a
    /// side file that nothing hashes is an unpinned input, which is the same
    /// class of defect as an unpinned decode config.
    ///
    /// `None` on non-VLM corpora, and on VLM items that are plain captioning
    /// (COCO-CIDEr style), where the engine's own default prompt applies.
    #[serde(default)]
    pub prompt: Option<String>,
    pub class: ContentClass,
    pub split: Split,
    /// License of THIS clip (mandatory — public repo, public corpora only).
    pub license: String,
    /// Hex SHA-256 of the media file's bytes.
    pub sha256: String,
}

/// The benchmark's OWN evaluator, declared by the corpus that needs it.
///
/// **Why this lives on the corpus and not on the engine or the reference.**
/// For ASR and OCR the metric is WER/CER — implementation-independent
/// arithmetic over two strings, so `crate::metrics` can own it. For a VLM
/// benchmark it is not: most VLM benchmarks are multiple-choice or
/// short-answer, so *answer extraction* — pulling "B" or "the third one" out
/// of free text — is part of the metric, and every benchmark does it
/// differently. A scorer we wrote would be a scorer tuned, however
/// unconsciously, to our own output. That is the exact mechanism that produced
/// a 2.8×-biased metric in the Carmenta campaign and four shipped mechanisms
/// that turned out to be artifacts of it.
///
/// So the evaluator travels with the benchmark, the way `omnidocbench-eval`
/// belongs to `OmniDocBench` and not to us. **This crate never scores a VLM
/// answer.** It runs predictions, hands them to the declared scorer, and
/// records what comes back.
///
/// # The contract
///
/// `command` is argv with `{predictions}` replaced by a path to a JSONL file
/// this crate writes, one line per holdout item:
///
/// ```jsonl
/// {"id": "ocrbench-0001", "path": "clips/x.png", "prediction": "the model's raw answer"}
/// ```
///
/// The scorer must print **one JSON object on stdout**:
///
/// ```json
/// {"score": 612.0, "metric": "OCRBench", "n": 1000}
/// ```
///
/// `score` is required; `metric` defaults to [`Self::metric`]; `n` is the item
/// count the scorer actually scored and is checked against the holdout size,
/// because a scorer that silently scored a subset has voided the comparison
/// (work-count parity — `codec-measurement` §4).
///
/// # Security — a corpus SELECTS a scorer, it does not DEFINE one
///
/// This block briefly carried an inline argv, which moved the trust boundary:
/// a `corpora/*.toml` had always been pure data, and suddenly a file shaped
/// like data could run code. The danger was never that execution is
/// *possible* — `references.toml` always could — but that it became
/// **invisible**, on one line among a thousand lines of hashes and prompts
/// that a reviewer will skim.
///
/// So [`Self::name`] now names a `[[scorer]]` declared in
/// `corpora/references.toml`, which is already read and reviewed as
/// executable. **Data selects from a set; code defines the set.**
///
/// [`Self::command`] survives only so that an older corpus fails loudly with
/// an explanation, and it is refused unless `FFAI_BENCH_ALLOW_CORPUS_SCORER=1`
/// is set deliberately. `metric` and `scale` stay here because they are facts
/// about the benchmark's numbers, not instructions to execute anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScorerSpec {
    /// The scorer this corpus selects — a NAME resolved against the
    /// `[[scorer]]` entries in `corpora/references.toml`.
    ///
    /// A name is data. An argv is code. This field is the safe one.
    pub name: String,
    /// **DEPRECATED and refused by default.** An inline argv.
    ///
    /// Present only so that a corpus written against the earlier format fails
    /// with an explanation instead of a parse error. Running it requires
    /// `FFAI_BENCH_ALLOW_CORPUS_SCORER=1`, because a data-shaped file that
    /// executes code is exactly the trust-boundary move this field exists to
    /// undo. See [`crate::reference::NamedScorer`].
    #[serde(default)]
    pub command: Option<Vec<String>>,
    /// Optional argv printing a version string, recorded in the ledger. A
    /// score is not reproducible without knowing which evaluator produced it.
    #[serde(default)]
    pub version_command: Option<Vec<String>>,
    /// The metric's name as this benchmark reports it (`"OCRBench"`, `"ANLS"`).
    pub metric: String,
    /// Divisor mapping the raw score onto 0..=1 for the gate arithmetic:
    /// 1000 for `OCRBench`, 100 for percentage accuracy, 1.0 for `ANLS`.
    ///
    /// **Declared, never inferred.** Guessing a scale from an observed value
    /// would silently rescale someone's benchmark; a wrong guess is invisible
    /// because the normalised number still looks plausible.
    pub scale: f64,
}

/// A corpus manifest (`corpora/*.toml`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: u32,
    /// Task this corpus measures: "asr", "tts", "ocr", "detect", "vlm".
    pub task: String,
    /// The benchmark's own evaluator. Required for `task = "vlm"` (see
    /// [`ScorerSpec`]); unused by every other task, whose metrics are
    /// implementation-independent arithmetic in `crate::metrics`.
    #[serde(default)]
    pub scorer: Option<ScorerSpec>,
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

    /// Deterministic hash over (name, version, clip ids + hashes + splits +
    /// prompts) — the fingerprint every ledger record pins its data to.
    ///
    /// **The prompt is hashed only when present, and that is what keeps this
    /// backwards-compatible.** For every corpus that predates VLM support the
    /// prompt is `None`, no bytes are fed to the hasher, and the fingerprint
    /// is byte-for-byte the one the old algorithm produced — so no existing
    /// ledger line's `corpus_manifest_hash` drifts. `hashes_are_stable_for_
    /// promptless_corpora` pins that by recomputing the OLD algorithm
    /// independently and asserting equality.
    ///
    /// It is hashed at all because a VLM benchmark item is an (image,
    /// question) pair: reword the question and the score changes. A question
    /// outside the fingerprint would be an unpinned input — the same class of
    /// defect as an unpinned decode config, and just as invisible.
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
            if let Some(p) = &c.prompt {
                // Length-prefixed so two adjacent fields cannot be re-parsed
                // as one: without it, prompts "ab"+"c" and "a"+"bc" would feed
                // the hasher identical bytes.
                hasher.update(b"p");
                hasher.update((p.len() as u64).to_le_bytes());
                hasher.update(p.as_bytes());
            }
        }
        // The SCORER is part of the fingerprint, for the same reason the
        // prompt is: it determines the number.
        //
        // Found the hard way. Two runs of the same clips under `scale = 1000`
        // and the corrected `scale = 50` produced normalised scores of 0.04
        // and 0.80 — a twentyfold difference — under an IDENTICAL corpus hash,
        // because the `[scorer]` block sat outside the fingerprint. The
        // ledger's contract is that a line is sufficient on its own to
        // reproduce the run; a scorer that can change without the fingerprint
        // moving breaks exactly that.
        //
        // Hashed only when present, so every pre-VLM corpus keeps its existing
        // fingerprint and no historical ledger line stops matching its data.
        if let Some(s) = &self.scorer {
            hasher.update(b"s");
            // The argv is no longer hashed here because it no longer lives
            // here — it is resolved from references.toml, and the RESOLVED
            // command is recorded on the ledger line instead (VlmScore::command),
            // which is what keeps a run reproducible from its own record.
            let inline = s.command.iter().flatten();
            for part in std::iter::once(&s.name)
                .chain(std::iter::once(&s.metric))
                .chain(inline)
            {
                hasher.update((part.len() as u64).to_le_bytes());
                hasher.update(part.as_bytes());
            }
            hasher.update(s.scale.to_le_bytes());
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

    /// The VLM prompt is part of the benchmark, so it must be part of the
    /// fingerprint. Reword a question and you are measuring a different thing;
    /// a fingerprint that did not move would let two incomparable runs sit in
    /// the ledger claiming the same corpus.
    #[test]
    fn rewording_a_prompt_moves_the_fingerprint() {
        let mut m = manifest();
        m.clips[0].prompt = Some("What is in this image?".into());
        let h1 = m.manifest_hash();
        m.clips[0].prompt = Some("What is shown in this image?".into());
        assert_ne!(h1, m.manifest_hash());
    }

    /// Adding the prompt to the hash must NOT move any existing corpus's
    /// fingerprint, or every historical ledger line silently stops matching
    /// its own data.
    ///
    /// This recomputes the OLD algorithm here, independently, rather than
    /// pinning a magic constant — a hardcoded expected hash would have to be
    /// captured from the new code, which proves nothing about the old.
    #[test]
    fn hashes_are_stable_for_promptless_corpora() {
        let m = manifest();
        assert!(
            m.clips.iter().all(|c| c.prompt.is_none()),
            "fixture must be promptless for this test to mean anything"
        );

        let mut hasher = Sha256::new();
        hasher.update(m.name.as_bytes());
        hasher.update(m.version.to_le_bytes());
        let mut clips: Vec<&ClipEntry> = m.clips.iter().collect();
        clips.sort_by(|a, b| a.id.cmp(&b.id));
        for c in clips {
            hasher.update(c.id.as_bytes());
            hasher.update(c.sha256.as_bytes());
            hasher.update(match c.split {
                Split::Train => b"t",
                Split::Holdout => b"h",
            });
        }
        assert_eq!(hex(&hasher.finalize()), m.manifest_hash());
    }

    /// **The regression test that actually matters**: a REAL shipped corpus
    /// must still hash to the value already recorded in `bench/ledger.jsonl`.
    ///
    /// Two fields were folded into `manifest_hash` for VLM support — the clip
    /// prompt and the `[scorer]` block. Both are hashed only when present, so
    /// every pre-VLM corpus is meant to be untouched. "Meant to be" is not
    /// evidence: if this drifts, every historical ledger line silently stops
    /// matching its own data, and the ledger's whole contract — that a line is
    /// sufficient on its own to reproduce a run — is void.
    ///
    /// The expected value is copied from a shipped ledger line, not from this
    /// code, which is what makes it a check rather than a tautology.
    #[test]
    fn a_shipped_corpus_still_hashes_to_its_ledger_value() {
        // Repo-relative from the crate dir.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../corpora/librispeech-test-clean-v2.toml");
        let Ok(m) = Manifest::load(&path) else {
            // A checkout without the corpus file is not a failure of this code.
            return;
        };
        assert_eq!(
            m.manifest_hash(),
            "bf32e97a1a4a29e6f1e4eb01cfb48e49c26f0f4155f137969940ad6153ce28a3",
            "librispeech-test-clean-v2's fingerprint moved — every ledger line \
             referencing it now points at data that no longer matches"
        );
    }

    /// Changing the scorer's scale changes every normalised score, so it must
    /// change the fingerprint too. This exact case shipped a 20x difference
    /// under an identical hash before the scorer was folded in.
    #[test]
    fn changing_the_scorer_moves_the_fingerprint() {
        let with = |scale: f64| -> Manifest {
            toml::from_str::<Manifest>(&format!(
                r#"
                name = "c"
                version = 1
                task = "vlm"

                [scorer]
                name = "s"
                command = ["python", "score.py", "{{predictions}}"]
                metric = "OCRBench"
                scale = {scale}
                "#
            ))
            .expect("parses")
        };
        assert_ne!(with(1000.0).manifest_hash(), with(50.0).manifest_hash());

        // …and so does changing the command, which is the other half of "what
        // produced this number".
        let other: Manifest = toml::from_str(
            r#"
            name = "c"
            version = 1
            task = "vlm"

            [scorer]
            name = "s"
            command = ["python", "different.py", "{predictions}"]
            metric = "OCRBench"
            scale = 1000.0
            "#,
        )
        .expect("parses");
        assert_ne!(with(1000.0).manifest_hash(), other.manifest_hash());
    }

    /// Two prompts that concatenate to the same bytes must not collide.
    #[test]
    fn adjacent_prompts_cannot_be_confused() {
        let mut a = manifest();
        a.clips[0].prompt = Some("ab".into());
        a.clips[1].prompt = Some("c".into());
        let mut b = manifest();
        b.clips[0].prompt = Some("a".into());
        b.clips[1].prompt = Some("bc".into());
        assert_ne!(a.manifest_hash(), b.manifest_hash());
    }
}
