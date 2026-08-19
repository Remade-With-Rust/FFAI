//! Mercury's model manifests, compiled into the crate.
//!
//! **Why these are embedded rather than read from `models/`.** A library must
//! not depend on its consumer's working directory. Every Mercury entry point
//! used to resolve manifests from a relative `"models"` path, which works in
//! this repo and fails everywhere else: `cargo add ffai-mercury` +
//! `WhisperCandle::new()` had nothing to read. `include_str!` fixes that at
//! the only cost that matters — the files must live inside the crate, since
//! anything outside the crate directory is absent from the published package.
//!
//! **Two copies exist on purpose.** `models/*.toml` at the repo root is what
//! `ffai models` lists and what the CLI and tools read; these are the
//! library's own. [`tests::embedded_manifests_match_the_repo_copies`] fails
//! if they drift, and skips when the repo files are absent — which is the
//! packaged-crate case, where there is nothing to drift from.
//!
//! Manifests describe *where weights come from and under what licence*; they
//! carry no weights themselves, so embedding them costs a few kilobytes and
//! changes nothing about principle 4.

use std::path::Path;

use ffai_core::error::{Error, Result};
use ffai_models::ModelManifest;

/// `(name, toml)` for every manifest Mercury can load on its own.
pub const EMBEDDED: &[(&str, &str)] = &[
    // ---- ASR: the Whisper sizes ----
    (
        "whisper-tiny-en",
        include_str!("../manifests/whisper-tiny-en.toml"),
    ),
    (
        "whisper-base-en",
        include_str!("../manifests/whisper-base-en.toml"),
    ),
    (
        "whisper-small-en",
        include_str!("../manifests/whisper-small-en.toml"),
    ),
    (
        "whisper-medium-en",
        include_str!("../manifests/whisper-medium-en.toml"),
    ),
    (
        "whisper-tiny",
        include_str!("../manifests/whisper-tiny.toml"),
    ),
    (
        "whisper-large-v3",
        include_str!("../manifests/whisper-large-v3.toml"),
    ),
    // ---- ASR: the WhisperX layer, fetched only when its flag is set ----
    (
        "wav2vec2-base-960h",
        include_str!("../manifests/wav2vec2-base-960h.toml"),
    ),
    (
        "ecapa-tdnn-voxceleb",
        include_str!("../manifests/ecapa-tdnn-voxceleb.toml"),
    ),
    // ---- TTS ----
    ("cmudict", include_str!("../manifests/cmudict.toml")),
    (
        "piper-vits-lessac-medium",
        include_str!("../manifests/piper-vits-lessac-medium.toml"),
    ),
];

/// The embedded TOML for `name`, if Mercury ships one.
#[must_use]
pub fn embedded(name: &str) -> Option<&'static str> {
    EMBEDDED
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, toml)| *toml)
}

/// Resolve a manifest by name.
///
/// `dir` is the caller's override: when given, the manifest is read from
/// `<dir>/<name>.toml` — falling back to a directory scan so a manifest whose
/// filename differs from its `name` field is still found, which is how
/// `ffai_models::load_dir` has always behaved. When `dir` is `None`, the
/// compiled-in copy is used and nothing touches the filesystem.
pub fn resolve(dir: Option<&Path>, name: &str) -> Result<ModelManifest> {
    if let Some(dir) = dir {
        let direct = dir.join(format!("{name}.toml"));
        if direct.exists() {
            return ModelManifest::load(&direct);
        }
        return ffai_models::load_dir(dir)?
            .into_iter()
            .find(|m| m.name == name)
            .ok_or_else(|| {
                Error::Model(format!(
                    "no model manifest named `{name}` in {}",
                    dir.display()
                ))
            });
    }
    let toml = embedded(name).ok_or_else(|| {
        Error::Model(format!(
            "no manifest named `{name}` is compiled into ffai-mercury (have: {}) — \
             pass a manifest directory to use your own",
            EMBEDDED
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;
    ModelManifest::from_toml(toml)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_manifest_parses_and_names_itself() {
        for (name, toml) in EMBEDDED {
            let m = ModelManifest::from_toml(toml)
                .unwrap_or_else(|e| panic!("{name} does not parse: {e}"));
            // The lookup key must equal the manifest's own name, or `resolve`
            // would hand back a manifest for a different model.
            assert_eq!(&m.name, name, "{name}.toml calls itself `{}`", m.name);
            // A manifest exists to surface a licence (principle 4); an empty
            // one would ship weights with their terms hidden.
            assert!(!m.license.is_empty(), "{name} declares no licence");
            assert!(!m.files.is_empty(), "{name} lists no files");
        }
    }

    #[test]
    fn embedded_manifests_match_the_repo_copies() {
        // Skipped when the repo files are absent — the packaged-crate case.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        for (name, embedded) in EMBEDDED {
            let Ok(text) = std::fs::read_to_string(root.join(format!("{name}.toml"))) else {
                continue;
            };
            assert_eq!(
                text.replace("\r\n", "\n"),
                embedded.replace("\r\n", "\n"),
                "crates/ffai-mercury/manifests/{name}.toml has drifted from \
                 models/{name}.toml — copy the repo version over the crate's"
            );
        }
    }

    #[test]
    fn resolve_reads_embedded_by_default_and_the_directory_when_given_one() {
        // No filesystem: works from any working directory, which is the whole
        // point of the module.
        let m = resolve(None, "whisper-tiny-en").expect("embedded");
        assert_eq!(m.name, "whisper-tiny-en");

        // An unknown name names what IS available rather than failing blankly.
        let err = resolve(None, "whisper-enormous").unwrap_err().to_string();
        assert!(err.contains("whisper-tiny-en"), "unhelpful error: {err}");

        // A directory override still works, and a missing one still errors.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        if repo.exists() {
            assert_eq!(
                resolve(Some(&repo), "whisper-base-en").unwrap().name,
                "whisper-base-en"
            );
        }
        assert!(resolve(Some(Path::new("no/such/dir")), "whisper-tiny-en").is_err());
    }
}
