//! The OCR model manifests, compiled into the crate.
//!
//! Every other constructor reads manifests from a `models/` directory on disk,
//! and that directory exists only in the `FFai` repository — not in the crate
//! crates.io ships. A user who added `ffai-carmenta` as a dependency got
//! `i/o error: The system cannot find the path specified` until they copied the
//! TOMLs in by hand (docs/plans/commercial-gaps.md, gap 10).
//!
//! The manifests are small, fixed per release, and carry the one thing a
//! caller must not get wrong — the SHA-256 of each weight file — so they belong
//! in the binary. Weights do not: they are licensed data, fetched or placed by
//! the caller, and never vendored.
//!
//! The copies under `crates/ffai-carmenta/models/` must stay byte-identical to
//! the repository's `models/`; `embedded_manifests_match_the_repository` fails
//! the moment they drift.

use ffai_core::error::{Error, Result};
use ffai_models::ModelManifest;

/// `(model name, manifest TOML)` for every model a Carmenta engine can load.
pub const EMBEDDED: &[(&str, &str)] = &[
    ("craft-mlt", include_str!("../models/craft-mlt.toml")),
    (
        "crnn-english-g2",
        include_str!("../models/crnn-english-g2.toml"),
    ),
    (
        "crnn-zh-sim-g2",
        include_str!("../models/crnn-zh-sim-g2.toml"),
    ),
    ("parseq-tiny", include_str!("../models/parseq-tiny.toml")),
    (
        "ppocrv5-mobile-det",
        include_str!("../models/ppocrv5-mobile-det.toml"),
    ),
    (
        "ppocrv5-mobile-rec",
        include_str!("../models/ppocrv5-mobile-rec.toml"),
    ),
];

/// Every embedded manifest, parsed.
pub fn embedded() -> Result<Vec<ModelManifest>> {
    EMBEDDED
        .iter()
        .map(|(name, toml)| {
            ModelManifest::from_toml(toml)
                .map_err(|e| Error::Model(format!("embedded manifest `{name}`: {e}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each entry's key is the manifest's own `name`, so a lookup by key can
    /// never return a different model's checksums.
    #[test]
    fn embedded_manifests_parse_and_are_keyed_by_their_own_name() {
        let parsed = embedded().unwrap();
        assert_eq!(parsed.len(), EMBEDDED.len());
        for ((key, _), m) in EMBEDDED.iter().zip(&parsed) {
            assert_eq!(*key, m.name);
            assert_eq!(m.task, "ocr");
            assert!(!m.files.is_empty(), "{key}: no files");
            assert!(
                m.files.iter().all(|f| f.sha256.is_some()),
                "{key}: every embedded file must carry a checksum"
            );
        }
    }

    /// The crate's copies are the repository's manifests, byte for byte. Run
    /// only where the repository exists — a crate unpacked from crates.io has
    /// no `../../models` and has nothing to drift from.
    #[test]
    fn embedded_manifests_match_the_repository() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        if !repo.is_dir() {
            eprintln!("SKIP embedded manifest sync: no repository models/ directory");
            return;
        }
        for (name, toml) in EMBEDDED {
            let on_disk = std::fs::read_to_string(repo.join(format!("{name}.toml")))
                .unwrap_or_else(|e| panic!("repository models/{name}.toml: {e}"));
            assert!(
                on_disk == *toml,
                "crates/ffai-carmenta/models/{name}.toml has drifted from models/{name}.toml — \
                 copy the repository file over the crate's"
            );
        }
    }
}
