//! CMUdict: word → ARPABET pronunciations.
//!
//! The permissively-licensed (BSD-2-Clause) lexicon backbone of Mercury's
//! G2P. The dictionary is DATA, not code — fetched into the model cache and
//! declared by `models/cmudict.toml` with its license and sha256, never
//! vendored (principle 4). ~134k entries parse in well under 100 ms and cost
//! ~15 MiB resident, which is noise next to any acoustic model.

use std::collections::HashMap;
use std::path::Path;

use ffai_core::error::{Error, Result};

/// One ARPABET phone: symbol without the stress digit, plus the stress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phone {
    /// Symbol index into [`SYMBOLS`] — stored as an index so a `Phone` is
    /// `Copy` and comparisons are integer comparisons.
    pub sym: u8,
    /// 0 = unstressed, 1 = primary, 2 = secondary; consonants carry 0.
    pub stress: u8,
}

/// The 39 ARPABET symbols, alphabetical. Index = `Phone::sym`.
pub const SYMBOLS: &[&str] = &[
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
    "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH",
    "UW", "V", "W", "Y", "Z", "ZH",
];

impl Phone {
    pub fn symbol(&self) -> &'static str {
        SYMBOLS[self.sym as usize]
    }

    /// ARPABET vowels are exactly the symbols that carry stress digits.
    pub fn is_vowel(&self) -> bool {
        matches!(
            self.symbol(),
            "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY" | "OW"
                | "OY" | "UH" | "UW"
        )
    }

    fn parse(token: &str) -> Option<Phone> {
        let (sym, stress) = match token.as_bytes().last() {
            Some(d @ b'0'..=b'2') => (&token[..token.len() - 1], d - b'0'),
            _ => (token, 0),
        };
        SYMBOLS.binary_search(&sym).ok().map(|i| Phone { sym: i as u8, stress })
    }
}

/// The dictionary: lowercase word → first-listed pronunciation.
///
/// CMUdict orders variants by frequency of use; espeak likewise picks one
/// form per word. Only the first is kept — variant selection by context is
/// out of scope until a corpus shows it costing intelligibility.
pub struct Lexicon {
    entries: HashMap<String, Vec<Phone>>,
}

impl Lexicon {
    /// Load from the `cmudict` manifest in a manifest directory (`models/`).
    pub fn from_manifest_dir(dir: &Path) -> Result<Self> {
        let manifests = ffai_models::load_dir(dir)?;
        let manifest = manifests.into_iter().find(|m| m.name == "cmudict").ok_or_else(|| {
            Error::Model(format!("no `cmudict` manifest in {} — see models/cmudict.toml", dir.display()))
        })?;
        let resolved = manifest.fetch()?;
        Self::load(resolved.file("cmudict.dict")?)
    }

    /// Parse a cmudict.dict file: `word PH0 PH1 ...` per line, variants as
    /// `word(2)`, comments after `#`.
    pub fn load(path: &Path) -> Result<Self> {
        // Latin-1: a handful of entries (déjà, café...) are not UTF-8.
        let bytes = std::fs::read(path)?;
        let text: String = bytes.iter().map(|&b| b as char).collect();
        let mut entries = HashMap::new();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let Some((word, phones)) = line.split_once(' ') else { continue };
            // `word(2)`+ are alternates; first listed wins (see type docs).
            if word.ends_with(')') {
                continue;
            }
            let parsed: Option<Vec<Phone>> =
                phones.split_whitespace().map(Phone::parse).collect();
            if let Some(p) = parsed {
                if !p.is_empty() {
                    entries.entry(word.to_ascii_lowercase()).or_insert(p);
                }
            }
        }
        if entries.is_empty() {
            return Err(Error::Model(format!(
                "{} parsed to an empty lexicon — wrong file?",
                path.display()
            )));
        }
        Ok(Lexicon { entries })
    }

    pub fn lookup(&self, word: &str) -> Option<&[Phone]> {
        self.entries.get(word).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp(content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join("ffai_lexicon_test.dict");
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn parses_words_stress_and_skips_variants() {
        let path = write_temp(
            "canoe K AH0 N UW1\nbirch B ER1 CH\nthe DH AH0\nthe(2) DH IY0\n# comment\n",
        );
        let lex = Lexicon::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // Content assertions, not shape: the exact phones and stresses.
        let canoe = lex.lookup("canoe").unwrap();
        assert_eq!(
            canoe.iter().map(|p| (p.symbol(), p.stress)).collect::<Vec<_>>(),
            vec![("K", 0), ("AH", 0), ("N", 0), ("UW", 1)]
        );
        assert!(canoe[1].is_vowel() && !canoe[0].is_vowel());
        // First variant wins; `the(2)` must not override.
        assert_eq!(lex.lookup("the").unwrap()[1].symbol(), "AH");
        assert_eq!(lex.lookup("missing"), None);
    }
}
