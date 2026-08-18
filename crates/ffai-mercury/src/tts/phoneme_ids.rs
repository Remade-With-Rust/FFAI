//! IPA phoneme string → the voice's model-input id sequence.
//!
//! A piper voice's `.onnx.json` carries `phoneme_id_map`: one espeak IPA
//! codepoint-cluster → id. The model was trained on sequences interleaved
//! with pad (id 0) and framed by BOS `^` / EOS `$`:
//!
//! ```text
//! ^ _ p1 _ p2 _ ... pN _ $      (ids: 1, 0, i1, 0, i2, 0, ..., iN, 0, 2)
//! ```
//!
//! — verified character-for-character against piper's own `phonemes_to_ids`
//! on the pinned fixtures (the `ids` tensor in tests/fixtures/vits).

use std::collections::HashMap;
use std::path::Path;

use ffai_core::error::{Error, Result};

pub struct PhonemeIdMap {
    map: HashMap<String, Vec<i64>>,
}

impl PhonemeIdMap {
    /// Parse from a voice config JSON (the `.onnx.json` shipped beside every
    /// piper voice, staged as `voice-config.json` in the model cache).
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Model(format!("voice config parse: {e}")))?;
        let obj = v
            .get("phoneme_id_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| Error::Model("voice config has no phoneme_id_map".into()))?;
        let mut map = HashMap::new();
        for (k, ids) in obj {
            let ids: Vec<i64> = ids
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_i64)
                .collect();
            map.insert(k.clone(), ids);
        }
        Ok(Self { map })
    }

    /// One sentence of IPA (one codepoint cluster per char, as
    /// [`crate::tts::phonemize`] emits) → interleaved id sequence.
    /// Unknown codepoints are skipped, counted, and reported — piper does the
    /// same (logs and drops); a hard error mid-sentence would turn one exotic
    /// character into no audio at all.
    #[must_use]
    pub fn sentence_to_ids(&self, ipa: &str) -> (Vec<i64>, usize) {
        let pad = self.id_of("_").unwrap_or(0);
        let bos = self.id_of("^").unwrap_or(1);
        let eos = self.id_of("$").unwrap_or(2);
        let mut ids = vec![bos];
        let mut unknown = 0usize;
        for c in ipa.chars() {
            match self.map.get(c.to_string().as_str()) {
                Some(v) => {
                    ids.push(pad);
                    ids.extend(v);
                }
                None => unknown += 1,
            }
        }
        ids.push(pad);
        ids.push(eos);
        (ids, unknown)
    }

    fn id_of(&self, key: &str) -> Option<i64> {
        self.map.get(key).and_then(|v| v.first().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaves_pad_and_frames_with_bos_eos() {
        // The exact framing piper's phonemes_to_ids produces (verified
        // against the pinned fixture ids for hvd-01-01).
        let map = PhonemeIdMap::from_json(
            r#"{"phoneme_id_map": {"^": [1], "$": [2], "_": [0], "ð": [41], "ə": [59], " ": [3]}}"#,
        )
        .unwrap();
        let (ids, unknown) = map.sentence_to_ids("ðə ə");
        assert_eq!(ids, vec![1, 0, 41, 0, 59, 0, 3, 0, 59, 0, 2]);
        assert_eq!(unknown, 0);

        let (_, unknown) = map.sentence_to_ids("ðəQ");
        assert_eq!(unknown, 1, "unknown codepoints are counted, not fatal");
    }
}
