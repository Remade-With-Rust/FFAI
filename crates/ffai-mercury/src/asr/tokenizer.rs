//! Whisper's tokenizer and special-token grammar.
//!
//! Whisper's decoder emits two interleaved languages: ordinary BPE text
//! tokens, and *control* tokens that steer the task (`<|transcribe|>`), mark
//! structure (`<|notimestamps|>`), and encode time (`<|0.42|>`). Getting the
//! control vocabulary wrong produces output that looks like text but has
//! timestamps spliced into it — so the grammar lives here, in one place,
//! rather than being re-derived at each call site.

use ffai_core::error::{Error, Result};
use std::path::Path;

/// Timestamp tokens quantize time to 20 ms steps, up to 30 s.
pub const TIMESTAMP_STEP_SECS: f64 = 0.02;

/// The tokenizer plus the resolved ids of every control token we rely on.
pub struct WhisperTokenizer {
    inner: tokenizers::Tokenizer,
    pub sot: u32,
    pub eot: u32,
    pub transcribe: u32,
    pub translate: u32,
    pub no_timestamps: u32,
    /// Probability mass on this token is Whisper's silence detector; the
    /// temperature-fallback logic in M2 uses it to reject hallucinated
    /// segments. Resolved now so a model missing it fails at load, not
    /// mid-decode.
    pub no_speech: u32,
    /// Prefix for `condition_on_previous_text` (long-audio context carry-over,
    /// M2). Same reasoning: fail at load.
    pub prev: u32,
    /// First timestamp token (`<|0.00|>`); timestamps run contiguously from here.
    pub timestamp_begin: u32,
    /// The lone-space token, suppressed at the first generated position so a
    /// transcript can't open with blank output.
    pub space: Option<u32>,
}

impl WhisperTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes, &path.display().to_string())
    }

    /// Build from the BYTES of `tokenizer.json` — the constructor a target
    /// with no filesystem needs.
    ///
    /// `load` is written in terms of this rather than beside it, so the two
    /// cannot drift: a browser and a server parse the same bytes through the
    /// same code. `whence` names the source in error messages only.
    pub fn from_bytes(bytes: &[u8], whence: &str) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| Error::Model(format!("loading tokenizer {whence}: {e}")))?;
        let id = |tok: &str| -> Result<u32> {
            inner.token_to_id(tok).ok_or_else(|| {
                Error::Model(format!(
                    "tokenizer {whence} has no `{tok}` token — is this a Whisper tokenizer?"
                ))
            })
        };
        Ok(Self {
            sot: id("<|startoftranscript|>")?,
            eot: id("<|endoftext|>")?,
            transcribe: id("<|transcribe|>")?,
            translate: id("<|translate|>")?,
            no_timestamps: id("<|notimestamps|>")?,
            no_speech: id("<|nospeech|>").or_else(|_| id("<|nocaptions|>"))?,
            prev: id("<|startofprev|>")?,
            timestamp_begin: id("<|0.00|>")?,
            space: inner
                .token_to_id("\u{0120}")
                .or_else(|| inner.token_to_id(" ")),
            inner,
        })
    }

    /// True if `token` encodes a timestamp rather than text.
    pub const fn is_timestamp(&self, token: u32) -> bool {
        token >= self.timestamp_begin
    }

    /// Seconds encoded by a timestamp token.
    pub fn timestamp_secs(&self, token: u32) -> f64 {
        f64::from(token.saturating_sub(self.timestamp_begin)) * TIMESTAMP_STEP_SECS
    }

    /// Whisper's `non_speech_tokens`: punctuation and symbol tokens that
    /// annotate audio rather than transcribe it (`♪`, `((`, `--`, brackets).
    ///
    /// openai-whisper and whisper.cpp both suppress these by default. We did
    /// not, and it showed up as a *uniform* accuracy deficit — a spurious
    /// symbol here and there across many clips rather than any single
    /// catastrophic failure, which is exactly the signature the per-clip
    /// analysis found (no clip over 50 % WER, worst five only 11.9 % of all
    /// errors).
    ///
    /// Ported from `whisper/tokenizer.py::non_speech_tokens`: a symbol
    /// qualifies when it encodes to a *single* token, alone or space-prefixed.
    pub fn non_speech_tokens(&self) -> Vec<u32> {
        const SYMBOLS: &str = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』";
        const MULTI: &[&str] = &[
            "<<",
            ">>",
            "<<<",
            ">>>",
            "--",
            "---",
            "-(",
            "-[",
            "('",
            "(\"",
            "((",
            "))",
            "(((",
            ")))",
            "[[",
            "]]",
            "{{",
            "}}",
            "♪♪",
            "♪♪♪",
        ];
        const MISC: &str = "♩♪♫♬♭♮♯";

        let mut out = std::collections::BTreeSet::new();
        // Whisper seeds the set with " -" and " '" unconditionally.
        for seed in [" -", " '"] {
            if let Ok(ids) = self.encode(seed)
                && let Some(&first) = ids.first()
            {
                out.insert(first);
            }
        }
        let singles: Vec<String> = SYMBOLS.chars().map(|c| c.to_string()).collect();
        let misc: Vec<String> = MISC.chars().map(|c| c.to_string()).collect();
        for symbol in singles
            .iter()
            .map(String::as_str)
            .chain(MULTI.iter().copied())
            .chain(misc.iter().map(String::as_str))
        {
            let is_misc = MISC.contains(symbol);
            for form in [symbol.to_string(), format!(" {symbol}")] {
                if let Ok(ids) = self.encode(&form)
                    && (ids.len() == 1 || is_misc)
                    && !ids.is_empty()
                {
                    out.insert(ids[0]);
                }
            }
        }
        out.into_iter().collect()
    }

    /// The id of a language token like `<|en|>`.
    pub fn language(&self, code: &str) -> Option<u32> {
        self.inner.token_to_id(&format!("<|{code}|>"))
    }

    /// Decode token ids to text, dropping control tokens.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let text: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|&t| t < self.eot_floor())
            .collect();
        self.inner
            .decode(&text, true)
            .map_err(|e| Error::Model(format!("decoding tokens: {e}")))
    }

    /// Encode text to token ids (used for prompting and for tests).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Model(format!("encoding text: {e}")))?
            .get_ids()
            .to_vec())
    }

    /// Everything at or above this id is a control token, not text.
    const fn eot_floor(&self) -> u32 {
        self.eot
    }

    /// The decoder's initial prompt: the fixed prefix every Whisper decode
    /// starts from.
    ///
    /// **English-only models take neither a language nor a task token.**
    /// `get_tokenizer(multilingual=False, ...)` sets both to `None`, so their
    /// prompt is bare `<|startoftranscript|>`. Prepending `<|transcribe|>`
    /// anyway feeds an `.en` model a control token it was never trained to
    /// see after SOT: tiny.en degrades to a spurious leading "." on every
    /// clip, and base.en collapses to empty output on some.
    pub fn initial_tokens(
        &self,
        language: Option<u32>,
        translate: bool,
        timestamps: bool,
        english_only: bool,
    ) -> Vec<u32> {
        let mut tokens = vec![self.sot];
        if !english_only {
            if let Some(lang) = language {
                tokens.push(lang);
            }
            tokens.push(if translate {
                self.translate
            } else {
                self.transcribe
            });
        }
        if !timestamps {
            tokens.push(self.no_timestamps);
        }
        tokens
    }
}
