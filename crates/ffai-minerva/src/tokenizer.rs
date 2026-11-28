//! Text in and text out: the model file's own byte-level BPE tokenizer
//! (`tokenizer.ggml.model = gpt2`, the Llama 3 pre-tokenizer), its detokenizer, a streaming
//! detokenizer that holds back incomplete UTF-8, and the file's chat template.
//!
//! Encoding, as Hugging Face `tokenizers` and llama.cpp do it for this vocabulary:
//!
//! 1. Special-token strings in the text (`<|eot_id|>`, ...) become their ids; the text between
//!    them is encoded normally.
//! 2. The Llama 3 split ([`split_llama3`]) cuts each piece into words.
//! 3. Each word's bytes map to GPT-2's printable byte alphabet. A word that is itself in the
//!    vocabulary is one token (Llama 3 sets `ignore_merges`); otherwise the lowest-ranked
//!    adjacent pair merges first, leftmost on a tie, until no pair has a rank.
//!
//! Gated on reference data from two independent oracles (Hugging Face and bitnet.cpp's server).

use std::collections::HashMap;

use rusty_ternary::gguf::{Gguf, Value};
use unicode_general_category::{GeneralCategory, get_general_category};

/// What can go wrong reading a tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerError(pub String);

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TokenizerError {}

fn err(s: impl Into<String>) -> TokenizerError {
    TokenizerError(s.into())
}

/// `tokenizer.ggml.token_type` values that are special (matched whole in text, never merged).
const TYPE_CONTROL: u64 = 3;
const TYPE_USER_DEFINED: u64 = 4;

/// A model's tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Each id's bytes when decoded (a special's own text).
    bytes: Vec<Vec<u8>>,
    /// Token text (GPT-2 byte alphabet) to id.
    vocab: HashMap<String, u32>,
    /// A mergeable pair of ids to (rank, merged id).
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Special-token text to id.
    specials: HashMap<String, u32>,
    /// Longest special-token text, in chars.
    special_max: usize,
    /// The chars a special-token text starts with.
    special_first: std::collections::HashSet<char>,
    /// Byte to its GPT-2 alphabet char.
    byte_char: [char; 256],
    /// Beginning of text, if the file names one.
    pub bos: Option<u32>,
    /// End of a turn / of text (`tokenizer.ggml.eos_token_id`).
    pub eos: Option<u32>,
    /// Whether to prepend [`Self::bos`] when encoding a prompt.
    pub add_bos: bool,
    /// The file's `tokenizer.chat_template`, if any.
    pub chat_template: Option<String>,
}

/// GPT-2's map from bytes to printable chars: printable Latin-1 bytes map to themselves, the
/// rest to U+0100 onwards in byte order.
fn byte_alphabet() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut next = 256u32;
    for b in 0u32..256 {
        let printable =
            (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if printable {
            b
        } else {
            next += 1;
            next - 1
        };
        out[b as usize] = char::from_u32(c).unwrap_or('\0');
    }
    out
}

fn strings<'a>(g: &Gguf<'a>, key: &str) -> Result<Vec<&'a str>, TokenizerError> {
    g.strings(key)
        .map_err(|_| err(format!("no {key} (an array of text)")))
}

impl Tokenizer {
    /// The tokenizer in a model file's metadata.
    ///
    /// # Errors
    /// When the file is not a `GGUF`, or holds no `gpt2` (byte-level BPE) tokenizer.
    pub fn from_gguf(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let g = Gguf::parse_lazy(bytes).map_err(|e| err(format!("not a readable GGUF: {e:?}")))?;
        let model = g
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .unwrap_or("");
        if model != "gpt2" {
            return Err(err(format!(
                "tokenizer `{model}`: only byte-level BPE (gpt2) is supported"
            )));
        }
        let pre = g
            .get("tokenizer.ggml.pre")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !matches!(pre, "llama-bpe" | "llama3" | "") {
            return Err(err(format!(
                "pre-tokenizer `{pre}`: only llama-bpe is supported"
            )));
        }
        let tokens = strings(&g, "tokenizer.ggml.tokens")?;
        let merges_text = strings(&g, "tokenizer.ggml.merges")?;
        let types: Vec<u64> = g.array("tokenizer.ggml.token_type").map_or_else(
            |_| vec![1; tokens.len()],
            |a| a.iter().map(|v| v.as_u64().unwrap_or(1)).collect(),
        );
        let byte_char = byte_alphabet();
        let char_byte: HashMap<char, u8> = byte_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, u8::try_from(b).unwrap_or(0)))
            .collect();
        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut specials = HashMap::new();
        let mut bytes = Vec::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            let id = u32::try_from(i).map_err(|_| err("vocabulary too large"))?;
            let ty = types.get(i).copied().unwrap_or(1);
            if ty == TYPE_CONTROL || ty == TYPE_USER_DEFINED {
                specials.insert((*t).to_owned(), id);
                bytes.push(t.as_bytes().to_vec());
            } else {
                // A normal token is GPT-2 alphabet text; a char outside it is kept as UTF-8.
                let mut b = Vec::with_capacity(t.len());
                for c in t.chars() {
                    match char_byte.get(&c) {
                        Some(&x) => b.push(x),
                        None => b.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
                    }
                }
                bytes.push(b);
            }
            vocab.insert((*t).to_owned(), id);
        }
        let mut merges = HashMap::with_capacity(merges_text.len());
        for (rank, m) in merges_text.iter().enumerate() {
            let Some((l, r)) = m.split_once(' ') else {
                continue;
            };
            let (Some(&a), Some(&b), Some(&ab)) =
                (vocab.get(l), vocab.get(r), vocab.get(&format!("{l}{r}")))
            else {
                continue;
            };
            let rank = u32::try_from(rank).unwrap_or(u32::MAX);
            merges.entry((a, b)).or_insert((rank, ab));
        }
        let id = |k: &str| {
            g.get(k)
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
        };
        let special_max = specials
            .keys()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(0);
        let special_first = specials.keys().filter_map(|s| s.chars().next()).collect();
        Ok(Self {
            bytes,
            vocab,
            merges,
            specials,
            special_max,
            special_first,
            byte_char,
            bos: id("tokenizer.ggml.bos_token_id"),
            eos: id("tokenizer.ggml.eos_token_id"),
            add_bos: !matches!(
                g.get("tokenizer.ggml.add_bos_token"),
                Some(Value::Bool(false))
            ),
            chat_template: g
                .get("tokenizer.chat_template")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }

    /// Vocabulary size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the vocabulary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The id of a special token's text (`<|eot_id|>`), if it is one.
    #[must_use]
    pub fn special(&self, text: &str) -> Option<u32> {
        self.specials.get(text).copied()
    }

    /// Encode `text`, special-token strings becoming their ids; with `bos`, [`Self::bos`] first.
    #[must_use]
    pub fn encode(&self, text: &str, bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if bos && let Some(b) = self.bos {
            out.push(b);
        }
        let mut rest = text;
        while !rest.is_empty() {
            match self.next_special(rest) {
                Some((at, len, id)) => {
                    self.encode_plain(&rest[..at], &mut out);
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    /// The first special-token string in `s`: its byte offset, byte length and id (the longest
    /// at the first position that has one).
    fn next_special(&self, s: &str) -> Option<(usize, usize, u32)> {
        if self.specials.is_empty() {
            return None;
        }
        for (at, c) in s.char_indices() {
            if !self.special_first.contains(&c) {
                continue;
            }
            let tail = &s[at..];
            // Char boundaries after 1..=special_max chars, longest first.
            let mut ends: Vec<usize> = tail
                .char_indices()
                .map(|(i, ch)| i + ch.len_utf8())
                .take(self.special_max)
                .collect();
            ends.reverse();
            if let Some((e, id)) = ends
                .into_iter()
                .find_map(|e| self.specials.get(&tail[..e]).map(|&id| (e, id)))
            {
                return Some((at, e, id));
            }
        }
        None
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for word in split_llama3(text) {
            self.encode_word(word, out);
        }
    }

    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        let mapped: String = word.bytes().map(|b| self.byte_char[b as usize]).collect();
        if let Some(&id) = self.vocab.get(&mapped) {
            out.push(id);
            return;
        }
        let mut syms: Vec<u32> = mapped
            .chars()
            .filter_map(|c| self.vocab.get(c.encode_utf8(&mut [0; 4]) as &str).copied())
            .collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(syms[i], syms[i + 1]))
                    && best.is_none_or(|b| rank < b.0)
                {
                    best = Some((rank, i, merged));
                }
            }
            let Some((_, i, merged)) = best else { break };
            syms[i] = merged;
            syms.remove(i + 1);
        }
        out.extend(syms);
    }

    /// The bytes `id` decodes to (a special's own text); empty for an unknown id.
    #[must_use]
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// Decode `ids` to text (invalid UTF-8, as from a cut multi-byte char, becomes U+FFFD).
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let b: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.token_bytes(i).iter().copied())
            .collect();
        String::from_utf8_lossy(&b).into_owned()
    }

    /// Render a conversation with the file's chat template. `messages` are (role, content).
    ///
    /// # Errors
    /// When the file has no template, or one this renderer does not know.
    pub fn apply_chat_template(
        &self,
        messages: &[(String, String)],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let t = self
            .chat_template
            .as_deref()
            .ok_or_else(|| err("the model has no chat template"))?;
        if t != BITNET_TEMPLATE {
            return Err(err(
                "chat template not supported by this renderer (only BitNet b1.58 2B-4T's)",
            ));
        }
        let mut s = String::new();
        for (role, content) in messages {
            s.push_str(&capitalize(role));
            s.push_str(": ");
            s.push_str(content.trim());
            s.push_str("<|eot_id|>");
        }
        if add_generation_prompt {
            s.push_str("Assistant: ");
        }
        Ok(s)
    }
}

/// BitNet b1.58 2B-4T's `tokenizer.chat_template`, rendered by [`Tokenizer::apply_chat_template`].
const BITNET_TEMPLATE: &str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['role'] | capitalize + ': '+ message['content'] | trim + '<|eot_id|>' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}";

/// Jinja's `capitalize`: the first char upper case, the rest lower case.
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or_else(String::new, |f| {
        f.to_uppercase()
            .chain(c.flat_map(char::to_lowercase))
            .collect()
    })
}

/// Decodes a token stream to text as it arrives, holding back bytes until they complete a
/// UTF-8 char (a char can span tokens), so every returned piece is valid text.
#[derive(Debug, Default, Clone)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    /// A decoder with nothing held.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `id`'s bytes; return the text that is now complete.
    pub fn push(&mut self, t: &Tokenizer, id: u32) -> String {
        self.pending.extend_from_slice(t.token_bytes(id));
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            // Bytes that can never become valid: emit them (as U+FFFD) rather than hold forever.
            Err(_) => self.pending.len(),
        };
        let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        out
    }

    /// Whatever is still held (an incomplete char at the end), as text.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

fn is_letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

fn is_number(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// The Llama 3 pre-tokenizer, by hand (the regex
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
/// whose last alternatives need a look-ahead), cutting `text` into the words BPE encodes.
#[must_use]
pub fn split_llama3(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| cs.get(i).map(|x| x.1);
    let byte = |i: usize| cs.get(i).map_or(text.len(), |x| x.0);
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        let end = 'm: {
            // Contractions, ASCII case-insensitive.
            if c == '\''
                && let Some(c1) = at(i + 1)
            {
                let l1 = c1.to_ascii_lowercase();
                if let Some(c2) = at(i + 2) {
                    let l2 = c2.to_ascii_lowercase();
                    if matches!((l1, l2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                        break 'm i + 3;
                    }
                }
                if matches!(l1, 's' | 't' | 'm' | 'd') {
                    break 'm i + 2;
                }
            }
            // [^\r\n\p{L}\p{N}]?\p{L}+
            let lead = !is_newline(c) && !is_letter(c) && !is_number(c);
            if is_letter(c) || (lead && at(i + 1).is_some_and(is_letter)) {
                let mut j = i + 1;
                while at(j).is_some_and(is_letter) {
                    j += 1;
                }
                break 'm j;
            }
            // \p{N}{1,3}
            if is_number(c) {
                let mut j = i + 1;
                while j < i + 3 && at(j).is_some_and(is_number) {
                    j += 1;
                }
                break 'm j;
            }
            // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
            let other = |x: char| !x.is_whitespace() && !is_letter(x) && !is_number(x);
            let start = if c == ' ' && at(i + 1).is_some_and(other) {
                Some(i + 1)
            } else if other(c) {
                Some(i)
            } else {
                None
            };
            if let Some(s) = start {
                let mut j = s;
                while at(j).is_some_and(other) {
                    j += 1;
                }
                while at(j).is_some_and(is_newline) {
                    j += 1;
                }
                break 'm j;
            }
            // Whitespace: c is whitespace here.
            let mut e = i;
            while at(e).is_some_and(char::is_whitespace) {
                e += 1;
            }
            // \s*[\r\n]+ : up to the last newline in the run.
            if let Some(last) = (i..e).rev().find(|&k| is_newline(cs[k].1)) {
                break 'm last + 1;
            }
            // \s+(?!\S) : the run, less its last char when a non-space follows.
            if e == n {
                break 'm e;
            }
            if e - i >= 2 {
                break 'm e - 1;
            }
            // \s+
            e
        };
        let end = end.max(i + 1);
        out.push(&text[byte(i)..byte(end)]);
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_matches_the_llama3_regex_on_known_cases() {
        assert_eq!(split_llama3("Hello world"), vec!["Hello", " world"]);
        assert_eq!(split_llama3("  two"), vec![" ", " two"]);
        assert_eq!(split_llama3("trailing   "), vec!["trailing", "   "]);
        assert_eq!(split_llama3("a 1234567"), vec!["a", " ", "123", "456", "7"]);
        assert_eq!(split_llama3("don't I'm"), vec!["don", "'t", " I", "'m"]);
        assert_eq!(split_llama3("x\r\n\r\ny"), vec!["x", "\r\n\r\n", "y"]);
        assert_eq!(split_llama3("ok!!! ?"), vec!["ok", "!!!", " ?"]);
        assert_eq!(split_llama3("a\n  b"), vec!["a", "\n", " ", " b"]);
        assert_eq!(split_llama3(""), Vec::<&str>::new());
    }

    #[test]
    fn byte_alphabet_is_gpt2s() {
        let a = byte_alphabet();
        assert_eq!(a[b'A' as usize], 'A');
        assert_eq!(a[b' ' as usize], '\u{120}');
        assert_eq!(a[b'\n' as usize], '\u{10a}');
        let mut seen: Vec<char> = a.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn capitalize_is_jinjas() {
        assert_eq!(capitalize("user"), "User");
        assert_eq!(capitalize("ASSISTANT"), "Assistant");
        assert_eq!(capitalize(""), "");
    }

    /// The reference gate: every text's `ids`, `ids_bos` and `decoded`, every chat's rendering
    /// and `ids_bos`, and the streamed pieces, exactly. Needs `MINERVA_GGUF` (the model) and
    /// `MINERVA_TOKREF` (the reference JSON).
    #[test]
    #[ignore = "needs MINERVA_GGUF and MINERVA_TOKREF"]
    fn reproduces_the_reference_exactly() {
        use serde_json::Value as J;
        let bytes =
            std::fs::read(std::env::var("MINERVA_GGUF").expect("MINERVA_GGUF")).expect("model");
        let t = Tokenizer::from_gguf(&bytes).expect("tokenizer");
        let r: J = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("MINERVA_TOKREF").expect("MINERVA_TOKREF"))
                .expect("reference"),
        )
        .expect("json");
        let ids = |v: &J| -> Vec<u32> {
            v.as_array()
                .expect("ids")
                .iter()
                .map(|x| u32::try_from(x.as_u64().expect("id")).expect("u32"))
                .collect()
        };
        let mut bad = Vec::new();
        let texts = r["texts"].as_array().expect("texts");
        for c in texts {
            let text = c["text"].as_str().expect("text");
            let got = t.encode(text, false);
            if got != ids(&c["ids"]) {
                bad.push(format!(
                    "ids {text:?}: got {got:?} want {:?}",
                    ids(&c["ids"])
                ));
            }
            if t.encode(text, true) != ids(&c["ids_bos"]) {
                bad.push(format!("ids_bos {text:?}"));
            }
            let dec = t.decode(&ids(&c["ids"]));
            if dec != c["decoded"].as_str().expect("decoded") {
                bad.push(format!("decoded {text:?}: got {dec:?}"));
            }
        }
        let chats = r["chats"].as_array().expect("chats");
        for c in chats {
            let msgs: Vec<(String, String)> = c["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .map(|m| {
                    (
                        m["role"].as_str().expect("role").to_owned(),
                        m["content"].as_str().expect("content").to_owned(),
                    )
                })
                .collect();
            let add_gen = c["add_generation_prompt"].as_bool().expect("flag");
            let rendered = t.apply_chat_template(&msgs, add_gen).expect("template");
            if rendered != c["rendered"].as_str().expect("rendered") {
                bad.push(format!("rendered: got {rendered:?}"));
            }
            if t.encode(&rendered, true) != ids(&c["ids_bos"]) {
                bad.push(format!("chat ids_bos for {rendered:?}"));
            }
        }
        for c in r["partial_pieces"].as_array().expect("pieces") {
            let mut d = StreamDecoder::new();
            let mut joined = String::new();
            for id in ids(&c["ids"]) {
                joined.push_str(&d.push(&t, id));
            }
            joined.push_str(&d.finish());
            if joined != c["joined_decoded"].as_str().expect("joined") {
                bad.push(format!("stream {joined:?}"));
            }
        }
        assert!(
            bad.is_empty(),
            "{} of {} cases differ:
{}",
            bad.len(),
            texts.len() * 3 + chats.len() * 2,
            bad.join(
                "
"
            )
        );
    }

    /// What building the tokenizer costs on a mapped model (best of 5). Needs `MINERVA_GGUF`.
    #[test]
    #[ignore = "a benchmark; needs MINERVA_GGUF"]
    fn bench_from_gguf() {
        let m = crate::model::Model::load(std::path::Path::new(
            &std::env::var("MINERVA_GGUF").expect("MINERVA_GGUF"),
        ))
        .expect("model");
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let tok = Tokenizer::from_gguf(m.file_bytes()).expect("tokenizer");
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            assert!(tok.len() > 100_000);
        }
        eprintln!("tokenizer build: {best:.1} ms (best of 5)");
    }
}
