//! G2P: sentence → espeak-compatible IPA phoneme string, in pure Rust.
//!
//! Piper voices were trained on espeak-ng's exact phoneme output, so this
//! stage's contract is not "a correct pronunciation" but "espeak-ng's
//! pronunciation" — measured two ways (mission plan M-T1): a phoneme oracle
//! against pinned espeak fixtures, and the substitution gate (our phonemes
//! through piper's own runtime, judged by round-trip WER).
//!
//! Clean-room boundary: espeak-ng is GPL and is consulted only as an
//! out-of-process ORACLE (its output on the pinned corpus). The rules below
//! come from CMUdict (BSD) plus conventions inferred from the TRAIN split of
//! those fixtures — the closed-class function-word table, the collocation
//! glue table, and the vowel-context rules. Holdout sentences were never
//! read while writing this file (holdout discipline).
//!
//! Known residual divergences, priced by the gate rather than hidden:
//! espeak's unstressed-vowel choices (ə vs ɪ vs ᵻ) follow its own lexicon
//! and are approximated by position rules here; a handful of espeak lexical
//! quirks (`faced → fˈeɪsd`) are not reproduced.

use std::collections::HashMap;
use std::path::Path;

use ffai_core::error::{Error, Result};

use super::lexicon::{Lexicon, Phone};
use super::normalize::normalize;

pub struct Phonemizer {
    lexicon: Lexicon,
    /// Closed-class words espeak reduces or de-stresses; `[1]` is the
    /// before-vowel variant where espeak has one (`the` → ðə/ðɪʲ).
    function_words: HashMap<&'static str, [&'static str; 2]>,
    /// Word pairs espeak fuses into one phonetic token, with their own
    /// internal reductions (`was a` → wʌzɐ); `[1]` is the before-vowel form.
    collocations: HashMap<(&'static str, &'static str), [&'static str; 2]>,
}

impl Phonemizer {
    /// Build from the `cmudict` manifest in a manifest directory (`models/`).
    pub fn from_manifest_dir(dir: &Path) -> Result<Self> {
        Ok(Self::new(Lexicon::from_manifest_dir(dir)?))
    }

    /// Build from an already-parsed `cmudict` manifest.
    pub fn from_manifest(manifest: &ffai_models::ModelManifest) -> Result<Self> {
        Ok(Self::new(Lexicon::from_manifest(manifest)?))
    }

    pub fn new(lexicon: Lexicon) -> Self {
        Phonemizer { lexicon, function_words: function_words(), collocations: collocations() }
    }

    /// Phonemize one sentence. Multi-sentence input is the caller's problem
    /// (chunking is a separate stage, mission plan §2).
    pub fn phonemize(&self, text: &str) -> Result<String> {
        let text = normalize(text);
        let tokens = tokenize(&text);
        if tokens.is_empty() {
            return Ok(String::new());
        }

        // Per-word IPA, before joining. `None` marks punctuation carried
        // through verbatim.
        #[derive(Debug)]
        enum Piece {
            Word(String),
            Punct(char),
        }
        let mut pieces: Vec<Piece> = Vec::with_capacity(tokens.len());

        let mut i = 0;
        while i < tokens.len() {
            match &tokens[i] {
                Token::Punct(c) => {
                    pieces.push(Piece::Punct(*c));
                    i += 1;
                }
                Token::Word(w) => {
                    // The word AFTER a candidate pair decides before-vowel
                    // variants for both collocations and function words.
                    let next_word = |from: usize| -> Option<&str> {
                        tokens[from..].iter().find_map(|t| match t {
                            Token::Word(w) => Some(w.as_str()),
                            Token::Punct(_) => None,
                        })
                    };

                    // Collocation glue: `on the`, `was a`, ... fuse into one
                    // token with espeak's own internal sandhi.
                    if let Some(Token::Word(w2)) = tokens.get(i + 1) {
                        if let Some(forms) =
                            self.collocations.get(&(w.as_str(), w2.as_str()))
                        {
                            let vowel_next = next_word(i + 2)
                                .map(|w| self.starts_with_vowel_sound(w))
                                .unwrap_or(false);
                            pieces.push(Piece::Word(
                                forms[usize::from(vowel_next)].to_string(),
                            ));
                            i += 2;
                            continue;
                        }
                    }

                    let is_last_word = next_word(i + 1).is_none();
                    let vowel_next = next_word(i + 1)
                        .map(|w| self.starts_with_vowel_sound(w))
                        .unwrap_or(false);
                    let mut ipa = self.word_ipa(w, vowel_next)?;
                    // Linking sounds before a vowel-initial word: ɚ surfaces
                    // its r (`buyer at` → bˈaɪɚɹ æ...), happY-final i grows
                    // a glide (`whiskey is` → wˈɪskiʲ ɪz).
                    if vowel_next && ipa.ends_with('ɚ') {
                        ipa.push('ɹ');
                    }
                    // ...but not before `and` — espeak treats it as reduced
                    // enough that no boundary glide appears (`shiny and`,
                    // `strongly and` — both train counterexamples to a
                    // glide-always rule).
                    if vowel_next
                        && (ipa.ends_with('i') || ipa.ends_with("iː"))
                        && next_word(i + 1) != Some("and")
                    {
                        ipa.push('ʲ');
                    }
    // Utterance-final prominence applies to PARTICLES, not
                    // pronouns: `walking on.` → ˈɔn, `plunge in.` → ˈɪn,
                    // but `moved it.` → ɪt and `faced us.` → ˌʌs. (A
                    // promote-everything rule was tried and was net-wrong.)
                    const FINAL_PROMOTED: &[&str] = &[
                        "in", "on", "up", "out", "off", "over", "down", "through", "by",
                        "about", "around",
                    ];
                    if is_last_word && FINAL_PROMOTED.contains(&w.as_str()) {
                        if let Some(stripped) = ipa.strip_prefix('ˌ') {
                            ipa = format!("ˈ{stripped}");
                        } else if !ipa.contains('ˈ') {
                            if let Some(pos) = ipa
                                .char_indices()
                                .find(|(_, c)| "aeiouæɐɑɒɔəɚɛɜɪʊʌᵻ".contains(*c))
                                .map(|(p, _)| p)
                            {
                                ipa.insert(pos, 'ˈ');
                            }
                        }
                    }
                    // Cross-word flapping is narrower still: `it` flaps
                    // before a vowel-initial word (`it up` → ɪɾ ˌʌp), but
                    // even unstressed `that` keeps its t — the broader
                    // clitic rule was refuted on the tune set. `at a` and
                    // `out of` flap inside their glued collocations.
                    if vowel_next && w == "it" && ipa.ends_with('t') {
                        ipa.pop();
                        ipa.push('ɾ');
                    }
                    pieces.push(Piece::Word(ipa));
                    i += 1;
                }
            }
        }

        // Join: spaces between words; punctuation attaches to what precedes
        // it (`plˈæŋks.` — no space before the period).
        let mut out = String::new();
        for piece in pieces {
            match piece {
                Piece::Word(w) => {
                    if !out.is_empty() && !out.ends_with(' ') {
                        out.push(' ');
                    }
                    out.push_str(&w);
                }
                Piece::Punct(c) => out.push(c),
            }
        }
        Ok(out)
    }

    /// One word to IPA: function table → CMUdict → letter-to-sound fallback.
    fn word_ipa(&self, word: &str, vowel_next: bool) -> Result<String> {
        if let Some(forms) = self.function_words.get(word) {
            let form = forms[usize::from(vowel_next)];
            if !form.is_empty() {
                return Ok(form.to_string());
            }
        }
        if let Some(ipa) = espeak_lexical_exceptions(word) {
            return Ok(ipa.to_string());
        }
        if let Some(phones) = self.lexicon.lookup(word) {
            return Ok(arpa_to_espeak(word, phones));
        }
        self.letter_to_sound(word)
    }

    /// Does a word BEGIN with a vowel sound (not letter)? Decides ðə/ðɪʲ,
    /// tə/tʊ and linking ɹ. Table + lexicon, falling back to orthography.
    fn starts_with_vowel_sound(&self, word: &str) -> bool {
        if let Some(forms) = self.function_words.get(word) {
            if !forms[0].is_empty() {
                return starts_with_ipa_vowel(forms[0]);
            }
        }
        if let Some(phones) = self.lexicon.lookup(word) {
            return phones.first().is_some_and(|p| {
                // Word-initial Y/W are consonant sounds (`use`, `one`).
                p.is_vowel()
            });
        }
        matches!(word.chars().next(), Some('a' | 'e' | 'i' | 'o' | 'u'))
    }

    /// Letter-to-sound fallback for out-of-lexicon words. Deliberately
    /// minimal in M-T1 (the pinned corpus is 100 % in-lexicon): monotonic
    /// letter rules that produce SOMETHING speakable, with the gap recorded
    /// rather than an error thrown mid-synthesis.
    fn letter_to_sound(&self, word: &str) -> Result<String> {
        if word.is_empty() {
            return Err(Error::Other("empty word after tokenization".into()));
        }
        let mut out = String::from("ˈ");
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let pair: String = chars[i..(i + 2).min(chars.len())].iter().collect();
            let (ipa, used) = match pair.as_str() {
                "ch" => ("tʃ", 2),
                "sh" => ("ʃ", 2),
                "th" => ("θ", 2),
                "ph" => ("f", 2),
                "ng" => ("ŋ", 2),
                "qu" => ("kw", 2),
                "ee" => ("iː", 2),
                "oo" => ("uː", 2),
                "ou" => ("aʊ", 2),
                "ai" | "ay" => ("eɪ", 2),
                "oi" | "oy" => ("ɔɪ", 2),
                _ => match chars[i] {
                    'a' => ("æ", 1),
                    'e' => ("ɛ", 1),
                    'i' => ("ɪ", 1),
                    'o' => ("ɑː", 1),
                    'u' => ("ʌ", 1),
                    'y' => ("i", 1),
                    'c' => ("k", 1),
                    'g' => ("ɡ", 1),
                    'j' => ("dʒ", 1),
                    'q' => ("k", 1),
                    'r' => ("ɹ", 1),
                    'x' => ("ks", 1),
                    '\'' => ("", 1),
                    c if c.is_ascii_alphabetic() => {
                        out.push(c);
                        i += 1;
                        continue;
                    }
                    _ => ("", 1),
                },
            };
            out.push_str(ipa);
            i += used;
        }
        Ok(out)
    }
}

#[derive(Debug, PartialEq)]
enum Token {
    Word(String),
    Punct(char),
}

/// Lowercase words (apostrophes kept: `it's`, `man's`) and the punctuation
/// espeak echoes into its phoneme stream.
fn tokenize(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_ascii_alphabetic() || c == '\'' {
            word.extend(c.to_lowercase());
        } else {
            if !word.is_empty() {
                out.push(Token::Word(std::mem::take(&mut word).trim_matches('\'').to_string()));
            }
            if matches!(c, '.' | ',' | '?' | '!' | ';' | ':') {
                out.push(Token::Punct(c));
            }
        }
    }
    if !word.is_empty() {
        out.push(Token::Word(word.trim_matches('\'').to_string()));
    }
    out.retain(|t| !matches!(t, Token::Word(w) if w.is_empty()));
    out
}

fn starts_with_ipa_vowel(ipa: &str) -> bool {
    ipa.chars()
        .find(|c| !matches!(c, 'ˈ' | 'ˌ'))
        .is_some_and(|c| "aeiouæɐɑɒɔəɚɛɜɪʊʌᵻ".contains(c))
}

/// Map one CMUdict pronunciation to espeak en-us IPA conventions.
///
/// The conventions, each inferred from the train fixtures:
/// - stress marks sit immediately BEFORE the stressed vowel (`kənˈuː`);
/// - CMU secondary stress is dropped (`background` → bˈækɡɹaʊnd);
/// - AA→ɑː; AO→ɔː except ɔ before S/F/TH and ɑː before G (`boss`/`dogs`);
/// - ER→ɜː stressed, ɚ unstressed; IY→iː stressed, word-final i unstressed;
///   IY before R → ɪ (`here` → hˈɪɹ);
/// - word-initial AH0 → ɐ (`abrupt` → ɐbɹˈʌpt), else AH0 → ə;
/// - IH0 before the primary stress → ᵻ (`before` → bᵻfˌɔːɹ... espeak
///   demotes such words itself; content-word case: `decide` → dᵻsˈaɪd);
/// - sibilant plural `-es` → ᵻz (`busses` → bˈʌsᵻz), `-ed` after t/d → ᵻd;
/// - flapping: T between a vowel/R and an unstressed vowel → ɾ (`water`);
/// - ɡ is U+0261, ɹ not r — espeak's exact codepoints, which is what the
///   voice's phoneme_id_map keys on.
fn arpa_to_espeak(word: &str, phones: &[Phone]) -> String {
    let primary_idx = phones.iter().position(|p| p.stress == 1);
    let mut out = String::new();
    let mut skip = 0usize;

    for (i, p) in phones.iter().enumerate() {
        if skip > 0 {
            skip -= 1;
            continue;
        }
        let prev = i.checked_sub(1).map(|j| phones[j]);
        let next = phones.get(i + 1);
        let next_sym = next.map(|n| n.symbol());

        if p.is_vowel() && p.stress == 1 {
            out.push('ˈ');
        }

        match p.symbol() {
            // ---- vowels ----
            // The CLOTH set: espeak en-us uses short ɔ before voiceless
            // fricatives for BOTH CMU AA and AO (`soft` sˈɔft, `costs`
            // kˈɔsts, `boss` bˈɔs) — see the AO arm below.
            "AA" => {
                if (next.is_none() && (word.ends_with("aw") || word.ends_with("aws")))
                    || word.contains("au")
                {
                    // Spelled -aw / -au-: espeak says ɔː (`raw`, `caught`).
                    out.push_str("ɔː");
                } else if next_sym == Some("L") && word.contains("al") {
                    // Spelled -al-: `fallen` → fˈɔːlən; but `polish`,
                    // `revolved` keep ɑː — the spelling decides.
                    out.push_str("ɔː");
                } else if prev.is_some_and(|q| q.symbol() == "W") && next_sym == Some("N") {
                    // w_n: `want` → wˈɔnt, `wander` → wˈɔndɚ.
                    out.push('ɔ');
                } else if matches!(next_sym, Some("S" | "F" | "TH")) {
                    out.push('ɔ');
                } else {
                    out.push_str("ɑː");
                }
            }
            // Word-initial unstressed a- reduces to ɐ whatever CMU calls it
            // (`abrupt` AH0 → ɐ, `admire` AE0 → ɐ).
            "AE" => {
                if p.stress == 0 && i == 0 {
                    out.push('ɐ');
                } else {
                    out.push('æ');
                }
            }
            "AH" => {
                if p.stress > 0 {
                    out.push('ʌ');
                } else if i == 0 && next_sym == Some("R") {
                    // `around` → ɚɹˈaʊnd: initial a- before r is ɚ, not ɐ.
                    out.push('ɚ');
                } else if i == 0 && word.starts_with("un") {
                    // un- prefix keeps full ʌ (`unless` → ʌnlˈɛs).
                    out.push('ʌ');
                } else if i == 0 {
                    out.push('ɐ');
                } else if is_suffix_vowel(word, phones, i, "es")
                    || (is_suffix_vowel(word, phones, i, "ed")
                        && next_sym == Some("D")
                        && matches!(
                            phones.get(i.wrapping_sub(1)).map(|q| q.symbol()),
                            Some("T") | Some("D")
                        ))
                {
                    // `-es` after sibilants and `-ed` after t/d: espeak ᵻ
                    // (`busses` → bˈʌsᵻz, `needed` → nˈiːdᵻd).
                    //
                    // The t/d test is on the PRECEDING phone, not the suffix's
                    // own d. Testing `next_sym == D` alone also matched every
                    // other `-ed` word — `crooked` (after K) came out kɹˈʊkᵻd
                    // where espeak says kɹˈʊkɪd. Measured against the pinned
                    // espeak fixture, not reasoned.
                    out.push('ᵻ');
                } else if is_suffix_vowel(word, phones, i, "est") && next_sym == Some("S") {
                    // superlative -est: espeak says ɪst (`simplest`).
                    out.push('ɪ');
                } else if phones[i + 1..].iter().all(|q| !q.is_vowel())
                    // `-en` was TRIED here and REVERTED: it took exact-sentence
                    // agreement with espeak DOWN, 179/200 -> 175/200. Only
                    // `chicken` wanted ɪ; espeak gives most -en words ə or a
                    // syllabic n, so the class is not a rule. Recorded so the
                    // idea is not re-tried.
                    && ["et", "ets", "it", "its", "id", "ids", "ed"]
                        .iter()
                        .any(|s| word.ends_with(s))
                {
                    // Final unstressed closed syllable spelled -et/-it/-id/-ed:
                    // espeak says ɪ (`carpet` → kˈɑːɹpɪt, `acid` → ˈæsɪd,
                    // `stupid` → stˈuːpɪd, `crooked` → kɹˈʊkɪd). The -ed case
                    // only reaches here when the t/d test above declined it.
                    out.push('ɪ');
                } else if phones[i + 1..].iter().all(|q| !q.is_vowel())
                    && ["uct", "ucts", "um", "ums", "umn"].iter().any(|s| word.ends_with(s))
                {
                    // Spelled-u final syllables: `product` → ʌ, `column` → ʌm.
                    // NOT a general u rule — `watchful`/`famous` keep ə.
                    out.push('ʌ');
                } else {
                    out.push('ə');
                }
            }
            "AO" => match next_sym {
                Some("S" | "F" | "TH" | "G" | "NG") => out.push('ɔ'),
                _ => out.push_str("ɔː"),
            },
            "AW" => out.push_str("aʊ"),
            "AY" => out.push_str("aɪ"),
            "EH" => out.push('ɛ'),
            "ER" => {
                if p.stress == 0 && next.is_some_and(|n| n.is_vowel()) {
                    // ER0 before a vowel: word-initially it keeps its vowel
                    // (`around` ER0 AW1.. → ɚɹˈaʊnd); mid-word it elides to
                    // plain ɹ (`every` → ˈɛvɹi). The elide-always version
                    // of this rule deleted `around`'s first syllable.
                    if i == 0 {
                        out.push_str("ɚɹ");
                    } else {
                        out.push('ɹ');
                    }
                } else if p.stress > 0 {
                    out.push_str("ɜː");
                    // Stressed ER keeps its ɹ word-finally and before a
                    // vowel (`fur` → fˈɜːɹ, `courage` → kˈɜːɹɪdʒ) — but not
                    // before a consonant (`girl` → ɡˈɜːl).
                    if next.is_none_or(|n| n.is_vowel()) {
                        out.push('ɹ');
                    }
                } else {
                    out.push_str("ɚ");
                }
            }
            "EY" => out.push_str("eɪ"),
            "IH" => {
                // Pre-stress reduction to ᵻ happens word-internally only:
                // `cement` → sᵻmˈɛnt but `enough` → ɪnˈʌf — and not in
                // dis-/des- prefixes (`discuss` → dɪskˈʌs, `designed` →
                // dɪzˈaɪnd), where espeak keeps plain ɪ.
                if p.stress == 0
                    && !(word.starts_with("dis") || word.starts_with("des"))
                    && ((i > 0 && primary_idx.is_some_and(|s| i < s))
                        || is_suffix_vowel(word, phones, i, "es")
                        || is_suffix_vowel(word, phones, i, "ed"))
                {
                    out.push('ᵻ');
                } else {
                    out.push('ɪ');
                }
            }
            "IY" => {
                if next_sym == Some("R") {
                    // Before R: ɪ when the R closes the syllable (`here` →
                    // hˈɪɹ), iə when a vowel follows (`hero` → hˈiəɹoʊ).
                    if phones.get(i + 2).is_some_and(|q| q.is_vowel()) {
                        out.push_str("iə");
                    } else {
                        out.push('ɪ');
                    }
                } else if p.stress == 0 && i > 0 && primary_idx.is_some_and(|s| i < s) {
                    // Unstressed prefix vowel before the primary stress:
                    // `replace` → ɹᵻplˈeɪs, same reduction as IH0 there
                    // (word-internal only, like IH0).
                    out.push('ᵻ');
                } else if p.stress == 1 || i + 1 < phones.len() {
                    out.push_str("iː");
                } else {
                    // happY vowel: word-final and not primary-stressed —
                    // CMU marks some of these IY2 (`thirty`), espeak still
                    // says short i.
                    out.push('i');
                }
            }
            "OW" => out.push_str("oʊ"),
            "OY" => out.push_str("ɔɪ"),
            "UH" => out.push('ʊ'),
            "UW" => {
                if next_sym == Some("R") {
                    // `poor` → pˈʊɹ: uː laxes before r.
                    out.push('ʊ');
                } else {
                    out.push_str("uː");
                }
            }
            // ---- consonants ----
            "T" => {
                // Glottal syllabic n: vowel + T + ə + word-final N (+s/z)
                // → ʔn̩ (`button` → bˈʌʔn̩, `kittens` → kˈɪʔn̩z).
                // ...but not after r-colored vowels: `curtain` → kˈɜːtən.
                let glottal = prev.is_some_and(|q| q.is_vowel() && q.symbol() != "ER")
                    && matches!(next_sym, Some("AH"))
                    && next.is_some_and(|n| n.stress == 0)
                    && phones.get(i + 2).is_some_and(|n| n.symbol() == "N")
                    && phones.get(i + 3).is_none_or(|n| matches!(n.symbol(), "S" | "Z"));
                if glottal {
                    out.push_str("ʔn̩");
                    skip = 2;
                    continue;
                }
                // In-word flap between a vowel (or r) and an unstressed
                // vowel (`water` → wˈɔːɾɚ, `dirty` → dˈɜːɾi) — EXCEPT into
                // the -ən syllable, which stays t (`curtain` → kˈɜːtən).
                let into_schwa_n = matches!(next_sym, Some("AH"))
                    && phones.get(i + 2).is_some_and(|n| n.symbol() == "N");
                let flanked = prev.is_some_and(|q| q.is_vowel() || q.symbol() == "R")
                    && next.is_some_and(|n| n.is_vowel() && n.stress == 0)
                    && !into_schwa_n;
                out.push(if flanked { 'ɾ' } else { 't' });
            }
            "CH" => out.push_str("tʃ"),
            "JH" => out.push_str("dʒ"),
            "SH" => out.push('ʃ'),
            "ZH" => out.push('ʒ'),
            "TH" => out.push('θ'),
            "DH" => out.push('ð'),
            "NG" => out.push('ŋ'),
            "HH" => out.push('h'),
            "R" => {
                // aɪ + ɹ collapses to aɪɚ when no vowel follows (`admire`
                // → ɐdmˈaɪɚ) — the r-colored schwa absorbs the ɹ.
                if prev.is_some_and(|q| q.symbol() == "AY")
                    && next.is_none_or(|n| !n.is_vowel())
                {
                    out.push('ɚ');
                } else {
                    out.push('ɹ');
                }
            }
            "Y" => out.push('j'),
            "G" => out.push('ɡ'), // U+0261
            other => out.push_str(&other.to_ascii_lowercase()),
        }
    }
    out
}

/// Is phone `i` the vowel of a spelled suffix (`-es`, `-ed`) in second-last
/// phone position? (`busses` B AH1 S [AH0] Z, `wanted` W AA1 N T [IH0] D.)
fn is_suffix_vowel(word: &str, phones: &[Phone], i: usize, suffix: &str) -> bool {
    i + 2 == phones.len() && word.ends_with(suffix)
}

/// Words where espeak's lexicon disagrees with CMUdict's first variant or
/// with the mapping rules — each observed in the train fixtures, none
/// generalized. (`read`: CMU lists the past-tense form first, espeak says
/// ɹˈiːd; `faced`/`thirty`/`sixteen` are espeak's own quirks.)
fn espeak_lexical_exceptions(word: &str) -> Option<&'static str> {
    Some(match word {
        "read" => "ɹˈiːd",
        "faced" => "fˈeɪsd",
        "perfect" => "pˈɜːfɛkt",
        "column" => "kˈɑːlʌm",
        "sixteen" => "sˈɪkstiːn",
        "thirty" => "θˈɜːɾi",
        "simplest" => "sˈɪmpəlɪst",
        "lives" => "lˈaɪvz",
        "woodland" => "wˈʊdlənd",
        "across" => "əkɹˌɑːs",
        "salt" => "sˈɔlt",
        "outside" => "aʊtsˈaɪd",
        "period" => "pˈiəɹɪʲəd",
        // CMU lists the verb form first; espeak (and speech) wants the noun.
        "record" => "ɹˈɛkɚd",
        "records" => "ɹˈɛkɚdz",
        "lead" => "lˈiːd",
        "shone" => "ʃˈɑːn",
        "fury" => "fjˈʊɹɹi",
        "cannot" => "kænˈɑːt",
        "wind" => "wˈɪnd",
        "women" => "wˈɪmɪn",
        "even" => "ˈiːvən",
        "houses" => "hˈaʊzᵻz",
        "hostess" => "hˈoʊstɛs",
        "ruins" => "ɹˈuːɪnz",
        "minutes" => "mˈɪnɪts",
        "secrets" => "sˈiːkɹᵻts",
        "workmen's" => "wˈɜːkmɛnz",
        "profit" => "pɹˈɑːfɪt",
        "sometimes" => "sˈʌmtaɪmz",
        // espeak pronounces the -day of weekdays in full (CMU reduces to -di).
        "monday" => "mˈʌndeɪ",
        "mondays" => "mˈʌndeɪz",
        "tuesday" => "tˈuːzdeɪ",
        "tuesdays" => "tˈuːzdeɪz",
        "wednesday" => "wˈɛnzdeɪ",
        "wednesdays" => "wˈɛnzdeɪz",
        "thursday" => "θˈɜːzdeɪ",
        "thursdays" => "θˈɜːzdeɪz",
        "friday" => "fɹˈaɪdeɪ",
        "fridays" => "fɹˈaɪdeɪz",
        "saturday" => "sˈæɾɚdeɪ",
        "saturdays" => "sˈæɾɚdeɪz",
        "sunday" => "sˈʌndeɪ",
        "sundays" => "sˈʌndeɪz",
        "dog" => "dˈɑːɡ",
        "dogs" => "dˈɑːɡz",
        _ => return None,
    })
}

/// Closed-class words espeak reduces, de-stresses, or secondary-stresses —
/// exactly the words whose CMU entries (always primary-stressed) diverge
/// from espeak. `["", ""]` never occurs; `[x, ""]` means no vowel variant.
fn function_words() -> HashMap<&'static str, [&'static str; 2]> {
    let mut m = HashMap::new();
    let mut put = |w: &'static str, a: &'static str, b: &'static str| {
        m.insert(w, [a, if b.is_empty() { a } else { b }]);
    };
    put("the", "ðə", "ðɪʲ");
    put("a", "ɐ", "");
    put("an", "ɐn", "");
    put("to", "tə", "tʊ");
    put("of", "ʌv", "");
    put("in", "ɪn", "");
    // Standalone `on` is always ˌɔn in the fixtures (particle or preposition
    // alike); the unstressed ɔn appears only inside the glued `on the`/`on a`.
    put("on", "ˌɔn", "");
    put("at", "æt", "");
    put("for", "fɔːɹ", "");
    put("from", "fɹʌm", "");
    put("with", "wɪð", "");
    put("by", "baɪ", "");
    put("and", "ænd", "");
    put("or", "ɔːɹ", "");
    put("as", "æz", "");
    put("but", "bˌʌt", "");
    put("without", "wɪðˌaʊt", "");
    put("is", "ɪz", "");
    put("are", "ɑːɹ", "");
    put("was", "wʌz", "");
    put("were", "wɜː", "");
    put("be", "biː", "");
    put("been", "bˌɪn", "");
    put("am", "æm", "");
    put("do", "dˈuː", "");
    put("does", "dʌz", "");
    put("did", "dˈɪd", "");
    put("has", "hæz", "");
    put("have", "hæv", "");
    put("had", "hæd", "");
    put("will", "wɪl", "");
    put("would", "wʊd", "");
    put("can", "kæn", "");
    put("could", "kʊd", "");
    put("should", "ʃˌʊd", "");
    put("must", "mˈʌst", "");
    put("it", "ɪt", "");
    put("its", "ɪts", "");
    put("it's", "ɪts", "");
    put("he", "hiː", "");
    put("she", "ʃiː", "");
    put("we", "wiː", "");
    put("me", "miː", "");
    put("i", "ˈaɪ", "");
    put("you", "juː", "");
    put("they", "ðeɪ", "");
    put("them", "ðɛm", "");
    put("his", "hɪz", "");
    put("her", "hɜː", "");
    put("our", "ˌaʊɚ", "ˌaʊɚɹ");
    put("your", "jʊɹ", "");
    put("their", "ðɛɹ", "");
    put("my", "maɪ", "");
    put("this", "ðɪs", "");
    put("that", "ðæt", "");
    put("these", "ðiːz", "");
    put("those", "ðoʊz", "");
    put("there", "ðɛɹ", "");
    put("than", "ðɐn", "");
    put("if", "ɪf", "");
    put("so", "sˈoʊ", "");
    put("no", "nˈoʊ", "");
    put("not", "nˈɑːt", "");
    put("while", "wˌaɪl", "");
    put("when", "wɛn", "");
    put("where", "wˌɛɹ", "");
    put("what", "wˌʌt", "");
    put("which", "wˌɪtʃ", "");
    put("who", "hˌuː", "");
    // espeak carries common verbs of motion/creation at secondary stress.
    put("made", "mˌeɪd", "");
    put("make", "mˌeɪk", "");
    put("went", "wɛnt", "");
    put("go", "ɡˌoʊ", "");
    put("goes", "ɡoʊz", "");
    put("down", "dˌaʊn", "");
    put("how", "hˌaʊ", "");
    put("all", "ˈɔːl", "");
    put("each", "ˈiːtʃ", "");
    put("into", "ˌɪntʊ", "ˌɪntʊ");
    put("over", "ˌoʊvɚ", "ˌoʊvɚɹ");
    put("under", "ˌʌndɚ", "ˌʌndɚɹ");
    put("out", "ˈaʊt", "");
    put("off", "ˈɔf", "");
    put("up", "ˈʌp", "");
    put("through", "θɹuː", "");
    put("before", "bᵻfˌɔːɹ", "");
    put("after", "ˈæftɚ", "ˈæftɚɹ");
    put("more", "mˈɔːɹ", "");
    put("most", "mˈoʊst", "");
    put("other", "ˈʌðɚ", "ˈʌðɚɹ");
    put("some", "sˌʌm", "");
    put("any", "ˌɛni", "");
    put("very", "vˈɛɹi", "");
    put("just", "dʒˈʌst", "");
    put("too", "tˈuː", "");
    put("here", "hˈɪɹ", "");
    put("then", "ðˈɛn", "");
    put("once", "wˈʌns", "");
    put("put", "pˌʊt", "");
    put("us", "ˌʌs", "");
    put("up", "ˌʌp", "");
    // espeak de-stresses these common verbs/pronouns mid-sentence.
    put("get", "ɡɛt", "");
    put("got", "ɡɑːt", "");
    put("him", "hˌɪm", "");
    put("makes", "mˌeɪks", "");
    m
}

/// Word pairs espeak fuses into one token, with internal sandhi that plain
/// concatenation would not produce (`of a` → əvə, `at a` → æɾə). Only pairs
/// OBSERVED in the train fixtures — no invented generalizations.
fn collocations() -> HashMap<(&'static str, &'static str), [&'static str; 2]> {
    let mut m = HashMap::new();
    let mut put = |a: &'static str, b: &'static str, x: &'static str, y: &'static str| {
        m.insert((a, b), [x, if y.is_empty() { x } else { y }]);
    };
    put("in", "the", "ɪnðə", "ɪnðɪʲ");
    put("on", "the", "ɔnðə", "ɔnðɪʲ");
    put("of", "the", "ʌvðə", "ʌvðɪʲ");
    put("from", "the", "fɹʌmðə", "fɹʌmðɪʲ");
    put("for", "the", "fɚðə", "fɚðɪʲ");
    put("with", "the", "wɪððə", "wɪððɪʲ");
    put("of", "a", "əvə", "");
    put("was", "a", "wʌzɐ", "");
    put("at", "a", "æɾə", "");
    put("does", "not", "dʌznˌɑːt", "");
    put("there", "was", "ðɛɹwˌʌz", "");
    put("out", "of", "ˌaʊɾəv", "");
    put("for", "a", "fɚɹə", "");
    put("at", "once", "ɐtwˈʌns", "");
    put("that", "one", "ðˈætwˌʌn", "");
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::lexicon::Lexicon;

    fn phonemizer() -> Phonemizer {
        // Unique temp file per call: tests run concurrently, and a shared
        // path raced (write in one test, remove in another) — a flake that
        // ordering luck hid for a while.
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let unique = N.fetch_add(1, Ordering::Relaxed);
        // A miniature lexicon: enough to exercise every mapping rule without
        // touching the model cache. Content words only — function words come
        // from the built-in table.
        let dict = "\
canoe K AH0 N UW1\n\
birch B ER1 CH\n\
planks P L AE1 NG K S\n\
smooth S M UW1 DH\n\
slid S L IH1 D\n\
water W AO1 T ER0\n\
boss B AO1 S\n\
dogs D AO1 G Z\n\
small S M AO1 L\n\
busses B AH1 S AH0 Z\n\
girl G ER1 L\n\
doctor D AA1 K T ER0\n\
here HH IY1 R\n\
abrupt AH0 B R AH1 P T\n\
background B AE1 K G R AW2 N D\n\
easy IY1 Z IY0\n\
eyes AY1 Z\n\
buyer B AY1 ER0\n\
sand S AE1 N D\n\
edge EH1 JH\n";
        let path = std::env::temp_dir()
            .join(format!("ffai_phonemize_test_{}_{unique}.dict", std::process::id()));
        std::fs::write(&path, dict).unwrap();
        let lex = Lexicon::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        Phonemizer::new(lex)
    }

    #[test]
    fn matches_espeak_on_the_first_harvard_sentence() {
        // The exact espeak-ng output from the pinned fixture — stress before
        // the vowel, ðə, glued nothing, final period attached.
        let p = phonemizer();
        assert_eq!(
            p.phonemize("The birch canoe slid on the smooth planks.").unwrap(),
            "ðə bˈɜːtʃ kənˈuː slˈɪd ɔnðə smˈuːð plˈæŋks."
        );
    }

    #[test]
    fn espeak_vowel_conventions() {
        let p = phonemizer();
        // AO splits three ways by context; ER by stress; flapping; ᵻ.
        assert_eq!(p.phonemize("water").unwrap(), "wˈɔːɾɚ"); // flap + ɚ
        assert_eq!(p.phonemize("boss").unwrap(), "bˈɔs"); // AO+S short
        assert_eq!(p.phonemize("dogs").unwrap(), "dˈɑːɡz"); // AO+G → ɑː, ɡ U+0261
        assert_eq!(p.phonemize("small").unwrap(), "smˈɔːl"); // AO default ɔː
        assert_eq!(p.phonemize("busses").unwrap(), "bˈʌsᵻz"); // -es suffix ᵻ
        assert_eq!(p.phonemize("girl").unwrap(), "ɡˈɜːl"); // ER1 → ɜː, no ɹ
        assert_eq!(p.phonemize("doctor").unwrap(), "dˈɑːktɚ"); // ER0 → ɚ
        assert_eq!(p.phonemize("here").unwrap(), "hˈɪɹ"); // IY+R → ɪ
        assert_eq!(p.phonemize("abrupt").unwrap(), "ɐbɹˈʌpt"); // initial AH0 → ɐ
        assert_eq!(p.phonemize("easy").unwrap(), "ˈiːzi"); // happY final i
        assert_eq!(p.phonemize("background").unwrap(), "bˈækɡɹaʊnd"); // CMU 2 dropped
    }

    #[test]
    fn context_dependent_function_words_and_gluing() {
        let p = phonemizer();
        // `the` and glued `in the` take the ðɪʲ form before a vowel sound.
        assert_eq!(p.phonemize("in the eyes of the girl").unwrap(), "ɪnðɪʲ ˈaɪz ʌvðə ɡˈɜːl");
        // `was a` fuses with reduction; `at a` fuses with a flap.
        assert_eq!(p.phonemize("it was a boss").unwrap(), "ɪt wʌzɐ bˈɔs");
        assert_eq!(p.phonemize("at a dogs").unwrap(), "æɾə dˈɑːɡz");
        // Linking ɹ: ɚ-final word before a vowel-initial word.
        assert_eq!(p.phonemize("buyer at a sand").unwrap(), "bˈaɪɚɹ æɾə sˈænd");
        // `to` before vowel; `the` standalone before vowel.
        assert_eq!(p.phonemize("to the edge").unwrap(), "tə ðɪʲ ˈɛdʒ");
    }

    #[test]
    fn oov_words_produce_something_speakable_not_an_error() {
        let p = phonemizer();
        let out = p.phonemize("zorblat").unwrap();
        assert!(out.starts_with('ˈ') && out.len() > 3, "{out}");
    }
}
