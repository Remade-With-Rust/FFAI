//! Text normalization for error-rate scoring — a port of OpenAI Whisper's
//! `whisper/normalizers/{basic,english}.py`.
//!
//! # Why this exists
//!
//! Raw WER between an ASR hypothesis and a LibriSpeech reference is mostly
//! noise about formatting. The reference says `MISTER QUILTER`, Whisper
//! writes `Mr. Quilter`; the reference says `TWENTY THREE`, Whisper writes
//! `23`. Scoring those as errors would make every implementation — ours and
//! the references alike — look far worse than it is, and would put our
//! published numbers nowhere near the world's.
//!
//! Normalization is applied identically to reference and hypothesis, and
//! identically to every implementation under test, so it cannot advantage
//! anyone.
//!
//! # Parity status (honest accounting)
//!
//! Implemented faithfully: lowercasing, bracket/parenthesis stripping, filler
//! removal, the contraction and title replacer table, comma-in-digit and
//! period handling, symbol stripping with the numeric keep-set, and
//! spelled-number → digit conversion.
//!
//! **Not yet at bit-parity with openai-whisper**, tracked as a Mercury M1
//! exit item:
//!
//! - the ~1,700-entry British→American spelling map (`english.json`),
//! - Unicode NFKD decomposition and general-category-based diacritic removal
//!   (we approximate: non-alphanumeric, non-keep characters become spaces),
//! - fractions, currency-suffix forms, and the year-pair heuristics in
//!   `EnglishNumberNormalizer`.
//!
//! Until those land, treat cross-implementation comparisons produced here as
//! sound (same normalizer for everyone) and absolute agreement with published
//! WER figures as approximate.

use regex::Regex;
use std::sync::OnceLock;

/// Which normalizer to score with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// No normalization — raw string comparison.
    None,
    /// Whisper's `BasicTextNormalizer`: language-agnostic.
    Basic,
    /// Whisper's `EnglishTextNormalizer`: the default for English corpora.
    #[default]
    English,
    /// OCR scoring: whitespace runs (including line breaks) collapse to one
    /// space, everything else is preserved.
    ///
    /// Deliberately NOT the ASR normalizer: reading case, punctuation, and
    /// digits correctly is OCR's job, so folding them away would score the
    /// task's hardest parts as free. Whitespace is collapsed because engines
    /// legitimately disagree about line-break placement, and the corpus keeps
    /// reading order unambiguous (single-column, top-to-bottom) so a flat
    /// comparison is fair.
    Ocr,
}

/// Normalize `text` under `mode`.
pub fn normalize(text: &str, mode: Mode) -> String {
    match mode {
        Mode::None => text.to_string(),
        Mode::Basic => basic(text),
        Mode::English => english(text),
        Mode::Ocr => collapse_whitespace(text),
    }
}

/// Whisper's `BasicTextNormalizer`.
fn basic(text: &str) -> String {
    let s = text.to_lowercase();
    let s = re(r"[<\[][^>\]]*[>\]]").replace_all(&s, "");
    let s = re(r"\(([^)]+?)\)").replace_all(&s, "");
    let s = strip_symbols(&s, "");
    collapse_whitespace(&s)
}

/// Whisper's `EnglishTextNormalizer`.
fn english(text: &str) -> String {
    let s = text.to_lowercase();
    let s = re(r"[<\[][^>\]]*[>\]]").replace_all(&s, "").into_owned();
    let s = re(r"\(([^)]+?)\)").replace_all(&s, "").into_owned();
    // Fillers Whisper drops outright.
    let s = re(r"\b(hmm|mm|mhm|mmm|uh|um)\b")
        .replace_all(&s, "")
        .into_owned();
    // Standardize a space before an apostrophe ("it 's" -> "it's").
    let s = re(r"\s+'").replace_all(&s, "'").into_owned();

    let mut s = s;
    for (pattern, replacement) in replacers() {
        s = pattern.replace_all(&s, *replacement).into_owned();
    }

    // Commas inside numbers: "1,234" -> "1234".
    let s = re(r"(\d),(\d)").replace_all(&s, "${1}${2}").into_owned();
    // Periods not followed by a digit are punctuation, not decimal points.
    let s = re(r"\.([^0-9]|$)").replace_all(&s, " ${1}").into_owned();
    // Keep the symbols that carry numeric meaning.
    let s = strip_symbols(&s, ".%$¢€£");
    let s = words_to_digits(&s);
    // Now drop numeric symbols that turned out not to be attached to digits.
    let s = re(r"[.$¢€£]([^0-9])").replace_all(&s, " ${1}").into_owned();
    let s = re(r"([^0-9])%").replace_all(&s, "${1} ").into_owned();
    collapse_whitespace(&s)
}

/// Replace symbol/punctuation characters with spaces, except `keep`.
/// (Approximates Whisper's Unicode-category pass — see module docs.)
fn strip_symbols(text: &str, keep: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || keep.contains(c) {
                c
            } else {
                ' '
            }
        })
        .collect()
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn re(pattern: &str) -> Regex {
    // Small, fixed set of patterns; compiled per call is fine at corpus
    // scale (thousands of strings, not millions). Hoist into a OnceLock table
    // if this ever shows up in a profile.
    Regex::new(pattern).expect("static pattern")
}

/// Whisper's contraction / title replacer table, applied in order.
fn replacers() -> &'static [(Regex, &'static str)] {
    static TABLE: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        [
            // common contractions
            (r"\bwon't\b", "will not"),
            (r"\bcan't\b", "can not"),
            (r"\blet's\b", "let us"),
            (r"\bain't\b", "aint"),
            (r"\by'all\b", "you all"),
            (r"\bwanna\b", "want to"),
            (r"\bgotta\b", "got to"),
            (r"\bgonna\b", "going to"),
            (r"\bi'ma\b", "i am going to"),
            (r"\bimma\b", "i am going to"),
            (r"\bwoulda\b", "would have"),
            (r"\bcoulda\b", "could have"),
            (r"\bshoulda\b", "should have"),
            (r"\bma'am\b", "madam"),
            // titles and abbreviations — the big win on read-speech corpora
            (r"\bmr\b", "mister "),
            (r"\bmrs\b", "missus "),
            (r"\bst\b", "saint "),
            (r"\bdr\b", "doctor "),
            (r"\bprof\b", "professor "),
            (r"\bcapt\b", "captain "),
            (r"\bgov\b", "governor "),
            (r"\bald\b", "alderman "),
            (r"\bgen\b", "general "),
            (r"\bsen\b", "senator "),
            (r"\brep\b", "representative "),
            (r"\bpres\b", "president "),
            (r"\brev\b", "reverend "),
            (r"\bhon\b", "honorable "),
            (r"\basst\b", "assistant "),
            (r"\bassoc\b", "associate "),
            (r"\blt\b", "lieutenant "),
            (r"\bcol\b", "colonel "),
            (r"\bjr\b", "junior "),
            (r"\bsr\b", "senior "),
            (r"\besq\b", "esquire "),
            // perfect tenses
            (r"'d been\b", " had been"),
            (r"'s been\b", " has been"),
            (r"'d gone\b", " had gone"),
            (r"'s gone\b", " has gone"),
            (r"'d done\b", " had done"),
            (r"'s got\b", " has got"),
            // general contractions
            (r"n't\b", " not"),
            (r"'re\b", " are"),
            (r"'s\b", " is"),
            (r"'d\b", " would"),
            (r"'ll\b", " will"),
            (r"'t\b", " not"),
            (r"'ve\b", " have"),
            (r"'m\b", " am"),
        ]
        .into_iter()
        .map(|(p, r)| (Regex::new(p).expect("static pattern"), r))
        .collect()
    })
}

fn unit_value(word: &str) -> Option<u64> {
    Some(match word {
        "zero" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        _ => return None,
    })
}

fn tens_value(word: &str) -> Option<u64> {
    Some(match word {
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fourty" => 40, // common misspelling, as Whisper tolerates
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => return None,
    })
}

fn scale_value(word: &str) -> Option<u64> {
    Some(match word {
        "thousand" => 1_000,
        "million" => 1_000_000,
        "billion" => 1_000_000_000,
        "trillion" => 1_000_000_000_000,
        _ => return None,
    })
}

/// Ordinal word → (value, suffix), e.g. "third" → (3, "rd").
fn ordinal_value(word: &str) -> Option<(u64, &'static str)> {
    let v = match word {
        "first" => 1,
        "second" => 2,
        "third" => 3,
        "fourth" => 4,
        "fifth" => 5,
        "sixth" => 6,
        "seventh" => 7,
        "eighth" => 8,
        "ninth" => 9,
        "tenth" => 10,
        "eleventh" => 11,
        "twelfth" => 12,
        "thirteenth" => 13,
        "fourteenth" => 14,
        "fifteenth" => 15,
        "sixteenth" => 16,
        "seventeenth" => 17,
        "eighteenth" => 18,
        "nineteenth" => 19,
        "twentieth" => 20,
        "thirtieth" => 30,
        "fortieth" => 40,
        "fiftieth" => 50,
        "sixtieth" => 60,
        "seventieth" => 70,
        "eightieth" => 80,
        "ninetieth" => 90,
        "hundredth" => 100,
        "thousandth" => 1000,
        _ => return None,
    };
    Some((v, ordinal_suffix(v)))
}

fn ordinal_suffix(v: u64) -> &'static str {
    match (v % 100, v % 10) {
        (11..=13, _) => "th",
        (_, 1) => "st",
        (_, 2) => "nd",
        (_, 3) => "rd",
        _ => "th",
    }
}

/// Convert spelled-out cardinals and ordinals to digits.
///
/// Composition rules mirror Whisper's behaviour on the cases that matter for
/// read speech: a unit after a tens word combines (`twenty three` → `23`), a
/// tens word after a partial number starts a new one (`eighteen seventy six`
/// → `18 76`, the year form), `hundred` multiplies, and `thousand`/`million`
/// scale. `and` is absorbed only when it sits inside a number.
fn words_to_digits(text: &str) -> String {
    /// The number being assembled: `total` holds completed scale groups
    /// ("two thousand"), `part` the 0–999 group under construction.
    #[derive(Default)]
    struct Acc {
        total: u64,
        part: u64,
        active: bool,
    }

    impl Acc {
        fn value(&self) -> u64 {
            self.total + self.part
        }

        /// Emit the pending number, if any, and reset.
        fn flush(&mut self, out: &mut Vec<String>) {
            if self.active {
                out.push(self.value().to_string());
                *self = Acc::default();
            }
        }
    }

    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut acc = Acc::default();

    for (i, token) in tokens.iter().enumerate() {
        let token = *token;
        if let Some(v) = unit_value(token) {
            // A unit after another unit/teen starts a new number rather than
            // summing: "six six" is two numbers, "twenty six" is one.
            if acc.active && acc.part % 10 != 0 {
                acc.flush(&mut out);
            }
            acc.part += v;
            acc.active = true;
        } else if let Some(v) = tens_value(token) {
            if acc.active && acc.part != 0 {
                acc.flush(&mut out);
            }
            acc.part += v;
            acc.active = true;
        } else if token == "hundred" && acc.active {
            acc.part = acc.part.max(1) * 100;
        } else if let Some(scale) = scale_value(token) {
            if acc.active {
                acc.total += acc.part.max(1) * scale;
                acc.part = 0;
            } else {
                out.push(token.to_string());
            }
        } else if token == "and"
            && acc.active
            && tokens
                .get(i + 1)
                .is_some_and(|n| unit_value(n).is_some() || tens_value(n).is_some())
        {
            // absorbed: "two hundred and five"
        } else if let Some((v, _)) = ordinal_value(token) {
            // An ordinal terminates the number it completes: "twenty third"
            // is 23rd, not 20 followed by 3rd.
            let value = if !acc.active {
                v
            } else if v >= 100 {
                // scale ordinal: "two hundredth" -> 200th
                acc.total + acc.part.max(1) * v
            } else if acc.part % 10 == 0 {
                acc.value() + v
            } else {
                // no valid composition ("six third"): emit the pending
                // number on its own, then the ordinal.
                out.push(acc.value().to_string());
                v
            };
            acc = Acc::default();
            out.push(format!("{value}{}", ordinal_suffix(value)));
        } else {
            acc.flush(&mut out);
            out.push(token.to_string());
        }
    }
    acc.flush(&mut out);
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn en(s: &str) -> String {
        english(s)
    }

    #[test]
    fn librispeech_reference_and_whisper_output_converge() {
        // The case that motivates the whole module: same sentence, one from a
        // LibriSpeech .trans.txt, one as Whisper would emit it.
        let reference = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL";
        let hypothesis = "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.";
        assert_eq!(en(reference), en(hypothesis));
    }

    #[test]
    fn titles_expand() {
        assert_eq!(
            en("Dr. Smith and Mrs. Jones"),
            "doctor smith and missus jones"
        );
    }

    #[test]
    fn contractions_expand_consistently() {
        assert_eq!(en("it's"), en("it is"));
        assert_eq!(en("won't"), "will not");
        assert_eq!(en("they've"), "they have");
    }

    #[test]
    fn fillers_are_dropped() {
        assert_eq!(en("um so uh yes"), "so yes");
    }

    #[test]
    fn spelled_numbers_match_digits() {
        assert_eq!(en("twenty three"), en("23"));
        assert_eq!(en("one hundred and five"), en("105"));
        assert_eq!(en("two thousand"), en("2000"));
        assert_eq!(en("twenty three thousand four hundred"), en("23400"));
    }

    #[test]
    fn year_pairs_split_rather_than_summing() {
        // "eighteen seventy six" must not become 94.
        assert_eq!(en("eighteen seventy six"), "18 76");
        assert_eq!(en("six six"), "6 6");
    }

    #[test]
    fn ordinals_become_suffixed_digits() {
        assert_eq!(en("the first day"), "the 1st day");
        assert_eq!(en("the twentieth"), "the 20th");
        // The composition case: an ordinal completes the pending number.
        assert_eq!(en("the twenty third"), "the 23rd");
        assert_eq!(en("one hundredth"), "100th");
        assert_eq!(en("the twenty third of may"), en("the 23rd of may"));
    }

    #[test]
    fn numbers_survive_surrounding_words() {
        assert_eq!(
            en("he had twenty three apples and left"),
            "he had 23 apples and left"
        );
    }

    #[test]
    fn scale_word_alone_is_left_as_a_word() {
        assert_eq!(en("thousands of people"), "thousands of people");
        assert_eq!(en("million"), "million");
    }

    #[test]
    fn digit_commas_and_trailing_periods() {
        assert_eq!(en("1,234 apples."), "1234 apples");
    }

    #[test]
    fn basic_mode_is_language_agnostic() {
        assert_eq!(basic("Hello, [noise] World! (aside)"), "hello world");
    }

    #[test]
    fn normalization_is_idempotent() {
        let once = en("Mr. Smith had twenty three apples, and he won't share.");
        assert_eq!(en(&once), once);
    }
}
