//! Long-form input: sentence segmentation for synthesis.
//!
//! VITS synthesizes one sentence at a time (its duration model was trained
//! on sentence-scale prosody); long-form input is split at sentence
//! boundaries, synthesized per sentence, and joined with a configurable
//! silence gap — the mission plan's `chunk.rs` stage, engine-agnostic.
//!
//! v1 boundary rule, stated honestly: split after `.`, `?`, `!`, `;` when
//! followed by whitespace and an uppercase/quote/digit start. Abbreviations
//! (`Dr.`, `e.g.`) are protected by a small list; a mid-sentence period in
//! unusual text will over-split, which costs a pause, not words.

/// Split text into synthesizable sentences (with their punctuation).
#[must_use]
pub fn sentences(text: &str) -> Vec<String> {
    // Single-token forms only: the lookback collects the alphabetic run
    // before the period, so `e.g.` / `i.e.` surface as their last letter.
    const ABBREV: &[&str] = &[
        "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "g", "e", "no",
    ];
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if matches!(c, '.' | '?' | '!' | ';') {
            // Look back for an abbreviation before a period.
            let is_abbrev = c == '.' && {
                let mut w_start = i;
                while w_start > start && chars[w_start - 1].is_ascii_alphabetic() {
                    w_start -= 1;
                }
                let word: String = chars[w_start..i].iter().collect::<String>().to_lowercase();
                ABBREV.contains(&word.as_str())
            };
            // A boundary needs following whitespace then a sentence-ish start.
            let next_non_ws = chars[i + 1..].iter().position(|c| !c.is_whitespace());
            let is_boundary = match next_non_ws {
                None => true, // end of text
                Some(rel) => {
                    let has_ws = rel > 0 || i + 1 == chars.len();
                    let nc = chars[i + 1 + rel];
                    has_ws && (nc.is_uppercase() || nc.is_ascii_digit() || nc == '"' || nc == '\'')
                }
            };
            if is_boundary && !is_abbrev {
                let sentence: String = chars[start..=i].iter().collect();
                let sentence = sentence.trim();
                if !sentence.is_empty() {
                    out.push(sentence.to_string());
                }
                start = i + 1;
            }
        }
        i += 1;
    }
    let tail: String = chars[start..].iter().collect();
    let tail = tail.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences_and_protects_abbreviations() {
        let text = "The birch canoe slid. Dr. Smith spoke! Was it loud? Yes; quite loud.";
        // `Yes; quite` does NOT split: the boundary rule requires an
        // uppercase/quote/digit start, so a lowercase continuation stays in
        // its sentence (better prosody than a hard gap).
        assert_eq!(
            sentences(text),
            vec![
                "The birch canoe slid.",
                "Dr. Smith spoke!",
                "Was it loud?",
                "Yes; quite loud."
            ]
        );
    }

    #[test]
    fn single_sentence_and_no_trailing_punct_pass_through() {
        assert_eq!(sentences("hello world"), vec!["hello world"]);
        assert_eq!(sentences("One. two three"), vec!["One. two three"]);
        assert_eq!(sentences(""), Vec::<String>::new());
    }
}
