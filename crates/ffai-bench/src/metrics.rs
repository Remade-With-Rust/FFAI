//! Task metrics. Phase 0: WER and CER (the ASR/OCR standards). Caption
//! metrics (CIDEr-class) arrive with Argus in Phase 4.
//!
//! Both sides of every comparison pass through [`crate::normalize`] under the
//! same [`Mode`], so formatting differences (`Mr.` vs `MISTER`, `23` vs
//! `twenty three`) don't masquerade as recognition errors.

use crate::normalize::{Mode, normalize};

/// Word error rate under the default (English) normalizer.
#[must_use]
pub fn wer(reference: &str, hypothesis: &str) -> f64 {
    wer_with(reference, hypothesis, Mode::default())
}

/// Character error rate under the default (English) normalizer.
#[must_use]
pub fn cer(reference: &str, hypothesis: &str) -> f64 {
    cer_with(reference, hypothesis, Mode::default())
}

/// Word error rate: Levenshtein distance over words / reference word count.
#[must_use]
pub fn wer_with(reference: &str, hypothesis: &str, mode: Mode) -> f64 {
    let r = normalize(reference, mode);
    let h = normalize(hypothesis, mode);
    let r: Vec<&str> = r.split_whitespace().collect();
    let h: Vec<&str> = h.split_whitespace().collect();
    error_rate(&r, &h)
}

/// Character error rate: Levenshtein distance over characters (spaces
/// included after normalization) / reference length.
#[must_use]
pub fn cer_with(reference: &str, hypothesis: &str, mode: Mode) -> f64 {
    let r = normalize(reference, mode);
    let h = normalize(hypothesis, mode);
    let r: Vec<char> = r.chars().collect();
    let h: Vec<char> = h.chars().collect();
    error_rate(&r, &h)
}

/// Levenshtein distance / reference length. Empty reference: 0.0 if the
/// hypothesis is empty too, else 1.0 per hypothesis token (capped at 1.0 by
/// convention would hide gross over-generation, so we don't cap).
fn error_rate<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> f64 {
    if reference.is_empty() {
        return if hypothesis.is_empty() {
            0.0
        } else {
            hypothesis.len() as f64
        };
    }
    levenshtein(reference, hypothesis) as f64 / reference.len() as f64
}

/// Two-row dynamic-programming Levenshtein — `O(len_a` × `len_b`) time,
/// `O(len_b)` space, no allocation in the inner loop.
fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ai) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bj) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ai != bj);
            curr[j + 1] = sub.min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formatting_differences_are_not_errors() {
        // The whole point of normalizing before scoring.
        let reference = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES";
        let hypothesis = "Mr. Quilter is the apostle of the middle classes.";
        assert_eq!(wer(reference, hypothesis), 0.0);
        // ...and without normalization, they very much are.
        assert!(wer_with(reference, hypothesis, Mode::None) > 0.0);
    }

    #[test]
    fn wer_exact_match_is_zero() {
        assert_eq!(wer("the cat sat", "The cat sat!"), 0.0);
    }

    #[test]
    fn wer_counts_substitutions_insertions_deletions() {
        // ref: 4 words; hyp: one substitution ("dog") + one deletion ("mat")
        let w = wer("the cat sat down", "the dog sat");
        assert!((w - 0.5).abs() < 1e-12, "got {w}");
    }

    #[test]
    fn cer_on_close_strings_is_small() {
        let c = cer("kitten", "sitten");
        assert!((c - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn empty_reference_with_output_is_penalized() {
        assert!(wer("", "spurious words") > 0.0);
        assert_eq!(wer("", ""), 0.0);
    }
}
