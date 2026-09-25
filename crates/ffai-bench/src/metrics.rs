//! Task metrics: WER and CER (the ASR/OCR standards).
//!
//! **Caption metrics deliberately do not live here.** Argus is live, and it
//! scores through the benchmark's OWN evaluator as an external process — see
//! `crate::vlm` and the `[[scorer]]` contract. Answer extraction is part of a
//! VLM metric, so a caption metric written here would be a scorer we grade
//! ourselves against, which is exactly the self-favouring comparison the
//! Carmenta campaign paid a year for.
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

/// Digit errors and reference digits, for a DIGIT ERROR RATE — the number a
/// financial-forms buyer asks for first (docs/plans/commercial-gaps.md,
/// gap 3). A page CER of 5 % says nothing about whether "$150,000" came back
/// right, and one wrong digit makes the value worthless.
///
/// Both strings are normalized under `mode` and aligned by a minimal-edit
/// Levenshtein alignment. Counted as digit errors:
/// * a substitution where EITHER side is a digit — a digit misread
///   (`1` -> `T`) and a phantom digit (`$` -> `8`, which turns "$10,000"
///   into "810,000") both corrupt the number;
/// * a reference digit deleted;
/// * a hypothesis digit inserted where the reference had nothing.
///
/// Returns `(errors, reference_digits)`, so a corpus total is summed
/// (micro-averaged) rather than a mean of per-page rates dominated by pages
/// with two digits.
#[must_use]
pub fn digit_errors_with(reference: &str, hypothesis: &str, mode: Mode) -> (usize, usize) {
    let truth: Vec<char> = normalize(reference, mode).chars().collect();
    let read: Vec<char> = normalize(hypothesis, mode).chars().collect();
    let digits = truth.iter().filter(|c| c.is_ascii_digit()).count();
    // Full DP table, then walk back one minimal alignment.
    let mut table = vec![vec![0usize; read.len() + 1]; truth.len() + 1];
    for (row_idx, row) in table.iter_mut().enumerate() {
        row[0] = row_idx;
    }
    for (col_idx, cell) in table[0].iter_mut().enumerate() {
        *cell = col_idx;
    }
    for ti in 1..=truth.len() {
        for ri in 1..=read.len() {
            table[ti][ri] = (table[ti - 1][ri - 1] + usize::from(truth[ti - 1] != read[ri - 1]))
                .min(table[ti - 1][ri] + 1)
                .min(table[ti][ri - 1] + 1);
        }
    }
    let (mut ti, mut ri, mut errors) = (truth.len(), read.len(), 0usize);
    while ti > 0 || ri > 0 {
        let diagonal = ti > 0
            && ri > 0
            && table[ti][ri] == table[ti - 1][ri - 1] + usize::from(truth[ti - 1] != read[ri - 1]);
        if diagonal {
            let (t, r) = (truth[ti - 1], read[ri - 1]);
            errors += usize::from(t != r && (t.is_ascii_digit() || r.is_ascii_digit()));
            ti -= 1;
            ri -= 1;
        } else if ti > 0 && table[ti][ri] == table[ti - 1][ri] + 1 {
            errors += usize::from(truth[ti - 1].is_ascii_digit()); // deleted
            ti -= 1;
        } else {
            errors += usize::from(read[ri - 1].is_ascii_digit()); // inserted
            ri -= 1;
        }
    }
    (errors, digits)
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

    /// The field misreads that motivated the metric: every one is a digit
    /// error, and letter errors elsewhere on the line are not.
    #[test]
    fn digit_errors_count_only_digits() {
        let m = Mode::Ocr;
        assert_eq!(
            digit_errors_with("$10,000", "810,000", m),
            (1, 5),
            "$ -> 8 invents a digit"
        );
        assert_eq!(digit_errors_with("1117 Oak", "TTT7 Oak", m), (3, 4));
        assert_eq!(
            digit_errors_with("#305 Main", "#30S Mian", m),
            (1, 3),
            "a letter swap is not counted"
        );
        assert_eq!(
            digit_errors_with("Unit 7", "Unit 77", m),
            (1, 1),
            "an invented digit counts"
        );
        assert_eq!(
            digit_errors_with("Total 42", "Total", m),
            (2, 2),
            "dropped digits count"
        );
        assert_eq!(digit_errors_with("no digits", "n0 digits", m), (1, 0));
    }

    #[test]
    fn empty_reference_with_output_is_penalized() {
        assert!(wer("", "spurious words") > 0.0);
        assert_eq!(wer("", ""), 0.0);
    }
}
