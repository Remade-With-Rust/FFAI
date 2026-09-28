//! Snap a read to a caller's lexicon (docs/plans/commercial-gaps.md, gap 12).
//!
//! A form field often has a known vocabulary — the state's counties, the
//! banks a filer has used before, the form's own printed choices. A caller
//! reading "Trish HTlI" where the lexicon holds "Irish Hill" should be told
//! so, with the evidence, rather than storing the misread or discarding the
//! line.
//!
//! The distance is Levenshtein with one change: substituting one member of a
//! LOOK-ALIKE class for another costs half an edit. The classes are the
//! confusions measured on real filings and on `carmenta-form-v1`:
//! `1 l I i | T`, `0 O o D U Q`, `5 S s`, `8 B $`, `2 Z z`, `6 G b`, `9 g q`,
//! `C G`, `c e`, `, .`. So "112l" is closer to "1121" than any word that
//! differs by a real letter.
//!
//! A snap is returned only when it is UNAMBIGUOUS: within `max_distance`, and
//! the runner-up is at least `margin` further away. Otherwise the caller gets
//! the candidates and decides. Nothing here changes what the engine read; it
//! is advice with its evidence attached.

/// Look-alike classes: substituting within one costs half an edit.
const LOOK_ALIKES: &[&str] = &[
    "1lIi|T", "0OoDUQ", "5Ss", "8B$", "2Zz", "6Gb", "9gq", "CG", "ce", ",.",
];

fn substitution_cost(a: char, b: char) -> f32 {
    if a == b {
        0.0
    } else if a.eq_ignore_ascii_case(&b)
        || LOOK_ALIKES
            .iter()
            .any(|class| class.contains(a) && class.contains(b))
    {
        0.5
    } else {
        1.0
    }
}

/// Edit distance with look-alike substitutions at half cost. Insertions and
/// deletions cost 1, so a dropped or invented character is never cheap.
#[must_use]
pub fn distance(a: &str, b: &str) -> f32 {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<f32> = (0..=b.len()).map(|j| j as f32).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut cur = vec![(i + 1) as f32; b.len() + 1];
        for (j, &cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + substitution_cost(ca, cb))
                .min(prev[j + 1] + 1.0)
                .min(cur[j] + 1.0);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// The outcome of [`Lexicon::snap`].
#[derive(Debug, Clone, PartialEq)]
pub enum Snap {
    /// The read is already an entry, exactly.
    Exact(String),
    /// One entry is clearly closest.
    Snapped { entry: String, distance: f32 },
    /// Nothing close enough, or two entries too close to call. The nearest
    /// candidates, closest first, for the caller to judge.
    Ambiguous(Vec<(String, f32)>),
}

/// A caller-supplied vocabulary for one field.
#[derive(Debug, Clone)]
pub struct Lexicon {
    entries: Vec<String>,
    /// Largest distance, as a fraction of the entry's length, that may snap.
    pub max_distance: f32,
    /// How much further the runner-up must be for a snap to be unambiguous.
    pub margin: f32,
}

impl Lexicon {
    /// A lexicon with conservative defaults: snap within 25 % of the entry's
    /// length, and only when the runner-up is at least 1.0 edit further.
    #[must_use]
    pub fn new(entries: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            entries: entries.into_iter().map(Into::into).collect(),
            max_distance: 0.25,
            margin: 1.0,
        }
    }

    /// Match `read` against the lexicon; see [`Snap`].
    #[must_use]
    pub fn snap(&self, read: &str) -> Snap {
        let read = read.trim();
        if let Some(e) = self.entries.iter().find(|e| e.as_str() == read) {
            return Snap::Exact(e.clone());
        }
        let mut scored: Vec<(String, f32)> = self
            .entries
            .iter()
            .map(|e| (e.clone(), distance(read, e)))
            .collect();
        scored.sort_by(|x, y| x.1.total_cmp(&y.1));
        let Some((best, d)) = scored.first().cloned() else {
            return Snap::Ambiguous(Vec::new());
        };
        let limit = self.max_distance * best.chars().count().max(1) as f32;
        let runner_up = scored.get(1).map_or(f32::INFINITY, |s| s.1);
        if d <= limit && runner_up - d >= self.margin {
            Snap::Snapped {
                entry: best,
                distance: d,
            }
        } else {
            scored.truncate(3);
            Snap::Ambiguous(scored)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_alike_substitutions_cost_half() {
        assert!((distance("112l", "1121") - 0.5).abs() < 1e-6);
        assert!((distance("TTT7", "1117") - 1.5).abs() < 1e-6);
        assert!((distance("810,000", "$10,000") - 0.5).abs() < 1e-6);
        assert!(
            (distance("cat", "cut") - 1.0).abs() < 1e-6,
            "a real letter change costs a full edit"
        );
        assert!(
            (distance("abc", "abcd") - 1.0).abs() < 1e-6,
            "insertions are never cheap"
        );
    }

    /// The misreads recorded on real filings snap to the right entries.
    #[test]
    fn field_misreads_snap_to_their_entries() {
        let lex = Lexicon::new([
            "Irish Hill",
            "Gulf Shores",
            "Merrill Lynch",
            "Cabell County",
            "Tri-State Bank",
        ]);
        for (read, want) in [
            ("Trish HTlI", "Irish Hill"),
            ("GuIF Shores", "Gulf Shores"),
            ("MerrITLynch", "Merrill Lynch"),
            ("Cabe]l County", "Cabell County"),
            ("Tr1-State Bank", "Tri-State Bank"),
        ] {
            match lex.snap(read) {
                Snap::Snapped { entry, .. } => assert_eq!(entry, want, "{read}"),
                other => panic!("{read}: expected a snap to {want}, got {other:?}"),
            }
        }
    }

    #[test]
    fn exact_reads_and_far_reads_are_reported_as_such() {
        let lex = Lexicon::new(["Irish Hill", "Gulf Shores"]);
        assert_eq!(lex.snap(" Irish Hill "), Snap::Exact("Irish Hill".into()));
        assert!(matches!(
            lex.snap("Something Else Entirely"),
            Snap::Ambiguous(_)
        ));
    }

    /// Two entries equally close must not be guessed between.
    #[test]
    fn a_close_call_is_ambiguous_not_a_guess() {
        let lex = Lexicon::new(["Unit 10", "Unit 1O"]);
        assert!(matches!(lex.snap("Unit 1o"), Snap::Ambiguous(_)));
    }
}
