//! Diarization Error Rate — the metric that gates `--diarize`.
//!
//! DER is the sum of three failures over total reference speech:
//!
//! ```text
//!            missed speech + false alarm + speaker confusion
//!    DER  =  ───────────────────────────────────────────────
//!                      total reference speech
//! ```
//!
//! - **missed** — reference speech the system said was silence.
//! - **false alarm** — silence the system said was speech.
//! - **confusion** — both agree someone spoke, and they disagree on who.
//!
//! **The label-mapping problem is the whole difficulty.** A system that
//! perfectly separates two speakers but calls them `SPEAKER_01` and
//! `SPEAKER_00` where the reference says `A` and `B` has made **zero**
//! errors — the labels are arbitrary names for clusters, not identities. So
//! DER is defined over the *optimal* one-to-one mapping between hypothesis
//! and reference labels, and computing anything else silently reports a
//! perfect system as a failed one.
//!
//! **Collar.** Human-annotated boundaries are imprecise to a few hundred
//! milliseconds, so the convention (NIST RT, and what pyannote reports by
//! default) is to score with a forgiveness collar around every reference
//! boundary. `0.25` is standard; `0.0` is the harsher "full" DER. Both are
//! reported rather than one being chosen, because the two answer different
//! questions and quoting a collared number as if it were the full one is a
//! way to look better than you are.
//!
//! DER can exceed 1.0. A system that emits speech everywhere on a mostly
//! silent recording accrues false alarm without bound, and clamping that to
//! 100 % would hide exactly the failure worth seeing.

/// One labelled stretch of audio, from either the reference or a system.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

impl Turn {
    pub fn new(start: f64, end: f64, speaker: impl Into<String>) -> Self {
        Self {
            start,
            end,
            speaker: speaker.into(),
        }
    }

    // Unused today: DER is computed from overlaps rather than per-turn
    // durations. Kept because it is the obvious accessor and every future
    // metric that weights by turn length wants exactly this, clamped at zero.
    #[allow(dead_code)]
    fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// The three error components, in seconds, plus the totals they divide into.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerBreakdown {
    pub missed_secs: f64,
    pub false_alarm_secs: f64,
    pub confusion_secs: f64,
    pub reference_secs: f64,
    /// Collar applied on each side of every reference boundary.
    pub collar: f64,
}

impl DerBreakdown {
    /// The headline number. `None` when the reference contains no speech —
    /// dividing by zero would report 0.0, which reads as a perfect score for
    /// a recording nobody could have got wrong.
    #[must_use]
    pub fn der(&self) -> Option<f64> {
        if self.reference_secs <= 0.0 {
            return None;
        }
        Some((self.missed_secs + self.false_alarm_secs + self.confusion_secs) / self.reference_secs)
    }
}

/// Cut points where either side changes state, so every resulting interval is
/// homogeneous — one reference speaker (or none) and one hypothesis speaker
/// (or none) throughout.
fn boundaries(reference: &[Turn], hypothesis: &[Turn]) -> Vec<f64> {
    let mut points: Vec<f64> = reference
        .iter()
        .chain(hypothesis.iter())
        .flat_map(|t| [t.start, t.end])
        .collect();
    points.sort_by(f64::total_cmp);
    points.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    points
}

fn speaker_at(turns: &[Turn], start: f64, end: f64) -> Option<&str> {
    let mid = (start + end) / 2.0;
    turns
        .iter()
        .find(|t| t.start <= mid && mid < t.end)
        .map(|t| t.speaker.as_str())
}

/// Regions excluded from scoring: within `collar` of any reference boundary.
fn in_collar(collared: &[(f64, f64)], start: f64, end: f64) -> bool {
    let mid = (start + end) / 2.0;
    collared.iter().any(|(a, b)| *a <= mid && mid < *b)
}

fn distinct(turns: &[Turn]) -> Vec<&str> {
    let mut names: Vec<&str> = turns.iter().map(|t| t.speaker.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// DER under the mapping that minimises it.
///
/// The mapping is found by maximising total overlap between mapped pairs.
/// With few speakers every permutation is tried, which is exact; past
/// [`MAX_EXACT_SPEAKERS`] it falls back to greedy assignment and **says so**
/// in the returned flag rather than quietly reporting an approximation as a
/// measurement.
#[must_use]
pub fn diarization_error_rate(
    reference: &[Turn],
    hypothesis: &[Turn],
    collar: f64,
) -> (DerBreakdown, bool) {
    let ref_names = distinct(reference);
    let hyp_names = distinct(hypothesis);

    // Collar regions around every reference boundary.
    let collared: Vec<(f64, f64)> = if collar > 0.0 {
        reference
            .iter()
            .flat_map(|t| {
                [
                    (t.start - collar, t.start + collar),
                    (t.end - collar, t.end + collar),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };

    let points = boundaries(reference, hypothesis);
    // (reference speaker index, hypothesis speaker index) -> overlap seconds
    let mut overlap = vec![0.0f64; ref_names.len().max(1) * hyp_names.len().max(1)];
    let mut scored: Vec<(f64, f64, Option<usize>, Option<usize>)> = Vec::new();

    for pair in points.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if b - a <= 1e-12 || in_collar(&collared, a, b) {
            continue;
        }
        let r = speaker_at(reference, a, b).and_then(|s| ref_names.iter().position(|n| *n == s));
        let h = speaker_at(hypothesis, a, b).and_then(|s| hyp_names.iter().position(|n| *n == s));
        if let (Some(ri), Some(hi)) = (r, h) {
            overlap[ri * hyp_names.len() + hi] += b - a;
        }
        scored.push((a, b, r, h));
    }

    // Best one-to-one mapping from reference speakers to hypothesis speakers.
    let (mapping, exact) = best_mapping(&overlap, ref_names.len(), hyp_names.len());

    let mut missed = 0.0;
    let mut false_alarm = 0.0;
    let mut confusion = 0.0;
    let mut reference_secs = 0.0;
    for (a, b, r, h) in scored {
        let dur = b - a;
        match (r, h) {
            (Some(ri), Some(hi)) => {
                reference_secs += dur;
                if mapping.get(ri).copied().flatten() != Some(hi) {
                    confusion += dur;
                }
            }
            (Some(_), None) => {
                reference_secs += dur;
                missed += dur;
            }
            (None, Some(_)) => false_alarm += dur,
            (None, None) => {}
        }
    }

    (
        DerBreakdown {
            missed_secs: missed,
            false_alarm_secs: false_alarm,
            confusion_secs: confusion,
            reference_secs,
            collar,
        },
        exact,
    )
}

/// Above this many speakers on either side, exhaustive search is abandoned.
/// 8! = 40320 mappings is instant; 12! is not.
pub const MAX_EXACT_SPEAKERS: usize = 8;

/// Returns the mapping and whether it is provably optimal.
fn best_mapping(overlap: &[f64], n_ref: usize, n_hyp: usize) -> (Vec<Option<usize>>, bool) {
    if n_ref == 0 || n_hyp == 0 {
        return (vec![None; n_ref], true);
    }
    if n_ref <= MAX_EXACT_SPEAKERS && n_hyp <= MAX_EXACT_SPEAKERS {
        let mut best: Option<(f64, Vec<Option<usize>>)> = None;
        let mut current = vec![None; n_ref];
        let mut used = vec![false; n_hyp];
        permute(
            overlap,
            n_hyp,
            0,
            n_ref,
            &mut current,
            &mut used,
            0.0,
            &mut best,
        );
        return (best.map_or_else(|| vec![None; n_ref], |(_, m)| m), true);
    }
    // Greedy: take the largest remaining overlap until none is left.
    let mut mapping = vec![None; n_ref];
    let mut used = vec![false; n_hyp];
    let mut pairs: Vec<(f64, usize, usize)> = (0..n_ref)
        .flat_map(|r| (0..n_hyp).map(move |h| (r, h)))
        .map(|(r, h)| (overlap[r * n_hyp + h], r, h))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (score, r, h) in pairs {
        if score > 0.0 && mapping[r].is_none() && !used[h] {
            mapping[r] = Some(h);
            used[h] = true;
        }
    }
    (mapping, false)
}

#[allow(clippy::too_many_arguments)]
fn permute(
    overlap: &[f64],
    n_hyp: usize,
    idx: usize,
    n_ref: usize,
    current: &mut Vec<Option<usize>>,
    used: &mut Vec<bool>,
    score: f64,
    best: &mut Option<(f64, Vec<Option<usize>>)>,
) {
    if idx == n_ref {
        if best.as_ref().is_none_or(|(s, _)| score > *s) {
            *best = Some((score, current.clone()));
        }
        return;
    }
    // Leaving a reference speaker unmapped is legal — there may be fewer
    // hypothesis clusters than real speakers.
    current[idx] = None;
    permute(overlap, n_hyp, idx + 1, n_ref, current, used, score, best);
    for h in 0..n_hyp {
        if used[h] {
            continue;
        }
        used[h] = true;
        current[idx] = Some(h);
        permute(
            overlap,
            n_hyp,
            idx + 1,
            n_ref,
            current,
            used,
            score + overlap[idx * n_hyp + h],
            best,
        );
        used[h] = false;
    }
    current[idx] = None;
}

/// Parse NIST RTTM, the standard interchange format for diarization ground
/// truth. Only `SPEAKER` lines are read; anything else is ignored.
///
/// `SPEAKER <file> <chan> <start> <dur> <ortho> <stype> <name> <conf> <slat>`
#[must_use]
pub fn parse_rttm(text: &str) -> Vec<Turn> {
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 8 || f[0] != "SPEAKER" {
                return None;
            }
            let start: f64 = f[3].parse().ok()?;
            let dur: f64 = f[4].parse().ok()?;
            Some(Turn::new(start, start + dur, f[7]))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn der_of(reference: &[Turn], hypothesis: &[Turn], collar: f64) -> f64 {
        diarization_error_rate(reference, hypothesis, collar)
            .0
            .der()
            .expect("reference has speech")
    }

    #[test]
    fn a_perfect_system_scores_zero() {
        let r = vec![Turn::new(0.0, 5.0, "A"), Turn::new(6.0, 10.0, "B")];
        assert!(der_of(&r, &r, 0.0) < 1e-9);
    }

    #[test]
    fn relabelling_is_not_an_error() {
        // THE case the metric exists to handle: same partition, different
        // names. A naive string comparison would call this 100 % wrong.
        let r = vec![Turn::new(0.0, 5.0, "A"), Turn::new(6.0, 10.0, "B")];
        let h = vec![
            Turn::new(0.0, 5.0, "SPEAKER_01"),
            Turn::new(6.0, 10.0, "SPEAKER_00"),
        ];
        assert!(der_of(&r, &h, 0.0) < 1e-9, "relabelling scored as error");
    }

    #[test]
    fn merging_two_speakers_into_one_is_confusion() {
        // 9 s of speech, 4 s of it given the wrong identity.
        let r = vec![Turn::new(0.0, 5.0, "A"), Turn::new(5.0, 9.0, "B")];
        let h = vec![Turn::new(0.0, 9.0, "SPEAKER_00")];
        let der = der_of(&r, &h, 0.0);
        assert!((der - 4.0 / 9.0).abs() < 1e-6, "{der}");
    }

    #[test]
    fn missing_speech_counts_as_missed() {
        let r = vec![Turn::new(0.0, 10.0, "A")];
        let h = vec![Turn::new(0.0, 6.0, "SPEAKER_00")];
        assert!((der_of(&r, &h, 0.0) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn speech_invented_in_silence_is_false_alarm() {
        let r = vec![Turn::new(0.0, 10.0, "A")];
        let h = vec![
            Turn::new(0.0, 10.0, "SPEAKER_00"),
            Turn::new(12.0, 15.0, "SPEAKER_00"),
        ];
        // 3 s invented against 10 s of reference speech.
        assert!((der_of(&r, &h, 0.0) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn der_may_exceed_one_and_is_not_clamped() {
        // Hallucinating speech across a mostly-silent hour must be visible,
        // not flattened to "100 % wrong".
        let r = vec![Turn::new(0.0, 1.0, "A")];
        let h = vec![Turn::new(0.0, 20.0, "SPEAKER_00")];
        assert!(der_of(&r, &h, 0.0) > 1.0);
    }

    #[test]
    fn a_collar_forgives_boundary_imprecision() {
        // Hypothesis is 0.1 s late starting and 0.1 s early ending — inside
        // any reasonable annotation tolerance.
        let r = vec![Turn::new(1.0, 9.0, "A")];
        let h = vec![Turn::new(1.1, 8.9, "SPEAKER_00")];
        let strict = der_of(&r, &h, 0.0);
        let collared = der_of(&r, &h, 0.25);
        assert!(strict > 0.0, "full DER should see the slip");
        assert!(collared < 1e-9, "collar should forgive it, got {collared}");
    }

    #[test]
    fn empty_reference_reports_none_not_zero() {
        let (b, _) = diarization_error_rate(&[], &[Turn::new(0.0, 1.0, "X")], 0.0);
        assert!(
            b.der().is_none(),
            "no reference speech must not read as perfect"
        );
    }

    #[test]
    fn empty_hypothesis_misses_everything() {
        let r = vec![Turn::new(0.0, 4.0, "A")];
        assert!((der_of(&r, &[], 0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn optimal_mapping_beats_the_greedy_order() {
        // Reference A overlaps h0 slightly more than h1, but taking it greedily
        // forces B onto a much worse partner. The exhaustive search must find
        // the assignment with the larger TOTAL overlap.
        let r = vec![Turn::new(0.0, 10.0, "A"), Turn::new(10.0, 20.0, "B")];
        let h = vec![
            Turn::new(0.0, 6.0, "h0"),
            Turn::new(6.0, 10.0, "h1"),
            Turn::new(10.0, 20.0, "h0"),
        ];
        let (b, exact) = diarization_error_rate(&r, &h, 0.0);
        assert!(exact, "should have used exhaustive search");
        // Whatever mapping wins, DER must be the minimum achievable.
        assert!(b.der().expect("has speech") <= 0.5 + 1e-9);
    }

    #[test]
    fn breakdown_components_sum_to_the_rate() {
        let r = vec![Turn::new(0.0, 5.0, "A"), Turn::new(5.0, 10.0, "B")];
        let h = vec![
            Turn::new(0.0, 8.0, "SPEAKER_00"),
            Turn::new(11.0, 12.0, "SPEAKER_01"),
        ];
        let (b, _) = diarization_error_rate(&r, &h, 0.0);
        let sum = b.missed_secs + b.false_alarm_secs + b.confusion_secs;
        assert!((b.der().expect("speech") - sum / b.reference_secs).abs() < 1e-9);
    }

    #[test]
    fn rttm_round_trips_the_standard_layout() {
        let text = "\
SPEAKER meeting 1 0.00 5.25 <NA> <NA> alice <NA> <NA>
SPEAKER meeting 1 5.25 3.75 <NA> <NA> bob <NA> <NA>
# a comment line that is not a SPEAKER record
SPKR-INFO meeting 1 <NA> <NA> <NA> unknown alice <NA> <NA>";
        let turns = parse_rttm(text);
        assert_eq!(turns.len(), 2, "{turns:?}");
        assert_eq!(turns[0], Turn::new(0.0, 5.25, "alice"));
        // Duration, not end time — the classic RTTM misreading.
        assert_eq!(turns[1], Turn::new(5.25, 9.0, "bob"));
    }
}
