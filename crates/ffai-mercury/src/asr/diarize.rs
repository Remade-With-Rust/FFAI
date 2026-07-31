//! Speaker diarization — who spoke when.
//!
//! The pipeline is four stages, and **only the third needs a neural model**:
//!
//! 1. [`subsegment`] — cut speech regions into short overlapping windows.
//! 2. embed each window (an acoustic model: [`super::speaker`]).
//! 3. [`cluster`] — group windows by voice.
//! 4. [`turns_from_labels`] then [`assign`] — turn labels into speaker turns
//!    and attach them to transcript segments.
//!
//! Stages 1, 3 and 4 live here and are model-free. That is deliberate and it
//! is the same split that paid off in [`super::align`]: the algorithm can be
//! written, tested and argued about against synthetic embeddings before the
//! expensive port exists, and a bug in the port cannot hide inside the
//! clustering.
//!
//! **Licence note, because it shaped the design.** The obvious embedding model
//! is pyannote's. It is MIT-licensed and **gated** — a licence that permits
//! use sitting behind an acceptance wall that prevents fetching, which cannot
//! go in a manifest under principle 4 ("weights are data, fetched from
//! manifests that surface each model's own licence"). SpeechBrain's
//! ECAPA-TDNN is Apache-2.0 and ungated, so that is what
//! [`super::speaker`] targets.

use ffai_core::types::TimedSegment;

/// Window length for embedding, in seconds. Long enough for a voice to be
/// characterised, short enough that a window rarely straddles a speaker
/// change — the two failure modes pull in opposite directions and 1.5 s is
/// the usual compromise.
pub const WINDOW_SECS: f64 = 1.5;

/// Hop between windows. Half the window, so a speaker change is at worst
/// half a window from a boundary.
pub const HOP_SECS: f64 = 0.75;

/// Cosine-distance threshold for merging clusters.
///
/// **Measured, and deliberately NOT the in-sample optimum.** Swept against
/// DER on `corpora/librispeech-diarization.toml`, 6 conversations, 0.25 s
/// collar (`examples/diarize_gate.rs sweep`):
///
/// ```text
///   0.55  34.00%   0.70   5.62%   0.85   2.71%  <- minimum
///   0.60  21.34%   0.75   5.20%   0.90   9.25%
///   0.65   8.60%   0.80   4.21%   0.95  44.65%   1.00+ 57.80%
/// ```
///
/// The minimum is 0.85. **0.80 ships instead**, because the curve is
/// violently asymmetric and the two directions fail differently:
///
/// - **Too low** over-splits. One speaker becomes two, and DER degrades
///   gently — 8.6 % at 0.65, still under 6 % at 0.70.
/// - **Too high** over-merges. Two speakers become one, every word of the
///   second is attributed to the first, and DER explodes — 9.3 % at 0.90,
///   **44.7 % at 0.95**, 57.8 % once everything collapses to a single
///   cluster.
///
/// Taking 0.85 would put the shipped default 0.05 from a 3.4x degradation on
/// audio that is not this corpus. 0.80 costs 1.5 pp in sample and buys a far
/// wider margin on everything else — the right trade when the penalty for
/// being wrong in one direction is six times the penalty in the other.
///
/// **This is in-sample.** Tuned on six conversations and reported on the same
/// six. The curve's shape is strong evidence the minimum is real rather than
/// a fluke — it is smooth and monotone on both sides, not scattered — but the
/// VALUE needs a holdout before it is a claim rather than a setting.
pub const DEFAULT_THRESHOLD: f32 = 0.80;

/// A stretch of audio attributed to one speaker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    /// Cluster index; rendered as `SPEAKER_00`, `SPEAKER_01`, ...
    pub speaker: usize,
}

/// Conventional label for a cluster index.
pub fn speaker_label(index: usize) -> String {
    format!("SPEAKER_{index:02}")
}

/// Cut speech regions into overlapping fixed-length windows for embedding.
///
/// Windows never cross a region boundary: a gap in speech is exactly where a
/// speaker change is most likely, so a window spanning one would blend two
/// voices into a single embedding and produce a cluster belonging to neither.
///
/// A region shorter than one window still yields one window covering it —
/// dropping it would silently leave that speech unattributed.
pub fn subsegment(regions: &[TimedSegment<()>], window: f64, hop: f64) -> Vec<(f64, f64)> {
    subsegment_at(regions, window, hop, 0.0)
}

/// [`subsegment`], with the buffer's absolute position in the stream so the
/// window grid does not move when the buffer does. See the snapping comment
/// below for why that matters.
pub fn subsegment_at(
    regions: &[TimedSegment<()>],
    window: f64,
    hop: f64,
    offset: f64,
) -> Vec<(f64, f64)> {
    let window = if window > 0.0 { window } else { WINDOW_SECS };
    let hop = if hop > 0.0 { hop } else { HOP_SECS };
    let mut out = Vec::new();
    for region in regions {
        let span = region.end - region.start;
        if span <= 0.0 {
            continue;
        }
        if span <= window {
            out.push((region.start, region.end));
            continue;
        }
        // Snap the chain's first window to the ABSOLUTE hop grid.
        //
        // Without this, windows are anchored to `region.start`, and a region
        // clipped by a sliding buffer's leading edge is anchored to the
        // BUFFER — which moves. Measured on a live 10 s window at a 1 s tick:
        // the leading region is pinned at 0.000 every tick while its content
        // slides underneath, so identical window bounds hold different audio
        // and every embedding is recomputed. Grids realigned only every 3 s
        // (`lcm(tick, hop)`), which is exactly where the hits appeared.
        //
        // `offset` is the buffer's position in the stream, so `offset + start`
        // is absolute and the same audio lands on the same bounds regardless
        // of where the buffer begins. With the default `offset = 0` and a
        // non-sliding caller this is the previous behaviour for any region
        // already on the grid, and shifts the chain by under one hop
        // otherwise.
        // ALWAYS cover the region's own start, THEN follow the absolute grid.
        //
        // The first version snapped the whole chain forward to the grid, which
        // skipped up to one hop of each region's leading audio — and a hop is
        // 0.75 s of speech at the moment a speaker begins, which is exactly
        // the evidence a cluster needs. Measured: blind DER 4.21 % -> 9.60 %,
        // oracle 5.00 % -> 8.11 %. It shipped in 0.6.0 described as "DER
        // unchanged" because the gate was re-run against a STALE example
        // binary (the library was rebuilt, the example was not).
        //
        // Emitting the region-start window first costs one extra forward per
        // region and restores that coverage; every window after it sits on the
        // absolute grid, which is what lets a sliding buffer's repeated audio
        // hit the embedding cache. Coverage and alignment, rather than a
        // trade between them.
        let snap = std::env::var("FFAI_DIARIZE_ABSGRID").as_deref() != Ok("off");
        out.push((region.start, (region.start + window).min(region.end)));
        let mut start = if snap {
            let abs = offset + region.start;
            region.start + (((abs / hop).ceil() * hop) - abs)
        } else {
            region.start
        };
        // Never re-emit the window just pushed.
        if start <= region.start {
            start = region.start + hop;
        }
        while start < region.end {
            let end = (start + window).min(region.end);
            out.push((start, end));
            if end >= region.end {
                break;
            }
            start += hop;
        }
    }
    out
}

/// Cosine distance, `1 - cos(a, b)`, in `[0, 2]`.
///
/// A zero-norm embedding is maximally distant from everything rather than
/// NaN: a silent or degenerate window should fail to join a cluster, not
/// poison every comparison it takes part in.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return 2.0;
    }
    1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Agglomerative clustering with average linkage over cosine distance.
///
/// Merges the closest pair of clusters until either the closest pair is
/// further apart than `threshold`, or `max_speakers` clusters remain. Average
/// linkage rather than single linkage because single linkage chains: one
/// ambiguous window between two speakers merges both into one cluster, which
/// is the characteristic diarization failure.
///
/// `max_speakers` is the caller's prior knowledge ("this is an interview, two
/// people"). When given it overrides the threshold — but note that this is
/// **not automatically the better option**, and measurement says so: with the
/// threshold tuned, blind clustering scores 4.21 % DER against 5.00 % when the
/// true count is supplied.
///
/// The reason is that forcing the count forces a MERGE. Hitting "exactly 4"
/// on a conversation the embeddings want to split 6 ways means joining two
/// clusters that should not join, and every word of one speaker is then
/// attributed to another — 8.87 s of confusion on the worst conversation
/// versus 2.89 s when left alone. Extra clusters are cheap under DER's
/// optimal mapping; a bad merge is not. Supply the count when it is certain,
/// not as a safety measure.
pub fn cluster(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_speakers: Option<usize>,
) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Each item starts in its own cluster; `members` tracks live clusters.
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let target = max_speakers.unwrap_or(1).max(1);

    loop {
        if members.len() <= 1 || (max_speakers.is_some() && members.len() <= target) {
            break;
        }
        // Closest pair by average linkage.
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let mut sum = 0.0f32;
                let mut count = 0usize;
                for &a in &members[i] {
                    for &b in &members[j] {
                        sum += cosine_distance(&embeddings[a], &embeddings[b]);
                        count += 1;
                    }
                }
                let d = if count > 0 { sum / count as f32 } else { f32::MAX };
                if best.as_ref().is_none_or(|(_, _, bd)| d < *bd) {
                    best = Some((i, j, d));
                }
            }
        }
        let Some((i, j, d)) = best else { break };
        // With a speaker count given, merge regardless of distance until the
        // count is reached; otherwise stop at the threshold.
        if max_speakers.is_none() && d > threshold {
            break;
        }
        let moved = members.remove(j);
        members[i].extend(moved);
    }

    // Label by first appearance, so SPEAKER_00 is whoever spoke first — a
    // stable, explainable ordering rather than whatever the merge order left.
    let mut order: Vec<(usize, usize)> = members
        .iter()
        .enumerate()
        .map(|(ci, m)| (*m.iter().min().expect("clusters are non-empty"), ci))
        .collect();
    order.sort_unstable();

    let mut labels = vec![0usize; n];
    for (speaker, (_, ci)) in order.into_iter().enumerate() {
        for &item in &members[ci] {
            labels[item] = speaker;
        }
    }
    labels
}

/// Collapse per-window labels into contiguous speaker turns.
///
/// Adjacent windows sharing a label become one turn. Because windows overlap,
/// a turn's end is the max end seen, not the last window's start plus a hop.
pub fn turns_from_labels(windows: &[(f64, f64)], labels: &[usize]) -> Vec<SpeakerTurn> {
    let mut turns: Vec<SpeakerTurn> = Vec::new();
    for (&(start, end), &speaker) in windows.iter().zip(labels.iter()) {
        match turns.last_mut() {
            Some(last) if last.speaker == speaker && start <= last.end => {
                last.end = last.end.max(end);
            }
            _ => turns.push(SpeakerTurn { start, end, speaker }),
        }
    }
    turns
}

/// Attach a speaker to each transcript segment by greatest temporal overlap.
///
/// Overlap rather than midpoint: a segment spanning a speaker change belongs
/// to whoever holds more of it, and a midpoint test would hand the whole
/// segment to whoever happened to own one instant. A segment overlapping no
/// turn gets `None` — unattributed, not guessed.
pub fn assign(
    segments: &[TimedSegment<String>],
    turns: &[SpeakerTurn],
) -> Vec<Option<usize>> {
    segments
        .iter()
        .map(|seg| {
            let mut best: Option<(usize, f64)> = None;
            for turn in turns {
                let overlap = seg.end.min(turn.end) - seg.start.max(turn.start);
                if overlap <= 0.0 {
                    continue;
                }
                match best {
                    Some((_, bo)) if bo >= overlap => {}
                    _ => best = Some((turn.speaker, overlap)),
                }
            }
            best.map(|(s, _)| s)
        })
        .collect()
}

/// Speaker turns as timed segments, ready to hang on a [`Transcript`].
pub fn labelled_turns(turns: &[SpeakerTurn]) -> Vec<TimedSegment<String>> {
    turns
        .iter()
        .map(|t| TimedSegment {
            start: t.start,
            end: t.end,
            value: speaker_label(t.speaker),
            confidence: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(start: f64, end: f64) -> TimedSegment<()> {
        TimedSegment { start, end, value: (), confidence: None }
    }

    fn seg(start: f64, end: f64) -> TimedSegment<String> {
        TimedSegment { start, end, value: "x".into(), confidence: None }
    }

    /// Two well-separated directions in embedding space stand in for two
    /// voices; no model needed to test the clustering.
    fn voice_a() -> Vec<f32> {
        vec![1.0, 0.0, 0.0, 0.0]
    }
    fn voice_b() -> Vec<f32> {
        vec![0.0, 1.0, 0.0, 0.0]
    }
    fn near(v: &[f32], jitter: f32) -> Vec<f32> {
        v.iter().map(|x| x + jitter).collect()
    }

    #[test]
    fn subsegment_never_spans_a_gap() {
        // The gap 2.0-5.0 is where a speaker change is most likely; no window
        // may cover it.
        let windows = subsegment(&[region(0.0, 2.0), region(5.0, 7.0)], 1.5, 0.75);
        assert!(windows.iter().all(|(s, e)| (*e <= 2.0) || (*s >= 5.0)), "{windows:?}");
    }

    #[test]
    fn short_region_still_produces_one_window() {
        let windows = subsegment(&[region(0.0, 0.4)], 1.5, 0.75);
        assert_eq!(windows, vec![(0.0, 0.4)]);
    }

    #[test]
    fn windows_cover_a_long_region_to_its_end() {
        let windows = subsegment(&[region(0.0, 5.0)], 1.5, 0.75);
        assert!(windows.first().expect("some").0 == 0.0);
        assert!((windows.last().expect("some").1 - 5.0).abs() < 1e-9, "{windows:?}");
    }

    #[test]
    fn zero_norm_embedding_is_distant_not_nan() {
        let d = cosine_distance(&[0.0, 0.0], &[1.0, 0.0]);
        assert!(d.is_finite() && d >= 2.0 - 1e-6, "{d}");
    }

    #[test]
    fn identical_voices_have_zero_distance() {
        assert!(cosine_distance(&voice_a(), &voice_a()).abs() < 1e-6);
    }

    #[test]
    fn two_voices_separate_into_two_clusters() {
        let e = vec![
            voice_a(),
            near(&voice_a(), 0.05),
            voice_b(),
            near(&voice_b(), 0.05),
        ];
        let labels = cluster(&e, DEFAULT_THRESHOLD, None);
        assert_eq!(labels[0], labels[1], "same voice split: {labels:?}");
        assert_eq!(labels[2], labels[3], "same voice split: {labels:?}");
        assert_ne!(labels[0], labels[2], "different voices merged: {labels:?}");
    }

    #[test]
    fn one_voice_stays_one_cluster() {
        let e = vec![voice_a(), near(&voice_a(), 0.02), near(&voice_a(), 0.04)];
        let labels = cluster(&e, DEFAULT_THRESHOLD, None);
        assert_eq!(labels.iter().collect::<std::collections::HashSet<_>>().len(), 1, "{labels:?}");
    }

    #[test]
    fn a_known_speaker_count_overrides_the_threshold() {
        // Four distinct directions; the threshold would keep four clusters,
        // but the caller says there are two people in the room.
        let e = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let labels = cluster(&e, DEFAULT_THRESHOLD, Some(2));
        assert_eq!(labels.iter().collect::<std::collections::HashSet<_>>().len(), 2, "{labels:?}");
    }

    #[test]
    fn speaker_zero_is_whoever_spoke_first() {
        // Item 0 is voice B here; it must still be labelled SPEAKER_00.
        let e = vec![voice_b(), voice_a(), near(&voice_b(), 0.05)];
        let labels = cluster(&e, DEFAULT_THRESHOLD, None);
        assert_eq!(labels[0], 0, "{labels:?}");
        assert_eq!(labels[2], 0, "{labels:?}");
        assert_eq!(labels[1], 1, "{labels:?}");
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert!(cluster(&[], DEFAULT_THRESHOLD, None).is_empty());
        assert!(subsegment(&[], 1.5, 0.75).is_empty());
        assert!(turns_from_labels(&[], &[]).is_empty());
    }

    #[test]
    fn turns_merge_adjacent_windows_of_one_speaker() {
        let windows = vec![(0.0, 1.5), (0.75, 2.25), (1.5, 3.0)];
        let turns = turns_from_labels(&windows, &[0, 0, 0]);
        assert_eq!(turns.len(), 1);
        assert_eq!((turns[0].start, turns[0].end), (0.0, 3.0));
    }

    #[test]
    fn turns_break_at_a_speaker_change() {
        let windows = vec![(0.0, 1.5), (1.5, 3.0), (3.0, 4.5)];
        let turns = turns_from_labels(&windows, &[0, 1, 1]);
        assert_eq!(turns.len(), 2, "{turns:?}");
        assert_eq!(turns[0].speaker, 0);
        assert_eq!(turns[1].speaker, 1);
        assert_eq!(turns[1].end, 4.5);
    }

    #[test]
    fn a_segment_goes_to_whoever_holds_more_of_it() {
        // Segment 0.0-1.0: speaker 0 holds 0.2, speaker 1 holds 0.8.
        let turns = vec![
            SpeakerTurn { start: 0.0, end: 0.2, speaker: 0 },
            SpeakerTurn { start: 0.2, end: 1.0, speaker: 1 },
        ];
        assert_eq!(assign(&[seg(0.0, 1.0)], &turns), vec![Some(1)]);
    }

    #[test]
    fn a_segment_overlapping_nothing_is_unattributed() {
        let turns = vec![SpeakerTurn { start: 10.0, end: 11.0, speaker: 0 }];
        assert_eq!(assign(&[seg(0.0, 1.0)], &turns), vec![None]);
    }

    #[test]
    fn labels_render_conventionally() {
        assert_eq!(speaker_label(0), "SPEAKER_00");
        assert_eq!(speaker_label(11), "SPEAKER_11");
    }
}

#[cfg(test)]
mod absolute_grid_tests {
    use super::*;

    fn region(start: f64, end: f64) -> TimedSegment<()> {
        TimedSegment { start, end, value: (), confidence: None }
    }

    /// THE invariant the live cache depends on: the same audio must produce
    /// the same ABSOLUTE window bounds no matter where the buffer starts.
    ///
    /// A live caller re-sends a sliding window, and a region clipped by the
    /// buffer's leading edge is anchored to the buffer — which moves. Before
    /// this, consecutive ticks re-cut identical audio at shifted offsets and
    /// every embedding was recomputed; measured hit rate ~24 %, and the
    /// window grids only realigned every `lcm(tick, hop)`.
    #[test]
    fn same_audio_lands_on_the_same_absolute_windows_as_the_buffer_slides() {
        // The same speech, seen from two buffers 1 s apart. Both are clipped
        // at the buffer's start, which is what pins the anchor.
        let a = subsegment_at(&[region(0.0, 6.0)], 1.5, 0.75, 10.0);
        let b = subsegment_at(&[region(0.0, 5.0)], 1.5, 0.75, 11.0);

        // Compare in ABSOLUTE time; buffer-relative bounds differ by design.
        let abs = |v: &[(f64, f64)], off: f64| -> Vec<(f64, f64)> {
            v.iter().map(|&(s, e)| ((s + off) * 1e6).round() / 1e6)
                .zip(v.iter().map(|&(_, e)| ((e + off) * 1e6).round() / 1e6))
                .collect()
        };
        let (aa, bb) = (abs(&a, 10.0), abs(&b, 11.0));

        // Every window of the later buffer must coincide with one from the
        // earlier: overlapping audio => identical bounds => a cache hit.
        let shared: Vec<_> = bb.iter().filter(|w| aa.contains(w)).collect();
        assert!(
            shared.len() >= bb.len() - 1,
            "overlapping audio must reuse the same absolute windows: {aa:?} vs {bb:?}"
        );
    }

    /// Snapping must never drop a region's audio entirely. A short region
    /// whose snap point lands past its end falls back to its own start.
    #[test]
    fn a_region_is_never_silently_dropped_by_snapping() {
        // Long enough to take the windowed path, offset so the snap moves.
        let w = subsegment_at(&[region(0.10, 2.00)], 1.5, 0.75, 0.70);
        assert!(!w.is_empty(), "snapping must not erase a region");
        assert!(w[0].0 >= 0.10 && w[0].0 < 2.00, "first window must start inside the region");
    }
}

/// The window geometry actually used, after any runtime override.
///
/// `FFAI_DIARIZE_HOP` / `FFAI_DIARIZE_WINDOW` exist so the geometry can be
/// SWEPT against DER rather than argued about. Embedding is ~100 % of
/// diarization's cost and the forward count is `span / hop`, so the hop is
/// the one knob that trades quality for compute linearly — and 0.75 s was
/// inherited convention, never measured on this corpus.
///
/// Read per call rather than cached: a sweep changes it between runs in one
/// process, and the cost of two env reads against a ~172 ms forward is
/// nothing.
pub fn geometry() -> (f64, f64) {
    let read = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0 && *v <= 30.0)
            .unwrap_or(default)
    };
    (read("FFAI_DIARIZE_WINDOW", WINDOW_SECS), read("FFAI_DIARIZE_HOP", HOP_SECS))
}

/// Embeddings already computed, in ABSOLUTE stream time.
///
/// The streaming path was re-deriving every window of the buffer on every
/// call. A content-keyed cache removed the repeated *forwards*, but the
/// pipeline still asked for windows it had already answered — and the ones it
/// could not reuse were the boundary windows, whose bounds move as the buffer
/// slides even when the speech does not.
///
/// This holds the answers instead of re-deriving the questions: window bounds
/// in absolute time, so audio settled by an earlier call is never
/// sub-segmented again. A tick then embeds only its NEW tail, and clustering
/// runs over the union of stored and new.
///
/// Bounded by a horizon rather than growing with the session: clustering is
/// O(n²) in windows, and a speaker registry already carries identity across
/// calls, so windows older than the horizon have nothing left to contribute.
#[derive(Default)]
pub struct StreamState {
    /// `(abs_start, abs_end, embedding)`, ascending by start.
    windows: Vec<(f64, f64, Vec<f32>)>,
    /// Absolute time through which windows have been generated. Audio before
    /// this is settled and is never re-cut.
    processed_to: f64,
}

/// How much stream history to cluster over. Long enough that a returning
/// voice is still represented, short enough that clustering stays cheap.
pub const STREAM_HORIZON_SECS: f64 = 30.0;

impl StreamState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Absolute time already covered.
    pub fn processed_to(&self) -> f64 {
        self.processed_to
    }

    /// Stored window count — the clustering input size.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Windows that still need embedding for this call: those of `regions`
    /// (already absolute) that start at or after what is settled.
    ///
    /// The `processed_to` cut is on the window's START, not its end: a window
    /// that merely *extends* past the mark was already generated with the
    /// audio available at the time, and regenerating it would produce a
    /// second embedding of overlapping speech that clustering would then
    /// weigh twice.
    pub fn pending(&self, abs_regions: &[(f64, f64)], window: f64, hop: f64) -> Vec<(f64, f64)> {
        let regions: Vec<TimedSegment<()>> = abs_regions
            .iter()
            .map(|&(s, e)| TimedSegment { start: s, end: e, value: (), confidence: None })
            .collect();
        // Offset 0: the bounds are already absolute, so the grid is absolute.
        subsegment_at(&regions, window, hop, 0.0)
            .into_iter()
            .filter(|&(s, _)| s >= self.processed_to - 1e-9)
            .collect()
    }

    /// Record newly embedded windows and advance the settled mark.
    pub fn extend(&mut self, new: Vec<(f64, f64, Vec<f32>)>, processed_to: f64) {
        self.windows.extend(new);
        self.windows.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.processed_to = self.processed_to.max(processed_to);
        let cutoff = self.processed_to - STREAM_HORIZON_SECS;
        self.windows.retain(|w| w.1 >= cutoff);
    }

    /// The stored windows and their embeddings, for clustering.
    pub fn parts(&self) -> (Vec<(f64, f64)>, Vec<Vec<f32>>) {
        (
            self.windows.iter().map(|w| (w.0, w.1)).collect(),
            self.windows.iter().map(|w| w.2.clone()).collect(),
        )
    }
}

#[cfg(test)]
mod stream_state_tests {
    use super::*;

    fn emb(v: f32) -> Vec<f32> {
        vec![v; 4]
    }

    /// The point of the whole design: settled audio is never re-cut, so a
    /// sliding buffer asks only for its new tail.
    #[test]
    fn settled_audio_is_never_resegmented() {
        let mut st = StreamState::new();
        // First call: a 10 s buffer at the start of the stream.
        let first = st.pending(&[(0.0, 10.0)], 1.5, 0.75);
        assert!(first.len() > 5, "a fresh 10 s region must yield windows");
        let n = first.len();
        st.extend(first.iter().map(|&(s, e)| (s, e, emb(1.0))).collect(), 10.0);

        // Second call one second later: the SAME speech plus 1 s of new audio.
        let second = st.pending(&[(0.0, 11.0)], 1.5, 0.75);
        assert!(
            second.len() <= 2,
            "only the new tail should be pending, got {} of {n} windows",
            second.len()
        );
        assert!(
            second.iter().all(|&(s, _)| s >= 10.0 - 1e-9),
            "pending windows must lie in the new tail: {second:?}"
        );
    }

    /// History is bounded, or clustering (O(n^2)) grows without limit over a
    /// long session.
    #[test]
    fn history_is_bounded_by_the_horizon() {
        let mut st = StreamState::new();
        for i in 0..200 {
            let t = i as f64;
            st.extend(vec![(t, t + 1.5, emb(t as f32))], t + 1.0);
        }
        assert!(
            st.len() as f64 <= STREAM_HORIZON_SECS + 2.0,
            "history must stay bounded, got {}",
            st.len()
        );
        assert!(st.processed_to() >= 199.0);
    }
}
