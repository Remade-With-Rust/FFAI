//! ByteTrack: multi-object tracking with no appearance model.
//!
//! # The idea, and why it is the right one here
//!
//! Most trackers throw away low-confidence detections as noise. ByteTrack's
//! observation is that a box the detector is unsure about is usually an
//! **occluded object**, not a hallucination — so it associates twice: high-score
//! detections against all tracks first, then low-score detections against
//! whatever tracks are still unmatched. The second pass is what recovers a
//! person walking behind a pillar, which is most of what MOT17 is made of.
//!
//! It is also **appearance-free**: no ReID network, no embeddings, no second
//! model. BoT-SORT would add all three and drag a new weight file and its
//! licence in with it. This keeps Diana weight-free, which is the whole reason
//! it could be built without a converter or a five-tier oracle.
//!
//! ```no_run
//! use ffai_diana::track::{ByteTrack, TrackerConfig};
//! let mut t = ByteTrack::new(TrackerConfig::default());
//! // per frame:
//! // let live = t.update(&boxes, &scores, &classes);
//! ```

pub mod assign;
pub mod kalman;

use kalman::{Cov, State, Xyah};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackState {
    /// Seen, but not yet confirmed for `min_hits` frames — not reported.
    New,
    Tracked,
    /// Matched nothing this frame; kept alive for `max_age` frames.
    Lost,
}

#[derive(Clone, Debug)]
pub struct Track {
    pub id: u32,
    pub class_id: u32,
    pub score: f32,
    pub state: TrackState,
    /// Frames since it was last matched. Zero on the frame it was seen.
    pub time_since_update: u32,
    /// Total matches, ever. Used for the `min_hits` confirmation.
    pub hits: u32,
    s: State,
    p: Cov,
}

impl Track {
    /// Current box as `[x0, y0, x1, y1]`.
    pub fn xyxy(&self) -> [f32; 4] {
        Xyah { cx: self.s[0], cy: self.s[1], a: self.s[2], h: self.s[3] }.to_xyxy()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrackerConfig {
    /// Detections at or above this are "high score" and drive the first pass.
    pub track_thresh: f32,
    /// Detections below `track_thresh` but at or above this join the second
    /// pass — ByteTrack's whole point.
    pub low_thresh: f32,
    /// A NEW track needs a detection at least this confident. Higher than
    /// `track_thresh` on purpose: recovering an existing track from a weak box
    /// is cheap, but starting a new identity from one creates a false
    /// trajectory that costs IDF1 for as long as it lives.
    pub new_track_thresh: f32,
    /// Maximum IoU DISTANCE (`1 - IoU`) for a match.
    pub match_thresh: f32,
    /// Frames a lost track survives before removal.
    pub max_age: u32,
    /// Matches required before a track is reported.
    ///
    /// **1, not the conventional 3** — measured. `min_hits` and
    /// `new_track_thresh` are the same filter twice: both exist to stop a
    /// false detection becoming a trajectory. With `new_track_thresh` at 0.7
    /// the detection that starts a track is already confident, so withholding
    /// it for two more frames rejects almost no false positives and throws
    /// away the opening frames of real ones — and every withheld frame of a
    /// real track is an identity false-negative straight off IDF1.
    ///
    /// Swept against MOT17, replaying cached detections so only the tracker
    /// varies, IDF1 delta per sequence at `min_hits = 1`:
    ///
    /// | 02 | 04 | 05 | 09 | 10 | 11 | 13 |
    /// |---|---|---|---|---|---|---|
    /// | +0.13 | +0.09 | +0.68 | +0.20 | +0.29 | +0.38 | +0.34 |
    ///
    /// Positive on all seven — every tune sequence AND every holdout one —
    /// for **IDF1 31.37 -> 31.63** overall, with MOTA also up (18.58 ->
    /// 18.71). A knob that improves both metrics on every clip is not a
    /// trade, it is a filter that was priced twice.
    pub min_hits: u32,
    /// Hold UNCONFIRMED tracks out of the main association passes and give
    /// them only the leftovers, in a third pass. See [`ByteTrack::update`].
    pub deferred_unconfirmed: bool,
    /// Frames of absence after which a revived track RE-INITIALISES its Kalman
    /// state from the detection instead of merging into a stale prediction.
    /// `u32::MAX` disables. See [`ByteTrack::apply`].
    pub reinit_after: u32,
    /// Fold detection confidence into the association cost — the reference
    /// tracker's `fuse_score`. EXPERIMENTAL, default off; see
    /// [`assign::cost_matrix_fused`]. 0 = off, 1 = rank-only, 2 = rank + gate.
    pub fuse_mode: u8,
    /// Detections/frame at or below which `new_track_thresh` is used as-is.
    pub crowd_lo: f32,
    /// Detections/frame at or above which `new_track_thresh_crowded` is used.
    pub crowd_hi: f32,
    /// The threshold a CROWDED scene gets. See [`ByteTrack::effective_new_thresh`].
    pub new_track_thresh_crowded: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        // The reference's values, kept rather than tuned. The tracking plan's
        // stop rule: no threshold tuning until MOT17 MOTA/IDF1 produces a
        // baseline, because four knobs against one corpus is how a tracker gets
        // fitted to a benchmark.
        TrackerConfig {
            track_thresh: 0.5,
            low_thresh: 0.1,
            // 0.7, not the reference's 0.6 — the one change a threshold sweep
            // earned. Swept on MOT17 02/05/10 and confirmed on HELD-OUT
            // 04/09/11/13: ID switches fall 38 % across all seven sequences
            // (548 -> 339) and NOT ONE sequence got worse. The MOTA/IDF1 gains
            // that looked real on the tune set did not survive holdout
            // (MOTA -0.13 pp, IDF1 +0.23 pp); the ID-switch reduction did, on
            // 7/7. That asymmetry is the whole result.
            new_track_thresh: 0.7,
            match_thresh: 0.8,
            // `max_age` was swept alongside it and PRUNED, not adopted.
            // Raising it to 60-70 is worth a further +0.67 IDF1 overall, but
            // regresses 02 (-0.26) and 09 (-0.22) while winning big on 04/10/11,
            // and the response is not even monotonic — 04 reads +0.09 at 30,
            // -0.54 at 35-50, then +1.18 at 60. Three losers against three
            // winners on seven clips is a truth table too small to fit a
            // dispatch to: no candidate signal separated them (dets/frame
            // +0.34, sequence length +0.54, box area -0.18, churn -0.29), and
            // the groups OVERLAP on the best of those. A rule found here would
            // be the default outcome of searching, not evidence.
            //
            // So: recorded as an available +0.67 that needs a real signal, and
            // left at 30. Revisit with more sequences, not more search.
            max_age: 30,
            min_hits: 1,
            // OFF, and the reason is one sequence. Ranking the first
            // association by `1 - IoU * score` instead of `1 - IoU` — the
            // reference's `fuse_score` — is worth +0.61 IDF1 overall (31.63 ->
            // 32.24), +0.11 MOTA, and drops ID switches 435 -> 389:
            //
            //   02 +0.52  04 +1.18  05 +0.02  09 +0.00  10 -0.86  11 +1.77  13 +0.01
            //
            // Six sequences non-negative and one loser. Fusing into the GATE as
            // well (the naive port) was much worse — it tightens the threshold
            // by a factor of the score, and the resulting sweep swung 05/09/13
            // by whole points in both directions. Separating the two, via
            // `assign::assign_gated`, collapsed those to ~0.00 and left exactly
            // one real regression. That separation is the finding here.
            //
            // Not shipped on, because a single loser cannot be dispatched
            // honestly: ANY feature on which sequence 10 is extremal
            // "separates" a 6-vs-1 table perfectly, and 10 is already extremal
            // on detection churn (0.168, highest of the seven). MOT17-train has
            // only these seven sequences, so the table cannot be grown today.
            // Turn it on to re-measure when there is more footage.
            reinit_after: u32::MAX,
            // BOTH ON, and together they are the largest tracking result in
            // this campaign: IDF1 31.63 -> 32.82, MOTA 18.71 -> 18.92, ID
            // switches 435 -> 375, with NO threshold touched.
            //
            // Found by asking a question the sweeps could not: we ran
            // Ultralytics' OWN ByteTrack over OUR cached detections. It scored
            // IDF1 34.09 against our 31.63 on identical boxes — so 2.46 pp of
            // the 3.25 pp gap was the tracker, and it was worth fixing. The
            // same run showed HOW: on the same detections it emitted 2.46x
            // more trajectories and 1.29x more boxes, covering 34-96 % of GT
            // boxes against our 24-75 %. IDF1 charges an identity
            // false-negative for every GT box we never put an id on, so
            // under-reporting caps it however good the identities are — which
            // is why our ID switches were HALF theirs and our IDF1 still lost.
            //
            // Every earlier attempt to report more (lower `new_track_thresh`,
            // lower `track_thresh`) made things worse, and `deferred_unconfirmed`
            // is why: a brand-new track was competing in the main association
            // on equal terms with one followed for a hundred frames, and could
            // take its detection. More tracks meant more theft. Holding them
            // out until a third pass removes that, and then it improves EVERY
            // configuration measured (+0.58 to +2.33).
            //
            // Per sequence, IDF1 then MOTA:
            //
            //   02 -1.21 / +0.04    04 +1.88 / +0.16    05 +1.62 / +1.14
            //   09 +0.11 / -0.13    10 -0.84 / +0.23    11 +6.62 / +0.41
            //   13 -0.26 / +0.04
            //
            // Three sequences lose IDF1 and ALL THREE gain MOTA — they are not
            // made worse, the tracker makes a different trade there. That is
            // the reason this ships where `max_age` did not: `max_age` was a
            // fitted constant whose losers lost on both metrics, this is a
            // structural fix that nothing was tuned to.
            deferred_unconfirmed: true,
            fuse_mode: 1,
            // ADAPTIVE, and the sign-flip is what forced it. Raising
            // `new_track_thresh` to 0.7 helped MOTA on sparse sequences
            // (09 +5.97, 05 +3.24) and HURT it on the crowded ones
            // (04 -1.49, 10 -0.10). Detections-per-frame correlates -0.83 with
            // the IDF1 change and -0.67 with the MOTA change across all seven.
            //
            // The mechanism is not subtle: in a crowd, most objects really are
            // there and half-occluded, so a high bar for starting an identity
            // rejects REAL people and drives FN up. In a sparse scene the same
            // bar mostly rejects false positives.
            //
            // A fixed compromise would ship a regression to whoever runs
            // crowded footage. The signal costs nothing — the tracker is
            // already handed the detection count every frame.
            crowd_lo: 15.0,
            crowd_hi: 30.0,
            new_track_thresh_crowded: 0.6,
        }
    }
}

pub struct ByteTrack {
    cfg: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u32,
    frame: u64,
    /// Exponential moving average of detections per frame — the dispatch signal.
    ///
    /// A running mean rather than this frame's count: one crowded frame in a
    /// quiet scene should not flip the threshold, and a tracker whose behaviour
    /// oscillates frame to frame produces exactly the identity churn the
    /// threshold exists to prevent.
    density: f32,
}

impl ByteTrack {
    pub fn new(cfg: TrackerConfig) -> Self {
        ByteTrack { cfg, tracks: Vec::new(), next_id: 1, frame: 0, density: 0.0 }
    }

    pub fn frame_index(&self) -> u64 {
        self.frame
    }

    /// Smoothed detections per frame — the crowding signal.
    pub fn density(&self) -> f32 {
        self.density
    }

    /// `new_track_thresh`, ramped down as the scene gets crowded.
    ///
    /// Linear between `crowd_lo` and `crowd_hi`; flat outside. A ramp rather
    /// than a step because a step at a fixed density makes two nearly-identical
    /// scenes behave differently, and the underlying effect is gradual — the
    /// correlation is continuous, not a cliff.
    pub fn effective_new_thresh(&self) -> f32 {
        let (lo, hi) = (self.cfg.crowd_lo, self.cfg.crowd_hi);
        let t = if hi <= lo {
            0.0
        } else {
            ((self.density - lo) / (hi - lo)).clamp(0.0, 1.0)
        };
        self.cfg.new_track_thresh
            + t * (self.cfg.new_track_thresh_crowded - self.cfg.new_track_thresh)
    }

    /// One frame of detections in, the live tracks out.
    ///
    /// Returned tracks are confirmed and matched THIS frame — a lost track is
    /// kept internally so it can be recovered, but reporting a box the detector
    /// did not see would be inventing evidence.
    pub fn update(&mut self, boxes: &[[f32; 4]], scores: &[f32], classes: &[u32]) -> Vec<Track> {
        self.frame += 1;
        // EMA over ~30 frames. Seeded with the first frame rather than 0 so a
        // crowded scene is not treated as empty for its opening second.
        let n = boxes.len() as f32;
        self.density = if self.frame == 1 { n } else { self.density * 0.967 + n * 0.033 };
        let new_thresh = self.effective_new_thresh();

        // Every track predicts forward, matched or not.
        for t in &mut self.tracks {
            kalman::predict(&mut t.s, &mut t.p);
            t.time_since_update += 1;
        }

        let hi: Vec<usize> = (0..boxes.len())
            .filter(|&i| scores[i] >= self.cfg.track_thresh)
            .collect();
        let lo: Vec<usize> = (0..boxes.len())
            .filter(|&i| scores[i] < self.cfg.track_thresh && scores[i] >= self.cfg.low_thresh)
            .collect();

        // ---- pass 1: high-score detections against every track ----
        // A track only just created has one observation, no velocity estimate
        // worth the name, and a covariance that has not settled — yet in the
        // shared pass it competes on equal terms with a track that has been
        // followed for a hundred frames, and can TAKE that track's detection.
        // The reference never lets that happen: unconfirmed tracks sit out the
        // main passes and get only what is left over.
        //
        // This is what makes a low creation threshold survivable. Lowering
        // `new_track_thresh` without it floods the first pass with speculative
        // tracks that steal from established identities, which is exactly what
        // our threshold sweeps measured and (correctly) rejected.
        let deferred = self.cfg.deferred_unconfirmed;
        let is_new = |t: &Track| t.state == TrackState::New;
        let mut unmatched_tracks: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| !deferred || !is_new(&self.tracks[i]))
            .collect();
        let unconfirmed: Vec<usize> = if deferred {
            (0..self.tracks.len()).filter(|&i| is_new(&self.tracks[i])).collect()
        } else {
            Vec::new()
        };
        let mut matched_dets: Vec<bool> = vec![false; boxes.len()];

        let t_boxes: Vec<[f32; 4]> = unmatched_tracks.iter().map(|&i| self.tracks[i].xyxy()).collect();
        let d_boxes: Vec<[f32; 4]> = hi.iter().map(|&i| boxes[i]).collect();
        // Pass 1 only. The second pass is by construction low-confidence, so
        // multiplying by a score that is always small would push every pair
        // past the gate and disable the very recovery it exists for.
        let gate = assign::cost_matrix(&t_boxes, &d_boxes);
        let pairs = match self.cfg.fuse_mode {
            // Rank by confidence-weighted overlap, ADMIT by overlap.
            1 => {
                let d_scores: Vec<f32> = hi.iter().map(|&i| scores[i]).collect();
                let rank = assign::cost_matrix_fused(&t_boxes, &d_boxes, &d_scores);
                assign::assign_gated(&rank, &gate, self.cfg.match_thresh)
            }
            // BOTH fused — the reference's literal behaviour. Gating on the
            // fused cost is what makes a LOW `track_thresh` survivable: a 0.25
            // detection carries cost >= 0.75 and so cannot claim a track on
            // geometry alone. Decoupling the gate (mode 1) is better at a high
            // threshold and worse at a low one, which is why both exist.
            2 => {
                let d_scores: Vec<f32> = hi.iter().map(|&i| scores[i]).collect();
                let rank = assign::cost_matrix_fused(&t_boxes, &d_boxes, &d_scores);
                assign::assign(&rank, self.cfg.match_thresh)
            }
            _ => assign::assign(&gate, self.cfg.match_thresh),
        };

        let mut still_unmatched: Vec<usize> = unmatched_tracks.clone();
        for (ti, di) in &pairs {
            let track_idx = unmatched_tracks[*ti];
            let det = hi[*di];
            self.apply(track_idx, boxes[det], scores[det], classes[det]);
            matched_dets[det] = true;
            still_unmatched.retain(|&x| x != track_idx);
        }
        unmatched_tracks = still_unmatched;

        // ---- pass 2: LOW-score detections against what is left ----
        //
        // This is the algorithm. A box the detector scored 0.2 is usually a
        // person behind something, and matching it keeps the identity alive
        // instead of ending the track and starting a new one when they emerge.
        if !unmatched_tracks.is_empty() && !lo.is_empty() {
            let t_boxes: Vec<[f32; 4]> =
                unmatched_tracks.iter().map(|&i| self.tracks[i].xyxy()).collect();
            let d_boxes: Vec<[f32; 4]> = lo.iter().map(|&i| boxes[i]).collect();
            // A LOOSER gate on the second pass: these boxes are already known to
            // be poor, and demanding the same IoU as a confident one would
            // discard exactly the occlusions this pass exists to recover.
            let pairs = assign::assign(&assign::cost_matrix(&t_boxes, &d_boxes), 0.5);
            let mut left = unmatched_tracks.clone();
            for (ti, di) in &pairs {
                let track_idx = unmatched_tracks[*ti];
                let det = lo[*di];
                self.apply(track_idx, boxes[det], scores[det], classes[det]);
                matched_dets[det] = true;
                left.retain(|&x| x != track_idx);
            }
            unmatched_tracks = left;
        }

        // ---- pass 3: leftovers to the UNCONFIRMED tracks ----
        //
        // Whatever the confirmed tracks did not claim. An unconfirmed track
        // that matches here is promoted by `apply`; one that does not is
        // removed outright below, never kept for `max_age` — a speculative
        // track has to be corroborated by the very next frame or it was noise.
        let mut unconfirmed_matched: Vec<bool> = vec![false; unconfirmed.len()];
        if !unconfirmed.is_empty() {
            let rest: Vec<usize> = hi.iter().copied().filter(|&d| !matched_dets[d]).collect();
            if !rest.is_empty() {
                let t_boxes: Vec<[f32; 4]> =
                    unconfirmed.iter().map(|&i| self.tracks[i].xyxy()).collect();
                let d_boxes: Vec<[f32; 4]> = rest.iter().map(|&i| boxes[i]).collect();
                let pairs =
                    assign::assign(&assign::cost_matrix(&t_boxes, &d_boxes), self.cfg.match_thresh);
                for (ti, di) in &pairs {
                    let det = rest[*di];
                    self.apply(unconfirmed[*ti], boxes[det], scores[det], classes[det]);
                    matched_dets[det] = true;
                    unconfirmed_matched[*ti] = true;
                }
            }
        }
        // An unmatched unconfirmed track is killed here rather than left to the
        // `New`-dies-on-miss rule, because `apply` has already promoted the
        // matched ones out of `New` and the survivors must not be swept with them.
        for (k, &i) in unconfirmed.iter().enumerate() {
            if !unconfirmed_matched[k] {
                self.tracks[i].time_since_update = u32::MAX / 2;
            }
        }

        // ---- lost / removed ----
        for &i in &unmatched_tracks {
            if self.tracks[i].state != TrackState::New {
                self.tracks[i].state = TrackState::Lost;
            }
        }
        let max_age = self.cfg.max_age;
        self.tracks.retain(|t| {
            // A track that never got confirmed dies the moment it misses;
            // keeping it for max_age would let one-frame false positives
            // linger for a second of video.
            if t.state == TrackState::New && t.time_since_update > 0 {
                return false;
            }
            t.time_since_update <= max_age
        });

        // ---- new tracks from confident, unclaimed detections ----
        for &i in &hi {
            if matched_dets[i] || scores[i] < new_thresh {
                continue;
            }
            let (s, p) = kalman::initiate(Xyah::from_xyxy(boxes[i]));
            self.tracks.push(Track {
                id: self.next_id,
                class_id: classes[i],
                score: scores[i],
                state: TrackState::New,
                time_since_update: 0,
                hits: 1,
                s,
                p,
            });
            self.next_id += 1;
        }

        let min_hits = self.cfg.min_hits;
        self.tracks
            .iter()
            .filter(|t| t.time_since_update == 0 && t.hits >= min_hits)
            .cloned()
            .collect()
    }

    fn apply(&mut self, idx: usize, b: [f32; 4], score: f32, class_id: u32) {
        let reinit_after = self.cfg.reinit_after;
        let t = &mut self.tracks[idx];
        // A track absent for many frames comes back with a prediction that has
        // coasted on constant velocity the whole time and a covariance grown
        // by every one of those `predict` steps. Merging the detection INTO
        // that places the revived box between where the object is and where a
        // stale model guessed it would be — and that bad box is then what the
        // next frame's IoU is computed against, so one poor revival degrades
        // the whole remaining trajectory.
        //
        // The reference re-activates instead: state re-seeded from the
        // detection, velocity unknown again. This is that, gated on how long
        // the track was actually gone — a one-frame miss has a perfectly good
        // prediction and re-seeding it would throw away real velocity.
        if t.time_since_update > reinit_after {
            let (ns, np) = kalman::initiate(Xyah::from_xyxy(b));
            t.s = ns;
            t.p = np;
        } else {
            kalman::update(&mut t.s, &mut t.p, Xyah::from_xyxy(b));
        }
        t.time_since_update = 0;
        t.hits += 1;
        t.score = score;
        t.class_id = class_id;
        t.state = TrackState::Tracked;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxes_at(x: f32) -> Vec<[f32; 4]> {
        vec![[x, 100.0, x + 40.0, 200.0]]
    }

    /// A steadily-moving object keeps ONE id, reported from the frame it is
    /// first seen — `min_hits` is 1, so nothing is withheld.
    #[test]
    fn one_object_keeps_one_id() {
        let mut t = ByteTrack::new(TrackerConfig::default());
        let mut ids = Vec::new();
        for k in 0..10 {
            let out = t.update(&boxes_at(100.0 + 5.0 * k as f32), &[0.9], &[0]);
            if let Some(tr) = out.first() {
                ids.push(tr.id);
            }
        }
        assert!(!ids.is_empty(), "never reported a track");
        assert!(ids.iter().all(|&i| i == ids[0]), "id switched: {ids:?}");
        assert_eq!(ids.len(), 10, "reported on {} frames, expected all 10", ids.len());
    }

    /// `min_hits` still WORKS when raised — the default moved, the mechanism
    /// did not. Without this, dropping the default to 1 would leave the
    /// confirmation path untested and free to rot.
    #[test]
    fn min_hits_withholds_a_track_until_confirmed() {
        let cfg = TrackerConfig { min_hits: 3, ..Default::default() };
        let mut t = ByteTrack::new(cfg);
        let mut reported = 0;
        for k in 0..10 {
            if !t.update(&boxes_at(100.0 + 5.0 * k as f32), &[0.9], &[0]).is_empty() {
                reported += 1;
            }
        }
        assert_eq!(reported, 8, "min_hits=3 should withhold the first two frames");
    }

    /// An unconfirmed track must not TAKE a detection from a confirmed one.
    ///
    /// This is the whole point of the deferred third pass, and it is what made
    /// reporting more tracks safe. Without it, every new speculative track was
    /// a competitor in the main association, so lowering the creation threshold
    /// cost more in stolen identities than it gained in coverage.
    #[test]
    fn an_unconfirmed_track_cannot_steal_from_a_confirmed_one() {
        let mut t = ByteTrack::new(TrackerConfig::default());
        // Establish one track over several frames so it is firmly confirmed.
        for k in 0..5 {
            t.update(&boxes_at(100.0 + 5.0 * k as f32), &[0.9], &[0]);
        }
        let established = t.update(&boxes_at(125.0), &[0.9], &[0])[0].id;

        // Now two overlapping detections: the object, plus a near-duplicate
        // that will spawn an unconfirmed track.
        let two = vec![[130.0, 100.0, 170.0, 200.0], [134.0, 100.0, 174.0, 200.0]];
        t.update(&two, &[0.9, 0.9], &[0, 0]);
        // Next frame only the original object is present. The established
        // identity must still own it.
        let out = t.update(&boxes_at(135.0), &[0.9], &[0]);
        assert!(
            out.iter().any(|x| x.id == established),
            "the established id {established} lost its detection to a newcomer: {:?}",
            out.iter().map(|x| x.id).collect::<Vec<_>>()
        );
    }

    /// THE ByteTrack test. An object goes to LOW confidence mid-sequence, the
    /// kind a naive tracker drops. The second association pass must hold the id.
    #[test]
    fn low_score_detections_keep_the_identity() {
        let mut t = ByteTrack::new(TrackerConfig::default());
        let mut first = 0;
        for k in 0..5 {
            if let Some(tr) = t.update(&boxes_at(100.0 + 5.0 * k as f32), &[0.9], &[0]).first() {
                first = tr.id;
            }
        }
        assert_ne!(first, 0);
        // Now occluded: still detected, but at 0.2 — below track_thresh.
        let mut kept = 0;
        for k in 5..10 {
            if let Some(tr) = t.update(&boxes_at(100.0 + 5.0 * k as f32), &[0.2], &[0]).first() {
                kept = tr.id;
            }
        }
        assert_eq!(kept, first, "identity lost to low-confidence boxes");
    }

    /// Occlusion shorter than max_age keeps the id; longer starts a new one.
    #[test]
    fn survives_a_gap_shorter_than_max_age() {
        let cfg = TrackerConfig { max_age: 5, ..Default::default() };
        let mut t = ByteTrack::new(cfg);
        let mut first = 0;
        for k in 0..6 {
            if let Some(tr) = t.update(&boxes_at(100.0 + 5.0 * k as f32), &[0.9], &[0]).first() {
                first = tr.id;
            }
        }
        for _ in 0..3 {
            t.update(&[], &[], &[]);
        }
        let out = t.update(&boxes_at(100.0 + 5.0 * 9.0), &[0.9], &[0]);
        assert_eq!(out.first().map(|x| x.id), Some(first), "id lost across a 3-frame gap");
    }

    #[test]
    fn a_long_gap_starts_a_new_identity() {
        let cfg = TrackerConfig { max_age: 3, ..Default::default() };
        let mut t = ByteTrack::new(cfg);
        let mut first = 0;
        for k in 0..6 {
            if let Some(tr) = t.update(&boxes_at(100.0 + 5.0 * k as f32), &[0.9], &[0]).first() {
                first = tr.id;
            }
        }
        for _ in 0..8 {
            t.update(&[], &[], &[]);
        }
        for _ in 0..4 {
            t.update(&boxes_at(400.0), &[0.9], &[0]);
        }
        let out = t.update(&boxes_at(400.0), &[0.9], &[0]);
        assert!(out.first().is_some_and(|x| x.id != first), "should be a NEW identity");
    }

    /// A single-frame false positive must never be reported.
    #[test]
    fn one_frame_blips_never_get_an_id() {
        // HONEST CHANGE OF PROPERTY. At `min_hits = 3` this asserted a blip is
        // never reported at all. At `min_hits = 1` it IS reported, for exactly
        // the one frame it exists — and that is the trade we measured and took:
        // suppressing the blip cost the first two frames of every REAL track,
        // which is an identity false-negative on each, and on all seven MOT17
        // sequences that cost more than the blips do. IDF1 and MOTA both rose.
        //
        // What must NOT change is that a blip cannot LINGER. An unconfirmed
        // track dies the moment it misses, rather than being kept for
        // `max_age`, so it can never accumulate frames it did not earn. That
        // invariant is the one worth a test, so this is it.
        let mut t = ByteTrack::new(TrackerConfig::default());
        let first = t.update(&boxes_at(10.0), &[0.9], &[0]);
        assert_eq!(first.len(), 1, "min_hits=1 reports on sight");
        for k in 0..5 {
            assert!(
                t.update(&[], &[], &[]).is_empty(),
                "a blip lingered into frame {}",
                k + 2
            );
        }
    }

    /// The crowding ramp must actually move, and in the right direction: a
    /// crowded scene gets the LOWER bar, because there a high bar rejects real
    /// people rather than false positives.
    #[test]
    fn crowding_lowers_the_new_track_threshold() {
        let cfg = TrackerConfig::default();
        let mut sparse = ByteTrack::new(cfg);
        for _ in 0..60 {
            sparse.update(&boxes_at(50.0), &[0.9], &[0]);
        }
        let mut crowded = ByteTrack::new(cfg);
        let many: Vec<[f32; 4]> = (0..40).map(|k| [k as f32 * 30.0, 10.0, k as f32 * 30.0 + 20.0, 80.0]).collect();
        let sc = vec![0.9f32; 40];
        let cl = vec![0u32; 40];
        for _ in 0..60 {
            crowded.update(&many, &sc, &cl);
        }
        assert!(sparse.density() < cfg.crowd_lo, "sparse density {}", sparse.density());
        assert!(crowded.density() > cfg.crowd_hi, "crowded density {}", crowded.density());
        assert!(
            crowded.effective_new_thresh() < sparse.effective_new_thresh(),
            "crowded {} should be BELOW sparse {}",
            crowded.effective_new_thresh(),
            sparse.effective_new_thresh()
        );
        assert!((sparse.effective_new_thresh() - cfg.new_track_thresh).abs() < 1e-6);
    }

    /// Two objects crossing must not swap ids — the case Hungarian exists for.
    #[test]
    fn two_objects_do_not_swap_ids() {
        let mut t = ByteTrack::new(TrackerConfig::default());
        let mut a = 0;
        let mut b = 0;
        for k in 0..12 {
            let x1 = 100.0 + 6.0 * k as f32;
            let x2 = 300.0 - 6.0 * k as f32;
            let out = t.update(
                &[[x1, 100.0, x1 + 40.0, 200.0], [x2, 100.0, x2 + 40.0, 200.0]],
                &[0.9, 0.9],
                &[0, 0],
            );
            if k == 5 && out.len() == 2 {
                a = out[0].id;
                b = out[1].id;
            }
        }
        assert_ne!(a, 0);
        assert_ne!(a, b, "two objects share an id");
    }
}
