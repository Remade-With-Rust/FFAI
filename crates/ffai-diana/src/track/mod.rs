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
    pub min_hits: u32,
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
            new_track_thresh: 0.6,
            match_thresh: 0.8,
            max_age: 30,
            min_hits: 3,
        }
    }
}

pub struct ByteTrack {
    cfg: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u32,
    frame: u64,
}

impl ByteTrack {
    pub fn new(cfg: TrackerConfig) -> Self {
        ByteTrack { cfg, tracks: Vec::new(), next_id: 1, frame: 0 }
    }

    pub fn frame_index(&self) -> u64 {
        self.frame
    }

    /// One frame of detections in, the live tracks out.
    ///
    /// Returned tracks are confirmed and matched THIS frame — a lost track is
    /// kept internally so it can be recovered, but reporting a box the detector
    /// did not see would be inventing evidence.
    pub fn update(&mut self, boxes: &[[f32; 4]], scores: &[f32], classes: &[u32]) -> Vec<Track> {
        self.frame += 1;

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
        let mut unmatched_tracks: Vec<usize> = (0..self.tracks.len()).collect();
        let mut matched_dets: Vec<bool> = vec![false; boxes.len()];

        let t_boxes: Vec<[f32; 4]> = unmatched_tracks.iter().map(|&i| self.tracks[i].xyxy()).collect();
        let d_boxes: Vec<[f32; 4]> = hi.iter().map(|&i| boxes[i]).collect();
        let pairs = assign::assign(&assign::cost_matrix(&t_boxes, &d_boxes), self.cfg.match_thresh);

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
            if matched_dets[i] || scores[i] < self.cfg.new_track_thresh {
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
        let t = &mut self.tracks[idx];
        kalman::update(&mut t.s, &mut t.p, Xyah::from_xyxy(b));
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

    /// A steadily-moving object keeps ONE id, and only appears after min_hits.
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
        // min_hits = 3, so the first two frames report nothing.
        assert_eq!(ids.len(), 8, "reported on {} frames, expected 8", ids.len());
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
        let mut t = ByteTrack::new(TrackerConfig::default());
        assert!(t.update(&boxes_at(10.0), &[0.9], &[0]).is_empty());
        for _ in 0..5 {
            assert!(t.update(&[], &[], &[]).is_empty());
        }
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
