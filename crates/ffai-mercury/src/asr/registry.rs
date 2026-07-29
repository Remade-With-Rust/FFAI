//! Speaker registry — identity that survives across calls.
//!
//! [`super::diarize`] answers "who spoke when" **inside one call**, and its
//! labels are arbitrary names for clusters: `SPEAKER_00` is whoever spoke
//! first in *that* audio. That is the standard definition, it is what DER
//! measures under an optimal label mapping, and it is what WhisperX does.
//!
//! It is also useless for a stream. Feed a live microphone in one-second
//! chunks and the same person becomes `SPEAKER_00` in one chunk and
//! `SPEAKER_01` in the next, because each call starts from nothing. FFai's
//! principle 5 says *"streaming-first — engines process chunks; whole-file is
//! the degenerate case"*, and diarization that only works whole-file inverts
//! that exactly.
//!
//! This is the missing half: a small amount of state that remembers what each
//! speaker sounds like, so a voice heard in chunk 1 is still the same label in
//! chunk 50.
//!
//! **The asymmetry that shapes every decision here.** Batch clustering may
//! reconsider — a merge it regrets is undone by the next merge, and the final
//! partition is what counts. A registry cannot. Once two people share a
//! centroid, every later window of either is matched against a blend of both,
//! and the error compounds for the rest of the session. Splitting one person
//! into two labels is ugly and survivable; merging two people is permanent.
//! So the enrolment rule errs toward **creating a new speaker**, and
//! [`ENROL_MARGIN`] exists to make that explicit rather than incidental.

use super::diarize::cosine_distance;

/// How much closer than the batch threshold a match must be before an
/// existing speaker claims a new cluster.
///
/// Batch clustering merges at distance < 0.80. A registry match is permanent,
/// so it demands `0.80 - ENROL_MARGIN`. The gap is the price of
/// irreversibility, and it is a starting point rather than a calibration —
/// E5 sweeps it against streaming DER.
pub const ENROL_MARGIN: f32 = 0.15;

/// One remembered voice.
#[derive(Debug, Clone)]
struct Known {
    /// Running mean of every embedding assigned to this speaker.
    centroid: Vec<f32>,
    /// How many embeddings contributed, so the mean can be updated in place.
    weight: f32,
}

/// Speaker identities that persist across [`assign`](SpeakerRegistry::assign)
/// calls.
#[derive(Debug, Clone)]
pub struct SpeakerRegistry {
    known: Vec<Known>,
    match_threshold: f32,
    /// Cap on remembered speakers. A stream that never stops would otherwise
    /// accumulate a centroid per stray noise burst forever; past this, the
    /// nearest existing speaker takes it however far away it is.
    max_speakers: Option<usize>,
}

impl SpeakerRegistry {
    /// `batch_threshold` is the same number [`super::diarize::cluster`] uses;
    /// the registry tightens it by [`ENROL_MARGIN`] itself, so callers pass
    /// one threshold and do not have to remember the relationship.
    pub fn new(batch_threshold: f32, max_speakers: Option<usize>) -> Self {
        SpeakerRegistry {
            known: Vec::new(),
            match_threshold: (batch_threshold - ENROL_MARGIN).max(0.05),
            max_speakers,
        }
    }

    pub fn len(&self) -> usize {
        self.known.len()
    }

    pub fn is_empty(&self) -> bool {
        self.known.is_empty()
    }

    /// Forget everyone. A new recording is a new set of people.
    pub fn reset(&mut self) {
        self.known.clear();
    }

    /// The distance to the nearest known speaker, if any — exposed so a
    /// caller can see *how* confident an assignment was rather than only
    /// which label came back.
    pub fn nearest(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        self.known
            .iter()
            .enumerate()
            .map(|(i, k)| (i, cosine_distance(embedding, &k.centroid)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Match this embedding to a known speaker, or enrol a new one.
    ///
    /// `weight` is how much evidence the embedding carries — the number of
    /// windows behind a cluster centroid. A centroid built from eight windows
    /// should move the running mean more than one built from a single 1.5 s
    /// window, and treating them alike lets a marginal fragment drag a
    /// well-established speaker's centroid toward it.
    pub fn assign(&mut self, embedding: &[f32], weight: f32) -> usize {
        let weight = weight.max(1.0);
        let capped = self.max_speakers.is_some_and(|m| self.known.len() >= m);

        match self.nearest(embedding) {
            // Close enough, or we are out of slots and must place it somewhere.
            Some((i, d)) if d <= self.match_threshold || capped => {
                let k = &mut self.known[i];
                let total = k.weight + weight;
                for (c, e) in k.centroid.iter_mut().zip(embedding.iter()) {
                    *c = (*c * k.weight + *e * weight) / total;
                }
                k.weight = total;
                i
            }
            _ => {
                self.known.push(Known { centroid: embedding.to_vec(), weight });
                self.known.len() - 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice_a() -> Vec<f32> {
        vec![1.0, 0.0, 0.0, 0.0]
    }
    fn voice_b() -> Vec<f32> {
        vec![0.0, 1.0, 0.0, 0.0]
    }
    /// Same direction, slightly perturbed — the same person on another day.
    fn near(v: &[f32], jitter: f32) -> Vec<f32> {
        v.iter().map(|x| x + jitter).collect()
    }

    #[test]
    fn a_returning_voice_keeps_its_label() {
        // THE point of the whole module: chunk 1 and chunk 5 agree.
        let mut r = SpeakerRegistry::new(0.80, None);
        let first = r.assign(&voice_a(), 4.0);
        let _ = r.assign(&voice_b(), 4.0);
        let again = r.assign(&near(&voice_a(), 0.03), 4.0);
        assert_eq!(first, again, "the same voice got a new label");
        assert_eq!(r.len(), 2, "expected two speakers, got {}", r.len());
    }

    #[test]
    fn a_new_voice_enrols_rather_than_joining_the_nearest() {
        let mut r = SpeakerRegistry::new(0.80, None);
        let a = r.assign(&voice_a(), 4.0);
        let b = r.assign(&voice_b(), 4.0);
        assert_ne!(a, b);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn matching_is_stricter_than_batch_clustering() {
        // The margin is the whole safety story: a pair that batch clustering
        // WOULD merge must not automatically be merged permanently.
        let batch = 0.80f32;
        let r = SpeakerRegistry::new(batch, None);
        assert!(
            r.match_threshold < batch,
            "registry threshold {} is not stricter than batch {batch}",
            r.match_threshold
        );
        assert!((r.match_threshold - (batch - ENROL_MARGIN)).abs() < 1e-6);
    }

    #[test]
    fn labels_are_assigned_in_order_of_first_appearance() {
        let mut r = SpeakerRegistry::new(0.80, None);
        assert_eq!(r.assign(&voice_b(), 1.0), 0, "first voice heard is SPEAKER_00");
        assert_eq!(r.assign(&voice_a(), 1.0), 1);
        assert_eq!(r.assign(&near(&voice_b(), 0.02), 1.0), 0);
    }

    #[test]
    fn a_heavier_cluster_moves_the_centroid_more() {
        // A centroid from eight windows is better evidence than one window,
        // and must not be dragged as far by a marginal fragment.
        let mut heavy = SpeakerRegistry::new(0.80, None);
        heavy.assign(&voice_a(), 8.0);
        heavy.assign(&near(&voice_a(), 0.10), 1.0);
        let heavy_drift = cosine_distance(&heavy.known[0].centroid, &voice_a());

        let mut light = SpeakerRegistry::new(0.80, None);
        light.assign(&voice_a(), 1.0);
        light.assign(&near(&voice_a(), 0.10), 1.0);
        let light_drift = cosine_distance(&light.known[0].centroid, &voice_a());

        assert!(
            heavy_drift < light_drift,
            "weight ignored: heavy {heavy_drift} light {light_drift}"
        );
    }

    #[test]
    fn the_centroid_tracks_a_voice_rather_than_freezing_on_first_sight() {
        let mut r = SpeakerRegistry::new(0.80, None);
        r.assign(&voice_a(), 1.0);
        let before = r.known[0].centroid.clone();
        r.assign(&near(&voice_a(), 0.05), 1.0);
        assert_ne!(before, r.known[0].centroid, "centroid never updated");
    }

    #[test]
    fn a_speaker_cap_stops_unbounded_growth() {
        // A long-running stream must not accrue a centroid per noise burst.
        let mut r = SpeakerRegistry::new(0.80, Some(2));
        r.assign(&voice_a(), 1.0);
        r.assign(&voice_b(), 1.0);
        let third = r.assign(&vec![0.0, 0.0, 1.0, 0.0], 1.0);
        assert_eq!(r.len(), 2, "cap exceeded");
        assert!(third < 2, "capped assignment must reuse an existing label");
    }

    #[test]
    fn reset_forgets_everyone() {
        let mut r = SpeakerRegistry::new(0.80, None);
        r.assign(&voice_a(), 1.0);
        r.assign(&voice_b(), 1.0);
        r.reset();
        assert!(r.is_empty());
        assert_eq!(r.assign(&voice_b(), 1.0), 0, "labels restart after reset");
    }

    #[test]
    fn nearest_reports_the_distance_not_just_the_label() {
        let mut r = SpeakerRegistry::new(0.80, None);
        r.assign(&voice_a(), 1.0);
        let (idx, d) = r.nearest(&voice_a()).expect("one speaker known");
        assert_eq!(idx, 0);
        assert!(d < 1e-6, "identical embedding should be at distance 0, got {d}");
        assert!(r.nearest(&voice_b()).expect("still one").1 > 0.5);
    }

    #[test]
    fn an_empty_registry_enrols_whatever_arrives_first() {
        let mut r = SpeakerRegistry::new(0.80, None);
        assert!(r.nearest(&voice_a()).is_none());
        assert_eq!(r.assign(&voice_a(), 1.0), 0);
    }
}
