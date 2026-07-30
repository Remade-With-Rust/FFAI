//! Loads the speaker model and runs the diarization pipeline end to end.
//!
//! The glue between [`super::speaker`] (embeddings), [`super::fbank`] (its
//! features) and [`super::diarize`] (clustering). Kept apart from all three so
//! each stays independently testable — the clustering has 15 tests and needs
//! no weights, which is only possible because none of it lives here.

use std::path::Path;

use ffai_core::candle::{DType, Device};
use ffai_core::error::{Error, Result};
use ffai_core::types::TimedSegment;

use super::diarize::{self, SpeakerTurn, HOP_SECS, WINDOW_SECS};
use super::fbank::Fbank;
use super::registry::SpeakerRegistry;
use super::speaker::{Config, EcapaTdnn};

/// Default manifest name.
pub const DEFAULT_MODEL: &str = "ecapa-tdnn-voxceleb";

pub struct Diarizer {
    model: EcapaTdnn,
    fbank: Fbank,
    sample_rate: usize,
}

impl Diarizer {
    pub fn from_manifest_dir(dir: &Path, name: &str, device: Device) -> Result<Self> {
        Self::from_manifest_source(Some(dir), name, device)
    }

    /// Load by name, from `dir` when given and from the manifests compiled
    /// into the crate otherwise — so diarization works without a `models/`
    /// directory beside the caller.
    pub fn from_manifest_source(dir: Option<&Path>, name: &str, device: Device) -> Result<Self> {
        let manifest = crate::manifests::resolve(dir, name).map_err(|e| {
            Error::Model(format!(
                "{e} — diarization needs a speaker embedding model (default: {DEFAULT_MODEL})"
            ))
        })?;
        Self::from_manifest(&manifest, device)
    }

    /// Load from an already-parsed manifest.
    pub fn from_manifest(
        manifest: &ffai_models::ModelManifest,
        device: Device,
    ) -> Result<Self> {
        let resolved = manifest.fetch()?;
        let weights = resolved.file("embedding_model.ckpt")?.to_path_buf();

        // A pickled state dict rather than safetensors, so this reads rather
        // than maps. It is 83 MB — worth knowing, but not worth the
        // complexity of a conversion step on first use.
        let vb = candle_nn::VarBuilder::from_pth(&weights, DType::F32, &device)
            .map_err(|e| Error::Model(format!("reading speaker weights: {e}")))?;
        let cfg = Config::default();
        let fbank = Fbank::new(cfg.n_mels);
        let model = EcapaTdnn::load(cfg, vb, device)?;
        Ok(Diarizer { model, fbank, sample_rate: super::fbank::SAMPLE_RATE })
    }

    /// Speech regions in, speaker turns out.
    ///
    /// A window whose features cannot be computed, or whose embedding fails,
    /// is **dropped rather than substituted**: a zero vector would join
    /// whichever cluster happened to be nearest and silently attribute that
    /// audio to a real speaker.
    pub fn diarize(
        &self,
        samples: &[f32],
        regions: &[TimedSegment<()>],
        threshold: f32,
        max_speakers: Option<usize>,
    ) -> Vec<SpeakerTurn> {
        let windows = diarize::subsegment(regions, WINDOW_SECS, HOP_SECS);
        if windows.is_empty() {
            return Vec::new();
        }

        let (kept, embeddings) = self.embed_windows(samples, &windows);
        if embeddings.is_empty() {
            return Vec::new();
        }

        let labels = diarize::cluster(&embeddings, threshold, max_speakers);
        diarize::turns_from_labels(&kept, &labels)
    }

    /// Diarize with identity that persists across calls.
    ///
    /// Same pipeline as [`Self::diarize`] up to clustering, then one extra
    /// step: each in-chunk cluster is reduced to a centroid and matched
    /// against `registry`, so a voice heard in an earlier call keeps its
    /// label.
    ///
    /// **Clusters are matched, not individual windows.** A cluster centroid
    /// averages every window that agreed with it, which is far better evidence
    /// than one 1.5 s window — and the first window of a new voice is exactly
    /// when the decision is least reliable and, in a registry, most permanent.
    /// Matching per window would let a single marginal fragment enrol a
    /// duplicate speaker or, worse, claim an existing one.
    pub fn diarize_streaming(
        &self,
        samples: &[f32],
        regions: &[TimedSegment<()>],
        threshold: f32,
        max_speakers: Option<usize>,
        registry: &mut SpeakerRegistry,
    ) -> Vec<SpeakerTurn> {
        let windows = diarize::subsegment(regions, WINDOW_SECS, HOP_SECS);
        if windows.is_empty() {
            return Vec::new();
        }
        let (kept, embeddings) = self.embed_windows(samples, &windows);
        if embeddings.is_empty() {
            return Vec::new();
        }

        let local = diarize::cluster(&embeddings, threshold, max_speakers);
        let n_local = local.iter().copied().max().map(|m| m + 1).unwrap_or(0);

        // Reduce each local cluster to its mean, then resolve that against
        // the persistent identities.
        let dim = embeddings[0].len();
        let mut sums = vec![vec![0.0f32; dim]; n_local];
        let mut counts = vec![0usize; n_local];
        for (e, &c) in embeddings.iter().zip(local.iter()) {
            for (acc, v) in sums[c].iter_mut().zip(e.iter()) {
                *acc += v;
            }
            counts[c] += 1;
        }

        let global: Vec<usize> = (0..n_local)
            .map(|c| {
                let n = counts[c].max(1) as f32;
                let centroid: Vec<f32> = sums[c].iter().map(|v| v / n).collect();
                registry.assign(&centroid, counts[c] as f32)
            })
            .collect();

        let labels: Vec<usize> = local.iter().map(|&c| global[c]).collect();
        diarize::turns_from_labels(&kept, &labels)
    }

    /// Shared front half: window -> features -> embedding, dropping whatever
    /// fails rather than substituting a zero vector.
    fn embed_windows(
        &self,
        samples: &[f32],
        windows: &[(f64, f64)],
    ) -> (Vec<(f64, f64)>, Vec<Vec<f32>>) {
        let sr = self.sample_rate as f64;
        let mut kept = Vec::with_capacity(windows.len());
        let mut embeddings = Vec::with_capacity(windows.len());
        for &(start, end) in windows {
            let a = ((start * sr) as usize).min(samples.len());
            let b = ((end * sr).ceil() as usize).clamp(a, samples.len());
            let (feats, frames) = self.fbank.compute(&samples[a..b]);
            if frames == 0 {
                continue;
            }
            match self.model.embed(&feats, frames) {
                Ok(e) if e.iter().all(|v| v.is_finite()) => {
                    kept.push((start, end));
                    embeddings.push(e);
                }
                _ => continue,
            }
        }
        (kept, embeddings)
    }
}
