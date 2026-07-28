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
        let manifests = ffai_models::load_dir(dir)?;
        let manifest = manifests.into_iter().find(|m| m.name == name).ok_or_else(|| {
            Error::Model(format!(
                "no model manifest named `{name}` in {} — diarization needs a speaker \
                 embedding model (see models/{DEFAULT_MODEL}.toml)",
                dir.display()
            ))
        })?;
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

        let sr = self.sample_rate as f64;
        let mut kept: Vec<(f64, f64)> = Vec::with_capacity(windows.len());
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(windows.len());
        for (start, end) in windows {
            let a = ((start * sr) as usize).min(samples.len());
            let b = ((end * sr).ceil() as usize).clamp(a, samples.len());
            let slice = &samples[a..b];
            let (feats, frames) = self.fbank.compute(slice);
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
        if embeddings.is_empty() {
            return Vec::new();
        }

        let labels = diarize::cluster(&embeddings, threshold, max_speakers);
        diarize::turns_from_labels(&kept, &labels)
    }
}
