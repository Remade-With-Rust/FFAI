//! Loads the speaker model and runs the diarization pipeline end to end.
//!
//! The glue between [`super::speaker`] (embeddings), [`super::fbank`] (its
//! features) and [`super::diarize`] (clustering). Kept apart from all three so
//! each stays independently testable — the clustering has 15 tests and needs
//! no weights, which is only possible because none of it lives here.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Mutex;

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
    /// Embeddings keyed by the CONTENT of the window that produced them.
    ///
    /// Streaming re-embeds the whole buffer every call: the live path hands
    /// over the trailing 10 s once a second, and at 1.5 s windows on a 0.75 s
    /// hop that is ~13 windows of which ~12 are byte-identical to last
    /// tick's. Each one cost a full ECAPA forward — measured at 172 ms, and
    /// the embedding stage is ~100 % of diarization (`examples/
    /// diarize_anatomy.rs`; fbank is 1.1 %, clustering ~0).
    ///
    /// Keyed on the SAMPLES, not on the window's time bounds, because the
    /// buffer slides: `(0.0, 1.5)` names different audio on consecutive
    /// ticks, so a time key would return a confidently wrong embedding.
    /// Content keying makes a hit numerically identical to a recompute, so
    /// this cannot move DER — it is the same vector, not an approximation.
    cache: Mutex<HashMap<u64, Vec<f32>>>,
}

/// Entries to retain. Each is 192 f32 plus a key — under 1 KB — so this is
/// ~0.4 MB at the cap, against the ~13 windows a live tick actually needs.
/// Cleared wholesale rather than evicted one at a time: the access pattern
/// is a sliding window, so the old entries are genuinely dead, and an LRU's
/// bookkeeping would cost more than the misses it saves.
const EMBED_CACHE_CAP: usize = 512;

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
        Ok(Diarizer {
            model,
            fbank,
            sample_rate: super::fbank::SAMPLE_RATE,
            cache: Mutex::new(HashMap::new()),
        })
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
    /// The sample rate the speaker front end expects.
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Filterbank features for one window — exposed so a profile can price
    /// the DSP separately from the network without reimplementing either
    /// (a probe that reimplements the path measures the probe).
    pub fn fbank_for(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        self.fbank.compute(samples)
    }

    /// One embedding forward. Profiling seam; see [`Self::fbank_for`].
    pub fn embed_for(&self, feats: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.model.embed(feats, frames)
    }

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
            let window = &samples[a..b];

            // Hash the audio, not the timestamps — see `cache`. ~96 KB per
            // window against a ~172 ms forward, so the hash is ~0.06 % of
            // what a hit saves; it does not need to be a fast hash to pay.
            let key = content_key(window);
            let caching = cache_enabled();
            if caching {
                if let Some(hit) = self.cache.lock().ok().and_then(|c| c.get(&key).cloned()) {
                    CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    kept.push((start, end));
                    embeddings.push(hit);
                    continue;
                }
            }
            CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            let (feats, frames) = self.fbank.compute(window);
            if frames == 0 {
                continue;
            }
            match self.model.embed(&feats, frames) {
                Ok(e) if e.iter().all(|v| v.is_finite()) => {
                    if let Some(mut c) = caching.then(|| self.cache.lock().ok()).flatten() {
                        // Only failures are excluded from the cache: a
                        // dropped window must stay droppable, or a transient
                        // failure would be memoized into a permanent one.
                        if c.len() >= EMBED_CACHE_CAP {
                            c.clear();
                        }
                        c.insert(key, e.clone());
                    }
                    kept.push((start, end));
                    embeddings.push(e);
                }
                _ => continue,
            }
        }
        (kept, embeddings)
    }
}

/// A key for a window's audio content.
///
/// Hashes the sample BITS rather than the floats, because `f32` is not
/// `Hash` — and bitwise equality is exactly the property wanted here: two
/// windows collide only if they are the same audio, in which case the model
/// would return the same embedding anyway.
fn content_key(samples: &[f32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    samples.len().hash(&mut h);
    // SAFETY-free reinterpretation: `to_bits` is a pure integer view of the
    // float, so this hashes the exact bytes without an unsafe cast.
    for s in samples {
        s.to_bits().hash(&mut h);
    }
    h.finish()
}

/// Embedding-cache hit/miss counters.
///
/// A speedup without a hit rate is a number you cannot attribute — and this
/// campaign has twice mistaken a stale binary for "the change does nothing".
/// These make the mechanism visible instead of inferred.
pub static CACHE_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static CACHE_MISSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `(hits, misses)` since process start.
pub fn cache_stats() -> (usize, usize) {
    (
        CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed),
        CACHE_MISSES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Runtime switch, read per window rather than cached in a `OnceLock`, so a
/// harness can interleave both arms in ONE process. The live cost varies with
/// how much speech VAD finds per tick, so arm-by-arm across two processes
/// compares different work — the mistake that produced the "2x slower"
/// reading this whole thread started from.
fn cache_enabled() -> bool {
    std::env::var("FFAI_DIARIZE_CACHE").as_deref() != Ok("off")
}

/// Drop every cached embedding. For a harness measuring a cold arm.
pub fn clear_embed_cache_counters() {
    CACHE_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    CACHE_MISSES.store(0, std::sync::atomic::Ordering::Relaxed);
}
