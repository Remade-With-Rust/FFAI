//! Loading a Whisper model onto a candle device.
//!
//! The transformer blocks themselves come from `candle-transformers`, the
//! canonical candle implementation of the Whisper architecture. That is a
//! deliberate boundary: attention and cross-attention are shared
//! infrastructure (the BLAS of this stack), while the stages Mercury owns —
//! the mel front-end, the tokenizer grammar, the decode loop, long-audio
//! seek, streaming — are the ones that determine our output and our speed.
//!
//! Weights are resolved through `ffai-models`, so they are cached, license-
//! surfaced, and shared with the Python reference implementations rather than
//! downloaded twice.

use candle_transformers::models::whisper::{self as m, Config};
use ffai_core::candle::{DType, Device, Result as CandleResult};
use ffai_core::error::{Error, Result};
use ffai_models::ModelManifest;
use std::path::Path;

use super::text_decoder::Precision;
use super::tokenizer::WhisperTokenizer;

/// A loaded Whisper: weights on a device, plus its config and tokenizer.
pub struct LoadedWhisper {
    pub model: m::model::Whisper,
    /// Mercury's incremental decoder — the fast path (see
    /// [`super::text_decoder`]). The candle decoder inside `model` is kept as
    /// the reference twin: `FFAI_CANDLE_DECODER=1` switches back to it, which
    /// is how the oracle test proves the two agree token for token.
    pub decoder: super::text_decoder::TextDecoder,
    /// Mercury's audio encoder — quantizable, unlike candle's. The candle
    /// encoder inside `model` remains the reference twin.
    pub encoder: super::audio_encoder::AudioEncoder,
    pub config: Config,
    pub tokenizer: WhisperTokenizer,
    pub device: Device,
    pub dtype: DType,
    /// Model name as declared in the manifest, for the ledger.
    pub name: String,
    /// Decoder weight precision actually in use.
    pub precision: Precision,
    /// Decoder weight dtype (may differ from the encoder's — see the loader).
    pub decoder_dtype: DType,
}

impl LoadedWhisper {
    /// Load by manifest name from a manifest directory (e.g. `models/`).
    pub fn from_manifest_dir(
        dir: &Path,
        name: &str,
        device: Device,
        precision: Precision,
    ) -> Result<Self> {
        Self::from_manifest_source(Some(dir), name, device, precision)
    }

    /// Load by name, from `dir` when given and from the manifests compiled
    /// into the crate otherwise — the path that lets `WhisperCandle::new()`
    /// work from any working directory.
    pub fn from_manifest_source(
        dir: Option<&Path>,
        name: &str,
        device: Device,
        precision: Precision,
    ) -> Result<Self> {
        let manifest = crate::manifests::resolve(dir, name)?;
        Self::from_manifest(&manifest, device, precision)
    }

    pub fn from_manifest(
        manifest: &ModelManifest,
        device: Device,
        precision: Precision,
    ) -> Result<Self> {
        let resolved = manifest.fetch()?;
        let config: Config = {
            let text = std::fs::read_to_string(resolved.file("config.json")?)?;
            serde_json::from_str(&text)
                .map_err(|e| Error::Model(format!("parsing config.json: {e}")))?
        };
        let tokenizer = WhisperTokenizer::load(resolved.file("tokenizer.json")?)?;

        // MIXED PRECISION, measured per stage (mission plan section 6.13).
        //
        // The encoder is compute-bound: its matmuls run at 87-94% of this
        // machine's peak, and candle has no f16 FMA path, so half precision
        // there upcasts and LOSES (615 -> 421 GFLOP/s, 1.46x slower).
        //
        // The decoder is memory-bound: it streams every weight once per
        // generated token, so halving the bytes wins even through candle's
        // less efficient f16 path (3.35 -> 2.14 ms on the vocabulary
        // projection, 1.56x faster).
        //
        // One dtype for both would give up one of these. FFAI_DECODER_DTYPE=f32
        // restores the uniform build for A/B.
        let dtype = DType::F32;
        // Half precision is applied SURGICALLY, to the vocabulary projection
        // only (see TextDecoder::load). Per-op measurement showed f16 helps
        // exactly the op that streams 80 MB per token (1.39x) and hurts the
        // attention path that does not (1.13x slower) — so a blanket f16
        // decoder gave back most of what it won.
        let decoder_dtype = DType::F32;
        let weights = resolved.file("model.safetensors")?.to_path_buf();
        // SAFETY: memory-mapping a file is unsound only if the bytes change
        // while mapped. These weights live in the immutable content-addressed
        // Hugging Face cache (a blob under `blobs/<sha>`), are never written
        // by FFai, and were checksum-verified above when the manifest
        // declares a hash. Mapping rather than reading matters here: a
        // large-v3 read would copy ~3 GB per load.
        #[allow(unsafe_code)]
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)
                .map_err(|e| Error::Model(format!("mapping safetensors: {e}")))?
        };
        let model = m::model::Whisper::load(&vb, config.clone())
            .map_err(|e| Error::Model(format!("building whisper: {e}")))?;
        // A second view of the same mmapped file at the decoder's dtype.
        #[allow(unsafe_code)]
        let vb_dec = if decoder_dtype == dtype {
            vb.clone()
        } else {
            // SAFETY: as above — the same immutable cache blob, mapped again.
            unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[resolved.file("model.safetensors")?.to_path_buf()],
                    decoder_dtype,
                    &device,
                )
                .map_err(|e| Error::Model(format!("mapping safetensors (decoder): {e}")))?
            }
        };
        // Ours loads from the same VarBuilder and the same tensor names, so
        // both decoders are views of one set of mmapped weights.
        let decoder = super::text_decoder::TextDecoder::load(
            vb_dec.pp("model").pp("decoder"),
            &config,
            precision,
        )
        .map_err(|e| Error::Model(format!("building mercury decoder: {e}")))?;
        // The encoder stays f32 REGARDLESS of the requested precision, and
        // that is a measured decision, not an oversight (mission plan §6.5).
        // candle's quantized kernels are tuned for the LLM decode shape — one
        // token against a large weight matrix — and quantize their activations
        // per call. The encoder's shape is the opposite: 1500 positions in one
        // batched matmul. Quantizing it measured 12.1x -> 5.0x realtime, a
        // 2.4x REGRESSION. Revisit when candle's quantized GEMM improves or on
        // GPU, where the tradeoff differs.
        let encoder = super::audio_encoder::AudioEncoder::load(
            vb.pp("model").pp("encoder"),
            &config,
            Precision::F32,
        )
        .map_err(|e| Error::Model(format!("building mercury encoder: {e}")))?;

        Ok(LoadedWhisper {
            model,
            decoder,
            encoder,
            config,
            tokenizer,
            device,
            dtype,
            name: manifest.name.clone(),
            precision,
            decoder_dtype,
        })
    }

    /// Mel bands this model expects (80 for tiny→large-v2, 128 for large-v3).
    pub fn n_mels(&self) -> usize {
        self.config.num_mel_bins
    }

    /// True when the model has no language-token slot (the `.en` variants).
    pub fn is_english_only(&self) -> bool {
        self.config.vocab_size < 51_865
    }

    /// Build the encoder input tensor from a mel chunk.
    pub fn mel_tensor(
        &self,
        mel: &super::mel::MelChunk,
    ) -> CandleResult<ffai_core::candle::Tensor> {
        ffai_core::candle::Tensor::from_slice(
            &mel.data,
            (1, mel.n_mels, mel.n_frames),
            &self.device,
        )?
        .to_dtype(self.dtype)
    }
}
