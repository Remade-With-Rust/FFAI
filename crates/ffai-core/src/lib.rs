//! # ffai-core
//!
//! The spine of FFai: shared media/AI types, one trait per task, and the
//! engine registry that makes implementations interchangeable.
//!
//! The design borrows ffmpeg's load-bearing idea: `AVCodec` is a registry of
//! interchangeable implementations selected by name (`-c:v libx264`). FFai's
//! equivalent: [`engine::AsrEngine`], [`engine::TtsEngine`],
//! [`engine::OcrEngine`], and [`engine::VlmEngine`] are traits with many
//! competing engines behind them, selected with `--engine <name>`.
//!
//! Candle is the tensor spine — re-exported here as [`candle`] so every engine
//! crate shares one `Tensor`/`Device` and buffers flow between models without
//! copies.

pub mod engine;
pub mod error;
pub mod registry;
pub mod types;

/// The shared tensor framework (Hugging Face candle).
pub use candle_core as candle;

pub use error::{Error, Result};

/// Pick the best available compute device.
///
/// CPU always works; CUDA/Metal are behind the crate features of the same
/// name and fall back to CPU when unavailable.
pub fn best_device() -> candle::Device {
    #[cfg(feature = "cuda")]
    if let Ok(dev) = candle::Device::new_cuda(0) {
        return dev;
    }
    #[cfg(feature = "metal")]
    if let Ok(dev) = candle::Device::new_metal(0) {
        return dev;
    }
    candle::Device::Cpu
}
