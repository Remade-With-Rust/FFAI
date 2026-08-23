//! # ffai-core
//!
//! The spine of `FFai`: shared media/AI types, one trait per task, and the
//! engine registry that makes implementations interchangeable.
//!
//! The design borrows ffmpeg's load-bearing idea: `AVCodec` is a registry of
//! interchangeable implementations selected by name (`-c:v libx264`). `FFai`'s
//! equivalent: [`engine::AsrEngine`], [`engine::TtsEngine`],
//! [`engine::OcrEngine`], and [`engine::VlmEngine`] are traits with many
//! competing engines behind them, selected with `--engine <name>`.
//!
//! Candle is the tensor spine — re-exported here as [`candle`] so every engine
//! crate shares one `Tensor`/`Device` and buffers flow between models without
//! copies.

pub mod cost;
pub mod engine;
pub mod fastmath;
pub mod fastops;
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
#[must_use]
pub const fn best_device() -> candle::Device {
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

/// Release memory the allocator is holding after a large one-off load.
///
/// Loading model weights allocates far more than the model keeps: dtype
/// conversions, quantization scratch, and the safetensors mapping all churn
/// through the heap and are then freed. Freed is not returned — the allocator
/// keeps the pages, and they stay counted against the process for the rest of
/// its life.
///
/// Measured on Whisper tiny.en: the process holds **345 MiB** after a run, but
/// trimming and then repeating the SAME work re-settles at **102 MiB**. So
/// ~240 MiB of what looked like footprint was never needed by the work — and a
/// reference implementation that manages its own arena does not carry it,
/// which is most of why we measured 2.2x its resident memory while needing
/// half of what it does.
///
/// This is a hint, not a free: pages the process still needs fault straight
/// back in on next use. Call it ONCE, after a known-large load, never in a hot
/// path — trimming what you are about to touch again just buys page faults.
///
/// No-op where the platform offers no equivalent.
//
// NOT const, and clippy::nursery is wrong here in a platform-specific way: on
// non-Windows every branch below is cfg'd out, the body is empty, and
// `missing_const_for_fn` fires. On Windows the same function calls a Win32
// entry point and cannot be const. Making it const would therefore compile on
// Linux and break on Windows - so the lint is allowed rather than obeyed.
// Caught only when CI first ran on Linux; it never fired on the author's box.
#[allow(clippy::missing_const_for_fn)]
pub fn release_load_arena() {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn SetProcessWorkingSetSizeEx(
                process: *mut core::ffi::c_void,
                min: usize,
                max: usize,
                flags: u32,
            ) -> i32;
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
        }
        // SAFETY: pseudo-handle needing no close; (SIZE_T)-1 for both bounds is
        // the documented request-a-trim form rather than a quota.
        unsafe {
            SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, 0);
        }
    }
}
