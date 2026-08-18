//! Runtime A/B knobs — the instrument that makes the three-probe rule affordable.
//!
//! Every one of these was a `OnceLock` resolved from the environment on first
//! read. That is correct for a shipped binary and **wrong for measurement**: a
//! value that can only be set before the process starts forces every A/B to be
//! arm-by-arm across two processes.
//!
//! This box's op-level spread is **28–49 %** (`examples/noise_floor.rs`), and
//! the reference's own throughput spreads 37 % of its median across runs. The
//! effects this campaign is still deciding are 2–15 %. A cross-process
//! arm-by-arm comparison at that ratio does not measure code; it measures which
//! machine each arm happened to get. `codec-six-whys-unknowns` is explicit
//! about the remedy — *alternate the arms inside one process, many rounds, and
//! read the paired win rate* — and that is impossible while the knob is frozen
//! at process start.
//!
//! So the value lives in an atomic, seeded lazily from the same environment
//! variable with the same spelling and the same default. Behaviour with the
//! variable unset is unchanged; what is new is that a harness can flip it
//! between rounds and that both arms can be alive at once.
//!
//! **The per-call caveat is real and preserved.** `text_decoder` carries a
//! comment earned the hard way — an earlier version of one of these overrides
//! read `std::env::var` inside the per-token path and cost 3× more than the
//! optimization it was gating. `std::env::var` allocates and locks; a relaxed
//! atomic load does neither. The knobs that sit on a hot path (`PAR_HEADS`,
//! `DECODE_MIN_KEYS`) are a single relaxed load and a compare, which is what
//! `OnceLock` already cost after initialization.
//!
//! Load-time knobs (`MLP_INT8_*`, `DEC_F16_DISABLED`) still only take effect
//! for models loaded *after* the change — so a harness A/B's them by holding
//! two engines, one built under each setting, not by flipping mid-run.

use std::sync::atomic::{AtomicI64, Ordering};

/// Sentinel meaning "not yet resolved from the environment".
const UNSET: i64 = i64::MIN;

/// A word-matched boolean knob: true exactly when the variable is set to
/// `word`, matching the `env::var(..) == Ok("on")` form these replaced.
pub struct Flag {
    name: &'static str,
    word: &'static str,
    v: AtomicI64,
}

impl Flag {
    #[must_use]
    pub const fn new(name: &'static str, word: &'static str) -> Self {
        Self {
            name,
            word,
            v: AtomicI64::new(UNSET),
        }
    }

    #[inline]
    pub fn get(&self) -> bool {
        let cur = self.v.load(Ordering::Relaxed);
        if cur != UNSET {
            return cur != 0;
        }
        let on = std::env::var(self.name).as_deref() == Ok(self.word);
        self.v.store(i64::from(on), Ordering::Relaxed);
        on
    }

    /// Override for the duration of a measurement. Not for shipped code paths.
    pub fn set(&self, on: bool) {
        self.v.store(i64::from(on), Ordering::Relaxed);
    }

    /// Drop back to whatever the environment says.
    pub fn reset(&self) {
        self.v.store(UNSET, Ordering::Relaxed);
    }
}

/// A numeric knob with a validity filter and a default, matching the
/// `parse().filter().unwrap_or()` form these replaced.
pub struct Num {
    name: &'static str,
    default: i64,
    valid: fn(i64) -> bool,
    v: AtomicI64,
}

impl Num {
    pub const fn new(name: &'static str, default: i64, valid: fn(i64) -> bool) -> Self {
        Self {
            name,
            default,
            valid,
            v: AtomicI64::new(UNSET),
        }
    }

    #[inline]
    pub fn get(&self) -> i64 {
        let cur = self.v.load(Ordering::Relaxed);
        if cur != UNSET {
            return cur;
        }
        let n = std::env::var(self.name)
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|&n| (self.valid)(n))
            .unwrap_or(self.default);
        self.v.store(n, Ordering::Relaxed);
        n
    }

    #[inline]
    pub fn get_usize(&self) -> usize {
        self.get().max(0) as usize
    }

    /// Override for the duration of a measurement. Not for shipped code paths.
    pub fn set(&self, n: i64) {
        self.v.store(n, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        self.v.store(UNSET, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// The knobs themselves. Names, words and defaults are exactly what the
// `OnceLock` readers used, so an unset environment behaves identically.
// ---------------------------------------------------------------------------

/// `FFAI_KV_F16=off` keeps the cross-attention cache at f32 — the A/B arm.
/// Prepare-time: takes effect on the next window's cache build.
pub static KV_F16_DISABLED: Flag = Flag::new("FFAI_KV_F16", "off");

/// `FFAI_DEC_F16=off` forces the decoder's single-row projections back to f32.
/// Load-time.
pub static DEC_F16_DISABLED: Flag = Flag::new("FFAI_DEC_F16", "off");

/// `FFAI_MLP_INT8=on` puts the decoder MLP on int8 instead of f16. Load-time.
pub static MLP_INT8_ENABLED: Flag = Flag::new("FFAI_MLP_INT8", "on");

/// `FFAI_MLP_INT8=off` forces the decoder MLP back to the GEMM path. Load-time.
pub static MLP_INT8_DISABLED: Flag = Flag::new("FFAI_MLP_INT8", "off");

/// `FFAI_PAR_HEADS=on` fans the decode-shape heads across cores. Per-call.
pub static PAR_HEADS: Flag = Flag::new("FFAI_PAR_HEADS", "on");

/// Key-count floor below which the fused decode kernel declines. Per-call.
///
/// Measured, not guessed: at 8 the fused kernel wins 24/25 paired rounds
/// (z = +4.6) on TOTAL transcription time, 1.128×, with the self-attention
/// stage going 0.051 → 0.034 s.
pub static DECODE_MIN_KEYS: Num = Num::new("FFAI_DECODE_MIN_KEYS", 8, |n| n >= 0);

/// Weights per int8 scale in the vocabulary projection. Setting it to `d`
/// reproduces a single per-row scale exactly, so the block-vs-row tradeoff can
/// be measured rather than argued. Load-time.
pub static VOCAB_BLK: Num = Num::new("FFAI_VOCAB_BLK", 32, |n| n >= 32 && n % 32 == 0);

/// `FFAI_GEMV_PAD=off` disables the adaptive row-padding of `m == 1` matmuls.
/// Calibration-time.
pub static GEMV_PAD_DISABLED: Flag = Flag::new("FFAI_GEMV_PAD", "off");

/// `FFAI_ENC_KT=off` reverts the encoder's K projection to the plain
/// orientation plus `transpose(2,3).contiguous()`. Per-call, so both arms can
/// be interleaved in one process.
pub static ENC_KT_DISABLED: Flag = Flag::new("FFAI_ENC_KT", "off");

// `FFAI_PREV_CONTEXT` and `FFAI_ADAPTIVE_CTX` are deliberately NOT here:
// they are per-transcribe decisions read directly from the environment in
// `whisper_candle::transcribe`, so `ab_clips` (which flips its B arm by
// setting the variable between calls) sees the flip. A cached Flag would
// hand that harness two identical arms.

#[cfg(test)]
mod tests {
    use super::*;

    // The point of the module: a knob must be settable more than once, or the
    // interleaved harness cannot exist.
    #[test]
    fn a_flag_can_be_flipped_repeatedly() {
        static F: Flag = Flag::new("FFAI_TEST_FLAG_UNSET_IN_ENV", "on");
        assert!(!F.get(), "unset environment must resolve to the default");
        F.set(true);
        assert!(F.get());
        F.set(false);
        assert!(!F.get());
        F.reset();
        assert!(!F.get(), "reset returns to the environment's answer");
    }

    #[test]
    fn a_num_honours_its_default_and_can_be_overridden() {
        static N: Num = Num::new("FFAI_TEST_NUM_UNSET_IN_ENV", 8, |n| n >= 0);
        assert_eq!(N.get(), 8);
        N.set(256);
        assert_eq!(N.get(), 256);
        N.reset();
        assert_eq!(N.get(), 8);
    }

    // The filter is what keeps an out-of-range override from reaching a kernel
    // that assumes 32-lane blocks.
    #[test]
    fn a_num_rejects_values_its_filter_refuses() {
        static N: Num = Num::new("FFAI_TEST_NUM_BLK", 32, |n| n >= 32 && n % 32 == 0);
        assert_eq!(N.get(), 32);
        assert!(!(N.valid)(31));
        assert!(!(N.valid)(48));
        assert!((N.valid)(64));
    }
}
