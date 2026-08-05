//! LIVE: streaming detection over a frame sequence.
//!
//! The function is the loop, not the model. A [`LiveSession`] wraps ANY
//! [`DetectEngine`] and never reaches inside it, adding the one thing a
//! stream needs that a single image does not: **the ability to not run.**
//!
//! # Why a gate exists here and nowhere else in Diana
//!
//! Diana's per-image path has no gate site at all. It is a fixed-compute
//! feedforward graph — a blank wall and a crowded street cost identical
//! FLOPs — with no search, no null arm and no per-unit trial to skip. The
//! only conditional in the hot path is optional NMS, off by default because
//! the one2one head is NMS-free.
//!
//! Across FRAMES the picture inverts. The expensive arm is the entire
//! forward pass, and the cheap signal — did the picture change — is directly
//! observable from the input, costs a single pass over pixels, and needs no
//! ground truth to evaluate. That is the whole argument for putting the gate
//! here.
//!
//! # The signal, and the mistake not to repeat
//!
//! Carmenta shipped this pattern first and **revised it after measurement**.
//! Its v1 used mean absolute pixel difference against a threshold of 2.0, and
//! a bench caught the gate swallowing real changes: the signal for a genuine
//! event was ~2.3 against a ~0.9 noise floor. Separated by a hair, so any
//! threshold either admitted noise or dropped events.
//!
//! The fix was to change the STATISTIC, not tune the constant: count the
//! FRACTION of pixels that move by more than a per-pixel delta the noise can
//! never cross. Sensor noise of plus or minus one level crosses a delta of 8
//! never; a real object moving crosses it on thousands of pixels. That
//! separates the two classes by orders of magnitude instead of a hair.
//!
//! This module inherits the lesson, not the constants — Carmenta gates on
//! text, which is small, static and high-contrast, while detection gates on
//! objects, which are large and can move behind low contrast. The threshold
//! here is picked from [`examples/live_harvest.rs`] on Diana's own content.

use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_core::error::Result;
use ffai_core::types::{DetectOutput, ImageBuffer, PixelFormat};

/// Fraction of pixels that must move by more than [`LiveConfig::pixel_delta`]
/// for a frame to count as changed.
///
/// Harvested, not chosen — see `examples/live_harvest.rs`.
pub const DEFAULT_CHANGE_FRACTION: f32 = 0.002;

/// Per-pixel grayscale delta that counts as a changed pixel.
///
/// 8 levels, the same value Carmenta arrived at, for the same reason: it sits
/// far above any plausible sensor or compression noise and far below a real
/// object edge moving. The point is not the number, it is that nothing in the
/// noise class can reach it.
pub const DEFAULT_PIXEL_DELTA: u8 = 8;

#[derive(Debug, Clone)]
pub struct LiveConfig {
    pub change_fraction: f32,
    pub pixel_delta: u8,
    /// Process every Nth frame (1 = consider every frame).
    pub sample_every: usize,
    /// Re-run regardless after this many consecutive gated frames.
    ///
    /// A gate that can never expire will serve a stale result forever if the
    /// signal is blind to some real change — a slow drift that never crosses
    /// the delta on enough pixels at once. This bounds that failure to a
    /// known number of frames instead of leaving it unbounded.
    pub max_gated_run: usize,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            change_fraction: DEFAULT_CHANGE_FRACTION,
            pixel_delta: DEFAULT_PIXEL_DELTA,
            sample_every: 1,
            max_gated_run: 30,
        }
    }
}

/// Per-session counters. Deterministic, and the numbers any gate claim reads.
#[derive(Debug, Clone, Default)]
pub struct LiveStats {
    /// Frames handed to [`LiveSession::process`].
    pub frames: usize,
    /// Frames that ran the model.
    pub processed: usize,
    /// Frames served from the previous result by the change gate.
    pub gated: usize,
    /// Frames skipped by the sampler before the gate saw them.
    pub sampled_out: usize,
    /// Times `max_gated_run` forced a run the signal would have gated.
    pub forced: usize,
}

impl LiveStats {
    /// Fraction of frames that never touched the model. This is the prize.
    pub fn skip_rate(&self) -> f32 {
        if self.frames == 0 {
            return 0.0;
        }
        (self.gated + self.sampled_out) as f32 / self.frames as f32
    }
}

/// Fraction of pixels differing by more than `delta` grayscale levels.
///
/// The cheap arm. One pass, no allocation, integer compares — it must stay
/// far below the cost of the model or the gate pays for itself and nothing
/// more. Returns 1.0 for a size or format change, which is not a "difference"
/// so much as a different picture, and must never be gated.
pub fn changed_fraction(prev: &ImageBuffer, cur: &ImageBuffer, delta: u8) -> f32 {
    if prev.width != cur.width || prev.height != cur.height || prev.format != cur.format {
        return 1.0;
    }
    let step = match cur.format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
    };
    let n = (cur.width as usize) * (cur.height as usize);
    if n == 0 {
        return 0.0;
    }
    let d = delta as i16;
    let mut changed = 0usize;
    // Compare the first channel only. A green-only change with red and blue
    // held is not a thing a camera produces, and one channel is a third of
    // the memory traffic — which matters because this runs on every frame,
    // including the ones it saves.
    for i in 0..n {
        let a = prev.data[i * step] as i16;
        let b = cur.data[i * step] as i16;
        if (a - b).abs() > d {
            changed += 1;
        }
    }
    changed as f32 / n as f32
}

/// A streaming detection session over an ordered frame sequence.
pub struct LiveSession<E: DetectEngine + ?Sized> {
    engine: std::sync::Arc<E>,
    cfg: LiveConfig,
    opts: DetectOptions,
    prev: Option<ImageBuffer>,
    last: Option<DetectOutput>,
    gated_run: usize,
    stats: LiveStats,
}

impl<E: DetectEngine + ?Sized> LiveSession<E> {
    pub fn new(engine: std::sync::Arc<E>, cfg: LiveConfig, opts: DetectOptions) -> Self {
        Self {
            engine,
            cfg,
            opts,
            prev: None,
            last: None,
            gated_run: 0,
            stats: LiveStats::default(),
        }
    }

    pub fn stats(&self) -> &LiveStats {
        &self.stats
    }

    /// Feed the next frame, in order.
    ///
    /// Returns the detections for it — either freshly computed, or the
    /// previous result reused. **A gated frame returns the PREVIOUS output
    /// verbatim**, which is what makes the gate an output stabiliser as well
    /// as a saving: an unchanged frame cannot churn, because nothing
    /// re-rolled the model.
    pub fn process(&mut self, frame: &ImageBuffer) -> Result<DetectOutput> {
        self.stats.frames += 1;

        // Sampler, ahead of the gate: a frame we never look at costs nothing,
        // not even the diff.
        if self.cfg.sample_every > 1 && (self.stats.frames - 1) % self.cfg.sample_every != 0 {
            if let Some(prev) = &self.last {
                self.stats.sampled_out += 1;
                return Ok(prev.clone());
            }
            // No previous result to serve; fall through and run.
        }

        let gate = match (&self.prev, &self.last) {
            (Some(p), Some(l)) => {
                let f = changed_fraction(p, frame, self.cfg.pixel_delta);
                if f < self.cfg.change_fraction {
                    if self.gated_run + 1 >= self.cfg.max_gated_run {
                        self.stats.forced += 1;
                        None
                    } else {
                        Some(l.clone())
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(out) = gate {
            self.gated_run += 1;
            self.stats.gated += 1;
            // The reference frame is NOT updated on a gated frame. Comparing
            // against the last PROCESSED frame is what stops a slow drift
            // from being gated forever one imperceptible step at a time.
            return Ok(out);
        }

        let out = self.engine.detect(frame, &self.opts)?;
        self.stats.processed += 1;
        self.gated_run = 0;
        self.prev = Some(frame.clone());
        self.last = Some(out.clone());
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffai_core::engine::{EngineInfo, EngineStatus, Task};
    use ffai_core::types::Detection;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Counts calls and returns a different box each time, so a reused result
    /// is distinguishable from a recomputed one.
    struct Counting(AtomicUsize, Vec<String>);

    impl DetectEngine for Counting {
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: "counting".into(),
                task: Task::Detect,
                status: EngineStatus::Experimental,
                description: "test double".into(),
            }
        }
        fn class_names(&self) -> &[String] {
            &self.1
        }
        fn detect(&self, _i: &ImageBuffer, _o: &DetectOptions) -> Result<DetectOutput> {
            let n = self.0.fetch_add(1, Ordering::Relaxed) as f32;
            Ok(DetectOutput {
                detections: vec![Detection {
                    x0: n,
                    y0: n,
                    x1: n + 1.0,
                    y1: n + 1.0,
                    class_id: 0,
                    confidence: 0.9,
                    track_id: None,
                }],
                letterbox: None,
            })
        }
    }

    fn frame(w: u32, h: u32, fill: u8) -> ImageBuffer {
        ImageBuffer {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            data: vec![fill; (w * h * 3) as usize],
        }
    }

    #[test]
    fn unchanged_frames_are_gated_and_cannot_churn() {
        let eng = Arc::new(Counting(AtomicUsize::new(0), vec!["thing".into()]));
        let mut s = LiveSession::new(eng.clone(), LiveConfig::default(), DetectOptions::default());
        let f = frame(64, 64, 100);

        let first = s.process(&f).unwrap();
        for _ in 0..10 {
            let out = s.process(&f).unwrap();
            // Byte-identical to the first result, not merely similar: the
            // gate's whole value as a stabiliser is that nothing re-rolled.
            assert_eq!(out.detections[0].x0, first.detections[0].x0, "gated frame churned");
        }
        assert_eq!(s.stats().processed, 1, "model ran more than once on a static stream");
        assert_eq!(s.stats().gated, 10);
        assert!((s.stats().skip_rate() - 10.0 / 11.0).abs() < 1e-6);
    }

    /// Noise below the per-pixel delta must not defeat the gate — this is the
    /// property Carmenta's v1 statistic lacked.
    #[test]
    fn sub_delta_noise_still_gates() {
        let eng = Arc::new(Counting(AtomicUsize::new(0), vec!["thing".into()]));
        let mut s = LiveSession::new(eng, LiveConfig::default(), DetectOptions::default());
        let base = frame(64, 64, 100);
        s.process(&base).unwrap();
        let mut noisy = base.clone();
        for (i, p) in noisy.data.iter_mut().enumerate() {
            *p = if i % 2 == 0 { 100 + 4 } else { 100 - 4 }; // ±4, under delta 8
        }
        s.process(&noisy).unwrap();
        assert_eq!(s.stats().processed, 1, "sub-delta noise defeated the gate");
    }

    #[test]
    fn a_real_change_is_not_gated() {
        let eng = Arc::new(Counting(AtomicUsize::new(0), vec!["thing".into()]));
        let mut s = LiveSession::new(eng, LiveConfig::default(), DetectOptions::default());
        let a = frame(64, 64, 100);
        s.process(&a).unwrap();
        let mut b = a.clone();
        // A 16x16 block moves well past the delta: 256 of 4096 pixels = 6.25%,
        // far above the 0.2% threshold.
        for y in 0..16 {
            for x in 0..16 {
                b.data[((y * 64 + x) * 3) as usize] = 200;
            }
        }
        s.process(&b).unwrap();
        assert_eq!(s.stats().processed, 2, "a real change was gated");
    }

    #[test]
    fn size_change_is_never_gated() {
        let eng = Arc::new(Counting(AtomicUsize::new(0), vec!["thing".into()]));
        let mut s = LiveSession::new(eng, LiveConfig::default(), DetectOptions::default());
        s.process(&frame(64, 64, 100)).unwrap();
        s.process(&frame(32, 32, 100)).unwrap();
        assert_eq!(s.stats().processed, 2, "a resolution change was gated");
    }

    /// An unbounded gate serves a stale result forever if the signal is blind
    /// to a real change. `max_gated_run` bounds that.
    #[test]
    fn gate_expires_after_max_run() {
        let eng = Arc::new(Counting(AtomicUsize::new(0), vec!["thing".into()]));
        let cfg = LiveConfig { max_gated_run: 4, ..Default::default() };
        let mut s = LiveSession::new(eng, cfg, DetectOptions::default());
        let f = frame(32, 32, 50);
        for _ in 0..10 {
            s.process(&f).unwrap();
        }
        assert!(s.stats().forced > 0, "the gate never expired");
        assert!(s.stats().processed > 1, "a blind gate ran the model only once in 10 frames");
    }
}
