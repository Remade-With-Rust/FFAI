//! LIVE: streaming OCR over a frame sequence (mission plan §4.1).
//!
//! The function is the loop, not the model: a [`LiveSession`] wraps ANY
//! `OcrEngine` (it never reaches inside — the §2 independence contract) and
//! adds the three things a stream needs:
//!
//! - **Change gate**: a frame whose mean absolute pixel difference from the
//!   previous processed frame is below threshold reuses the previous result
//!   at zero model cost — which is also the v1 output stabilizer: an
//!   unchanged frame *cannot* churn, because it never re-rolls the model.
//! - **Sampler**: process every Nth frame; skipped frames cost nothing.
//! - **Timed segments**: output text becomes `TimedSegment<String>` spans
//!   (start = first frame that showed the text, end = first frame that
//!   didn't), rendered as SRT/VTT with the same shape Mercury's transcript
//!   converters use.
//!
//! Frame SOURCE is deliberately out of scope: v1 consumes decoded
//! `ImageBuffer`s in order (the CLI feeds it an image-sequence directory).
//! When rff publishes video ingest, `ffai-media::sample_frames` slots in
//! front of this unchanged — that decision is recorded in the mission plan
//! §9, not silently assumed.

use ffai_core::engine::{OcrEngine, OcrOptions};
use ffai_core::error::Result;
use ffai_core::types::{ImageBuffer, OcrOutput, TimedSegment};

use crate::image::to_gray_f32;

#[derive(Debug, Clone)]
pub struct LiveConfig {
    /// Fraction of pixels that must differ by more than [`Self::pixel_delta`]
    /// grayscale levels for a frame to count as changed.
    ///
    /// Measured revision: v1 used mean absolute difference with threshold
    /// 2.0, and the screencast bench caught it swallowing REAL single-slot
    /// text changes (~2.3 mean diff against a ~0.9 noise floor — 9 OCR calls
    /// for 24 change events, stale text scored 12% CER). A changed-pixel
    /// FRACTION with a per-pixel delta the noise can never cross separates
    /// the classes by orders of magnitude instead of a hair: ±1-level noise
    /// crosses delta 8 never; one changed text line crosses it on thousands
    /// of pixels.
    pub change_fraction: f32,
    /// Per-pixel grayscale delta that counts as a changed pixel.
    pub pixel_delta: f32,
    /// Process every Nth frame (1 = every frame).
    pub sample_every: usize,
}

impl Default for LiveConfig {
    fn default() -> Self {
        LiveConfig { change_fraction: 0.0005, pixel_delta: 8.0, sample_every: 1 }
    }
}

/// Per-session counters — the numbers the M-C2 gates read.
#[derive(Debug, Default, Clone)]
pub struct LiveStats {
    pub frames: usize,
    /// Frames that ran the full model.
    pub ocr_calls: usize,
    /// Frames served by the change gate.
    pub gated: usize,
    /// Frames skipped by the sampler.
    pub sampled_out: usize,
    /// Wall seconds of each full model call, for p50/p95.
    pub call_secs: Vec<f64>,
}

impl LiveStats {
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.call_secs.is_empty() {
            return None;
        }
        let mut v = self.call_secs.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        Some(v[((v.len() - 1) as f64 * p).round() as usize])
    }
}

pub struct LiveSession<'e> {
    engine: &'e dyn OcrEngine,
    opts: OcrOptions,
    cfg: LiveConfig,
    prev_gray: Option<Vec<f32>>,
    prev_out: OcrOutput,
    prev_text: String,
    /// Open span: (start time, text).
    open: Option<(f64, String)>,
    segments: Vec<TimedSegment<String>>,
    pub stats: LiveStats,
}

impl<'e> LiveSession<'e> {
    pub fn new(engine: &'e dyn OcrEngine, opts: OcrOptions, cfg: LiveConfig) -> Self {
        LiveSession {
            engine,
            opts,
            cfg,
            prev_gray: None,
            prev_out: OcrOutput::default(),
            prev_text: String::new(),
            open: None,
            segments: Vec::new(),
            stats: LiveStats::default(),
        }
    }

    /// Feed the next frame with its presentation time; returns the current
    /// (possibly reused) recognition result.
    pub fn push_frame(&mut self, img: &ImageBuffer, t_secs: f64) -> Result<&OcrOutput> {
        self.stats.frames += 1;
        let n = self.stats.frames;
        if self.cfg.sample_every > 1 && (n - 1) % self.cfg.sample_every != 0 {
            self.stats.sampled_out += 1;
            return Ok(&self.prev_out);
        }

        let gray = to_gray_f32(img)?;
        let unchanged = match &self.prev_gray {
            Some(prev) if prev.len() == gray.len() => {
                let delta = self.cfg.pixel_delta;
                let changed =
                    prev.iter().zip(&gray).filter(|(a, b)| (**a - **b).abs() > delta).count();
                (changed as f64 / gray.len() as f64) < self.cfg.change_fraction as f64
            }
            _ => false,
        };

        if unchanged {
            self.stats.gated += 1;
            return Ok(&self.prev_out);
        }

        let t0 = std::time::Instant::now();
        let out = self.engine.recognize(img, &self.opts)?;
        self.stats.call_secs.push(t0.elapsed().as_secs_f64());
        self.stats.ocr_calls += 1;

        let text = out.text();
        if text != self.prev_text {
            if let Some((start, prev)) = self.open.take() {
                if !prev.is_empty() {
                    self.segments.push(TimedSegment { start, end: t_secs, value: prev, confidence: None });
                }
            }
            if !text.is_empty() {
                self.open = Some((t_secs, text.clone()));
            }
            self.prev_text = text;
        }
        self.prev_gray = Some(gray);
        self.prev_out = out;
        Ok(&self.prev_out)
    }

    /// Close the session at `end_secs` and take the timed track.
    pub fn finish(mut self, end_secs: f64) -> (Vec<TimedSegment<String>>, LiveStats) {
        if let Some((start, text)) = self.open.take() {
            if !text.is_empty() {
                self.segments.push(TimedSegment { start, end: end_secs, value: text, confidence: None });
            }
        }
        (self.segments, self.stats)
    }
}

/// Render timed OCR spans as SRT.
pub fn to_srt(segments: &[TimedSegment<String>]) -> String {
    let mut out = String::new();
    for (i, s) in segments.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            srt_time(s.start, ','),
            srt_time(s.end, ','),
            s.value
        ));
    }
    out
}

/// Render timed OCR spans as WebVTT.
pub fn to_vtt(segments: &[TimedSegment<String>]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for s in segments {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            srt_time(s.start, '.'),
            srt_time(s.end, '.'),
            s.value
        ));
    }
    out
}

fn srt_time(secs: f64, ms_sep: char) -> String {
    let ms = (secs.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{ms_sep}{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffai_core::engine::{EngineInfo, EngineStatus, Task};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts real calls; answers with a text derived from the frame's first
    /// pixel so tests can force "same pixels" vs "changed pixels".
    struct CountingEngine(AtomicUsize);

    impl OcrEngine for CountingEngine {
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: "counting".into(),
                task: Task::Ocr,
                status: EngineStatus::Experimental,
                description: String::new(),
            }
        }
        fn recognize(&self, img: &ImageBuffer, _o: &OcrOptions) -> Result<OcrOutput> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(OcrOutput::from_lines([format!("pixel {}", img.data[0])]))
        }
    }

    fn frame(v: u8) -> ImageBuffer {
        ImageBuffer {
            width: 8,
            height: 8,
            format: ffai_core::types::PixelFormat::Gray8,
            data: vec![v; 64],
        }
    }

    #[test]
    fn unchanged_frames_are_gated_and_cannot_churn() {
        let eng = CountingEngine(AtomicUsize::new(0));
        let mut s = LiveSession::new(&eng, OcrOptions::default(), LiveConfig::default());
        for i in 0..5 {
            s.push_frame(&frame(100), i as f64).unwrap();
        }
        // frame changes at t=5
        s.push_frame(&frame(200), 5.0).unwrap();
        let calls = eng.0.load(Ordering::Relaxed);
        assert_eq!(calls, 2, "one call per distinct frame, gate served the rest");
        let (segments, stats) = s.finish(6.0);
        assert_eq!(stats.gated, 4);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].value, "pixel 100");
        assert!((segments[0].start, segments[0].end) == (0.0, 5.0));
        assert_eq!(segments[1].value, "pixel 200");
    }

    #[test]
    fn srt_renders_spans() {
        let segs = vec![TimedSegment { start: 0.0, end: 2.5, value: "REC 00:12".to_string(), confidence: None }];
        let srt = to_srt(&segs);
        assert!(srt.contains("00:00:00,000 --> 00:00:02,500"), "{srt}");
        assert!(srt.contains("REC 00:12"));
    }
}
