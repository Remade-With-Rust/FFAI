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
    /// Auto-ROI (mission plan §4.1): after calibration, run recognition on
    /// the learned horizontal text bands instead of the full frame, with a
    /// periodic full-frame sweep to catch new elements. Opt-in — landed
    /// because the observe-only harvest measured 100% box coverage at 38.8%
    /// of frame area on the screencast corpus (61% detection-pixel ceiling).
    pub auto_roi: bool,
    /// OCR calls spent calibrating (full-frame) before bands activate.
    pub calib_calls: usize,
    /// Every Nth OCR call runs full-frame anyway (new-element sweep, and the
    /// band set is rebuilt from its result).
    pub full_sweep_every: usize,
}

impl Default for LiveConfig {
    fn default() -> Self {
        LiveConfig {
            change_fraction: 0.0005,
            pixel_delta: 8.0,
            sample_every: 1,
            auto_roi: false,
            calib_calls: 8,
            full_sweep_every: 10,
        }
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
    /// OCR calls served by band (auto-ROI) recognition.
    pub roi_calls: usize,
    /// Bands actually re-recognized across all roi_calls (dirty bands).
    pub dirty_bands: usize,
    /// Wall seconds of each STEADY-STATE call (band recognition), for
    /// p50/p95 — the number the latency gate reads.
    pub call_secs: Vec<f64>,
    /// Wall seconds of calibration + synchronous full-frame calls — the
    /// LOAD_S of this loop: one-time/async-maintenance cost, reported
    /// beside steady p95, never inside it.
    pub full_secs: Vec<f64>,
    /// Background sweeps completed (band geometry refreshes).
    pub sweeps_landed: usize,
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

pub struct LiveSession {
    engine: std::sync::Arc<dyn OcrEngine + Send + Sync>,
    opts: OcrOptions,
    cfg: LiveConfig,
    prev_gray: Option<Vec<f32>>,
    prev_out: OcrOutput,
    prev_text: String,
    /// Open span: (start time, text).
    open: Option<(f64, String)>,
    segments: Vec<TimedSegment<String>>,
    /// In-flight background full-frame sweep (band-geometry refresh). Its
    /// result replaces the band set when it lands; it is NEVER on the
    /// serving path — that is what moves full-frame cost out of steady p95.
    pending_sweep: Option<std::thread::JoinHandle<Result<OcrOutput>>>,
    /// Auto-ROI bands with cached per-band output blocks (absolute coords),
    /// rebuilt on every full-frame call. The dirty-band gate re-recognizes
    /// ONLY bands whose pixels moved: on single-slot HUD changes that is 1/N
    /// of the frame's work.
    bands: Vec<Band>,
    pub stats: LiveStats,
}

#[derive(Clone)]
struct Band {
    y0: usize,
    y1: usize,
    /// Cached output for this band, bboxes in FRAME coordinates.
    cached: Vec<ffai_core::types::OcrBlock>,
}

/// Build bands from a full-frame result and cache each band's blocks:
/// loose +-8 px unions of line y-bands (the harvest's construction), output
/// lines assigned to their band for the dirty-band cache.
fn calibrate_bands(full: &OcrOutput, frame_h: usize) -> Vec<Band> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for l in full.lines() {
        if let Some(b) = &l.bbox {
            let (y0, y1) = ((b.y - 8.0).max(0.0) as usize, ((b.y + b.height + 8.0) as usize).min(frame_h));
            if let Some(x) = spans.iter_mut().find(|x| y0 <= x.1 && y1 >= x.0) {
                x.0 = x.0.min(y0);
                x.1 = x.1.max(y1);
            } else {
                spans.push((y0, y1));
            }
        }
    }
    spans.sort_unstable();
    spans
        .into_iter()
        .map(|(y0, y1)| {
            let lines: Vec<_> = full
                .lines()
                .filter(|l| {
                    l.bbox.as_ref().is_some_and(|b| {
                        let c = b.y + b.height / 2.0;
                        c >= y0 as f32 && c < y1 as f32
                    })
                })
                .cloned()
                .collect();
            Band { y0, y1, cached: vec![ffai_core::types::OcrBlock { lines, bbox: None }] }
        })
        .collect()
}

/// Copy a horizontal band of rows out of a frame (any pixel format).
fn crop_rows(img: &ImageBuffer, y0: usize, y1: usize) -> ImageBuffer {
    let bpp = img.format.bytes_per_pixel();
    let stride = img.width as usize * bpp;
    let y1 = y1.min(img.height as usize);
    let y0 = y0.min(y1);
    ImageBuffer {
        width: img.width,
        height: (y1 - y0) as u32,
        format: img.format,
        data: img.data[y0 * stride..y1 * stride].to_vec(),
    }
}

impl LiveSession {
    pub fn new(
        engine: std::sync::Arc<dyn OcrEngine + Send + Sync>,
        opts: OcrOptions,
        cfg: LiveConfig,
    ) -> Self {
        LiveSession {
            engine,
            opts,
            cfg,
            prev_gray: None,
            prev_out: OcrOutput::default(),
            prev_text: String::new(),
            open: None,
            segments: Vec::new(),
            pending_sweep: None,
            bands: Vec::new(),
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
        // One diff pass feeds three decisions: the global gate, per-band
        // dirtiness, and the outside-band escape (a change where no band is
        // means a NEW element: full sweep now, not at the next scheduled one).
        let w = img.width as usize;
        let (mut global_changed, mut outside_changed) = (0usize, 0usize);
        let mut band_changed = vec![0usize; self.bands.len()];
        if let Some(prev) = self.prev_gray.as_ref().filter(|p| p.len() == gray.len()) {
            let delta = self.cfg.pixel_delta;
            for (i, (a, b)) in prev.iter().zip(&gray).enumerate() {
                if (a - b).abs() > delta {
                    global_changed += 1;
                    let y = i / w;
                    match self.bands.iter().position(|bd| y >= bd.y0 && y < bd.y1) {
                        Some(k) => band_changed[k] += 1,
                        None => outside_changed += 1,
                    }
                }
            }
        } else {
            global_changed = usize::MAX; // first frame: always process
        }
        let frac = self.cfg.change_fraction as f64;
        if (global_changed as f64) < gray.len() as f64 * frac {
            self.stats.gated += 1;
            return Ok(&self.prev_out);
        }

        // Harvest a finished background sweep: refresh band geometry.
        if self.pending_sweep.as_ref().is_some_and(|h| h.is_finished()) {
            if let Some(handle) = self.pending_sweep.take() {
                if let Ok(Ok(full)) = handle.join() {
                    // GEOMETRY ONLY. The sweep ran on an older frame, so its
                    // TEXT is stale by construction — caching it clobbered
                    // fresh dirty-band re-reads and resurrected old text
                    // (measured: CER 8.74% vs 1.85%, a swallowed segment).
                    // Overlapping bands inherit their current (fresher)
                    // caches; only genuinely new bands take the sweep's
                    // lines, and they'll re-read on their first dirty frame.
                    let mut new_bands = calibrate_bands(&full, img.height as usize);
                    for nb in &mut new_bands {
                        if let Some(ob) = self
                            .bands
                            .iter()
                            .find(|ob| nb.y0 < ob.y1 && nb.y1 > ob.y0)
                        {
                            nb.cached = ob.cached.clone();
                            nb.y0 = nb.y0.min(ob.y0);
                            nb.y1 = nb.y1.max(ob.y1);
                        }
                    }
                    self.bands = new_bands;
                    self.stats.sweeps_landed += 1;
                }
            }
        }

        let bands_ready = self.cfg.auto_roi
            && !self.bands.is_empty()
            && self.stats.ocr_calls >= self.cfg.calib_calls;
        // Outside-band change: something appeared where no band is; the band
        // set is stale by construction — a sweep must run NOW, synchronously,
        // or the new element would be served stale.
        let new_element = global_changed != usize::MAX
            && (outside_changed as f64) >= gray.len() as f64 * frac;
        let use_bands = bands_ready && !new_element;

        // Periodic sweep: dispatched in the BACKGROUND on its schedule; the
        // serving path below stays on bands. At most one in flight.
        if use_bands
            && self.pending_sweep.is_none()
            && (self.stats.ocr_calls + 1) % self.cfg.full_sweep_every.max(1) == 0
        {
            let engine = self.engine.clone();
            let opts = self.opts.clone();
            let frame = img.clone();
            self.pending_sweep =
                Some(std::thread::spawn(move || engine.recognize(&frame, &opts)));
        }

        let t0 = std::time::Instant::now();
        let out = if use_bands {
            self.stats.roi_calls += 1;
            // Re-recognize DIRTY bands only, IN PARALLEL (a multi-slot
            // change re-reads each dirty band independently); clean bands
            // answer from cache.
            use rayon::prelude::*;
            let dirty: Vec<usize> = (0..self.bands.len())
                .filter(|&k| {
                    let band_px = (self.bands[k].y1 - self.bands[k].y0) * w;
                    (band_changed[k] as f64) >= (band_px as f64 * frac).max(1.0)
                })
                .collect();
            self.stats.dirty_bands += dirty.len();
            let engine = &self.engine;
            let opts = &self.opts;
            let fresh: Vec<(usize, Vec<ffai_core::types::OcrBlock>)> = dirty
                .par_iter()
                .map(|&k| -> Result<_> {
                    let (y0, y1) = (self.bands[k].y0, self.bands[k].y1);
                    let sub = crop_rows(img, y0, y1);
                    let band_out = engine.recognize(&sub, opts)?;
                    let blocks = band_out
                        .blocks
                        .into_iter()
                        .map(|mut block| {
                            for line in &mut block.lines {
                                if let Some(b) = &mut line.bbox {
                                    b.y += y0 as f32;
                                }
                            }
                            block
                        })
                        .collect();
                    Ok((k, blocks))
                })
                .collect::<Result<Vec<_>>>()?;
            for (k, blocks) in fresh {
                self.bands[k].cached = blocks;
            }
            OcrOutput { blocks: self.bands.iter().flat_map(|b| b.cached.clone()).collect() }
        } else {
            let full = self.engine.recognize(img, &self.opts)?;
            self.bands = calibrate_bands(&full, img.height as usize);
            full
        };
        if use_bands {
            self.stats.call_secs.push(t0.elapsed().as_secs_f64());
        } else {
            self.stats.full_secs.push(t0.elapsed().as_secs_f64());
        }
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
        let eng = std::sync::Arc::new(CountingEngine(AtomicUsize::new(0)));
        let mut s = LiveSession::new(eng.clone(), OcrOptions::default(), LiveConfig::default());
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
