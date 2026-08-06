//! Shared media and AI result types.
//!
//! These are the "packets and frames" of FFai: every engine consumes and
//! produces these types, which is what lets engines compose into pipelines.
//! Timestamps are `f64` seconds throughout (Whisper convention).

/// Interleaved PCM audio, always `f32` in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    /// Interleaved samples: `[L, R, L, R, ...]` for stereo.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
    }

    /// Downmix to mono by averaging channels (what ASR models expect).
    pub fn to_mono(&self) -> AudioBuffer {
        if self.channels <= 1 {
            return self.clone();
        }
        let ch = self.channels as usize;
        let samples = self
            .samples
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect();
        AudioBuffer { samples, sample_rate: self.sample_rate, channels: 1 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Gray8,
}

impl PixelFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 => 4,
            PixelFormat::Gray8 => 1,
        }
    }
}

/// A decoded raster image (row-major, tightly packed).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageBuffer {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: Vec<u8>,
}

/// A decoded video frame with its presentation time.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoFrame {
    pub image: ImageBuffer,
    /// Presentation timestamp in seconds.
    pub timestamp: f64,
}

/// A value anchored to a time range — transcript lines, captions, chapters.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedSegment<T> {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    pub value: T,
    pub confidence: Option<f32>,
}

/// ASR output: ordered timed text segments.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
    /// BCP-47-ish language tag if detected/forced (e.g. "en").
    pub language: Option<String>,
    pub segments: Vec<TimedSegment<String>>,
    /// Per-word timings from forced alignment.
    ///
    /// `None` means word timestamps were **not requested** — an absent
    /// result, not an empty one. `Some(vec![])` means they were requested and
    /// nothing was produced, which is a different fact and usually a bug
    /// worth seeing. Collapsing the two into a bare `Vec` would make a
    /// skipped stage indistinguishable from a stage that found nothing,
    /// which is the same class of mistake as a skipped gate reading as a
    /// pass.
    pub words: Option<Vec<TimedSegment<String>>>,
    /// Speaker turns from diarization, each value a label like `SPEAKER_00`.
    ///
    /// `None` when not requested, exactly as [`Self::words`]. Turns are kept
    /// as their own timeline rather than stamped onto segments because the
    /// two do not align: one segment can span a speaker change, and one
    /// speaker turn can cover several segments. Collapsing them would force a
    /// lossy choice at the wrong moment.
    pub speakers: Option<Vec<TimedSegment<String>>>,
}

impl Transcript {
    /// Plain text, one segment per line.
    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.value.trim())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// SubRip subtitle rendering.
    pub fn to_srt(&self) -> String {
        let mut out = String::new();
        for (i, seg) in self.segments.iter().enumerate() {
            out.push_str(&format!(
                "{}\n{} --> {}\n{}\n\n",
                i + 1,
                srt_time(seg.start),
                srt_time(seg.end),
                seg.value.trim()
            ));
        }
        out
    }

    /// WebVTT.
    ///
    /// When word timings are present each cue carries inline `<mm:ss.mmm>`
    /// tags ahead of each word — the form players use to highlight a caption
    /// word by word as it is spoken. Without them this is a plain cue list,
    /// which is what a segment-level transcript can honestly support.
    pub fn to_vtt(&self) -> String {
        let mut out = String::from("WEBVTT\n\n");
        for seg in &self.segments {
            out.push_str(&format!("{} --> {}\n", vtt_time(seg.start), vtt_time(seg.end)));
            // A word belongs to the cue whose span contains its start. Using
            // the start rather than any overlap keeps each word in exactly one
            // cue, so nothing is duplicated across a boundary.
            let inline: Vec<&TimedSegment<String>> = self
                .words
                .iter()
                .flatten()
                .filter(|w| w.start >= seg.start && w.start < seg.end)
                .collect();
            if inline.is_empty() {
                out.push_str(seg.value.trim());
            } else {
                for w in inline {
                    out.push_str(&format!("<{}>{} ", vtt_time(w.start), w.value.trim()));
                }
            }
            out.push_str("\n\n");
        }
        out
    }

    /// JSON, hand-rolled.
    ///
    /// `ffai-core` depends on candle and `thiserror` and nothing else; adding
    /// a serialisation framework to a published crate so one function can emit
    /// an object this shallow is a poor trade. `words` is `null` when
    /// alignment was not requested and `[]` when it was and found nothing —
    /// the distinction the field exists to carry.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n  \"language\": ");
        match &self.language {
            Some(l) => out.push_str(&format!("\"{}\"", json_escape(l))),
            None => out.push_str("null"),
        }
        out.push_str(",\n  \"segments\": [");
        for (i, s) in self.segments.iter().enumerate() {
            out.push_str(if i == 0 { "\n" } else { ",\n" });
            out.push_str(&json_timed(s, 4));
        }
        out.push_str(if self.segments.is_empty() { "]" } else { "\n  ]" });
        out.push_str(",\n  \"words\": ");
        match &self.words {
            None => out.push_str("null"),
            Some(ws) => {
                out.push('[');
                for (i, w) in ws.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    out.push_str(&json_timed(w, 4));
                }
                out.push_str(if ws.is_empty() { "]" } else { "\n  ]" });
            }
        }
        out.push_str(",\n  \"speakers\": ");
        match &self.speakers {
            None => out.push_str("null"),
            Some(ts) => {
                out.push('[');
                for (i, t) in ts.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    out.push_str(&json_timed(t, 4));
                }
                out.push_str(if ts.is_empty() { "]" } else { "\n  ]" });
            }
        }
        out.push_str("\n}\n");
        out
    }
}

fn json_timed(s: &TimedSegment<String>, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let conf = match s.confidence {
        Some(c) => format!("{c:.4}"),
        None => "null".to_string(),
    };
    format!(
        "{pad}{{ \"start\": {:.3}, \"end\": {:.3}, \"text\": \"{}\", \"confidence\": {conf} }}",
        s.start,
        s.end,
        json_escape(s.value.trim())
    )
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // Control characters are not legal raw in JSON strings.
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn vtt_time(secs: f64) -> String {
    let ms = (secs.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

fn srt_time(secs: f64) -> String {
    let ms = (secs.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02},{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

/// Axis-aligned box in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// One recognized word.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrWord {
    pub text: String,
    pub bbox: Option<BoundingBox>,
    /// Recognition confidence in 0..1, when the engine reports one.
    pub confidence: Option<f32>,
}

/// One text line: the recognition unit of most OCR stacks.
///
/// `text` is authoritative; `words` is optional detail (empty when the
/// engine recognizes whole lines without word segmentation, as line-level
/// transformer decoders do).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OcrLine {
    pub text: String,
    pub words: Vec<OcrWord>,
    pub bbox: Option<BoundingBox>,
    pub confidence: Option<f32>,
}

/// One block: lines that belong together (a paragraph, a column cell, a HUD
/// element). Region classification (title/table/figure…) arrives as typed
/// payloads with the DOCUMENT milestone — deliberately absent from v1 (see
/// carmenta-mission-plan §2.1).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OcrBlock {
    pub lines: Vec<OcrLine>,
    pub bbox: Option<BoundingBox>,
}

/// OCR output for one page/frame: blocks in reading order.
///
/// **Reading order is the Vec order at every level** — blocks within the
/// output, lines within a block, words within a line. An explicit sequence,
/// not a geometric convention, so a future layout stage can reorder without
/// changing the type. LIVE wraps this in `TimedSegment<OcrOutput>` for timed
/// tracks; the hierarchy (page → block → line → word) is the AVFrame of the
/// Carmenta mission and every engine and the bench harness build against it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OcrOutput {
    pub blocks: Vec<OcrBlock>,
}

impl OcrOutput {
    /// Flat text: lines joined with `\n`, blocks separated by a blank line.
    pub fn text(&self) -> String {
        self.blocks
            .iter()
            .map(|b| b.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n"))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Convenience constructor: one block per line, no geometry — the
    /// degenerate shape for engines (or tests) without layout information.
    pub fn from_lines<I: IntoIterator<Item = String>>(lines: I) -> Self {
        OcrOutput {
            blocks: vec![OcrBlock {
                lines: lines
                    .into_iter()
                    .map(|text| OcrLine { text, ..Default::default() })
                    .collect(),
                bbox: None,
            }],
        }
    }

    /// Every line in reading order, across blocks.
    pub fn lines(&self) -> impl Iterator<Item = &OcrLine> {
        self.blocks.iter().flat_map(|b| b.lines.iter())
    }
}

/// How a detector letterboxed an image, kept so boxes can be mapped back.
///
/// The inverse transform **travels with the output** rather than living in
/// the caller's head: a detection in letterboxed coordinates is not wrong,
/// it is unusable, and every consumer would otherwise re-derive the same
/// arithmetic. `scale` is the factor the original was multiplied by;
/// `pad_x`/`pad_y` are the pixels added on the left/top.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    /// Original image size, so mapping back can clamp to it.
    pub orig_width: u32,
    pub orig_height: u32,
}

impl Letterbox {
    /// Map a point from letterboxed input coordinates back to the original,
    /// clamped to the original image bounds.
    pub fn invert(&self, x: f32, y: f32) -> (f32, f32) {
        let ox = ((x - self.pad_x) / self.scale).clamp(0.0, self.orig_width as f32);
        let oy = ((y - self.pad_y) / self.scale).clamp(0.0, self.orig_height as f32);
        (ox, oy)
    }
}

/// One detected object.
///
/// Geometry is **xyxy in original-image pixels** — the form every consumer
/// and every scorer wants. `class_id` indexes the model's own class list
/// (surfaced by the engine's manifest); `track_id` is populated only by a
/// streaming loop that associates across frames.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub class_id: u32,
    pub confidence: f32,
    pub track_id: Option<u64>,
}

impl Detection {
    pub fn width(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y1 - self.y0).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Intersection over union with another detection, ignoring class.
    pub fn iou(&self, other: &Detection) -> f32 {
        let ix = (self.x1.min(other.x1) - self.x0.max(other.x0)).max(0.0);
        let iy = (self.y1.min(other.y1) - self.y0.max(other.y0)).max(0.0);
        let inter = ix * iy;
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    pub fn as_bbox(&self) -> BoundingBox {
        BoundingBox { x: self.x0, y: self.y0, width: self.width(), height: self.height() }
    }
}

/// What a detection engine returns for one image.
///
/// Detections are in **confidence-descending order** — the order the
/// end-to-end heads already produce, and the order every consumer that
/// takes a top-N expects. Class *names* live on the engine (they come from
/// the weight manifest), not on every output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DetectOutput {
    pub detections: Vec<Detection>,
    /// The letterbox that produced these coordinates, when one was applied.
    /// `None` means the engine consumed the image at native resolution.
    pub letterbox: Option<Letterbox>,
}

impl DetectOutput {
    /// Keep only detections at or above `min_confidence`.
    pub fn filter_confidence(&mut self, min_confidence: f32) {
        self.detections.retain(|d| d.confidence >= min_confidence);
    }

    /// Greedy class-wise non-maximum suppression.
    ///
    /// Not used by the NMS-free one-to-one path — it exists for engines
    /// whose head emits overlapping candidates, and so a LIVE loop can
    /// merge across sources. Assumes the confidence-descending order the
    /// type documents.
    pub fn suppress_overlaps(&mut self, iou_threshold: f32) {
        let mut kept: Vec<Detection> = Vec::with_capacity(self.detections.len());
        for det in self.detections.drain(..) {
            if !kept
                .iter()
                .any(|k| k.class_id == det.class_id && k.iou(&det) > iou_threshold)
            {
                kept.push(det);
            }
        }
        self.detections = kept;
    }

    /// Drop DUPLICATE boxes while sparing genuinely overlapping objects.
    ///
    /// The one-to-one head is NMS-free by construction, so plain
    /// [`suppress_overlaps`](Self::suppress_overlaps) is off by default — and
    /// rightly, since running it costs IDF1. But the construction does not
    /// hold in practice: on MOT17 our detections carried **1285 intra-frame
    /// pairs above IoU 0.7 against the reference's 855** on identical frames.
    ///
    /// Labelling every such pair against ground truth (a pair is a DUPLICATE
    /// if both boxes match the same object, GENUINE if they match different
    /// ones) gave 9,582 duplicates against 562 genuine, and three observable
    /// features that separate them — standardised mean difference in brackets:
    ///
    /// | feature | duplicate | genuine | sep |
    /// |---|---|---|---|
    /// | IoU | 0.822 | 0.656 | 1.37 |
    /// | area ratio | 0.863 | 0.775 | 0.71 |
    /// | height ratio | 0.942 | 0.900 | 0.44 |
    ///
    /// Confidence gap separates nothing (0.04) and box SIZE separates nothing
    /// (0.09) — both were measured and dropped rather than assumed.
    ///
    /// A conjunction of the three, thresholds chosen by scoring candidates on
    /// **IDF1 itself rather than classifier accuracy**, measured across all
    /// seven MOT17 sequences:
    ///
    /// ```text
    /// IDF1 32.82 -> 33.28   MOTA 18.92 -> 19.24   ID switches 375 -> 239
    /// 02 +2.84  04 -0.22  05 -1.29  09 -0.33  10 +1.48  11 +0.54  13 -0.02
    /// ```
    ///
    /// **Sequence 05 loses 1.29 and that is not noise** — it is stable across
    /// every gate formulation tried: absolute thresholds, a per-frame gate, and
    /// a population-relative percentile cut on each sequence's own overlap
    /// distribution. 05 carries boxes at 11.9 % of frame against 1.0-5.5 %
    /// elsewhere, so its people genuinely overlap more, and no observable
    /// feature separated its real pairs from duplicates. Recorded as a known
    /// cost rather than hidden: whoever runs large-subject footage pays it.
    ///
    /// The percentile form scored the same (33.29) and was rejected for a
    /// reason that has nothing to do with accuracy — it needs the whole clip's
    /// overlap distribution before it can threshold frame 1, so it cannot run
    /// online. This one is causal and per-frame.
    pub fn suppress_duplicates(&mut self, iou: f32, area_ratio: f32, height_ratio: f32) {
        let mut kept: Vec<Detection> = Vec::with_capacity(self.detections.len());
        for det in self.detections.drain(..) {
            let a_det = (det.x1 - det.x0).max(0.0) * (det.y1 - det.y0).max(0.0);
            let h_det = (det.y1 - det.y0).max(0.0);
            let dup = kept.iter().any(|k| {
                if k.class_id != det.class_id || k.iou(&det) <= iou {
                    return false;
                }
                let a_k = (k.x1 - k.x0).max(0.0) * (k.y1 - k.y0).max(0.0);
                let h_k = (k.y1 - k.y0).max(0.0);
                let ar = a_det.min(a_k) / a_det.max(a_k).max(f32::MIN_POSITIVE);
                let hr = h_det.min(h_k) / h_det.max(h_k).max(f32::MIN_POSITIVE);
                ar > area_ratio && hr > height_ratio
            });
            if !dup {
                kept.push(det);
            }
        }
        self.detections = kept;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x0: f32, y0: f32, x1: f32, y1: f32, class_id: u32, confidence: f32) -> Detection {
        Detection { x0, y0, x1, y1, class_id, confidence, track_id: None }
    }

    /// The gate must drop a near-identical twin and SPARE two real people who
    /// happen to overlap. Both halves matter: the second is why plain NMS was
    /// rejected, and a change that only satisfies the first would look correct
    /// on a duplicate-only fixture while costing a trajectory in the field.
    #[test]
    fn suppress_duplicates_spares_genuine_overlap() {
        // A duplicate: same object, nearly the same box.
        let mut out = DetectOutput {
            detections: vec![
                det(100.0, 100.0, 140.0, 220.0, 0, 0.90),
                det(102.0, 101.0, 141.0, 219.0, 0, 0.60),
            ],
            letterbox: None,
        };
        out.suppress_duplicates(0.80, 0.88, 0.95);
        assert_eq!(out.detections.len(), 1, "near-identical twin should be dropped");

        // Genuine overlap: one person in front of another, so the boxes
        // overlap heavily but differ in height — a real occlusion.
        let mut out = DetectOutput {
            detections: vec![
                det(100.0, 100.0, 140.0, 220.0, 0, 0.90),
                det(104.0, 60.0, 144.0, 215.0, 0, 0.70),
            ],
            letterbox: None,
        };
        out.suppress_duplicates(0.80, 0.88, 0.95);
        assert_eq!(out.detections.len(), 2, "differing heights means two people, not a duplicate");
    }

    #[test]
    fn letterbox_inverts_to_original_pixels() {
        // A 586x640 image letterboxed into 640x640: scale 1.0, pad_x 27.
        let lb = Letterbox {
            scale: 1.0,
            pad_x: 27.0,
            pad_y: 0.0,
            orig_width: 586,
            orig_height: 640,
        };
        assert_eq!(lb.invert(27.0, 0.0), (0.0, 0.0));
        assert_eq!(lb.invert(127.0, 50.0), (100.0, 50.0));
        // Clamped, never negative or past the original bounds.
        assert_eq!(lb.invert(0.0, 0.0), (0.0, 0.0));
        assert_eq!(lb.invert(9999.0, 9999.0), (586.0, 640.0));
    }

    #[test]
    fn letterbox_inverts_a_scaled_box() {
        let lb = Letterbox {
            scale: 0.5,
            pad_x: 10.0,
            pad_y: 20.0,
            orig_width: 1000,
            orig_height: 800,
        };
        let (x, y) = lb.invert(110.0, 120.0);
        assert!((x - 200.0).abs() < 1e-5, "got {x}");
        assert!((y - 200.0).abs() < 1e-5, "got {y}");
    }

    #[test]
    fn iou_matches_hand_computed() {
        let a = det(0.0, 0.0, 10.0, 10.0, 0, 0.9);
        let b = det(0.0, 0.0, 10.0, 5.0, 0, 0.8);
        // inter 50, union 100
        assert!((a.iou(&b) - 0.5).abs() < 1e-6);
        let far = det(100.0, 100.0, 110.0, 110.0, 0, 0.7);
        assert_eq!(a.iou(&far), 0.0);
    }

    #[test]
    fn suppression_is_class_wise() {
        let mut out = DetectOutput {
            detections: vec![
                det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
                det(0.5, 0.5, 10.0, 10.0, 0, 0.8), // same class, heavy overlap -> dropped
                det(0.5, 0.5, 10.0, 10.0, 1, 0.7), // different class -> kept
            ],
            letterbox: None,
        };
        out.suppress_overlaps(0.5);
        assert_eq!(out.detections.len(), 2);
        assert_eq!(out.detections[0].class_id, 0);
        assert_eq!(out.detections[1].class_id, 1);
    }

    #[test]
    fn confidence_filter_keeps_the_threshold_itself() {
        let mut out = DetectOutput {
            detections: vec![det(0.0, 0.0, 1.0, 1.0, 0, 0.25), det(0.0, 0.0, 1.0, 1.0, 0, 0.24)],
            letterbox: None,
        };
        out.filter_confidence(0.25);
        assert_eq!(out.detections.len(), 1);
    }

    #[test]
    fn mono_downmix_averages_channels() {
        let stereo = AudioBuffer {
            samples: vec![1.0, 0.0, 0.5, 0.5],
            sample_rate: 16_000,
            channels: 2,
        };
        let mono = stereo.to_mono();
        assert_eq!(mono.channels, 1);
        assert_eq!(mono.samples, vec![0.5, 0.5]);
    }

    #[test]
    fn srt_renders_timestamps() {
        let t = Transcript {
            language: Some("en".into()),
            segments: vec![TimedSegment {
                start: 1.5,
                end: 3.25,
                value: "hello world".into(),
                confidence: None,
            }],
            words: None,
            speakers: None,
        };
        let srt = t.to_srt();
        assert!(srt.contains("00:00:01,500 --> 00:00:03,250"));
        assert!(srt.contains("hello world"));
    }
}

#[cfg(test)]
mod transcript_output_tests {
    use super::*;

    fn seg(start: f64, end: f64, text: &str) -> TimedSegment<String> {
        TimedSegment { start, end, value: text.into(), confidence: None }
    }

    fn word(start: f64, end: f64, text: &str) -> TimedSegment<String> {
        TimedSegment { start, end, value: text.into(), confidence: Some(0.9) }
    }

    #[test]
    fn vtt_without_words_is_a_plain_cue_list() {
        let t = Transcript {
            language: Some("en".into()),
            segments: vec![seg(1.5, 3.25, "hello world")],
            words: None,
            speakers: None,
        };
        let vtt = t.to_vtt();
        assert!(vtt.starts_with("WEBVTT"), "{vtt}");
        assert!(vtt.contains("00:00:01.500 --> 00:00:03.250"), "{vtt}");
        assert!(vtt.contains("hello world"), "{vtt}");
        assert!(!vtt.contains("<00:"), "no inline tags without words: {vtt}");
    }

    #[test]
    fn vtt_with_words_carries_inline_timing_tags() {
        let t = Transcript {
            language: Some("en".into()),
            segments: vec![seg(1.0, 3.0, "hello world")],
            words: Some(vec![word(1.0, 1.4, "hello"), word(2.0, 2.5, "world")]),
            speakers: None,
        };
        let vtt = t.to_vtt();
        assert!(vtt.contains("<00:00:01.000>hello"), "{vtt}");
        assert!(vtt.contains("<00:00:02.000>world"), "{vtt}");
    }

    #[test]
    fn a_word_lands_in_exactly_one_cue() {
        // The boundary word starts exactly at the second cue's start; it must
        // appear once, in that cue, not in both.
        let t = Transcript {
            language: None,
            segments: vec![seg(0.0, 2.0, "a"), seg(2.0, 4.0, "b")],
            words: Some(vec![word(2.0, 2.5, "boundary")]),
            speakers: None,
        };
        let vtt = t.to_vtt();
        assert_eq!(vtt.matches("boundary").count(), 1, "{vtt}");
    }

    #[test]
    fn json_distinguishes_not_requested_from_found_nothing() {
        let absent = Transcript { language: None, segments: vec![], words: None, speakers: None };
        assert!(absent.to_json().contains("\"words\": null"), "{}", absent.to_json());

        let empty = Transcript { language: None, segments: vec![], words: Some(vec![]), speakers: None };
        assert!(empty.to_json().contains("\"words\": []"), "{}", empty.to_json());
    }

    #[test]
    fn json_escapes_quotes_backslashes_and_controls() {
        let t = Transcript {
            language: None,
            segments: vec![seg(0.0, 1.0, "he said \"hi\"\\ok\u{7}")],
            words: None,
            speakers: None,
        };
        let j = t.to_json();
        assert!(j.contains(r#"\"hi\""#), "{j}");
        assert!(j.contains(r"\ok"), "{j}");
        assert!(j.contains(r"\u0007"), "{j}");
        // Nothing raw that would break a parser.
        assert!(!j.contains('\u{7}'), "{j}");
    }

    #[test]
    fn json_carries_word_timings_and_confidence() {
        let t = Transcript {
            language: Some("en".into()),
            segments: vec![seg(0.0, 1.0, "hi")],
            words: Some(vec![word(0.1, 0.4, "hi")]),
            speakers: None,
        };
        let j = t.to_json();
        assert!(j.contains("\"start\": 0.100"), "{j}");
        assert!(j.contains("\"end\": 0.400"), "{j}");
        assert!(j.contains("\"confidence\": 0.9000"), "{j}");
    }
}

#[cfg(test)]
mod speaker_output_tests {
    use super::*;

    #[test]
    fn json_distinguishes_diarization_not_requested_from_no_turns_found() {
        let absent = Transcript::default();
        assert!(absent.to_json().contains("\"speakers\": null"), "{}", absent.to_json());

        let empty = Transcript { speakers: Some(vec![]), ..Default::default() };
        assert!(empty.to_json().contains("\"speakers\": []"), "{}", empty.to_json());
    }

    #[test]
    fn json_carries_speaker_turns() {
        let t = Transcript {
            speakers: Some(vec![
                TimedSegment { start: 0.0, end: 2.5, value: "SPEAKER_00".into(), confidence: None },
                TimedSegment { start: 2.5, end: 5.0, value: "SPEAKER_01".into(), confidence: None },
            ]),
            ..Default::default()
        };
        let j = t.to_json();
        assert!(j.contains("SPEAKER_00"), "{j}");
        assert!(j.contains("SPEAKER_01"), "{j}");
        assert!(j.contains("\"end\": 5.000"), "{j}");
    }
}
