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

/// One recognized run of text (a word, line, or block depending on engine).
#[derive(Debug, Clone, PartialEq)]
pub struct TextSpan {
    pub text: String,
    pub bbox: Option<BoundingBox>,
    pub confidence: Option<f32>,
}

/// OCR output: spans in reading order as produced by the engine.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OcrOutput {
    pub spans: Vec<TextSpan>,
}

impl OcrOutput {
    pub fn text(&self) -> String {
        self.spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
