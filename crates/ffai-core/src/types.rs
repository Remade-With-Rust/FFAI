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
        };
        let srt = t.to_srt();
        assert!(srt.contains("00:00:01,500 --> 00:00:03,250"));
        assert!(srt.contains("hello world"));
    }
}
