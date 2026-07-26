//! # ffai-media
//!
//! Media ingest/egress for FFai — the `libavformat` seat.
//!
//! Policy: all container/codec work routes through **remade_ffmpeg_rs**
//! (`rff-*` crates) as the default backend — we own it, it is pure Rust, and
//! it keeps the zero-C/C++ promise. Phase 0 ships native WAV support (the one
//! format every ASR/TTS engine needs on day one); everything else returns a
//! clear "pending rff integration" error rather than silently failing.

use std::path::Path;

use ffai_core::error::{Error, Result};
use ffai_core::types::{AudioBuffer, ImageBuffer, VideoFrame};

/// Load an audio file into a normalized f32 [`AudioBuffer`].
///
/// Phase 0: WAV only (PCM int/float). Other containers/codecs land with the
/// remade_ffmpeg_rs integration in Phase 1.
pub fn load_audio(path: &Path) -> Result<AudioBuffer> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "wav" => load_wav(path),
        other => Err(Error::Media(format!(
            "`.{other}` decode is not wired yet — audio beyond WAV arrives with the \
             remade_ffmpeg_rs (rff) backend in Phase 1; for now convert with \
             `ffmpeg -i in.{other} -ar 16000 -ac 1 out.wav`"
        ))),
    }
}

fn load_wav(path: &Path) -> Result<AudioBuffer> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| Error::Media(format!("WAV open failed: {e}")))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::Media(format!("WAV read failed: {e}")))?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| Error::Media(format!("WAV read failed: {e}")))?
        }
    };
    Ok(AudioBuffer {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

/// Write an [`AudioBuffer`] as 32-bit float WAV.
pub fn save_wav(path: &Path, audio: &AudioBuffer) -> Result<()> {
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| Error::Media(format!("WAV create failed: {e}")))?;
    for &s in &audio.samples {
        writer
            .write_sample(s)
            .map_err(|e| Error::Media(format!("WAV write failed: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| Error::Media(format!("WAV finalize failed: {e}")))?;
    Ok(())
}

/// Decode a still image. Pending the rff image decoders (PNG/JPEG/WebP/…).
pub fn load_image(path: &Path) -> Result<ImageBuffer> {
    Err(Error::Media(format!(
        "image decode for `{}` is not wired yet — lands with the remade_ffmpeg_rs \
         image decoders (Phase 3, ahead of Carmenta OCR going live)",
        path.display()
    )))
}

/// Sample frames from a video at `fps` frames/second (for Argus video
/// understanding). Pending the rff demux/decode integration.
pub fn sample_frames(path: &Path, fps: f64) -> Result<Vec<VideoFrame>> {
    let _ = fps;
    Err(Error::Media(format!(
        "video frame sampling for `{}` is not wired yet — lands with the \
         remade_ffmpeg_rs (rff) backend",
        path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_preserves_samples() {
        let audio = AudioBuffer {
            samples: (0..1600).map(|i| (i as f32 / 100.0).sin() * 0.5).collect(),
            sample_rate: 16_000,
            channels: 1,
        };
        let path = std::env::temp_dir().join("ffai_media_roundtrip_test.wav");
        save_wav(&path, &audio).unwrap();
        let loaded = load_audio(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.sample_rate, 16_000);
        assert_eq!(loaded.channels, 1);
        assert_eq!(loaded.samples.len(), audio.samples.len());
        // f32 WAV is lossless.
        assert_eq!(loaded.samples, audio.samples);
    }

    #[test]
    fn unknown_extension_names_the_backend_plan() {
        let err = load_audio(Path::new("clip.mp3")).unwrap_err();
        assert!(err.to_string().contains("remade_ffmpeg_rs"));
    }
}
