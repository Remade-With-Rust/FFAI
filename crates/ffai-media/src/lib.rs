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

/// Decode a still image — PNG and JPEG today; WebP/AVIF/GIF follow.
///
/// **JPEG comes from home** (Principle 7): it decodes through
/// `rff-codec-jpeg` in our own pure-Rust ffmpeg, on rff's `CodecRegistry`
/// seam, so improvements to rff land here without a change on this side.
///
/// **PNG has not moved yet, and the reason is measured rather than
/// cautious.** `rff-codec-png` expands grayscale to packed `Rgb24`, while
/// this path returns `Gray8` — and Carmenta's OCR consumes `load_image`, so
/// the swap changes which preprocessing branch every grayscale page takes.
/// That is a behavioural change and it needs Carmenta's corpus gates green,
/// not an assumption that it is equivalent. JPEG was safe to move first
/// because nothing decoded JPEG before, so there was no behaviour to break.
///
/// One consequence of depending on rff, recorded so it is not rediscovered:
/// rff is not on crates.io, and `cargo publish` refuses a git dependency
/// ("all dependencies must have a version requirement specified when
/// publishing"). Any FFai crate downstream of this one therefore cannot be
/// published until rff publishes — a deliberate trade of distribution for
/// owning the stack.
pub fn load_image(path: &Path) -> Result<ImageBuffer> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    match ext.as_str() {
        "png" => load_png(path),
        "jpg" | "jpeg" => load_jpeg(path),
        other => Err(Error::Media(format!(
            "`.{other}` decode is not wired yet — PNG and JPEG are supported; WebP/AVIF \
             arrive with the rff image decoders. Convert with `ffmpeg -i in.{other} out.png`"
        ))),
    }
}

/// JPEG, decoded by **our own ffmpeg** — `rff-codec-jpeg` through rff's
/// codec registry, the same path `rff` itself takes.
///
/// Going through `CodecRegistry` rather than calling the decoder directly is
/// deliberate: it is the seam every other rff codec arrives on, so adding
/// WebP/AVIF/GIF later is a `register()` line rather than another bespoke
/// function. rff always hands back packed `Rgb24` for JPEG (it expands
/// grayscale itself), so callers see one layout regardless of how the file
/// was encoded.
fn load_jpeg(path: &Path) -> Result<ImageBuffer> {
    use ffai_core::types::PixelFormat;
    use rff_codec::CodecRegistry;
    use rff_core::{CodecId, Frame, Packet};

    let data = std::fs::read(path).map_err(|e| Error::Media(format!("open failed: {e}")))?;

    let mut registry = CodecRegistry::new();
    rff_codec_jpeg::register(&mut registry);
    let mut decoder = registry
        .find_decoder(CodecId::Jpeg)
        .map_err(|e| Error::Media(format!("rff jpeg decoder: {e}")))?;

    // A still image is one packet: JPEG is self-describing, so the file IS
    // the packet (rff's own framing for still-image codecs).
    let packet = Packet { data, ..Default::default() };
    decoder.send_packet(&packet).map_err(|e| Error::Media(format!("JPEG decode: {e}")))?;
    decoder.flush();
    let frame = decoder.receive_frame().map_err(|e| Error::Media(format!("JPEG decode: {e}")))?;

    let Frame::Video(v) = frame else {
        return Err(Error::Media("rff returned an audio frame for a JPEG".into()));
    };
    let format = match v.format {
        rff_core::PixelFormat::Rgb24 => PixelFormat::Rgb8,
        rff_core::PixelFormat::Rgba => PixelFormat::Rgba8,
        other => {
            return Err(Error::Media(format!(
                "rff decoded this JPEG as {other:?}; ffai-media handles packed RGB/RGBA"
            )))
        }
    };
    let bpp = if format == PixelFormat::Rgba8 { 4 } else { 3 };
    let (w, h) = (v.width as usize, v.height as usize);
    let plane = v
        .planes
        .into_iter()
        .next()
        .ok_or_else(|| Error::Media("rff video frame has no planes".into()))?;
    let stride = v.strides.first().copied().unwrap_or(w * bpp);

    // Honour the stride rather than assuming it equals the row: rff states
    // it "may exceed width for alignment", and a packed codec today can
    // become an aligned one tomorrow without changing this signature.
    let data = if stride == w * bpp {
        plane
    } else {
        let mut packed = Vec::with_capacity(w * h * bpp);
        for row in 0..h {
            let start = row * stride;
            packed.extend_from_slice(&plane[start..start + w * bpp]);
        }
        packed
    };
    Ok(ImageBuffer { width: v.width, height: v.height, format, data })
}

fn load_png(path: &Path) -> Result<ImageBuffer> {
    use ffai_core::types::PixelFormat;
    let file = std::fs::File::open(path).map_err(|e| Error::Media(format!("open failed: {e}")))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().map_err(|e| Error::Media(format!("PNG header: {e}")))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| Error::Media(format!("PNG decode: {e}")))?;
    buf.truncate(info.buffer_size());

    // 16-bit depths and palettes are expanded by the png crate only with the
    // right transformations; keep v1 strict and honest about what it read.
    if info.bit_depth != png::BitDepth::Eight {
        return Err(Error::Media(format!("PNG bit depth {:?} unsupported (8-bit only for now)", info.bit_depth)));
    }
    let format = match info.color_type {
        png::ColorType::Grayscale => PixelFormat::Gray8,
        png::ColorType::Rgb => PixelFormat::Rgb8,
        png::ColorType::Rgba => PixelFormat::Rgba8,
        png::ColorType::GrayscaleAlpha => {
            // Drop alpha: OCR reads luminance.
            buf = buf.chunks_exact(2).map(|p| p[0]).collect();
            PixelFormat::Gray8
        }
        png::ColorType::Indexed => {
            return Err(Error::Media("indexed PNG unsupported (expand the palette first)".into()))
        }
    };
    Ok(ImageBuffer { width: info.width, height: info.height, format, data: buf })
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
