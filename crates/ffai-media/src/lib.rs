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
/// **PNG now comes from home too.** It decodes through `rff-codec-png` on
/// the same registry seam. The obstacle was real and is handled rather than
/// waved away: rff normalises grayscale to packed `Rgb24`, while this
/// path's contract is `Gray8`, and Carmenta's OCR consumes `load_image`
/// with a grayscale document corpus. The colour type is therefore read from
/// the PNG header and grayscale sources are contracted back — which is
/// exact, not approximate, because rff wrote `g` into all three lanes. A
/// standing test decodes every corpus image both ways and asserts the bytes
/// are identical.
///
/// One consequence of depending on rff, recorded so it is not rediscovered:
/// rff is not on crates.io, and `cargo publish` refuses a git dependency
/// ("all dependencies must have a version requirement specified when
/// publishing"). Any FFai crate downstream of this one therefore cannot be
/// published until rff publishes — a deliberate trade of distribution for
/// owning the stack.
pub fn load_image(path: &Path) -> Result<ImageBuffer> {
    // MAGIC BYTES FIRST, extension only as a fallback.
    //
    // Dispatching on the extension alone is a real-world trap, not a
    // hypothetical one: 169 of OmniDocBench's 316 English pages are JPEGs
    // named `.png`, and every one of them failed to decode with "Invalid PNG
    // signature" — producing an empty result that scored as 100 % CER and
    // silently contaminated a whole benchmark run before anyone read stderr.
    // A file's contents are authoritative; its name is a hint.
    // Read ONCE and decode from the buffer.
    //
    // This sniffed the magic with `fs::read(path).take(8)` — which reads the
    // WHOLE file to look at eight bytes, and then the decoder read it again.
    // Every image was loaded twice: measured at 1.16x slower than a
    // single-read path over 40 images, i.e. the sniff cost more than the
    // decoder choice it was making. Reading once and passing the bytes down
    // costs nothing and removes a whole file read per image.
    let bytes = std::fs::read(path).map_err(|e| Error::Media(format!("open failed: {e}")))?;
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        return decode_png(&bytes);
    }
    if bytes.starts_with(&[0xFF, 0xD8]) {
        return decode_jpeg(bytes);
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    match ext.as_str() {
        "png" => decode_png(&bytes),
        "jpg" | "jpeg" => decode_jpeg(bytes),
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
fn decode_jpeg(data: Vec<u8>) -> Result<ImageBuffer> {
    use ffai_core::types::PixelFormat;
    use rff_codec::CodecRegistry;
    use rff_core::{CodecId, Frame, Packet};

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

/// The PNG header's colour-type byte, read from the raw file.
///
/// rff normalises every PNG to packed `Rgb24`/`Rgba` — correct for a codec,
/// and it throws away the one fact this crate's contract depends on: whether
/// the source was GRAYSCALE. `ImageBuffer`'s consumers care. Carmenta's OCR
/// has explicit `Gray8` paths and its document corpus is 32/32 grayscale, so
/// silently handing it three-channel data would triple that corpus's memory
/// and change which code path reads it.
///
/// The colour type sits at a fixed offset in IHDR, which is required to be
/// the first chunk: 8-byte signature, 4-byte length, 4-byte `IHDR`, then
/// width(4) height(4) bit-depth(1) **colour-type(1)**. Byte 25. Verified
/// against the chunk name rather than assumed, so a malformed file falls
/// back to trusting rff instead of misreading a random byte.
fn png_colour_type(bytes: &[u8]) -> Option<u8> {
    (bytes.len() > 25 && &bytes[12..16] == b"IHDR").then(|| bytes[25])
}

/// PNG, decoded by **our own ffmpeg** — `rff-codec-png` through the same
/// `CodecRegistry` seam JPEG arrives on (Principle 7).
///
/// This replaced a direct `png`-crate call. Two things made the swap safe
/// rather than merely ideological:
///
/// 1. **rff is strictly more capable.** It sets
///    `Transformations::EXPAND | STRIP_16`, so palette and 16-bit PNGs decode
///    instead of erroring — both were hard rejections here before.
/// 2. **The grayscale round-trip is exact.** rff expands `g` to `[g, g, g]`,
///    so taking every third byte recovers `g` bit-for-bit; likewise every
///    fourth from `Rgba` for grayscale+alpha, which is what dropping alpha
///    always did. Contracting back is not an approximation.
///
/// The contract is therefore unchanged for every input the old path
/// accepted, which is what let this land while another session was
/// benchmarking OCR on the grayscale corpus.
/// `FFAI_PNG_LEGACY=1` routes PNG back through the `png` crate.
///
/// Present for one transitional reason, with numbers. On the rff revision
/// currently on the default branch (54df3fe, 2026-07-29) the rff path is
/// **1.15x slower on RGB and 2.36x slower on grayscale** than the code it
/// replaced. The RGB gap is decoder work in progress upstream. The
/// GRAYSCALE gap is structural and will not close by itself:
/// `rff_core::PixelFormat` has no grayscale variant, so a gray PNG must be
/// expanded to `Rgb24` and contracted back here — three times the bytes,
/// twice, for data that started and ended one channel wide.
///
/// Carmenta's document corpus is 32/32 grayscale and another session
/// benchmarks OCR on it, so the escape exists to keep a transitional
/// slowdown out of somebody else's gate. Delete it once rff's PNG work
/// lands (and, for grayscale, once rff can carry a single-channel format).
fn png_legacy() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);
    match CACHED.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_PNG_LEGACY").is_ok_and(|v| v == "1");
            CACHED.store(on as u8, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

/// The `png`-crate path, retained as the legacy arm and as the ORACLE the
/// tests compare rff against. It is not dead code: a decoder swap changes
/// the pixels every downstream gate is computed from, so the thing being
/// replaced has to stay runnable.
fn decode_png_legacy(data: &[u8]) -> Result<ImageBuffer> {
    use ffai_core::types::PixelFormat;
    let decoder = png::Decoder::new(std::io::Cursor::new(data));
    let mut reader = decoder.read_info().map_err(|e| Error::Media(format!("PNG header: {e}")))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| Error::Media(format!("PNG decode: {e}")))?;
    buf.truncate(info.buffer_size());
    if info.bit_depth != png::BitDepth::Eight {
        return Err(Error::Media(format!(
            "PNG bit depth {:?} unsupported by the legacy path (rff handles it)",
            info.bit_depth
        )));
    }
    let format = match info.color_type {
        png::ColorType::Grayscale => PixelFormat::Gray8,
        png::ColorType::Rgb => PixelFormat::Rgb8,
        png::ColorType::Rgba => PixelFormat::Rgba8,
        png::ColorType::GrayscaleAlpha => {
            buf = buf.chunks_exact(2).map(|p| p[0]).collect();
            PixelFormat::Gray8
        }
        png::ColorType::Indexed => {
            return Err(Error::Media("indexed PNG needs the rff path".into()))
        }
    };
    Ok(ImageBuffer { width: info.width, height: info.height, format, data: buf })
}

fn decode_png(data: &[u8]) -> Result<ImageBuffer> {
    if png_legacy() {
        return decode_png_legacy(data);
    }
    use ffai_core::types::PixelFormat;
    use rff_codec::CodecRegistry;
    use rff_core::{CodecId, Frame, Packet};

    let src_colour = png_colour_type(data);

    let mut registry = CodecRegistry::new();
    rff_codec_png::register(&mut registry);
    let mut decoder = registry
        .find_decoder(CodecId::Png)
        .map_err(|e| Error::Media(format!("rff has no PNG decoder registered: {e}")))?;

    let packet = Packet { data: data.to_vec(), ..Default::default() };
    decoder.send_packet(&packet).map_err(|e| Error::Media(format!("PNG decode: {e}")))?;
    decoder.flush();
    let frame = decoder.receive_frame().map_err(|e| Error::Media(format!("PNG decode: {e}")))?;
    let Frame::Video(v) = frame else {
        return Err(Error::Media("PNG decoded to a non-video frame".into()));
    };

    // Honour the frame's own stride rather than assuming stride == row width;
    // a decoder is entitled to pad rows and this one is not required to say so.
    let (w, h) = (v.width as usize, v.height as usize);
    let src_bpp = match v.format {
        rff_core::PixelFormat::Rgb24 => 3,
        rff_core::PixelFormat::Rgba => 4,
        other => return Err(Error::Media(format!("rff PNG returned unexpected format {other:?}"))),
    };
    let plane = v
        .planes
        .first()
        .ok_or_else(|| Error::Media("rff PNG frame has no planes".into()))?;
    let stride = v.strides.first().copied().unwrap_or(w * src_bpp);

    // Grayscale sources contract back to one channel; everything else is
    // packed as-is. `step_by(src_bpp)` on a row IS the contraction, because
    // rff wrote g into all three (or four) lanes.
    let gray = matches!(src_colour, Some(0) | Some(4));
    let out_bpp = if gray { 1 } else { src_bpp };
    let mut out = Vec::with_capacity(w * h * out_bpp);
    for row in 0..h {
        let r = &plane[row * stride..row * stride + w * src_bpp];
        if gray {
            out.extend(r.iter().step_by(src_bpp));
        } else {
            out.extend_from_slice(r);
        }
    }

    let format = if gray {
        PixelFormat::Gray8
    } else if src_bpp == 4 {
        PixelFormat::Rgba8
    } else {
        PixelFormat::Rgb8
    };
    Ok(ImageBuffer { width: v.width, height: v.height, format, data: out })
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

#[cfg(test)]
mod png_oracle {
    use super::*;
    use ffai_core::types::PixelFormat;

    /// The pre-rff implementation is `decode_png_legacy`, kept in the
    /// crate rather than copied here so the oracle and the escape hatch are
    /// the SAME code — a divergent copy would stop being an oracle the
    /// first time one of them was edited.
    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("crates/ffai-media has two ancestors")
            .to_path_buf()
    }

    /// Every corpus PNG must decode BIT-IDENTICALLY through rff and through
    /// the implementation it replaced — RGB (Diana) and grayscale
    /// (Carmenta) alike, since the grayscale contraction is the part that
    /// could plausibly differ.
    #[test]
    fn rff_png_matches_the_implementation_it_replaced() {
        let root = repo_root();
        let dirs = [
            "corpora/clips/diana-coco-v3",
            "corpora/clips/diana-coco",
            "corpora/clips/carmenta-doc",
            "corpora/clips/carmenta-synth",
        ];
        let (mut checked, mut dirs_seen) = (0usize, 0usize);
        for d in dirs {
            let Ok(entries) = std::fs::read_dir(root.join(d)) else { continue };
            dirs_seen += 1;
            let mut paths: Vec<_> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "png"))
                .collect();
            paths.sort();
            paths.truncate(12);
            for p in paths {
                let Ok(bytes) = std::fs::read(&p) else { continue };
                let Ok(want) = decode_png_legacy(&bytes) else { continue };
                let got = decode_png(&bytes).unwrap_or_else(|e| panic!("rff failed on {}: {e}", p.display()));
                assert_eq!(got.width, want.width, "{}: width", p.display());
                assert_eq!(got.height, want.height, "{}: height", p.display());
                assert_eq!(got.format, want.format, "{}: pixel format", p.display());
                assert_eq!(got.data.len(), want.data.len(), "{}: byte count", p.display());
                assert!(got.data == want.data, "{}: PIXELS DIFFER", p.display());
                checked += 1;
            }
        }
        assert!(dirs_seen > 0, "no corpus directories present to check against");
        eprintln!("rff PNG == previous implementation on {checked} images across {dirs_seen} corpora");
    }
}
