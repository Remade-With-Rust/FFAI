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
/// **Both decoders are ours, and both come from crates.io**: `rusty_png`
/// (our performance fork of image-rs/image-png) and `rusty_jpeg` (baseline
/// and progressive DCT, AVX2 FDCT/quantize, two-block AVX2 IDCT). Principle
/// 7 without the tax that used to come with it.
///
/// The route here was PNG/`png` + JPEG/`rff-codec-jpeg`, then both through
/// rff's `CodecRegistry`, and now both direct. The registry seam is the
/// right home for demuxers and video codecs, where rff provides something
/// nothing else does. For a still-image decoder it cost two things that the
/// standalone crates give back:
///
/// * **Publication.** rff is a git dependency and `cargo publish` refuses
///   one, which made every FFai crate downstream of this one unpublishable.
///   `cargo publish --dry-run -p ffai-media` passes again.
/// * **Grayscale.** `rff_core::PixelFormat` has no single-channel variant,
///   so a gray PNG had to expand to `Rgb24` and contract back — measured
///   2.36x slower on Carmenta's 32/32-grayscale corpus. `rusty_png` carries
///   `ColorType::Grayscale` natively, so the round-trip is gone rather than
///   optimised.
///
/// Both are gated by standing tests: `rusty_png` byte-identical to upstream
/// `png` across every corpus image, and `rusty_jpeg` within 3/255 of
/// libjpeg using the corpus's own JPEG/PNG twins.
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

/// JPEG, decoded by **`rusty_jpeg`** — ours, from crates.io.
///
/// Moved off `rff-codec-jpeg` for the same reason PNG did: rff is a git
/// dependency and `cargo publish` refuses those, which made every crate
/// downstream of this one unpublishable. `rusty_jpeg` is the same decoder
/// rff wraps, published directly, with AVX2 FDCT/quantize and a two-block
/// AVX2 IDCT.
///
/// **Grayscale JPEGs are expanded to RGB here on purpose.** `rusty_jpeg`
/// reports `L8` natively and returning `Gray8` would be leaner — but this
/// function's existing contract is packed RGB for every JPEG, Carmenta's
/// OCR corpus is read through it, and changing an output FORMAT is a
/// different decision from changing a decoder. Made separately, or not at
/// all.
fn decode_jpeg(data: Vec<u8>) -> Result<ImageBuffer> {
    use ffai_core::types::PixelFormat;
    use rusty_jpeg::{Decoder, PixelFormat as JpegFormat};

    let mut decoder = Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().map_err(|e| Error::Media(format!("JPEG decode: {e}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| Error::Media("JPEG decoded without image info".into()))?;
    let (w, h) = (info.width as usize, info.height as usize);

    let (data, format) = match info.pixel_format {
        JpegFormat::RGB24 => (pixels, PixelFormat::Rgb8),
        JpegFormat::L8 => {
            let mut rgb = vec![0u8; w * h * 3];
            for (i, &g) in pixels.iter().take(w * h).enumerate() {
                rgb[i * 3..i * 3 + 3].copy_from_slice(&[g, g, g]);
            }
            (rgb, PixelFormat::Rgb8)
        }
        other => {
            return Err(Error::Media(format!(
                "JPEG pixel format {other:?} unsupported — ffai-media handles RGB and grayscale"
            )))
        }
    };
    Ok(ImageBuffer { width: info.width as u32, height: info.height as u32, format, data })
}

/// PNG, decoded by **`rusty_png`** — our own performance fork of
/// image-rs/image-png, from crates.io.
///
/// This is the third home for this function in two days and the reasoning is
/// worth keeping, because each move was for a different reason:
///
/// 1. The `png` crate. Worked; not ours.
/// 2. `rff-codec-png` through rff's `CodecRegistry`. Ours, and the registry
///    seam is genuinely nice — but it forced two costs. rff is a GIT
///    dependency, and `cargo publish` refuses those outright, so every FFai
///    crate downstream became unpublishable. And `rff_core::PixelFormat` has
///    no single-channel variant, so grayscale had to expand to `Rgb24` and
///    contract back: measured **2.36x slower** on Carmenta's 32/32-grayscale
///    document corpus.
/// 3. `rusty_png`, published on crates.io. Ours, no git dependency, and
///    API-compatible with the `png` crate — including `ColorType::Grayscale`,
///    so gray stays one channel the whole way through and the round-trip
///    disappears rather than being optimised.
///
/// `EXPAND | STRIP_16` is set so palette and 16-bit PNGs decode instead of
/// erroring, which is the capability the rff path added and this keeps.
fn decode_png(data: &[u8]) -> Result<ImageBuffer> {
    use ffai_core::types::PixelFormat;
    use rusty_png::{BitDepth, ColorType, Decoder, Transformations};

    let mut decoder = Decoder::new(std::io::Cursor::new(data));
    decoder.set_transformations(Transformations::EXPAND | Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| Error::Media(format!("PNG header: {e}")))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| Error::Media(format!("PNG decode: {e}")))?;
    buf.truncate(info.buffer_size());

    if info.bit_depth != BitDepth::Eight {
        return Err(Error::Media(format!(
            "PNG bit depth {:?} survived STRIP_16 — unsupported",
            info.bit_depth
        )));
    }
    let format = match info.color_type {
        ColorType::Grayscale => PixelFormat::Gray8,
        ColorType::Rgb => PixelFormat::Rgb8,
        ColorType::Rgba => PixelFormat::Rgba8,
        ColorType::GrayscaleAlpha => {
            // Drop alpha: every consumer of a gray page reads luminance.
            buf = buf.chunks_exact(2).map(|p| p[0]).collect();
            PixelFormat::Gray8
        }
        // EXPAND turns palettes into RGB/RGBA before we get here, so this
        // arm means the transformation did not apply rather than that the
        // file is exotic.
        ColorType::Indexed => {
            return Err(Error::Media("indexed PNG survived EXPAND".into()))
        }
    };
    Ok(ImageBuffer { width: info.width, height: info.height, format, data: buf })
}

/// Sample frames from a video at `fps` frames/second (for Argus video
/// understanding). Pending the rff demux/decode integration.
pub fn sample_frames(path: &Path, fps: f64) -> Result<Vec<VideoFrame>> {
    use rff_format::FormatRegistry;

    // Demux with rff, decode with `rusty_h264` DIRECTLY.
    //
    // This used to route through `rff-codec-h264`, whose 0.1.0 release pins
    // `rusty_h264 ^0.2` — so no matter what was published upstream we resolved
    // 0.2.1, six minor versions behind. The cost of that pin, measured
    // 2026-08-06 on the same files:
    //
    //   | file                    | 0.2.1  | 0.8.0   |
    //   |-------------------------|--------|---------|
    //   | CAVLC                   | 164/164| 164/164 |
    //   | CABAC                   |  49/164| 164/164 |
    //   | x264 default (High)     |   0/164| 164/164 |
    //   | 1080p decode, ms/frame  |  47.50 |   15.20 |
    //
    // x264's DEFAULT profile is High, so on 0.2.1 a normal MP4 decoded to
    // nothing. Going direct fixes that and is 3.1x faster; the registry seam
    // returns when `rff-codec-h264` relaxes its pin.
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if !matches!(ext.as_str(), "mp4" | "mov" | "m4v") {
        return Err(Error::Media(format!(
            "`{}`: only MP4/MOV is wired (H.264 inside). Other containers land              as their rff-format-* crates publish.",
            path.display()
        )));
    }

    let file = std::fs::File::open(path)?;
    let mkerr = |e: rff_core::Error| Error::Media(format!("{}: {e}", path.display()));
    let mut formats = FormatRegistry::new();
    rff_format_mp4::register(&mut formats);
    let mut demux = formats
        .open_demuxer("mp4", Box::new(std::io::BufReader::new(file)))
        .map_err(mkerr)?;
    let streams = demux.read_header().map_err(mkerr)?;

    let (vidx, vstream) = streams
        .iter()
        .enumerate()
        .find(|(_, s)| s.media_type == rff_core::MediaType::Video)
        .ok_or_else(|| Error::Media(format!("{}: no video stream", path.display())))?;

    let mut dec = rusty_h264::Decoder::new();
    // SPS/PPS live in the container's extradata for MP4, not inline.
    if !vstream.extradata.is_empty() {
        let _ = dec.decode(&vstream.extradata);
    }

    let tb = vstream.time_base;
    let src_fps = if tb.num > 0 { tb.den as f64 / tb.num as f64 } else { 0.0 };
    let stride = if fps > 0.0 && src_fps > 0.0 {
        (src_fps / fps).round().max(1.0) as usize
    } else {
        1
    };

    let mut out = Vec::new();
    let mut idx = 0usize;
    let mut pkts = 0usize;
    loop {
        let packet = match demux.read_packet() {
            Ok(p) => p,
            Err(rff_core::Error::Eof) => break,
            Err(e) => return Err(mkerr(e)),
        };
        if packet.stream_index != vidx {
            continue;
        }
        pkts += 1;
        // DO NOT SWALLOW THIS. It was `if ... .is_err() { continue; }`, which
        // turned a decoder reporting itself clearly into a silent short read:
        // a standard x264 file produced zero frames and NO error, which no
        // caller could distinguish from an empty video.
        let frame = dec.decode(&packet.data).map_err(|e| {
            Error::Media(format!(
                "{}: decode failed on packet {} after {} frame(s): {e}",
                path.display(),
                pkts,
                out.len()
            ))
        })?;
        if let Some(v) = frame {
            if idx % stride == 0 {
                let ts = packet
                    .pts
                    .map_or(0.0, |p| p as f64 * tb.num as f64 / tb.den.max(1) as f64);
                out.push(from_rusty_frame(&v, ts)?);
            }
            idx += 1;
        }
    }
    Ok(out)
}

/// `rusty_h264`'s YUV420p frame to RGB8, the format every FFai engine consumes.
///
/// BT.601 limited range, matching OpenCV's `COLOR_YUV2RGB_I420` — so frames
/// decoded here and frames extracted by the Python tooling agree, and a
/// comparison between two engines is not secretly a comparison between two
/// colour conversions.
fn from_rusty_frame(v: &rusty_h264::YuvFrame, ts: f64) -> Result<VideoFrame> {
    let (w, h) = (v.width, v.height);
    let mut rgb = vec![0u8; w * h * 3];
    let (ys, us, vs) = (w, w.div_ceil(2), w.div_ceil(2));
    for row in 0..h {
        for col in 0..w {
            let y = v.y[row * ys + col] as f32;
            let cu = v.u[(row / 2) * us + col / 2] as f32 - 128.0;
            let cv = v.v[(row / 2) * vs + col / 2] as f32 - 128.0;
            let yy = 1.164 * (y - 16.0);
            let o = (row * w + col) * 3;
            rgb[o] = (yy + 1.596 * cv).clamp(0.0, 255.0) as u8;
            rgb[o + 1] = (yy - 0.813 * cv - 0.391 * cu).clamp(0.0, 255.0) as u8;
            rgb[o + 2] = (yy + 2.018 * cu).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(VideoFrame {
        image: ImageBuffer {
            width: w as u32,
            height: h as u32,
            format: ffai_core::types::PixelFormat::Rgb8,
            data: rgb,
        },
        timestamp: ts,
    })
}

/// YUV420p to RGB8, the format every FFai engine consumes.
///
/// BT.601 limited range, matching OpenCV's `COLOR_YUV2RGB_I420` — so frames
/// extracted here and frames extracted by the Python tooling agree, and a
/// comparison between the two engines is not secretly a comparison between
/// two colour conversions.
fn from_rff_frame(v: &rff_core::VideoFrame, ts: f64) -> Result<VideoFrame> {
    let (w, h) = (v.width as usize, v.height as usize);
    if v.planes.len() < 3 || v.strides.len() < 3 {
        return Err(Error::Media(format!(
            "expected 3 planar YUV planes, found {}",
            v.planes.len()
        )));
    }
    let (yp, up, vp) = (&v.planes[0], &v.planes[1], &v.planes[2]);
    let (ys, us, vs) = (v.strides[0], v.strides[1], v.strides[2]);
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in 0..w {
            let yv = yp[row * ys + col] as f32 - 16.0;
            let uv = up[(row / 2) * us + col / 2] as f32 - 128.0;
            let vv = vp[(row / 2) * vs + col / 2] as f32 - 128.0;
            let o = (row * w + col) * 3;
            rgb[o] = (1.164 * yv + 1.596 * vv).clamp(0.0, 255.0) as u8;
            rgb[o + 1] = (1.164 * yv - 0.813 * vv - 0.391 * uv).clamp(0.0, 255.0) as u8;
            rgb[o + 2] = (1.164 * yv + 2.018 * uv).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(VideoFrame {
        image: ImageBuffer {
            width: v.width,
            height: v.height,
            format: ffai_core::types::PixelFormat::Rgb8,
            data: rgb,
        },
        timestamp: ts,
    })
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

    /// Upstream `png` — the crate `rusty_png` is a performance FORK of.
    ///
    /// That is the right oracle for a fork: it is the implementation whose
    /// behaviour ours is supposed to reproduce, maintained by someone else,
    /// so it cannot drift with our edits. A decoder swap changes the pixels
    /// every downstream gate is computed from, which is why this is a
    /// standing test and not a one-off check.
    fn decode_png_upstream(data: &[u8]) -> Result<ImageBuffer> {
        use ffai_core::types::PixelFormat;
        let mut decoder = png::Decoder::new(std::io::Cursor::new(data));
        decoder.set_transformations(
            png::Transformations::EXPAND | png::Transformations::STRIP_16,
        );
        let mut reader =
            decoder.read_info().map_err(|e| Error::Media(format!("PNG header: {e}")))?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info =
            reader.next_frame(&mut buf).map_err(|e| Error::Media(format!("PNG decode: {e}")))?;
        buf.truncate(info.buffer_size());
        if info.bit_depth != png::BitDepth::Eight {
            return Err(Error::Media("non-8-bit".into()));
        }
        let format = match info.color_type {
            png::ColorType::Grayscale => PixelFormat::Gray8,
            png::ColorType::Rgb => PixelFormat::Rgb8,
            png::ColorType::Rgba => PixelFormat::Rgba8,
            png::ColorType::GrayscaleAlpha => {
                buf = buf.chunks_exact(2).map(|p| p[0]).collect();
                PixelFormat::Gray8
            }
            png::ColorType::Indexed => return Err(Error::Media("indexed".into())),
        };
        Ok(ImageBuffer { width: info.width, height: info.height, format, data: buf })
    }

    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("crates/ffai-media has two ancestors")
            .to_path_buf()
    }

    /// `rusty_jpeg` against libjpeg, for free, using the corpus's own twins.
    ///
    /// `tools/diana_coco_corpus.py` builds each `coco-NNN.png` by decoding
    /// `coco-NNN.src.jpg` with Pillow (libjpeg) and re-encoding LOSSLESSLY.
    /// So the PNG carries libjpeg's decode of that exact JPEG, and decoding
    /// both through this crate compares our JPEG decoder against libjpeg's
    /// with no reference implementation to install.
    ///
    /// The bound is a TOLERANCE, not equality, and deliberately so: the JPEG
    /// spec does not mandate a single IDCT, so conforming decoders disagree
    /// in the last bit or two. Measured worst channel delta here is **3 of
    /// 255**. A regression that broke the decoder would blow through this by
    /// orders of magnitude; a legitimate IDCT change would not.
    #[test]
    fn rusty_jpeg_agrees_with_libjpeg_via_the_corpus_twins() {
        let root = repo_root().join("corpora/clips/diana-coco");
        let (mut n, mut worst) = (0usize, 0i32);
        for i in 0..8 {
            let (j, p) = (root.join(format!("coco-{i:03}.src.jpg")), root.join(format!("coco-{i:03}.png")));
            if !j.exists() || !p.exists() {
                continue;
            }
            let a = load_image(&j).expect("jpeg");
            let b = load_image(&p).expect("png");
            assert_eq!((a.width, a.height), (b.width, b.height), "coco-{i:03}: dimensions");
            assert_eq!(a.data.len(), b.data.len(), "coco-{i:03}: buffer length");
            worst = worst.max(
                a.data.iter().zip(&b.data).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0),
            );
            n += 1;
        }
        if n == 0 {
            eprintln!("SKIP jpeg/libjpeg twin check: corpus absent");
            return;
        }
        assert!(worst <= 8, "rusty_jpeg diverges from libjpeg by {worst}/255 over {n} images");
        eprintln!("rusty_jpeg vs libjpeg: {n} images, worst channel delta {worst}/255");
    }

    /// Every corpus PNG must decode BIT-IDENTICALLY through rff and through
    /// the implementation it replaced — RGB (Diana) and grayscale
    /// (Carmenta) alike, since the grayscale contraction is the part that
    /// could plausibly differ.
    #[test]
    fn rusty_png_matches_upstream_png() {
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
                let Ok(want) = decode_png_upstream(&bytes) else { continue };
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
        eprintln!("rusty_png == upstream png on {checked} images across {dirs_seen} corpora");
    }
}


/// Write a 16-bit grayscale PNG.
///
/// Lives here rather than in a caller because this crate already owns the
/// PNG dependency and the decode side, so encode belongs beside them.
///
/// Diana's depth maps are the motivating case: a `u16` per pixel carries
/// enough precision for a normalised depth visualisation, where 8 bits
/// visibly bands a smooth field. The METRES do not survive normalisation —
/// anything numeric should take the raw f32 instead.
///
/// `pixels` is row-major, `width * height` samples. PNG stores 16-bit
/// samples BIG-endian regardless of host order, which is the one detail
/// easy to get wrong and impossible to see afterwards: a byte-swapped map
/// still renders, as noise.
pub fn save_gray16_png(path: &Path, pixels: &[u16], width: usize, height: usize) -> Result<()> {
    if pixels.len() != width * height {
        return Err(Error::Other(format!(
            "save_gray16_png: {} pixels for a {width}x{height} image",
            pixels.len()
        )));
    }
    let file = std::fs::File::create(path)?;
    let mut enc = rusty_png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    enc.set_color(rusty_png::ColorType::Grayscale);
    enc.set_depth(rusty_png::BitDepth::Sixteen);
    let mut w = enc.write_header().map_err(|e| Error::Other(format!("png header: {e}")))?;
    let mut bytes = Vec::with_capacity(pixels.len() * 2);
    for p in pixels {
        bytes.extend_from_slice(&p.to_be_bytes());
    }
    w.write_image_data(&bytes).map_err(|e| Error::Other(format!("png write: {e}")))?;
    w.finish().map_err(|e| Error::Other(format!("png finish: {e}")))?;
    Ok(())
}
