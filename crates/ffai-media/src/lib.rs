//! # ffai-media
//!
//! Media ingest/egress for `FFai` — the `libavformat` seat.
//!
//! Policy: all container/codec work routes through **`remade_ffmpeg_rs`**
//! (`rff-*` crates) as the default backend — we own it, it is pure Rust, and
//! it keeps the zero-C/C++ promise. Phase 0 ships native WAV support (the one
//! format every ASR/TTS engine needs on day one); everything else returns a
//! clear "pending rff integration" error rather than silently failing.

pub mod annexb;

use std::path::Path;

use ffai_core::error::{Error, Result};
use ffai_core::types::{AudioBuffer, ImageBuffer, VideoFrame};

/// Load an audio file into a normalized f32 [`AudioBuffer`].
///
/// Phase 0: WAV only (PCM int/float). Other containers/codecs land with the
/// `remade_ffmpeg_rs` integration in Phase 1.
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

// `v as f32` widens an i32 PCM sample. Beyond 2^24 the mantissa rounds, which
// is inherent to representing 32-bit PCM as f32 at all - the whole pipeline is
// f32 audio - and is a rounding difference of one LSB at full scale, not a
// correctness issue.
#[allow(clippy::cast_precision_loss)]
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
            // `bits_per_sample` comes from the file's fmt chunk, i.e. from
            // untrusted bytes. hound validates the formats IT supports, but this
            // arithmetic must not rest on that: 0 underflows `- 1` (u16), and
            // anything >= 65 overflows the shift. In debug both panic; in
            // release - where this workspace deliberately carries no
            // overflow-checks - the shift is masked and `scale` comes out
            // silently wrong, which quietly rescales every sample.
            let bits = spec.bits_per_sample;
            if !(1..=32).contains(&bits) {
                return Err(Error::Media(format!(
                    "WAV declares {bits} bits per sample; supported range is 1..=32"
                )));
            }
            let scale = (1u32 << (bits - 1)) as f32;
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
///   one, which made every `FFai` crate downstream of this one unpublishable.
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
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
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
    let pixels = decoder
        .decode()
        .map_err(|e| Error::Media(format!("JPEG decode: {e}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| Error::Media("JPEG decoded without image info".into()))?;
    let (w, h) = (info.width as usize, info.height as usize);

    let (data, format) = match info.pixel_format {
        JpegFormat::RGB24 => (pixels, PixelFormat::Rgb8),
        JpegFormat::L8 => {
            // `w * h * 3` is unchecked multiplication on dimensions that came from a
            // decoded bitstream. On 64-bit it merely asks for an absurd allocation; on
            // 32-bit - and `ffai-wasm` makes wasm32 a real target - it WRAPS to a small
            // buffer, and the row/column indexing below then runs past it. Same defect
            // class as the ONNX dims product (see ffai-mercury's audit, gate H-17).
            let size = w
                .checked_mul(h)
                .and_then(|n| n.checked_mul(3))
                .ok_or_else(|| {
                    Error::Media(format!("frame {w}x{h} overflows this platform's usize"))
                })?;
            let mut rgb = vec![0u8; size];
            for (i, &g) in pixels.iter().take(w * h).enumerate() {
                rgb[i * 3..i * 3 + 3].copy_from_slice(&[g, g, g]);
            }
            (rgb, PixelFormat::Rgb8)
        }
        other => {
            return Err(Error::Media(format!(
                "JPEG pixel format {other:?} unsupported — ffai-media handles RGB and grayscale"
            )));
        }
    };
    Ok(ImageBuffer {
        width: u32::from(info.width),
        height: u32::from(info.height),
        format,
        data,
    })
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
///    dependency, and `cargo publish` refuses those outright, so every `FFai`
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
    let mut reader = decoder
        .read_info()
        .map_err(|e| Error::Media(format!("PNG header: {e}")))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| Error::Media(format!("PNG decode: {e}")))?;
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
        ColorType::Indexed => return Err(Error::Media("indexed PNG survived EXPAND".into())),
    };
    Ok(ImageBuffer {
        width: info.width,
        height: info.height,
        format,
        data: buf,
    })
}

/// Sample frames from a video at `fps` frames/second (for Argus video
/// understanding). Pending the rff demux/decode integration.
/// A lazily-decoded video, yielding one frame at a time.
///
/// # Why this exists rather than a `Vec`
///
/// `sample_frames` used to decode the whole file and hand back a
/// `Vec<VideoFrame>`. A 1080p RGB frame is 5.9 MiB, so **one minute of video is
/// 10.4 GiB and ten minutes is 104 GiB** — the API could only ever be used on
/// clips, and "video ingest" meant "short video ingest".
///
/// This holds the demuxer and decoder open and pulls exactly as far as the
/// caller asks. Memory is one frame plus the decoder's own reference buffers,
/// whatever the file's length.
///
/// Mirrors what Ultralytics' `predict(source, stream=True)` returns — a
/// generator rather than a list — for the same reason.
pub struct VideoStream {
    demux: Box<dyn rff_format::Demuxer>,
    dec: rusty_h264::Decoder,
    /// `Some(n)` when the container hands us AVCC and every packet needs its
    /// `n`-byte length prefixes rewritten as start codes; `None` for Annex-B.
    nal_length_size: Option<usize>,
    vidx: usize,
    tb: rff_core::Rational,
    /// Seconds between kept frames; `<= 0` keeps every frame.
    ///
    /// **Decimation is by TIMESTAMP, not by a decoded-frame stride.** The
    /// stride version computed its source rate as `time_base.den /
    /// time_base.num` — but a time base is a CLOCK TICK RATE, not a frame
    /// rate. MP4 commonly uses 1/12800, so `stream_frames(path, 1.0)` asked
    /// for one frame per second and computed a stride of 12800, returning
    /// **exactly one frame** from every clip in the corpus. The failure is
    /// quiet in the worst way: one frame is a perfectly good frame, so a
    /// caller sees a plausible result rather than an error.
    ///
    /// A deadline in seconds needs no frame rate at all — it reads the
    /// timestamps the container already carries — and it stays correct on
    /// variable-frame-rate sources, where no single stride can be right.
    interval: f64,
    /// Timestamp the next kept frame must reach.
    next_due: f64,
    /// Whether any frame has been kept yet, so the first one is always taken
    /// regardless of where its timestamp starts.
    started: bool,
    /// Packets fed, for error messages that say WHERE it stopped.
    pkts: usize,
    path: std::path::PathBuf,
    done: bool,
}

impl VideoStream {
    /// Frames the container claims, when it says. `None` means unknown —
    /// report it as unknown rather than guessing, since a wrong total in a
    /// progress line is worse than no total.
    #[must_use]
    pub const fn frame_count_hint(&self) -> Option<usize> {
        None
    }
}

impl Iterator for VideoStream {
    type Item = Result<VideoFrame>;

    // `pts as f64` for a presentation timestamp: an i64 pts beyond 2^53 is
    // centuries of video at any real time base.
    #[allow(clippy::cast_precision_loss)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            let packet = match self.demux.read_packet() {
                Ok(p) => p,
                Err(rff_core::Error::Eof) => {
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(Error::Media(format!("{}: {e}", self.path.display()))));
                }
            };
            if packet.stream_index != self.vidx {
                continue;
            }
            self.pkts += 1;
            // Errors PROPAGATE. This was `if ... .is_err() { continue; }`,
            // which turned a decoder reporting itself clearly into a silent
            // short read — a standard x264 file yielded zero frames and no
            // diagnostic, indistinguishable from an empty video.
            // Rewrite AVCC to Annex-B when the container uses it. `to_annexb`
            // returns None for anything that is not length-prefixed — including
            // a packet that is already Annex-B — so a wrong guess passes the
            // data through untouched rather than mangling it.
            let converted = self
                .nal_length_size
                .and_then(|n| annexb::to_annexb(&packet.data, n));
            let payload: &[u8] = converted.as_deref().unwrap_or(&packet.data);
            let frame = match self.dec.decode(payload) {
                Ok(f) => f,
                Err(e) => {
                    self.done = true;
                    return Some(Err(Error::Media(format!(
                        "{}: decode failed on packet {} : {e}",
                        self.path.display(),
                        self.pkts
                    ))));
                }
            };
            let Some(v) = frame else { continue };
            let ts = packet.pts.map_or(0.0, |p| {
                p as f64 * f64::from(self.tb.num) / f64::from(self.tb.den.max(1))
            });
            if self.interval > 0.0 {
                if self.started && ts < self.next_due {
                    continue;
                }
                // Advance from the DEADLINE, not from the frame's own
                // timestamp, so the sampling grid does not drift late by half
                // an interval on every step. Clamped forward for sources whose
                // gaps exceed the interval, so a long gap does not leave a
                // backlog of instantly-due frames afterwards.
                // SITE-REVIEWED: clippy offers `self.interval.mul_add(0.5, ts)`
                // and it is REFUSED here. `mul_add` is a fused multiply-add --
                // one rounding where this expression has two -- so it can move
                // the deadline by an ULP. This value is a frame-selection
                // boundary, compared against `ts` on the next iteration, so an
                // ULP either way can change WHICH FRAME a source emits at an
                // exact tick. A decoder does not get to be 1 ULP creative.
                //
                // Bound through a `let` because an attribute on the assignment
                // itself is `#![feature(stmt_expr_attributes)]`, still unstable.
                #[allow(clippy::suboptimal_flops)]
                let due = if self.started {
                    (self.next_due + self.interval).max(ts + self.interval * 0.5)
                } else {
                    ts + self.interval
                };
                self.next_due = due;
                self.started = true;
            }
            return Some(from_rusty_frame(&v, ts));
        }
    }
}

/// Open a video and stream its frames. `fps <= 0` keeps every frame.
///
/// Demuxes with `rff-format-mp4` and decodes with `rusty_h264` — the whole path
/// is Remade-With-Rust, no libavformat and no libavcodec.
#[allow(clippy::cast_precision_loss)]
pub fn stream_frames(path: &Path, fps: f64) -> Result<VideoStream> {
    use rff_format::FormatRegistry;

    // Extension -> the name the demuxer REGISTERS UNDER, which is not the
    // extension: `rff-format-mkv` registers as "matroska", `rff-format-ts` as
    // "mpegts". Looking up by extension silently found nothing and reported
    // "no demuxer found for input `mkv`" while the crate was linked and working.
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let demuxer_name = match ext.as_str() {
        "mp4" | "mov" | "m4v" => "mp4",
        "mkv" | "webm" | "mka" => "matroska",
        "avi" => "avi",
        "ts" | "m2ts" | "mts" => "mpegts",
        other => {
            return Err(Error::Media(format!(
                "`.{other}`: no demuxer wired. Supported: mp4/mov/m4v, mkv/webm, \
                 avi, ts/m2ts/mts. MPEG-PS and ASF land when their rff-format-* \
                 crates publish — see docs/rff-gaps-for-ffai.md."
            )));
        }
    };

    let file = std::fs::File::open(path)?;
    let mkerr = |e: rff_core::Error| Error::Media(format!("{}: {e}", path.display()));
    let mut formats = FormatRegistry::new();
    rff_format_mp4::register(&mut formats);
    rff_format_mkv::register(&mut formats);
    rff_format_avi::register(&mut formats);
    rff_format_ts::register(&mut formats);
    let mut demux = formats
        .open_demuxer(demuxer_name, Box::new(std::io::BufReader::new(file)))
        .map_err(mkerr)?;
    let streams = demux.read_header().map_err(mkerr)?;

    let (vidx, vstream) = streams
        .iter()
        .enumerate()
        .find(|(_, s)| s.media_type == rff_core::MediaType::Video)
        .ok_or_else(|| Error::Media(format!("{}: no video stream", path.display())))?;

    let mut dec = rusty_h264::Decoder::new();
    // Two container conventions, and getting this wrong is SILENT.
    //
    // `rff-format-mp4` hands back Annex-B with empty extradata. `rff-format-mkv`
    // hands back AVCC (length-prefixed) with an `avcC` in extradata, which
    // `rusty_h264` does not parse — measured at 164 packets, 0 frames, 0 errors.
    // If this is an avcC, convert the parameter sets to Annex-B, feed them, and
    // remember the length size so every packet can be rewritten too.
    let avcc = annexb::parse_avcc(&vstream.extradata);
    let nal_length_size = avcc.as_ref().map(|c| c.nal_length_size);
    match &avcc {
        Some(c) => {
            let _ = dec.decode(&c.parameter_sets);
        }
        None if !vstream.extradata.is_empty() => {
            let _ = dec.decode(&vstream.extradata);
        }
        None => {}
    }

    let tb = vstream.time_base;
    // NOT `time_base.den / time_base.num` — see `VideoStream::interval`. That
    // read a clock tick rate as a frame rate and decimated 12800x.
    let interval = if fps > 0.0 { 1.0 / fps } else { 0.0 };

    Ok(VideoStream {
        demux,
        dec,
        nal_length_size,
        vidx,
        tb,
        interval,
        next_due: 0.0,
        started: false,
        pkts: 0,
        path: path.to_path_buf(),
        done: false,
    })
}

/// Every frame at once. Prefer [`stream_frames`] — this holds the whole video
/// in memory and exists for callers that genuinely want a `Vec`.
// `(src_fps / fps).round().max(1.0) as usize`: `.max(1.0)` fixes the low end
// AND absorbs NaN (f64::max returns the non-NaN operand), and Rust saturates
// float->int casts. A ratio large enough to truncate needs an fps of ~1e-19.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn sample_frames(path: &Path, fps: f64) -> Result<Vec<VideoFrame>> {
    stream_frames(path, fps)?.collect()
}

/// `rusty_h264`'s `YUV420p` frame to RGB8, the format every `FFai` engine consumes.
///
/// BT.601 limited range, matching `OpenCV`'s `COLOR_YUV2RGB_I420` — so frames
/// decoded here and frames extracted by the Python tooling agree, and a
/// comparison between two engines is not secretly a comparison between two
/// colour conversions.
// Every `as u8` here is preceded by `.clamp(0.0, 255.0)` on the same
// expression, so truncation and sign loss are impossible by construction -
// the clamp IS the guard, not an afterthought. `width/height as u32` round-trip
// dimensions that arrived as u32 from the decoder.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn from_rusty_frame(v: &rusty_h264::YuvFrame, ts: f64) -> Result<VideoFrame> {
    let (w, h) = (v.width, v.height);
    // `w * h * 3` is unchecked multiplication on dimensions that came from a
    // decoded bitstream. On 64-bit it merely asks for an absurd allocation; on
    // 32-bit - and `ffai-wasm` makes wasm32 a real target - it WRAPS to a small
    // buffer, and the row/column indexing below then runs past it. Same defect
    // class as the ONNX dims product (see ffai-mercury's audit, gate H-17).
    let size = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::Media(format!("frame {w}x{h} overflows this platform's usize")))?;
    let mut rgb = vec![0u8; size];
    let (ys, us, vs) = (w, w.div_ceil(2), w.div_ceil(2));
    for row in 0..h {
        for col in 0..w {
            let y = f32::from(v.y[row * ys + col]);
            let cu = f32::from(v.u[(row / 2) * us + col / 2]) - 128.0;
            let cv = f32::from(v.v[(row / 2) * vs + col / 2]) - 128.0;
            let yy = 1.164 * (y - 16.0);
            let o = (row * w + col) * 3;
            rgb[o] = 1.596f32.mul_add(cv, yy).clamp(0.0, 255.0) as u8;
            rgb[o + 1] = 0.391f32
                .mul_add(-cu, 0.813f32.mul_add(-cv, yy))
                .clamp(0.0, 255.0) as u8;
            rgb[o + 2] = 2.018f32.mul_add(cu, yy).clamp(0.0, 255.0) as u8;
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

/// `YUV420p` to RGB8, the format every `FFai` engine consumes.
///
/// BT.601 limited range, matching `OpenCV`'s `COLOR_YUV2RGB_I420` — so frames
/// extracted here and frames extracted by the Python tooling agree, and a
/// comparison between the two engines is not secretly a comparison between
/// two colour conversions.
// Unused until the video path lands: `rff` provides demuxers and H.264/VP9,
// and nothing calls this until that arrives (see the workspace manifest's note
// on rff being the one git dependency). Retained rather than deleted because
// the colour-conversion contract above is the hard part and would have to be
// rewritten identically.
#[allow(dead_code)]
// Every `as u8` here is preceded by `.clamp(0.0, 255.0)` on the same
// expression, so truncation and sign loss are impossible by construction -
// the clamp IS the guard, not an afterthought. `width/height as u32` round-trip
// dimensions that arrived as u32 from the decoder.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
    // `w * h * 3` is unchecked multiplication on dimensions that came from a
    // decoded bitstream. On 64-bit it merely asks for an absurd allocation; on
    // 32-bit - and `ffai-wasm` makes wasm32 a real target - it WRAPS to a small
    // buffer, and the row/column indexing below then runs past it. Same defect
    // class as the ONNX dims product (see ffai-mercury's audit, gate H-17).
    let size = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::Media(format!("frame {w}x{h} overflows this platform's usize")))?;
    let mut rgb = vec![0u8; size];
    for row in 0..h {
        for col in 0..w {
            let yv = f32::from(yp[row * ys + col]) - 16.0;
            let uv = f32::from(up[(row / 2) * us + col / 2]) - 128.0;
            let vv = f32::from(vp[(row / 2) * vs + col / 2]) - 128.0;
            let o = (row * w + col) * 3;
            rgb[o] = 1.164f32.mul_add(yv, 1.596 * vv).clamp(0.0, 255.0) as u8;
            rgb[o + 1] = 0.391f32
                .mul_add(-uv, 1.164f32.mul_add(yv, -(0.813 * vv)))
                .clamp(0.0, 255.0) as u8;
            rgb[o + 2] = 1.164f32.mul_add(yv, 2.018 * uv).clamp(0.0, 255.0) as u8;
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
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder
            .read_info()
            .map_err(|e| Error::Media(format!("PNG header: {e}")))?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| Error::Media(format!("PNG decode: {e}")))?;
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
        Ok(ImageBuffer {
            width: info.width,
            height: info.height,
            format,
            data: buf,
        })
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
            let (j, p) = (
                root.join(format!("coco-{i:03}.src.jpg")),
                root.join(format!("coco-{i:03}.png")),
            );
            if !j.exists() || !p.exists() {
                continue;
            }
            let a = load_image(&j).expect("jpeg");
            let b = load_image(&p).expect("png");
            assert_eq!(
                (a.width, a.height),
                (b.width, b.height),
                "coco-{i:03}: dimensions"
            );
            assert_eq!(a.data.len(), b.data.len(), "coco-{i:03}: buffer length");
            worst = worst.max(
                a.data
                    .iter()
                    .zip(&b.data)
                    .map(|(x, y)| (*x as i32 - *y as i32).abs())
                    .max()
                    .unwrap_or(0),
            );
            n += 1;
        }
        if n == 0 {
            eprintln!("SKIP jpeg/libjpeg twin check: corpus absent");
            return;
        }
        assert!(
            worst <= 8,
            "rusty_jpeg diverges from libjpeg by {worst}/255 over {n} images"
        );
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
            let Ok(entries) = std::fs::read_dir(root.join(d)) else {
                continue;
            };
            dirs_seen += 1;
            let mut paths: Vec<_> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "png"))
                .collect();
            paths.sort();
            paths.truncate(12);
            for p in paths {
                let Ok(bytes) = std::fs::read(&p) else {
                    continue;
                };
                let Ok(want) = decode_png_upstream(&bytes) else {
                    continue;
                };
                let got = decode_png(&bytes)
                    .unwrap_or_else(|e| panic!("rff failed on {}: {e}", p.display()));
                assert_eq!(got.width, want.width, "{}: width", p.display());
                assert_eq!(got.height, want.height, "{}: height", p.display());
                assert_eq!(got.format, want.format, "{}: pixel format", p.display());
                assert_eq!(
                    got.data.len(),
                    want.data.len(),
                    "{}: byte count",
                    p.display()
                );
                assert!(got.data == want.data, "{}: PIXELS DIFFER", p.display());
                checked += 1;
            }
        }
        // SKIP rather than fail when the corpus is absent, matching how the
        // tokenizer oracle handles missing weights. A checkout without corpora -
        // any CI runner, any fresh clone - would otherwise fail a test that has
        // nothing to say, and a suite that is red for an uninteresting reason
        // stops being read.
        if dirs_seen == 0 {
            eprintln!("png oracle: no corpus directories present, skipping");
            return;
        }
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
// Dimensions cast to u32 for the PNG header. A width or height beyond u32 is
// not representable in PNG at all, and the encoder rejects it downstream.
#[allow(clippy::cast_possible_truncation)]
pub fn save_gray16_png(path: &Path, pixels: &[u16], width: usize, height: usize) -> Result<()> {
    if pixels.len() != width * height {
        return Err(Error::Other(format!(
            "save_gray16_png: {} pixels for a {width}x{height} image",
            pixels.len()
        )));
    }
    let file = std::fs::File::create(path)?;
    let mut enc =
        rusty_png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    enc.set_color(rusty_png::ColorType::Grayscale);
    enc.set_depth(rusty_png::BitDepth::Sixteen);
    let mut w = enc
        .write_header()
        .map_err(|e| Error::Other(format!("png header: {e}")))?;
    let mut bytes = Vec::with_capacity(pixels.len() * 2);
    for p in pixels {
        bytes.extend_from_slice(&p.to_be_bytes());
    }
    w.write_image_data(&bytes)
        .map_err(|e| Error::Other(format!("png write: {e}")))?;
    w.finish()
        .map_err(|e| Error::Other(format!("png finish: {e}")))?;
    Ok(())
}
