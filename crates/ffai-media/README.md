# ffai-media

Media ingest and egress for [FFai](https://github.com/Remade-With-Rust/FFAI): get audio and images off disk and into the shapes engines expect.

```rust
let audio = ffai_media::load_audio("talk.wav")?;   // -> ffai_core::types::AudioBuffer
println!("{:.1}s at {} Hz", audio.duration_secs(), audio.sample_rate);
```

## What it guarantees

An `AudioBuffer` is always `f32` samples in `[-1.0, 1.0]` with a known sample rate and channel count. Engines resample and downmix from there — `AudioBuffer::to_mono()` averages channels, and ASR engines reject audio at the wrong rate rather than silently resampling badly.

## Images, from a file or from bytes

```rust
let page = ffai_media::load_image("page.png")?;            // from a path
let page = ffai_media::decode_image(&bytes_from_a_pdf)?;   // from memory, same output
```

Both dispatch on the bytes' magic, not the file name: 169 of OmniDocBench's English pages are JPEGs named `.png`. `decode_image` exists for callers whose pages come out of a PDF, an archive or a network response, so they need no temporary file and no copy of the grayscale/RGB handling.

## PDF pages as a viewer shows them (`pdf` feature)

```toml
ffai-media = { version = "0.6", features = ["pdf"] }
```

```rust
for page in ffai_media::pdf::pdf_pages(&pdf_bytes)? {
    if let Some(image) = &page.image { /* OCR it */ }
    else if page.has_text_layer && !page.redaction_risk { /* use page.text */ }
}
```

**A scanned PDF page is an image with things drawn on top of it, and extracting the image alone reads through redactions.** A real filing hid a home address under a black patch; the patch was a separate object, and OCR of the extracted scan returned the address. `pdf_pages` therefore returns the scan **composited with everything drawn after it**, turned upright:

- filled boxes, later images (painted black) and nested form objects;
- redaction, square, circle, polygon and stamp annotations.

A page it cannot walk returns **no image** rather than the raw scan.

A born-digital text layer has the same trap: text under a black box stays in the content stream. The page's `text` comes with a `redaction_risk` flag, set when anything could be hiding part of it.

Decoders: JPEG, CCITT Group 4 and Group 3 1-D (fax scans), and raw or Flate 1- and 8-bit samples in gray, RGB or CMYK, including stencil masks. JBIG2 and JPEG 2000 are reported per page, not guessed. Every allocation is bounded before it happens: images at 100 MP, decompression at the size the dimensions justify, and form content at 64 MiB.

## Where the codecs come from

Codecs come from home, FFai's principle 7. Still images decode through [`rusty_jpeg`](https://crates.io/crates/rusty_jpeg) and [`rusty_png`](https://crates.io/crates/rusty_png), and video ingest through [remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs) (`rff-*`). All are pure Rust and published on crates.io, so everything downstream of this crate is publishable. They are gated against what they replaced: `rusty_png` byte-identical to upstream `png` across every corpus image, and `rusty_jpeg` within 3/255 of libjpeg. The PDF feature adds `lopdf` and the `fax` CCITT decoder, both pure Rust and MIT.

## License

MIT OR Apache-2.0.
