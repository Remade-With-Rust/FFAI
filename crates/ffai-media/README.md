# ffai-media

Media ingest and egress for [FFai](https://github.com/Remade-With-Rust/FFAI): get audio and images off disk and into the shapes engines expect.

```rust
let audio = ffai_media::load_audio("talk.wav")?;   // -> ffai_core::types::AudioBuffer
println!("{:.1}s at {} Hz", audio.duration_secs(), audio.sample_rate);
```

## What it guarantees

An `AudioBuffer` is always `f32` samples in `[-1.0, 1.0]` with a known sample rate and channel count. Engines resample and downmix from there — `AudioBuffer::to_mono()` averages channels, and ASR engines reject audio at the wrong rate rather than silently resampling badly.

## Where the codecs come from

Container and codec work routes through [remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs) (`rff-*`), our own pure-Rust ffmpeg — FFai's principle 7, "codecs come from home". Formats land as that project lands them.

Today: **WAV** (native), **JPEG** (`rff-codec-jpeg`, through rff's `CodecRegistry`), **PNG** (the `png` crate). PNG has not moved to `rff-codec-png` yet for a measured reason rather than a cautious one — rff expands grayscale to packed `Rgb24` where this path returns `Gray8`, and Carmenta's OCR reads `load_image`, so the swap changes which preprocessing branch every grayscale page takes and needs Carmenta's corpus gates green first.

Note on distribution: rff is not on crates.io, and `cargo publish` refuses a git dependency outright. Crates downstream of `ffai-media` therefore cannot be published until rff publishes — a deliberate trade of distribution for owning the stack, made so the deployment is Rust end to end.

## License

MIT OR Apache-2.0.
