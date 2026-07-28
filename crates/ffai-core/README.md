# ffai-core

Shared types, engine traits, and the registry for [FFai](https://github.com/Remade-With-Rust/FFAI) — the AI media toolkit, remade with Rust.

This is the crate every other FFai crate depends on and no FFai crate depends on. It holds no models and no algorithms: it defines the shapes everything else speaks in.

## The idea

FFai is built the way ffmpeg is built — **one trait per task, many engines per trait, selected by name**. `AsrEngine` is the `AVCodec` of speech; `--engine whisper-candle` is `-c:v libx264`.

```rust
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_core::registry::EngineRegistry;

let mut reg = EngineRegistry::new();
ffai_mercury::register(&mut reg);

let engine = reg.asr(Some("whisper-candle"))?;   // or None for the default
let transcript = engine.transcribe(&audio, &AsrOptions::default())?;
```

## What's in here

| Module | What it holds |
|---|---|
| `types` | `AudioBuffer`, `ImageBuffer`, `VideoFrame`, `TimedSegment<T>`, `Transcript` |
| `engine` | `AsrEngine` / `TtsEngine` / `OcrEngine` / `VlmEngine`, their option structs, `EngineStatus` |
| `registry` | name → engine lookup, used by the CLI and embeddable anywhere |
| `error` | one `Error` enum across the toolkit |

Candle is re-exported as `ffai_core::candle`, so every engine shares one `Tensor` and one `Device` and buffers move between models without conversion.

## Two conventions worth knowing

**`EngineStatus` is honest.** A registered engine is `Stub`, `Experimental`, or `Stable`, and `Stable` means it has been gated against a reference implementation — not that it works. `ffai engines` prints it.

**Absent is not empty.** `Transcript::words` and `Transcript::speakers` are `Option<Vec<_>>`. `None` means the stage was not requested; `Some(vec![])` means it ran and found nothing. Collapsing those into a bare `Vec` would make a skipped stage indistinguishable from a stage that found nothing.

## License

MIT OR Apache-2.0. Model weights carry their own licenses, surfaced at selection time by `ffai-models`.
