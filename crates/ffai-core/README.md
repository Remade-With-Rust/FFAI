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
| `fastmath` | scalar `exp`/`ln`/`tanh`/`erf`/`silu`/`gelu` that **vectorize** — no libm call in the loop |
| `fastops` | those kernels as candle `CustomOp1`s: drop-in replacements for `.gelu()`, `.silu()`, `.tanh()`, `.erf()` |
| `cost` | deterministic work counters — matmul FLOPs, elementwise visits, transcendentals, bytes moved |

Candle is re-exported as `ffai_core::candle`, so every engine shares one `Tensor` and one `Device` and buffers move between models without conversion.

## `fastmath` / `fastops` — why they live here

candle's CPU backend evaluates `tanhf`/`erf` **per element on one core**; its
rayon use covers `conv2d` and nothing else. On the shape a `SigLIP` MLP actually
runs, `(1, 1024, 3072)`, candle's `.gelu()` took **44.01 ms** and the kernel here
**1.22 ms** — with the caption it feeds byte-identical to the reference.

They live in `ffai-core` because three engines had independently written their
own range-reduced `exp`, and the three had **drifted in exactly the line that
decides whether the win happens**: one left an `f32::round` in the loop and one
an `f32::floor`, both of which are libm calls that keep the loop scalar — so two
of the three had removed the call and put an equivalent barrier straight back.
One module with the oracle tests is what stops that happening a fourth time.

Two things they are careful about: `gelu_erf` and `gelu_tanh` are **different
functions** (they differ by ~1e-3, and a test asserts they disagree so a
refactor cannot alias them), and `tanh` switches to a Maclaurin series below
`|x| = 0.02` because the `1 - 2/(e^{2x}+1)` form catastrophically cancels there.

## `cost` — a win is a counter that went down

A loaded box swings more than most optimizations are worth, so verdicts that can
be deterministic are. `cost` counts the work rather than timing it: same input,
same number, on any machine under any load. It also separates **scalar** from
**vectorised** transcendentals, because conflating them (they are ~36x apart)
once produced a cost model that predicted 32 s against a 16 s measurement.

## Two conventions worth knowing

**`EngineStatus` is honest.** A registered engine is `Stub`, `Experimental`, or `Stable`, and `Stable` means it has been gated against a reference implementation — not that it works. `ffai engines` prints it.

**Absent is not empty.** `Transcript::words` and `Transcript::speakers` are `Option<Vec<_>>`. `None` means the stage was not requested; `Some(vec![])` means it ran and found nothing. Collapsing those into a bare `Vec` would make a skipped stage indistinguishable from a stage that found nothing.

## License

MIT OR Apache-2.0. Model weights carry their own licenses, surfaced at selection time by `ffai-models`.
