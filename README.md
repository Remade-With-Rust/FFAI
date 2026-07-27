# FFAI

**The AI media toolkit, remade with rust.** OCR, speech recognition, speech
synthesis, and vision-language understanding in one pure-Rust toolkit — built
the way ffmpeg was built: libraries first, one binary on top, everything
swappable by name.

Part of [Remade With Rust](https://github.com/Remade-With-Rust).

```text
ffai asr -i talk.wav -o talk.srt --engine whisper-candle
ffai tts "hello world" -o hello.wav --voice kokoro
ffai ocr -i receipt.png --engine easy-ocr
ffai caption -i frame.png --prompt "what is happening here?"
ffai engines        # list every engine + status, like `ffmpeg -codecs`
ffai models         # list model manifests, licenses, cache status
```

## Components

| Component | Crate | Task | Namesake |
|---|---|---|---|
| **Mercury** | `ffai-mercury` | ASR + TTS | Roman god of language and messages |
| **Carmenta** | `ffai-carmenta` | OCR | Roman goddess who adapted the Greek alphabet into Latin letters |
| **Argus** | `ffai-argus` | VLM captioning / video understanding | Argus Panoptes, the all-seeing watchman |

Infrastructure: `ffai-core` (types, engine traits, registry — candle is the
tensor spine), `ffai-media` (ingest/egress, backed by
[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)),
`ffai-models` (weight manifests + cache), `ffai-bench` (the analyzer — see
below), `ffai-cli` (the `ffai` binary).

## The analyzer: `ffai bench`

One call compares any FFai engine against world-standard implementations on
a pinned, hash-verified corpus:

```text
ffai bench asr --corpus corpora/asr-smoke-v1.toml            # us vs the world
ffai bench asr --corpus corpora/asr-smoke-v1.toml --baseline-only   # world only
```

It runs our engine and every reference declared in `corpora/references.toml`
(whisper.cpp, faster-whisper, …) over the same holdout clips, scores them
with the same metric code (WER/CER, ×-realtime), and appends an audit-grade
record to `bench/ledger.jsonl` — corpus fingerprint, reference versions,
environment, and a four-gate verdict (correctness / quality / speed /
footprint) where **a skipped gate is never a pass**. Every public claim FFai
makes should trace to a ledger line that makes it reproducible.

The measurement discipline is ported from Prometheus, the private refinery
built for remade_ffmpeg_rs; this public crate carries only the spine (gates,
best-of-N timing, hashed corpora, ledger), no private content. Full
methodology — including why every timing is reported both warm and
end-to-end — is in [docs/benchmarking.md](docs/benchmarking.md).

## Architecture

The load-bearing idea is ffmpeg's: **one trait per task, many engines per
trait, selected by name.** `AsrEngine` / `TtsEngine` / `OcrEngine` /
`VlmEngine` are the `AVCodec` of FFai; `--engine whisper-candle` is
`-c:v libx264`. New models are new engines, not rewrites — and a future
plugin is just an engine registered at runtime.

## Principles

1. **Pure Rust, zero C/C++ by default.** GPU/accelerator backends sit behind
   feature flags; pure-Rust replacements are adopted as they mature (see
   [ROADMAP.md](ROADMAP.md) § Watchlist).
2. **Library-first.** The CLI contains no logic; every capability is a crate
   you can embed.
3. **Candle is the tensor spine.** One `Tensor`/`Device` across all engines —
   buffers flow between models without copies.
4. **Weights are data, not code.** Never vendored; fetched into a cache from
   TOML manifests that surface each model's *own* license (often more
   restrictive than FFai's).
5. **Streaming-first.** Engines process chunks; whole-file is the degenerate
   case.
6. **Oracle-gated.** An engine is `stable` only when validated against its
   reference implementation (WER/CER/perceptual metrics on public corpora).
   Until then it is honestly labeled `experimental` — or `stub`.
7. **Codecs come from home.** Container/codec work routes through
   [remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)
   (`rff-*`), our own pure-Rust ffmpeg.

## Status

**Mercury ASR transcribes today, in pure Rust.** `whisper-candle` runs
OpenAI Whisper on candle with our own mel front-end (STFT + Slaney
filterbank), tokenizer grammar, and decode loop. On a hash-pinned 11-clip
LibriSpeech test-clean holdout, at matched greedy decoding on a CPU:

| Implementation | WER % | ×realtime (warm) |
|---|---:|---:|
| **whisper-candle** tiny.en (Rust) | **3.00** | 7.0 |
| openai-whisper tiny.en (Python) | 3.37 | 21.5 |
| **whisper-candle-base** base.en (Rust) | **1.72** | 3.3 |
| openai-whisper base.en (Python) | 3.12 | 13.5 |

Accuracy is at or better than the reference; **speed is roughly 4× behind**,
which is the honest state of an unoptimized bring-up and the whole subject of
M2. Note that 11 clips is a smoke corpus — enough to gate a milestone, not
enough to publish a comparative accuracy claim. Full details, including the
six methodology and bring-up defects found along the way, are in the
[Mercury mission plan](docs/mercury-mission-plan.md); every number traces to a
line in [`bench/ledger.jsonl`](bench/ledger.jsonl).

TTS, OCR (Carmenta), and VLM (Argus) remain honest `stub`s — visible as such
in `ffai engines`. See [ROADMAP.md](ROADMAP.md) for the build-out order.

## Build

```sh
cargo build --release
cargo test --workspace
target/release/ffai engines
```

## License

MIT OR Apache-2.0 (code). Model weights carry their own licenses — check
`ffai models`.
