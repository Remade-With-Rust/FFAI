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

| Component | Crate | Task | Namesake | Compare |
|---|---|---|---|---|
| **Mercury** | `ffai-mercury` | ASR + TTS | Roman god of language and messages | ASR live: memory gate PASS vs whisper.cpp (167–183 vs 194 MiB), ahead on noisy speech, ~1.12× behind on speed ([Status](#status)) |
| **Carmenta** | `ffai-carmenta` | OCR | Roman goddess who adapted the Greek alphabet into Latin letters | Pending Build |
| **Argus** | `ffai-argus` | VLM captioning / video understanding | Argus Panoptes, the all-seeing watchman | Pending Build |

Infrastructure: `ffai-core` (types, engine traits, registry — candle is the
tensor spine), `ffai-media` (ingest/egress, backed by
[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)),
`ffai-models` (weight manifests + cache), `ffai-bench` (the analyzer — see
below), `ffai-cli` (the `ffai` binary).

## The analyzer: `ffai bench`

One call compares any FFai engine against world-standard implementations on
a pinned, hash-verified corpus:

```text
ffai bench asr --corpus corpora/librispeech-test-clean-v2.toml          # us vs the world
ffai bench asr --corpus corpora/librispeech-test-other-v1.toml --baseline-only   # world only
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

**Mercury ASR transcribes today, in pure Rust, within ~1.12× of whisper.cpp's
throughput — ahead of it on noisy speech, 0.41 pp behind on clean.**
`whisper-candle` runs OpenAI Whisper on
candle with our own mel front-end (STFT + Slaney filterbank), tokenizer
grammar, decode loop, audio encoder, and four hand-written AVX2 kernels.

Measured on two hash-pinned **134-clip** LibriSpeech holdouts, matched greedy
decoding, CPU only, tiny.en:

| Corpus | Implementation | WER % | CER % | ×realtime (warm) | steady MiB |
|---|---|---:|---:|---:|---:|
| test-clean | **whisper-candle** (Rust) | 7.99 | 3.27 | 28.1–33.1 | **183** |
| test-clean | whisper.cpp (C++/ggml) | **7.58** | **2.87** | **35.5–36.6** | 194 |
| test-other | **whisper-candle** (Rust) | **16.79** | **8.34** | 23.4–26.7 | **167** |
| test-other | whisper.cpp (C++/ggml) | 16.82 | 8.41 | **27.3–29.5** | 194 |

**Quality: split, and the split is a decision we made on purpose.** On the
noisy half we are **ahead** of whisper.cpp (16.79 % vs 16.82 %). On the
clean half we are 0.41 pp behind — **5.4 % relative, outside the 5 % band**.
0.22 pp of that gap is bought, not inherited: Mercury annotates non-speech
events (`[Laughs]`, `(coughs)`) the way whisper.cpp does, instead of
suppressing those tokens the way openai-whisper does. Suppressing them
returns test-clean to 7.77 % / 3.25 % and moves nothing on test-other, so
the cost is real, priced, and one flag away —
`DecodeConfig::suppress_non_speech`. We take it to match the reference we
benchmark against, and carry recovering it as open quality work.

**Footprint: PASS — 167–183 MiB steady against whisper.cpp's 194 MiB
(0.86–0.94×)**, and ~49–63 MiB of ours is audio the harness holds for the
speed comparison, so the engine itself sits near 120 MiB. **Speed: FAIL at
~1.12×** (1.088–1.137× across repeat runs, corroborated by a 21-round paired
test at z = −4.15).

So the standing is a **Pareto position rather than a deficit**: behind on
speed, ahead on memory, ahead on noisy speech, behind on clean. The verdict
is **not claimable yet** — all four gates must pass, and speed does not.
Note that the harness judges quality against the *best* reference of all
those it runs (openai-whisper-base, 5.96 % / 12.14 %), not against
whisper.cpp, so its own verdict line reads `quality FAIL` on both corpora.
The comparison above is the narrower one against whisper.cpp specifically.

Footprint is judged on **steady** resident memory with peak recorded beside
it, sampled the same way on both sides. Peak is dominated by model load — a
spike over in half a second that never recurs — so judging on it would compare
our load transient against theirs and call the result footprint.

Worth knowing what the bar is: whisper.cpp is not a naive baseline. It runs
**flash attention on by default**, an OpenBLAS backend, runtime ISA dispatch
selecting an AVX-VNNI build, blocked weight repacking, and f16 weights.
Toggling its own `-nfa` flag prices that fused attention at **1.65×** — and
against its *unfused* encoder ours is **1.38× faster**.

Two cautions this project keeps on the record. Single-run gap ratios are
worthless here: across six ledger runs of the same code the test-clean gap
reads 1.01×–1.29× purely on machine state, so progress is reported as our own
throughput (22.9 → 33.1 ×RT) and the ratio only as standing. And the widest
quality signal is not WER but **test-clean CER — 3.27 % vs 2.87 %, 14 %
relative** — a deficit that does not appear on test-other and is only
partly explained by the annotation decision above (suppressing annotations
moves CER just 3.27 → 3.25, so ~13 % of the gap remains unexplained).

Full details, including every reverted experiment and the methodology defects
found along the way, are in the [Mercury mission plan](docs/mercury-mission-plan.md)
and [docs/whys/](docs/whys/); every number traces to a line in
[`bench/ledger.jsonl`](bench/ledger.jsonl).

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
