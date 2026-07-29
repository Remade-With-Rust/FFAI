# FFAI

**The AI media toolkit, remade with rust.** OCR, speech recognition, speech
synthesis, and vision-language understanding in one pure-Rust toolkit — built
the way ffmpeg was built: libraries first, one binary on top, everything
swappable by name.

Part of [Remade With Rust](https://github.com/Remade-With-Rust).

```text
ffai asr -i talk.wav -o talk.srt --engine whisper-candle
ffai asr -i talk.wav -o talk.json --word-timestamps   # per-word times (CTC alignment)
ffai asr -i meeting.wav --diarize --max-speakers 3    # who spoke when
ffai asr -i talk.wav -o talk.vtt --word-timestamps    # VTT with inline word timing
ffai tts "hello world" -o hello.wav --voice kokoro
ffai ocr -i receipt.png --engine easy-ocr
ffai caption -i frame.png --prompt "what is happening here?"
ffai engines        # list every engine + status, like `ffmpeg -codecs`
ffai models         # list model manifests, licenses, cache status
```

## Components

| Component | Crate | Task | Namesake | Compare |
|---|---|---|---|---|
| **Mercury** | `ffai-mercury` | ASR + TTS | Roman god of language and messages | ASR live, with the full WhisperX layer (VAD · word timestamps · diarization) in pure Rust: ahead of whisper.cpp on WER+CER on both holdouts and on memory, 1.01–1.09× on speed ([Status](#status)) |
| **Carmenta** | `ffai-carmenta` | OCR | Roman goddess who adapted the Greek alphabet into Latin letters | Pending Build |
| **Argus** | `ffai-argus` | VLM captioning / video understanding | Argus Panoptes, the all-seeing watchman | Pending Build |

Infrastructure: `ffai-core` (types, engine traits, registry — candle is the
tensor spine), `ffai-media` (ingest/egress, backed by
[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)),
`ffai-models` (weight manifests + cache), `ffai-bench` (the analyzer — see
below), `ffai-cli` (the `ffai` binary), `ffai-demo` + `demo-ui` (a live
side-by-side demo — speak into the mic and read Mercury and whisper.cpp
transcribing the same audio in real time: `cargo run --release -p ffai-demo`).

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

**Mercury ASR transcribes today, in pure Rust, ahead of whisper.cpp on WER and
CER on both holdouts, at 1.01–1.09× its throughput — with one caveat about
where part of that quality margin comes from, stated below rather than
buried.**
`whisper-candle` runs OpenAI Whisper on
candle with our own mel front-end (STFT + Slaney filterbank), tokenizer
grammar, decode loop, audio encoder, and four hand-written AVX2 kernels.

Measured on two hash-pinned **134-clip** LibriSpeech holdouts, matched greedy
decoding, CPU only, tiny.en:

| Corpus | Implementation | WER % | CER % | ×realtime (warm) | steady MiB |
|---|---|---:|---:|---:|---:|
| test-clean | **whisper-candle** (Rust) | **6.79** | **2.74** | 32.9 | **183** |
| test-clean | whisper.cpp (C++/ggml) | 7.58 | 2.87 | **33.2–36.6** | 194 |
| test-other | **whisper-candle** (Rust) | **16.43** | **8.07** | 26.7 | **167** |
| test-other | whisper.cpp (C++/ggml) | 16.82 | 8.41 | **29.0–29.5** | 194 |

**Quality: ahead of whisper.cpp on both corpora, and here is the asterisk.**
6.79 vs 7.58 on clean, 16.43 vs 16.82 on noisy, better on CER on both. Those
are the shipped default's real numbers.

Part of that margin comes from **speech segmentation being on by default**,
which whisper.cpp does not do — and segmentation is **not** a quality
mechanism. Turning it off moves us to 7.99 / 16.79. That looks like a 1.20 pp
quality win and is not one: decomposed per clip over 400 clips it is **38
improved, 38 worsened, a sign test of z = 0.00**, with correlation −0.09
between silence removed and WER gained — the opposite sign to the mechanism
originally proposed for it. It shifts where speech sits inside Whisper's fixed
30 s context and re-rolls the decode on about a fifth of clips, half each way.
The aggregate moved because WER is dominated by a handful of high-delta clips.
**Do not expect this margin to transfer to your audio.** Full descent:
[docs/whys/vad-quality.md](docs/whys/vad-quality.md).

Segmentation ships for its speed, which *is* a mechanism: 2.2–4.2× on audio
with trailing silence at a byte-identical transcript, and silence producing an
empty transcript with no encoder pass.

A second deliberate cost sits in these numbers: Mercury annotates non-speech
events (`[Laughs]`, `(coughs)`) the way whisper.cpp does rather than
suppressing those tokens the way openai-whisper does. That costs 0.22 pp on
test-clean and nothing on test-other, and is one flag away —
`DecodeConfig::suppress_non_speech`.

**Footprint: PASS — 167–183 MiB steady against whisper.cpp's 194 MiB
(0.86–0.95×)**, and ~49–63 MiB of ours is audio the harness holds for the
speed comparison, so the engine itself sits near 120 MiB. **Speed: still
behind, but close** — 32.9 vs 33.2 ×RT on clean (1.01×) and 26.7 vs 29.0 on
noisy (1.09×). Single-run ratios on this machine are worthless (the same code
has read 1.01×–1.29× across six ledger runs), so read the throughput, not the
ratio.

So the standing is **ahead on quality against whisper.cpp, ahead on memory,
marginally behind on speed** — with the segmentation asterisk above attached
to the quality half. The verdict is still **not claimable**, for two reasons
worth stating plainly. The harness judges quality against the *best* reference
it runs — `openai-whisper-base` at 5.96 % / 12.14 %, a 74M beam-search model —
not against whisper.cpp, so its own verdict line reads `quality FAIL` on both
corpora. Against **matched** references (tiny, greedy) our 6.79 % is first,
ahead of faster-whisper-tiny-greedy 7.04 %, openai-whisper-tiny-greedy 7.41 %
and whisper.cpp 7.58 %. Those are two different questions and the harness
currently conflates them; splitting the gate is open work.

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
throughput (22.9 → 32.9 ×RT) and the ratio only as standing. And the
long-standing **test-clean CER deficit (3.27 % vs 2.87 %) is now a lead
(2.74 %)** — but by the same segmentation change whose per-clip effect is
z = 0.00, so the deficit is better described as *displaced* than as
explained. It went unexplained through int8, the f16 cache and every kernel
change since §6.7, and nothing since has said what caused it.

### The WhisperX layer, in pure Rust

Everything WhisperX does, without Python, CUDA, a HuggingFace token, or a
single gated weight — as flags on the same engine rather than a fork.

| Flag | What it adds | Model | Gate |
|---|---|---|---|
| *(default)* | speech segmentation before transcription | none — energy VAD | silence corpus **8/8 empty** |
| `--word-timestamps` | per-word times by CTC forced alignment | wav2vec2-base-960h, Apache-2.0 | containment **100 %**, 1105 words |
| `--diarize` | speaker turns (`SPEAKER_00`…) | ECAPA-TDNN, Apache-2.0 | **DER 4.21 %** |

Segmentation is **on by default for measured speed** — 2.2–4.2× on audio with
trailing silence at a byte-identical transcript, and an empty result on
silence with no encoder pass. It also moves corpus WER, and that is **not** a
quality win: 38 improved / 38 worsened across 400 clips, a sign test of
z = 0.00 ([why](docs/whys/vad-quality.md)). The other two stages are opt-in:
they add models and change the output's shape, and nothing that has not
earned a default gets one.

**Licences shaped the design, not just the paperwork.** WhisperX's diarization
depends on pyannote weights that are MIT-licensed *and gated* — permission
granted, access walled behind a browser click. That cannot live in a manifest
under principle 4, so Mercury uses SpeechBrain's ECAPA-TDNN, which is
Apache-2.0 and ungated. Every model FFai fetches is fetchable without an
account.

Three corpora gate this layer, and all three were written **after** the code
they gate — the wrong order, recorded as such. Two found real defects the
moment they ran: the clustering threshold over-splitting five-fold (DER
34 % → 4.21 %), and forced alignment placing every segment-initial word a
systematic 0.17 s early (containment 97.8 % → 100 %). Neither was visible on
the short-clip corpora.

Full details, including every reverted experiment and the methodology defects
found along the way, are in the [Mercury mission plan](docs/finished/mercury-mission-plan.md),
the [Mercury-X plan](docs/finished/mercury-X-mission.md)
and [docs/whys/](docs/whys/); every number traces to a line in
[`bench/ledger.jsonl`](bench/ledger.jsonl).

TTS, OCR (Carmenta), and VLM (Argus) remain honest `stub`s — visible as such
in `ffai engines`. See [ROADMAP.md](ROADMAP.md) for the build-out order.

## Install

All crates are on [crates.io](https://crates.io/crates/ffai-cli) as of 0.4.0:

```sh
cargo install ffai-cli      # the `ffai` binary
cargo add ffai-mercury      # or embed a component as a library
```

Weights are never bundled — they are fetched into a local cache on first use
(or ahead of time with `ffai models --fetch <name>`).

## Build from source

```sh
cargo build --release
cargo test --workspace
target/release/ffai engines
```

## License

MIT OR Apache-2.0 (code). Model weights carry their own licenses — check
`ffai models`.
