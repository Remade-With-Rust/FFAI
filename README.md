# FFAI

**The AI media toolkit, remade with rust.** OCR, speech recognition (ASR),
text-to-speech (TTS), and vision-language understanding in one pure-Rust
toolkit — built the way ffmpeg was built: libraries first, one binary on top,
everything swappable by name. No Python runtime, no C/C++ by default, no
gated weights.

Part of [Remade With Rust](https://github.com/Remade-With-Rust).

```text
ffai asr -i talk.wav -o talk.srt --engine whisper-candle
ffai asr -i talk.wav -o talk.json --word-timestamps   # per-word times (CTC alignment)
ffai asr -i meeting.wav --diarize --max-speakers 3    # who spoke when
ffai asr -i talk.wav -o talk.vtt --word-timestamps    # VTT with inline word timing
ffai tts "Hello from FFai." -o hello.wav               # piper voices, pure Rust
ffai tts -o out.wav --seed 42 "Same seed, same bytes."  # byte-stable synthesis
ffai tts -o long.wav "Long form works. Sentences split; silence is a knob."
ffai ocr -i page.png                                  # CRAFT + CRNN, pure Rust
ffai ocr -i photo.png --engine craft-parseq           # word-level AR rec for photos
ffai ocr --live --watch 5 -i captures/ -o screen.srt  # LIVE: point it at a screen
ffai caption -i frame.png --prompt "what is happening here?"
ffai engines        # list every engine + status, like `ffmpeg -codecs`
ffai models         # list model manifests, licenses, cache status
```

## Components

| Component | Crate | Task | Namesake | Compare |
|---|---|---|---|---|
| **Mercury** | `ffai-mercury` | ASR + TTS | Roman god of language and messages | **ASR live**: full WhisperX layer (VAD · word timestamps · diarization) in pure Rust, **all four gates PASS vs whisper.cpp on both holdouts** — 1.07× / 1.70× its throughput, 0.84–0.92× its memory, line-ball on WER. **TTS live**: piper's own voices on candle, oracle-exact vs piper's runtime, quality parity through a frozen judge, smaller and faster-loading, behind on synthesis speed ([Status](#status)) |
| **Carmenta** | `ffai-carmenta` | OCR | Roman goddess who adapted the Greek alphabet into Latin letters | **OCR live**, with a LIVE streaming mode no mainstream tool ships: change-gated, zero-churn, p95 230 ms/frame vs per-frame Tesseract's 377. Recognition beats PaddleOCR's own recognizer on identical real-photo crops (1.5 % vs 3.0 % CER); full-pipeline photo accuracy still trails PaddleOCR, causes diagnosed ([Status](#status)) |
| **Argus** | `ffai-argus` | VLM captioning / video understanding | Argus Panoptes, the all-seeing watchman | Pending Build |

Infrastructure: `ffai-core` (types, engine traits, registry — candle is the
tensor spine), `ffai-media` (ingest/egress, backed by
[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)),
`ffai-models` (weight manifests + cache), `ffai-bench` (the analyzer — see
below), `ffai-cli` (the `ffai` binary), `ffai-demo` + `demo-ui` (a live
two-tab demo: **Listen** puts Mercury and whisper.cpp on the same microphone
in real time with speaker labels holding steady across chunks, and **Speak**
synthesizes what you type while showing the phonemes our G2P produced, the
sentence split, and a byte-identical-under-a-seed determinism check —
`cargo run --release -p ffai-demo`).

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

**Mercury ASR transcribes today, in pure Rust, and as of 2026-07-30 it passes
all four gates against whisper.cpp on both holdouts — quality, speed,
footprint, correctness.** Speed was the one gate that had never passed;
`whisper-candle` runs OpenAI Whisper on candle with our own mel front-end
(STFT + Slaney filterbank), tokenizer grammar, decode loop, audio encoder,
and four hand-written AVX2 kernels.

Measured on two hash-pinned **134-clip** LibriSpeech holdouts, matched greedy
decoding, CPU only, tiny.en (ledger `bench-asr-1785387940`, `-1785388172`):

| Corpus | Implementation | WER % | CER % | ×realtime (warm) | steady MiB |
|---|---|---:|---:|---:|---:|
| test-clean | **whisper-candle** (Rust) | **7.27** | **2.79** | **27.8** | **179** |
| test-clean | whisper.cpp (C++/ggml) | 7.58 | 2.87 | 25.9 | 195 |
| test-other | **whisper-candle** (Rust) | 16.89 | **8.40** | **33.3** | **163** |
| test-other | whisper.cpp (C++/ggml) | **16.82** | 8.41 | 19.6 | 194 |

**Speed: PASS, and it came from the padding, not from a kernel.** Whisper
pads every window to 30 s, so on ordinary utterances ~78 % of encoder work
was encoding silence — and the encoder is measured O(n) in sequence length.
**Adaptive encoder context** encodes each window at a bucketed context sized
to the audio actually present, with the timestamp grammar masked at the
encoded extent and three guards that escalate a suspect decode back to the
full 30 s context (which is byte-for-byte the old path, so the worst case is
a small extra cost rather than a different transcript). Function-by-function
against whisper.cpp running its own flash-attention default, Mercury now wins
**every stage** — encode ~2.0×, decode 1.1–1.2× (0.77–0.85× per token), mel
1.4×, sampling 1.7–2.0×.

This deserves a note on what it is *not*: a variable-length encoder window was
implemented and **pruned at 268 % WER** earlier in the project. The difference
is not cleverness, it is that the prune predated the repetition guard, the
temperature ladder and the seek loop — the machinery that lets a bad
short-context decode be *detected and re-run* rather than accepted. A
refutation expires when its baseline moves. Full descent, including the two
levers that failed on the way and the four measured iterations it took to
gate clean: [docs/whys/adaptive-context.md](docs/whys/adaptive-context.md).

**Quality: line-ball, and read the per-clip column, not the aggregate.** 7.27
vs 7.58 on clean; on test-other 16.89 vs 16.82 is 0.07 pp *behind* on WER and
0.01 pp ahead on CER. Both sit inside the harness's own gate band, and the
per-clip decomposition of the adaptive-context change is neutral on both
corpora (test-clean 9 improved / 11 worsened, z = −0.45; test-other 12 / 17,
z = −0.93; ~88 % of clips byte-unchanged). Aggregate WER on this box also
moves ±0.2–0.3 pp across rebuilds of identical-behaviour code, so a
sub-0.3 pp corpus delta is not a result here — the paired per-clip counts are.

A related caution the project keeps on the record: **speech segmentation is on
by default**, which whisper.cpp does not do — and segmentation is **not** a
quality mechanism. Turning it off once moved the corpus by 1.20 pp, which
looked like a quality win and was not one: decomposed per clip over 400 clips
it is **38 improved, 38 worsened, a sign test of z = 0.00**, with correlation −0.09
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

**Footprint: PASS — 163–179 MiB steady against whisper.cpp's 194–195 MiB
(0.84–0.92×)**, and ~49–63 MiB of ours is audio the harness holds for the
speed comparison, so the engine itself sits near 120 MiB.

So the standing against whisper.cpp is **ahead on speed, ahead on memory,
line-ball on quality** — and, for the first time, the harness's own verdict
line reads `ALL GATES PASS — claimable` on both holdouts. That is a verdict
against **matched** references (tiny, greedy). It is not a claim against the
field: `openai-whisper-base`, a 74M beam-search model, sits at 5.96 % /
12.14 % and Mercury does not beat it. The harness used to conflate those two
questions by judging quality against the best reference it ran whatever its
size; it now bands the comparison against matched configs, which is the
honest version of the question "is our implementation good" as distinct from
"is tiny.en a good model".

**Long-form is the one corpus still behind** — 6.89 % against whisper.cpp's
6.47 % (ledger `bench-asr-1785388224`), down from **10.55 %** at the start of
the day. The remaining gap is one known failure class: an utterance absorbed
between two contiguous, individually-plausible segment spans, which
span-level coverage accounting structurally cannot see. Two fixes were built
and both measured worse (a lower repair threshold, and narrower windows —
which turned out worse *and* slower at every width). The named bridge is
word-level coverage via the CTC aligner, and it is the next long-form brick
rather than a threshold to tune.

Footprint is judged on **steady** resident memory with peak recorded beside
it, sampled the same way on both sides. Peak is dominated by model load — a
spike over in half a second that never recurs — so judging on it would compare
our load transient against theirs and call the result footprint.

Worth knowing what the bar is: whisper.cpp is not a naive baseline. It runs
**flash attention on by default**, an OpenBLAS backend, runtime ISA dispatch
selecting an AVX-VNNI build, blocked weight repacking, and f16 weights.
Toggling its own `-nfa` flag prices that fused attention at **1.65×** — and
against its *unfused* encoder ours is **1.38× faster**.

Two cautions this project keeps on the record. **Single-run gap ratios are
worthless here** — across ledger runs of the same code the test-clean gap has
read 1.01×–1.29× purely on machine state, and the reference's own throughput
spreads ~37 % of its median — so progress is reported as our own throughput
(22.9 → 27.8–46 ×RT depending on the box's mood) and cross-implementation
ratios only as standing. It is also why the speed claim above rests on the
paired, single-process, interleaved measurements rather than on any one
ledger line's ratio. And the **test-clean CER deficit (3.27 % vs 2.87 %) is
now a lead (2.79 %)** — but it was first closed by a segmentation change
whose per-clip effect is z = 0.00, so it is better described as *displaced*
than as explained. It went unexplained through int8, the f16 cache and every
kernel change since §6.7, and nothing since has said what caused it.

### The WhisperX layer, in pure Rust

Everything WhisperX does, without Python, CUDA, a HuggingFace token, or a
single gated weight — as flags on the same engine rather than a fork.

| Flag | What it adds | Model | Gate |
|---|---|---|---|
| *(default)* | speech segmentation before transcription | none — energy VAD | silence corpus **8/8 empty** |
| `--word-timestamps` | per-word times by CTC forced alignment | wav2vec2-base-960h, Apache-2.0 | containment **100 %**, 1105 words |
| `--diarize` | speaker turns (`SPEAKER_00`…) | ECAPA-TDNN, Apache-2.0 | **DER 4.21 %** |
| `persist_speakers` | those labels survive the **next call** | — | streaming **DER 5.68 %** |

Segmentation is **on by default for measured speed** — 2.2–4.2× on audio with
trailing silence at a byte-identical transcript, and an empty result on
silence with no encoder pass. It also moves corpus WER, and that is **not** a
quality win: 38 improved / 38 worsened across 400 clips, a sign test of
z = 0.00 ([why](docs/whys/vad-quality.md)). The other two stages are opt-in:
they add models and change the output's shape, and nothing that has not
earned a default gets one.

**Streaming is a capability, not a mode.** Diarization labels are, by
convention, arbitrary names for clusters *within one call* — `SPEAKER_00` in
two separate calls need not be the same person. Fine for a file, useless for a
live stream, where the same voice is renamed every chunk. Measured on
conversations fed as 8 s chunks with DER scored over the whole concatenated
timeline: **53.58 % without persistence, 5.68 % with it.** Registry matching is
deliberately stricter than in-call clustering, because a registry merge is
permanent — two people who share a centroid stay merged for the session.

That gap existed because principle 5 says *streaming-first* while diarization
only worked whole-file. WhisperX has the same gap; WhisperX does not claim
streaming-first.

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

### Mercury TTS: piper's voices, pure Rust, deterministic

**`piper-candle` synthesizes speech today** — the full VITS stack (text
encoder with relative-position attention, stochastic duration predictor
with spline flows, residual coupling flow, HiFi-GAN) implemented on candle,
running the SAME voice files [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)
runs, converted locally from the voice's own `.onnx`. The phonemizer is a
clean-room pure-Rust G2P built on CMUdict (BSD): espeak-ng — GPL, and the
reason piper itself is GPL — participates only as an out-of-process test
oracle, and nothing GPL ships. English (en-US) only for now; that is the
honest cost of the license line, stated rather than buried.

**Correctness is oracle-exact.** Against piper's own onnxruntime at zero
noise, every stage matches: text encoder to 4e-6, per-phoneme durations
integer-exact, end-to-end waveform to 3e-5. The pure-Rust phonemizer passed
its substitution gate — our phonemes through piper's runtime score within
the 5 % round-trip band of espeak's own.

**Quality: parity, measured the hard way.** Round-trip WER through a frozen
whisper.cpp judge (never our own ASR — no self-grading): **Mercury 5.91 %,
byte-stable on every run**. Piper's own audio scores 4.8–6.5 % across
ledger runs — it samples noise in-graph and cannot repeat a number, so the
harness scores it as the mean of independent draws with the range recorded.
Read that as parity through this instrument, not superiority; the
instrument cannot support more, and we say so.

**Deterministic by default** — same text, same seed, byte-identical WAV,
verified at both the library and the file-hash level. Piper structurally
cannot offer this. Long-form chunking, speed/noise/seed knobs, and
sentence-silence control ship on `ffai tts`.

**Footprint and load: ahead.** 172–208 MiB steady against piper's 217–240
(0.71–0.87×, gate PASS), model load 0.26–0.35 s against ~1.8–2.6 s.

**Speed: behind, closing, and not claimable — stated plainly.** Synthesis
went from 3.2× realtime at bring-up to **19–20× warm** on a quiet machine
across five profiled campaigns (cache-blocked AVX2 conv kernels,
phase-decomposed upsamplers, a flat decoder, GEMM-shaped flow with
vectorized gates — with nine pruned attempts recorded alongside the wins).
Piper measures 25–32× warm here. Function-by-function against piper's own
runtime, Mercury's upsamplers and duration predictor are ~1.9× FASTER and
the text encoder is at parity; the remaining gap lives in two convolution
kernels with measured targets. The speed gate reads FAIL on every fair
ledger line, and three unfair lines (one flattering us, two flattering
piper) are explicitly disowned in the mission plan with reading
instructions.

Voice weights are converted, never vendored, and the voice's own license is
surfaced per manifest (`models/piper-vits-lessac-medium.toml` — see its
MODEL_CARD note before commercial use). Full campaign history, every
reverted experiment included:
[docs/mercury-tts-mission.md](docs/mercury-tts-mission.md); every number
above traces to a line in [`bench/ledger.jsonl`](bench/ledger.jsonl).

### Carmenta OCR: live, measured against PaddleOCR — the honest split

Two engines run today, both the EasyOCR-lineage stack reimplemented on
candle and oracle-matched against PyTorch (detection maps to <5e-3,
recognition to the exact per-step argmax): `craft-crnn` (line-level CTC,
the default) and `craft-parseq` (word-level AR with the refinement pass).

**Where Carmenta wins, measured:**

- **The recognition stage beats PaddleOCR's own mobile recognizer** on 400
  identical real-photo word crops with quad-level ground truth: **1.5 % vs
  3.0 % CER, 93 % vs 88 % exact-match, at 2.6× lower latency.**
- **LIVE streaming** (a capability the incumbents don't have): change-gated
  frame reading at steady **p95 230 ms vs 377 ms** for per-frame Tesseract,
  **zero output churn** across 156 unchanged frames where stateless engines
  churn 24 times, 24/24 text changes caught, memory flat over a 30-minute
  soak (ratio 1.041).
- Against the lineage it reimplements: **~5× better CER than the EasyOCR
  pipeline on pages** (0.73 % vs 3.65 %) — line-level composition dodges its
  word-segmentation errors.

**Where PaddleOCR still wins, stated plainly:** the full pipeline on real
photographs. On 45 photographed receipts (CORD-v2, CC-BY), PaddleOCR mobile
scores **15.6 % CER to our 27.2 %** — despite our stronger recognition
stage. Stage-level instrumentation localized the gap: tilt-sensitive line
grouping between detection and recognition (deskew is the named fix), a
share of ground-truth inflation from CORD's privacy-blurred regions that
taxes every engine, and detection latency (5.6 s vs 2.9 s per receipt after
adaptive scaling; was 10.5 s). Synthetic corpora tell the same story in
miniature: Paddle 0.02 % vs our 0.31 % on clean pages. The refuted
hypotheses (thresholds, color input) are recorded alongside the confirmed
ones in the mission plan's campaign log.

Full campaign history:
[docs/Carmenta-mission-plan.md](docs/Carmenta-mission-plan.md) §8; every
number traces to [`bench/ledger.jsonl`](bench/ledger.jsonl).

VLM (Argus) remains an honest `stub` — visible as such in `ffai engines`.
See [ROADMAP.md](ROADMAP.md) for the build-out order.

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
