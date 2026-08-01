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
ffai asr -i talk.wav --engine whisper-candle-small    # accuracy tier: 3.05% WER
ffai tts "Hello from FFai." -o hello.wav               # piper voices, pure Rust
ffai tts -o out.wav --seed 42 "Same seed, same bytes."  # byte-stable synthesis
ffai tts -o long.wav "Long form works. Sentences split; silence is a knob."
ffai ocr -i page.png                                  # CRAFT + CRNN, pure Rust
ffai ocr -i photo.png --engine craft-parseq           # word-level AR rec for photos
ffai ocr --live --watch 5 -i captures/ -o screen.srt  # LIVE: point it at a screen
ffai detect -i street.jpg --engine yolo26n                # YOLO26, pure Rust
ffai detect -i street.png -o boxes.jsonl --conf 0.4       # structured out
ffai caption -i frame.png --prompt "what is happening here?"
ffai engines        # list every engine + status, like `ffmpeg -codecs`
ffai models         # list model manifests, licenses, cache status
```

## Components

| Component | Crate | Task | Namesake | Compare |
|---|---|---|---|---|
| **Mercury** | `ffai-mercury` | ASR + TTS | Roman god of language and messages | **ASR live**: full WhisperX layer (VAD · word timestamps · diarization) in pure Rust, **all four gates PASS vs whisper.cpp on both holdouts** — and at matched model size ahead on WER, CER *and* speed. Sizes tiny→medium, beam search, 0.84–0.92× its memory. **TTS live**: piper's own voices on candle, oracle-exact vs piper's runtime, **quality parity** through a frozen judge (5.49 % vs 5.27 % WER), **1.58× faster wall-clock at 5 % less CPU**, 10× faster load, and byte-identical output per seed — which piper structurally cannot offer ([Status](#status)) |
| **Carmenta** | `ffai-carmenta` | OCR | Roman goddess who adapted the Greek alphabet into Latin letters | **OCR live**, with a LIVE streaming mode no mainstream tool ships: change-gated, zero-churn, all four gates PASS. Two detector lineages: the mobile-det engines **pass the speed and footprint gates against PaddleOCR on every corpus** — 5.8x faster than CRAFT at 1/12th its memory on screen text — while photo accuracy still trails PaddleOCR, causes diagnosed ([Status](#status)) |
| **Diana** | `ffai-diana` | Object detection | Roman goddess of the hunt — fast, precise detection | **Detection live**: YOLO26 on candle from official Ultralytics `.pt` via an audited offline conversion, **all five tiers from one tier-agnostic graph**. **mAP matches PyTorch to within 0.08 pp across all ten tier/geometry configurations on a 450-image holdout**, exact at n and **every detection identical** — same count, classes and order across 1161 detections — at **1.6–5.6× less memory and up to 10× faster load**, byte-identical to itself at any thread count, JPEG and PNG in, and a concurrent batch path PyTorch's GIL cannot match. **Per-image latency is ~1.75× behind at every tier — the one gate that fails, published with its number** ([Status](#status)) |
| **Argus** | `ffai-argus` | VLM captioning / video understanding | Argus Panoptes, the all-seeing watchman | Pending Build |

Infrastructure: `ffai-core` (types, engine traits, registry — candle is the
tensor spine), `ffai-media` (ingest/egress, backed by
[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)),
`ffai-models` (weight manifests + cache), `ffai-bench` (the analyzer — see
below), `ffai-cli` (the `ffai` binary), `ffai-demo` + `demo-ui` (a live
three-tab demo: **Listen** puts Mercury and whisper.cpp on the same microphone
in real time with speaker labels holding steady across chunks, **Speak**
synthesizes what you type while showing the phonemes our G2P produced, the
sentence split, and a byte-identical-under-a-seed determinism check, and
**Read** takes an image you drop or paste and runs both OCR lineages over the
identical pixels — with the content classifier showing which one the pipeline
would dispatch to, so the measured sign-flip is something you can falsify on
your own screenshot rather than take on trust —
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

**Quality at tiny.en: line-ball, and read the per-clip column, not the
aggregate.** 7.27 vs 7.58 on clean; on test-other 16.89 vs 16.82 is 0.07 pp
*behind* on WER and 0.01 pp ahead on CER. Both sit inside the harness's own
gate band. Aggregate WER on this box moves ±0.2–0.3 pp across rebuilds of
identical-behaviour code, so a sub-0.3 pp corpus delta is not a result here —
the paired per-clip counts are.

That parity is not a limitation of the implementation, it is the *model*:
`whisper-candle` runs OpenAI's tiny.en weights, so its accuracy ceiling is
Whisper's. Two levers change that, and both now exist.

### Accuracy tiers, and beam search

**`small.en` more than halves the error rate** (test-clean, greedy, same
harness, 200 clips):

| model | WER % | CER % | ×realtime |
|---|---:|---:|---:|
| tiny.en | 6.39 | 2.31 | 19.9 |
| base.en | 5.16 | 2.10 | 8.3 |
| **small.en** | **3.05** | **0.88** | 4.2 |

And at the **same** model size — the only comparison that prices the
implementation rather than the weights — Mercury is ahead of whisper.cpp on
quality *and* speed:

| implementation | WER % | CER % | ×realtime |
|---|---:|---:|---:|
| **mercury** small.en | **3.05** | **0.88** | **4.2** |
| whisper.cpp small.en | 3.38 | 1.16 | 3.7 |

16 clips better, 8 worse, 176 tied — z = +1.63, which is *under* this
project's |z| > 2 bar, so that reads "ahead, direction consistent, not yet
significant" rather than as a proven win.

**Beam search** (`FFAI_BEAM_SIZE=5`, greedy stays the default) is the other
lever — what all three references run by default, and what Mercury lacked
until now. It clears the bar, but only pooled: test-clean 6.36 → 5.76 % (19
improved / 9 worsened, z = +1.89) and test-other 13.87 → 13.12 % (25 / 15,
z = +1.58) are each below |z| > 2 alone; **pooled, 44 / 24 gives z = +2.43**
on WER and +2.32 on CER. It costs ~5×, spending the whole speed surplus above.

Full descent, including a harness that briefly reported an 18 pp lead that
did not exist: [docs/whys/quality-routes.md](docs/whys/quality-routes.md).

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

So the standing against whisper.cpp is **ahead on speed, ahead on memory, and
— at matched size — ahead on quality too**, with the harness's own verdict
line reading `ALL GATES PASS — claimable` on both holdouts. Every one of
those is a verdict against **matched** references: same weights, same
decoding strategy. Comparing our `small.en` against their `tiny.en` would
price the model rather than the implementation, which is the error the
reference file exists to prevent.

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

One caution this project keeps on the record: **single-run gap ratios are
worthless here.** Across ledger runs of the same code the test-clean gap has
read 1.01×–1.29× purely on machine state, and the reference's own throughput
spreads ~37 % of its median. So progress is reported as our own throughput
and cross-implementation ratios only as standing — and the speed claims above
rest on paired, single-process, interleaved measurements rather than on any
one ledger line's ratio.

### The WhisperX layer, in pure Rust

Everything WhisperX does, without Python, CUDA, a HuggingFace token, or a
single gated weight — as flags on the same engine rather than a fork.

| Flag | What it adds | Model | Gate |
|---|---|---|---|
| *(default)* | speech segmentation before transcription | none — energy VAD | silence corpus **8/8 empty** |
| `--word-timestamps` | per-word times by CTC forced alignment | wav2vec2-base-960h, Apache-2.0 | containment **100 %**, 1105 words |
| `--diarize` | speaker turns (`SPEAKER_00`…) | ECAPA-TDNN, Apache-2.0 | **DER 4.21 %** |
| `persist_speakers` | those labels survive the **next call** | — | streaming **DER 5.68 %**, 3.16× faster live |

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

**Streaming diarization is also 3.16× cheaper than it was**, which is what
makes it usable live rather than merely correct. Speaker embedding is ~100 %
of diarization's cost (~172 ms per 1.5 s window; the filterbank is 1.1 % and
clustering ~0), and a live caller re-sending a sliding window was
re-embedding almost the same audio every tick. Two changes fixed that: a
cache keyed on each window's **samples** rather than its timestamps, and an
**absolute window grid** (`AsrOptions::stream_offset_secs`) so the same audio
lands on the same window bounds no matter where the buffer starts. Paired
against the same ticks: median 1258 → 720 ms, **10/10 paired wins**, at
**4.20 % DER against 4.21 %** region-anchored.

That accuracy line is a correction. The grid shipped in 0.6.0 described as
DER-neutral and it was not — the first version snapped each window chain
forward, skipping up to 0.75 s of every region's leading audio, and cost
**4.21 % → 9.60 %**. It read as neutral because the gate was re-run against a
stale example binary. The fix emits the region-start window before following
the grid, which restores coverage; the honest speedup is 1.75×, not the 3.16×
first reported, because part of that was accuracy nobody had priced.

The diagnosis is worth more than the number, and it was not what reading the
code suggested — the leading speech region is *clipped by the buffer's edge*,
so its windows anchor to the buffer and slide with it while the audio beneath
moves. Full descent, including a hypothesis stated, withdrawn, and then
restored by the trace: [docs/whys/diarization-cost.md](docs/whys/diarization-cost.md).

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
runs — fetched from the public, ungated `rhasspy/piper-voices` and read
straight from ONNX by our own pure-Rust reader, with no conversion step, no
Python, and no ONNX runtime. The phonemizer is a clean-room pure-Rust G2P
built on CMUdict (BSD): espeak-ng — GPL, and the reason piper itself is GPL
— participates only as an out-of-process test oracle, and nothing GPL ships.
English (en-US) only for now; that is the honest cost of the license line,
stated rather than buried.

**Correctness is oracle-exact.** Against piper's own onnxruntime at zero
noise, every stage matches: text encoder to 4e-6, per-phoneme durations
integer-exact, end-to-end waveform to 3e-5. The pure-Rust phonemizer passed
its substitution gate — our phonemes through piper's runtime score within
the 5 % round-trip band of espeak's — and the ONNX reader that replaced the
Python converter is byte-identical to it across 350 tensors and 132
convolution geometries.

**Quality: parity, measured the hard way.** Round-trip WER through a frozen
whisper.cpp judge (never our own ASR — no self-grading): **5.49 % against
piper's 5.27 %** on the same holdout, same judge, same run.

**Both engines are scored across draws, because one of them has to be.**
Piper samples noise inside its ONNX graph and cannot repeat a run, so its
WER is the mean of independent draws with the range recorded. Ours is
seeded and byte-stable, so it would otherwise report one fixed draw
forever — and our own seed-to-seed spread is **1.11 pp** (4.99–6.10 %),
several times larger than any recent engine change. Scoring both the same
way is the only comparison that means anything.

**Deterministic by default** — same text, same seed, byte-identical WAV,
verified at both the library and the file-hash level. Piper structurally
cannot offer this. Long-form chunking, speed/noise/seed knobs, and
sentence-silence control ship on `ffai tts`.

**Speed: ahead, on a reference that will not hold still.** Recent ledger
lines read **19.3–21.2× realtime warm against piper's 11.2–23.0×**, and
**load 0.64 s against 6.69 s**. But piper's own throughput spans
**4.5×–29× across the ledger's TTS lines** on this machine and its WER
4.8–6.5 %, both wider than any difference between the two engines — so the
gates flip run to run on *its* variance, and single-run ratios are not a
claim. The number that survives is measured on both engines simultaneously:
**1.58× faster wall-clock while using 5 % less total CPU.** Footprint is
parity, 214 MiB steady against 217.

**Per stage, we are within ~2.5 points of share on all four.** That took
correcting the instrument twice: onnxruntime's own profiler slows it
1.75–1.93× with a non-uniform per-node tax (correcting for it drove a stage
to a *negative* time, which is how it was caught), so the reference is now
timed by cutting its graph into cumulative prefixes and running each
unprofiled. Every per-stage figure recorded before that correction is
withdrawn. The full descent, including the levers that were measured and
rejected, is in
[docs/whys/tts-speed-gap.md](docs/whys/tts-speed-gap.md).

Voice weights are converted, never vendored, and the voice's own license is
surfaced per manifest (`models/piper-vits-lessac-medium.toml` — see its
MODEL_CARD note before commercial use). Full campaign history, every
reverted experiment included:
[docs/mercury-tts-mission.md](docs/mercury-tts-mission.md); every number
above traces to a line in [`bench/ledger.jsonl`](bench/ledger.jsonl).

### Carmenta OCR: live, measured against PaddleOCR — the honest split

Four engines run today, from two detector lineages crossed with two
recognizers, all oracle-matched against their references. The EasyOCR
lineage on candle (detection maps to <5e-3, recognition to the exact
per-step argmax) gives `craft-crnn` (line-level CTC, the default) and
`craft-parseq` (word-level AR with the refinement pass). Swapping in a
PP-OCRv5 mobile detector — DBNet on PP-LCNetV3, 4.7 MB, reproducing
paddle's own exported program to **zero binarised disagreement** across a
pinned page — gives `mobiledet-crnn` and `mobiledet-parseq`.

The two detector lineages trade against each other rather than ranking, and
the measurement says so plainly. On real-photo receipts mobile-det is **3×
faster and 3.2× leaner** than CRAFT — the first configuration to pass the
speed and footprint gates there, and leaner than PaddleOCR itself — and
**worse on quality** (37.3 % vs 27.3 % CER), because DBNet emits line regions
and a receipt separates its labels from its amounts across two columns.
CRAFT's character components carry structure a line detector discards. Pick by
corpus class, not by leaderboard.

**Where Carmenta wins, measured:**

- **The recognition stage beats PaddleOCR's own mobile recognizer** on 400
  identical real-photo word crops with quad-level ground truth: **1.5 % vs
  3.0 % CER, 93 % vs 88 % exact-match, at 2.6× lower latency.**
- **LIVE streaming** (a capability the incumbents don't have): change-gated
  frame reading at steady **p95 257 ms vs 343 ms** for per-frame Tesseract,
  **zero output churn** across 156 unchanged frames where stateless engines
  churn 24 times, 24/24 text changes caught, memory flat over a 30-minute
  soak (ratio 1.041).
- Against the lineage it reimplements: **~5× better CER than the EasyOCR
  pipeline on pages** (0.73 % vs 3.65 %) — line-level composition dodges its
  word-segmentation errors.

**Where PaddleOCR still wins, stated plainly:** the full pipeline on real
photographs. On 45 photographed receipts (CORD-v2, CC-BY), PaddleOCR mobile
scores **15.6 % CER to our 20.9 %** (`--engine craft-parseq`; the default
`craft-crnn` reads 27.3 % there but wins the screen/HUD class, so the
engine is a per-content choice, not a ranking) — despite our stronger recognition
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

### Diana: YOLO26 detection, from the official `.pt`, in pure Rust

**Diana runs the real YOLO26 graph on candle — C3k2, SPPF, C2PSA, the
NMS-free end-to-end head — built from official Ultralytics checkpoints by an
audited offline conversion. No Python at inference. No weights in this
repo:** they are AGPL-3.0 and stay the user's to fetch, convert and license.

Measured on a hash-pinned **450-image** COCO holdout, CPU only, every tier in
both geometries, each graded only against the reference declaring the same
configuration. mAP at conf 0.001 / 100 dets like the reference. Memory is
steady working set with the harness's own pre-decoded image cache subtracted
from both sides (ledger `bench-detect-1785550365` and the ten rows after it):

| Tier | Geometry | Diana mAP50 | Ultralytics | Δ pp | Diana MiB | Ultralytics MiB | leaner |
|---|---|---:|---:|---:|---:|---:|---:|
| n | rect | 61.36 | 61.36 | **−0.01** | **71** | 403 | 5.6× |
| n | square | 61.01 | 61.01 | **−0.00** | **63** | 317 | 5.0× |
| s | rect | 68.07 | 68.16 | −0.08 | **110** | 447 | 4.1× |
| s | square | 68.88 | 68.84 | +0.04 | **120** | 382 | 3.2× |
| m | rect | 73.14 | 73.07 | +0.06 | **178** | 544 | 3.1× |
| m | square | 73.30 | 73.23 | +0.07 | **201** | 456 | 2.3× |
| l | rect | 74.07 | 74.02 | +0.06 | **197** | 535 | 2.7× |
| l | square | 74.13 | 74.16 | −0.03 | **220** | 490 | 2.2× |
| x | rect | 77.31 | 77.38 | −0.07 | **346** | 678 | 2.0× |
| x | square | 77.69 | 77.67 | +0.03 | **380** | 603 | 1.6× |

**Worst deviation across all ten: 0.08 pp**, exact at n, and Diana reads
*higher* than the reference in five of the ten. Memory is 1.6–5.6× leaner,
the ratio narrowing as the model grows because the weights come to dominate
what is left. The footprint gate passes on all ten.

Those memory figures are what the ledger says, which was not true a day ago.
The harness pre-decodes every clip so the *speed* measurement excludes image
decoding — then holds 355 MiB of buffers that the reference, running as a
subprocess, never pays. At 45 images that cache was 37 MiB and invisible; at
450 it inverted the gate, reading Diana as a **regression** at 413 MiB
against 396 while it actually used 58. The harness now charges its own cache
to itself, and the table above is a re-run rather than a hand-correction. A
measurement that only works at one input size has an unstated precondition.

The stronger statement sits
under it: at n the *detections themselves* are identical — same count, same
classes, same order, across all 1161 boxes (measured on the 45-image
corpus; the 450-image run reproduces the mAP exactly but the box-by-box
comparison has not been re-run at that size). And Diana is byte-identical to
itself across runs and across thread counts, which PyTorch does not promise.

Two disciplines produced that. Every layer was checked against a tracked
oracle digest dumped from the reference before any Rust ran, and **geometry
is a required argument, not a default.** `predict()` silently letterboxes
rectangularly, so "YOLO's mAP" is two different numbers and M-D0 lost
1.5–1.8 pp to an unpinned one. Diana names which one it is reporting.

**That 1.5–1.8 pp was mostly the corpus.** It came from 45 images. Re-measured
on 450, the geometry effect collapses to **0.14–0.68 pp** — at or under the
band the quality gate itself uses to decide whether two numbers differ. The
*sign* still flips by tier (n prefers rectangular, every larger tier prefers
square) and that was tempting to wire as a per-tier default; the magnitude
says don't. Naming the geometry matters because comparing a rectangular
engine against a square baseline is a 1.5 pp error of pure bookkeeping.
Choosing between them barely matters at all.

**All five tiers — n/s/m/l/x — come from one tier-agnostic graph**, 2.4 M
parameters to 55.7 M, and each is oracled independently against its own
reference dump (worst relative delta 3.8e-6 to 8.7e-6). Widths, repeat
counts and block kinds are derived from the checkpoint's own scale row, and
the strict loader fails closed on any mismatch.

**Every new tier found a bug the previous ones hid, which is the argument
for having them.** s found a head width that was right at n only because
`max(64, 80)` and `nc` both equal 80. m found that `parse_model` *overrides*
the YAML's `c3k` flag for scales in `mlx`, turning two `Bottleneck` layers
into `C3k` — a branch n and s never reach, on a board that was green at
3.869e-6. A constant that is right only because the configurations you have
tested happen to agree is found by configuration N+1, never by testing
configuration 1 harder.

**Speed is the open gate: ~1.75× behind Ultralytics on per-image latency,
consistently at every tier.** The repo's rule is that `verdict: claimable`
needs all four gates; Diana does not get it. Quality, footprint and
correctness are each claimed individually because each is gated. The losing
row goes in the table.

That number took three tries to state correctly, and the two wrong versions
are worth more than the right one. It was first published as "~2.5× behind",
generalised from the n tier alone. The corpus sweep then appeared to show
the gap narrowing monotonically with model size — 2.24× at n down to 0.94×
at x, apparently *ahead* at the largest tier — which would have been a much
better story. It was an artifact: **the sweep runs tiers in order n→x over
several hours, so tier index and wall-clock time are perfectly confounded**,
and this box's load drifts. Two checks killed it. A *null arm* — the same
engine, same corpus, same config, run twice — moved the headline ratio 27%
with nothing changed, while the reference's own throughput moved 37%. And
re-running the tiers in **reverse order** flattened the trend to nothing:

| tier | n→x order | x→n order | mean |
|---|---:|---:|---:|
| n | 1.67× | 1.86× | 1.77× |
| s | 1.79× | 1.90× | 1.85× |
| m | 1.33× | 2.12× | 1.73× |
| l | 1.65× | 1.70× | 1.68× |
| x | 1.57× | 1.62× | 1.60× |

The m cell alone swung 59% on running order. No tier is at parity; the gate
fails everywhere. Underneath, Diana does ~1.7× more CPU work than PyTorch to
produce the identical answer.

What is *behind* the row is now diagnosed rather than admitted. A six-whys
descent ([docs/whys/diana-latency.md](docs/whys/diana-latency.md)) found the
**activation at 30.9 % of a detection** — larger than any convolution shape,
and the third time this repo has found an activation at the top of a profile.
The cause was one function: `f32::round` is ties-away-from-zero, which no x86
instruction implements, so it blocked vectorisation of the whole loop. The
module had already removed `exp` for precisely that reason and left `round`
behind. Rounding by float addition instead is **4.71× on the kernel and
bit-identical**, worth **1.079× on the pipeline** (17/21 paired rounds,
z = +2.84).

Two levers were priced and **pruned**, which is worth as much as the fix.
Cutting rayon's thread count saves 1.66× of CPU work but is *not* faster on
wall (9/24, z = −1.22) — for a single image the alternative is 15 idle cores.
And the crate compiles for the x86-64 baseline with no `target-cpu`, worth
1.39× on the *old* SiLU kernel and **1.017× on the pipeline once the SiLU fix
landed** (z = +0.65, inside the noise), because AVX2 was mostly rescuing
`round()`. A confirmation expires when its baseline moves; runtime ISA
dispatch would have been built on a number that no longer existed.

What remains is structural: **~120 fork-joins per image cost 2.3–3.8× the CPU
of the work they perform.** That tax does fall with model size — 3.81× at n
to 0.83× at x, measured on our own serial baseline — so our parallel
efficiency genuinely improves as tensors grow. It just doesn't open a gap
against Ultralytics, because theirs improves in step. Both amortize; neither
pulls ahead. That is the largest named thing between Diana and the
reference, and it is not a flag.

**One gate turned out to be measuring luck, and the way that was
established is the point.** The oracle asserted that our 300-row top-k
matched the reference's row for row. On the l tier it failed while every
layer above it matched at 4.3e-6. Rather than hunt a decode bug, we
perturbed the *reference's own* tensors by exactly the f32 divergence we
measure against it and re-ran *its own* selection: it could not reproduce
its own ordering either — 581 px and 6 class flips on the **n** case that
was passing. The two-stage top-k ranks 8400x80 candidates to 300, and where
there are no real detections the rows near the cut sit 1e-8 apart, so their
order is undefined and asserting on it tests the weather. The gate now
checks the tie-robust confidence sequence for all 300 rows and positional
box/class identity only above a measured confidence floor — and when no row
clears it, it says so rather than reporting a pass.

One methodology note worth repeating, because it inverted a result: A/B on
**CPU time**, not wall time, when the box is loaded — but CPU time sums
across threads, so it *dilutes* a change that removes serial work by the
thread count. Zero-copy read 22/22 wins on wall, 8/22 (z = −1.28) on
24-thread CPU time, and 19/22 (z = +3.41) at one thread. The A/B harness
encodes the rule.

Full campaign history:
[docs/diana-mission-plan.md](docs/diana-mission-plan.md) §8; every number
traces to [`bench/ledger.jsonl`](bench/ledger.jsonl).

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
