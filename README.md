# FFAI

**The AI media toolkit, remade with rust.** OCR, speech recognition (ASR),
text-to-speech (TTS), and vision-language understanding in one pure-Rust
toolkit — built the way ffmpeg was built: libraries first, one binary on top,
everything swappable by name. No Python runtime, no gated weights.

**One correction to a claim this page carried for weeks:** it said "no C/C++
by default", and that is not true. `candle-core` takes `tokenizers` as a
hard, non-optional dependency with `features = ["onig"]`, so `onig_sys`
compiles Oniguruma — a C regex engine — into every FFai build that touches
candle. It is build-time only, it produces an ordinary binary with no runtime
dependency, and it disappears on `wasm32` where candle target-gates the
dependency out. `tokenizers` marks `onig` optional and ships a pure-Rust
alternative, so removing it is one feature line upstream in candle rather
than a redesign here. Stated at the top because a supply-chain claim that is
wrong is worse than one that is qualified.

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
| **Carmenta** | `ffai-carmenta` | OCR | Roman goddess who adapted the Greek alphabet into Latin letters | **OCR live**, with a LIVE streaming mode no mainstream tool ships: change-gated, **zero churn across 156 unchanged frames** where stateless Tesseract churns 24 times. On the full **OmniDocBench** holdout: **20.3 % CER, 236/236 correctness**, reading order computed by projection rather than learned — and **89 % of the remaining gap to PP-StructureV3 is sequence, not characters** (order-free CER within 1.40 pp). Against Baidu Unlimited-OCR: 25.9 % vs 15.5 % on a matched 43-page subset, at **17x the throughput on CPU** from 4.7 MB of detector weights against 6.4 GB. Photo accuracy still trails PaddleOCR, causes diagnosed ([Status](#status)) |
| **Diana** | `ffai-diana` | Object detection | Roman goddess of the hunt — fast, precise detection | **Detection live**: YOLO26 on candle from official Ultralytics `.pt` via an audited offline conversion, **all five tiers from one tier-agnostic graph**. **mAP matches PyTorch to within 0.08 pp across all ten tier/geometry configurations on a 450-image holdout**, exact at n and **every detection identical at n, m, l and x** — same count, classes and order across 724 detections, boxes within 0.30 px — at **1.6–5.6× less memory and up to 10× faster load**, byte-identical to itself at any thread count, JPEG and PNG in. **Ahead of Ultralytics per image at identical mAP — 0.70x and 0.58x across two paired runs after a harness bug that cost us ~14 % was fixed — on 121 MiB against 310 MiB** (yolo26n/640 rect, 45-image holdout). **Accuracy holds on MOT17: 0.029 pp mean gap over 5,316 frames of a public benchmark.** Speed is the one failing gate: against **ONNX Runtime** it is **2.89× at matched square geometry**, which Diana beats on accuracy (0.7014 vs 0.6865). Video in through the pure-Rust `rff` stack; a LIVE change gate that fires on 0.8 % of real surveillance frames ([Status](#status)) |
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
   (`rff-*`), our own pure-Rust ffmpeg. Still images decode through the
   standalone crates from that same workspace —
   [`rusty_png`](https://crates.io/crates/rusty_png) (a performance fork of
   image-rs/image-png) and
   [`rusty_jpeg`](https://crates.io/crates/rusty_jpeg) (baseline +
   progressive DCT, AVX2 FDCT/quantize, two-block AVX2 IDCT) — both from
   crates.io, so owning the stack no longer costs publishability. Each is
   gated against the implementation it replaces: `rusty_png` byte-identical
   to upstream `png` across every corpus image, `rusty_jpeg` within 3/255 of
   libjpeg.

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

**Documents, against the best in class.** On 43 pages of
[OmniDocBench](https://github.com/opendatalab/OmniDocBench) (Apache-2.0 — the
benchmark Baidu's Unlimited-OCR states its record on), identical pixels, every
number a ledger line:

| 43 real document pages | CER | pages/s | memory |
|---|---:|---:|---:|
| Unlimited-OCR (Baidu, 3B MoE, **GPU**) | **15.51 %** | 0.01 | 8745 MiB peak |
| PP-StructureV3 | 19.14 % | 0.02 | 1481 MiB steady |
| **`mobiledet-crnn` (ours, CPU)** | **23.76 %** | **0.15** | **425 MiB steady** |

**10.4 points behind the model that holds the record, at 17× its throughput on
a machine with no GPU**, from 4.7 MB of detector weights against 6.4 GB —
`correctness PASS · quality FAIL · speed PASS · footprint PASS`. Not parity.
The same order of magnitude, in a deployment class neither reference can enter.

Full campaign history:
[docs/Carmenta-mission-plan.md](docs/Carmenta-mission-plan.md) §8; every
number traces to [`bench/ledger.jsonl`](bench/ledger.jsonl).

### Diana: YOLO26 detection, from the official `.pt`, in pure Rust

**Diana runs the real YOLO26 graph on candle — C3k2, SPPF, C2PSA, the
NMS-free end-to-end head — built from official Ultralytics checkpoints by an
audited offline conversion. No Python at inference. No weights in this
repo:** they are AGPL-3.0 and stay the user's to fetch, convert and license.

Standalone landing page:
[Remade-With-Rust/diana](https://github.com/Remade-With-Rust/diana).

**Embedding it pulls detection only.** Diana has no dependency on Mercury,
Carmenta or Argus — they are siblings, not layers underneath:

| build | transitive crates | compiles C? |
|---|---:|---|
| `ffai-diana`, default | **138** | yes — `onig_sys` |
| `ffai-diana` + `ffai-models/fetch` | 308 | yes — `onig_sys`, `aws-lc-sys` |
| **`wasm32-unknown-unknown`** | **95** | **no** |

The 170-crate gap on native is the Hugging Face downloader — `reqwest`,
`hyper`, `rustls`, `aws-lc-sys` — which Diana never calls; its whole use of
`ffai-models` is `load_dir`, reading TOML off disk. Off by default there.

**The tree does compile C on native, and it is worth stating plainly because
this README claimed otherwise for weeks.** `candle-core` takes `tokenizers` as
a hard, non-optional dependency with `features = ["onig"]`, and `onig_sys`
compiles Oniguruma. It is build-time only — the output is an ordinary binary
with no runtime dependency — and it vanishes on wasm32, where candle
target-gates that dependency out. `tokenizers` marks `onig` optional and ships
a pure-Rust alternative, so the fix is one feature line upstream in candle.

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

The stronger statement sits
under it: the *detections themselves* are identical — same count, same
classes, same order — at **four tiers**, not just the smallest:

| tier | detections | class agreement | max box delta |
|---|---:|---|---:|
| n | 131 | 131/131 | 0.084 px |
| m | 204 | 204/204 | 0.300 px |
| l | 187 | 187/187 | 0.170 px |
| x | 202 | 202/202 | 0.213 px |

**The speed picture changed when the harness was fixed.** The bench
pre-decoded the whole corpus before timing, which looks like it favours us —
our reader sits outside the timed region while the reference pays for its
decode inside it — and does the opposite. The reference decodes each image
just before using it and reads a buffer its own decoder just wrote; ours was
written 45 images ago and had to be fetched. **The harness was handicapping us
by ~14 %.** ABBA, three reps: just-in-time decode is 11-17 % faster despite
ADDING the decode to the timed region, and a working-set sweep shows holding 1
image versus 45 is FLAT — the mechanism is recency, not residency. JIT is now
the default.

With that fixed, two clean paired runs at rect put Diana **ahead of
Ultralytics**: 32 ms against 46 (0.70x) and 37 against 64 (0.58x). **That is
two runs and is labelled as two runs** — a third read 382 ms on a box that had
been benchmarking for hours (rtf 2.6 against 25-29) and is excluded as machine
noise, stated here rather than buried in a median. Before the fix, seven
paired runs read a median of 1.11x BEHIND, and a single favourable run had
been quoted as "faster than Ultralytics" and retracted. The current reading is
ahead; it is not yet a settled result.

Against **ONNX Runtime** the gate still fails: **2.89x at matched square
geometry**, measured under the old harness and not re-measured since, so it is
an upper bound. ORT has no rect export, so the widely-quoted 1.25x compared
our reduced-work rect against its full-work square — rect is 70-75 % of
square's pixels.

mAP is identical to PyTorch to four decimals in both geometries, and memory is
the unambiguous win at 0.4x Ultralytics and 0.75x ORT. The repo's rule is
that `verdict: claimable` needs all four gates; Diana does not get it. The
losing row goes in the table.

### Accuracy holds on a public benchmark, not just our corpus

All seven MOT17 training sequences, **5,316 frames**, dataset ground truth,
identical extracted frames to both engines:

| seq | camera | frames | Diana AP50 | ultralytics | gap pp |
|---|---|---:|---:|---:|---:|
| 02 | static | 600 | 21.55 % | 21.54 % | +0.01 |
| 04 | static | 1050 | 23.07 % | 23.06 % | +0.01 |
| 05 | moving | 837 | 56.14 % | 56.04 % | +0.10 |
| 09 | static | 525 | 62.35 % | 62.37 % | -0.02 |
| 10 | moving | 654 | 35.92 % | 35.95 % | -0.03 |
| 11 | moving | 900 | 56.96 % | 56.92 % | +0.03 |
| 13 | moving | 750 | 25.62 % | 25.62 % | -0.00 |

**Mean absolute gap 0.029 pp** across scenes spanning 21.55 % to 62.35 %, so
the agreement is not an artefact of one easy sequence. Ahead on four, behind
on three, every one inside 0.1 pp. Reproduce with
`tools/diana_mot_bench.py --all`.

The **LIVE change gate** was measured on the same 5,316 frames and the result
is a qualifier, not a headline: it fires on **0.8 %** of them, at an accuracy
cost of **-0.006 pp**. On a synthetic still scene it gates 46 of 48 frames;
on real surveillance footage with pedestrians in it, almost nothing. It is for
a static SCENE, not merely a fixed CAMERA. Forced to gate a scene that WAS
changing, AP50 fell 45 points — the threshold is a correctness boundary, not a
tuning knob.

| tier | n→x order | x→n order | pinned floor |
|---|---:|---:|---:|
| n | 1.67× | 1.86× | 1.98× |
| s | 1.79× | 1.90× | 2.20× |
| m | 1.33× | 2.12× | 1.82× |
| l | 1.65× | 1.70× | 1.83× |
| x | 1.57× | 1.62× | 1.54× |

### Watch it run against Ultralytics

```
python tools/diana_sbs_viewer.py --frames corpora/clips/mot17-09/img1     --weights yolo26n.pt        # your own AGPL checkpoint
```

Both engines on screen, fed the identical frames in the identical order, with
per-frame latency, rolling frame rate, running medians, object counts and box
agreement. `--live` puts the change gate in the loop and badges the frames it
skips; `--video clip.mp4` extracts frames first; `--record out.mp4` saves the
composed view.

**The engines alternate and never run concurrently** — Diana's reply is read
to completion before Ultralytics is called, so each has the whole machine
while the other is idle. Run at the same time, each latency would be measuring
the other. Both decode inside their own timed region. It is still a demo, not
the benchmark: no min-of-N, no warm-up discipline, display in the loop, and
[`bench/ledger.jsonl`](bench/ledger.jsonl) is what the claims trace to.

It has already shown something the 640 corpus does not — though not what was
first published here. Two observations taken from it, *"larger frames erase the
advantage"* and *"Diana's latency tail is heavier"*, were both chased down and
**both are refuted**; the descent is
[docs/whys/diana-1080p-and-tail.md](docs/whys/diana-1080p-and-tail.md).

**Frame size does not erode the advantage.** In rect mode a 1920x1080 frame
letterboxes to 640x384 = 246 kpx, while a near-square COCO image letterboxes to
608x640 = 389 kpx — a 1080p frame is **37 % LESS model work**, not more.
Measured ABBA on CPU time, both engines in child processes, model load
excluded, the ratio is **3.26x at 1080p and 3.25x at 640**: identical to three
significant figures across a resolution change that moves model work by 37 %.
The "parity" readings were wall-clock, on a box whose null arm — Diana against
Diana, identical code — read a **10.4 % floor with 47 % within-arm spread**.
They were never distinguishable from each other.

That CPU ratio carries its own caveat, and it is the honest one: **3.25x less
CPU at a wall ratio near 1.0 means Ultralytics converts more cores into the
same wall clock.** The claim is "Diana does the work for a third of the CPU",
not "Diana is 3.25x faster" — different products, one for a shared server and
one for an idle laptop.

**And the tail is ours only in the good direction.** The original figures came
from running all of Diana and then all of Ultralytics, which puts machine drift
between the blocks. Interleaved per frame, so both see the same machine within
each frame, Diana's mean/p50 is **1.14 and 1.21** across two runs against
Ultralytics' **1.79 and 3.33**, with 5-7 outliers to their 16. Diana has the
lighter tail.

Underneath it sat one real mechanism: Diana's slow frames carry **4,200 page
faults against a normal 31**. `MIMALLOC_PURGE_DELAY=-1` removes them completely
— faults go flat at 15 and halve overall — and it is **not shipped**, because
it buys 0.3 % of latency for +9.6 MiB peak RSS against a footprint gate with
1 MiB of headroom. Real mechanism, refuted lever, and the two are recorded
separately.

### The same structure that loses latency wins throughput

Latency is one question. A server asks a different one: given N images and a
whole machine, how many per second? There the answer inverts, at every tier —
**Diana is 1.5–2.4× ahead:**

| tier | run 1 | run 2 |
|---|---:|---:|
| n | 1.61× | 1.66× |
| s | 1.72× | 2.09× |
| m | 1.66× | 2.13× |
| l | 1.62× | 2.35× |
| x | 1.52× | 2.32× |

Two independent runs, because one was not enough to know which part of the
result was real. **The direction is: ahead at every tier in both runs.** The
magnitude is not — it moves by up to 50% between runs on this box, so the
honest claim is a range, not the 1.6× a single session would have supported.

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
