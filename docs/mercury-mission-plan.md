# Mercury Mission Plan

**Component:** Mercury — FFai's voice component (`ffai-mercury`)
**Tasks:** ASR (speech → text) and TTS (text → speech)
**Status:** Phase 0 stubs registered · this plan takes both to `stable`
**Prime directive:** pure Rust end-to-end, measured against the non-Rust
world standards by `ffai bench` at every milestone. No claim without a
ledger line.

---

## 1. Mission

Ship the first production-grade, pure-Rust voice stack:

1. **ASR** with the full Whisper capability set, plus the WhisperX
   enhancement layer (VAD, word-level timestamps, diarization) switchable
   **per call via flags** — not a separate engine, not a fork.
2. **TTS** that is a fully functional tool on day one: text in, natural
   speech out, voice-selectable, streaming-capable.
3. Every milestone exits through the analyzer: `ffai bench` compares our
   Rust code to the **non-Rust global benchmarks** (openai-whisper and
   WhisperX on PyTorch, faster-whisper on CTranslate2, whisper.cpp; Coqui
   TTS and Kokoro's Python pipeline) on pinned corpora, and the result —
   win or loss — is appended to `bench/ledger.jsonl`.

**Success =** WER parity (within the 5% relative band) with openai-whisper
per model size, at better ×-realtime on the same hardware, in safe Rust —
and a TTS voice a listener rates as natural, at faster-than-realtime
synthesis. Claims traceable to ledger lines, reproducible by anyone from
the public repo.

---

## 2. Design rule: independent functions, composed pipelines

Mercury is a **toolbox of independently callable functions**, each with its
own contract and tests. The CLI composes them; so can any embedder. Nothing
is welded together.

```
ffai-mercury
├── audio/            preprocess: resample, mono, chunk
├── asr/
│   ├── mel.rs        log-mel spectrogram (the Whisper front-end)
│   ├── encoder.rs    Whisper audio encoder forward pass (candle)
│   ├── decoder.rs    token decode loop: greedy + beam, temperature fallback
│   ├── tokenizer.rs  BPE tokenizer + special-token grammar
│   ├── detect.rs     language identification
│   └── engine.rs     WhisperCandle: composes the above → Transcript
├── x/                the WhisperX layer (each stage independent, engine-agnostic)
│   ├── vad.rs        voice activity detection → speech regions
│   ├── align.rs      forced alignment → word-level timestamps
│   └── diarize.rs    speaker embedding + clustering → speaker labels
└── tts/
    ├── normalize.rs  text normalization (numbers, dates, abbreviations)
    ├── phonemize.rs  G2P: graphemes → phonemes
    ├── acoustic.rs   phonemes → acoustic features / audio latents (candle)
    ├── vocoder.rs    latents → waveform
    └── engine.rs     KokoroCandle etc.: composes the above → AudioBuffer
```

Contracts that make the functions independent:

- Every stage consumes/produces `ffai-core` types (`AudioBuffer`,
  `TimedSegment`, `Transcript`) or plain candle tensors — no stage knows its
  neighbors.
- Each stage has **its own oracle test** (§6): mel output vs
  openai-whisper's mel to tolerance, tokenizer round-trip vs tiktoken,
  alignment timestamps vs WhisperX's on the same clip.
- The WhisperX layer operates on *any* `AsrEngine`'s output — it wraps
  transcripts, it does not reach into Whisper internals. When a better ASR
  model replaces Whisper someday, VAD/align/diarize survive unchanged.

## 3. ASR specification

### 3.1 Whisper core (`whisper-candle` engine)

| Capability | Contract |
|---|---|
| Model sizes | tiny → large-v3 (+ turbo), selected via `ffai-models` manifests |
| Transcription | greedy + beam search, temperature fallback on repetition/logprob failure (Whisper paper semantics) |
| Language | auto-detect (`detect.rs`) or forced via `--language` |
| Translate | `--translate` task token → English |
| Timestamps | segment-level from decoder timestamp tokens |
| Long audio | 30 s windows, context carry-over (`condition_on_previous_text`), seek logic |
| Streaming | chunked incremental transcription behind the same API (streaming-first core principle) |
| Quantization | f16 + GGUF-style int8/int4 paths for CPU viability |

### 3.2 The WhisperX layer — flags, not forks

```
ffai asr -i talk.mp4                                   # Whisper core
ffai asr -i talk.mp4 --vad                             # + VAD segmentation
ffai asr -i talk.mp4 --word-timestamps                 # + forced alignment
ffai asr -i talk.mp4 --diarize                         # + speaker labels
ffai asr -i talk.mp4 --word-timestamps --diarize       # full WhisperX mode
```

| Stage | Model class | Independent contract |
|---|---|---|
| `vad.rs` | Silero-VAD-class, ported to candle | `AudioBuffer → Vec<TimedSegment<()>>` (speech regions); cuts hallucination on silence, enables batching windows by speech, not by 30 s |
| `align.rs` | wav2vec2-CTC-class per language | `(AudioBuffer, Transcript) → Transcript` with per-word `TimedSegment`s |
| `diarize.rs` | speaker-embedding + clustering | `(AudioBuffer, Transcript) → Transcript` with `speaker` labels on segments |

`AsrOptions` already carries `word_timestamps` / `diarize` (Phase 0); `vad`
is added alongside. The engine trait does not change.

### 3.3 ASR world-standard references (non-Rust)

Declared in `corpora/references.toml`; versions recorded per ledger line.

| Reference | Stack | Why it's the bar |
|---|---|---|
| openai-whisper | Python/PyTorch | the accuracy definition of each model size |
| WhisperX | Python/PyTorch | the bar for word timestamps + diarization quality |
| faster-whisper | Python + CTranslate2 (C++) | the speed bar most products actually use |
| whisper.cpp | C++ | the native-CPU speed bar |

## 4. TTS specification

### 4.1 Fully functional means

```
ffai tts "Hello from FFai." -o hello.wav                 # just works, default voice
ffai tts --voice af_heart --speed 1.2 -t script.txt -o narration.wav
ffai tts --list-voices
```

- Long-form input: sentence segmentation, prosody-safe chunking, seamless
  concatenation.
- Voice selection from `ffai models`-managed voice packs (licenses surfaced —
  some voices are CC BY-NC and must be flagged at selection time).
- Streaming synthesis: audio starts before the full text is processed.
- Deterministic mode (fixed seed) for testing and caching.

### 4.2 Engine ladder

1. **`kokoro-candle` (M4):** Kokoro-82M implemented natively on candle —
   small, Apache-2.0 weights, proven quality; our reference TTS engine and
   the one we oracle-gate hardest.
2. **`any-tts` (M5):** dependency on the any-tts crate → instantly adds
   Qwen3-TTS / VibeVoice / OmniVoice tiers behind the same `TtsEngine` trait.
3. **`voirs` (later):** the full G2P→acoustic→vocoder framework as an
   alternative lineage.

The independent functions (§2) are OURS regardless of engine: normalize and
phonemize are shared infrastructure every TTS engine uses, each oracle-tested
on its own.

### 4.3 TTS world-standard references (non-Rust)

| Reference | Stack | Bar |
|---|---|---|
| Kokoro (Python pipeline) | Python/PyTorch | same-model quality + speed bar for kokoro-candle: output should be near-identical audio, faster |
| Coqui TTS (XTTS/VITS) | Python/PyTorch | the general open-source TTS quality bar |
| Piper | C++ | the fast-local-TTS speed bar |

### 4.4 Measuring TTS quality without ears

- **Round-trip intelligibility (primary, automated):** synthesize a pinned
  text corpus → transcribe with a FROZEN third-party ASR (whisper.cpp
  pinned version, never our own engine — no self-grading) → WER between
  input text and round-trip transcript, ours vs each reference's audio.
- **Speed:** ×-realtime synthesis + time-to-first-audio, best-of-N.
- **Naturalness (secondary):** UTMOS-class MOS predictor when practical;
  human spot-listens logged in ledger notes — never claimed as measurements.

## 5. Analyzer integration (`ffai-bench`)

Already live for ASR; Mercury work extends it:

- **M1:** adopt Whisper's official text normalizer for WER so our numbers
  are comparable with published ones; LibriSpeech corpus manifests
  (test-clean + test-other excerpts, CC-BY, hash-pinned).
- **M3:** word-timestamp accuracy metric (mean |Δms| vs WhisperX alignment)
  and DER-style diarization scoring; noisy/accented corpus classes.
- **M4:** `ffai bench tts` — round-trip WER + synthesis RTF + time-to-first-
  audio vs the §4.3 references.
- **Footprint gate goes live (M2):** peak RSS measured for us and references;
  "pure Rust, no Python runtime, N× smaller install" becomes a measured
  claim, not a vibe.
- Cadence: `--baseline-only` lands references on the board at milestone
  start; full bench runs at milestone exit; ledger records both.

## 6. Milestones and exit gates

Every milestone exits through all four gates (correctness / quality / speed /
footprint) on the holdout split — a skipped gate blocks exit.

| # | Deliverable | Exit gate (ledger-recorded) |
|---|---|---|
| **M0** ✅ | Baselines: LibriSpeech manifests + `bench --baseline-only` against the installable ASR references | reference WER/RTF on the board; corpus hashes pinned — **see §6.1** |
| **M1** | Whisper bring-up: mel → encoder → decoder → tokenizer on candle, tiny + base, greedy | per-stage oracles pass (mel/tokenizer/logits tolerance vs openai-whisper); WER within band on tiny+base |
| **M2** | Full core: beam + fallback, long-audio seek, all sizes, f16/quant, streaming API | WER band on large-v3; RTF ≥ faster-whisper CPU; footprint gate live and passing |
| **M3** | WhisperX layer: `--vad`, `--word-timestamps`, `--diarize` | word-timestamp Δ vs WhisperX within tolerance; diarization within DER band; flags compose with any engine |
| **M4** | TTS v1: kokoro-candle + normalize/phonemize infra, `ffai tts` fully functional (§4.1) | round-trip WER ≤ Python Kokoro's + RTF faster; deterministic mode byte-stable |
| **M5** | TTS breadth: any-tts engine tier, streaming synthesis, voice-pack licensing UX | same gates per added voice tier; time-to-first-audio < 500 ms on reference hardware |
| **M6** | Mercury `stable`: docs, `mercury` lib examples, claims page generated FROM the ledger | every README claim maps to a ledger line id |

Sequencing note: M4 (TTS v1) can start in parallel with M3 — it shares no
code with the WhisperX layer, only `ffai-core` types. Two tracks, one
analyzer.

### 6.1 M0 result — baselines on the board

**Corpus.** `librispeech-test-clean` v1: 16 utterances from 16 distinct
speakers of LibriSpeech test-clean (CC BY 4.0), 157.2 s total audio,
converted to 16 kHz mono WAV. 11 clips holdout (80.9 s), 5 train. Every clip
SHA-256-pinned; a run aborts if the bytes drift. Regenerate deterministically
with `cargo run -p ffai-bench --example prepare_librispeech`. Manifest:
[`corpora/librispeech-test-clean-v1.toml`](../corpora/librispeech-test-clean-v1.toml).

**Baselines** (ledger `bench-asr-1785084337`, corpus `2117fcaa8ba1`,
best-of-3, all pinned to `beam_size=5`, Windows x86_64 / Intel Raptor Lake,
CPU only):

| Implementation | Stack | WER % | CER % | ×RT warm | ×RT e2e | load s |
|---|---|---:|---:|---:|---:|---:|
| openai-whisper tiny.en | Python/PyTorch | 2.99 | 1.15 | 13.4 | 10.2 | 0.31 |
| faster-whisper tiny.en | Python/CTranslate2 (C++), int8 | 2.99 | 1.15 | **22.1** | 13.5 | 0.80 |
| openai-whisper base.en | Python/PyTorch | **1.72** | **0.77** | 6.5 | 5.7 | 0.53 |
| faster-whisper base.en | Python/CTranslate2 (C++), int8 | 1.97 | 0.82 | 11.3 | 8.8 | 0.94 |

Read the table three ways:

- **The two tiny.en rows agree on WER to the digit (2.99 %).** Same model,
  same decode config, different runtime — identical output is exactly what
  should happen, and it is the strongest available evidence that the harness
  measures the implementation rather than an accident of configuration. This
  agreement did *not* exist before the decode config was pinned.
- **faster-whisper is 1.65–1.74× faster than openai-whisper** at both sizes,
  the direction and rough magnitude its authors claim. The first (unpinned)
  run had it looking *slower*, which is how we found defect 2.
- **int8 quantization costs base.en about 0.25 points of WER** (1.72 → 1.97)
  to buy 1.74× throughput. That is the accuracy/speed frontier Mercury has to
  land on, not a single point to beat.

**These are the numbers `whisper-candle` must answer to.** The M1/M2 targets
follow directly: WER within the 5 % relative band of the openai-whisper row
at the same model size, at ≥ the faster-whisper row's warm throughput.

**What M0 established beyond the numbers** — three methodology defects were
found and fixed while standing this up, each of which would have produced
indefensible claims later:

1. **Per-clip invocation put interpreter startup and model load inside every
   timed run.** Fixed by making batch invocation the reference contract, and
   by reporting warm and end-to-end throughput as two separate numbers that
   are always recorded together.
2. **Decode configuration was unpinned.** openai-whisper defaults to greedy,
   faster-whisper to beam_size=5 — the first run had faster-whisper looking
   *slower*, which is backwards. All references are now pinned to
   `beam_size=5` and the exact argv is stored in the ledger line.
3. **Raw WER scored formatting as error.** `MISTER QUILTER` vs `Mr. Quilter`
   and `twenty three` vs `23` are not recognition failures. A port of
   Whisper's `EnglishTextNormalizer` now scores both sides identically, with
   its parity gaps documented rather than glossed.

**Reference coverage.** The plan named four reference families; two are
installed and measured (openai-whisper, faster-whisper — at two model sizes
each). The gap, recorded honestly rather than quietly dropped:

| Reference | Status | Why |
|---|---|---|
| openai-whisper | ✅ measured | tiny.en, base.en |
| faster-whisper | ✅ measured | tiny.en, base.en, int8 |
| whisper.cpp | ⬜ not yet | no C++ toolchain on the bench machine (`cl`/`gcc` absent); prebuilt release binary is the intended route. **This is the most important missing baseline** — it is the native, no-Python comparison our "pure Rust is competitive" claim will be judged against. Required before M2 exit. |
| WhisperX | ⬜ deferred to M3 | its distinctive output is word timestamps and diarization, which M3 measures; it adds nothing to a plain WER/RTF baseline |

## 7. Engineering discipline (inherited, non-negotiable)

- **One brick per commit;** revert if the bench says it's not better.
- **Profile before optimizing** (the codec-optimize lesson): bring-up is
  scalar-clear candle first; SIMD/kernel work only where the profiler points,
  each behind the scalar-twin-as-oracle discipline.
- **Stage oracles before end-to-end oracles:** a WER regression must be
  attributable to one stage in minutes (mel? tokenizer? decode loop?).
- **No self-grading:** quality judgments always come from a frozen external
  implementation or ground truth, never from another FFai engine.
- **Holdout discipline:** anything used for tuning lives in `train` clips;
  exit gates run on `holdout` only.
- **Weight licensing surfaced** at voice/model selection; CC BY-NC voices
  clearly marked in `--list-voices`.

## 8. Risks

| Risk | Mitigation |
|---|---|
| candle Whisper exists as an example, tempting a copy-paste bring-up that can't hit our per-stage oracle bar | use it as a cross-check, not a source; our stage decomposition (§2) is the deliverable |
| WER "parity" disputes from normalizer differences | adopt Whisper's official normalizer verbatim (M1) before publishing any WER |
| Diarization quality is a research-grade problem | scope M3 to "within DER band of WhisperX's pipeline", not SOTA; it's a flag, shippable later than M2 without blocking |
| Python reference environments drift | reference versions recorded per ledger line; baseline re-runs at every milestone start |
| TTS naturalness is subjective | claim only what's measured (round-trip WER, RTF, footprint); MOS-predictor numbers labeled as predictions |
| GPU features drag C/C++ back in | CPU-first targets; CUDA/Metal stay behind features; oxicuda watchlist reviewed at each milestone (ROADMAP § Watchlist) |
