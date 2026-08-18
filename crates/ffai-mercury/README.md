# ffai-mercury

**Speech recognition (ASR) and text-to-speech (TTS) in pure Rust.** Whisper/WhisperX-class recognition — voice activity detection, word-level timestamps, speaker diarization — and VITS/Piper-class synthesis running piper's own voices on candle. No Python runtime, no C/C++ by default, no HuggingFace token, no gated weights, and nothing GPL.

Mercury is [FFai](https://github.com/Remade-With-Rust/FFAI)'s voice component. Standalone landing page: [Remade-With-Rust/mercury](https://github.com/Remade-With-Rust/mercury).

```toml
[dependencies]
ffai-mercury = "0.7"
ffai-core = "0.6"
ffai-media = "0.6"
```

## Transcribe

```rust
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

let engine = WhisperCandle::new();                  // whisper-tiny-en
let audio  = ffai_media::load_audio("talk.wav")?;   // 16 kHz mono
let transcript = engine.transcribe(&audio, &AsrOptions::default())?;

for seg in &transcript.segments {
    println!("[{:.2}-{:.2}] {}", seg.start, seg.end, seg.value);
}
```

Weights download once, into a local cache, from hash-verified manifests.

### Trading speed for accuracy

`WhisperCandle::new()` is `tiny.en` — the fast default. Accuracy is a dial:

```rust
use ffai_mercury::asr::text_decoder::Precision;

// tiny 6.39 % WER @ 19.9x · base 5.16 @ 8.3 · small 3.05 @ 4.2 (test-clean)
let engine = WhisperCandle::model("whisper-small-en", Precision::F32);
```

Sizes run `whisper-{tiny,base,small,medium}-en`; `Precision::Q8_0` halves the
decoder's memory on tiny and base. Above `medium.en` Whisper is multilingual
only, which needs language detection Mercury does not yet have — so that is
where the ladder honestly stops.

Beam search is the other dial, and it is what the reference implementations
run by default. Greedy is Mercury's default because every benchmark in this
repo pins the references to greedy so the comparison measures implementations
rather than decoding strategies:

```rust
// pooled across both holdouts: z = +2.43 on WER, ~5x the cost
let cfg = DecodeConfig { beam_size: 5, ..Default::default() };
```

## The WhisperX layer

| Option | What it adds | Model fetched |
|---|---|---|
| `vad` *(on by default)* | speech segmentation before transcription | none — energy VAD |
| `word_timestamps` | per-word times by CTC forced alignment | wav2vec2-base-960h (Apache-2.0) |
| `diarize` | speaker turns, `SPEAKER_00`… | ECAPA-TDNN (Apache-2.0) |
| `persist_speakers` | keeps those labels stable **across calls** | — |

```rust
let opts = AsrOptions {
    word_timestamps: true,
    diarize: true,
    max_speakers: Some(2),      // when you know; see the caution below
    ..Default::default()
};
let t = engine.transcribe(&audio, &opts)?;

for w in t.words.iter().flatten() {
    println!("{:6.2}-{:6.2} {}", w.start, w.end, w.value);
}
for turn in t.speakers.iter().flatten() {
    println!("{:6.2}-{:6.2} {}", turn.start, turn.end, turn.value);
}
```

The extra stages are **lazy**: without the flag, their models are not fetched, not read, and not resident.

## Streaming

Diarization labels are, by convention, arbitrary names for clusters *within one call* — `SPEAKER_00` in two separate calls need not be the same person. That is fine for a file and useless for a live stream, where the same voice would be renamed every chunk.

`persist_speakers` keeps a speaker registry between calls:

```rust
let opts = AsrOptions { diarize: true, persist_speakers: true, ..Default::default() };

for chunk in microphone_chunks {
    let t = engine.transcribe(&chunk, &opts)?;   // SPEAKER_00 means the same
    // ...                                        // person in every chunk
}
engine.reset_speakers();                          // a new recording is new people
```

Measured on conversations fed as 8 s chunks, scoring DER over the whole concatenated timeline: **53.58 % without it, 5.68 % with it.** Matching is deliberately stricter than in-call clustering, because a registry merge is permanent — two people who share a centroid stay merged for the session.

Tell the engine where each buffer sits in the stream and streaming diarization gets **3.16× cheaper**:

```rust
let opts = AsrOptions {
    diarize: true,
    persist_speakers: true,
    stream_offset_secs: elapsed,   // where THIS buffer starts in the session
    ..Default::default()
};
```

Speaker embedding is ~100 % of diarization's cost (~172 ms per 1.5 s window). A sliding window re-sends mostly the same audio each tick, and without the offset the window grid is anchored to the *buffer* — so identical audio gets re-cut at new offsets and every embedding is recomputed. With it, windows land on an absolute grid and a content-keyed cache actually hits: median 1258 → 720 ms across paired ticks, 10/10, at **4.20 % DER against 4.21 %** region-anchored. Leave it at `0.0` for whole-file use and nothing changes.

*Correction:* 0.6.0 described this as DER-neutral on a stale measurement. Its first version snapped window chains forward and skipped 0.75 s of each region's leading audio, costing 4.21 % → 9.60 % DER; 0.6.1 emits the region-start window before following the grid, restoring coverage at a smaller — and real — 1.75× speedup.

## Things measurement taught us, which you may want

**Segmentation is on by default for speed, not quality.** 2.2–4.2× on audio with trailing silence, byte-identical transcript, and an empty result on silence with no encoder pass at all. It also moves corpus WER — and that is *not* a quality win (38 improved / 38 worsened across 400 clips, sign test z = 0.00). Set `vad: false` for the unsegmented fixed-30 s grid.

**`max_speakers` is not the safe option.** With the clustering threshold tuned, blind clustering scores **4.21 % DER** against **5.00 %** with the true count supplied. Forcing a count forces a merge, and a bad merge attributes one speaker's words to another. Supply it when it is certain, not as insurance.

**Silence produces nothing.** Whisper hallucinates fluent text on silence (`you`, `Thank you.`); Mercury reads `P(<|nospeech|>)` and drops the window. Gated at 8/8 on a silence corpus.

**Non-speech events are annotated** (`[Laughs]`, `(coughs)`) as whisper.cpp does rather than suppressed as openai-whisper does. That costs 0.22 pp WER on clean read speech and is one flag away: `DecodeConfig::suppress_non_speech`.

## Output formats

`Transcript` renders to plain text, SRT, WebVTT (with inline `<mm:ss.mmm>` word tags when word timestamps are on), and JSON carrying words and speaker turns.

## Synthesize

```rust
use ffai_core::engine::{TtsEngine, TtsOptions};
use ffai_mercury::tts::PiperCandle;

let engine = PiperCandle::new();
let audio  = engine.synthesize("The birch canoe slid on the smooth planks.",
                               &TtsOptions::default())?;
ffai_media::save_wav("out.wav".as_ref(), &audio)?;
```

`piper-candle` is the full VITS stack on candle — text encoder with relative-position attention, stochastic duration predictor with spline flows, residual coupling flow, HiFi-GAN — running the **same voice files** [piper](https://github.com/OHF-Voice/piper1-gpl) runs. The `.onnx` and its config are fetched from the (public, ungated) `rhasspy/piper-voices` repo and read **directly by a pure-Rust ONNX reader**: no conversion step, no Python, no ONNX runtime. The first `synthesize` call fetches them; `ffai models --fetch piper-vits-lessac-medium` does it ahead of time.

| Option | Effect |
|---|---|
| `speed` | playback rate; 1.0 = the voice's own timing |
| `noise_scale` / `noise_w` | acoustic and duration variation; `0.0` = fully deterministic audio |
| `seed` | noise seed — same text + same seed = **byte-identical WAV** |
| `sentence_silence_s` | gap between sentences of long-form input |

Long-form input is segmented into sentences, synthesized per sentence, and joined — `ffai tts` on a paragraph just works.

### The phonemizer is ours, and that is a licensing decision

piper is GPL-3.0 because it embeds espeak-ng. Mercury's G2P is a clean-room pure-Rust implementation over CMUdict (BSD-2-Clause) that emits espeak-compatible IPA; espeak-ng participates **only as an out-of-process test oracle** over pinned corpora, and nothing GPL is linked, vendored, or shipped. The honest cost, stated rather than buried: **en-US only** for now, where piper covers 40+ languages.

### What is measured

Every number here comes from `ffai bench tts` on a hash-pinned 200-sentence
corpus (134-clip holdout), scored by a **frozen third-party judge** —
whisper.cpp transcribes the synthesized audio and WER is computed against the
input text. Never our own ASR; no self-grading.

- **Oracle-exact against piper's own runtime** at zero noise: text encoder to
  4e-6, per-phoneme durations **integer-exact**, end-to-end waveform to 3e-5.
  Every stage is pinned separately, so an error in one cannot hide in the next.
- **The Rust ONNX reader is byte-identical to the Python converter it
  replaced** — 350 tensors, 15.65 M floats, 132 convolution geometries and the
  audio itself. The Python script stays in the repo as the oracle that gate is
  measured against, not as a step anyone runs.
- **Quality: parity.** 5.49 % round-trip WER against piper's 5.27 % on the same
  holdout, same judge, same run.
- **Speed: ahead, on a reference that will not hold still.** Recent lines read
  19.3–21.2 ×realtime warm against piper's 11.2–23.0 ×. Piper spans **4.5×–29×
  across the ledger's TTS lines** on this machine, so single-run ratios are not
  a claim. The number that survives is measured simultaneously on both engines:
  **1.58× faster wall-clock while using 5 % less total CPU.**
- **Load: 0.64 s against 6.69 s** — the one large difference that barely varies.
- **Footprint: parity**, 214 MiB steady against 217.
- **Determinism**: same text + same seed = byte-identical WAV, verified at
  library and file-hash level. Piper structurally cannot offer this — it samples
  noise inside its ONNX graph with no seed control.

**Both engines are scored across draws, because one of them has to be.** Piper
cannot repeat a run, so its WER is the mean of independent draws with the range
recorded. Ours is seeded and byte-stable, so it would otherwise report a single
fixed draw forever — and our own seed-to-seed spread is **1.11 pp**
(4.99–6.10 %), several times any recent engine change. Scoring both the same way
is the only comparison that means anything; the shipped default seed is still
reported beside the distribution it came from.

Full campaign history, every reverted experiment and withdrawn claim included:
[the TTS mission](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/mercury-tts-mission.md)
and [the six-whys descent](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/whys/tts-speed-gap.md).

## Status: `experimental`, honestly

**ASR — all four gates PASS on both holdouts.** Measured against whisper.cpp
on two hash-pinned 134-clip LibriSpeech holdouts, matched greedy decoding,
CPU only, tiny.en: **7.27 % / 16.89 % WER** against their 7.58 % / 16.82 %,
at **27.8× / 33.3× realtime** against their 25.9× / 19.6×, on 0.84–0.92× the
steady memory. Speed had failed every previous ledger line; **adaptive
encoder context** closed it — each window is encoded at a context sized to
the audio present rather than always 30 s, with guards that escalate a
suspect decode back to the full context. Function-by-function, Mercury is
now ahead of whisper.cpp on **every** stage: encode ~2.0×, decode 1.1–1.2×,
mel 1.4×, sampling 1.7–2.0×.

**Accuracy is a dial, not a fixed point.** tiny.en's WER is Whisper's, not
ours — so the lever is the model size, and `small.en` more than halves it:
6.39 % → 5.16 % → **3.05 %** for tiny → base → small on test-clean, at
19.9 → 8.3 → 4.2 ×realtime. At **matched size**, the comparison that prices
the implementation rather than the weights, Mercury leads on both axes:
**3.05 % WER / 0.88 % CER at 4.2 ×RT** against whisper.cpp small.en's
3.38 % / 1.16 % at 3.7 ×RT (16 clips better, 8 worse, 176 tied — z = +1.63,
under our |z| > 2 bar, so "ahead, not yet significant").

**Beam search** landed too — `beam_size: 5`, what every reference runs by
default. Pooled across both holdouts it is a significant improvement over
greedy (WER 44 improved / 24 worsened, **z = +2.43**; CER z = +2.32), worth
0.6–0.75 pp, at roughly 5× the cost. Greedy remains the default.

**TTS** — measured against piper1-gpl on a pinned 200-sentence corpus
(134-clip holdout): correctness 134/134, quality **5.49 % vs 5.27 %**
round-trip WER, footprint 214 vs 217 MiB, and 19.3–21.2 ×realtime against
piper's 11.2–23.0 ×. The gates flip run to run — **on the reference's variance,
not ours**: piper's throughput spans 4.5×–29× and its WER 4.8–6.5 % across the
ledger, both wider than any difference between the two engines. Read the
distributions, not a line.

Every number traces to a line in [`bench/ledger.jsonl`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl). Full campaign histories, every reverted experiment included: [ASR](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/finished/mercury-mission-plan.md) · [TTS](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/mercury-tts-mission.md).

Not yet `stable`: ASR word-timestamp error is gated at utterance granularity rather than milliseconds and the diarization corpus has no speaker overlap — those gate regression, not readiness. TTS is **en-US and single-voice**, against piper's 40+ languages. `any-tts` and `voirs` remain registered stubs.

## License

MIT OR Apache-2.0. Model weights carry their own licenses, surfaced at selection time.

---

## Security

- **Threat model**: [docs/threat-model.md](docs/threat-model.md) — trust boundaries, STRIDE, and the highest-value attack path (the model cache is trusted and this crate does not verify it).
- **What personal data this processes**: [docs/data-inventory.md](docs/data-inventory.md) — including the fact that diarization derives **voiceprints**, which are GDPR Art 9 special-category biometric data.
- **`unsafe` inventory**: [UNSAFE.md](UNSAFE.md) — all 28 sites, classified, with the invariant each one rests on.
- **Reporting a vulnerability**: [SECURITY.md](../../SECURITY.md) at the repo root.

<!-- HARDENING-TABLE:BEGIN generated by use-protection-please — edit docs/plans/use-protection-please.md, not this block -->
## Hardening status

**Tier** critical-path · **Audited** 2026-08-15 (survey) · **v1.0.0 gates** 5/16 · [Full checklist](docs/plans/use-protection-please.md)

`████████░░░░░░░░░░░░` **40%** &nbsp;·&nbsp; 17 Completed · 0 Scheduled · 25 Incomplete · 13 N/A

| Phase | ✅ Completed | 🗓 Scheduled | ⬜ Incomplete | · N/A |
|---|--:|--:|--:|--:|
| 0 — Threat modeling | 2 | 0 | 0 | 0 |
| 1 — Toolchain | 4 | 0 | 0 | 0 |
| 2 — Supply chain | 5 | 0 | 3 | 0 |
| 3 — Code level | 0 | 0 | 7 | 0 |
| 4 — Static analysis | 0 | 0 | 1 | 0 |
| 5 — Dynamic analysis | 0 | 0 | 3 | 0 |
| 6 — Fuzzing and properties | 0 | 0 | 4 | 0 |
| 7 — Formal verification | 0 | 0 | 1 | 0 |
| 8 — Build and binary | 0 | 0 | 0 | 2 |
| 9 — Runtime privilege | 0 | 0 | 0 | 1 |
| 10 — Cryptography | 0 | 0 | 0 | 3 |
| 11 — CI/CD, release, and operations | 2 | 0 | 3 | 0 |
| 12 — Compliance controls | 4 | 0 | 3 | 7 |
| **Total** | **17** | **0** | **25** | **13** |

**Compliance** — *technical controls evidenced by this audit. NOT a certification, an attestation, or an auditor's opinion.*

| Framework | Technical controls met | |
|---|--:|:--|
| GDPR / PII | 5 / 8 | ⬜ |

**Architect** — [Nick Overlock](https://www.linkedin.com/in/nick-overlock-593235b9/)
<!-- HARDENING-TABLE:END -->
