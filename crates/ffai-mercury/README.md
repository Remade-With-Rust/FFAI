# ffai-mercury

**Speech recognition and speech synthesis in pure Rust.** ASR with the full WhisperX layer — voice activity detection, word-level timestamps, speaker diarization — and TTS running piper's own voices on candle. No Python, no C/C++ by default, no HuggingFace token, no gated weights, and nothing GPL.

Mercury is [FFai](https://github.com/Remade-With-Rust/FFAI)'s voice component. Standalone landing page: [Remade-With-Rust/mercury](https://github.com/Remade-With-Rust/mercury).

```toml
[dependencies]
ffai-mercury = "0.4"
ffai-core = "0.4"
ffai-media = "0.4"
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

`piper-candle` is the full VITS stack on candle — text encoder with relative-position attention, stochastic duration predictor with spline flows, residual coupling flow, HiFi-GAN — running the **same voice files** [piper](https://github.com/OHF-Voice/piper1-gpl) runs, converted locally from the voice's own `.onnx` (see `models/piper-vits-lessac-medium.toml`).

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

- **Oracle-exact against piper's own runtime** at zero noise: text encoder to 4e-6, per-phoneme durations **integer-exact**, end-to-end waveform to 3e-5.
- **The phonemizer passed its substitution gate** — our phonemes fed through piper's own runtime score round-trip WER inside the 5 % band of espeak's.
- **Quality: parity.** Round-trip WER through a frozen whisper.cpp judge (never our own ASR — no self-grading): **5.91 %, byte-stable on every run**, against piper's own 4.8–6.5 % across runs. Piper samples noise in-graph and cannot repeat a number, so it is scored as the mean of independent draws with the range recorded. Parity through this instrument — not superiority; the instrument cannot support more.
- **Determinism**: verified at both library and file-hash level. Piper structurally cannot offer it.
- **Footprint and load ahead**: 172–208 MiB steady vs piper's 217–240; load 0.26–0.35 s vs ~1.8–2.6 s.
- **Speed behind, closing, not claimable**: 3.2× → 19–20× realtime warm across five profiled campaigns; piper measures 25–32× here. Function-by-function against piper's runtime, Mercury's upsamplers and duration predictor are ~1.9× **faster** and the text encoder is at parity — the gap lives in two convolution kernels with measured targets. The speed gate reads FAIL on every fair ledger line.

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

**TTS** — three of four gates pass against piper1-gpl on a pinned 200-sentence corpus (correctness, quality, footprint); speed does not, and is reported as failing.

Every number traces to a line in [`bench/ledger.jsonl`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl). Full campaign histories, every reverted experiment included: [ASR](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/finished/mercury-mission-plan.md) · [TTS](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/mercury-tts-mission.md).

Not yet `stable`: ASR word-timestamp error is gated at utterance granularity rather than milliseconds and the diarization corpus has no speaker overlap — those gate regression, not readiness. TTS is en-US, single-voice, and behind on synthesis speed. `any-tts` and `voirs` remain registered stubs.

## License

MIT OR Apache-2.0. Model weights carry their own licenses, surfaced at selection time.
