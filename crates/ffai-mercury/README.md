# ffai-mercury

**Speech recognition in pure Rust, with the full WhisperX layer** — voice activity detection, word-level timestamps, and speaker diarization. No Python, no C/C++ by default, no HuggingFace token, no gated weights.

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

## Status: `experimental`, honestly

Measured against whisper.cpp on two hash-pinned 134-clip LibriSpeech holdouts, matched greedy decoding, CPU only, tiny.en: **6.79 % / 16.43 % WER** against their 7.58 % / 16.82 %, ahead on CER on both, at 1.01–1.09× throughput and lower memory. Every number traces to a line in [`bench/ledger.jsonl`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl).

Not yet `stable`: word-level timestamp error is gated at utterance granularity rather than milliseconds, and the diarization corpus has no speaker overlap — it gates regression, not readiness. TTS is a registered stub.

## License

MIT OR Apache-2.0. Model weights carry their own licenses, surfaced at selection time.
