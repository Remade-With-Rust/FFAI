# FFai demo — Mercury, both directions

Two tabs:

- **Listen** — Mercury and whisper.cpp transcribing the **same microphone
  audio**, next to each other, so you can read whether they agree.
- **Speak** — Mercury synthesizing whatever you type, with the phonemes, the
  sentence split, and a determinism check on screen.

```sh
# 1. build the UI (wasm)
cd demo-ui && dx build --platform web --release && cd ..

# 2. run the server (loads Mercury ASR + the voice, finds whisper.cpp)
cargo run --release -p ffai-demo

# 3. open http://127.0.0.1:8787
#    Listen → Start → talk → Stop
#    Speak  → type → Speak
```

## How it fits together

```
browser  ──getUserMedia → PCM → resample 16 kHz → WAV──▶  POST /transcribe
                                                              │
                                                    one temp .wav file
                                                        ╱          ╲
                                          Mercury (in-process)   whisper-cli
                                                        ╲          ╱
   two transcripts + timings  ◀────────────────────  JSON response
```

Both engines are handed **the same file**, not the same bytes decoded twice.
That removes the "did they actually get identical input" question, which is
the failure mode this project has been bitten by before — a reference invoked
with `-nt` was quietly doing 23 % less work for months (mission plan §6.17).

## The millisecond numbers are not a speed comparison

Mercury runs **warm, in-process**. whisper.cpp is a **subprocess that reloads
its model on every chunk**, so it is charged model load each time. Putting
startup inside a timed run is exactly the defect §6.1 fixed at the benchmark
level, and it is reproduced here only because a demo that keeps a warm
whisper-server alive is a different piece of software.

For real throughput use the analyzer, which loads each implementation once and
runs best-of-N over a hash-pinned corpus:

```sh
ffai bench asr --corpus corpora/librispeech-test-clean-v2.toml
```

## What the Listen tab is for

Reading the **text**. Both run tiny.en at matched greedy settings, so where the
two panes differ, that difference is the implementations — which is the only
thing this demo is evidence of.

## The Speak tab

```
browser  ──text + knobs as JSON──▶  POST /synthesize
                                        │
                    phonemize (our G2P) → sentence split → VITS on candle
                                        │
   WAV as a data URL, plus phonemes, timings, sha256  ◀── JSON response
```

Three things are on screen that a WER table cannot show:

**The phonemes.** Pronunciation bugs are invisible in a waveform and obvious
in IPA. What is displayed comes from `PiperCandle::phonemes`, which runs the
same chunker and phonemizer `synthesize` does — so it is what the model
received, not a re-derivation that could drift from it.

**The sentence split.** Long-form text is synthesized per sentence and joined
with a silence gap, so where the cuts land is a real decision, shown.

**Determinism, checkable.** *Speak twice · prove determinism* synthesizes the
same input twice and compares a SHA-256 of the **samples** (not the WAV — a
header carries a length that would make two renderings match for a trivial
reason). Identical hashes are the claim piper structurally cannot make: it
samples noise inside its ONNX graph with no seed control, so it cannot
reproduce its own output. Set `noise_scale` and `noise_w` to 0 for audio that
is deterministic by construction rather than by seed.

The ×realtime figure is **one warm call on a machine you are also using**. It
collapses under load — measured 20× on an idle box and under 2× with the cores
saturated — so read it as a liveness check. For the measured comparison:

```sh
ffai bench tts --corpus corpora/harvard-sentences-v1.toml
```

That gate currently reads **FAIL**: Mercury synthesizes at 19–20× realtime
against piper's 25–32× on a quiet machine. The demo does not hide it and
neither does the tab.

## Notes

- No storage. The temp WAV is deleted before the response returns.
- Chunks are 5 s. Whisper is trained on 30 s windows and gets *better* with
  more context, so short chunks cost both engines accuracy equally — another
  reason to read this for agreement, not for quality.
- If whisper.cpp is missing, Mercury still runs and the other pane says so.
  Install it per [docs/benchmarking.md](../docs/benchmarking.md).
- The UI is excluded from the cargo workspace: it is a wasm app with its own
  lockfile, and `cargo build --workspace` would otherwise try to build it for
  the host.
