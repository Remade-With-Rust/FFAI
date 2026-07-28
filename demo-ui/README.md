# FFai side-by-side demo

Mercury and whisper.cpp transcribing the **same microphone audio**, next to
each other, so you can read whether they agree.

```sh
# 1. build the UI (wasm)
cd demo-ui && dx build --platform web --release && cd ..

# 2. run the server (loads Mercury, finds whisper.cpp)
cargo run --release -p ffai-demo

# 3. open http://127.0.0.1:8787  → Start → talk → Stop
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

## What the demo is for

Reading the **text**. Both run tiny.en at matched greedy settings, so where the
two panes differ, that difference is the implementations — which is the only
thing this demo is evidence of.

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
