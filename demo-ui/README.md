# FFai demo — four tabs, four engines, one honest comparison each

- **Listen** — Mercury and whisper.cpp transcribing the **same microphone
  audio**, next to each other, so you can read whether they agree.
- **Speak** — Mercury synthesizing whatever you type, with the phonemes, the
  sentence split, and a determinism check on screen.
- **Read** — both OCR lineages on the same image, and the content class that
  decides which one the pipeline would dispatch to.
- **See** — Argus captioning an image beside PyTorch on the **same checkpoint
  and the same file**, plus a full breakdown of where every millisecond went.

```sh
# 1. build the UI (wasm)
cd demo-ui && dx build --platform web --release && cd ..

# 2. run the server (loads Mercury ASR + the voice, finds whisper.cpp;
#    Argus and its PyTorch reference load in the BACKGROUND, ~40 s)
cargo run --release -p ffai-demo

# 3. open http://127.0.0.1:8787
#    Listen → Start → talk → Stop
#    Speak  → type → Speak
#    Read   → choose or paste an image
#    See    → type a question, choose or paste an image
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

### Speakers are laid out, not prefixed

`/transcribe` returns the Mercury pane a `turns` array — one entry per segment,
each with its speaker and its start and end **in session time** — beside the
flat `text`. The times are absolute because the browser posts a *sliding*
window: a time measured from the buffer's start moves under the same audio
every tick, so nothing measured that way can be laid out against anything else.
`stream_offset_secs` already goes in for the diarizer's window grid; the same
number comes back out.

The pane renders one row per utterance, a colour lane per speaker, the clock
time on the left, and a ribbon above each chunk showing who held which seconds
— drawn against the 10 s window rather than each chunk's own span, so a 1.3 s
exchange draws a short bar and a 4 s one draws a long bar. A repeated speaker
drops its chip so a monologue reads as one block.

`SPEAKER_00:` glued to the front of a string was doing three things wrong: it
threw away the timing the diarizer had just computed, it left every turn in one
paragraph, and it is indistinguishable from someone actually saying it. The
string is still sent, and a pane that gets no `turns` (whisper.cpp, which does
not diarize, or Mercury with speakers off) renders it exactly as before.

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

## The See tab — why it is a diagram and not just two panes

Listen and Read are races: two implementations, identical input, read the
answers. See is a race too — Argus against **PyTorch + `transformers` running
the identical `SmolVLM-256M-Instruct` checkpoint**, through
`corpora/refs/smolvlm_hf_ref.py`, which is the same file the benchmark's
quality gate uses and the place the decode config (greedy / 64 tokens /
float32 / seed 0) is pinned. Pointing the demo at the bench's own reference is
what stops "the reference" meaning two different things in two places.

But the race is the less interesting half. Ask why captioning took four seconds
and the answer everyone reaches for is "the language model". It is almost never
the language model. A measured run on a 104×27 banner:

```
vision        12468 ms   ← 77% of the total, 9 SigLIP passes
generate       2061 ms   ← 32 tokens, the part everyone blames
prefill        1470 ms
preprocess       92 ms
decode           20 ms

tokens: 576 image · 36 text  →  94% of the prompt is the picture
```

A still is cut into tiles — longest edge to 2048, then a grid of 512×512 tiles
plus a global thumbnail — and **every tile is its own vision forward pass**.
The tab draws that grid over your image so the tile count is a thing you can
see rather than a number you are told, then shows per-tile and per-token costs
underneath. The per-tile bars are near-flat on purpose: vision cost is a
function of the tile COUNT, not of how complicated the picture is.

Both arms are warmed before serving, and the model-load time is reported
separately. A latency demo that folds a one-off 15 s weight load into its first
reading is telling you something false about every later one; the first click
says `COLD` if it paid that cost.

**One image is an anecdote.** The measured result on a pinned 50-image corpus
is in `bench/ledger.jsonl`: quality an exact tie with this reference, 49/50
answers byte-identical, footprint 0.71×, speed 2.4× slower. The tab says so
rather than inviting you to generalise from whatever you just pasted.

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
