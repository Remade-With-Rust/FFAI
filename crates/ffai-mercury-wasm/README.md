# ffai-mercury-wasm

**Mercury in the browser** — pure-Rust Whisper speech recognition compiled to
WebAssembly. The mel front end, the encoder, the decoder and the tokenizer all
run inside the wasm module. There is no whisper.cpp, no ONNX Runtime Web, and
no JavaScript inference engine underneath: the same Rust that runs natively is
the Rust that runs here.

**Status: `0.1.0` — experimental.** It runs, and it agrees with the native
build byte for byte on the clip below. No corpus gate has been run on a wasm
build; do not quote `ffai-mercury`'s accuracy record for this target without
re-measuring it here.

Executed on `wasm32-unknown-unknown` (Node VM, no emscripten). Requires a
browser with WebAssembly — no threads, no SIMD, and no C toolchain anywhere in
the build.

## What it is

- **A `cdylib` wrapper over [`ffai-mercury`](https://crates.io/crates/ffai-mercury)**,
  exposing Whisper through `wasm-bindgen` as three calls.
- **Weights are yours** — `Recognizer.whisper` takes `model.safetensors`,
  `config.json` and `tokenizer.json` as bytes. Nothing is bundled and nothing
  is fetched for you. There is no filesystem to read and no mmap to take.
- **Gated against native.** `WhisperCandle::from_bytes` funnels into the same
  assembler the manifest constructor uses, so a browser and a server build the
  same model from the same tensors. `examples/from_bytes_smoke.rs` in
  `ffai-mercury` runs that constructor natively so it is debuggable.
- **16 kHz mono, and it will not resample for you** — see below.

## Performance

`whisper-tiny.en`, a 10.4 s LibriSpeech clip, Node on x86-64 Windows,
single-threaded. Absolute times here run roughly 3× a browser's, so read the
ratios.

| | wasm | native, same bytes and same audio |
|---|---:|---|
| weights load | 887 ms | — |
| decode | 1.4 s (**7.66× realtime**) | — |
| peak linear memory | 649 MiB | — |
| transcript | *"He hoped there would be stew for dinner…"* | **byte-identical** |
| segments | 2 | 2 |

The one word that differs from LibriSpeech's ground truth is `flower` for
`flour` — a homophone. Native produces the same slip, which is the point of
running both: it is `tiny.en`, not the port.

## Usage

```js
import init, { Recognizer } from './ffai_mercury_wasm.js';
await init();

const r = Recognizer.whisper(weights, configJson, tokenizerJson, 'whisper-tiny-en');
console.log(r.text(mono16k, 16000));
```

| | |
|---|---|
| `Recognizer.whisper(weights, config, tokenizer, name)` | load; throws a descriptive error on a bad checkpoint |
| `r.text(samples, sampleRate)` | the whole utterance as one string |
| `r.transcribe(samples, sampleRate)` | `Segment[]` — `{ text, start, end }` |
| `r.detectLanguage(samples, sampleRate)` | the detected tag, or `undefined` |
| `allocator()` | which allocator this module was built with |
| `linearMemoryBytes()` | current linear memory — the only honest memory instrument here |

Build it with:

```
cargo build --release --target wasm32-unknown-unknown -p ffai-mercury-wasm
wasm-bindgen target/wasm32-unknown-unknown/release/ffai_mercury_wasm.wasm \
    --out-dir crates/ffai-mercury-wasm/pkg --target web
```

`wasm-bindgen` the CLI and the crate **must be the same version** — the crate
is pinned with `=` for that reason. `ffai-mercury` is depended on with
`default-features = false`, which is not optional: `fetch` pulls `hf-hub`,
whose 1.0 release calls `reqwest::blocking`, a module that does not exist on
wasm32.

## 16 kHz mono, and why this crate will not resample for you

Whisper is a 16 kHz mono model, and the engine refuses anything else rather
than resampling silently — a transcript produced from audio played at the wrong
speed is wrong in a way that reads as fluent. The browser already has a
correct, fast resampler that a wasm module has no business duplicating:

```js
const off = new OfflineAudioContext(1, Math.ceil(buf.duration * 16000), 16000);
const src = off.createBufferSource();
src.buffer = buf;
src.connect(off.destination);
src.start();
const mono16k = (await off.startRendering()).getChannelData(0);
```

## Sizing

These are 32-bit address spaces, and wasm linear memory **only ever grows** —
nothing you free is returned to the browser, so peak is the whole budget.
`whisper-tiny.en` is ~145 MB of f32 weights and comfortable; `base` is larger
but workable; the large checkpoints are not a browser proposition. Loading two
pipelines into one module instance is not one either — `ffai-carmenta-wasm`
measured ~118 MB of weights in a single linear memory failing to complete.

Word timestamps and diarization are **off** on this target. Both load a second
model on first use, which a space already holding Whisper cannot afford; they
stay off rather than failing halfway through a transcript.

## What is different on this target

**Serial.** `wasm32-unknown-unknown` has no threads, so `ffai_mercury::par`
supplies the identical serial iterators and `rayon` is not a dependency of this
build at all. Mercury reaches rayon on the hot path — `flash_attn` over
attention heads, `text_decoder` over logit chunks, `vocab_int8` over vocabulary
blocks — so without the shim a browser build panics inside the first forward
pass rather than at load. Serial costs less than it sounds: candle's
`default_num_threads()` calls `num_cpus::get_physical()`, whose wasm32 branch
returns a literal `1`, so every matmul beneath those sites is already
`Parallelism::None` here.

**No clock, and one decision had to stop depending on one.** `Instant::now()`
panics on this target, and `adaptive::matmul_dtype` was *timing* two candidate
matmuls to choose between f32 and f16. On wasm it now answers `F32` directly —
the same verdict its own "cannot be timed" fallback documents, reached without
running probe matmuls to learn nothing. Everything else that timed now reports
zero rather than lying about a number the target cannot take.

**No SIMD in candle, and that is an upstream defect rather than a limit.**
`candle-core` 0.11 does not compile with `-C target-feature=+simd128`, so
`gemm`'s wasm SIMD kernels and LLVM's auto-vectorisation are both unavailable.
That is the largest outstanding lever for every FFai wasm crate, it is a few
lines in someone else's repository, and it is tracked in
`docs/plans/carmenta-wasm-plan.md` §2.

## The Remade With Rust ecosystem

<!-- ORG BOILERPLATE — keep identical across repos -->

**Remade With Rust** is an initiative by **[Mata Network](https://www.mata.network/)**
to rebuild essential C and C++ tools in Rust — for the memory safety, the
predictable performance, and the freedom of a permissive license. Each project
is a reimplementation, not a fork: same wire protocols and file formats, new
code you can actually depend on. No copyleft. No surprises.

| Project | What it is |
|---|---|
| 🎬 **[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)** | **Our FFmpeg alternative.** Drop-in `ffmpeg` and `ffprobe` binaries — demux → decode → filter → encode → mux, rebuilt as composable Rust crates with **zero GPL/LGPL**. Apache-2.0. `rusty_h264` is its H.264 codec. |
| 🧠 **[FFAI](https://github.com/Remade-With-Rust/FFAI)** | **Our sister project: media *for* AI.** "The AI media toolkit, remade with rust." Embedded ASR + TTS (**Mercury**), OCR (**Carmenta**) and vision-language captioning (**Argus**) behind an ffmpeg-style, swap-by-name architecture — no Python, no CUDA. MIT OR Apache-2.0. |
| 🌐 **[Mata Network](https://www.mata.network/)** | **The home page.** *"Stop sacrificing your privacy for convenience."* Sovereign, self-hostable privacy infrastructure — wallet & identity, password manager, contact manager, and a browser extension that stops information leaking as you browse. Remade With Rust is its open-source arm. |

→ All projects: **[github.com/Remade-With-Rust](https://github.com/Remade-With-Rust)**

<!-- /ORG BOILERPLATE -->

## License

MIT OR Apache-2.0, at your option. **Model weights are not covered by it** —
each carries its own license, surfaced at selection time and in
`ffai models`.
