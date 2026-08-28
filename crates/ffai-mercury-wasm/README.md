# ffai-mercury-wasm

**Mercury in the browser** — pure-Rust Whisper speech recognition compiled to
WebAssembly. The mel front end, the encoder, the decoder and the tokenizer all
run inside the wasm module. There is no whisper.cpp, no ONNX Runtime Web, and
no JavaScript inference engine underneath: the same Rust that runs natively is
the Rust that runs here.

```js
import init, { Recognizer } from './ffai_mercury_wasm.js';
await init();

const r = Recognizer.whisper(weights, configJson, tokenizerJson, 'whisper-tiny-en');
console.log(r.text(mono16k, 16000));
```

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

## Weights are yours, and are never bundled

There is no filesystem to read and no mmap to take, so `Recognizer.whisper`
takes the three artefacts as bytes — `model.safetensors`, the text of
`config.json`, and the bytes of `tokenizer.json`. These are the same three
files the manifest resolves natively, and `WhisperCandle::from_bytes` funnels
into the same assembler the manifest constructor uses, so a browser and a
server build the same model from the same tensors. It is not a more lenient
door into the engine.

**Pick the smallest checkpoint you can live with.** These are 32-bit address
spaces: `whisper-tiny.en` is ~75 MB of f32 weights and comfortable, `base` is
~145 MB, and the large checkpoints are not a browser proposition. Loading two
pipelines into one module instance is not one either — `ffai-carmenta-wasm`
measured ~118 MB of weights in a single linear memory failing to complete.

## API

| | |
|---|---|
| `Recognizer.whisper(weights, config, tokenizer, name)` | load; throws a descriptive error on a bad checkpoint |
| `r.text(samples, sampleRate)` | the whole utterance as one string |
| `r.transcribe(samples, sampleRate)` | `Segment[]` — `{ text, start, end }` |
| `r.detectLanguage(samples, sampleRate)` | the detected tag, or `undefined` |
| `allocator()` | which allocator this module was built with |
| `linearMemoryBytes()` | current linear memory — the only honest memory instrument here |

Word timestamps and diarization are **off** on this target. Both load a second
model on first use, which a 32-bit address space already holding Whisper cannot
afford; they stay off rather than failing halfway through a transcript.

## Build

```
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p ffai-mercury-wasm
wasm-bindgen target/wasm32-unknown-unknown/release/ffai_mercury_wasm.wasm \
    --out-dir crates/ffai-mercury-wasm/pkg --target web
```

`wasm-bindgen` the CLI and `wasm-bindgen` the crate **must be the same
version** — the crate is pinned with `=` for that reason.

`ffai-mercury` is depended on with `default-features = false`, which is not
optional: `fetch` pulls `hf-hub`, whose 1.0 release calls `reqwest::blocking`,
a module that does not exist on wasm32.

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

## Status

`experimental`. It builds, it binds, and it loads and fails cleanly against a
bad checkpoint. **No corpus gate has been run on a wasm build** — text matching
native is the claim that would need one, and it has not been made.
