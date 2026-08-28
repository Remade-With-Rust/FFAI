# ffai-argus-wasm

**Argus in the browser** — pure-Rust SmolVLM image captioning compiled to
WebAssembly. Preprocessing, the SigLIP vision tower, the pixel-shuffle
connector and the Llama text decoder all run inside the wasm module. There is
no ONNX Runtime Web, no transformers.js, and no Python anywhere: the same Rust
that runs natively is the Rust that runs here.

```js
import init, { Captioner } from './ffai_argus_wasm.js';
await init();

const c = Captioner.smolvlm(weights, configJson, tokenizerJson);
console.log(c.describe(rgbaFromCanvas, width, height, 'What is in this image?'));
```

## Size is the binding constraint

`SmolVLM-256M` at f32 is roughly 1 GB of weights against a 4 GB linear memory
that **only ever grows** — nothing you free is returned to the browser. It
fits, with room for the vision tower's activations and a KV cache, and nothing
larger is a browser proposition.

Tiling is what makes an image expensive rather than pixels: a still is 17 tiles
and 1088 image tokens. `describeWithLimit` bounds the other half — decode is
autoregressive and the KV cache grows with every token, so the token budget is
the knob that bounds both time and memory on a long answer.

`linearMemoryBytes()` is exported for exactly this reason. It is the only
honest memory instrument on a target whose memory never shrinks;
`performance.memory` measures the JS heap and says nothing about ours.

## Weights are yours, and are never bundled

There is no filesystem to read and no mmap to take, so `Captioner.smolvlm`
takes the three artefacts as bytes — `model.safetensors`, the text of
`config.json`, and the bytes of `tokenizer.json`. These are the same three
files the manifest resolves natively, and `SmolVlm::from_bytes` funnels into
the same `build` the manifest path uses, so a browser and a server assemble the
same model.

The geometry, the `<image>` token id and the end-of-turn token set are all read
from the checkpoint's own config on both paths, never assumed — a different
SmolVLM size loads correctly rather than silently mis-tiling.

## API

| | |
|---|---|
| `Captioner.smolvlm(weights, config, tokenizer)` | load; throws a descriptive error on a bad checkpoint |
| `c.describe(rgba, w, h, prompt)` | caption; empty prompt uses the model's default |
| `c.describeWithLimit(rgba, w, h, prompt, maxNewTokens)` | the same, with a token budget |
| `allocator()` | which allocator this module was built with |
| `linearMemoryBytes()` | current linear memory |

## Build

```
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p ffai-argus-wasm
wasm-bindgen target/wasm32-unknown-unknown/release/ffai_argus_wasm.wasm \
    --out-dir crates/ffai-argus-wasm/pkg --target web
```

`wasm-bindgen` the CLI and `wasm-bindgen` the crate **must be the same
version** — the crate is pinned with `=` for that reason.

`ffai-argus` is depended on with `default-features = false`, which is not
optional: `fetch` pulls `hf-hub`, whose 1.0 release calls `reqwest::blocking`,
a module that does not exist on wasm32.

## What is different on this target

**Serial.** `wasm32-unknown-unknown` has no threads, so `ffai_argus::par`
supplies the identical serial iterators and `rayon` is not a dependency of this
build at all. Argus reaches rayon at roughly a dozen sites across the vision
tower — `siglip` over rows, hidden units and attention chunks, `preprocess`
over resampled image rows, `text` over the decoder's key/value chunks — so
without the shim a browser build panics inside the first tile rather than at
load.

**No clock.** `Instant::now()` panics on this target and Argus called it 51
times. Every one of those feeds a millisecond report rather than a decision —
that was checked across all twelve consumption sites, not assumed — so on wasm
the profile tables read all-zero instead of lying about a number the target
cannot take.

**One `VarBuilder`, cloned.** The loader used to map the checkpoint twice, once
renamed for candle's tower and once raw for ours. That is nearly free with an
mmap and is a second full COPY of ~1 GB in a 32-bit address space, so it now
clones a builder instead. Native gets the same saving.

**No SIMD in candle, and that is an upstream defect rather than a limit.**
`candle-core` 0.11 does not compile with `-C target-feature=+simd128`, so
`gemm`'s wasm SIMD kernels and LLVM's auto-vectorisation are both unavailable —
and Argus is a transformer, so essentially all of its arithmetic lands in
`gemm`. That is the largest outstanding lever for this crate by a wide margin,
it is a few lines in someone else's repository, and it is tracked in
`docs/plans/carmenta-wasm-plan.md` §2.

## Status

`experimental`. It builds, it binds, and it loads and fails cleanly against a
bad checkpoint. **No corpus gate has been run on a wasm build**, and the
caption-quality record in `ffai-argus` was earned natively — do not quote it
for this target without re-measuring.
