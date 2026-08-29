# ffai-argus-wasm

**Argus in the browser** — pure-Rust SmolVLM image captioning compiled to
WebAssembly. Preprocessing, the SigLIP vision tower, the pixel-shuffle
connector and the Llama text decoder all run inside the wasm module. There is
no ONNX Runtime Web, no transformers.js, and no Python anywhere: the same Rust
that runs natively is the Rust that runs here.

**Status: `0.1.0` — experimental.** It runs, and it agrees with the native
build byte for byte on the image below. No corpus gate has been run on a wasm
build; the caption-quality record in `ffai-argus` was earned natively and must
not be quoted for this target without re-measuring.

Executed on `wasm32-unknown-unknown` (Node VM, no emscripten). Requires a
browser with WebAssembly — no threads, no SIMD, and no C toolchain anywhere in
the build.

## What it is

- **A `cdylib` wrapper over [`ffai-argus`](https://crates.io/crates/ffai-argus)**,
  exposing SmolVLM through `wasm-bindgen` as two calls.
- **Weights are yours** — `Captioner.smolvlm` takes `model.safetensors`,
  `config.json` and `tokenizer.json` as bytes. Nothing is bundled and nothing
  is fetched for you. There is no filesystem to read and no mmap to take.
- **Gated against native.** `SmolVlm::from_bytes` funnels into the same `build`
  the manifest path uses, so a browser and a server assemble the same model.
  `examples/from_bytes_smoke.rs` in `ffai-argus` runs that constructor natively
  so it is debuggable.
- **Geometry is read, never assumed** — the tile size, the `<image>` token id
  and the end-of-turn token set all come from the checkpoint's own config on
  both paths, so a different SmolVLM size loads correctly rather than silently
  mis-tiling.

## Performance

`SmolVLM-256M-Instruct`, a COCO photo, 20-token budget, Node on x86-64 Windows,
single-threaded.

| path | tiles / image tokens | time | caption |
|---|---|---:|---|
| **`describeFast`** | **1 / 64** | **37.6 s** | *"In the foreground of the picture there is a bear. In the background there is grass."* |
| `describe` | 17 / 1088 | 544.1 s | *"A grizzly bear is in the center of the frame."* |

**14.5x, and read the captions before assuming what it cost.** The unsplit run
keeps the subject and adds background the split run missed; what splitting buys
is fine print and small objects, not the gist. Use `describe` when you need to
read text in the image and can afford nine minutes.

**Image size does not matter, and that is the whole insight.** `describe` takes
505 s on a 224x224 image and 504 s on a 586x640 one, because both are resized
into the same fixed tile grid. The grid is the cost, not the pixels — shrinking
the input does nothing, dropping the split does everything.

### On SIMD, since it is the obvious suspect

It is already on, and it is already load-bearing. Turning gemm's simd128
microkernels off costs **3.53x** (505 s -> 1786 s, identical caption), which is
the proof they execute. They compile without `-C target-feature=+simd128`
because they are gated on `target_arch = "wasm32"` and carry
`#[target_feature(enable = "simd128")]` per function.

The workspace does now set `+simd128` (see `.cargo/config.toml`), which is worth
**1.32-1.40x on Mercury** by letting LLVM vectorise candle's own scalar loops.
On Argus it measures ~3%, inside noise, because Argus's hot path is gemm and
gemm was never scalar. Do not expect SIMD work to move this crate further.

The remaining gap to native is threads: `wasm32-unknown-unknown` has none, and
Argus fans out across roughly a dozen rayon sites natively.

## Usage

```js
import init, { Captioner } from './ffai_argus_wasm.js';
await init();

const c = Captioner.smolvlm(weights, configJson, tokenizerJson);
// describeFast, not describe — see Performance. 14.5x, and the caption holds up.
console.log(c.describeFast(rgbaFromCanvas, width, height, 'What is in this image?', 40));
```

| | |
|---|---|
| `Captioner.smolvlm(weights, config, tokenizer)` | load; throws a descriptive error on a bad checkpoint |
| `c.describeFast(rgba, w, h, prompt, maxNewTokens)` | **start here** — 1 tile, 14.5x faster |
| `c.describe(rgba, w, h, prompt)` | 17 tiles; reads fine print, costs ~9 minutes |
| `c.describeWithLimit(rgba, w, h, prompt, maxNewTokens)` | `describe` with a token budget |
| `allocator()` | which allocator this module was built with |
| `linearMemoryBytes()` | current linear memory — the only honest memory instrument here |

Build it with:

```
cargo build --release --target wasm32-unknown-unknown -p ffai-argus-wasm
wasm-bindgen target/wasm32-unknown-unknown/release/ffai_argus_wasm.wasm \
    --out-dir crates/ffai-argus-wasm/pkg --target web
```

`wasm-bindgen` the CLI and the crate **must be the same version** — the crate
is pinned with `=` for that reason. `ffai-argus` is depended on with
`default-features = false`, which is not optional: `fetch` pulls `hf-hub`,
whose 1.0 release calls `reqwest::blocking`, a module that does not exist on
wasm32.

## Sizing is the binding constraint

`SmolVLM-256M` is ~500 MB of weights against a 4 GB linear memory that **only
ever grows** — nothing you free is returned to the browser, so peak is the
whole budget. It fits, with room for the vision tower's activations and a KV
cache, and nothing larger is a browser proposition.

Tiling is what makes an image expensive rather than pixels: a still is 17 tiles
and 1088 image tokens. `describeWithLimit` bounds the other half — decode is
autoregressive and the KV cache grows with every token, so the token budget is
the knob that bounds both time and memory on a long answer.

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
mmap and is a second full copy of ~500 MB in a 32-bit address space, so it now
clones a builder instead. Native gets the same saving.

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
