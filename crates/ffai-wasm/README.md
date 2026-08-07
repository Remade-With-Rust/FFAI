# ffai-wasm

**YOLO26 object detection in the browser, in pure Rust.** The whole graph
compiles to WebAssembly — backbone, neck, the NMS-free one2one head, the
letterbox, the decode. There is **no ONNX Runtime Web underneath, no
TensorFlow.js, and no JavaScript inference engine**: the same Rust that runs
natively is the Rust that runs here.

Part of [FFai](https://github.com/Remade-With-Rust/FFAI). Diana is its
detection component.

## It runs, and it agrees with the native build

Measured on `coco-032.png` (428x640), yolo26n, rect geometry, conf 0.25, in
Node 24 on `wasm32-unknown-unknown`:

| | native | wasm |
|---|---:|---:|
| detections | 13 | **13** |
| class mismatches | — | **0** |
| largest box/confidence difference | — | **0.000000** |
| checksum over all boxes | 16094.598199 | 16094.598099 |

The checksum differs in the ninth significant figure — **6.2 parts per
billion** — because native uses AVX2 and wasm has no SIMD, so floating-point
addition reassociates differently. Every box and every class is identical at
display precision. This is the expected shape of a cross-target float
comparison, and it is stated rather than rounded away.

| | |
|---|---|
| module | **1.82 MB** (`wasm-bindgen --target web`, release) |
| model load | **18 ms** from bytes |
| detect | **257 ms** min, 288 ms median, single-threaded |

Native on the same image is ~34 ms with a thread pool, so wasm is roughly 8x
slower in wall clock. That is the honest number and it is mostly threads: see
below.

## Serial is the arm that was already winning on CPU

`wasm32-unknown-unknown` has no threads to spawn, so `ffai-diana`'s
`crate::par` supplies the identical serial iterators and **`rayon` is not a
dependency of the wasm build at all**. Dropping it entirely is what proves the
shim is complete — a call site that still reaches for a parallel iterator fails
to compile rather than panicking in someone's browser.

That is not purely a concession. Diana measured the intra-image fan-out at:

| rayon threads | CPU ms/image |
|---:|---:|
| 1 | **363** |
| 24 | 844 |

**The work is 363 ms; twenty-four threads spend 844 ms doing it.** Wasm loses
wall-clock parallelism. It does not lose the efficient arm.

## Weights are yours, and are never bundled

YOLO26 checkpoints are **AGPL-3.0**. This crate ships none and cannot: the
constructor takes the safetensors bytes and the manifest JSON, which is also
the only thing that works in a browser, since there is no filesystem to read.

```js
import init, { Detector } from './pkg/ffai_wasm.js';
await init();

const det = new Detector(safetensorsBytes, manifestJson, 'n');

const ctx = canvas.getContext('2d');
const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
for (const d of det.detect(data, width, height, 0.25)) {
  console.log(d.name, d.confidence, d.x0, d.y0, d.x1, d.y1);
}
```

`detect` takes exactly the RGBA layout `getImageData` hands out. Alpha is
ignored.

## Build

```
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p ffai-wasm
wasm-bindgen target/wasm32-unknown-unknown/release/ffai_wasm.wasm \
    --out-dir crates/ffai-wasm/pkg --target web
```

`wasm-bindgen` the CLI and `wasm-bindgen` the crate **must be the same
version** — the crate is pinned with `=` for that reason, and a mismatch is a
confusing runtime failure rather than a build error.

`.cargo/config.toml` supplies `--cfg getrandom_backend="wasm_js"`, which
`getrandom` 0.3 requires *in addition to* its `wasm_js` feature; it errors if
either half is missing.

## The allocator, because a cdylib is a binary

A library cannot set `#[global_allocator]` — only the final artifact can, and
here that is this crate. wasm32's Rust default is **dlmalloc**. This module
uses [`rusty_alloc`](https://crates.io/crates/rusty_alloc), the pure-Rust
mimalloc remake that replaced the C allocator in our native binary at parity,
and which already ships a wasm backend with a measured wasm-specific decision
(no arena pre-reservation, since `memory.grow` is never returned to the host).

`allocator()` reports which one the module was built with. **The A/B has now
been run.** Two modules differing in exactly one `#[global_allocator]` line,
both loaded into one Node process and rotated per iteration, N=40, warm-up
discarded, with a NULL arm (two detectors from the same module):

| arm | min | median | p90 |
|---|---:|---:|---:|
| rusty_alloc | 170.3 ms | 184.0 ms | 190.3 ms |
| dlmalloc | 171.7 ms | 185.2 ms | 193.5 ms |
| **NULL** (rusty against itself) | 177.8 ms | 184.5 ms | 188.6 ms |

**Parity.** rusty_alloc reads 1.007x with 26/40 paired wins at **z = 1.90**,
against a null arm at 1.003x and z = -0.32 — a hair ahead, and short of the
|z| > 2 the repo requires before calling anything a win. An earlier 24-iteration
pass without the null arm read the OPPOSITE (dlmalloc 2.7 % ahead), which is
what a difference inside the noise looks like and why the null arm is not
optional.

Matching dlmalloc — the wasm default, and a mature one — is the useful result
for a young allocator. `--no-default-features` builds the dlmalloc arm if you
want to re-run it.

## Status

`experimental`. It loads, it detects, and it agrees with native to display
precision. Not yet measured: SIMD
(`wasm32-unknown-unknown` supports 128-bit SIMD behind a target feature and
Diana's kernels have never been built for it), and threads via
`wasm-bindgen-rayon`, which would need SharedArrayBuffer and cross-origin
isolation from whoever serves it.
