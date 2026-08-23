# ffai-carmenta-wasm

**OCR in the browser, in pure Rust.** Detection, recognition and reading order
all run inside the WebAssembly module. There is **no Tesseract.js, no ONNX
Runtime Web, and no JavaScript inference engine**: the same Rust that runs
natively is the Rust that runs here.

Part of [FFai](https://github.com/Remade-With-Rust/FFAI). Carmenta is its OCR
component.

## It runs, and it agrees with the native build

Measured in Node 24 on `wasm32-unknown-unknown`, `mobiledet-crnn`:

| | |
|---|---|
| module | **2.47 MB** (release) |
| weights | **4.7 MB** detector + 15 MB recognizer |
| model load from bytes | **41 ms** |
| `readLine`, 200×40 crop | **206 ms** |
| output | identical text to the native build |

The last row is the one that matters most. `CraftCrnn::from_bytes` is gated
against the filesystem constructor for byte-identical text
(`ffai-carmenta/tests/from_bytes.rs`), so the browser is not getting a more
lenient engine — only a differently-fed one.

**These are not browser numbers and not a comparison.** They were taken in
Node, on one crop, on a machine with other tenants. Nothing here should be
repeated as "Carmenta is X ms in the browser".

## Read the region, not the page

**Detection is the whole cost, and in wasm it is not interactive.** On a
620×200 capture, `det_fwd` is **88.8 %** of native runtime — CRAFT and DBNet
run a fixed canvas no matter how little text is on it — and wasm multiplies
that by roughly two orders of magnitude. A full `read` of that capture did not
finish inside ten minutes in Node.

`readLine` skips detection entirely and returns in **~200 ms**. So the shape
that works in a browser is:

> the user selects a region — a receipt field, a subtitle bar, a screenshot
> crop, a camera frame already cropped to the text — and you call `readLine`.

`read` and `text` are there, they are correct, and they are for offline or
server-side use until threads (`wasm-bindgen-rayon`) and SIMD (below) land.

```js
import init, { Reader } from './pkg/ffai_carmenta_wasm.js';
await init();

const r = Reader.mobiledetCrnn(detBytes, crnnBytes);

const ctx = canvas.getContext('2d');
const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
for (const line of r.readLine(data, width, height)) {
  console.log(line.text, line.x, line.y, line.width, line.height);
}
```

`readLine` takes exactly the RGBA layout `getImageData` hands out; alpha is
ignored. `text()` returns the whole page in the reading order the engine
computed.

## Weights are yours, and are never bundled

There is no filesystem to read and no mmap to take, so every constructor takes
bytes. Three pairs are offered, and **the right one is not the one the native
ranking predicts**:

| constructor | detector | recognizer, 200×40 `readLine` |
|---|---:|---:|
| **`mobiledetCrnn`** | **4.7 MB** | **206 ms** |
| `craftCrnn` | 83 MB | 206 ms |
| `mobiledetSvtr` | 4.7 MB | **8905 ms** |

**SVTR is 43× slower here.** Natively it is the document default and only
~1.9× slower than CRNN, which is a price worth paying for its accuracy. It is
a transformer, so every matmul lands in candle's `gemm` — and `gemm` has no
SIMD on wasm (below). CRNN is convolutions and an LSTM. A native benchmark
cannot tell you this; only running it in the target can.

Load **one pair per module instance.** Loading two at once put ~118 MB of
weights into a single wasm linear memory and did not complete.

## Build

```
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p ffai-carmenta-wasm
wasm-bindgen target/wasm32-unknown-unknown/release/ffai_carmenta_wasm.wasm \
    --out-dir crates/ffai-carmenta-wasm/pkg --target web
```

`wasm-bindgen` the CLI and `wasm-bindgen` the crate **must be the same
version** — the crate is pinned with `=` for that reason.

`ffai-carmenta` is depended on with `default-features = false`, which is not
optional: `fetch` pulls `hf-hub`, whose 1.0 release calls `reqwest::blocking`,
a module that does not exist on wasm32 (24 compile errors before any of our
code is reached).

Then serve `demo.html` beside `pkg/` and open it.

## What is different on this target

**Serial.** `wasm32-unknown-unknown` has no threads, so `ffai_carmenta::par`
supplies the identical serial iterators and `rayon` is not a dependency of this
build at all. That costs less than it sounds: three rayon levels nest natively,
and a one-line strip measured **177 ms/line under `par_iter` against 82 ms
serial**. On wasm the nesting cannot happen anyway — candle's
`default_num_threads()` calls `num_cpus::get_physical()`, whose wasm32 branch
returns a literal `1`, so candle takes `Parallelism::None`.

**No SIMD in candle, and that is an upstream defect rather than a limit.**
`candle-core` 0.11 does not compile with `-C target-feature=+simd128`:
`cpu/mod.rs` gates `vec_add_f16` on `any(neon, avx2, simd128)` and uses
`CurrentCpuF16`, which the simd128 path never defines. So the global flag is
unavailable, and with it `gemm`'s wasm SIMD kernels and LLVM's
auto-vectorisation of every scalar loop. **That is the single largest
outstanding lever for this crate, it explains both the detection cost and the
SVTR gap, and it is a few lines in someone else's repository.**

**Our own kernel does vectorise, and it did not help.** A function-level
`#[target_feature(enable = "simd128")]` emits real `f32x4` instructions with the
global flag off — so `ffai_carmenta::conv3x3` has a SIMD128 twin that builds
today. Measured interleaved on the CRNN path in Node: 206 ms against candle's
222 ms, and 255 ms on a second round — **inside the noise**. It ships as the
default because it is correct (identical text to candle) and because it is the
arm that will matter once `gemm` is vectorised too, but it is not a win to
advertise. `--features wasm-candle-conv` builds the other arm; wasm cannot
switch at run time, so the A/B is two builds.

**SIMD128 is a baseline requirement, not a runtime upgrade.** wasm validates a
whole module ahead of time, so a `v128` instruction anywhere makes the module
require SIMD support — there is no `is_wasm_feature_detected!`. Every browser
has shipped it since Safari 16.4 (March 2023); a non-SIMD fallback would have
to be a second module.

## Instruments

* `node_smoke.mjs` — does it run, and how long does a full `read` take.
* `pair_ab.mjs` — which recognizer, interleaved, min-of-rounds.
* `crnn_ab.mjs` — the SIMD128-vs-candle conv A/B, two builds.

## Status

`experimental`. It loads, it reads, and it agrees with the native build on the
crops tested. Not yet done: threads via `wasm-bindgen-rayon` (needs
`SharedArrayBuffer` and cross-origin isolation from whoever serves it), the
upstream candle fix, an accuracy gate on a wasm build, and any measurement in
an actual browser.
