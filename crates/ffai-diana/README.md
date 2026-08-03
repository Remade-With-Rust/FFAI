# ffai-diana

**Object detection in pure Rust.** YOLO26 — the real graph, C3k2 / SPPF / C2PSA / attention and the NMS-free one2one head — running on candle from official Ultralytics checkpoints. No Python runtime, no ONNX, and **no vendored weights**.

Diana is [FFai](https://github.com/Remade-With-Rust/FFAI)'s detection component. Standalone landing page: [Remade-With-Rust/diana](https://github.com/Remade-With-Rust/diana).

```toml
[dependencies]
ffai-diana = "0.6"
ffai-core = "0.6"
ffai-media = "0.6"
```

## Detect

```rust
use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_diana::{engine::Yolo26, image::Geometry};
use std::path::Path;

let engine = Yolo26::build("n", Geometry::Rect, "models");
let image  = ffai_media::load_image(Path::new("street.jpg"))?;   // JPEG or PNG
let found  = engine.detect(&image, &DetectOptions::default())?;
let names  = engine.class_names();

for d in &found.detections {
    println!("{} {:.2} @ [{:.0} {:.0} {:.0} {:.0}]",
             names[d.class_id as usize], d.confidence, d.x0, d.y0, d.x1, d.y1);
}
```

This example is compiled as
[`examples/quickstart.rs`](https://github.com/Remade-With-Rust/FFAI/blob/master/crates/ffai-diana/examples/quickstart.rs),
so it type-checks against the real API rather than against memory.

Or through the registry, which registers all five tiers in both geometries:

```rust
let mut reg = ffai_core::registry::EngineRegistry::new();
ffai_diana::register(&mut reg);
let engine = reg.detect(Some("yolo26s"))?;
```

## Embedding it: what you actually pull in

Diana does **not** drag in the rest of FFai. It has no dependency on
`ffai-mercury`, `ffai-carmenta` or `ffai-argus` — those are sibling crates,
not layers underneath. Bundling detection alone is the normal case, not a
special one.

| build | transitive crates | compiles C? |
|---|---:|---|
| `ffai-diana`, default | **138** | yes — `onig_sys` |
| `ffai-diana` + `ffai-models/fetch` | 308 | yes — `onig_sys`, `aws-lc-sys` |
| **`wasm32-unknown-unknown`** | **95** | **no** |

The 170-crate difference on native is the Hugging Face downloader —
`reqwest`, `hyper`, `rustls`, `aws-lc-sys` — and Diana never calls it. Its
whole use of `ffai-models` is `load_dir`, which reads TOML off disk. The
fetch stack is **off by default here**; enable `ffai-models/fetch` if you want
weights pulled from the hub rather than shipped alongside.

### The C dependency, and where it is not

A native build **needs a C compiler**, for one library that is not ours:

```
cc → onig_sys → onig → tokenizers → candle-core → ffai-diana
```

`candle-core` takes `tokenizers` as a hard, non-optional dependency with
`features = ["onig"]` — a C regex engine, for text models Diana never touches,
reached through one candle module (`quantized::tokenizer`) it never calls.
`tokenizers` itself marks `onig` **optional** and ships a pure-Rust
alternative, so nothing technical requires this; it is one hardcoded feature
line upstream. It cannot be gated from here.

This is **build-time only**. The result is an ordinary native binary with
Oniguruma statically linked; there is no shared library to ship and no runtime
dependency. It matters for musl/static builds, cross-compilation, minimal
containers, and any no-C-in-the-supply-chain policy — and nowhere else.

**On wasm32 it disappears entirely.** candle declares that dependency as
`[target.'cfg(not(target_arch = "wasm32"))'.dependencies.tokenizers]`, so a
wasm build is 95 crates with no `onig` and no `cc`.
`cargo check --target wasm32-unknown-unknown` is clean.

Compiling is not deploying, and two runtime pieces are **not** done: weights
load through `std::fs`, which a browser does not have (a from-bytes
constructor is the missing API), and `rayon` compiles for wasm but needs
atomics plus a threaded build to do anything — without it the numbers below,
which were measured with a 4-worker pool, do not carry.

### The allocator is not inherited

Diana's latency depends on it heavily: the system allocator re-faults nearly
every byte it hands back — **58,634 page faults per image** — and costs
**1.66×**. A library cannot set a global allocator, so an embedding
application gets its own default unless it opts in:

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

That is a trade, not a free win: it buys the 1.66× and costs roughly 120 MiB
of retained pages, because retention *is* the mechanism. On a
size-constrained target the system allocator is the leaner and slower choice,
and both halves of that are measured.

## Five tiers from one graph

`n`, `s`, `m`, `l`, `x` all run the **same tier-agnostic graph**. There is no
per-tier code path: depth, width and the `c3k` promotion (`m`/`l`/`x` build
their inner blocks as `C3k`, `n`/`s` do not) come from the checkpoint's own
scale rule, reproduced from Ultralytics' `parse_model`.

Two geometries: `Rect` reproduces Ultralytics' `auto=True` letterbox — the
smallest multiple-of-32 rectangle containing the scaled image — and `Square`
pads to `imgsz × imgsz`, matching the usual ONNX export. They are not
interchangeable and the flag is not cosmetic; it moves mAP.

## The weights are AGPL, and that is why they are not here

Ultralytics' YOLO checkpoints are **AGPL-3.0**. This crate is MIT OR
Apache-2.0 and **ships no weights, vendors none, and redistributes none**.

You bring your own `.pt` and convert it offline with `tools/diana_convert.py`,
which is an audited, deterministic transcription into safetensors plus a
manifest — no retraining, no fine-tuning, no derivation. The AGPL obligations
that attach to the weights stay with the weights you obtained yourself.

The converter fails closed: a shape that does not match what the graph
expects is an error, never a silent partial load. That rule caught a real
bug — `model.6.m.0.m.0.cv1` built as 32→16 where the checkpoint has 32→32 —
on its first run.

## Status: `experimental`, honestly

**Three of four gates PASS.** Measured against Ultralytics 8.4.113 and ONNX
Runtime on a hash-pinned 45-image COCO holdout, CPU only, yolo26n at 640
rect ([`bench-detect-1785728764`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl)):

| | mAP50 | p50 latency | ×realtime | steady RSS |
|---|---:|---:|---:|---:|
| **Diana** | **0.7014** | **41 ms** | **22.3×** | **160 MiB** |
| ultralytics-yolo26n-rect | 0.7014 | 45 ms | 19.4× | 310 MiB |
| ort-yolo26n | 0.6865 | 31 ms | 31.6× | 160 MiB |

**mAP is identical to PyTorch to four decimals**, at lower latency and
**half the memory**. Model load is 68 ms.

**The speed gate still FAILS**, and against ONNX Runtime rather than
PyTorch: ORT is 31 ms to our 41. It is the fastest thing in the field and
the gate compares against the fastest, which is the right rule. We are ahead
of it on accuracy (0.7014 vs 0.6865) and level on memory.

That gate closed a long way in one campaign — p50 **85 → 41 ms** — and the
four things that moved it were not the four anyone would guess:

* **the allocator**, worth 1.64×. 58,634 page faults per image; the system
  allocator was returning memory to the OS and re-faulting nearly every byte;
* **the thread pool**, worth 1.21× wall and 3.5× CPU — one image wants ~4
  workers, not 24, and candle keeps its own pool besides;
* **preprocessing**, 5.7× on its own — a serial bilinear resize recomputing
  the horizontal sample position for every column of every row;
* **epilogue fusion** on both convolution paths, 12.5% — bias and SiLU in one
  traversal instead of three.

What did **not** move it, each refuted with numbers: im2col tiling (twice),
direct convolution (four shapes, including an AVX2 microkernel), implicit
GEMM, the im2col zero-fill, and elementwise traffic fusion — because
**nothing in this graph leaves L3**, so every prize priced at DRAM bandwidth
was overstated by 3–15×.

Not yet `stable`: the footprint gate passes by **1 MiB** (160 against ORT's
161), which is not a margin — mimalloc retains ~130 MiB of allocator churn
against 26 MiB actually live, and the durable fix is upstream of the
allocator. Detection is single-image; video ingest is not wired. Only COCO's
80 classes are exercised.

Every number traces to a line in
[`bench/ledger.jsonl`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl).
The full campaign, every reverted experiment included:
[docs/whys/diana-latency.md](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/whys/diana-latency.md).

## License

MIT OR Apache-2.0. **Model weights are not covered by it** — YOLO26
checkpoints are AGPL-3.0 and you supply your own.
