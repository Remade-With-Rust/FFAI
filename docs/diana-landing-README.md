<!--
Landing README for a standalone `Remade-With-Rust/diana` repository, mirroring
what `Remade-With-Rust/mercury` and `Remade-With-Rust/carmenta` do for those
components. It is kept here so it versions with the code whose numbers it
quotes; copy it to that repo's root as README.md.

Rule inherited from Mercury's page: every figure is a ledger line or it does not
appear. Where a claim is qualified, the qualifier ships with it.
-->

# Diana

**Object detection in pure Rust.**

Diana runs the real YOLO26 graph — C3k2, SPPF, C2PSA, attention, and the
NMS-free one2one head — on [candle](https://github.com/huggingface/candle),
from official Ultralytics checkpoints converted offline. No Python runtime,
no ONNX, no gated models, and **no vendored weights**.

It is the detection component of [FFai](https://github.com/Remade-With-Rust/FFAI),
named for the Roman goddess of the hunt.

```toml
[dependencies]
ffai-diana = "0.6"
ffai-core = "0.6"
ffai-media = "0.6"
```

---

## What it does well, measured

Against Ultralytics 8.4.113 and ONNX Runtime on a hash-pinned 45-image COCO
holdout, CPU only, yolo26n at 640 rect
([`bench-detect-1785728764`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl)):

| | mAP50 | p50 latency | ×realtime | steady RSS |
|---|---:|---:|---:|---:|
| **Diana** | **0.7014** | **41 ms** | **22.3×** | **160 MiB** |
| ultralytics-yolo26n-rect | 0.7014 | 45 ms | 19.4× | 310 MiB |
| ort-yolo26n | 0.6865 | 31 ms | 31.6× | 160 MiB |

**mAP is identical to PyTorch to four decimals**, at lower latency and half
the memory. Model load is 68 ms.

Beyond the aggregate: across all ten tier/geometry configurations mAP matches
PyTorch to within 0.08 pp on a 450-image holdout, and at n, m, l and x **every
detection is identical** — same count, same classes, same order across 724
detections, boxes within 0.30 px.

## What it does not do yet

**The speed gate FAILS**, and against ONNX Runtime rather than PyTorch: ORT is
31 ms to our 41. It is the fastest thing in the field and the gate compares
against the fastest, which is the right rule. Diana is ahead of it on accuracy
(0.7014 vs 0.6865) and level on memory.

**The footprint gate passes by 1 MiB** — 160 against ORT's 161. That is not a
margin. mimalloc retains ~130 MiB of allocator churn against 26 MiB actually
live, and the durable fix is upstream of the allocator.

Detection is single-image; video ingest is not wired. Only COCO's 80 classes
are exercised.

---

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
an audited deterministic transcription into safetensors plus a manifest — no
retraining, no fine-tuning, no derivation. The AGPL obligations that attach to
the weights stay with the weights you obtained yourself.

The converter fails closed: a shape that does not match what the graph expects
is an error, never a silent partial load. That rule caught a real bug —
`model.6.m.0.m.0.cv1` built as 32→16 where the checkpoint has 32→32 — on its
first run.

---

## Embedding it

Diana does **not** drag in the rest of FFai. It has no dependency on
`ffai-mercury`, `ffai-carmenta` or `ffai-argus`; those are sibling crates, not
layers underneath.

| build | transitive crates | compiles C? |
|---|---:|---|
| `ffai-diana`, default | **138** | yes — `onig_sys` |
| `ffai-diana` + `ffai-models/fetch` | 308 | yes — `onig_sys`, `aws-lc-sys` |
| **`wasm32-unknown-unknown`** | **95** | **no** |

The 170-crate difference on native is the Hugging Face downloader —
`reqwest`, `hyper`, `rustls`, `aws-lc-sys`. Diana never calls it; its whole
use of `ffai-models` is `load_dir`, which reads TOML off disk. Off by default.

**One C dependency remains on native and it is not ours.** `candle-core` takes
`tokenizers` as a hard, non-optional dependency with `features = ["onig"]` — a
C regex engine, for text models Diana never touches, reached through one candle
module it never calls. `tokenizers` itself marks `onig` optional and ships a
pure-Rust alternative, so nothing technical requires this; it is one hardcoded
feature line upstream, and it cannot be gated from here.

It is **build-time only**: the output is an ordinary native binary with
Oniguruma statically linked, no shared library to ship, no runtime dependency.
It matters for musl/static builds, cross-compilation, minimal containers and
no-C supply-chain policies, and nowhere else.

**On wasm32 it disappears.** candle declares that dependency as
`[target.'cfg(not(target_arch = "wasm32"))'.dependencies.tokenizers]`, so a
wasm build is 95 crates with no `onig` and no `cc`.
`cargo check --target wasm32-unknown-unknown` is clean — but compiling is not
deploying, and two runtime pieces are **not** done: weights load through
`std::fs`, which a browser does not have, and `rayon` compiles for wasm yet
needs atomics plus a threaded build to do anything.

**The allocator is not inherited.** The system allocator re-faults nearly every
byte it hands back — 58,634 page faults per image — and costs **1.66×**. A
library cannot set a global allocator, so an embedder opts in itself:

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

That is a trade: it buys the 1.66× and costs roughly 120 MiB of retained
pages, because retention *is* the mechanism.

---

## How the speed was found, including what did not work

p50 went **85 ms → 41 ms** in one campaign. The four things that moved it were
not the four anyone would guess:

* **the allocator** — 1.64×, from 58,634 page faults per image;
* **the thread pool** — 1.21× wall and 3.5× CPU; one image wants ~4 workers,
  not 24, and candle keeps its own pool besides;
* **preprocessing** — 5.7× alone; a serial bilinear resize was recomputing the
  horizontal sample position for every column of every row;
* **epilogue fusion** on both convolution paths — 12.5%; bias and SiLU in one
  traversal instead of three.

What did **not** work, each refuted with numbers: im2col tiling (twice), direct
convolution in four shapes including a hand-written AVX2 microkernel, implicit
GEMM, the im2col zero-fill, and elementwise traffic fusion. One cause explains
most of them — **nothing in this graph leaves L3**, so every prize priced at
DRAM bandwidth was overstated by 3–15×.

The full campaign, every reverted experiment and every retracted number
included:
[docs/whys/diana-latency.md](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/whys/diana-latency.md).

## License

MIT OR Apache-2.0. **Model weights are not covered by it** — YOLO26
checkpoints are AGPL-3.0 and you supply your own.
