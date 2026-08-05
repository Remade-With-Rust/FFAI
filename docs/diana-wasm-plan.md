# Diana on WebAssembly — Plan

**Component:** Diana — FFai's detection component (`ffai-diana`)
**Status:** compiles for `wasm32-unknown-unknown`; does **not** run in a
browser and is **~2.9× slower** than native even where it does
**Prime directive inherited:** no claim without a number. Every figure below
was measured on this machine or read out of a manifest, and the ones that
were **not** measured are labelled as such.

---

## 0. Where this actually stands

`cargo check -p ffai-diana --target wasm32-unknown-unknown` is **clean**.
That is the whole of the good news, and "compiles" is a long way from
"deploys".

| | native | wasm32 |
|---|---:|---:|
| transitive crates | 138 | **95** |
| compiles C | yes — `onig_sys` | **no** |
| SIMD | AVX2 (8×f32) | **none today** |
| threads | 4-worker pool | **none today** |
| p50, 640 rect, yolo26n | **32.5 ms** | *not runnable yet* |

Two things were fixed to get even this far, both ours:

* `direct3x3.rs` called `direct3x3_avx2::tile` **without** the
  `#[cfg(target_arch = "x86_64")]` that guards the kernel itself, so the call
  compiled on x86 and nowhere else. Latent since the microkernel landed;
  found by compiling for wasm32, never by reading.
* `getrandom` arrives transitively through candle and cannot select its own
  browser backend, so a wasm consumer hit `compile_error!` twice before
  reaching any of our code. Now declared, target-gated, free on native.

**The C question is settled and is not a wasm problem.** candle declares
`[target.'cfg(not(target_arch = "wasm32"))'.dependencies.tokenizers]`, so
`onig` — and the only C in Diana's tree — is excluded upstream on wasm32.

---

## 1. The cost of what is missing, measured

The AVX2 kernels are `#[cfg(target_arch = "x86_64")]` and rayon has no
threads on `wasm32-unknown-unknown`. Both were simulated **natively** so the
number is real rather than projected — `FFAI_DIANA_NO_AVX2=1` and
`FFAI_DIANA_THREADS=1`, min-of-80, three runs each:

| configuration | min-of-80 | vs native |
|---|---:|---:|
| AVX2 on, 4 workers — native today | **32.5 ms** | 1.00× |
| AVX2 off, 4 workers | 40.6 ms | 1.25× |
| **AVX2 off, 1 worker — plain wasm** | **93.9 ms** | **2.9×** |

Against Ultralytics' 45 ms native p50, a plain wasm build lands near 94 ms.
**The parity claims in the README do not carry to a browser** and must not be
repeated there.

This is a *simulation of the two missing capabilities on native hardware*,
not a wasm measurement. It bounds the problem; it does not measure the
target. wasm's own overheads — bounds checking, linear-memory addressing,
JIT warmup — are **not** in it and will make the real figure worse. No wasm
number can be quoted until something runs.

---

## 2. Why SIMD is lost, which is not the reason anyone assumes

wasm has SIMD. All three layers exist already:

* **SIMD128 is standardised** and shipped in every major browser;
* **Rust exposes it** — `core::arch::wasm32::{v128, f32x4_splat, f32x4_add,
  f32x4_mul, …}`, every primitive our SiLU polynomial needs;
* **candle already has a kernel module for it** — `src/cpu/simd128.rs`, with
  a `Cpu` impl.

It is lost to an **upstream defect**. `candle-core/src/cpu/mod.rs` exports
different symbol sets per ISA:

```rust
pub use avx::{CurrentCpu, CurrentCpuBF16, CurrentCpuF16};   // x86
pub use neon::{CurrentCpu, CurrentCpuBF16, CurrentCpuF16};  // arm
pub use simd128::CurrentCpu;                                //  wasm — one only
```

and then gates `vec_add_f16` on `any(neon, avx2, simd128)`, where it uses
`CurrentCpuF16` — which the simd128 path never defines. Verified:

```
RUSTFLAGS='-C target-feature=+simd128' cargo check --target wasm32-unknown-unknown
error[E0433]: cannot find type `CurrentCpuF16` in this scope
   --> candle-core-0.11.0/src/cpu/mod.rs:259:15
```

**candle-core 0.11 does not build for wasm32 with SIMD enabled.** A wasm
build today is therefore necessarily `+simd128`-off and necessarily scalar.
That is where the 2.9 × comes from, and it is a bug rather than a limit —
the distinction matters because it changes the work from "write a vector
backend" to "unblock one that exists".

---

## 3. The path, in dependency order

Each step is gated. Nothing downstream is worth starting before the step
above it lands, and **step 1 is not ours to write**.

### Step 1 — unblock SIMD upstream *(candle, not us)*

Either drop `simd128` from that `cfg(any(...))` for the f16/bf16 paths, or
implement `CurrentCpuF16`/`CurrentCpuBF16` in `simd128.rs`. Small either way,
and it unblocks anyone doing vision on candle in a browser, not just Diana.

* **Action:** file the issue with the reproduction above; offer the PR.
* **Interim:** `[patch.crates-io]` on a fork carries the same maintenance
  cost as any fork and should be taken only if step 2 is urgent.
* **Gate:** `RUSTFLAGS='-C target-feature=+simd128' cargo check --target
  wasm32-unknown-unknown -p ffai-diana` exits 0.

### Step 2 — turn `+simd128` on and take what is free

With step 1 landed, `+simd128` alone buys candle's vectorised f32 kernels
**and** LLVM auto-vectorisation of our scalar loops. **No code from us.**

* **Gate:** re-run the step-1 check, then measure. Expect somewhere between
  the two rows of §1 — 1.25 × is the ceiling this can reach, since that is
  what AVX2 is worth over scalar at 4 workers, and SIMD128 is half the width.
* **Prune first:** if the measured gain is under the harness's resolution on
  whatever wasm runtime is chosen, stop here and say so.

### Step 3 — weights without a filesystem — **DONE**

`Yolo26::from_bytes` and `Yolo26Depth::from_bytes` take the converted
safetensors and the manifest JSON directly, with no `std::fs` anywhere on the
path. Validation is identical to the filesystem constructor — a manifest whose
scale disagrees with the requested tier is refused, and the test asserts that
— so the bytes path is not the lenient door.

Gated by `from_bytes_matches_the_filesystem_path`: **byte-identical** depth
maps from the two constructors, since they differ only in where the bytes came
from. `cargo check --target wasm32-unknown-unknown` still passes.

That leaves **threading** (step 4) as the only remaining runtime blocker for a
browser, and it is a deployment constraint (cross-origin isolation) rather
than code. The same API also serves embedded targets that ship weights in
flash, which was the second reason to build it first.

*Original scoping, kept:*

`Yolo26::build(tier, geometry, manifest_dir)` reaches `ffai_models::load_dir`,
which is `std::fs` against a filesystem a browser does not have. **This blocks
running at all, independently of speed**, and it is the piece to build first
if the goal is "works in a browser" rather than "is fast in a browser".

* **Shape:** a from-bytes constructor — caller fetches the safetensors and
  the manifest, hands Diana the buffers. No I/O inside the engine.
* **Second beneficiary:** embedded targets that ship weights in flash rather
  than on a filesystem want exactly this API, so it is not wasm-only work.
* **Gate:** the five-tier oracle passes with weights supplied as bytes, and
  the byte path and the path path produce byte-identical detections.

### Step 4 — threads, or an honest single-threaded story

`rayon` is in the wasm tree and compiles, but does nothing without
`+atomics,+bulk-memory` and a threaded build (`wasm-bindgen-rayon`, plus
cross-origin isolation headers on the serving side).

* Worth **1.21 × wall / 3.5 × CPU** natively — the largest single lever in
  §1, since 40.6 → 93.9 ms is the thread loss and 32.5 → 40.6 is the SIMD
  loss.
* Cross-origin isolation is a **deployment** constraint on whoever embeds
  Diana, not a code change we can make for them. If the target cannot set
  those headers, single-threaded is the honest answer and the number to
  publish is the single-threaded one.
* **Gate:** cores-busy > 1.0 measured in the runtime, not assumed from the
  build flags.

### Step 5 — port the two kernels

`silu_avx2` and `direct3x3_avx2` to `core::arch::wasm32`. Same polynomial,
**4 lanes instead of 8**. Mechanical rather than novel: the scalar twin
already exists as the oracle for both, and `silu_avx2` already documents why
its Horner form uses separate mul and add rather than FMA — that reasoning
transfers unchanged.

* **Do not start before step 2 is measured.** If auto-vectorisation under
  `+simd128` already recovers most of the 1.25 ×, hand-written kernels are
  competing for what is left, and this campaign has already spent three
  levers on prizes that arithmetic showed did not exist.
* **Gate:** each kernel matches its scalar twin within the bound already in
  `silu_avx2`'s tests, and the five-tier oracle passes.

---

## 4. What must not be claimed

* **No wasm performance number** until something runs in a wasm runtime. §1
  is a native simulation of two missing capabilities and is labelled as one.
* **Nothing about parity.** Diana is at or slightly better than Ultralytics
  *native*. A browser build is a different engine on a different ISA with a
  different threading model, and inherits none of it.
* **The 95-crate figure is not a performance claim.** It is lower than
  native's 138 because candle excludes `tokenizers` there — the same target
  change that removes SIMD. Fewer crates and slower execution have the same
  cause and are not related as cause and effect.

---

## 5. Order of work, if the goal is a browser demo

1. **Step 3** (weights from bytes) — without it nothing runs, and it helps
   embedded targets too.
2. **Step 1** (file the candle issue) — costs an hour, unblocks step 2 for
   everyone, and the fix may land without us.
3. **Step 2** (measure `+simd128`) — free once step 1 lands.
4. **Step 4** (threads) — largest lever, but gated on a deployment
   constraint the embedder controls.
5. **Step 5** (port kernels) — only if step 2 leaves enough on the table to
   justify it.
