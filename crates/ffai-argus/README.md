# ffai-argus

[![crates.io](https://img.shields.io/crates/v/ffai-argus?logo=rust)](https://crates.io/crates/ffai-argus)
[![docs.rs](https://img.shields.io/docsrs/ffai-argus?logo=docsdotrs)](https://docs.rs/ffai-argus)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](https://github.com/Remade-With-Rust/FFAI)
[![Remade With Rust](https://img.shields.io/badge/Remade%20With-Rust-000?logo=rust&logoColor=fff)](https://github.com/Remade-With-Rust)
[![By Mata Network](https://img.shields.io/badge/by-Mata%20Network-5b2be0)](https://www.mata.network/)

> **ffai-argus** is vision-language captioning and VQA in pure **Rust**:
> `SmolVLM-256M-Instruct` — the real architecture, a `SigLIP` tower, a
> pixel-shuffle connector and a Llama decoder — running on **candle**, and
> reproducing the reference implementation's caption **byte-identically from a
> raw image file**. No Python runtime, no ONNX, no `llama.cpp`. Images and
> video, still captions or a timed `.srt` track, behind one engine trait.

Part of **[Remade With Rust](https://github.com/Remade-With-Rust)** by
**[Mata Network](https://www.mata.network/)** — the vision-language component of
**[FFAI](https://github.com/Remade-With-Rust/FFAI)**, the AI media toolkit,
alongside **[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)**,
our memory-safe FFmpeg alternative.
[Jump to the ecosystem ↓](#the-remade-with-rust-ecosystem)

Argus is named for Argus Panoptes, the all-seeing hundred-eyed watchman.

---

## ⚡ The headline

A VLM that runs as **one Rust binary** and is gated not on "looks plausible" but
on **token equality with the Python reference**:

- **Byte-identical captions.** Not close, not "high SNR" — the same bytes. Raw
  image in through the public surface (`ImageBuffer` → `String`), our resize,
  our tiling, our tower, our prompt assembly, our decode loop: **32/32 output
  tokens identical** to `transformers` on the same checkpoint. Six gates, each
  isolated so a mismatch names a stage
  ([below](#gated-against-the-reference-stage-by-stage)).
- **The real architecture, ported — not wrapped.** Idefics3 AnyRes tiling
  (longest edge 2048 → 512 px tiles plus a global thumbnail = 17 tiles, 1088
  image tokens), a `SigLIP` encoder written here rather than borrowed, the
  pixel-shuffle connector at `scale_factor: 4`, and candle's `llama` as the text
  tower — because `SmolVLM`'s `text_config.model_type` **is** `llama`.
- **Scored against a published number, before anything was optimized.**
  **525/1000** on OCRBench through VLMEvalKit against the checkpoint's published
  **526** — inside a pre-registered 496–556 band by 30×. The scoreboard was
  stood up first so the port could be *priced*, not just admired.
- **Video without a video claim.** `ffai caption -i clip.mp4 --fps 2 --window 8`
  emits `.srt`/`.vtt`/`.json`, with peak memory a function of the window rather
  than the clip. The checkpoint is an image model with no published video row,
  so **no video quality number is invented** — what is gated is the *track*
  ([below](#video)).
- **1.20x off PyTorch end to end** — down from 2.4x — measured stage by stage
  on an idle machine, with the deficit concentrated in the vision tower. That
  tower has since taken a further **1.199x (16.6 %)**, verified by interleaved
  in-process A/B against a null arm; the end-to-end figure has not been
  re-measured since, and is not extrapolated here ([below](#performance)).
- **Deterministic by default.** Greedy decode, reproducible with no seed.
  Sampling is opt-in and *requires* `--seed`: passing `--temperature` alone is
  refused rather than quietly returning a caption you cannot reproduce.

| | `transformers` + PyTorch (Python) | **ffai-argus (Rust)** |
|---|---|---|
| Runtime to deploy | Python, PyTorch, ~GB of wheels | **one binary** |
| Caption vs the reference | *is* the reference | **byte-identical** |
| C/C++ in the tree | PyTorch, its BLAS, ONNX | candle's `onig_sys`, **build-time only**, gated off wasm |
| `unsafe` in this crate | — | `unsafe_code = "warn"`; the few sites are `#[target_feature]` kernels and `spare_capacity_mut` |
| Steady-state footprint | 1852 MiB | **1309 MiB — 0.71×** |
| Licence | mixed | **MIT OR Apache-2.0** |
| Determinism | by convention | **unseeded sampling is unrepresentable in the type** |

<sub>**On the C claim, precisely:** `candle-core` takes `tokenizers` with
`features = ["onig"]` as a hard dependency, so `onig_sys` compiles a C regex
engine into every *native* candle build. It is build-time only, with no runtime
component, and it is target-gated out on `wasm32`. We will not tell you there is
"no C in the tree" for a candle build — there is, and this is where it is.</sub>

### Performance

**Against PyTorch, stage by stage** — the same image (17 tiles, 1142 prompt
tokens, 32 generated), the same checkpoint, both warm, on an idle machine.
Ours from `examples/stage_split`; the reference from
`corpora/refs/smolvlm_hf_profile.py`. Both arms repeated; each figure moved
under 1 % between repeats:

| stage | ffai-argus | `transformers` | |
|---|---:|---:|---|
| preprocess | **52 ms** | 134 ms | **we win 2.6x** |
| vision tower | 8172 ms | **7277 ms** | we lose 1.12x |
| text side (prefill + decode) | 2690 ms | **1695 ms** | we lose 1.59x |
| **whole caption** | **10 918 ms** | **9 106 ms** | **1.20x slower** |

**1.20x**, and the deficit is concentrated in the vision tower. That is the
number to hold this crate to; everything below is how it got there.

**Three optimization rounds.** Measured on the same instrument, same image, at
a matched 24-token budget, against the figures this README previously published:

| stage | before | after | |
|---|---:|---:|---:|
| **prefill** | 3870 ms | **1261 ms** | **3.07x** |
| **generate** | 2181 ms | **1063 ms** | **2.05x** |
| vision tower | 8452 ms | 8181 ms | 1.03x |
| **whole caption** | **14 561 ms** | **10 562 ms** | **1.38x** |

Every gate held throughout: 32/32 tokens, caption byte-identical to the
reference, full suite green.

What moved the text side was **our own `SmolLM2` tower** (`src/text.rs`).
candle's `llama` spends 114 ms per layer — 3424 ms of a caption, **20 % of the
whole thing** — in `masked_fill`, at **0.8 GB/s**, writing `-inf` into an upper
triangle. It costs that much because it is `where_cond` against two *broadcast*
operands, so all 11.7 M elements are a strided gather on one core. Deleting it
is **bit-identical**, not an approximation: softmax takes the row max,
`max(finite, -inf)` never selects `-inf`, and `exp(-inf)` is exactly zero — so
skipping a masked column produces precisely the floats that materialising it
does. Causality is now fused into the softmax kernel, the `1/sqrt(head_dim)`
scale is folded into q's weights (exact — it is a power of two), and
`silu(gate) * up` is one pass.

On the vision side the two wins were both **`broadcast_matmul`**, which
stretches a weight to the batch shape when the batch is 1:

| | as written | fixed | |
|---|---:|---:|---:|
| connector projection | 64.4 ms | **4.6 ms** | **14.1x** |
| patch embedding (`conv2d` -> matmul) | 10.9 ms | **4.1 ms** | **2.6x** |

The patch conv is stride-16 with a 16x16 kernel, so it is non-overlapping and
its im2col is a pure permutation — the matmul form is the same operation, and
it emits `(seq, hidden)` directly, deleting the transpose that followed.

**Round 4-5: the bias was the polluter.** The vision tower's four projections
measured **288-347 GF/s in situ against 486-570 isolated**, and three rounds of
looking for the cause blamed cache pollution and weight residency — both
refuted. The actual cause was that an isolated benchmark calls `matmul`, while
the tower calls `candle_nn::Linear::forward`, which is `matmul` **plus
`broadcast_add(bias)`** — and candle's binary ops are single-threaded. That
second pass re-reads and re-writes the whole activation between every GEMM,
evicting its working set.

Each bias was folded into an op that already touches every element, so the add
is free against traffic already being paid:

| bias | folded into | |
|---|---|---|
| qkv `768->2304` | the packed q/k/v permute-copy | `PackedQkvOp` |
| fc1 `768->3072` | GELU | `GeluBiasOp` |
| fc2 / out_proj | the residual add | `AddBiasOp` |

Two more passes went the same way. Softmax's normalising divide **commutes with
the matmul that follows it** —

```text
out[i,:] = SUM_j (p[i,j] / S_i) * v[j,:]  ==  (SUM_j p[i,j] * v[j,:]) / S_i
```

— so it moved past `attn.v` to divide the `(1024, 64)` output instead of the
`(1024, 1024)` scores: **786 K divides instead of 12.6 M**, and one 100 MB round
trip per layer deleted. And the connector's pixel shuffle, written as
reshape/transpose/reshape/transpose, was **two** generic strided permutes
performing a single permutation; composed into one pass it is bit-identical.

| win | measured |
|---|---:|
| bias fusion (4 sites) | **1.133x** tower, **1.172x** on a whole caption in production config |
| deferred normalisation | **1.058x**, reproduced exactly twice |
| fused pixel shuffle | **1.92x** on the op, **bit-identical** |
| **composed** | **1.199x — 16.6 % off the vision tower** |

The arithmetic closes: after fusing, the four projections measure **fc1 563,
fc2 537, qkv 535, out_proj 500 GF/s** — the isolated rate, recovered.

**Every verdict here is an interleaved in-process A/B**, both arms in one
binary, alternated ABBA, against a null arm that computes the identical tensor.
This box has read the same code at 8172 and 14382 ms in one session — a **1.76x
spread** — so a before/after across two builds measures the afternoon, not the
change. The tower A/Bs were then re-run through the real engine
(`examples/caption_arm_ab`), because on 24 cores `tile_workers` yields 6 and the
engine sets `kernels_parallel = workers*6 <= cores` = **false**: production runs
six concurrent towers with these kernels *serial*, which is not the regime a
single-tower benchmark measures.

**Where a caption's time goes now** (`examples/stage_split`, min of 3, warm):

| stage | time | share |
|---|---:|---:|
| preprocess | 52 ms | 0.5 % |
| **vision (tower)** | **8172 ms** | **74.8 %** |
| assemble | 4 ms | 0.0 % |
| prefill | 1283 ms | 11.8 % |
| generate | 1407 ms | 12.9 % |
| **total** | **10 918 ms** | |

Vision's share rose from 58 % to 75 % because the text side got three times
faster, not because vision got slower. It is now the whole of the remaining gap
to PyTorch, and its layer is **77 % matmul** — the elementwise phase that
dominated round 1 has been spent.

**What was tried and refuted**, so it is not re-litigated. The vision tower has
now rejected the same idea eight ways, and the pattern is worth more than any
individual result: **candle's GEMM rewards large batched calls, and every
attempt to trade call size for cache locality has lost.**

| attempt | verdict |
|---|---:|
| attention one head at a time (4.2 MB of scores, not 50.3) | **0.888x** |
| blocked attention, query-block 128 / 256 / 512 | **0.732 / 0.824 / 0.926x** |
| dropping a vestigial batch-of-1 to rank 3 | **0.74x** |
| batching all 17 tiles into one pass | **1.09x** — needs to beat the **2.50x** tile concurrency it would replace |
| a hand-written blocked `q.k^T` kernel | **0.11x** — 9x slower than candle |
| pre-transposing k so the GEMM gets a contiguous operand | 1.02x, under the noise floor |
| `kernels_parallel = 1` in production | **0.932x** — the shipped `workers*6 <= cores` heuristic is right |
| tile workers 4 / 8 against the shipped 6 | **0.932 / 0.999x** |

Also refuted: replacing candle's softmax (four attempts), fusing q/k/v under
GQA, pre-transposing weights at load, and a fused LayerNorm — which moves a
third of candle's bytes and is still not faster, because the 3.1 MB activation
fits L3, so candle's extra passes are cache hits rather than trips to DRAM. It
read **1.09x on one run and 0.89x on the next**; a verdict that changes sign
between runs is the instrument, not the code.

The wins all have the opposite shape: none restructures a GEMM, each deletes a
**separate pass** beside one. All measured, all in
[`docs/plans/argus-launch-plan.md`](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/plans/argus-launch-plan.md)
§19-27.

**The four-gate verdict** (`ffai bench vlm`, both arms, no SKIPs — a skipped
gate is never a pass), measured **2026-08-21** — see the note below, which
qualifies every row of it:

| gate | verdict | measured |
|---|---|---|
| correctness | **PASS** | caption byte-identical; 32/32 tokens |
| quality | **PASS** | exact tie — **49/50** answers byte-identical on OCRBench-lite |
| footprint | **PASS** | 1309 MiB steady vs 1852 — **0.71x** |
| speed | **PENDING RE-RUN** | last corrected run **0.969x**; the stale 2.4x below is not current |

> ⚠ **The speed row is stale and is not the current number.** The 2.4x predates
> every optimization round on this page.
>
> **The harness defect that used to block the re-run is fixed.** It was reported
> here as "`ffai bench vlm`'s engine arm segfaults on the second
> `describe_image` call in one process". The cause was never in Argus: it was
> `rusty_alloc` **0.3.2**, whose use-after-free reproduces on every target. On
> the pinned **1.1.4** the engine arm now completes a **50-item corpus in one
> process** without crashing. Six further harness defects were fixed alongside
> it — an unpinned token budget that had the engine generating 256 tokens
> against the reference's 64, a speed gate that compared across *different
> decode configs*, no warm-up (so model load landed inside run 1), and a
> per-item timer that excluded image decode in our favour.
>
> A corrected 50-item run put the speed gate at **0.969x** and end-to-end at
> **1.013x** with quality an exact tie — but that run predates the round 4-5
> vision work above, and re-running it on a quiet box is pending. **Neither
> 2.4x nor 0.969x is this build's number**, and rather than print an
> extrapolation, this row stays marked pending until the harness has actually
> been re-run. The vision figures above are separately measured and stand on
> their own instruments.

**Verdicts here are counters, not milliseconds, wherever that is possible.**
This box has been measured at a **4x spread** within one configuration, so
[`src/cost.rs`](https://github.com/Remade-With-Rust/FFAI/blob/master/crates/ffai-argus/src/cost.rs)
counts matmul FLOPs and calls, elementwise visits, scalar vs vectorised
transcendentals, bytes moved and layout copies — every one exactly reproducible
on any machine under any load. A win is a counter that went down. The timings
on this page were taken with the machine deliberately quiesced and every figure
repeated.

---

## What is this?

`ffai-argus` captions images and video in pure, safe Rust. It is a
reimplementation of `SmolVLM-256M-Instruct`'s inference path on candle — the
vision tower, the connector, the chat template, the tiling and the greedy decode
loop — not a binding to the Python original and not an ONNX export of it.

The bar it was built to is unusual and worth stating: **an image pipeline that
is visually indistinguishable from the reference is not good enough.** Our first
Lanczos resampler matched PIL to within a single quantisation level everywhere,
about 50 dB SNR, and produced **8 of 32** correct tokens. The story of why is
[below](#the-one-quantisation-level-story), and it is the reason every gate here
is token equality rather than a tensor tolerance.

**Most users want the whole toolkit — [FFAI](https://github.com/Remade-With-Rust/FFAI)
and its `ffai` binary.** Depend on this crate directly when captioning is the
only component you need: it brings the tower, the connector, the prompt
assembly, the decode loop and the content path, and nothing else.

## The Remade With Rust ecosystem

<!-- ORG BOILERPLATE — keep identical across repos -->

**Remade With Rust** is an initiative by **[Mata Network](https://www.mata.network/)**
to rebuild essential C and C++ tools in Rust — for the memory safety, the
predictable performance, and the freedom of a permissive license. Each project
is a reimplementation, not a fork: same wire protocols and file formats, new
code you can actually depend on.

We build the core to production grade and open-source it so the community can
extend it. No copyleft. No surprises. Just the tools we rely on, made faster and
safer.

| Project | What it is |
|---|---|
| 🎬 **[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)** | **Our FFmpeg alternative.** Drop-in `ffmpeg` and `ffprobe` binaries — demux → decode → filter → encode → mux, rebuilt as composable Rust crates with **zero GPL/LGPL**. Apache-2.0. `rusty_h264` is its H.264 codec. |
| 🧠 **[FFAI](https://github.com/Remade-With-Rust/FFAI)** | **Our sister project: media *for* AI.** "The AI media toolkit, remade with rust." Embedded ASR + TTS (**Mercury**), OCR (**Carmenta**) and vision-language captioning (**Argus**) behind an ffmpeg-style, swap-by-name architecture — no Python, no CUDA. MIT OR Apache-2.0. |
| 🌐 **[Mata Network](https://www.mata.network/)** | **The home page.** *"Stop sacrificing your privacy for convenience."* Sovereign, self-hostable privacy infrastructure — wallet & identity, password manager, contact manager, and a browser extension that stops information leaking as you browse. Remade With Rust is its open-source arm. |

→ All projects: **[github.com/Remade-With-Rust](https://github.com/Remade-With-Rust)**

<!-- /ORG BOILERPLATE -->

## Install

```sh
cargo add ffai-argus ffai-core ffai-media
```

or in `Cargo.toml`:

```toml
[dependencies]
ffai-argus = "0.6"    # the engine
ffai-core  = "0.7"    # the VlmEngine trait, the registry, the cost counters
ffai-media = "0.6"    # decoding an image/video file is the CALLER's job
```

`ffai-media` is separate on purpose: the engine takes an already-decoded
`ImageBuffer`, so nothing forces an image-codec dependency on a caller who
already has pixels.

| Crate | Role | Docs |
|---|---|---|
| [`ffai-argus`](https://crates.io/crates/ffai-argus) | **this crate — the VLM engine** | [docs.rs](https://docs.rs/ffai-argus) |
| [`ffai-core`](https://crates.io/crates/ffai-core) | engine traits, registry, cost counters, `fastmath`/`fastops` | [docs.rs](https://docs.rs/ffai-core) |
| [`ffai-media`](https://crates.io/crates/ffai-media) | image/video decode, frame streaming | [docs.rs](https://docs.rs/ffai-media) |
| [`ffai-models`](https://crates.io/crates/ffai-models) | manifest-driven checkpoint resolution | [docs.rs](https://docs.rs/ffai-models) |
| [`ffai-cli`](https://crates.io/crates/ffai-cli) | the `ffai` binary | [docs.rs](https://docs.rs/ffai-cli) |

## Quick start

```rust
use ffai_core::engine::{VlmEngine, VlmOptions};
use ffai_argus::SmolVlm;
use std::path::Path;

let engine = SmolVlm::with_manifest_dir("models".into());
let image  = ffai_media::load_image(Path::new("street.jpg"))?;   // JPEG or PNG
let caption = engine.describe_image(&image, &VlmOptions {
    prompt: Some("What is written in this image?".into()),
    ..VlmOptions::default()
})?;
println!("{caption}");
```

This example is **compiled**, not transcribed:
[`examples/quickstart.rs`](https://github.com/Remade-With-Rust/FFAI/blob/master/crates/ffai-argus/examples/quickstart.rs) is type-checked against the
real API by `cargo build --examples`, because a published README is permanent.

Or through the registry, which is how the `ffai` binary reaches it — engines are
swapped by name, ffmpeg-style:

```rust
let mut reg = ffai_core::registry::EngineRegistry::new();
ffai_argus::register(&mut reg);
let engine = reg.vlm(Some("smolvlm"))?;
```

Command-line:

```sh
ffai caption -i street.jpg --prompt "What is written in this image?"
ffai caption -i street.jpg --seed 42 --temperature 0.7 --top-p 0.9
ffai caption -i clip.mp4 --fps 2 --window 8 --output captions.srt
```

**Decoding is greedy by default and deterministic without a seed.** Sampling is
opt-in and *requires* `--seed`: passing `--temperature` alone is refused rather
than quietly handing you a caption you cannot reproduce.

## Gated against the reference, stage by stage

The engine reports `stable` in `ffai engines`, which in this tree means
*oracle-gated against a reference implementation*. Each stage was gated in
isolation first, so a mismatch names the stage rather than the pipeline:

| stage | gate | result |
|---|---|---|
| vision tower + connector | reference `pixel_values` in, tensors compared | **104.8 / 113.2 dB** |
| prompt assembly | token ids, and the spliced embedding tensor | **1142/1142 ids**, bit-exact splice |
| decode loop | greedy tokens, reference embeddings in | **32/32 identical** |
| tower → assembly → decode | greedy tokens, reference `pixel_values` in | **32/32 identical** |
| **the whole content path** | raw image in — our resize, tiling, tower, assembly, decode | **32/32 identical** |
| **`describe_image`** | the public surface: `ImageBuffer` in, `String` out | **byte-identical caption** |

The last two rows are the ones that matter. Everything above them can pass while
the engine still captions the wrong thing, because they are fed the reference's
own intermediate tensors.

## The one-quantisation-level story

The content path was the hard part, and it is worth stating plainly because the
lesson generalises past this crate.

Our first Lanczos resampler matched PIL's **to within a single quantisation
level** — `7.843e-3` in `[-1,1]` units, exactly `1/255`, everywhere. Visually
indistinguishable. ~50 dB SNR. By any tolerance an image pipeline would
normally apply, correct.

It produced **8 of 32** correct tokens.

The vision tower's own error, `2.06e-4`, flips nothing across the same 32
argmaxes. Preprocessing's `7.8e-3` — about 40× larger — flips a token at step 5.
Same pipeline, qualitatively different outcome, and **no tensor tolerance could
have told us which side of the line we were on.** Only token equality could.

The fix was to stop approximating PIL and implement it: `i32` coefficients
scaled by `1<<22`, integer accumulation seeded to round rather than truncate,
and the detail that carries most of the difference — **a `u8` intermediate
between the horizontal and vertical passes.** PIL resamples into an 8-bit image
and then resamples *that*, so the round-off happens twice, on purpose. Staying
in `f32` between the passes is "more accurate" and gives a different picture.

`resize_oracle.rs` gates that claim directly: `u8` in, `u8` out, against PIL's
own output, **zero differing pixels in both directions**.

## Video

```sh
ffai caption -i clip.mp4 --fps 2 --window 8 --output captions.srt
```

Frames are sampled uniformly at `--fps`, captioned `--window` at a time, and
emitted as a timed track (`.srt` / `.vtt` / `.json`, or timestamped text). The
CLI drives the engine one window at a time, so **peak memory is a function of
the window, not of the length of the clip.**

**Tile splitting is off for video, and that is what makes it work.** A still is
17 tiles — 1088 image tokens — and the text tower holds 8192, so a split window
caps at seven frames. Unsplit, a frame is one tile at 64 tokens and the same
budget holds a hundred. The unsplit tile is *exactly* the global thumbnail the
still path already produces, so the video path inherits that path's bit-exact
oracle gate rather than needing one of its own.

**No video quality claim is made.** `SmolVLM-256M-Instruct` is an image model
with no temporal training and no published Video-MME or MVBench row — its
captions say "the image shows", because that is what it sees. What *is* gated
is the track: windows tile the timeline with no gaps or overlap, the remainder
window is never dropped, `--window 1` degenerates exactly to per-frame, an
empty clip is an empty track rather than an error, and an oversized window is
refused in milliseconds by a geometry check that names the knob.

## What the content path handles

`Rgb8` passes through, `Gray8` is replicated across three channels (`SigLIP` has
no grayscale variant), and `Rgba8` **drops** alpha rather than compositing it —
compositing needs a background colour, and inventing one changes the picture.
The test asserts the consequence rather than the code: an opaque RGBA image must
caption *identically* to its RGB original.

Non-square images get their own gate, because every fixture in the crate is
square — which makes `rows == cols` and hides any place the two are swapped. A
transposed tile grid does not error; it produces a valid prompt of the right
token count describing the image sideways.

## Architecture

3 487 lines of source, 56 tests, no `build.rs`, no vendored model code.

```
crates/ffai-argus/src/
  lib.rs         the registry hook and the public re-export (SmolVlm)
  engine.rs      the VlmEngine impl — ImageBuffer in, String out; tile workers,
                 video windowing, stop-sequence truncation
  preprocess.rs  image -> pixel_values: PIL-exact fixed-point Lanczos (rayon,
                 both passes), Idefics3 AnyRes tile geometry
  vision.rs      SmolVLM's vision tower + the pixel-shuffle connector, on candle
  siglip.rs      OUR SigLIP encoder — same maths, fewer passes over memory:
                 fused QKV with the attention scale folded into the q weights at
                 load, a one-copy layout, a zero-copy GELU CustomOp1 with AVX2
                 dispatch
  prompt.rs      sequence assembly — image + question -> the exact token
                 sequence the model was trained on (chat template, image splice)
  text.rs        OUR SmolLM2 text tower — causality fused into the softmax
                 (deleting candle's 3424 ms `masked_fill`), the attention scale
                 folded into q's weights, a one-pass SwiGLU and a parallel
                 zero-copy RMSNorm
  decode.rs      the decode loop — inputs_embeds -> tokens, greedy, KV-cached
  cost.rs        re-export of ffai_core::cost — the deterministic counters

crates/ffai-argus/examples/
  quickstart.rs         the README's example, compiled
  stage_split.rs        where a caption's wall-clock actually goes
  cost_report.rs        the deterministic counters for one caption
  vision_ops_probe.rs   one encoder layer priced at the real shapes
  kernel_ab.rs          our kernels vs candle's, op by op
  gemm_probe.rs         candle's GEMM against PyTorch's
  tile_parallel_ab.rs   tile-worker scaling
  tile_batching_ab.rs   the batching hypothesis (refuted: 1.07x, +384 MiB)
  build_inputs_embeds.rs
```

The `SigLIP` encoder is ours rather than candle's because half of every layer
was elementwise and layout work running single-threaded (see
[Performance](#performance)). The maths is unchanged and gated on it: the tower
still matches the reference at 104.8 dB and the caption is still byte-identical.

## Why candle rather than mistral.rs

House doctrine says *don't hand-roll an LLM **serving** loop on raw candle* —
paging, quantization, sampling and constrained decoding are solved there. That
rule is about serving. This is a greedy prefill plus one-token-at-a-time decode
for a single sequence, which candle supports directly: `models::llama` **is**
`SmolVLM`'s text tower (its `text_config.model_type` is literally `llama`), and
`forward_input_embed` takes injected embeddings — exactly what a VLM needs and
what `forward` cannot do.

The decisive constraint was publication: `mistralrs` is on crates.io at 0.8.1,
but the version proven to serve `SmolVLM` is a **git revision**, and
`cargo publish` refuses a git dependency. **mistral.rs is not rejected** — it
remains the documented path for the serving concerns it owns, and the
`mistralrs-backend` feature is reserved for it.

<sub>**And the alternative tensor stack was evaluated, not assumed away.**
[`trustformers`](https://github.com/cool-japan/trustformers) is on crates.io
(0.2.0) and does ship a Llama implementation — two claims to the contrary were
made during evaluation and both were wrong, so they are corrected here. The
surviving objection is measured rather than architectural: its GEMM runs ~23×
slower than candle's, and candle's is already at PyTorch parity.</sub>

## Weights

Resolved through the `ffai-models` manifest seam
(`models/smolvlm-256m-instruct.toml`), never a hardcoded cache path, and shared
with the Hugging Face cache so a checkpoint `transformers` already downloaded is
not downloaded twice. Loading is lazy — registering the engine does not read a
gigabyte of safetensors.

`SmolVLM-256M-Instruct` is Apache-2.0.

## Platform support

| Platform | Status |
|---|---|
| Windows (x86-64) | ✅ builds + tests |
| Linux (x86-64) | ✅ builds + tests |
| macOS | ✅ builds + tests |
| `wasm32-unknown-unknown` | ❌ not supported — see below |

CPU inference, one machine, no CUDA and no Metal required. AVX2+FMA kernels are
selected at **runtime** through `is_x86_feature_detected!`, each with a scalar
twin that stays the oracle and the fallback, so a published binary still runs on
a machine without AVX2.

**Not wasm.** A 256M-parameter checkpoint and a 17-tile tower is not a browser
workload; FFai's wasm story is [`ffai-wasm`](https://github.com/Remade-With-Rust/FFAI/tree/master/crates/ffai-wasm) (Diana / YOLO26) and
[`ffai-carmenta-wasm`](https://github.com/Remade-With-Rust/FFAI/tree/master/crates/ffai-carmenta-wasm) (OCR). Note that candle's
`onig_sys` C dependency is target-gated out on `wasm32` — the barrier here is
the model, not the toolchain.

## Roadmap

- [x] `ffai-bench` VLM harness — `run_vlm`, the four-gate verdict, the ledger record
- [x] VLMEvalKit as a scoring adapter — **525/1000 OCRBench against a published 526**
- [x] Model selection by port-cost triage — `SmolVLM-256M-Instruct`, pixel-shuffle connector
- [x] Trait surface — `Decoding` makes unseeded sampling unrepresentable; `VlmPrompt`/`VlmPart` carry interleaved multi-image
- [x] **Vision tower + connector**, oracle-matched — 104.8 dB / 113.2 dB
- [x] **Prompt assembly** — 1142/1142 ids, bit-exact embedding splice
- [x] **Decode loop** — candle `llama` with KV cache, 32/32 tokens
- [x] **Content path** — PIL-exact fixed-point Lanczos, AnyRes tiling, `resize_oracle` at zero differing pixels
- [x] **`describe_image` end to end — byte-identical caption**, 49/50 on OCRBench-lite
- [x] **`describe_video`** — frame sampling → windowed captions → `.srt`/`.vtt`/`.json`, six structural gates
- [x] **Streaming frame ingest** in `ffai-media` — and a `stream_frames` defect found and fixed along the way: it returned one frame per clip at any fps
- [x] **Vision speed campaign** — our `SigLIP` encoder, six-way tile parallelism, zero-copy AVX2 GELU: **2.84× tower / 1.86× caption**, gates unchanged
- [x] **Deterministic cost counters** (`src/cost.rs`) — a win is a counter that went down
- [x] **Our own `SmolLM2` text tower** — causality fused into the softmax, the attention scale folded into q, a one-pass SwiGLU: **prefill 3.07x, generate 2.05x**, caption still byte-identical
- [x] **Two `broadcast_matmul` traps removed from vision** — the connector projection (**14.1x**) and the patch embedding as a matmul (**2.6x**)
- [x] **Stage-by-stage measurement against PyTorch** — the gap is **1.20x** and it is entirely the vision tower
- [ ] **Fix `ffai bench vlm`'s engine arm**, which segfaults on the second `describe_image` in one process — this blocks re-running the four-gate verdict
- [ ] **Re-run the four-gate verdict** so the speed row stops being a 2026-08-21 number
- [ ] Close the vision gap — the layer is now **77 % matmul**; tile batching is bounded at ~10 % and peaks at chunk 4
- [ ] `mistralrs-backend` — Qwen-VL / LLaVA-class models, once a crates.io release serves them
- [ ] A checkpoint with real temporal training, so a video quality number can be earned rather than invented
- [ ] Interleaved multi-image prompts through the public surface (the trait already carries them)

## License

MIT OR Apache-2.0, at your option. No copyleft anywhere in the dependency tree.
Model weights carry their own licences, surfaced at selection time;
`SmolVLM-256M-Instruct` is Apache-2.0.

## About Mata Network

<!-- ORG BOILERPLATE — keep identical across repos -->

**[Mata Network](https://www.mata.network/)** builds sovereign, self-hostable
privacy infrastructure — *"stop sacrificing your privacy for convenience"*:
wallet & identity, a password manager, a contact manager, and a browser
extension that stops your information leaking as you browse.

**Remade With Rust** is our open-source home for the permissively-licensed
building blocks that work depends on — including
[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs) (the
FFmpeg alternative) and [FFAI](https://github.com/Remade-With-Rust/FFAI) (the
AI media toolkit).

→ **[www.mata.network](https://www.mata.network/)**

<!-- /ORG BOILERPLATE -->
