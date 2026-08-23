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

**The four-gate verdict** (`ffai bench vlm`, both arms, no SKIPs — a skipped
gate is never a pass), measured **2026-08-21** against `transformers` on CPU
running the identical checkpoint and decode config:

| gate | verdict | measured |
|---|---|---|
| correctness | **PASS** | caption byte-identical; 32/32 tokens |
| quality | **PASS** | exact tie — **49/50** answers byte-identical on OCRBench-lite |
| footprint | **PASS** | 1309 MiB steady vs 1852 — **0.71×** |
| speed | **FAIL** | 0.08 vs 0.20 it/s — **2.4× slower** |

**Then the speed campaign ran** (2026-08-22). Vision was 77–82 % of a caption;
min-of-4, ABBA-interleaved, same image, every gate unchanged (32/32 tokens,
caption still byte-identical):

| | before | after | |
|---|---:|---:|---:|
| vision tower | — | — | **2.84×** |
| **whole caption** | **27 213 ms** | **14 627 ms** | **1.86×** |

> ⚠ **The four-gate table above has not been re-run since.** The 2.4× is a
> 2026-08-21 number and the 1.86× landed the day after, so the *current* gap to
> PyTorch is smaller than the table says — but by how much is not measured, and
> this project does not publish arithmetic in place of a measurement.

**Where the time actually goes** (`examples/stage_split`, min of 3, warm). The
instrument exists because a per-tile *work model* was once used to claim vision
was no longer the majority, and that claim was wrong:

| stage | time | share |
|---|---:|---:|
| preprocess | 54 ms | 0.4 % |
| **vision (tower)** | **8452 ms** | **58.0 %** |
| assemble | 3 ms | 0.0 % |
| prefill | 3870 ms | 26.6 % |
| generate | 2181 ms | 15.0 % |
| **total** | **14 561 ms** | |

**What the campaign found, and what it refuted.** One encoder layer at real
shapes costs 61.9 ms — **50 % matmul, 50 % everything else**. The matmuls were
never the problem: candle's GEMM runs **589–702 GF/s against PyTorch's
680–697** — parity. The cause was one line of candle's CPU backend, which calls
rayon for **`conv2d` and nothing else**, so for half of every layer a 24-core
box ran one core:

| op, real shape | candle | ours (parallel) | ours (serial) |
|---|---:|---:|---:|
| GELU `(1,1024,3072)` | 44.01 ms | **3.03 ms (14.5×)** | 8.37 ms (5.3×) |
| softmax `(1,12,1024,1024)` | **4.71 ms** | 11.74 ms (0.40×) | 50.79 ms (0.09×) |
| `layer_norm (1,1024,768)` | **0.64 ms** | 0.70 ms (0.91×) | 1.35 ms (0.47×) |

Read that table the way it was eventually read: **a win on one op is not a
licence to rewrite its neighbour.** Replacing candle's GELU won 14.5×;
replacing its softmax lost, four attempts running, and was reverted
permanently. Blocked/flash attention was refuted at 0.23–0.51× while
bit-identical. Tile batching measured 1.07× and cost +384 MiB against a
footprint gate we pass. Those refutations are recorded in
[`docs/plans/argus-launch-plan.md`](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/plans/argus-launch-plan.md)
§19–20 so they are not re-litigated.

**Verdicts here are counters, not milliseconds, wherever that is possible.**
This box swings ±12 %, which is larger than most wins worth having, so
[`src/cost.rs`](https://github.com/Remade-With-Rust/FFAI/blob/master/crates/ffai-argus/src/cost.rs) counts matmul FLOPs and calls, elementwise visits,
scalar vs vectorised transcendentals, bytes moved and layout copies — every one
exactly reproducible on any machine under any load. A win is a counter that went
down.

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
- [ ] **Re-run the four-gate verdict** so the speed column reflects the campaign instead of predating it
- [ ] Close the remaining speed gap — the profile says memory bandwidth and seventeen tower passes, not GEMM
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
