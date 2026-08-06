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

## Depth: metric distance per pixel

The same backbone and neck, a different head. `yolo26{n,s,m,l,x}-depth.pt`
converted the same audited way, giving a dense map in **metres**.

```rust
use ffai_core::engine::{DepthEngine, DepthOptions};
use ffai_diana::{depth_engine::Yolo26Depth, image::Geometry};

let engine = Yolo26Depth::build("n", Geometry::Rect, "models");
let out = engine.depth(&image, &DepthOptions::default())?;
let (near, far) = out.range().unwrap();
println!("{}x{} map, {near:.2}-{far:.2} m", out.width, out.height);
```

```
ffai depth -i street.jpg -o depth.bin      # raw f32 metres, row-major
ffai depth -i street.jpg -o depth.png      # 16-bit grayscale, normalised
```

Output is `(1, H/4, W/4)`, unbounded — `exp` of a clamped logit rather than a
scaled sigmoid, which is what lets one model span an indoor corridor and an
outdoor street. `--full-res` maps it back onto the source image.

**Gated at all five tiers against Ultralytics**, per-pixel, through the engine
rather than the bare graph:

| tier | worst relative error | range on the fixture |
|---|---:|---|
| n | 4.05e-6 | 1.48–14.11 m |
| s | 4.46e-6 | 1.08–16.90 m |
| m | 7.40e-6 | 1.35–23.51 m |
| l | 5.54e-6 | 1.47–25.79 m |
| x | 6.19e-6 | 1.45–31.83 m |

No `ffai bench depth` line yet, so **no speed or memory claim for depth** —
only this correctness one, which is the stronger statement anyway: a
ground-truth metric would grade Ultralytics' weights, while this grades the
port.

## Faster than Ultralytics, measured the hard way

**1.069x on 1080p frames — Diana faster in 25 of 32 paired runs, z = +3.18.**
Solo passes, ABBA-alternated at the pass level, both engines decoding their own
JPEGs, median of 60 frames per pass. The ratio was watched as N grew, because a
cross-implementation number that is still moving has not been measured yet:

| N | median | paired |
|---:|---:|---|
| 8 | 1.035x | 5/8, z = +0.71 |
| 16 | 1.045x | 11/16, z = +1.50 |
| 24 | 1.070x | 18/24, z = +2.45 |
| **32** | **1.069x** | **25/32, z = +3.18** |

Stable from N=24. At 640 on the COCO holdout the same comparison reads
**0.70x and 0.58x** across two paired runs — ahead by more, on smaller frames.

### The harness shape is a bigger variable than the engines

Five arrangements, **identical work**, only the order and co-residency changed:

| arrangement | ratio | vs solo |
|---|---:|---:|
| **solo, ABBA** | **1.077x** | 1.00x |
| alternating per frame | 1.031x | 0.96x |
| block-wise (all A then all B) | 1.417x | **1.31x** |
| other engine resident but idle | 1.284x | 1.19x |
| frames shuffled | 1.405x | **1.30x** |

**A 31 % swing from arrangement alone**, against a ~7 % effect. Two of those
arrangements flatter Diana badly — block-wise most of all, which is exactly the
shape `codec-measurement` §3 forbids. Only **solo** reproduces across runs
(1.069x at N=32, 1.077x at N=5), so that is the only one quoted here.

**Memory stays the unambiguous win: 0.4x Ultralytics, 0.75x ONNX Runtime.** And
on CPU-seconds rather than wall, Diana does the same work for roughly **a third
of the machine** — the figure that matters on a host packing many streams.

## The codec stack is ours, and it is not a toy

Diana's pipeline decodes with Remade-With-Rust codecs end to end — no libjpeg,
no libpng, no libavcodec, no OpenCV:

| format | crate | measured |
|---|---|---|
| JPEG | `rusty_jpeg` 0.3.2 | **7.05 ms** at 1080p — **1.14x** libjpeg-turbo's 6.18 |
| PNG | `rusty_png` 0.3.2 | gated against upstream `png`, byte-identical |
| H.264 | `rusty_h264` 0.8.0 | **22.72 ms/frame** at 1080p, full-file decode |

A pure-Rust JPEG decoder within 14 % of libjpeg-turbo is the number worth
pausing on — that is a C library with two decades of hand-written SIMD in it.

**H.264 went from unusable to working in one version bump.** The 0.2.1 the
dependency graph was pinned to could not decode x264's DEFAULT profile at all —
0 of 164 frames, silently, because CABAC entropy decoding was broken:

| | rusty_h264 0.2.1 | **0.8.0** |
|---|---:|---:|
| CAVLC | 164/164 | 164/164 |
| CABAC | 49/164 | **164/164** |
| x264 default (High) | **0/164** | **164/164** |
| 1080p decode | 47.50 ms | **22.72 ms** |

## Video ingest: the decoder is ready, the API is not

Stated plainly because it is the difference between a demo and a deployment.

**What works:** `ffai_media::sample_frames` demuxes MP4 with `rff-format-mp4`
and decodes with `rusty_h264` 0.8, handling everything x264 emits by default.
Decode errors propagate with the packet index and the decoder's own message —
they used to be discarded, which turned a normal file into zero frames and no
diagnostic.

**What does not:** it returns `Vec<VideoFrame>` — **every frame in memory at
once**. One minute of 1080p is 10.4 GiB; ten minutes is 104 GiB. There is also
no CLI path (`ffai detect` takes an image or a directory of frames), and it has
only been exercised against MP4/H.264 from x264.

**So: not deployable for streaming video yet.** It needs an iterator API that
yields frames as it decodes, and a CLI verb to drive it. The decoding half of
that work is done and measured; the plumbing half is not.

## Video in, not just stills

`ffai_media::sample_frames` decodes H.264/MP4 through the pure-Rust `rff`
stack — no libavcodec, no OpenCV — and hands back RGB frames on a fixed
stride. The YUV→RGB conversion reproduces OpenCV's `COLOR_YUV2RGB_I420`
BT.601 matrix exactly, because a benchmark against a Python reference that
converts colour differently is measuring the colour conversion.

```rust
let frames = ffai_media::sample_frames(Path::new("clip.mp4"), 8)?;  // every 8th
```

## LIVE: skip frames that did not change

```
ffai detect --live -i frames/          # a directory, sorted by name
```

A frame whose pixels have not moved reuses the previous detections at zero
model cost — which also makes it an output stabiliser, since nothing re-rolls
the model on a static scene.

**Read the qualifier before enabling it.** On a genuinely still scene the
saving is large; on real footage with people in it, it is close to nothing:

| content | frames gated |
|---|---|
| fixed camera, still scene, sensor noise only | **46 of 48** |
| MOT17 static-camera sequences (02, 04, 09) | **41 of 2,175 — 1.9 %** |
| MOT17 moving-camera sequences | **4 of 3,141** |

The gate is for a **static SCENE**, not merely a fixed CAMERA. A surveillance
view with pedestrians crossing it changes every frame, and the gate correctly
refuses — costing a per-frame pixel diff and saving nothing. Across all 5,316
MOT17 frames it fired on **0.8 %**, at an accuracy cost of **−0.006 pp**.

Its failure mode is not graceful. Forced to gate 507 of 525 frames on a scene
that WAS changing, AP50 fell from 62.35 % to 17.15 % — **45 points**. The
threshold is a correctness boundary, not a tuning knob, which is why the
per-pixel delta is set from a harvest on real compressed video rather than
chosen.

## Measured on MOT17 — a public benchmark, not our own corpus

All seven MOT17 training sequences, **5,316 frames**, ground truth from the
dataset. Diana and Ultralytics are handed the identical extracted frames, so
neither pays for a decode the other does not.

| seq | camera | frames | Diana AP50 | ultralytics | gap |
|---|---|---:|---:|---:|---:|
| 02 | static | 600 | 21.55 % | 21.54 % | +0.01 |
| 04 | static | 1050 | 23.07 % | 23.06 % | +0.01 |
| 05 | moving | 837 | 56.14 % | 56.04 % | +0.10 |
| 09 | static | 525 | 62.35 % | 62.37 % | −0.02 |
| 10 | moving | 654 | 35.92 % | 35.95 % | −0.03 |
| 11 | moving | 900 | 56.96 % | 56.92 % | +0.03 |
| 13 | moving | 750 | 25.62 % | 25.62 % | −0.00 |

**Mean absolute gap: 0.029 pp**, across scenes spanning 21.55 % to 62.35 % —
so the agreement is not an artefact of one easy sequence. Ahead on four,
behind on three, every one inside 0.1 pp.

Reproduce with `tools/diana_mot_bench.py --all`.

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

## The allocator knob, measured — and why it is not the default

Diana's slow frames carry a page-fault spike: **~31 faults on a normal frame,
4,200 on a slow one.** `MIMALLOC_PURGE_DELAY=-1` removes it completely — faults
go flat and halve overall. mimalloc reads that variable itself, so it needs no
code from us and no rebuild.

It is **not the default**, and the reason is that the fault count does not
convert into time:

| | CPU ms/frame | wall ms/frame | steady RSS |
|---|---:|---:|---:|
| default | 130.2 | 66.7 | **182 MiB** |
| `MIMALLOC_PURGE_DELAY=-1` | 125.9 | 68.9 | **239 MiB** |

Same binary, ABBA-interleaved, 8 reps, work parity constant at 401 detections.
**CPU 0.968x, wall 1.033x** — both inside this box's 10.4 % null-arm floor, and
wall is marginally *worse*. The footprint number is not inside anything:
**+57 MiB, +31 %**, measured by `ffai bench detect` — the same instrument the
footprint gate scores, against ONNX Runtime's 161 MiB.

So the trade is 57 MiB for nothing the clock can see. The faults are real work
removed; they are **soft** faults against pages still resident in RAM, which is
why removing thousands of them per frame is worth ~0.2 ms of a 60 ms frame.

There is a sharper reason to leave it off. Those faults come from the OS
trimming our working set, which happens **under system memory pressure** — so
the setting does least when the machine is idle and safe, and most when the
machine is already short of memory and retaining 57 MiB is the worst available
response. It helps least where it is safe and most where it is dangerous.

Turn it on if you have measured your own workload and want flat page behaviour:

```
MIMALLOC_PURGE_DELAY=-1 ffai detect -i frame.jpg
```

The full descent, including the two measurements that contradicted each other
and how the contradiction was resolved:
[docs/whys/diana-1080p-and-tail.md](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/whys/diana-1080p-and-tail.md).

## Status: `experimental`, honestly

Measured against Ultralytics 8.4.113 and ONNX Runtime on a hash-pinned
45-image COCO holdout, CPU only, yolo26n at 640 rect
([`bench/ledger.jsonl`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl)):

| | mAP50 | steady RSS |
|---|---:|---:|
| **Diana** (rect) | **0.7014** | **121 MiB** |
| ultralytics-yolo26n-rect | 0.7014 | 310 MiB |
| ort-yolo26n (square only) | 0.6865 | 163 MiB |

**mAP is identical to PyTorch to four decimals**, and memory is the
unambiguous win — **0.4x Ultralytics, 0.75x ONNX Runtime**. Model load is
68 ms.

### The latency number moved, because the harness was wrong

The bench pre-decoded the whole corpus before timing. That looks like it
favours us — our PNG reader sits outside the timed region while the reference
pays for its own decode inside it. It does the opposite. The reference decodes
each image *just before* using it and reads a buffer its own decoder has just
written; under pre-decode ours was written 45 images ago and has to be fetched.
**The harness was handicapping us by ~14 %.** Measured ABBA, three reps,
just-in-time decode is 11–17 % faster despite ADDING the decode to the timed
region — and a working-set sweep shows holding 1 image versus 45 is FLAT, so
the mechanism is recency, not residency. JIT is now the default;
`FFAI_BENCH_PREDECODE=1` restores the old behaviour.

With that fixed, two clean paired runs at rect (both numbers from the same run,
so machine drift cancels):

| | ours | ultralytics | ort |
|---|---:|---:|---:|
| run 1 | 32 ms | 46 ms — **0.70x** | 28 ms — 1.14x |
| run 2 | 37 ms | 64 ms — **0.58x** | 33 ms — 1.12x |

**This is two runs, and it is labelled as two runs.** A third read 382 ms
against ultralytics' 49 — rtf 2.6 where the others sit at 25–29 — on a box
that had been benchmarking for hours; it is excluded as machine noise and the
exclusion is stated here rather than buried in a median. An earlier version of
this page claimed "faster than Ultralytics" on one favourable run and had to
retract it across three published places. **Ahead at rect is the current
reading, not yet a settled result.**

The ORT column is **not like-for-like**: ORT has no rect export, so our
reduced-work rect runs against its full-work square, and rect is 70–75 % of
square's pixels on this corpus. At matched square geometry the last honest
figure was **2.89x behind ORT** (81 ms against 28) — measured under the OLD
harness and **not re-measured since**, so treat it as an upper bound rather
than a current number.

### What moved it, and what did not

p50 went **85 → 41 ms** in one campaign, and the four levers were not the four
anyone would guess:

* **the allocator**, worth 1.64x. 58,634 page faults per image; the system
  allocator was returning memory to the OS and re-faulting nearly every byte;
* **the thread pool**, worth 1.21x wall and 3.5x CPU — one image wants ~4
  workers, not 24, and candle keeps its own pool besides. **4 is the
  latency-optimal width, not the only sensible one**: measured against it,
  2 workers costs 1.35x wall and saves **15 % of the CPU**, and 1 worker costs
  2.0x wall and saves **21 %**. A host packing many concurrent streams should
  measure `FFAI_DIANA_THREADS=2`; a single latency-bound stream wants 4;
* **preprocessing**, 5.7x on its own — a serial bilinear resize recomputing
  the horizontal sample position for every column of every row;
* **epilogue fusion** on both convolution paths, 12.5 % — bias and SiLU in one
  traversal instead of three.

What did **not** move it, each refuted with numbers: im2col tiling (twice),
direct convolution (four shapes, including an AVX2 microkernel), implicit
GEMM, the im2col zero-fill, elementwise traffic fusion, Intel MKL (within
noise of candle's GEMM), thread width beyond 4–6, cache pollution (flat across
44x residency), and content-adaptive dispatch. **Nothing in this graph leaves
L3**, so every prize priced at DRAM bandwidth was overstated by 3–15x.

Not yet `stable`: the footprint gate passes by **1 MiB** (160 against ORT's
161), which is not a margin — mimalloc retains ~130 MiB of allocator churn
against 26 MiB actually live, and the durable fix is upstream of the
allocator. Only COCO's 80 classes are exercised.

Every number traces to a line in
[`bench/ledger.jsonl`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl).
The full campaign, every reverted experiment and every retracted number
included:
[docs/whys/diana-latency.md](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/whys/diana-latency.md).

## License

MIT OR Apache-2.0. **Model weights are not covered by it** — YOLO26
checkpoints are AGPL-3.0 and you supply your own.
