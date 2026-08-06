# rusty_alloc 0.3.1 — reproducible segfault on the JPEG decode path

**For the rusty_alloc team. This is a release blocker for 0.3.1.**
Windows 11 / x86_64-pc-windows-msvc, 16 logical cores, 2026-08-06.

**0.3.0 is clean. 0.3.1 crashes.** We are pinned to `=0.3.0` and staying there
until this is resolved.

## The bisect

Identical FFai build, identical workload, only the allocator version changed.
`detect --track` over 50 frames, 8 runs each:

| version | segfaults |
|---|---|
| 0.1.0-alpha.1 | 0/8 |
| 0.1.0-alpha.2 | 0/8 |
| **0.3.0** | **0/8** |
| **0.3.1** | **6/8** |

## Minimal reproduction

The smallest arm that reproduces has **no model, no tracker, no threading of
our own** — it decodes JPEGs in a loop and does nothing else:

```
cargo build --release -p ffai-diana --example alloc_decode --features alloc-rusty
./alloc_decode corpora/clips/mot17-09/img1 1
```

| arm | segfaults |
|---|---|
| decode 525 JPEGs @ 1920x1080 | **6/6** |
| decode 40 JPEGs @ 640x480 | **6/6** |
| decode 5 JPEGs @ 1920x1080 | **4/4** |
| decode 40 PNGs (small) | **0/4** |
| detect one image x60, no decode | **0/6** |

Five JPEG decodes is enough.

## What it is NOT — each of these was measured, not assumed

- **Not a threading race.** It crashes with `RAYON_NUM_THREADS=1` (5/6). More
  threads made it *less* frequent, not more: 1→5/6, 2→3/6, 4→4/6, 16→1/6.
- **Not allocation size.** 640x480 JPEG buffers crash as reliably as 1920x1080
  ones (6/6 both). ~0.9 MiB and ~5.9 MiB alike.
- **Not the model or tensor churn.** 60 consecutive detects on a preloaded
  image never crash (0/6). Every crash needs a decode.
- **Not simple over-alignment.** A dependency-free repro doing 200 rounds of
  64 K over-aligned (`repr(align(32))`) allocations plus a growth `reserve`
  did not crash on 0.3.1 (0/4), nor did the same at align 1. So it is not
  merely "the `align > 8` path" in `GlobalAlloc`, though something more
  specific about that path is still the leading suspect.

## Where we would look

The discriminator is **PNG decode fine / JPEG decode fatal**, at any buffer
size, single-threaded, within five iterations. Both go through the same
`GlobalAlloc`, so the difference is the *pattern* `rusty_jpeg` produces —
mixed alignments, many small table allocations interleaved with a few large
plane buffers, and realloc-shaped growth — rather than any single call.

The 0.3.0 → 0.3.1 diff is small enough to bisect directly, and 0.3.0 passing
0/8 on the identical workload makes that bisect cheap.

## What we lose by pinning to 0.3.0

Nothing — 0.3.0 is the release that fixed the RSS escalation we reported, and
it holds up:

| | floor | median | max | max/min |
|---|---|---|---|---|
| mimalloc | 77.4 | 97.1 | 118.3 | 1.53x |
| 0.1.0-alpha.1 | 92.0 | 191.5 | 403.2 | 4.38x |
| **0.3.0** | 91.4 | **91.4** | **190.7** | **2.09x** |

Speed at N=21 with a null arm: wall 1.026x, CPU 0.988x against mimalloc.
Detection output byte-identical to the mimalloc build over a 525-frame run.

## One process note

`rusty_alloc = "0.3.0"` is a *caret* requirement, so it silently resolved to
0.3.1 the moment 0.3.1 was published — mid-session, under a lockfile we had
already validated. We shipped a segfaulting default for several commits without
touching the version string, and only caught it because an unrelated end-to-end
check crashed.

We are now pinned with `=0.3.0`. Until 0.3.1 is understood we would suggest
treating 0.3.x as pre-release for downstream consumers, or yanking 0.3.1.
