# rusty_alloc in FFai — evaluation report (0.1.0-alpha.1 and 0.3.0)

**For the rusty_alloc team.** Written 2026-08-06 against `rusty_alloc` and
`rusty_alloc-api` **0.3.0**, on Windows 11 / x86_64-pc-windows-msvc, 16 logical
cores. First measured against 0.1.0-alpha.1; **0.3.0 landed mid-evaluation and
roughly halved both the median and the maximum peak RSS** — that comparison is
§3 and is the most useful thing in this document.

**Headline: it works, it is fast enough to ship, and we have shipped it.** The
`ffai` binary now sets `rusty_alloc_api::RustyAlloc` as its global allocator by
default, replacing `mimalloc`, across FFai and Diana.

There is **one finding for your current RSS-max work** (§3), and one
documentation bug (§4).

This is not a routine dependency swap for us. The allocator is by a wide margin
the **largest single effect in Diana's entire latency campaign**: against the
system allocator, mimalloc measured **1.64×** (74.6 → 45.5 ms per image),
because the system allocator was re-faulting nearly every byte it handed back —
58,634 page faults per image. Nothing else we found came close. So mimalloc was
simultaneously the C dependency we most wanted to remove and the one that cost
the most to remove wrongly.

---

## Method

Applied to every number below: **not** core-pinned (the workload is
rayon-parallel across 28 threads, so single-core affinity would measure a
different program), `High` priority, arms **ABBA-rotated**, CPU time from
`TotalProcessorTime`, peak RSS polled at 150 ms, and a **NULL arm** — mimalloc
against itself under a second label — to establish the resolution floor.

The null arm earned its place. At N=15 it read **1.029×**, which was most of an
apparent 5% rusty_alloc "win". At N=31 it settled to **0.998×, z = −0.90**.

Every arm also reports a **work count** — detection count plus a checksum over
all box coordinates and confidences — because an allocator changes where bytes
live and never what they contain.

Two caveats recorded rather than hidden:

- A neighbouring project's `python.exe` was burning CPU throughout (5,750 CPU-s
  observed mid-session). ABBA plus the null arm absorbed it, and the null floor
  above is the evidence, but absolute wall times are not quiet-box numbers.
- **Five first-pass results in this evaluation did not reproduce and were
  retracted** (§6). They are listed because a report showing only the findings
  that survived tells you nothing about how reliable the surviving ones are.

## 1. Correctness — PASS, unqualified

`rusty_alloc` produced **byte-identical output to mimalloc on every workload**,
verified by SHA-256 of the result file:

| workload | scale | result |
|---|---|---|
| detect, single image | 40–2000 reps | identical detection count + coordinate checksum |
| detect + track, MOT17 frames | 525 frames, 3,918 boxes | identical SHA-256 across 10 runs |
| detect + track, frames | 2,100 frames | identical SHA-256 |
| decode + detect + track, H.264 video | 3,560 boxes | identical SHA-256 |
| JPEG decode only | 1,575 decodes | identical total pixel count |

No crashes, no hangs, no nonzero exits across ~200 process runs, under candle's
tensor churn and rayon's work stealing. For an `alpha.1` allocator that is a
strong result and it is the main thing we want to say.

## 2. Speed — parity

| probe | N | rusty_alloc vs mimalloc |
|---|---|---|
| detect, n tier | 31 rounds | wall **1.015×**, z = +1.98, CPU 1.000× |
| detect, s tier | 9 rounds | wall **1.015×** |
| detect, m tier | 9 rounds | wall **1.017×** |
| decode only, 1,575 JPEGs | 7 rounds | wall **1.020×**, CPU 0.970× |
| CLI end-to-end, 525 frames | 5 rounds | wall **0.988×**, CPU 1.010× |
| CLI end-to-end, 2,100 frames | 1 round | wall 1.000× (98.2 s vs 98.3 s) |

Pooling the 49 paired detect rounds: rusty_alloc wins 33, **z = +2.43**,
against a null arm winning 22 at z = −0.71. A real but small edge — call it
**~1.5%** — so the honest claim is **parity or slightly better**, not a speed
win. Critically, it does not give back the 1.64× that made the allocator matter,
which was the only question that could have blocked the swap.

## 3. Peak RSS — 0.3.0 halved it, and the shape says where the rest is

**There is no growth in either version.** Median peak RSS is flat in workload
length (0.1.0-alpha.1: 92.1 / 92.2 / 92.3 MB at 100 / 500 / 1500 detect reps).
We initially reported unbounded growth extrapolating to ~10 GB/hour; that was
three N=1 points drawn from a multimodal distribution and is retracted (§6).

The quantity that matters is the **distribution**. 15 runs each, 400 detect
reps, ABBA-interleaved, identical work, raw values sorted:

```
mimalloc     77.4 77.5 77.5 77.5 77.5 77.5 79.6 86.9 97.1 | 110.8 110.8 110.8 | 118.2 118.3 118.3
0.1.0-alpha1 92.0 92.1 92.1 92.1 92.2 93.3 | 157.1 157.1 | 191.5 195.7 195.7 195.9 195.9 | 222.0 | 403.2
0.3.0        91.4 91.4 91.4 91.4 91.4 91.4 91.4 91.4 91.4 94.6 | 156.3 156.3 156.3 156.4 | 190.7
```

| | floor | median | max | max/min | runs at floor |
|---|---:|---:|---:|---:|---:|
| mimalloc | 77.4 | 97.1 | 118.3 | 1.53x | 7/15 |
| rusty_alloc 0.1.0-alpha.1 | 92.0 | 191.5 | 403.2 | **4.38x** | 6/15 |
| **rusty_alloc 0.3.0** | 91.4 | **91.4** | **190.7** | **2.09x** | **10/15** |

**0.3.0 halved the median (191.5 -> 91.4) and halved the max (403.2 -> 190.7),
and its median is now BELOW mimalloc's (91.4 vs 97.1).** The 195/222/403 modes
are gone; 10 of 15 runs now sit at the floor against 6 before. Whatever changed
between alpha.1 and 0.3.0 worked.

It cost nothing in speed: at N=21 with a null arm, 0.3.0 reads wall **1.026x**
(z = 1.53) and CPU **0.988x** versus mimalloc, with the null arm at 0.994x
(z = -0.65).

Where the remainder is, if it is worth chasing:

- **The floor is a fixed ~+18%** — 91.4 vs mimalloc's 77.4, stable across every
  run and both versions. Untouched by 0.3.0, so presumably a deliberate arena
  reservation rather than retention.
- **Escalation still exists, in one step.** 5 of 15 runs leave the floor, and
  when they do they land at ~156 or ~190 — a ~70-108% jump, against mimalloc's
  ~43-53% steps (77 -> 111 -> 118). So the remaining lever is the SIZE of the
  first escalation step, not its frequency and not the baseline.
- Both allocators are multimodal here; this is not a rusty_alloc pathology, and
  a fair target is mimalloc's 1.53x spread rather than 1.0x.

**`collect()` did not reclaim any of it** (measured on alpha.1). Three arms, ONE
binary, trim the only variable, ABBA-rotated, N=5 at 1,500 detect reps:

| arm | RSS median | RSS range | latency |
|---|---:|---:|---:|
| mimalloc | 111.1 MB | 106.7-134.2 | 31.97 ms |
| rusty_alloc, no trim | 195.8 MB | 92.2-403.3 | 30.63 ms |
| rusty_alloc, trim 200 ms | 195.8 MB | 91.2-402.5 | 30.49 ms |

**0.1 MB, 0%.** Worth re-checking on 0.3.0 if the escalation path changed. Our
trimmer therefore ships OFF (`FFAI_TRIM_MS=<ms>` enables it), kept wired only so
re-testing your next release is one env var rather than one patch.

**Context that may matter:** Diana runs 4 worker threads plus candle's 24, and
**none of them ever exit** — a persistent rayon pool. Your M4 notes describe
returning a thread's pages at thread exit and via segment reclaim; on this
workload the thread-exit path never runs. We did not measure that as the cause,
so treat it as where we would look rather than as a finding.

## 4. Documentation bug in `rusty_alloc-api`

**Still present in 0.3.0** (`src/lib.rs:129`), so this is not a stale
complaint about the alpha. The `unsafe impl GlobalAlloc` SAFETY comment ends:

> `…regardless of which thread frees it` **`(M2: one global locked heap)`**

But `rusty_alloc` core's own `lib.rs` says:

> `Milestone status: **M4** — per-thread heaps, lock-free cross-thread frees …`
> **`No global lock anywhere on the alloc/free paths.`**

The api crate's comment is stale by two milestones, in both releases. It cost
us real time: we
read it first, predicted lock contention on a 28-thread workload, and went to
verify the mechanism before measuring anything. For anyone evaluating the crate,
"one global locked heap" reads as close to a disqualifier, and it is not true.

## 5. What went well beyond the numbers

`rusty_alloc` exposes its reclaim as a safe Rust function. The mimalloc
equivalent had to be hand-declared:

```rust
#[allow(unsafe_code)]
unsafe extern "C" {
    fn mi_collect(force: bool);
}
// ...
unsafe { mi_collect(false) };
```

That whole block is now `rusty_alloc::alloc::collect(false)`. One `unsafe
extern` block deleted from our tree — the unglamorous form the pure-Rust
argument usually takes.

One packaging note, also unchanged in 0.3.0: `collect` lives in `rusty_alloc`
core while
`rusty_alloc-api` exposes reclaim only per-`Heap`, so anyone using the api crate
as a global allocator must add the core crate as a second direct dependency to
reach it. Re-exporting a global `collect` from the api crate would remove that.

## 6. Retracted during this evaluation

Recorded because they show what this measurement costs to get right:

| first read | what it became | why it was wrong |
|---|---|---|
| rusty_alloc **+3.2% CPU** | 1.000× at N=31 | N=15 sat under the null floor (1.029×) |
| CLI **24% slower, 37% more CPU** | 0.988× / 1.010× at N=5 ABBA | two sequential N=1 runs with a cold file cache; times fell 25.3 → 22.8 s once warm |
| **RSS grows unboundedly**, 0.10 MB/frame, ~10 GB/hour extrapolated | median is FLAT at 92.1/92.2/92.3 MB | the "curve" was three N=1 points drawn from a 92–403 MB multimodal distribution |
| **`collect()` halves RSS for free** (405.6 → 195.0) | 0.1 MB, 0% at N=5 | same distribution, one sample each side |
| predicted **lock contention** on 28 threads | no such lock exists | the stale doc comment in §4 |

Every one of those would have been published as fact from a single run.

## Reproducing

```
cargo build --release -p ffai-diana --example alloc_ab                        # mimalloc arm
cargo build --release -p ffai-diana --example alloc_ab --features alloc-rusty # rusty_alloc arm
cargo build --release -p ffai-diana --example alloc_ab --features alloc-sys   # system arm
cargo build --release -p ffai-diana --example alloc_decode                    # decode-only probe

FFAI_TRIM_MS=200 ./alloc_ab n 1500     # with reclaim
FFAI_TRIM_MS=0   ./alloc_ab n 1500     # without
```

Each arm prints `ARM=… min=… med=… ndet=… sum=…` on one line; `sum` is the work
checksum that must match across arms.

`ffai` itself pins `rusty_alloc` 0.3.0 and defaults to it; `--features mimalloc` rebuilds against
the C library as the oracle, which is what keeps "our pure-Rust allocator
matches the C one" a claim anyone can re-measure rather than one they have to
take on trust.
