# Diana — the 1080p gap and the heavy tail

Two unknowns, both raised by the side-by-side viewer, both recorded in the root
README and one commit message as open observations:

1. *"At MOT17's 1080p the latency gap closes to parity — 1.12x ahead, 0.92x
   behind, 0.96x behind across three runs, against 0.70x ahead at 640. Larger
   frames erase the advantage."*
2. *"Diana's latency tail is heavier than Ultralytics'. Mean 117.6 ms against a
   median of 51.0; Ultralytics in the same loop, taking the same interruptions,
   sits at 59.8 against 46.9. Both engines see identical scheduling noise, so
   the asymmetry is ours."*

**Both are refuted, and the second is inverted.** Neither was a property of the
engine; both were properties of how they were measured. The descent also found
three defects in our own instruments and one real — but small — mechanism.

Depth 6 was run first, per `codec-six-whys-unknowns`, and it is where both
unknowns terminated.

---

## D6a — is the box quiet? **No, and that voids every wall number above.**

- ASKED: what is the noise floor these observations were taken against?
- COUNTED: 10 foreign processes; `cargo test --release --workspace` plus a
  Carmenta gate script, started 16:13, from a concurrent session in the shared
  worktree. Total CPU load 66 %.
- MEASURED: **null arm — Diana against Diana, identical code, ABBA, 6 reps:**
  CPU ratio **0.896x**, within-arm spread **21.5 % / 22.8 % CPU** and
  **46.6 % / 38.1 % wall**.
- ANSWER: the harness floor on this box is **10.4 % on the CPU ratio**. The
  original readings — 1.12x, 0.92x, 0.96x — are an **8-12 % spread that sits
  entirely inside it**. They were never distinguishable from each other.
- CONFIDENCE: high. Work parity held (401 detections, constant across all reps).
- STATUS: closed. `codec-measurement` §15's corollary applies — the floor moved
  because a neighbouring project's build started, and the start time brackets
  the measurements that went bad.

## D6b — do both arms do identical work? **No. Our own timer excluded the decode.**

- ASKED: does `--serve` time the same thing `predict(path)` times?
- COUNTED: one `load_image` call per frame, sitting **before** `Instant::now()`.
- ANSWER: **no.** Diana's reported `ms` excluded JPEG decode; Ultralytics'
  `predict(path)` includes its own. At 1080p that is a 2.07 MP decode —
  measured at **8.4 ms, 16 % of the frame** — handed to us for free.
- The viewer's docstring and the root README both asserted the opposite
  ("Both engines decode inside their own timed region"). The claim was written
  from intent rather than from the code.
- FIXED: decode is inside the timed region and reported split, as
  `ms` / `detect_ms` / `decode_ms`, so neither half can hide again.
- CONFIDENCE: high — it is a line ordering, not a statistic.
- STATUS: closed.

## D6c — `Profile::report()` had ZERO callers

- COUNTED: `grep -rn "profile::report" crates/ --include=*.rs` → **0 hits.**
- ANSWER: the stage instrument existed, was maintained, was documented at
  length in its own module header, and **had never once been read**. Wired to
  the serve loop's exit under `FFAI_PROFILE=1`.
- STATUS: closed. Worth its own line because it is the cheapest class of
  finding available and nothing in the tree pointed at it.

## D6d — psutil read 0.000 s of CPU for an arm doing 84 ms/frame

- COUNTED: CPU time exactly `0.0` across 6 reps while wall read 83.9 ms/frame.
- ANSWER: the venv's `Scripts/python.exe` is a **trampoline** that execs the
  real interpreter as a child; the watched pid did nothing. Summing the process
  tree fixed it, and an `assert cpu > 0` now makes the failure loud.
- CONFIDENCE: high. §7 — an impossible number is the instrument asking for help,
  and this one would have silently reported Ultralytics as infinitely efficient.
- STATUS: closed.

---

## D1/D2 — Unknown 1: "larger frames erase the advantage"

### The premise is false in counts, before any clock

- COUNTED, letterbox geometry at `imgsz=640`, stride 32, from `image.rs`:

  | corpus | source | letterbox = model work | format |
  |---|---|---|---|
  | COCO-640 | 586x640 = 0.27 MP | **608x640 = 389 kpx** | PNG 448 KiB |
  | MOT17 | 1920x1080 = 2.07 MP | **640x384 = 246 kpx** | JPEG 194 KiB |

- ANSWER: **a 1080p frame is 37 % LESS model work than a COCO frame.** In rect
  mode a 16:9 source letterboxes to 640x384; a near-square COCO image
  letterboxes to 608x640. "Larger frames" describes the *source*, which the
  model never sees. The phrase in the README was wrong as written.
- STATUS: closed.

### D2 — the preprocessing hypothesis, refuted

- ASKED: source pixels grow 7.7x; is our resize losing to OpenCV's?
- MEASURED (`FFAI_PROFILE=1`, now that it prints): `pre` = **1.96 ms/image at
  1080p against 4.48 ms at COCO.** Preprocessing is *cheaper* on the larger
  source.
- ANSWER: **refuted.** A bilinear resize costs per OUTPUT pixel, and the output
  is 246 kpx against 389 kpx. The 7.7x input growth buys nothing because each
  output pixel samples a fixed 2x2 neighbourhood regardless of how far apart
  those samples are.
- Stage shares are otherwise near-identical at both resolutions — pre 3.6 % vs
  4.7 %, backbone 56.6 % vs 56.7 %, neck 23.2 % vs 25.2 %, head 15.4 % vs
  12.4 %. There is no resolution-specific stage.
- STATUS: closed, refuted.

### D1 — the gap itself, on the metric that survives a loaded box

- MEASURED: ABBA-interleaved, **CPU time** via `GetProcessTimes`, both engines
  in child processes behind the same stdin/stdout protocol, model load
  excluded, 50 frames x 6 reps, work parity checked per arm:

  | corpus | diana cpu | ultralytics cpu | **CPU ratio** | wall ratio |
  |---|---:|---:|---:|---:|
  | MOT17 1080p | 131.4 ms | 428.6 ms | **3.262x** | 1.217x |
  | COCO 640 | 129.1 ms | 418.8 ms | **3.245x** | 1.667x |

- ANSWER: **the advantage does not erode. It is 3.26x at 1080p and 3.25x at
  640 — a 0.5 % difference.** What varied was wall, whose within-arm spread on
  the COCO run was **122 % and 84 %**.
- CONFIDENCE: high. Two corpora, ABBA, CPU time, constant detection counts per
  arm, and the answer agrees to three significant figures across a resolution
  change that moves model work by 37 %.
- CAVEAT, and it is the honest one: **CPU time is not wall time, and a user
  experiences wall.** 3.25x less CPU with a wall ratio near 1 means Ultralytics
  spends far more CPU to reach a similar wall clock — it is using more cores.
  The right claim is "Diana does the same work for a third of the CPU", not
  "Diana is 3.25x faster". Those are different products: one matters on a
  shared server, the other on an idle laptop.
- STATUS: closed. The README's "larger frames close the gap" is withdrawn.

---

## D1 — Unknown 2: "Diana's tail is heavier"

### The original measurement was block-wise, which §3 forbids

- ASKED: are the tails comparable?
- ANSWER: the 117.6/51.0 vs 59.8/46.9 figures came from running **all of Diana,
  then all of Ultralytics**. `codec-measurement` §3: that puts machine drift
  *between the blocks*, and the same quantity read 3.9 % / 34.1 % / 49.4 %
  block-wise against a tight 16.0-20.2 % interleaved on that skill's own
  campaign.

### Interleaved per frame, the result inverts

- MEASURED: both engines process frame *i* back to back before either moves to
  *i+1*, so within a frame they see the same machine. Two runs:

  | | diana mean/p50 | ultralytics mean/p50 | diana outliers | ultra outliers |
  |---|---:|---:|---:|---:|
  | run 1, n=140 | **1.14** | 1.79 | 5 | 16 |
  | run 2, n=110 | **1.21** | 3.33 | 7 | 16 |

- ANSWER: **refuted and inverted. Diana's tail is the LIGHTER of the two**, by a
  wide margin and in both runs. Ultralytics carries 16 outliers to our 5-7 and a
  mean/p50 up to 3.33 against our 1.21.
- CONFIDENCE: high on the direction (two runs, interleaved, consistent);
  low on the magnitude (the box is at 66 % foreign load, and outlier counts are
  small integers).
- STATUS: closed, refuted.

### D5 — but a real Diana-specific mechanism was found underneath it

- COUNTED: page faults per frame, Diana, split by whether the frame was an
  outlier: **31 normal, 4,200 on slow frames** in one run; **152 normal, 1,257
  on slow frames** in another. A 8-135x spike, ~5-17 MB of pages re-faulted on
  the frames that stalled.
- MECHANISM: mimalloc returning pages to the OS and re-faulting them — the same
  mechanism as this campaign's largest single measured effect, the 1.64x
  allocator finding.
- PROBE: `MIMALLOC_PURGE_DELAY=-1`.

  | | normal faults/frame | faults on slow frames | total faults, 80 frames |
  |---|---:|---:|---:|
  | default | 152 | 1,257 | 77,305 |
  | purge off | **15** | **15 — flat** | **39,149** |

- The counter confirms the mechanism **exactly as predicted**: the spikes do not
  shrink, they disappear, and total faults halve.
- **And the clock shows nothing.** mean/p50 1.21 → 1.22, median 69.9 → 70.1 ms,
  outliers 7 → 6. Sized: ~137 faults/frame removed at ~1.5 us each is **~0.2 ms
  of a 70 ms frame, 0.3 %** — correctly below what any clock here resolves
  (§15).
- COST: **+9.6 MiB peak RSS (128.8 → 138.3), +37.8 MiB steady.** The footprint
  gate passes by **1 MiB** (160 against ORT's 161).
- VERDICT: **not shipped, and not shipped as a default specifically.** It buys
  0.3 % of latency and spends ten times the gate's entire headroom. It is
  mimalloc's own environment variable, so any embedder who wants flat page
  behaviour already has it, and it is documented with these numbers rather than
  recommended.
- STATUS: closed. Refuted as a *lever*; confirmed as a *mechanism*. Recorded
  separately because the distinction is what stops it being re-litigated — the
  page faults are real, they are simply not worth 9.6 MiB.

---

## What changed as a result

| | before | after |
|---|---|---|
| `--serve` decode | outside the timed region | inside, reported split |
| `Profile::report()` | zero callers | printed on `FFAI_PROFILE=1` |
| CPU-time A/B harness | none | `tools/diana_cpu_ab.py`, ABBA, null arm |
| tail attribution | none | `tools/diana_tail_paired.py`, coincidence test |
| README: "larger frames close the gap" | published | withdrawn — 1080p is *less* model work |
| README: "Diana's tail is heavier" | published | withdrawn — it is lighter, 1.14-1.22 vs 1.79-3.33 |

## Refuted hypotheses, with their numbers

- **"Our resize loses to OpenCV at 1080p"** — refuted. `pre` is 1.96 ms at
  1080p against 4.48 ms at COCO; resize costs per output pixel, and 1080p
  produces fewer.
- **"Larger frames mean more model work"** — refuted. Rect letterbox makes a
  1080p frame 246 kpx against a COCO frame's 389 kpx.
- **"Diana's tail is heavier than Ultralytics'"** — refuted, inverted. 1.14-1.22
  against 1.79-3.33, interleaved, two runs.
- **"Disabling mimalloc purge is worth taking"** — refuted on price. Real work
  removal (faults halved, spikes eliminated) worth 0.3 % of latency for 9.6 MiB
  against 1 MiB of gate headroom.


---

## The rebuild — the one lever the descent left, and its price

The descent ended on unspent CPU: **3.25x less than Ultralytics at a wall ratio
near 1.0.** A user feels wall, so unspent CPU is unspent latency. Thread width
was refuted at 4-6 workers earlier in this campaign, and §11 says a refutation
expires when its baseline moves — epilogue fusion, JIT decode and the
decode-timing fix have all landed since, so it was re-run rather than inherited.

### First attempt: void, and the reason is worth keeping

Every width was spawned up front and round-robined between. Six live thread
pools contend with each other; the curve came back non-monotonic (1 -> 156 ms,
4 -> 77, 6 -> 167, 12 -> 54) with a **105.8 % within-arm spread**. A width's
cost is only its own when nothing else of ours is resident. Re-run with one
process alive at a time.

### The sweep, one process at a time, both corpora

Work parity held at every width on both corpora — 297 detections on MOT17, 180
on COCO, constant — so thread count is not changing results.

| threads | MOT17 wall | MOT17 cpu | COCO wall | COCO cpu |
|---:|---:|---:|---:|---:|
| 1 | 101.3 | 97.3 | 124.9 | 108.6 |
| **4 (shipped)** | **58.0** | **112.9** | **57.3** | **114.8** |
| 8 | 56.2 | 169.9 | 54.7 | 194.5 |
| 12 | 49.6 | 244.1 | 49.4 | 255.9 |

### Verdict: keep 4. The trade is bad and it is bad consistently.

| corpus | 4 -> 12 wall gain | 4 -> 12 CPU cost |
|---|---:|---:|
| MOT17 1080p | 1.171x | **2.16x** |
| COCO 640 | 1.158x | **2.23x** |

- **The wall gain is not a result.** 1.16-1.17x sits under a within-arm spread
  of 73.6 % (COCO) and 197.1 % (MOT17) on this box. It cannot be claimed.
- **The CPU cost IS a result.** The CPU curve is monotonic, agrees across two
  corpora to within 3 %, and CPU is the drift-immune metric (§2). Doubling the
  pool doubles the bill.
- **And it would spend the thing the descent actually proved.** At 4 threads
  Diana uses 113 ms CPU against Ultralytics' 429 — 3.8x better. At 12 it uses
  244, which is 1.76x better. That trades a **measured, certain 2.2x efficiency
  advantage for an unproven 1.16x wall gain.**

Refuted on three varied probes as §11 requires: two corpora that differ in
model work by 37 %, and the level above the change (whole-pipeline wall, not a
kernel microbench). **No code change. The default stays at 4.**

The idea that survives: the right width is a *deployment* choice, not a
constant. A latency-bound single-stream user on an idle 24-core box may well
want 12; a server packing many streams wants 4 and would be actively harmed by
12. `FFAI_DIANA_THREADS` already exposes it, and it is now documented with the
numbers above instead of a recommendation.

## Still open

- **A quiet box.** Every wall figure here carries a within-arm spread between
  38 % and 197 %. The thread sweep's 1.16x is the number most likely to change
  when one is available; the CPU costs beside it will not.
- Nothing here was measured on a quiet box. Every number above is either a
  COUNT (immune) or a CPU-time ratio (5x tighter than wall, per §2), and the
  wall figures are reported only to show how far they moved.
