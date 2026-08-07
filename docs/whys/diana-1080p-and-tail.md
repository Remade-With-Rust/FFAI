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
- COST: **+57 MiB steady (182 → 239)** on `ffai bench detect`, the instrument
  the footprint gate scores, against ORT's 161 MiB. An earlier ad-hoc RSS
  sampler said +9.6 MiB peak; it was not measuring steady state and is
  superseded — see the re-measurement below.
### Re-measured when shipping it as the default was proposed

The first pass priced this on one interleaved run (0.3 %) and an ad-hoc RSS
sampler. Both were re-done, and one of them was WRONG:

- **The RSS sampler was incoherent.** A purge-delay sweep read steady RSS of
  142.4 MiB at default against **91.5 MiB with purging disabled** — physically
  impossible, since disabling purge can only retain more. A single-instant
  `rss` reading on a box under memory pressure is not "steady RSS"; the OS is
  trimming working sets underneath it. Replaced with `ffai bench detect`, the
  instrument the footprint gate actually scores.
- **The latency question was genuinely open.** One unpaired bench run read p50
  57 -> 50 ms (-12 %), against the first pass's 0.3 %. Resolved with a
  same-binary ABBA A/B, 8 reps, work parity constant at 401 detections:
  **CPU 0.968x, wall 1.033x**, both inside the 10.4 % null-arm floor and wall
  marginally WORSE. §12: same-binary deltas are the reliable evidence, and this
  one says there is no latency effect.
- **The cost, on the right instrument: 182 -> 239 MiB steady, +57 MiB, +31 %.**
  Not +9.6 MiB as first measured.

- SHARPER REASON, found while re-measuring: the faults come from the OS
  trimming our working set, which happens under **system memory pressure**.
  The setting therefore does least when the machine is idle and safe, and most
  when the machine is already short of memory and retaining 57 MiB is the worst
  available response. It helps least where it is safe and most where it is
  dangerous.

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

---

# Orientation, broken open — and refuted at real shapes

The synthetic sweep's 1.31-1.68x did not survive contact with the shapes the
graph actually runs.

## D5a — the island probe measured a copy nobody performs

First conclusion: "a per-layer NHWC island is 100-200x underwater." **Wrong.**
It transposed the COL MATRIX (460 MB). im2col *builds* the col matrix, so its
layout is free to choose, and weights transpose once at load. The only thing
that ever changes layout is the ACTIVATION, at `cout*H*W`. Re-measured: 1.245
ms/img saving against 2.053 ms/img of activation transpose — **1.65x
underwater, not 200x.** The conclusion survived; the mechanism was wrong.

## D5b — the chain probe buried its own signal

A 6-layer chain read 1.03x, and NET -0.068 ms once boundary transposes were
counted. Its im2col was written for the probe rather than being the shipped
kernel, and at 0.6-1.0 ms against the GEMM's 0.15-0.36 it DOMINATED the total.
A probe whose scaffolding outweighs what it measures reports the scaffolding.

Caught by its own data: two rows with IDENTICAL im2col work — same `cin,h,w`,
and im2col does not depend on `cout` — read **1.715 and 11.649 ms**. That is
also what all-of-A-then-all-of-B produces (§3); interleaving at min-of-31
collapsed both totals 7x.

## D5c — the verdict, with the floor measured INSIDE the probe

GEMM only, real shapes, min-of-101, ABBA-interleaved. The last two rows are
DUPLICATES of the first two under different labels, so the probe carries its
own floor:

| cin->cout | HxW | calls | M=cout | M=HW | speedup |
|---|---|---:|---:|---:|---:|
| 32->32 | 24x40 | 12 | 0.1236 | 0.0973 | 1.27x |
| 16->16 | 48x80 | 5 | 0.1028 | 0.1245 | 0.83x |
| 64->64 | 12x20 | 4 | 0.0705 | 0.0916 | 0.77x |
| 64->16 | 48x80 | 1 | 0.2427 | 0.2449 | 0.99x |
| 16->8 | 96x160 | 1 | 0.2059 | 0.1544 | 1.33x |
| 128->16 | 24x40 | 1 | 0.1762 | 0.1442 | 1.22x |
| 32->16 | 48x80 | 1 | 0.1860 | 0.1902 | 0.98x |
| 16->32 | 48x80 | 1 | 0.2057 | 0.2576 | 0.80x |
| **32->32 (dup)** | 24x40 | - | 0.1194 | 0.1138 | **1.05x** |
| **16->16 (dup)** | 48x80 | - | 0.0730 | 0.0588 | **1.24x** |

`16->16 48x80` reads **0.83x in one row and 1.24x in its identical twin**. The
floor is ~±25 % and every per-shape number sits inside it.

**Weighted by calls per image: 1.047x — inside the floor, and not a 4.7 % gain.**
§3: any delta smaller than the spread is not a result.

### Why the synthetic probe said 1.68x

It held total FLOPs constant, forcing **N = 25,000-400,000**. Real feature maps
are `H*W` = 3,840 at 48x80 and 960 at 24x40 — **10 to 400x smaller**. At
synthetic sizes one dimension was genuinely huge, so which one was small
mattered. At real sizes NCHW is M=cout(16-32) x N=HW(960-3,840) and NHWC is
M=HW x N=cout: **both have a small dimension.** Swapping which one is small
does not help a squat matrix. §18 is the rule this broke.

## Verdict

**NHWC / orientation: REFUTED at real shapes.** The whole-graph conversion —
every conv, the epilogue, attention, the detect head, five tiers of oracle —
would have been built for 1.047x inside a ±25 % floor. That prune is worth more
than the probes cost.

NOT refuted: the narrow-cout deficit itself. The +0.823 correlation with
log2(cout) stands and candle's isolated M=8 vs M=512 gap of 6.3x stands. What
is refuted is that ORIENTATION reaches it.

## What is left, and it is one thing

| approach | status | why |
|---|---|---|
| im2col fusion | refuted | 1x1 has no col matrix, same deficit |
| direct convolution | refuted | 0.19x per-shape, worse as cout grows |
| cache tiling | refuted | operand size correlates -0.048 |
| NHWC / orientation | refuted | 1.047x at real shapes, inside a ±25 % floor |
| MKL | refuted earlier | within noise of candle's GEMM |
| thread width | refuted | 2.2x CPU for 1.16x wall |

**Winograd F(2x2,3x3) is the one major technique untried.** Four multiplies per
output instead of nine — a 2.25x arithmetic reduction — applying exactly to the
3x3 stride-1 convolutions that are 22.9 % of detect. Real implementations net
1.5-2x on those layers after transform overhead, so roughly **8-9 % of detect**.
Its weakness is numerical rather than structural, so it arrives with a
correctness gate the five-tier oracle already holds.

Price it with an in-context probe on ONE layer before writing a transform. Every
structural idea in the table above looked good until it was measured in context.

## D5d — the aggregate was hiding a DISPATCH, and the table said so

The 1.047x whole-probe ratio was reported as "orientation refuted". That was
wrong, and the reason is the one `codec-content-adaptive-dispatch` names: **a
mixed per-shape result is an UNFINISHED dispatch, not a verdict.** Averaging a
2x win against a 0.6x loss reports neither.

The objection that reopened it: the per-shape table "reads as an adaptive
dispatch". It does. The duplicate-row check only proved that SOME rows were
noise; it never established that ALL of them were.

**The correct test is reproducibility, not size.** Each of 47 shapes measured in
4 independent passes, arms ABBA-interleaved, min-of-25 per pass. A shape counts
as dispatchable only if **every pass agrees on the sign**.

| shape | pass 1-4 | prefers |
|---|---|---|
| 48->64 96x160 | 2.13 1.72 2.10 1.99 | **NHWC ~2x** |
| 32->16 48x80 | 1.62 1.53 1.54 1.43 | NHWC ~1.5x |
| 96->128 48x80 | 1.39 1.45 1.44 1.40 | NHWC ~1.4x |
| 32->32 96x160 | 1.44 1.35 1.40 1.45 | NHWC ~1.4x |
| 128->16 24x40 | 1.24 1.24 1.22 1.28 | NHWC |
| 16->16 48x80 | 0.57 0.69 0.81 0.59 | **NCHW** |
| 64->64 12x20 | 0.81 0.75 0.65 0.80 | NCHW |
| 256->80 12x20 | 0.68 0.88 0.82 0.86 | NCHW |

**28 of 47 shapes reproducible: 21 prefer NHWC, 7 prefer NCHW, 19 sign-flip.**

**Prize of a perfect per-shape dispatch, weighted by calls per image: 1.173x on
the GEMM = 4.39 % of detect.**

### Why this dispatch is cheaper than the usual kind

Every dimension is **static**. cin, cout, H and W are fixed once the tier and
geometry are chosen, so the layout per layer is a constant decided at model
load — not a runtime probe, not a heuristic on content, and with **zero
per-call dispatch overhead**. The skill's warning about a runtime arm-selector
in the hot loop (§15's corollary) does not apply.

### What is NOT yet measured, and it decides the real number

The 4.39 % assumes **zero transition cost**. It is an upper bound:

* For a 3x3 conv, im2col builds the col matrix, so its layout is free — but the
  INPUT activation's layout decides which im2col is cheap, and the OUTPUT
  arrives in the corresponding layout.
* For a 1x1 conv there is no col matrix at all: the activation IS the operand,
  so changing its orientation means an actual transpose.
* 21 shapes want NHWC and 7 want NCHW. **If they alternate, transposes eat the
  win**; if they cluster into runs, they do not.

The roofline aggregates by shape and therefore cannot answer this — it does not
record layer ORDER. That is the next instrument, and the honest statement until
it exists is: **4.39 % of detect is the ceiling, and the floor depends on how
many layout transitions the real layer sequence forces.**

### The correction worth keeping

Two probes said "orientation is refuted" — the 1.03x chain and the 1.047x
aggregate — and both were true as stated and wrong as generalised. The first
buried its signal under a reimplemented im2col; the second averaged a real
bimodal result. **A refutation built on an aggregate is only as good as the
homogeneity of what it averaged**, and nothing had checked that.

## D5e — solved over the real layer sequence, and the dispatch is not the win

The 4.39 % per-shape figure assumed neighbours never disagree. The missing
instrument was layer ORDER, which no aggregate keyed by shape can recover.
`record_order` now emits the 88 dense convolutions in execution order, and the
layout assignment becomes a shortest path over two states per layer — exact,
not bounded.

Three independent runs, min-of-15 per layer per run:

| run | all-NCHW | all-NHWC | optimal | NHWC x | opt x | NHWC layers |
|---|---:|---:|---:|---:|---:|---|
| 1 | 20.409 | 18.559 | 18.005 | 1.100x | 1.134x | 56/88, 2 transitions |
| 2 | 24.661 | 20.869 | 20.670 | 1.182x | 1.193x | 52/88, 7 transitions |
| 3 | 22.532 | 19.761 | 19.348 | 1.140x | 1.165x | 60/88, 4 transitions |

**Median: all-NHWC 1.140x, optimal 1.165x. Positive in all three runs.**

* **all-NHWC, no dispatch at all: ~3.67 % of detect**
* **optimal per-layer dispatch: ~4.21 % of detect**
* **the dispatch is worth +0.55 pp over a single global layout**

### The engineering conclusion inverts

The finding began as "orientation is a per-shape dispatch". Solved over the real
sequence it is **not**: a single global layout captures 87 % of the achievable
win with **no per-layer logic, no transitions, and no dispatch table**. The
optimum chooses NHWC for 52-60 of 88 layers, but forcing the remaining layers
into NHWC too costs less than the transitions avoiding it would cost.

That is the opposite of the usual dispatch lesson and it is only visible with
order in hand. **Build the global layout; treat the dispatch as a later
refinement worth half a point.**

### What this number does NOT include

GEMM only. A real conversion also moves im2col, the epilogue and the activation
plumbing, and the chain probe's reimplemented `im2col_nhwc` was slightly SLOWER
than its NCHW twin. Some of the 3.67 % will be given back. **Re-measure im2col
and the epilogue in NHWC before committing to the refactor** — that is exactly
the in-context check every structural idea in this campaign has failed.

---

# Retrospective: which past decisions were made on an AGGREGATE?

Every A/B toggle in the tree is a decision that averaged over shapes. The
orientation case shows what that can hide — a 2x win and a 0.6x loss reported
as 1.047x. Ranked by the prior that the same thing happened:

| toggle | what it decides | why it might be bimodal | prior |
|---|---|---|---|
| `FFAI_DIANA_NO_INPLACE` | epilogue fused in place | **Two A/B runs DISAGREED IN DIRECTION** (+5.5 % at 8 images, -2.8 % at 45). Kept on counters with no speed claim. Disagreement in sign is the exact signature of an averaged bimodal result. | **highest** |
| `FFAI_DIANA_THREADS` | worker count | The sweep was WHOLE-GRAPH. Layers span 12x20 (960 px) to 192x320 (61,440 px) — a 64x range. A tiny layer at 12 threads pays barrier cost on almost no work; a large one wants every core. Per-LAYER width has never been tested. | **highest** |
| `FFAI_DIANA_NESTED_PAR` | nested parallelism | Refuted at a 2.32x CPU tax measured across the whole graph. Might still pay for the handful of very large early layers. | medium |
| `FFAI_DIANA_NO_DWCONV` | our depthwise vs candle grouped | 7 depthwise shapes at AI 2.2 f/B, spanning very different sizes. | medium |
| `FFAI_DIANA_NO_PW` / `NO_CONV3` / `NO_S2` | our kernel vs candle per KIND | Already dispatched by kind, never within a kind. | medium |
| `FFAI_DIANA_TILE` | im2col tiling | Refuted 3x on aggregate — but now has a MECHANISM (operand size correlates -0.048 with throughput), which is a reason to expect uniform failure rather than a hidden win. | low |
| `FFAI_DIANA_DIRECT` | direct convolution | **Already re-tested per shape.** Wins only at cout=8, 1.14x, worth 0.05 ms/img. Dispatch exists and is negligible. **Closed.** | closed |
| MKL | GEMM backend | "Within noise of candle's GEMM" was an aggregate, and two tuned GEMMs are exactly the kind of pair that trades wins by shape. Not currently wired. | medium-high |

**The two to test next are the epilogue and per-layer thread width**, and the
epilogue first because its own record already contains the tell: a direction
disagreement that was correctly refused as a speed claim and never diagnosed.

---

# Hammering both dispatch candidates - and the content axis

Both retrospective candidates tested per shape, in context, under the skill's
governing rule: **a shape is dispatchable only if every independent rep agrees
on the SIGN.** Both refuted, and so is the content axis.

## Candidate 1 - epilogue in place vs out of place: REFUTED

54 shapes, 4 ABBA reps, in context through the roofline.
**Only 2 of 54 reproducible. 52 sign-flip.**

| | ms/img | |
|---|---:|---:|
| all in-place (shipped) | 18.990 | - |
| all out-of-place | 19.155 | 0.991x |
| **per-shape dispatch** | **18.920** | **1.004x** |

The "+5.5 % at 8 images, -2.8 % at 45" direction disagreement that made this the
top candidate was **noise, not a hidden dispatch**. The original decision - keep
on the counted work removal, make no speed claim - was correct, and is now
correct for a measured reason rather than a cautious one.

## Candidate 2 - per-layer thread width: REFUTED

54 shapes, widths {1,2,4,8,12}, 3 reps. **Only 7 of 54 have a stable optimum**,
and they disagree (2 want 1 thread, 3 want 4, 2 want 8).

| all layers at | ms/img | vs shipped |
|---|---:|---:|
| 1 thread | 47.320 | 0.427x |
| 2 threads | 29.990 | 0.674x |
| **4 threads (shipped)** | **20.220** | **1.000x** |
| 8 threads | 25.250 | 0.801x |
| 12 threads | 48.790 | 0.414x |
| per-layer dispatch | 20.140 | **1.004x** |

A clean optimum at 4; the dispatch buys 0.4 %. The 64x work range across layers
was a good reason to look and produced no signal.

## The content axis - there is no surface to dispatch on

A convolution graph has **static shapes and dense arithmetic**, so the only
thing an image can change is the VALUES. Two ways that could have mattered:

### Denormals: 0 in 80,640,000 values

`silu.rs` claimed unbounded activations "almost certainly land in denormals,
which are orders of magnitude slower on x86", offered to explain why ablating
SiLU made the pipeline SLOWER (49.7 ms vs 45.1). It read like a finding and had
never been checked.

| | count | share |
|---|---:|---:|
| values | 80,640,000 | |
| **subnormal** | **0** | **0.000000 %** |
| exact zero | 4 | 0.00 % |

**Zero. Not few.** Flush-to-zero would change nothing. Corrected in place: the
anomaly is now **unexplained rather than explained**, which is worse than it
looked, because a plausible story kept it feeling closed.

### Detection count: correlation -0.009

115 diverse COCO images, per-image latency as the median of 3 runs. Detections
ranged 1 to 15. **correlation(detect_ms, detections) = -0.009.** Images with
<=2 objects: 53.5 ms; with >=8 objects: 43.8 ms - busy images nominally FASTER,
the signature of no relationship. The raw 4.63x p10-p90 spread is box noise, and
the -0.009 is what proves it.

**No content-adaptive dispatch is available in this engine** - the mechanism is
absent rather than unfound: static shapes, dense arithmetic, no data-dependent
branching, no denormals, and a head whose cost does not move with what it finds.

## Where the campaign stands

| lever | status |
|---|---|
| **global NHWC layout** | **LIVE - 1.140x median on the GEMM, ~3.67 % of detect, positive 3/3 runs** |
| per-layer layout dispatch | +0.55 pp over global; a later refinement |
| epilogue dispatch | refuted, 1.004x, 52/54 sign-flip |
| thread-width dispatch | refuted, 1.004x, 7/54 stable |
| content-adaptive anything | refuted - no mechanism exists |
| denormals / flush-to-zero | refuted - 0 in 80.6M values |
| Winograd F(2x2,3x3) | untried, ~8-9 % of detect, the largest remaining |

---

# The three uncovered pieces, measured — NHWC still pays, at 5.2 %

The global-NHWC figure covered convolution matmuls and nothing else. Three
things change layout with it. All three are now measured, min-of-21, four real
shapes each, both arms using the same code on the same element counts.

| piece | NHWC/NCHW | direction | consistency |
|---|---:|---|---|
| **im2col** | **0.81x** | NHWC **FASTER** | 0.78-0.86 across 4 shapes |
| epilogue (bias+SiLU) | 1.37x | NHWC slower | 1.12-1.51 |
| channel plumbing | 1.98x | NHWC slower | see below |

## im2col is FASTER in NHWC, and the chain probe was wrong about it

The earlier chain probe reported `im2col_nhwc` slightly SLOWER. At min-of-21 it
is faster on every shape. The mechanism is plain once measured: the NHWC gather
copies a contiguous run of `cin` floats per (pixel, tap), while NCHW copies
element by element with a stride. That was a block-wise measurement inside a
probe whose scaffolding dominated, and it should not have been reported.

## The epilogue is slower, exactly as predicted

In NCHW one channel bias is a scalar broadcast over a contiguous run. In NHWC
the bias VECTOR repeats along the fast axis. 1.37x, consistent across shapes.
Small in absolute terms because the epilogue is ~1 ms/img.

## Channel plumbing is the real structural risk, and `narrow` is the worst of it

Counted in the real graph: **15 `cat(dim=1)` and 9 `narrow(dim=1)` per image
over 16.3 MiB.**

| shape | op | NCHW | NHWC | ratio |
|---|---|---:|---:|---:|
| 32c 48x80 | cat | 0.0123 | 0.0396 | 3.22x |
| 32c 48x80 | **narrow** | **0.0008** | **0.0184** | **23x** |
| 64c 96x160 | cat | 1.1234 | 1.3879 | 1.24x |
| 64c 96x160 | **narrow** | **0.0008** | **0.7250** | **906x** |

**In NCHW a channel slice is contiguous, so `narrow` is a free VIEW.** In NHWC
the channel is the fastest axis, so the same call is a strided copy. A ratio of
906x is what "free becomes a real copy" looks like, and no GEMM benchmark can
ever show it.

**It is also mitigable, which is why it is not fatal.** A `narrow` feeding a
convolution never needs materialising — the convolution can read the sub-range
directly, or the split can be folded into the weight indexing. The 906x is the
cost of the naive translation, not of the layout.

## The accounting, on real in-context shares

The probe's own absolutes are NOT usable: its im2col arms are ~7.6x slower than
the shipped kernel (21.997 ms against a measured 2.887), which would inflate
im2col's weight by that factor. The ratios are the durable output; they are
applied to shares measured in context.

In-context, ms/image: detect 26.5, conv 20.77, gemm 16.9, im2col 2.887,
epilogue+rest 0.98, plumbing 0.91.

| | ms/img |
|---|---:|
| gemm, 1.140x faster on 16.9 ms | **+2.075** |
| im2col, 0.81x faster on 2.89 ms | **+0.549** |
| epilogue, 1.37x slower on 0.98 ms | -0.364 |
| plumbing, 1.98x slower on 0.91 ms | -0.893 |
| **NET** | **+1.367 ms/img = 5.2 % of detect** |

**NHWC still pays, and by more than the GEMM-only estimate of 3.67 %** — im2col
adds 2.1 pp, while the epilogue and plumbing take back 4.7 pp.

## What would change this number

* **The plumbing estimate is the weakest input.** Its in-context cost was never
  separately profiled; 0.911 ms comes from this probe running candle's real
  `cat`/`narrow` on real shapes, which is defensible but is not a measurement of
  the shipped path. If the true figure is 2x higher, the net halves.
* **Folding `narrow` into the consuming convolution is worth more than the
  epilogue and plumbing losses combined**, and it is a change that helps NCHW
  too. It should be measured before the layout work, not after.

---

# In-engine NHWC: the win is CONFIRMED and it is NOT a dispatch

Every NHWC number until now was ISOLATED - synthetic tensors inside an example.
`crate::conv3x3::conv3x3_nhwc` puts it in the engine behind `FFAI_DIANA_NHWC=1`:
`im2col_t` produces `[OHW, Cin*9]`, the GEMM runs `col @ Wt`, and the result is
transposed back so the surrounding NCHW graph is untouched.

**Correctness gate: 41 detections both arms, 0 mismatches, worst box delta
0.0000 px.** Identical output, so every timing below is work-parity by
construction.

## The GEMM win is real, and LARGER in context than isolated

8 ABBA reps, 30 frames each, in the engine:

| | |
|---|---|
| ratios | 2.57 1.61 0.56 1.15 1.23 1.67 2.13 1.77 |
| median | **1.645x** |
| paired | **7/8 reps, z = +2.12** |

The isolated probe said 1.140x. **In context it is 1.385-1.645x.** For once an
isolated number UNDERSTATED the effect, which is the other half of the warning
in §6 and the reason it says both directions.

## My first `im2col_t` was the bottleneck, not NHWC

The island first measured **0.452x overall** - 2.2x slower. The cause was mine:
the gather looped `(c, ky, kx)` outside and `ox` inside, so every write landed
`k` floats from the last, a pure stride-k scatter. Rewritten to emit one output
pixel's K values contiguously:

| stage | before | after |
|---|---:|---:|
| detect TOTAL | 0.452x | **0.919x** |
| `>im2col` | 0.176x | 0.450x |
| `>gemm` | 0.715x | **1.385x** |

**The island is now at parity while still paying two handicaps a converted run
does not pay**: `im2col_t` gathers from an NCHW input (inherently strided on the
read side), and every conv output is transposed back.

## There is NO per-shape dispatch: 0 of 21 shapes

Per-shape, 3x3 convolutions, 4 ABBA reps, requiring all four to agree:

**0 shapes always prefer NHWC. 14 always prefer NCHW. 7 sign-flip.**

| | ms/img over these shapes |
|---|---:|
| all-NCHW (shipped) | 30.355 |
| all-NHWC island | 57.370 (0.529x) |
| **per-shape dispatch** | **30.355 (1.000x)** |

The island's two handicaps exceed the GEMM win on **every** shape, so there is
nothing to gate on. This is the same conclusion the layer-order DP reached -
**global layout, not dispatch** - now confirmed with a working implementation
and a correctness gate rather than a projection.

## The projected prize for a converted run

From in-context shares (detect 2.537 s, gemm 0.800 = 31.5 %, im2col 0.317 =
12.5 %):

| | |
|---|---:|
| gemm at the in-context 1.385x | +8.8 % |
| im2col at 0.81x from an NHWC input | +2.4 % |
| transpose - vanishes in a run | 0.0 % |
| epilogue 1.37x and plumbing 1.98x slower | -1.5 % |
| **NET** | **~9.6 % of detect** |

Up from the earlier 5.2 %, and the reason is precisely that the GEMM ratio was
measured isolated at 1.140x and in context at 1.385x.

## What ships and what does not

* **Ships now:** nothing. The island is at parity and its per-shape dispatch is
  1.000x. `FFAI_DIANA_NHWC=1` stays OFF by default, gated, as the A/B arm and
  the foundation for the conversion.
* **The work that pays** is converting a RUN of consecutive convolutions so the
  activation stays NHWC across them - which removes the transpose AND makes
  `im2col_t` read contiguously. Both handicaps disappear together; that is why
  the island cannot show the win and no per-layer gate can rescue it.
* **The correctness gate already exists** and is byte-identical, so the
  conversion has an oracle from the first line.

---

# Going after the prize — and finding it is a third of what was claimed

## The im2col_t variants, all measured in the engine

| variant | loop order | im2col ratio |
|---|---|---:|
| v1 | (c,ky,kx) outer, ox inner | 0.176x |
| v2 | ox outer, c inner | 0.278x |
| **v3 (kept)** | **c outer, ox inner** | **0.315x** |

v3 streams one channel plane at a time and writes 3 contiguous floats per
(pixel, channel, ky) at stride k, with the row's `ow*k` destination (92 KB)
L2-resident.

**v3 was briefly reverted on a bad comparison** — a LOADED-box v2 run (0.919x
total) against a QUIET-box v3 run (0.675x total). Different boxes, opposite
conclusion. That is precisely the cross-run error this campaign exists to catch,
and it was made here after documenting it four times.

## The GEMM ratio, once the box went quiet

| box state | NCHW-arm spread | gemm ratio |
|---|---:|---:|
| loaded (NCHW total 2.5 s) | ~200 % | **1.645x, z=+2.12** |
| quiet (NCHW total 0.77 s) | ~10 % | **1.104x** |
| quiet | tight | 1.146x |
| 12 % load | 62 % | 1.185x |

**The isolated probe said 1.140x and was right all along.** The "confirmed in
context at 1.645x with z=+2.12" reading was noise on a box whose own NCHW arm
varied 3x between reps — and it happened to flatter the hypothesis, which is
why it was believed. `codec-measurement` §16 warns that a cross-implementation
ratio trends with N; this one trended with the neighbouring build.

## The prize, honestly

| | |
|---|---:|
| gemm, 27.4 % of detect at 1.10-1.19x | +2.6 to +4.3 % |
| im2col, 9.9 % at 0.81x from an NHWC input | +1.9 % |
| epilogue 1.37x and plumbing 1.98x slower | -1.5 % |
| **NET** | **~3 to 5 % of detect** |

**Not 9.6 %.** That figure used the loaded-box 1.385x. Every favourable NHWC
number this campaign produced was taken on a busy box.

## Verdict: do NOT do the conversion

A whole-graph NHWC conversion touches every convolution, the epilogue, the
attention blocks, the channel plumbing, the detect head, and the five-tier
oracle across two geometries. **For ~4 % of detect that is a bad trade** — it is
a large, correctness-dense refactor priced at roughly what a single good kernel
brick returns, and the campaign has cheaper unspent options (Winograd at ~8-9 %).

## What is kept

* `conv3x3_nhwc` behind `FFAI_DIANA_NHWC=1`, **OFF by default**, with a
  byte-identical correctness gate (41 detections, 0 mismatches, 0.0000 px). It
  is the A/B arm and the foundation if the decision is ever revisited.
* `im2col_t` at v3, the best of three measured variants.
* The measured ratios for all four pieces, so nobody re-derives them.

## The lesson worth more than the brick

**Three separate NHWC numbers were reported as wins and all three shrank when
re-measured**: 1.68x synthetic became 1.047x at real shapes; 1.047x-plus-a-
dispatch became global-only; 1.645x in context became 1.10-1.19x on a quiet box.
Each intermediate reading was taken honestly and each was wrong in the same
direction — toward the hypothesis being pursued.

The defence that worked every time was the same one: **a duplicate arm, or an
identical-work pair, measured inside the same run.** Where that was present the
error surfaced immediately; where it was absent the number survived for hours.

---

# Taking the prize: two bricks landed, one step left

Decision taken to pursue the conversion despite the ~4 % pricing. Two contained
pieces landed, both gated byte-identical, both needed by the conversion anyway.

## Brick 1 — the transpose fused into the epilogue

The NHWC convolution produces `[OHW, Cout]` and the graph wants `[Cout, OHW]`.
Doing that as its own `t()?.contiguous()?` is a full pass over the activation
that the epilogue is about to make regardless. `epilogue::apply_transposed`
does both in one pass, blocked over pixel tiles so each `TILE * Cout` slab is
L1-resident and the strided read is paid once rather than `Cout` times.

| | wrap ratio | island TOTAL |
|---|---:|---:|
| separate transpose | 0.118x | 0.695x |
| **fused** | **0.267x** | **0.755x** |

Unit-tested against transpose-then-epilogue as an equality, not a tolerance.

## Brick 2 — the weight transpose cached, found in the RESIDUE

`Wt[K, Cout]` was rebuilt on every call. It never appeared in `im2col`, `gemm`
or `wrap` — it showed up only as **0.046 s of the NHWC arm's total that no scope
claimed**. `codec-measurement` §6: a residue no scope claims is the profiler
asking where you did not look.

Cached by candle's tensor id, which is unique and stable, with the cache holding
a `Tensor` so the id cannot be recycled.

| | unscoped residue gap | island TOTAL |
|---|---:|---:|
| per-call transpose | +0.046 s | 0.755x |
| **cached** | **-0.001 s** | **0.805x** |

The residue closing to zero is the confirmation the fix landed rather than moved.

## Where the arm stands, on a quiet box (4-12 % spread)

| stage | NCHW | NHWC | ratio |
|---|---:|---:|---:|
| TOTAL | 0.758 | 0.924 | 0.805x |
| `>im2col` | 0.077 | 0.232 | **0.328x** |
| `>gemm` | 0.209 | 0.170 | **1.196x** |
| `>wrap` | 0.020 | 0.071 | 0.267x |

Progression: **0.452 -> 0.919 -> 0.695 -> 0.755 -> 0.805x**.

**im2col is now the entire remaining gap.** The GEMM win is stable at 1.196x
across every quiet measurement, matching the original isolated 1.140x.

## The remaining step, and its exact price

| | | |
|---|---:|---:|
| island today | 0.924 s | 0.805x |
| + NHWC-input im2col (0.81x) | 0.754 s | 1.005x |
| + no transpose in the epilogue | 0.703 s | **1.078x** |
| − NHWC channel plumbing (1.98x) | 0.730 s | **1.038x** |

**Both remaining steps need the same thing: consecutive convolutions keeping the
activation in NHWC.** That removes the transpose AND makes im2col read
contiguously — which is why neither can be bought separately and why the island
has a floor it cannot cross.

Net for the full conversion: **~3.8 % of detect**, consistent with the earlier
3-5 % once the loaded-box readings were discarded.

## A shortcut that was tried and does not work

candle is backed by `gemm`, which accepts arbitrary strides, so `col.t()` handed
straight to `matmul` might have given the orientation with the fast NCHW im2col
and no conversion at all. Measured on four real shapes: **0.35x, 0.53x, 0.70x,
1.07x** — candle materialises it, and worse than doing it explicitly. Dead.

## Status

`FFAI_DIANA_NHWC=1` is **still OFF by default and still a net loss at 0.805x**.
Nothing here changes the shipped path. What landed is two real bricks the
conversion requires, each gated byte-identical, plus the exact remaining price.

---

# The run conversion, built and measured — and the architecture is too short

The last step was built: `conv3x3_pair_nhwc` keeps the activation NHWC across
two consecutive convolutions, so the pair pays **one** transpose instead of two
and its second im2col gathers **contiguously** (`im2col_from_nhwc`
`copy_from_slice`s `Cin` floats per (pixel, tap) instead of striding). The
weight is permuted to a tap-major K once at load (`weight_nhwc_cached`).

Wired into `Bottleneck`, which is the one place in this graph where both ends of
a layout region are known statically. **Gate: 55 detections both arms, 0
mismatches, 0.0000 px, conf delta 0.000000.**

| mode | TOTAL | reps |
|---|---:|---|
| island + pairs (`=1`) | 0.810x | 0/6 |
| **pairs only (`=pair`)** | **0.894x** | **0/8, z = -2.83** |

Pairs-only is the honest dispatch — convert the runs, leave isolated
convolutions alone — and it is **10.6 % slower**, consistently, on a 6 % box.

## Why, in per-convolution costs

Measured over 1,170 conv-calls:

| | per conv |
|---|---:|
| im2col | 70.1 us |
| gemm | 192.3 us |
| epilogue | 17.9 us |

| | |
|---|---:|
| saving per convolution INSIDE a run | **+38.2 us** |
| ENTRY penalty — im2col from NCHW is ~3x | **-153.5 us** |
| EXIT penalty — fused transpose vs plain epilogue | **-41.8 us** |

**Break-even run length: 5.1 consecutive convolutions.**

| run | net |
|---|---:|
| 2 (Bottleneck) | -118.9 us — LOSES |
| 4 (C3k, both bottlenecks chained) | -42.6 us — LOSES |
| 6 | +33.8 us — pays |

## The verdict, and it is architectural

**YOLO26 does not contain runs of 5 consecutive convolutions.** Every sequence
is broken within two or three by a 1x1, a channel split, a concat or a residual
add. `Bottleneck` gives 2; chaining across `C3k`'s two bottlenecks gives 4 at
the absolute best. Both are under the bar.

The entry cost is what kills it: producing `[OHW, K]` from an NCHW activation is
~3x producing `[K, OHW]`, because the destination stride is wrong no matter the
loop order (three orders were measured: 0.176x, 0.278x, 0.315x). A run pays that
once, and needs five convolutions of +38 us to earn it back.

**This is not a tuning failure and no longer-run heuristic rescues it.** The
prize was real — the GEMM orientation is worth 1.196x and that held across every
quiet measurement — but the cost of ENTERING the layout exceeds what this
architecture's run lengths can repay.

## What is kept, all off by default

* `conv3x3_nhwc` (`=1`) and `conv3x3_pair_nhwc` (`=pair`), both gated
  byte-identical, as the A/B arms and as a working NHWC implementation should a
  future architecture with longer runs make it worth revisiting.
* `epilogue::apply_transposed` — transpose fused into the epilogue, unit-tested
  as an equality. Worth 0.695x -> 0.755x on the NHWC path.
* `epilogue::apply_nhwc` — the NHWC-native bias+SiLU.
* `weight_t_cached` / `weight_nhwc_cached` — both permutations hoisted to load.
* `im2col_from_nhwc` — the contiguous gather, which is the piece that would
  matter if the runs were ever long enough.

**The shipped NCHW path is untouched and nothing has regressed.**

## The number that should have been computed first

Break-even run length is `(entry + exit) / per-conv-saving`. Every input to it
was measurable before a single line of the pair path was written: the entry
penalty from the three im2col variants, the exit penalty from the transpose
probe, and the per-conv saving from the GEMM ratio. **It would have read 5.1
against an architecture offering 2, and priced the whole conversion out in an
afternoon** — `codec-measurement` §11, prune on arithmetic before building.

---

# Winograd, priced before it was built — dead on arithmetic

The largest remaining lever, given the treatment the NHWC campaign earned the
hard way: **compute the break-even first.** Total cost, twenty minutes.

## What F(2x2,3x3) changes, and what it does not

It produces a 2x2 output tile from a 4x4 input tile with **16 element-wise
multiplies instead of 36** — a 2.25x reduction. Those multiplies become 16
independent GEMMs, one per position in the tile:

| | current | Winograd |
|---|---|---|
| GEMMs per conv | 1 | **16** |
| M | cout | cout — **unchanged** |
| K | 9 * cin | **cin** (9x smaller) |
| N | H*W | **H*W / 4** |

**M does not change**, and M is the only axis that governs this graph's GEMM
efficiency — the per-layer roofline found +0.823 correlation with log2(cout)
and −0.048 with operand size. Winograd shrinks the two dimensions that were
never the problem and leaves the one that is.

## Measured on the real shapes

| | 1 big GEMM | 16 small | ratio |
|---|---:|---:|---:|
| weighted per image | 5.022 ms | 5.015 ms | **1.00x** |

**Exactly parity, from 2.25x fewer multiplies.** The whole arithmetic saving is
consumed by the GEMMs being smaller. Transforms — input ~32 adds per (tile,cin),
output ~24 per (tile,cout), priced generously at GEMM throughput, which is a
LOWER bound for an elementwise memory-bound pass — add 0.154 ms.

**NET −0.147 ms/img. Dead before a transform was written.**

## And there is no dispatch either

Four independent runs, sign-consistency required:

| shape | calls | reps | verdict |
|---|---:|---|---|
| 32->32 24x40 | 12 | 1.52 0.70 1.09 1.17 | sign-flips |
| 16->16 48x80 | 5 | 1.17 0.96 2.09 1.35 | sign-flips |
| 64->64 12x20 | 4 | 0.72 0.81 0.69 0.75 | NCHW |
| 16->8 96x160 | 1 | 0.52 0.46 0.43 0.47 | NCHW |
| 128->16 24x40 | 1 | 0.74 0.72 0.80 0.83 | NCHW |

**0 of 8 shapes reproducibly favour Winograd**; 5 always prefer the shipped
path. Dispatch prize 1.000x.

The two shapes that looked like winners on the first run are the two with the
highest call counts — exactly the pattern that would have made this feel like a
discovery. They sign-flip.

## Why this was predictable, and the standing conclusion

Winograd attacks the MULTIPLY COUNT. This graph is not multiply-bound at these
shapes — it is bound by GEMM shape efficiency, which the roofline established
and which every subsequent probe confirmed. A technique that trades multiplies
for more, smaller GEMMs is pushing on the wrong axis, and one afternoon of
measurement says so where a week of implementation would have said it louder.

**Every structural lever this campaign identified is now closed with a
mechanism**, and they all close for the same reason: the convolution GEMMs are
small in M, M is fixed by the architecture, and nothing that leaves M alone
reaches the problem.

## Winograd, RE-OPENED — the "dead" verdict compared a 12-thread arm to a 2-thread one

The verdict above was reached on WALL time. Asked whether fewer multiplies is
worth anything when it is not faster, the honest instrument is **CPU-seconds**,
because that is what energy tracks. Measuring it broke the verdict.

| arm | cpu s | wall s | **cores busy** |
|---|---:|---:|---:|
| 1 big GEMM | 4.33 | 0.347 | **12.4** |
| Winograd, candle batched | 1.30 | 0.680 | **1.9** |

**candle's batched matmul barely parallelises.** The 16 tile-position GEMMs are
independent — embarrassingly parallel — and nothing was fanning them out. The
first comparison therefore put a 12-thread arm against a ~2-thread one and
called the slower one dead. That is a work-parity failure in spirit if not in
letter, and the tell was sitting in a column that was never printed.

### Fanned out over rayon, 6 ABBA reps

| arm | wall (spread) | cpu (spread) |
|---|---:|---:|
| big GEMM | 0.347 (6 %) | 4.33 (18 %) |
| **Winograd parallel** | **0.310 (25 %)** | **3.75 (36 %)** |
| Winograd serial | 0.680 (10 %) | 1.30 (89 %) |

**Parallel Winograd: 1.117x wall, faster in 5/6 reps, and 0.87x the CPU.**
Better on both axes. `z = +1.63`, so it is a lead rather than a verdict — the
25 % spread means this needs N >= 20 before anyone claims a number.

### Two real operating points, and this is the interesting part

| | wall | CPU |
|---|---:|---:|
| latency-optimised: parallel Winograd | **1.12x** | 0.87x |
| CPU-lean: serial Winograd | 0.51x | **0.30x** |

The serial form is exactly "less processing, not faster" — **a third of the
CPU for half the speed.** For a shared server packing many streams that is a
better trade than the fast one, and it sits directly on the positioning the
CPU-time work established: Diana already does the job for ~a third of
Ultralytics' CPU, and this would extend that on the 3x3 layers.

**That is a genuine dispatch axis** — not by shape, but by what the deployment
is optimising. It is the first one this campaign has found that survives its own
measurement.

### What is NOT established

* **GEMM only.** No transforms, no kernel, nothing built. Transforms were priced
  at ~3 % of GEMM time and are not in these numbers.
* **z = +1.63 at 6 reps** with a 25 % spread. Tonight's repeated lesson is that
  favourable numbers shrink on re-measurement; this one needs N >= 20.
* **Sized against the whole engine it is small**: 3x3 stride-1 convolutions are
  22.9 % of detect, so 1.12x on their GEMM is roughly 2-3 % of detect for the
  latency arm.
* The 0.30x CPU figure is the one worth chasing, and it is the one nobody would
  have looked for while measuring wall.

### The lesson, again and more expensively

**"Cores busy" is CPU/wall and costs nothing to print.** It was absent from
every probe in this campaign until the last one, and the moment it appeared it
overturned a verdict reached an hour earlier. A parallelism column belongs in
every arm-vs-arm table by default, because two arms at different thread counts
are not comparable and nothing else in the output says so.

---

# CPU audit: which levers decided on WALL are wins on CPU?

Every refutation in this campaign was decided on wall time. The Winograd
correction says that is half a measurement, so all of them were re-run with
CPU-seconds and a cores-busy column. 4 ABBA reps each against the shipped
default, 30 frames.

| arm | wall ms/f | cpu ms/f | cores | wall x | **CPU x** | |
|---|---:|---:|---:|---:|---:|---|
| **SHIPPED DEFAULT** | 38.49 | 91.93 | 2.39 | 1.00 | 1.00 | |
| epilogue out-of-place | 45.50 | 92.71 | 2.04 | 0.85 | 1.01 | no saving |
| im2col tiling | 54.69 | 108.85 | 1.99 | 0.70 | **1.18** | worse on both |
| direct convolution | 44.59 | 87.50 | 1.96 | 0.86 | **0.95** | real trade |
| 2 threads | 51.78 | 77.86 | 1.50 | 0.74 | **0.85** | real trade |
| 1 thread | 78.41 | 72.40 | 0.92 | 0.49 | **0.79** | real trade |

## Two verdicts are confirmed rather than overturned

* **Epilogue in-place** is the right default on BOTH axes: out-of-place is
  0.85x wall for no CPU saving at all (1.01x). The counted work removal that
  justified it — 81 SliceOps and 39.6 MiB/image — was real, and the CPU column
  now says so where the clock never could.
* **im2col tiling** is 0.70x wall AND 1.18x CPU. Refuted three times on wall,
  and worse on the axis nobody had checked. That one is closed for good.

## Three are genuine "less processing, not faster" trades

**Thread width is the big one, and it was hiding in plain sight.** The campaign
established that one image wants ~4 workers rather than 24, and later refuted
per-LAYER width. Both are about going WIDER. Going NARROWER was never priced:

| threads | wall | CPU |
|---|---:|---:|
| 1 | 0.49x | **0.79x** |
| 2 | 0.74x | **0.85x** |
| **4 (shipped)** | **1.00x** | **1.00x** |
| 8 | — | 1.51x |
| 12 | — | 2.16x |

**2 threads is 15 % less CPU for 35 % more wall.** For a host packing many
concurrent streams that is very likely the better setting, and it is the same
phenomenon the throughput result already reports from the other end — the
structure that loses latency wins throughput.

**Direct convolution: 0.95x CPU at 0.86x wall.** Marginal, and it matches the
kernel's own premise — it never builds the 9x-expanded operand, so it moves far
less memory. The wall refutation stands; the CPU column says the traffic
argument was right even though the clock was not.

## What changes

Nothing ships differently. `FFAI_DIANA_THREADS` already exposes the knob; what
was wrong was the DOCUMENTATION, which called 4 optimal rather than naming the
trade. A latency-bound single stream wants 4; a throughput-bound host packing
streams should measure 2.

## The audit is the transferable part

**Every lever refuted on a stopwatch deserves one CPU-time re-run**, and it is
cheap — five levers, one harness, minutes. Two of tonight's refutations got
stronger, one got a caveat, and two settings turned out to be operating points
rather than losers.


## 2026-08-06 — the tail is the MACHINE, and Diana is the robust one

Standing claim, now retracted: *"Diana's latency tail is heavier than
Ultralytics'."* It is not, and the runs that said so were measuring the desktop.

Three arms, same 220 frames of MOT17-09, engines ALTERNATED per frame so
neither owns a quiet moment, Normal priority throughout.

**Quiet box** — no tail on either engine, and they are indistinguishable:

| engine | p50 | p99 | max | p99/p50 |
|---|---|---|---|---|
| diana | 44.0 | 51.3 | 61.4 | **1.17** |
| ultralytics | 37.8 | 45.0 | 47.4 | **1.19** |

**Same box, 16 synthetic burn threads** — both degrade, very unequally:

| engine | p50 | p99 | max | p99/p50 |
|---|---|---|---|---|
| diana | **89.3** | **168.7** | 357.1 | **1.89** |
| ultralytics | **171.5** | **934.4** | 1424.4 | **5.45** |

Under load Diana is **1.92x faster on the median** and its tail ratio is a third
of the reference's. Quiet-to-loaded degradation is 2.0x for Diana against 4.5x
for Ultralytics.

**What the tail actually is.** Priority alone removes it, with identical work
and identical CPU time (200 frames, rusty_alloc build):

| priority | p50 | p99 | max | cpu_s | cores_busy |
|---|---|---|---|---|---|
| Normal | 64.0 | 384.2 | 408.1 | 29.5 | 2.05 |
| High | 64.7 | **100.5** | **113.4** | 30.0 | 2.43 |
| RealTime | 68.9 | 147.9 | 177.8 | 37.3 | 2.17 |

Same median, same CPU, 3.8x better tail — so the spikes are timeslices lost, not
work done. Note `cores_busy ~2.1`: Diana is a two-core workload, not the
28-thread fan-out previously assumed, so it never saturates the box. And
RealTime is WORSE than High; the answer is High, not "more".

**Two wrong attributions this cost.** The tail was first blamed on
`rusty_alloc` (sequential run: p99 618 vs mimalloc 204) — ABBA-interleaved it
reads 409 vs 480 in rusty_alloc's favour, and the effect was entirely which arm
ran while a neighbouring build was hot. It was then blamed on Diana, from a
single contended run showing mean/median 1.93 against Ultralytics' 1.23; that
does not reproduce, and under controlled load the ordering is emphatically the
other way.

**Rule.** A latency tail measured at Normal priority on a shared desktop is a
measurement of the desktop. Pin the priority or report the CPU time, and never
attribute a tail to a component without an arm that holds that component fixed.
