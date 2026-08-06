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
