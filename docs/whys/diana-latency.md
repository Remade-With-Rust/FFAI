# Diana latency — the six-whys descent

**Unknown:** Diana's per-image detect latency trails Ultralytics ~2.5x and
ONNX Runtime ~3-5x, on the same corpus at matched geometry. The speed gate
reads FAIL and has since M-D2 opened.

Depth 6 first, per `codec-six-whys-unknowns`, because three consecutive
descents in this repo terminated there.

---

## D6a — do both arms time the same span?

- **ASKED:** the bench hands our engine a pre-decoded `ImageBuffer`
  (`crates/ffai-bench/src/runner.rs`, the `decoded` loop) and times the
  reference on `model.predict(path)`, which reads and decodes the file
  inside the timed region. Is that worth anything?
- **MEASURED:** `tools/diana_d6_span.py`, 24 ABBA-interleaved rounds of
  8 images, `predict(path)` against `predict(pre_decoded_array)`.
  **12/24, z = +0.00.** Median share 6.5 %, but individual rounds ranged
  -73 % to +25 %.
- **ANSWER: INCONCLUSIVE — and the inconclusiveness is the finding.** The
  spread swamps the effect. Recorded as unresolved rather than as "6.5 %",
  because 6.5 % is exactly the sort of number that would get quoted.
- **CONFIDENCE:** high that it is unresolved; the run's own absolute figures
  prove why (below).
- **DIRECTION:** note the bias runs AGAINST the reference — it does strictly
  more work than we do inside the timed region — so whatever this is worth,
  it makes our published gap flattering rather than harsh.
- **STATUS:** open, blocked on a quiet box.

## D6b — is the instrument sound at all right now?

- **ASKED:** noise floor.
- **MEASURED:** during the v3 reference sweep, the same reference that the
  ledger records at **67 ms/image** measured **252 ms** in one probe and
  **19 372 ms** in another. Our own engine's `cpu/wall` occupancy read
  3.32x in a window where 12 of 12 images should have been consistent.
- **ANSWER: no wall-clock measurement is valid while a benchmark is running
  on this box.** A 289x artifact is not a noise floor, it is an unusable
  instrument.
- **CONSEQUENCE:** the entire D1-D5 descent is gated on the box being quiet.
  Sequenced accordingly: mAP work first (load-independent), latency after.
- **CONFIDENCE:** certain.

## D6c — are both arms configured to the same parallelism? ★

- **ASKED:** thread counts, as a configuration difference rather than a code
  one. This is the class of finding that closed three previous descents.
- **MEASURED:**

  | | threads |
  |---|---:|
  | `torch.get_num_threads()` | **8** |
  | `torch.get_num_interop_threads()` | 16 |
  | `rayon::current_num_threads()` (ours) | **24** |

  Machine: **16 physical cores, 24 logical.** That combination is an Intel
  hybrid — 8 P-cores with SMT (16 logical) plus 8 E-cores (8 logical).
  **Torch's 8 is exactly the P-core count.** It is not a default that
  happens to be low; it is the topology-aware choice.

- **ANSWER:** we run every fork-join across 24 threads including 8 E-cores
  and both SMT siblings of each P-core. Our kernels (`dwconv`, `conv3x3`
  im2col, `silu`) are `par_chunks_mut` — **barriers**. A barrier's cost is
  set by its SLOWEST participant, so scheduling equal-sized chunks onto
  cores of unequal speed makes every barrier wait on an E-core.
- **HYPOTHESIS (H1):** restricting rayon to the P-cores lowers per-image
  latency despite using a third of the threads.
- **TEST:** `RAYON_NUM_THREADS` sweep — no code change, so the probe costs
  nothing but a quiet box.
- **PRIZE (arithmetic, before building):** unknown until occupancy is
  re-measured quiet. If our real occupancy is ~3.3x of 24, we are 14 % busy
  and perfect occupancy is worth 7.2x — far more than the 2.5x gap. That
  would mean the gap is **scheduling, not work**, and it would invert the
  campaign from "write faster kernels" to "stop over-subscribing".
- **CONFIDENCE:** medium-high on the mechanism, unmeasured on the size.
- **SPAWNED:** D6d (is equal-sized chunking the real defect, rather than the
  thread count? Unequal cores want unequal chunks, or work-stealing with
  more, smaller tasks).
- **STATUS:** open, blocked on a quiet box.

## D6d — H1 tested: does cutting threads help WALL? **REFUTED**

- **ASKED:** if 24 threads costs 2.32x the CPU that 1 thread does, is a
  smaller pool faster in wall time — the thing the gate measures?
- **MEASURED:** `examples/pool_ab.rs`, one process, two rayon pools, ABBA
  per round, 24 rounds. **8 threads faster in 9/24, z = -1.22, median A/B
  0.979x.**
- **ANSWER: no.** For a SINGLE image the fan-out is worth its barrier cost,
  because the alternative is leaving 15 cores idle. H1 is refuted as a
  latency lever.
- **BUT the CPU saving is real** and belongs to a different objective — see
  D6e. A refutation constrains the claim it tested, not the idea it came
  from.
- **CONFIDENCE:** high. Paired design, |z| < 2 with N = 24.

## D6e — the same knob, the other objective. Kept, PARTIALLY

- **ASKED:** in `detect_batch`, images already fill every core, so the
  per-layer fan-out nested inside them should be pure overhead. Worth what?
- **PRIZE, computed first:** serial work is 363 ms/image against 844 ms at
  24 threads, so the arithmetic said up to 2.32x.
- **MEASURED:** `examples/batch_ab.rs`, 21 ABBA rounds of a 24-image batch,
  re-exec per arm so the env knob is honoured.

  | metric | verdict |
  |---|---|
  | CPU/image | B cheaper **20/21, z = +4.15, median 1.080x** — REAL |
  | wall | B faster 10/21, **z = -0.22**, median 0.983x — inside the noise |

- **ANSWER: 8 %, not 132 %.** The prize arithmetic was wrong and the reason
  is worth keeping: **rayon's work-stealing already collapses most nested
  fan-out when the workers are busy.** The 2.32 x overhead is real in the
  SINGLE-image path, where the pool genuinely spreads, and mostly absent in
  the batch path. Pricing an optimisation with a number measured in a
  different regime is the same error as pricing it in a different shape.
- **DISPOSITION: KEPT.** Confirmed on the load-robust metric, no effect on
  the latency path (`serial_scope` is only entered from `detect_batch`),
  identical detections (57 = 57), and the old behaviour stays reachable via
  `FFAI_DIANA_NESTED_PAR=1` so the decision can be re-measured rather than
  re-argued. **Wall is UNPROVEN, not proven-neutral** — this box could not
  resolve 8 %.
- **CONFIDENCE:** high on CPU, none on wall.

---

## D3 — which stage owns the 363 ms of real work?

`RAYON_NUM_THREADS=1 FFAI_PROFILE=1`, n tier, 15 runs, median 288.6 ms.
Ranked by ABSOLUTE cost, per the skill's first learning:

| stage | share | ms/image |
|---|---:|---:|
| backbone | 53.0 % | 152.6 |
| neck | 25.2 % | 72.7 |
| head | 16.5 % | 47.6 |
| pre | 2.8 % | 8.2 |
| decode | 2.1 % | 6.1 |
| residue | 0.3 % | 1.0 |

The nested breakdown is where the answer is, and it is not a convolution:

| op | share | calls |
|---|---:|---:|
| **silu** | **30.9 %** | 1305 |
| 3x3 s1 | 18.6 % | 480 |
| gemm | 17.7 % | 585 |
| 3x3 s2 | 16.5 % | 105 |
| 1x1 | 15.7 % | 735 |
| im2col | 13.5 % | 585 |
| attention | 2.7 % | 30 |
| depthwise | 1.7 % | 120 |

**★ The activation is the largest single line item.** That is the third time
this repo has found it — Mercury's `tanh` GELU at 39 % of the encoder,
Diana's own earlier SiLU find at 25.8 %, and now 30.9 % again after that fix
shipped. The lesson from the skill applies literally: *the same class of bug
does not recur in the same place, so pattern-matching to the last bug is how
you miss this one* — except here it recurred in the SAME place, because the
previous fix removed one libm call and left another.

## D5 — why is a pointwise streaming op the most expensive thing here?

- **MEASURED:** `examples/silu_ceiling.rs`, 16 M elements, best of 7, single
  thread. The shipped kernel runs at **1.03 GB/s**. This machine's measured
  memory bandwidth is **17.6 GB/s**. A read-4-write-4 elementwise function
  17x below memory speed is not vectorised at all.
- **MECHANISM:** the module's own doc says it removed `exp` because "a
  scalar libm call blocks vectorization of the whole loop". It then calls
  **`f32::round`**, which is ties-AWAY-FROM-ZERO — a mode no x86
  instruction implements, since `vroundps` is ties-to-even. So the libm call
  came back under a different name, in the same loop, for the same reason.
- **VARIANTS**, all bit-identical to the shipped kernel over the activation
  range (max relative disagreement exactly **0.000e0**):

  | variant | ms / 16 M | GB/s | vs shipped |
  |---|---:|---:|---:|
  | `round()` (shipped) | 130.90 | 1.03 | — |
  | `round_ties_even()` | 104.22 | 1.29 | 1.26x |
  | **magic-number `+1.5*2^23`** | **67.45** | **1.99** | **1.94x** |

- **PRIZE:** SiLU is 30.9 % of detect, so 1.94x on it is **15.0 % off the
  pipeline (1.176x)**. The noise floor of the profile harness is 9.2 %, so
  this is resolvable. A *perfect* SiLU would be worth 1.447x — recorded as
  the hard ceiling so the next person knows what is left.
- **DISPOSITION: SHIPPED AND GATED.** Bit-identical output means the
  five-tier oracle is the correctness gate and it passes unchanged. The
  SPEED claim is gated one level up, on the serial path the bench actually
  times, paired ABBA with the old rounding behind
  `FFAI_DIANA_SILU_ROUND=1`:

  | metric | result |
  |---|---|
  | wall | **17/21, z = +2.84, median 1.0787x — REAL** |
  | CPU | 12/21, z = +0.65, median 1.023x — inside the noise |

  **1.079x on the pipeline against 1.176x predicted.** Half the microbench's
  prize survived contact, which is the expected discount and worth writing
  down: the microbench streamed a cold 64 MiB buffer, while the real
  activations are smaller and partly cache-resident, so the isolated kernel
  was more memory-bound than the in-context one. `Isolation misleads in
  both directions` — here it over-promised by 2x, and the level-above probe
  is what caught it.

  That CPU time did NOT move while wall did is consistent with D6e: at 24
  threads the fork-join overhead dominates the CPU column and swamps a 7 %
  work saving, while wall sees the shortened critical path.
- **★★ THE TABLE ABOVE IS WRONG — the harness was measuring itself.** Two
  defects, both mine, both while quoting the skill that names them:

  1. `bench` took `f: fn(f32) -> f32`, a function POINTER — an indirect
     call per element, so nothing inlined or vectorised in ANY arm.
  2. Once that was made generic, LLVM deleted every loop, because `dst` is
     never read. All five arms reported **0.00 ms**, which is the
     instrument screaming rather than a result.

  Fixed by monomorphising and adding `std::hint::black_box(&*dst)` after
  the loop **identically in every arm**, including the roofline — an
  asymmetric `black_box` is itself a listed way to change what vectorises
  in one arm only. Re-measured:

  | variant | ms / 16 M | GB/s | vs shipped |
  |---|---:|---:|---:|
  | memcpy (roofline) | 5.58 | **24.04** | — |
  | `round()` (was shipped) | 60.84 | 2.21 | — |
  | `round_ties_even()` | 38.85 | 3.45 | 1.57x |
  | **magic-number** | **12.91** | **10.40** | **4.71x** |
  | magic + `max`/`min` | 13.61 | 9.86 | 4.47x |

  **The real kernel win is 4.71x, not 1.94x**, and at 10.4 GB/s against a
  24 GB/s copy the fixed kernel is within 2.3x of pure memory traffic — it
  DOES vectorise, and `round()` was what stopped it. The earlier "17x below
  bandwidth / not vectorising at all" was the function pointer.

  **What survived the correction: the RANKING.** Every arm paid the same
  indirection, so the ordering that selected the change was right, and the
  in-context gate (1.079x, z = +2.84) never depended on the microbench at
  all. What did not survive: every magnitude, and the "a perfect SiLU is
  worth 1.447x" ceiling, which was computed against a corrupted v0. Left in
  the log rather than quietly rewritten, because *isolation misleads in both
  directions* is a rule I was reciting at the time.

- **"A concurrent batch path PyTorch's GIL cannot match": WITHDRAWN, then
  RE-EARNED at a smaller size.** The original 22.73 vs 17.29 img/s was a
  single-run wall comparison taken before the null arm and the ordering
  confound were known, and it was withdrawn. Re-measured properly
  (`tools/diana_throughput.py`, min-of-N, arms alternated, counts compared)
  it holds at **~1.6x ahead at every tier** — n 1.61x, s 1.72x, m 1.66x,
  l 1.62x, x 1.52x.

  **Crucially the reference is given its BEST configuration, not its
  default.** `predict(list)` defaults to batch=1 and loops at 15.76 img/s;
  explicit `batch=4` reaches 18.79. Measuring the default would have made
  our margin look like 1.58x against a hobbled baseline. Caveat kept in the
  open: detection counts differ by 1-3 of 68-86 at conf 0.25 (n exact) —
  the same threshold-boundary effect as the 0.08 pp mAP delta.

  **The old, withdrawn form for reference:** It was a wall-clock
  throughput comparison on a loaded box, single run, measured before the
  null arm established a 27 % resolution and before the ordering confound
  was known. The batch API is real and is still described; the comparative
  is not re-verified and no longer claimed. Re-measure it with the paired,
  order-reversed instruments if it is wanted back.

- **`f32::clamp` as the remaining blocker: REFUTED.** On the corrected
  harness `max`/`min` is 13.61 ms against clamp's 12.91 — the clamp is
  marginally FASTER, i.e. it was never blocking anything. Refuted on the
  good instrument, which is the only reason this is stated as closed.

- **STILL OPEN:** the fixed kernel sits 2.3x off a memory copy. That is a
  reasonable place for a transcendental and the remaining headroom is
  small; the next latency lever is NOT here. It is `target-cpu` (D6f) and
  the ~120 fork-joins per image (D6c).

## D6f — ★ the build has no `target-cpu`

- **ASKED:** the standing depth-6 question — is the measurement comparing
  like with like, or is the configuration doing the work?
- **MEASURED:** there is no `.cargo/config.toml`, `RUSTFLAGS` is unset, and
  `[profile.release]` sets only `lto = "thin"`. **Every hand-written kernel
  in this crate is compiled for the x86-64 baseline: SSE2, no AVX2, no FMA,
  no `vroundps`.** Rebuilding with `-C target-cpu=native` moved the shipped
  SiLU from 130.90 ms to 94.35 ms — **1.39x, from a build flag.**
- **SCOPE, stated so it is not over-claimed:** candle's GEMM comes from the
  `gemm` crate, which dispatches on CPU features at RUNTIME, so the matmuls
  were never affected. This is a tax on OUR kernels specifically —
  `dwconv`, `conv3x3`, `silu` — which is 34 % of the profile by itself.
- **NOT simply fixed by setting the flag.** `target-cpu=native` bakes the
  build machine's ISA into the binary; for a crate published to crates.io
  that is a portability bug, not an optimisation. The correct shape is
  runtime dispatch (`is_x86_feature_detected!` + `#[target_feature]`) —
  which is a real piece of work, so it was priced before being started.

- **★★ PRICED AT THE PIPELINE LEVEL: WORTH NOTHING. Do not build it.**
  `tools/diana_native_ab.sh` alternates two BINARIES of identical source —
  baseline x86-64 against `-C target-cpu=native` — ABBA over a whole detect
  pass on the serial path the gate times.

  **12/21, z = +0.65, median 1.0171x. Inside the noise.**

  The 1.39x was real and it was measured **on the old SiLU kernel**, where
  AVX2's contribution was mostly making `f32::round` less catastrophic.
  Once the magic-number rounding shipped, the polynomial vectorises 4-wide
  under plain SSE2 and the flag has almost nothing left to buy: what remains
  is GEMM (the `gemm` crate dispatches on CPU features at RUNTIME and was
  never affected) and im2col (memory-bound).

  **A confirmation expires when its baseline moves.** The skill states this
  for refutations — "int8 was refuted against f32; once f16 shipped, the
  same idea's prize halved" — and the symmetric case is what happened here:
  a confirmed lever, measured honestly, was made worthless by a change
  landed twenty minutes later in the same session. Had the two been done in
  the other order, runtime ISA dispatch would have been built on a 1.39x
  that no longer existed.

- **STATUS: CLOSED, negative.** Recorded as a measured prune, not as a
  backlog item.

---

## ★★★ D6g — THE NULL ARM. Per-tier speed verdicts are not resolvable in one run

Run this before believing any cross-implementation ratio on this box,
including the ones above.

- **ASKED:** what is the resolution limit of the bench's engine-vs-reference
  comparison? Not "is the box noisy" — what delta can this harness actually
  distinguish?
- **MEASURED:** two full v3 runs of `yolo26n`, **same code, same corpus, same
  configuration**, hours apart. That is a null arm: any difference is the
  instrument.

  | run | our p50 | ref p50 | ratio | ref throughput |
  |---|---:|---:|---:|---:|
  | `bench-detect-1785551014` | 132 ms | 75 ms | **1.76x** | 12.73 img/s |
  | `bench-detect-1785595308` | 123 ms | 55 ms | **2.24x** | 17.38 img/s |

- **ANSWER: the headline ratio moved 27 % with nothing changed, and the
  REFERENCE's own throughput moved 37 %.** The denominator drifts further
  than most of the effects being reported.

- **★ WHAT THIS INVALIDATES, including my own correction.** The blanket
  claim "speed FAILS, ~2.5x behind" was wrong because it generalised the n
  tier. But the fix I proposed — "the gate PASSES at m and we are at parity
  at x" — is wrong the same way, in the other direction:

  | tier | single-run ratio | inside a +/-27 % band? |
  |---|---:|---|
  | n | 2.24x | outside — really behind |
  | s | 1.67x | outside — really behind |
  | m | 1.17x | **INSIDE — not resolvable** |
  | l | 1.06x | **INSIDE — not resolvable** |
  | x | 0.94x | **INSIDE — not resolvable** |

  The harness's own per-row `speed: pass` at m rests on a 9 % margin against
  a 27 % resolution. It is a coin flip wearing a gate.

- **I THEN CLAIMED THE TREND SURVIVED. IT DOES NOT — see D6h.** The ratio
  goes 2.24 -> 1.67 -> 1.17 -> 1.06 -> 0.94 across five tiers, a 2.4x span
  against a 27 % per-point resolution, and it looked like a real finding
  because its range exceeded the noise by ~9x. It is an artifact of the
  sweep's ORDER.

- **CONSEQUENCE:** per-tier verdicts need a load-robust instrument. CPU time
  is the one this campaign already trusts (`tools/diana_cpu_ratio.py`), and
  the structural prediction is separately falsifiable
  (`tools/diana_overhead_amortization.py`).

- **METHOD NOTE:** the null arm cost nothing — it was two runs already in
  the ledger. It was not run until the measurement core skill demanded it,
  after two sessions of quoting ratios from single runs. Cheapest instrument
  in the box, and the last one reached for.

## ★★★ D6h — THERE IS NO TIER TREND. Order and tier were confounded

- **ASKED:** the bench sweep runs tiers in the order n, s, m, l, x, and takes
  hours. So tier index and wall-clock time are perfectly correlated. Any
  monotone drift in this box's load over the sweep manufactures a monotone
  "tier trend". Is the trend a property of the models or of the clock?
- **MEASURED:** `tools/diana_cpu_ratio.py`, both arms in-process minutes
  apart rather than hours, median of 9, three warm calls each side — and run
  BOTH WAYS. A real effect survives reversal; drift flips sign.

  | tier | forward (n->x) | reverse (x->n) | mean |
  |---|---:|---:|---:|
  | n | 1.67x | 1.86x | **1.77x** |
  | s | 1.79x | 1.90x | **1.85x** |
  | m | 1.33x | 2.12x | **1.73x** |
  | l | 1.65x | 1.70x | **1.68x** |
  | x | 1.57x | 1.62x | **1.60x** |

- **ANSWER: the gap is FLAT at ~1.6-1.9x across every tier.** The sweep's
  2.24 -> 0.94 was the ordering confound. The m cell alone swung 1.33 -> 2.12
  (59 %) purely on running order, which is the whole story in one number.

- **★ SO BOTH CLAIMS WERE WRONG, AND THE FIRST ONE'S VERDICT WAS RIGHT BY
  ACCIDENT.** "Speed FAILS, ~2.5x behind" overstated the magnitude (~1.75x)
  but reached the right verdict. The correction — "the gate PASSES at m and
  we are at parity at x" — reversed the verdict on an artifact. **A wrong
  claim corrected into a differently-wrong claim is the failure mode this
  whole descent exists to prevent**, and it took the null arm plus an
  order-reversal to catch, neither of which costs more than ten minutes.

- **What is now defensible:** Diana is **~1.75x behind Ultralytics on
  per-image latency, consistently, at every tier** — and does ~1.7x more CPU
  work to get there. The speed gate fails at every tier. No tier is at
  parity.

- **CONFIDENCE:** high. Two orders, two metrics (CPU and wall) agreeing per
  tier, tight pairing, medians, matched warmup.

## D6i — the quiet-box requirement, discharged by pinning instead of waiting

The plan said "re-measure on a quiet box". This box was never quiet — 98 %
with another session's benchmark on it at the time of writing — and waiting
for quiet is unfalsifiable. Every measurement above was therefore taken
under foreign load, which is a shared weakness the three probes do NOT
control for: they vary order and metric, not LOAD.

`tools/diana_pinned_floor.ps1` varies it. Each arm pinned to the same cores
at High priority (removes scheduler migration, the thing that turns a 1.06x
spread into 2.02x), **minimum** per-image wall over repetitions rather than
mean or median (foreign load only ever ADDS time, so the minimum is the
floor of the code's own cost — and the skill's own data shows unpinned and
pinned MINIMA agreeing at 1047 vs 1038 ms even when the spreads do not),
arms alternated on both tier index and rep so ordering cannot align with
tier.

| tier | Diana min | ref min | ratio |
|---|---:|---:|---:|
| n | 118.9 ms | 60.2 ms | 1.98x |
| s | 196.1 | 89.0 | **2.20x** |
| m | 370.4 | 204.0 | 1.82x |
| l | 483.2 | 264.1 | 1.83x |
| x | 837.1 | 542.8 | **1.54x** |

**Mean 1.87x, range 1.54-2.20x, and NOT monotone in tier size — s is the
worst tier, not n.** So the ordering-confound verdict survives a change of
load regime, which is the axis it had not been tested on.

Three instruments now agree on the shape and roughly on the size:

| instrument | mean gap | trend? |
|---|---:|---|
| unpinned median, forward+reverse order | 1.75x | none |
| unpinned CPU work ratio | ~1.7x | none |
| **pinned floor, min-of-N** | **1.87x** | none (s worst, x best, non-monotone) |

The pinned floor reads slightly WORSE for us than the median probe, which is
itself informative: at the floor the reference benefits more from clean
conditions than we do, consistent with our having parallel overhead that
does not vanish when the machine frees up.

**Weak residual signal, stated as weak:** x is the lowest ratio in some wall
instruments and s the highest. It is not parity, it is not monotone, and it
is inside the spread — recorded so nobody re-derives it as a trend.

## D6j — the corpus axis, and the refutation's final tally

Every tight probe above read **9 images from v2** while the bench that
suggested per-tier parity read **450 from v3**. Corpus — and therefore the
image-size distribution the letterbox produces — was the last axis the
refutation had not varied, and it is the axis the published claim rests on.

Pinned floor, min-of-N, arms alternated, **v3 images**:

| tier | Diana min | ref min | ratio |
|---|---:|---:|---:|
| n | 127.6 ms | 78.7 ms | 1.62x |
| s | 287.1 | 103.1 | **2.78x** |
| m | 507.4 | 237.5 | 2.14x |
| l | 593.9 | 256.3 | 2.32x |
| x | 920.5 | 496.9 | 1.85x |

Mean **2.14x**, range 1.62-2.78x, still non-monotone, still nowhere near
parity. x is not the best tier here — n is.

### Tally: seven axes, one answer

| axis varied | outcome |
|---|---|
| order (n->x vs x->n) | no trend |
| metric (wall vs CPU work) | no trend |
| load regime (unpinned vs pinned floor) | no trend |
| statistic (mean / median / min-of-N) | no trend |
| warmup (1 vs 3 calls) | no trend, and fixed a 5x artifact |
| work parity (conf, max_det, detection COUNTS) | comparison valid; found a real parity bug |
| corpus (v2 9-image vs v3 12-image) | no trend |

**"We pass at m and are at parity at x" is refuted, overdetermined.** Across
seven independent axes the gap sits between 1.5x and 2.8x with no tier at
parity and no monotone dependence on model size; the tier that reads worst
changes with the instrument, which is the signature of noise rather than of
an effect.

The refutation standard this repo sets is three varied probes with one at
the level above. This has seven, including the level above (whole-corpus
bench) and a deterministic COUNT (detection counts identical per image).
Nothing further would change it, and the honest headline is **~1.9x behind
at every tier, gate FAILS everywhere.**

## D5b — the amortization mechanism is real, and it does NOT reach the gap

The trend needs a mechanism or it is a coincidence with five points. The
claim is that Diana pays ~120 fork-joins per image whose cost is set by
thread count and barrier latency rather than by tensor size, so bigger
tiers spread a slowly-growing fixed cost over rapidly-growing arithmetic.

That predicts: **overhead should grow much more slowly than the work.**
Measured with `tools/diana_overhead_amortization.py` — CPU ms/image at 1
thread (the work) against 24 threads (work + overhead), which is load-robust
because CPU time does not accrue while descheduled:

| tier | serial work | at 24t | overhead | **tax = overhead / work** |
|---|---:|---:|---:|---:|
| n | 232 | 1115 | 883 | **3.81x** |
| s | 688 | 1677 | 990 | 1.44x |
| m | 1344 | 3664 | 2320 | 1.73x |
| l | 1729 | 5268 | 3539 | 2.05x |
| x | 4177 | 7656 | 3479 | **0.83x** |

**Work grows 18.0x from n to x; overhead grows 3.9x.** The prediction holds
in direction and comfortably in magnitude.

It does NOT hold in the stronger form I first wrote ("roughly constant").
Overhead grows nearly 4x, and it should: m/l/x have more layers (224 and
332 tensors against n's 204) so there are more barriers, and larger tensors
mean more chunks to schedule at each one. The honest mechanism is *a cost
that grows sub-linearly in the work*, not a fixed one.

**★ AND THE CORROBORATION I DREW FROM IT WAS FALSE.** I wrote that the tax
falling 4.81x -> 1.83x "agrees to within 10 %" with the wall gap improving
2.4x over the same span, and called that two instruments confirming each
other. D6h then showed the wall trend does not exist. Two numbers agreeing
is not corroboration when one of them is an artifact — it is a coincidence
that made an artifact look explained, which is worse than leaving it
unexplained.

**What the amortization measurement still supports, on its own:** our
parallel efficiency genuinely improves with model size, from a 3.81x tax at
n to 0.83x at x. That is real, load-robust, and our own numbers on both
sides. It simply does not show up as a narrowing gap against Ultralytics —
which means the reference's efficiency improves with size too, roughly in
step. Both implementations amortize; neither pulls ahead.

That is the honest reading, and it is a smaller claim than the one it
replaces.

**Sanity check that fell out of it:** n's serial work reads 232 ms here
against 363 ms measured before the SiLU fix — 1.56x, in the range that fix
predicted (1.32x from its 30.9 % share, plus the measurement's own spread).
The instrument agrees with a change made hours earlier for unrelated
reasons.

---

## Refuted / parked

- **H1 — "fewer rayon threads lowers single-image latency": REFUTED**,
  9/24, z = -1.22, paired ABBA, N = 24. Scope of the refutation: it kills
  the LATENCY claim only. The CPU-work claim it came from survives and
  shipped as D6e.
- **"The reference's timed span includes decode, so the gap is worse than
  reported": UNRESOLVED, not refuted.** 12/24, z = +0.00. The direction is
  known (the reference does strictly more work); the size is not. Re-run on
  a quiet box.
- **"The batch fan-out costs 2.32x": REFUTED AT THE BATCH LEVEL**, worth
  1.08x. The 2.32x figure is correct for the single-image path and was
  wrongly carried across regimes.

## Sequencing decision

mAP is load-independent; latency is not. The v3 corpus work (items 2 and 3)
therefore runs FIRST and to completion, and the latency descent resumes on a
quiet box. Attempting them concurrently produced the 19 372 ms reading above.

---

## ★★★ D3b — im2col is a MEMORY FLOOR, and that is why threads stopped helping

The parallel-efficiency descent kept finding ops that would not scale and
kept assuming the cause was scheduling. It is not, for the largest of them.

- **ASKED:** per-op speedup from 1 to 24 threads showed gemm at 3.33x, silu
  at 2.52x, but **im2col at 1.55x** and the 3x3/1x1 convolutions that
  contain it at ~1.62x. Overall 1.72x. Why does im2col refuse to scale?

- **FIRST ATTEMPT, REFUTED.** im2col chunks one task per INPUT CHANNEL, and
  counting showed the stem convolution (3 -> 16 at 640x640) produces **3
  tasks for 24 threads** — 21 idle on the largest-spatial layer. Chunking
  per (channel, tap) instead gives 9x the tasks. Measured: serial im2col
  went **0.611 -> 1.366 s, 2.2x WORSE**, and 24-thread 0.394 -> 0.568.
  Reverted. The old per-channel task reads the source plane once and reuses
  it across all nine taps; the finer task re-reads it nine times. **Occupancy
  bought less than locality cost** — and the occupancy defect was real, so
  this is a case where the diagnosis was right and the fix was still wrong.

- **THE INSTRUMENT THEN FAILED OUTRIGHT.** A thread sweep to locate the
  saturation point read im2col at **6.855 s ascending and 1.112 s
  descending at the same thread count**; 8 threads read 0.700 and 2.159.
  Drift exceeded the effect in both directions. No wall measurement on this
  box could answer the question.

- **MEASURED WITH A COUNTER INSTEAD** (`FFAI_DIANA_COUNT=1`), which is
  immune to every timing artifact:

  | tier | im2col writes / image | + GEMM read-back | at 17.6 GB/s | at 24 GB/s |
  |---|---:|---:|---:|---:|
  | n | 108.4 MiB | **216.9 MiB** | 12.9 ms | 9.5 ms |
  | s | — | 419.0 MiB | 24.9 ms | 18.3 ms |
  | m | — | 855.2 MiB | 50.9 ms | 37.3 ms |
  | x | — | 1939.8 MiB | 115.4 ms | 84.6 ms |

- **★ ANSWER: im2col is a BANDWIDTH FLOOR, not a scheduling problem.**
  A 3x3 convolution's im2col buffer is NINE TIMES its input, materialised to
  memory and read straight back by the GEMM. Memory bandwidth is shared
  across cores, so that time does not divide by the thread count — which is
  precisely what a 1.55x speedup on 24 threads looks like.

  At the n tier this is **9.5-12.9 ms per image against a ~66 ms reference
  total**: a seventh to a fifth of the entire budget we are trying to match,
  spent materialising a buffer a direct convolution would never write.

- **CONSEQUENCE — the campaign's direction inverts.** Every lever tried so
  far (thread count, pool size, batch dispatch, per-tap chunking) moved
  work between cores. None of them can move a shared-bandwidth floor. The
  levers that CAN are the ones that move fewer bytes: tile the im2col so
  each block is GEMM'd while still in L2, or skip materialisation entirely
  with an implicit-GEMM/direct convolution.

- **CONFIDENCE:** high, and it needs no quiet box — this is arithmetic on a
  deterministic counter, cross-checked against a memcpy-measured roofline.

## D5c — tiling the im2col: priced BEFORE building, and the price is a curve

The bandwidth floor says the lever is fewer bytes: tile the im2col so each
block is GEMM'd while it is still in cache instead of materialising 108 MiB
and reading it straight back. That replaces one big GEMM per convolution
with several small ones, and small GEMMs have a per-call cost — so the
arithmetic comes first.

Measured (`examples/matmul_overhead.rs`), layer-0 shapes `w[16,27] x
col[27,N]`, best of 9:

| tile N | ms/call | tiles to cover ohw=102400 | total |
|---:|---:|---:|---:|
| 102400 (today) | 1.309 | 1 | 1.309 ms |
| **10240** | 0.097 | 10 | **0.967 ms** |
| 4855 | 0.215 | 22 | 4.721 ms |
| 1024 | 0.056 | 100 | 5.560 ms |

**A sweet spot exists and it is not where intuition put it.** Ten tiles beat
the single call by 1.35x — the block fits cache and the GEMM gets faster
despite ten times the call overhead. Twenty-two tiles is 3.6x WORSE than
ten, and a hundred is worse still. The curve is sharp on the small side.

So the tile size has to target a BYTE BUDGET, not a tile count:
`N = budget / (c_in * 9 * 4)` with budget ~1 MiB. That behaves across the
network because depth trades one for the other — the stem has c_in=3 and
ohw=102400 (N~9700, ten tiles), while a deep layer has c_in=256 and
ohw=400, which is a handful of tiles either way.

**STATUS: specified, priced, NOT BUILT.** What remains is real work — the
convolution has to assemble its output from block results without paying
back the traffic it saved, and the output assembly is the part most likely
to eat the win. It is the correct next brick and it is not a small one.

## Where parity actually stands

Honest arithmetic against the 1.0x target, n tier:

| quantity | value |
|---|---:|
| our serial work | 232-304 ms |
| our wall at 24 threads | ~114-128 ms (pinned floor) |
| reference wall | ~47-79 ms (pinned floor) |
| **im2col bandwidth floor (thread-invariant)** | **9.5-12.9 ms** |
| our parallel efficiency | 1.72x from 24 threads |
| efficiency needed for parity | ~4x |

Parity is **not reached and was not reached in this campaign.** What changed
is that the gap is no longer a mystery: a measurable part of it is a shared
bandwidth floor that no amount of threading can touch, the fix for that is
specified and priced, and the levers that cannot work (thread count, pool
size, per-tap chunking, ISA flags) are refuted with numbers so nobody spends
a week on them.

## ★★ D6k — CORRECTION: the bandwidth floor was priced against the wrong roofline

The im2col finding above is real in KIND and was overstated in SIZE, by
about 2x, and the error is the same one this repo has now made three times:
a roofline measured at one working-set size applied to another.

216.9 MiB/image was priced at **17.6-24 GB/s** — a figure measured with a
64 MiB memcpy, i.e. DRAM. But this machine is an i7-14650HX with **30 MB of
L3**, and the n-tier col buffer is **11 MB per layer**. It fits. That traffic
largely never reaches DRAM.

Bandwidth by working set (`examples/bw_by_size.rs`, best of 9):

| buffer | GB/s |
|---:|---:|
| 1 MB | 68.8 |
| 4 MB | 48.1 |
| 8 MB | 44.2 |
| **11 MB — the layer-0 col buffer** | **41.7** |
| 24 MB | 17.5 |
| 64 MB | 22.8 |
| 128 MB | 14.1 |

**Corrected: 216.9 MiB at 41.7 GB/s is ~5.5 ms, not 9.5-12.9 ms.** Against a
~66 ms reference total that is ~8 % of the budget, not "a fifth to a
seventh".

Two consequences:

1. **The tiling result is now consistent rather than puzzling.** Tiling saves
   a DRAM round trip that was mostly not happening; the prize was roughly
   half what was computed, and the implementation's added copies were larger
   than the halved prize. Measured -24 % CPU, and now explained.
2. **im2col's 1.55x scaling is still unexplained.** The bandwidth story
   covered it at the DRAM price and does not at the L3 price. That question
   is reopened, not closed — the next descent should ask it fresh rather than
   inherit this answer.

The skill's own words, earned again: *"a roofline is only valid for the SHAPE
it was measured at."* Recorded rather than quietly rewritten, because the
overstated version was committed and would otherwise stand.

## ★★★ D3c — THE GAP IS KERNEL WORK, NOT PARALLELISM. The campaign was aimed wrong

Every lever in this descent targeted how work is spread across cores. That
was the wrong axis, and one measurement never taken settles it: **the
reference's SERIAL cost.** Our own serial figure had been in hand for days;
theirs had not.

Torch forced to one thread — verified by occupancy 0.97x, not by the thread
count it reports, because ultralytics resets `set_num_threads` and both the
env vars and process affinity failed to stick:

| | serial wall | serial CPU |
|---|---:|---:|
| Ultralytics, 1 thread | 129.5 ms | **125.0 ms** |
| Diana, 1 thread | 278.8 ms | **281.2 ms** |

**We do 2.25x more WORK.** Not spread worse — more of it.

And the parallel picture inverts with it:

| | serial | threads | wall | efficiency |
|---|---:|---:|---:|---:|
| Ultralytics | 125 ms | 8 | 47-93 ms | **1.3-2.7x** |
| Diana | 281 ms | 24 | 114-128 ms | **2.2-2.5x** |

**Our parallel efficiency is as good as theirs or better.** The 1.9x wall gap
is 2.25x more work, partly clawed back by using three times the threads.

### What this retires

- The whole thread-count / pool-size / batch-dispatch line of enquiry. Those
  were measured individually and each came back neutral or negative; now
  there is a reason rather than a coincidence.
- The earlier inference that "their parallel CPU is ~484-562 ms, so their
  serial work must be ~340-400 ms." That divided parallel CPU by an assumed
  efficiency. Their parallel CPU is mostly OVERHEAD: 125 ms of work costing
  484-562 ms of CPU across 8 threads. **Never infer a serial cost from a
  parallel one.**

### Re-tested, since a refutation expires when its baseline moves

`target-cpu=native` was pruned earlier on a noisy WALL measurement. Re-tested
on SERIAL CPU, which isolates kernel quality: baseline 265-281 ms, native
250-281 ms — roughly **8 %**. Real, small, and nowhere near 2.25x. The prune
stands, now for a better-founded reason.

### Instrument note

`GetProcessTimes` has a **15.625 ms quantum**: every CPU figure above is a
multiple of it (281.2 = 18 ticks, 250.0 = 16, 265.6 = 17). At ~280 ms that is
+/-5 %, which is fine for a 2.25x gap and useless for an 8 % one. Any future
kernel A/B at this scale needs many more images per sample, or a different
clock.

### Where parity is

Parity needs our serial work at ~125 ms, from 281. That is a **kernel**
project — fused, vectorised convolution of roughly oneDNN quality — not a
scheduling one. It is a real target with a known technique, and it is the
first time in this campaign the target has been the right one.

## D4b — the 2x located: GEMM efficiency collapses at small M

With the target corrected to kernel work, the question is where 281 (now
251) ms goes against their 125. candle's matmul is 18.4 % of serial detect
and it is the framework's code, so it gets measured before anything is
written. Single thread, real shapes, `examples/gemm_efficiency.rs`:

| M x K x N | GFLOP/s | % of ~128 peak | layer |
|---|---:|---:|---|
| 16 x 27 x 102400 | 27.3 | **21 %** | stem 3x3 |
| 32 x 144 x 25600 | 40.5 | 32 % | l1 |
| 64 x 288 x 25600 | 48.1 | 38 % | l2 |
| 128 x 576 x 6400 | 63.2 | 49 % | deep |
| 256 x 1152 x 1600 | **88.9** | **69 %** | deepest |
| 64 x 64 x 25600 | 43.3 | 34 % | 1x1 |
| 256 x 256 x 1600 | 75.3 | 59 % | 1x1 deep |

**Efficiency tracks M, and M is `c_out`.** The deepest layers already reach
69 % of peak — candle's GEMM is not the problem there. The stem runs at
21 %, and the stem is where the spatial work is largest, so the badly-served
shapes carry the most work.

Cross-check against the whole network: YOLO26n is ~8.7 GFLOP at 640x640, so
our 251 ms is **~35 GFLOP/s average** and their 125 ms is **~70**. The 2x is
exactly this table.

### What that implies, priced

* Lifting average GEMM efficiency from ~40 to ~80 GFLOP/s saves ~23 ms of
  the 46 ms gemm bucket -> 251 -> 228 ms (1.10x).
* Removing im2col entirely (13.4 %, ~34 ms of pure data movement a direct
  convolution never does) -> ~194 ms (1.29x combined).
* **Both together reach ~194 ms, not 125.** The remainder is in the
  convolution wrappers — bias add, reshape, the `SliceOp` round trip — which
  the profile currently hides inside the conv parents.

So parity is a convolution-path rewrite: direct/implicit-GEMM convolution
with spatial blocking, which is what removes im2col AND fixes small-M at the
same time, because a direct kernel blocks over output pixels rather than
over a matrix dimension the architecture pins at 16.

**That is the next campaign, and it now has a target, a ceiling, and a
per-shape table to gate against.**

## ★★ D4c — CORRECTION: it is not small M. It is ARITHMETIC INTENSITY, and im2col destroys it

The table above concluded "efficiency tracks M, and M is c_out". That was
committed and it is wrong. M and K co-vary across real layers, so the table
could not separate them. Holding each fixed:

| sweep | GFLOP/s |
|---|---|
| K=27, M = 16 / 64 / 256 | 27.5 / 26.0 / **32.9** — flat, M is not the driver |
| M=16, K = 27 / 288 / 1152 | 31.4 / 26.1 / **19.7** — *worse* with larger K |

Neither dimension explains it. Arithmetic intensity does —
`2MKN / (4(MK + KN + MN))`:

| shape | working set | flops/byte | GFLOP/s |
|---|---:|---:|---:|
| 256x1152x1600 | 10 MB | **92** | **78.0** |
| 256x27x102400 | 116 MB | 6 | 32.9 |
| 16x27x102400 | 18 MB | 5 | 27.5 |
| 16x1152x6400 | 30 MB | 7.9 | 19.7 |

**Every conv GEMM we issue is memory-bound, and im2col is why.** The B
operand IS the ninefold-expanded buffer, so the GEMM reads nine times the
bytes for the same arithmetic. Intensity collapses from ~90 flops/byte (a
balanced GEMM) to 5-8, and a matmul at 5 flops/byte cannot reach peak on any
machine.

This unifies everything the descent found and could not previously connect:

* im2col scaling 1.55x on 24 threads — memory-bound, as measured.
* conv GEMMs at 20-33 GFLOP/s against a 128 peak — memory-bound, not a bad
  microkernel.
* the whole-network ~35 GFLOP/s against the reference's ~70 — they use
  direct/blocked convolution and never build the expanded operand.

**And it explains why the tiled im2col failed for the right reason after
all.** Tiling reduced where the buffer LIVED; it did not reduce how many
bytes the GEMM READS, because the operand is still 9x. Intensity was
unchanged, so the ceiling was unchanged.

The lever is therefore not tiling, not threads, not ISA: it is **not
materialising the expanded operand at all**. A direct convolution reads the
original activation once and holds the output tile in registers, so its
intensity is set by the output tile rather than by a 9x-redundant matrix.

**Recorded as a correction rather than an edit**, because "efficiency tracks
M" is already committed and would otherwise send the next campaign after a
microkernel that is not the problem.

## ★★★ D4b — the GEMM's ORIENTATION, an axis never tested here

I closed this thread with "what's left is candle's GEMM at our shapes, which
is someone else's kernel." That was wrong, and the skill names the axis that
shows why: **orientation**. Its learnings carry a case where identical
arithmetic ran 54 vs 539 GFLOP/s purely on which operand was M.

Our convolution computes `w[c_out, K] x col[K, ohw]`, so **M = c_out is the
SMALL dimension** — 16 at the stem — and N = ohw is large. Measured on the
real yolo26n shapes (`examples/gemm_orient.rs`, best of 7):

| c_out | K | ohw | as-is | transposed | speedup |
|---:|---:|---:|---:|---:|---:|
| 16 | 27 | 102400 | 1.278 ms | 0.632 | **2.02x** |
| 32 | 144 | 25600 | 1.044 | 0.670 | 1.56x |
| 64 | 288 | 25600 | 2.719 | 2.059 | 1.32x |
| 128 | 576 | 6400 | 2.088 | 1.723 | 1.21x |
| 256 | 1152 | 1600 | 1.838 | 1.701 | 1.08x |
| 256 | 2304 | 400 | 1.253 | 1.097 | 1.14x |

**Faster in every shape. 1.30x aggregate, 2.02x at the stem**, where the
tensors are biggest. The GEMM is not someone else's kernel — it is ours to
FEED, and we have been feeding it the slow way round.

**NOT YET EXPLOITABLE, and the tax is the open question.** Taking it needs:

* weights as `[K, c_out]` — free, transposed once at load;
* im2col writing `[ohw, K]` instead of `[K, ohw]` — a different access
  pattern in the kernel, which could go either way and must be measured, not
  assumed (the per-tap chunking already proved a "more parallel" im2col can
  be 2.2x SLOWER);
* the output arriving as `[ohw, c_out]` and needing a transpose back to
  NCHW — **this is the tax, it is unpriced, and it could eat the whole win.**

Prize arithmetic before anyone builds it: GEMM is 17.2 % of serial detect, so
1.30x on it is ~3.9 % of the pipeline — the same scale as the AVX2 silu win,
minus a transpose. Worth doing only if the transpose is cheap, and the honest
next step is to price THAT before writing the im2col variant.

### The tax, priced — and it eats the win six times over

Measured in the same probe rather than left as an open question:

| totals across the six shapes | ms |
|---|---:|
| as-is GEMM | 10.63 |
| transposed GEMM | 7.97 (**1.33x faster**) |
| transpose `[ohw, c_out]` back to NCHW | **+16.55** |
| **net** | **24.52 = 0.43x — 2.3x WORSE** |

The orientation win is real and worth 2.66 ms across these shapes. Undoing
the layout costs **16.55 ms** — six times the saving. `t().contiguous()` on
`[ohw, c_out]` is a strided gather over a large matrix and it dwarfs the GEMM
it was meant to accelerate.

**PRUNED, on arithmetic, before writing the im2col variant.** Fifteen minutes
of probe against what would have been a rewrite of the im2col kernel, the
weight layout and the output path — and it would have landed 2.3x slower.

The prune has a SCOPE, stated so it does not close more than it tested: it
kills *transpose the GEMM and transpose the result back*. It does not kill
*consume `[ohw, c_out]` directly*, which would need every downstream
consumer — bias add, SiLU, concat, the head — to accept channels-last. That
is a whole-graph layout change, not a kernel swap, and it is the only form in
which the measured 1.33x is reachable.

**The lesson is about the prune, not the number.** "Someone else's kernel"
was a category, not a measurement, and it very nearly closed a door with a
measured 2x behind it. A prune constrains the approach it tested; it does not
license a conclusion about the whole component.

### The tax re-priced twice — and candle's transpose is 4.3x slower than a loop

The prune above stands, but two of its numbers were wrong and the second
error was mine.

**First: 16.55 ms to move ~2.5 M elements is ~20 MB of traffic**, which at
this machine's bandwidth should cost about 1 ms. A 16x discrepancy is the
instrument asking for help, not a law of physics. So: candle's
`t().contiguous()` against a plain blocked transpose.

**Second: my blocked transpose had its inner loop over the wrong axis.**
With `c` innermost the destination writes stride by `rows` and touch a fresh
cache line every iteration. Swapping to `r` innermost makes them contiguous —
same tile, same work:

| transpose | total over the six shapes |
|---|---:|
| candle `t().contiguous()` | 18.12 ms |
| blocked, inner loop over `c` (wrong) | 16.67 ms |
| **blocked, inner loop over `r`** | **4.19 ms** |

**Candle's generic transpose is 4.3x slower than a naive blocked loop**, and
my first blocked version was only 1.1x better than candle because it had the
same defect candle does.

Re-priced: `7.97 + 4.19 = 12.17` against `10.63` as-is — **0.874x. Still
dead**, but the margin fell from 2.3x worse to 1.5 ms worse. A hand-written
AVX2 8x8 transpose could plausibly reach ~1.5 ms and flip it to ~1.12x on the
GEMM, which is 17.2 % of serial, so ~2 % of pipeline. **Not worth the
rewrite** — recorded so the next person can see the arithmetic rather than
redo it.

**The finding worth more than the idea it was pricing:** if the graph
transposes anywhere HOT, it is paying 4.3x. `blocks.rs` has two —
`transpose(...).contiguous()` on both sides of the attention matmul — and
attention is the op measuring **1.02x scaling from 1 to 24 threads**, the
only one in the profile that does not parallelise at all. That is the next
place to look, and it is a much better lead than the one that produced it.


---

# THE ACCOUNTING — why parity is not reachable by kernel work

The attention-transpose lead died the way every other one did, and cheaply:
bucketed, measured at **0.017 s of 5.228 s = 0.3 % of the pipeline** — 9 % of
attention itself. Fixing it 4.3x would return 0.23 %. One measurement, one
line of arithmetic, one dead lead.

That was the last unexamined component. The full ledger of this campaign:

| lever | status | worth |
|---|---|---:|
| SiLU `round()` -> magic rounding | **SHIPPED** | 1.079x pipeline (z = +2.84) |
| SiLU double-write removal | **SHIPPED** | 1.12x serial |
| SiLU AVX2 kernel | **SHIPPED** | <=3.9 % (z = +2.84) |
| top-k decoding `max_detections` | **SHIPPED** | ~0.5 % + a parity fix |
| batch parallelism dispatch | **SHIPPED** | CPU 1.08x; wall unproven |
| thread count / pool size | REFUTED | 9/24, z = -1.22 |
| `target-cpu=native` | PRUNED | 1.017x, z = +0.65 |
| per-(channel,tap) im2col | REFUTED | 2.2x WORSE |
| direct convolution (3 variants) | REFUTED | 9/9 against, z = +3.00 |
| tiled im2col | PRUNED on arithmetic | call overhead |
| transposed GEMM | PRUNED | 0.874x after its tax |
| attention transposes | **REFUTED** | 0.3 % of pipeline |
| conv wrapper / SliceOp | REFUTED | 1.8 % / ~0 |
| silu division -> rcp | PRUNED | no measurable cost |

**Sum of every remaining identified lever: ~2-3 %. The gap is ~90 %.**

There is no combination of the things above that reaches 1.0x, and that is
now a measured statement rather than a tired one. Every component of the
forward pass has been bucketed, and every bucket has been either shipped,
refuted with a z-score, or pruned on arithmetic.

## What parity would actually require

Not a kernel. An **algorithm**: an implicit-GEMM or channels-last
convolution that never materialises the im2col buffer. The evidence for that
being the real answer, rather than a guess:

* im2col moves **216.9 MiB per image at the n tier** (deterministic counter,
  not a timing) — 9.5-12.9 ms of pure memory traffic against a ~66 ms
  reference total, and bandwidth is shared so it does not divide by threads;
* candle's GEMM is at **69 % of peak** at balanced shapes, so the multiply
  itself is not the problem;
* the transposed orientation is **1.33x faster** and unreachable only because
  the layout change has to propagate through every downstream consumer.

Those three point the same way: the cost is the DATA MOVEMENT the im2col
formulation forces, and the fix is to stop forcing it. That is a graph-wide
layout change plus a fused convolution kernel — weeks, not an afternoon, and
it would be building what oneDNN already is.

## The honest verdict on the bet

**Not reached.** ~1.9x at the start of this descent, and the shipped wins
(SiLU x3, top-k, batch dispatch) are worth perhaps 10-15 % of it. What the
campaign bought instead is that nobody has to guess again: fourteen levers
are closed with numbers attached, the one remaining path is named and priced,
and the instruments that would have hidden all of it — a null arm, an order
reversal, a cycle counter, a self-checking A/B harness — now exist in the
tree.

## D5d — silu, closed: it is COMPUTE-bound, and the ceiling was wrong

"2.3x remains in silu" came from comparing the kernel to a **memcpy**
roofline. That was the wrong ceiling, and the AVX2 kernel is what made it
obvious.

| | GB/s | Gelem/s | vs memcpy |
|---|---:|---:|---:|
| memcpy (zero arithmetic) | 20.6-22.5 | 2.58-2.82 | — |
| scalar (before) | 9.20 | 1.15 | 2.45-2.81x off |
| **AVX2 (shipped)** | **9.0-12.9** | **1.12-1.61** | **1.75-2.29x off** |

The kernel gained **1.23-1.40x**, taking roughly 40 % of what was measured
as available. Then two things closed the rest of the question:

**The divide is 3.3 % of the AVX2 kernel.** It was pruned once on a SCALAR
ablation, which is weak evidence because other overheads masked it there —
and in an 8-wide kernel `vdivps` has about a tenth of `vmulps`'s throughput,
so the baseline had moved and the refutation deserved re-testing. Re-tested:
still 3.3 %. `vrcpps` + Newton stays pruned, now on the right baseline.

**A transcendental cannot reach memcpy speed, and that is arithmetic not
opinion.** silu does ~16 vector ops per 8 lanes — a degree-5 Horner (10), a
clamp (2), the exponent-field write (3), the divide (1). At 2.4 GHz and ~2
vector ops/cycle that is a **compute** ceiling near 2.4 Gelem/s, against
memcpy's 2.58-2.82 which does no arithmetic at all. Measured 1.12-1.61.

So the honest remaining headroom is **~1.5x to the COMPUTE ceiling**, not
2.3x to a memory one — and closing it means fewer ops, i.e. a shorter
polynomial, i.e. trading accuracy against an oracle that a single FMA fusion
already breached by 2 %.

**silu is done.** Three fixes shipped (magic rounding 1.079x pipeline, the
double-write removal 1.12x serial, AVX2 1.23-1.40x kernel), the divide
refuted twice on two baselines, and the remaining gap is a precision trade
rather than an engineering one.


---

## The roofline, which should have been the first measurement

Every experiment above assumed the same thing without ever testing it: that
the reference does **our arithmetic, faster**. Tiled im2col, three scalar
direct convolutions, the transposed GEMM, the AVX2 microkernel, the blocking
inversion — all of them search the space of "execute these multiplies
better". If that assumption is false, the whole search was in the wrong
space.

Testing it needs no timer. Count the multiply-accumulates with a
deterministic counter, divide by the reference's published latency, and
compare the implied throughput against what the machine can physically
retire (`examples/roofline.rs`):

| | per image |
|---|---:|
| 3x3 convolution arithmetic, yolo26n @ 640 rect | **2.050 GFLOP** |
| ultralytics p50 | 59 ms -> **35 GFLOP/s** |
| ours p50 | 142 ms -> **14 GFLOP/s** |
| this box, AVX2 FMA x 24 cores | **1920-2688 GFLOP/s** |

**The reference runs at about 1.8 % of arithmetic peak. We run at about
0.8 %.** Neither implementation is remotely arithmetic-bound.

### What that costs every conclusion above

The convolution microkernel went from ~1.3x behind to **1.111x** behind over
this campaign — real work, correctly measured. It is also **tuning something
that occupies under two percent of the machine.** Driving the convolution to
100 % of peak, which nobody has ever done in a real framework, would move a
142 ms image by roughly the 20 ms the arithmetic actually costs. The 2.4x
gap at n survives it.

This retroactively explains the shape of every result in this document: a
long series of carefully-measured single-digit outcomes, several refutations,
and no movement in the gate. That is the signature of optimising the wrong
term. The measurement discipline was sound throughout — each individual
number here is defensible — and it was aimed at a term that cannot produce
the answer.

### Where the time actually is

With arithmetic at ~20 ms of a 142 ms image, ~120 ms is elsewhere, and the
other counters in this document already bound the candidates:

* im2col traffic, **216.9 MiB/image**, ~9.5 ms at this box's 24 GB/s;
* `silu`, measured at **30.7 % of serial detect**;
* the remainder is framework: per-op allocation, tensor copies, and the sheer
  op count of a graph executed one node at a time.

None of those are convolution. The reference is at 1.8 % of peak rather than
0.8 % not because its GEMM is twice as good — its GEMM is also nowhere near
the roofline — but because it carries **less of everything else per image**.

### The honest status of the parity bet

Parity was never one kernel away, and no amount of the work catalogued above
was going to reach it. The gap is structural: an eager, allocate-per-op
execution model against one that has had years of memory-traffic and
operator-fusion work poured into it. Closing it is a **fusion and allocation**
project — keeping activations in cache across ops instead of round-tripping
them through DRAM — and that is a different piece of work from anything
attempted here, with a different risk profile and no guarantee at the end.

Recorded at the point where it was found rather than folded quietly into a
plan, because the counter that produced it costs nothing and could have been
run on day one.


---

## Fusion campaign — D6 first, then a refutation that cost the prize

Run under `codec-six-whys-unknowns` with `codec-measurement` loaded, on the
question "fuse the eager graph to reach parity with Ultralytics".

### D6 (run first) — the arms do not do identical work

`crates/ffai-bench/src/runner.rs:539` decodes every clip BEFORE starting the
per-image timer and times `detect()` alone. `corpora/refs/ultralytics_ref.py`
times `model.predict(path)` — a FILE PATH, so its decode happens INSIDE the
timed region.

COUNTED: PNG decode over 45 corpus images, min-of-5 each, same decoder the
harness uses: **p50 3.4 ms**.

So the reference carries ~3.4 ms/image that we do not, and the published
ratio flatters us. Corrected, the n-tier gap is **~1.9x, not 1.81x** — which
is what the README already claimed, arrived at independently.

Two further D6 findings, both about our own tooling:

* **`ffai bench detect --baseline-only` is not honoured** — it ran all eleven
  references over 450 images. A "quick baseline" took 60+ minutes and its
  reference subprocesses were the CPU load that voided a whole A/B round.
* The `ffai` binary **had not compiled since Diana landed** (02849e3 added the
  registration without the dependency). The gate harness is only load-bearing
  if the path to it builds; nothing in the test suite covers that.

### D3 — the fusion prize, counted then refuted

COUNTED (`FFAI_DIANA_COUNT=1`, one run):

| quantity | n tier, per image |
|---|---:|
| conv MACs (3x3 + 1x1) | 1.828 G = **3.655 GFLOP** |
| output activation elements | 13.4 M = **51.2 MiB** |
| im2col buffer | **216.9 MiB** |

Every activation element was written by the GEMM, read and rewritten by
`broadcast_add` for the bias, then read and rewritten again by SiLU. Four
extra touches of 51.2 MiB = 204.8 MiB, **~8.9 ms at this box's 24 GB/s
against a 96 ms image**. That was the prize, and it was wrong.

**Three shapes were built and measured. All three lost.**

| shape | result |
|---|---|
| serial `extend` with bias+SiLU mapped in | 15/21 against, z = +1.96, 8.4% (inconclusive, 31.8% floor) |
| `extend_from_slice` + parallel bias+SiLU | **18/21 against, z = +3.27, 6.0% — direction REAL** |
| single pass into uninit capacity, parallel | 12/21 against, z = +0.65, 11.4% (inconclusive, 14.6% floor) |

Two distinct defects, then the real reason:

1. **A fused op inherits the PARALLELISM of what it replaced, not only its
   arithmetic.** Shape 1 ran on the calling thread; the SiLU it replaced
   fanned out over `par_chunks_mut`. Removing four traversals is worth
   nothing if it also removes three quarters of the workers.
2. **The traffic arithmetic counted the touches REMOVED and not the ones
   ADDED.** Shape 2 replaced four touches (broadcast_add read+write, SiLU
   read+write) with four touches (copy read+write, apply read+write). It saved
   nothing by construction, and measured exactly that.
3. **The prize was priced at the wrong level of the memory hierarchy.** 51.2
   MiB is the per-IMAGE total across ~120 convolutions — **~430 KiB each,
   which is L2-resident.** Those touches were never DRAM traffic. At L2
   bandwidth the real prize is on the order of 1 ms, comfortably under this
   box's floor, which is why even shape 3 — a genuine halving of touches from
   four to two — could not be resolved.

**Reverted: measured worse (shape 2, z = +3.27) and unresolvable at best
(shape 3).** Not "reverted inside the noise" — shape 2 has a direction.

The counter that produced the wrong prize is KEPT (`take_acts`), because it
is also the counter that explains the refutation, and `examples/roofline.rs`
now prints the per-convolution figure next to the DRAM one so the same
mistake cannot be made from the same output.

### What this rules out, and what it does not

It rules out **elementwise epilogue fusion** as a route to parity: bias and
activation are L2-resident and their traffic is not the bottleneck.

It does NOT rule out the **im2col buffer**, which is a different animal:
216.9 MiB per image, written once and read back by the GEMM, at a size that
does NOT stay in L2. That is 4.2x the activation traffic and the one
remaining structural lever. `examples/zerocost.rs` also prices its zero-fill
at **4.3 ms/image** — and note that probe's 1 MiB row flips sign, where the
allocator switches to fresh OS pages, so any fix there must be size-aware.


### Iteration 2 — the im2col zero-fill costs nothing in context

The fusion refutation pointed here: the im2col buffer is 216.9 MiB/image, is
NOT L2-resident, and is allocated with `vec![0f32; k * b]` — zero-filled and
then overwritten by a fill that preserves on the order of one percent of
those zeros as genuine padding.

`examples/zerocost.rs` priced it at the real size distribution (585
allocations, ~380 KiB mean): **4.3 ms per image's worth**, ~4.5 % of a 96 ms
image. It also found the effect INVERTING at 1 MiB, where the allocator stops
recycling and takes fresh, already-zero pages from the OS.

Built: `Vec::with_capacity` plus `spare_capacity_mut`, the fill writing every
element — data where the tap is in bounds, an explicit zero where it is
padding — guarded to sizes under 1 MiB, `set_len` after.

Gated hard, because uninitialised memory that is *incidentally* zero hides
exactly this bug: `padding_is_written_not_inherited` POISONS the allocator
with a recognisable value across the sizes im2col uses, drops it so the
convolution recycles those blocks, and compares against candle at both
strides and two shapes. Plus the five-tier oracle. All passed.

**Measured, single process, ABBA, null arm, CPU time: median 0.9997x, 10/21
rounds, z = -0.22.** Ten of twenty-one is exactly chance. The same harness
resolved a 6 % effect at 18/21 (z = +3.27) earlier the same session, so a
4.5 % effect would not have produced this.

An intermediate measurement said the change was **16 % SLOWER**, 6/6, arms
fully separated. That number was an artifact of its own harness: arms were
selected by `git stash` and always run in the order new-then-old, so the old
arm collected every warming benefit. It is recorded because it was wrong in
the direction that would have made the revert look justified for the wrong
reason.

**Reverted: no measured effect.** Not "measured worse".

The lesson is `codec-measurement` §9 and the D6 rule, in the form that costs
the most: **the microbench and the in-context number disagreed, and the
in-context number wins.** `zerocost.rs` allocates and frees in a tight loop,
which keeps recycling the same heap block and so pays a real `memset` every
time. The pipeline's allocation pattern is not that, and `alloc_zeroed` there
is largely free — the OS hands back pages that are already zero.

`examples/zerocost.rs` is kept, with this outcome noted in it, because the
probe is correct about what it measures and only wrong about what that
implies for the pipeline. That distinction is the whole finding.


### Iteration 3 — the ceiling probe that closed the whole campaign

Implicit GEMM was the last traffic lever standing: generate im2col columns
into an L1-resident panel inside the GEMM's blocking, so the 108.4 MiB buffer
is never materialised and never read back. It is a large build, and the two
preceding refutations were both the SAME error — an L2-resident buffer priced
at DRAM bandwidth — so the ceiling was measured first.

`examples/im2col_sizes.rs` bins im2col BYTES by the size of the buffer
carrying them, because the mean (380 KiB/call) is the wrong statistic when
early layers are wide and shallow and late layers narrow and deep:

| buffer size | bytes | share |
|---|---:|---:|
| under L2 (2 MiB) | 22.3 MiB | 20.6 % |
| L2 .. L3 (32 MiB) | 86.1 MiB | 79.4 % |
| **over L3** | **0.0 MiB** | **0.0 %** |

**Not one byte of the im2col buffer reaches DRAM.** The largest buffer in the
graph is under 16 MiB against a 32 MiB L3. At L3 bandwidth the write-plus-
read-back is on the order of **1 ms**, not the 9 ms the DRAM figure implied.

Pruned on arithmetic, before building. Cost: one probe.

## The campaign's conclusion — Diana is not memory-bound anywhere

Three levers, three prizes, all computed at DRAM bandwidth and all wrong for
the same reason:

| lever | DRAM-priced prize | actual residency | outcome |
|---|---:|---|---|
| bias + SiLU epilogue fusion | 8.9 ms | ~430 KiB/conv, **L2** | built, 18/21 against, z = +3.27 |
| im2col zero-fill removal | 4.3 ms | fresh OS pages, free | built, 10/21, z = -0.22, **no effect** |
| implicit GEMM | ~9 ms | 0 % over L3 | **pruned before building** |

The unifying finding is worth more than any of the three:

> **Nothing in this graph is memory-bandwidth-bound.** Every working set —
> activations at ~430 KiB per convolution, im2col at up to 16 MiB against a
> 32 MiB L3 — is cache-resident. A YOLO26n at 640 is simply not a large
> enough model to leave cache on this machine.

That closes "fuse the eager graph" as a route to parity, with arithmetic
rather than with a series of disappointments. Every traffic-reduction lever
is bounded by cache bandwidth, which is 3-15x DRAM, so every prize computed
against 24 GB/s in this document was overstated by that factor.

**What it redirects to.** If the time is not bandwidth, it is the work and how
well it is spread. Two numbers already in this file point there and neither
has been chased to the bottom:

* the pipeline gets **1.53x from 24 cores** and degrades past 4 — measured,
  and the reason ~65 % of it is serial is still unexplained;
* `attn` scales at **0.93x** — it gets slower with more cores, and unlike
  `silu` (fixed) nothing has been done about it.

Both are parallel-efficiency questions, not memory questions, and this
campaign has established that memory questions cannot produce the answer.


### Iteration 4 — utilisation, measured instead of fitted

`examples/utilisation.rs`: `QueryProcessCycleTime` over wall, divided by the
calibrated single-core rate, gives MEAN CORES BUSY directly — no Amdahl fit.

Readings ran 0.65 to 2.01 against a 4-worker pool, contaminated by another
project's `bench_ocr` occupying ~12 cores of this shared worktree. The ratio
is CPU over WALL, so contention can only DEPRESS it; the maximum is the least
contaminated reading and a valid bound:

> **At best 2.01 of 4 workers busy. Half the pool is idle.**

~2x of headroom against a 1.9x gap — the first lever in this campaign large
enough to REACH parity rather than shave it. A bound, not a result.

### Iteration 5 — candle runs its own thread pool, and shrinking it does nothing

Chasing the idle half found a structural fact worth recording regardless of
its performance verdict.

**candle 0.11 builds its OWN rayon pool** (`utils.rs:368`, `candle_pool()`),
sized by `get_num_threads()` — which is rayon's default, 24 here. Our
`latency_pool().install(...)` therefore never governed matmuls at all. The
process runs **4 of our workers plus 24 of candle's on 24 logical cores**, and
the 1.21x pool win measured earlier moved only our half.

That is why `gemm` (13.4 %) and `1x1` (19.0 %) — about a third of the pipeline
— scale independently of anything `crate::parallel` decides.

`RAYON_NUM_THREADS` does reach candle's pool. Tested on CPU TIME, which
contention cannot inflate, 8 pairs ABBA:

| | median CPU/image |
|---|---:|
| default (candle at 24) | **197 ms** |
| `RAYON_NUM_THREADS=4` | **199 ms** |

No effect. Oversubscription between the two pools costs no measurable WORK,
so matching their widths is not the lever — the structure is real and the
prize is not there.

What that leaves: the idle half is not explained by pool oversubscription,
and the wall-time measurement needed to localise it further requires a quiet
box. Recorded as OPEN, with the instrument (`examples/utilisation.rs`) and the
statistic to read (the maximum) both in place for the next quiet window.


---

## D1, finally answered: we do LESS work and spread it over a quarter of the cores

Every iteration of this campaign assumed, without testing, that the reference
executes our arithmetic more efficiently. The measurement that settles it was
never taken because the harness only ever compared WALL: the reference's CPU
time per image.

Taken by polling the python process tree (`TotalProcessorTime` on the venv
launcher reads ~0 — the shim re-execs and the work is in a child, an
impossible number that had to be chased before any figure was usable), 60
images against 1 so the model load cancels, exit codes and output line counts
checked on both arms:

| | CPU/image | wall/image | **cores busy** |
|---|---:|---:|---:|
| ultralytics-yolo26n-rect | **312.5 ms** | 39.0 ms | **8.01** |
| Diana | **~225 ms** | ~96 ms | **~2.2** |

**We do 28 % LESS CPU WORK than the reference.** The 1.9x latency gap is not
an efficiency gap, an arithmetic gap, a memory gap or a kernel gap. It is
entirely a SPREADING gap: torch puts 312 ms of work on 8 cores; we put 225 ms
of work on 2.2.

The wall figure cross-checks: 53.0 ms/image marginal on an earlier run of the
same probe, against the ledger's 53 ms p50 for this reference. The 39.0 ms
here is the same quantity on a less contended pass.

### What this retires

* **Kernel tuning.** The AVX2 microkernel, the blocking experiments, the
  direct-convolution variants — all were trying to reduce work we already do
  less of than the winner.
* **Every memory lever**, already closed by the cache-residency finding.
* My own reading three commits ago that "the useful work is small and
  spreading it wider costs more in barriers than it recovers." Torch spreads
  the same kind of work over 8 cores at **8.01/8 = essentially perfect**
  utilisation. The barriers are OUR design, not a property of the problem.

### What it makes the target

Our parallel efficiency is **2.2 / 4 = 55 %** against torch's ~100 %.

| if we reached | wall/image |
|---|---:|
| 4 cores busy (100 % on the current pool) | **56 ms** |
| 8 cores busy (torch's width) | **28 ms** |

Parity is 53 ms. **Perfect efficiency on the pool we already have clears it**,
and matching torch's thread count would beat the reference by 1.4x.

That is the first target this campaign has produced that is both quantified
and sufficient. It is also the first evidence that Diana's implementation is
COMPETITIVE — we lose a race we are equipped to win, on scheduling alone.

### The open question, stated precisely

Where do 45 % of our worker-cycles go? Candidates already measured, none yet
attributed:

* ~1320 matmuls per image, each a cross-pool handoff into candle's own
  24-thread pool (candle-core 0.11 `utils.rs:368`);
* a per-layer fan-out with a barrier per convolution and per activation;
* `attn` at 0.93x scaling — slower with more cores.

The instrument for it (`examples/utilisation.rs`) exists and reads cores-busy
directly. It needs a quiet box, which this one has not been.
