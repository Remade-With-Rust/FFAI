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

- **"A concurrent batch path PyTorch's GIL cannot match" (22.73 vs 17.29
  img/s): WITHDRAWN from the README, not refuted.** It was a wall-clock
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
