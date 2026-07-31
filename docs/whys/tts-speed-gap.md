# WHYS — where Mercury TTS is slower than piper1

Campaign: after M-T0..M-T3 and four speed sub-campaigns, our VITS synthesis was
believed to be ~1.31x behind piper1 overall, losing ~2.5x on flow and ~2.2x on
the decoder's plain convs, at parity on the text encoder, and winning on the
upsamplers and duration predictor.

**Every one of those per-stage beliefs turned out to rest on a broken
comparison.** The descent below ran depth 6 first, as the skill demands, and
depth 6 invalidated the campaign's map before a single line of code was touched.

---

## D6 — is the measurement sound? (run FIRST)

### D6a — does the hardware hold still?

- ASKED: this box has been blamed for noise all campaign; three ledger lines
  were disowned as machine-compromised. Is that an environment problem or a
  harness bug?
- MEASURED: `Win32_Processor` + `GetLogicalProcessorInformationEx` report an
  i7-14650HX: **16 physical cores, 24 logical, in two classes** — 8 cores at
  `efficiency_class=1` with SMT (P-cores, logical 0..15) and 8 at
  `efficiency_class=0` without (E-cores, logical 16..23).
  `examples/pin_probe.rs` runs the identical synthesis under a fixed affinity
  mask and reports CPU time alongside wall:

  | pin | wall ms | cpu ms | xRT | cpu/wall |
  |---|---|---|---|---|
  | 1 P-core | 7694 | 6953 | 5.85 | 0.90 |
  | 1 E-core | 11977 | 11906 | 3.76 | 0.99 |
  | 8 P-cores | 2669 | 11469 | 16.85 | 4.30 |
  | 8P + 8E | 2465 | 15641 | 18.25 | 6.35 |
  | unpinned | 2153 | 17625 | 20.89 | 8.19 |

- ANSWER: an E-core runs the same code **1.56x slower** than a P-core, and the
  most core-class-sensitive stage is **flow at 1.71x** — the exact stage the
  campaign spent the most effort on. The same binary on the same data spans
  5.85x to 20.89x realtime purely on placement.
- CONFIDENCE: high — monotone, large, reproduced across five configs.
- SPAWNED: D6b (thread counts), D6c (is the reference's instrument sound?)
- STATUS: **closed — instrument defect confirmed and priced.**

### D6a' — refuted: E-core straggling

- ASKED: rayon's default pool spans all 24 logical processors. On a hybrid CPU
  the classic pathology is that a parallel region splits evenly, E-cores retire
  late, and the join waits for the slowest worker — so excluding E-cores can be
  FASTER.
- MEASURED: 8 P-cores 16.85xRT vs 8P+8E 18.25xRT vs unpinned 20.89xRT.
- ANSWER: **REFUTED.** Using E-cores helps, and letting the scheduler use SMT
  siblings too helps further. Pinning is not a speed lever for us; it is only an
  instrument control. Recorded so the idea does not return.
- CONFIDENCE: medium-high — one axis (three configs, monotone). Not re-probed on
  a second axis, so this is "measured worse", not "impossible".
- STATUS: closed, negative.

### D6b — do both arms do identical work?

- ASKED: the campaign compared "ms per 20 sentences" between our engine and
  onnxruntime. Are those 20 sentences, and that machine allocation, the same?
- MEASURED: two independent mismatches.
  1. **Different text.** `profile_piper_stages.py` takes the first 20 sentences
     in CORPUS order; `profile_tts.rs` takes the first 20 HOLDOUT sentences.
     Different sentences synthesize different amounts of audio, so the raw ms
     were never commensurable. Fixed by dumping our exact phoneme-id rows
     (`FFAI_DUMP_IDS`) and feeding them to ORT
     (`corpora/refs/profile_piper_matched.py`); audio now matches to 0.4%
     (45.0s vs 45.2s).
  2. **Different thread counts.** ORT's `intra_op_num_threads` was left at its
     default. Measured via `GetProcessTimes` on both sides:

     | arm | cpu/wall | wall ms | cpu ms | xRT |
     |---|---|---|---|---|
     | ORT, 1 thread | 0.98 | 4224 | 4156 | 10.45 |
     | ORT, 4 threads | 3.97 | 2072 | 8219 | 22.00 |
     | ORT, 8 threads | 7.92 | 1811 | 14344 | 24.58 |
     | ORT, default | 15.62 | 1682 | 26266 | 26.86 |
     | ours, unpinned | 8.19 | 2153 | 17625 | 20.89 |

- ANSWER: **ORT runs ~15.6 effective threads to our 8.2.** At its default it
  beats us on wall by 1.28x while burning **49% more CPU** (26.3s vs 17.6s) —
  per unit of work we are already the cheaper engine. At MATCHED thread
  occupancy (8) the gap is **1.19x**, not the 2.5x on flow / 2.2x on decoder the
  campaign was chasing.
- CONFIDENCE: high — audio length matched, both sides measured with the same
  Win32 instrument, monotone thread sweep.
- STATUS: **closed. The headline is restated: 1.19x at matched threads, 1.28x at
  each engine's default.**

### D6c — is the reference's own instrument sound?

- ASKED: every ORT per-stage figure in the ledger came from
  `sess.enable_profiling`. Is a profiled ORT run comparable to our un-profiled
  stage timings?
- MEASURED: same config (matched ids, 8 threads) reads **3170-3500 ms/pass
  profiled against 1811 ms unprofiled — a 1.75-1.93x profiler tax.** The
  principled per-node correction (tax spread over the 50,560 node events per
  pass, 26.9 us/node) drives the duration predictor to **-111.7 ms**.
- ANSWER: an arithmetically impossible stage time is the instrument asking for
  help, not a rounding artifact. The tax is not uniform per node and cannot be
  corrected this way. **The ORT per-stage split is discarded as unusable.** Only
  its op-SHARES within a profiled run survive, and only qualitatively: `dec` is
  ~90% ConvTranspose+Conv, `enc_p` only ~25% conv, `dp` is 25,740 tiny nodes per
  pass with no conv at all.
- CONFIDENCE: high — the cross-check is a direct wall-clock comparison of the
  same config with the flag on and off.
- STATUS: **closed. Every per-stage "we lose Nx to ORT" claim in this campaign
  is withdrawn as instrument-contaminated.**

---

## D1 — is the gap real, at matched settings?

- ASKED: with matched text, matched audio length, matched threads, and both
  arms un-profiled — are we slower?
- MEASURED: ours 2153 ms wall vs ORT-at-8-threads 1811 ms; ORT default 1682 ms.
- ANSWER: **yes, real, but 1.19x — not 1.31x and not 2.5x.**
- CONFIDENCE: high.
- STATUS: closed.

## D2 — which stage owns it, in absolute cost?

- ASKED: our own stage split, with the residue named.
- MEASURED (`pin_probe`, unpinned, matched ids, best-of-5), now carrying CPU
  time so each stage's effective core occupancy is visible:

  | stage | wall ms | cpu ms | **cores** |
  |---|---|---|---|
  | text_encoder | 523.1 | 2843.8 | 5.44 |
  | duration_pred | 356.4 | 312.5 | **0.88** |
  | flow | 281.0 | 3296.9 | 11.73 |
  | decoder | 1025.6 | 11156.2 | 10.88 |
  | TOTAL | 2210.8 | 18468.8 | 8.35 |

  Stage sum 2186 ms against 2211 ms wall — **residue 1.1%, named.**
- ANSWER: the decoder is the largest absolute cost, but the **anomaly** is the
  duration predictor: 16% of wall executed on **0.88 cores** while flow and
  decoder reach 11+. That single stage is why our whole-process occupancy is
  8.19 against ORT's 15.62.
- CONFIDENCE: high — residue is 1.1% and the occupancy column is a direct
  measurement, not an inference.
- SPAWNED: D3a (what is serial inside dp?), D3b (why is text_encoder only 5.44?)
- STATUS: closed.

## D3a — which op inside the duration predictor is serial?

- ASKED: is dp's cost arithmetic or plumbing, and where?
- MEASURED (`examples/dp_anatomy.rs`, 20 sentences, 1764 phoneme columns):
  - `durations()` whole: 248-313 ms across runs
  - conditioning half (pre -> DDSConv -> proj): 87-115 ms at **5.7-7.6 GFLOP/s**
  - flows + tail: 161-198 ms = **63-65% of the stage**
- ANSWER: the stage runs at single-digit GFLOP/s on a machine that does
  hundreds — overhead-bound by definition. Reading the code closed the shape
  question: `conv_flow_reverse` runs a **full 192-channel DDSConv stack per
  flow**, so the duration predictor executes **four** 192-channel DDS stacks
  (1 conditioning + 3 flows), not one. The flow half is not cheap 2-channel
  plumbing, as its tensor shapes suggest at a glance.
- CONFIDENCE: high.
- STATUS: closed.

## D4a/D5a — why is that primitive slow as executed here?

- ASKED: what routes dp's work onto a serial path?
- MEASURED/READ: `Vits::conv` routes on name. `dec.*` and `dp.*` go to the
  direct kernels; `enc_p.*` and `flow.*` were explicitly REVERTED to candle,
  with the reason recorded in the comment above the rule:

  > *"enc_p regressed 1.35x (serial 1x1s lose to candle's threaded matmul at
  > 192x192xT)"*

  dp's dense convs are exactly 192x192xT. The finding was measured for the text
  encoder and never applied to the duration predictor, so every dense dp conv
  lands on `conv1d_direct`, which is serial.
- ANSWER: **the mechanism was already written down in our own source comment.**
  dp sits at 0.88 cores because the routing rule sends it the one shape the
  campaign had already proven loses on that path.
- CONFIDENCE: high — a code-reading fact that predicts exactly the occupancy
  anomaly D2 measured independently.
- FIX: route dp's DENSE convs to candle's threaded matmul; keep dp's DEPTHWISE
  convs on the flat kernel (measured a win earlier, 395 -> 254 ms).
  `FFAI_DP_DIRECT_1X1=1` restores the old routing for A/B.
- STATUS: fix implemented; see the rebuild below.

---

## Rebuild — climbing back through the gates

Gate at every level climbed, per the skill.

- **D5 microbench / stage (`dp_anatomy`)**: 312.84 -> 210.75 ms, **1.48x**;
  conditioning throughput 5.7 -> 10.4 GFLOP/s. Re-run in reversed (BA) order to
  check position bias.
- **D2 whole pipeline (`pin_probe`)**: pending — the gate that matters, since a
  change that helps an op can tax its consumer.
- **D1 corpus (`ffai bench tts`)**: pending — WER must stay byte-stable; the
  routing change is arithmetic-reordering only (same math, different kernel), so
  it is gated on tolerance, not bit-identity.

---

## Standing corrections to the ledger

1. The headline gap is **1.19x at matched thread occupancy** (1.28x at each
   engine's default), not 1.31x and not 2.5x on any stage.
2. **We use 33% less CPU than ORT** at each engine's default settings. Wall-clock
   parity is a threading problem on our side, not a kernel-efficiency deficit.
3. All per-stage "ours vs ORT" ms comparisons recorded before this descent are
   **withdrawn**: they compared different sentences, at different thread counts,
   with the reference's profiler inflating its side 1.75-1.93x.
4. Pinning to P-cores is NOT a speed lever (refuted); it is an instrument
   control. Benchmarks should keep reporting cpu/wall so occupancy regressions
   are visible.
