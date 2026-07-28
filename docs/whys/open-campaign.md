# WHYS — closing out OPEN.md

**Unknown:** the speed gate still fails at ~1.2–1.25× behind
`whisper-cpp-tiny-greedy-t24`, and [`OPEN.md`](OPEN.md) lists four levers that
were refuted on one measurement each, one unexplained result, and three
instrumentation gaps.

Method: `codec-six-whys-unknowns`, depth 6 first. One entry per level; a level
may spawn siblings; refuted hypotheses keep their number.

---

## D6a — is the instrument capable of the measurement the plan requires?

- **ASKED:** `OPEN.md` §1 says each of the four refutations needs three varied
  probes, **at least one at the level above the change**. Every one of them is
  an op- or stage-level change, so the level above is total transcription time.
  Can this repo measure that?
- **MEASURED:** every A/B knob (`FFAI_MLP_INT8`, `FFAI_PAR_HEADS`,
  `FFAI_KV_F16`, `FFAI_DECODE_MIN_KEYS`, `FFAI_GEMV_PAD`, `FFAI_VOCAB_BLK`) was
  a `OnceLock` resolved from the environment on first read. A value fixed at
  process start can only be A/B'd **arm-by-arm across two processes**.
  `examples/noise_floor.rs` on this box: op-level spread **28–49 %**; the
  reference's own throughput spreads 37 % of its median across runs.
- **ANSWER:** **no.** The effects still in play are 2–15 %. Arm-by-arm across
  processes at a 28–49 % spread does not measure code, it measures which
  machine each arm drew. This is the exact failure `codec-six-whys-unknowns`
  prescribes interleaving for, and interleaving was structurally impossible.
  **Every "inconclusive" verdict in `OPEN.md` §1 was taken with an instrument
  that could not have concluded.**
- **CONFIDENCE:** high — read from the source, not inferred.
- **FIX:** [`asr/knobs.rs`](../../crates/ffai-mercury/src/asr/knobs.rs) — the
  same names, words and defaults, backed by atomics instead of `OnceLock`, so a
  harness can flip them between rounds and hold both arms alive at once. Unset
  environment behaves identically; 65 tests pass (62 before, +3 for the knobs).
  The per-call hot-path caveat the codebase earned the hard way is preserved: a
  relaxed atomic load, never `std::env::var`, inside the per-token path.
- **STATUS:** closed. **Fourth consecutive descent in this campaign to
  terminate at D6.**

## D6b — what is the harness's own noise floor?

- **ASKED:** before trusting any verdict, what does the harness report for two
  arms that are *definitionally identical*? A VP9 campaign found its A/B read
  +0.2 % to +10.8 % for two identical encoders — wider than every delta it had
  been publishing.
- **MEASURED:** [`examples/pipeline_ab.rs`](../../crates/ffai-mercury/examples/pipeline_ab.rs)
  `--test null`, two separately-constructed engines with identical settings, 41
  paired ABBA rounds, 4 clips / 27.8 s audio:

  ```
  null-A   med 1154.2 ms  [987.2 .. 1261.5]
  null-B   med 1149.0 ms  [1012.1 .. 1246.4]
  paired: A won 19/41 (46 %) · median ratio 0.996x · ranges OVERLAP
  transcripts: IDENTICAL
  VERDICT: harness floor is +/-0.4 % (|z|=0.5)
  ```

- **ANSWER:** the harness is **sound**. Under a true null it returns
  INCONCLUSIVE (z = −0.5) with a 0.4 % median bias, while the raw per-round
  medians swing 23 % — which is the interleaving doing its job. Effects of
  roughly **≥2 %** are resolvable at 41 rounds; anything under ~1 % is not.
- **Also established:** the pipeline is **deterministic** across 41 repeats on
  these clips (identical transcripts). §6.22's run-to-run WER alternation is
  therefore not per-call nondeterminism on this path.
- **CONFIDENCE:** high.
- **STATUS:** closed. Re-run `--test null` at the start of every session; it
  drifts with the machine.

## D1 — which levers are large enough to be worth a probe at all?

- **ASKED:** `codec-six-whys-unknowns` is blunt about this — *expected pipeline
  gain = stage share × speedup*, and an experiment whose ceiling lands under
  the noise floor cannot succeed however good the kernel is. Two minutes of
  multiplication before any building.
- **MEASURED:** `FFAI_PROFILE=1`, tiny.en, 13.5 s clip, 41 tokens, total
  0.271 s. Stage shares of **total**, not of their parent:

  | stage | share of total |
  |---|---:|
  | encoder attention | **29.3 %** |
  | encoder MLP | 18.9 % |
  | decoder cross-attention | **19.3 %** |
  | decoder MLP | 9.4 % |
  | decoder self-attention | 6.8 % |
  | final projection | 5.8 % |
  | conv front end | 4.6 % |
  | sampling | 3.6 % |
  | mel | 2.3 % |

- **ANSWER — the prize table.** Pipeline gain = share × (1 − 1/speedup):

  | lever (OPEN.md) | target | share | speedup for a 2 % prize | plausible | prize | verdict |
  |---|---|---:|---:|---|---:|---|
  | int8 decoder MLP | dec MLP | 9.4 % | 1.27× | 1.5–2× (halves traffic vs f16) | **3.1–4.7 %** | probe |
  | f16 KV cache | dec cross-attn | 19.3 % | 1.12× | 1.16× (measured on the stage) | **2.7 %** | probe |
  | parallel decode heads | dec cross-attn | 19.3 % | 1.12× | 1.14× (measured on the stage) | **2.4 %** | probe |
  | NT-form kernel | encoder `ea kernel` (13.3 %) | 13.3 % | 1.18× | measured **slower** in context | ≤ 0 | see §1b |

  All three surviving levers clear the 2 % floor established in D6b — which is
  precisely why they read "inconclusive" before: the old cross-process
  instrument's floor was far above their ceilings. **The levers were never
  refuted; the instrument was never able to decide them.**
- **CONFIDENCE:** high on the shares (single-process profiler, one clip);
  medium on the plausible speedups (two are prior stage measurements, one is a
  traffic argument).
- **SPAWNED:** D1a–D1c, one per surviving lever, each answered at the pipeline
  level by `pipeline_ab`.
- **STATUS:** open.

## D2 — the quantization asymmetry (OPEN.md §3) is not a phenomenon

- **ASKED:** `OPEN.md` §3 records that the *harder* corpus is *less* sensitive
  to quantization, calls the direction "predicted backwards both times", and
  asks for a mechanism. What does the campaign's own significance bar say about
  the effect before anyone explains it?
- **MEASURED:** the reported argmax-flip counts, run through the two-proportion
  test this campaign uses for every other verdict:

  | corpus | flips | rate |
  |---|---:|---:|
  | test-clean | 15 / 848 | 1.77 % |
  | test-other | 6 / 560 | 1.07 % |

  pooled p = 21/1408 = 0.01491; SE = √(p(1−p)(1/848 + 1/560)) = 0.00660;
  **z = (0.01769 − 0.01071) / 0.00660 = +1.06**, two-sided p ≈ 0.29.

- **ANSWER:** **there is no effect to explain.** z = 1.06 is far below the
  |z| > 2 bar this campaign applies to everything else — §6.25 refused to
  accept z = +2.3 without 25 more rounds. Reaching |z| = 2 at these
  proportions needs ≈ 3.6× the tokens (~3000 clean / ~2000 other).
- **The honest entry is not "predicted backwards, no mechanism found" but
  "the effect is within noise at these sample sizes; the question is not yet
  well-posed."** Inventing a mechanism for it would have been fitting a story
  to 21 events.
- **What survives:** the *conclusion* §6.23 drew from the second corpus is
  unaffected — the int8 projection is still not the source of the test-clean
  deficit, which rests on the gap moving +0.35 → +0.01 pp, not on the flip
  rates. The separate WER-cost figures (+0.62 pp vs +0.05 pp) are a different
  measurement and remain open.
- **CONFIDENCE:** high — arithmetic on the campaign's own reported counts.
- **STATUS:** closed. Same class as the two wrong ceilings and the retracted
  WER win: **a single favourable sample promoted to a finding.** Third instance
  in this campaign, first one caught before it drove any work.

---

## Open: quality debts carried out of the demo campaign (2026-07-28)

Two separate quality gaps, different causes, tracked apart so neither hides
the other. Both surfaced from the side-by-side demo, not from the corpora —
which is itself the finding.

### Q1 — the annotation cost (0.22 pp, priced and chosen)

- **WHAT:** `suppress_non_speech: false` ships as the default, matching
  whisper.cpp. Cost on LibriSpeech test-clean: WER 7.77 → 7.99, CER
  3.25 → 3.27. Cost on test-other: **zero** (16.79 / 8.34 both ways).
- **WHY IT COSTS:** on clean read speech the model occasionally spends an
  annotation token where words belong. On noisy speech the annotation never
  wins argmax, which is why the corpora disagree.
- **WHY IT MATTERS MORE THAN 0.22 PP LOOKS:** it moves test-clean from 2.5 %
  to 5.4 % relative against whisper.cpp — from inside the 5 % band to
  outside it. The gate criterion flips on a change that reads small.
- **THE OPEN QUESTION:** all-or-nothing is not the only rule available.
  Allowing annotation tokens *only when `no_speech_prob` is elevated* would
  plausibly keep `(coughs)` on real non-speech events while denying the
  model an annotation mid-sentence on clean speech. That is a testable
  hypothesis with an existing signal — `no_speech_prob` is already computed
  at position 0 for the §6.31 gate and currently thrown away after one
  comparison.
- **FIRST EXPERIMENT:** condition the suppression list on the same
  probability the no-speech gate reads. Null arm: current behaviour. Gate:
  test-clean WER returns toward 7.77 while the demo's cough still annotates.
- **STATUS:** open. Cheap, well-posed, and the signal is already in hand.

### Q2 — the test-clean CER deficit (~13 %, unexplained since §6.7)

- **WHAT:** CER 3.27 % vs whisper.cpp 2.87 %. Suppressing annotations moves
  it only 3.27 → 3.25, so Q1 explains ~1 pp of a 14 pp relative gap and
  **~13 % remains unexplained.**
- **WHY IT IS THE MORE INTERESTING ONE:** it is character-level, it does not
  appear on test-other, and it has survived int8 (§6.23 cleared the
  projection as the source), the f16 cache, and every kernel change since.
  A deficit that is corpus-specific and metric-specific is a clue, not noise.
- **NOT YET ASKED:** what the *character* errors actually are. No one has
  looked at the confusion classes — substitutions vs insertions, digits,
  casing, punctuation, hyphenation. The campaign has measured this gap
  repeatedly and never once inspected it.
- **FIRST EXPERIMENT:** dump per-clip CER deltas, take the worst 20, and
  read them. Before any hypothesis.
- **STATUS:** open, and explicitly **not** to be folded into the Mercury-X
  milestones ([mercury-X-mission.md](../mercury-X-mission.md)) — that layer
  does not address it and would obscure it.
