# Entropy inflection audit — where does FFai break as content gets harder?

**Question:** which parts of the pipeline work on simple content and start
breaking on high-entropy content? A stage whose cost rises FASTER than the
entropy that drives it is a dispatch gate nobody has built yet.

**Status:** plan + corpus + execution, this document carries all three.

---

## 0. Scope — where a content axis can even exist

FFai has two candidate surfaces and only one of them can vary with content:

| surface | content-dependent? | evidence |
|---|---|---|
| **Diana detection** | **NO — closed** | static shapes, dense arithmetic, no data-dependent branching, **0 subnormals in 80.6M values**, latency correlating **-0.009** with detection count |
| **rff decode + ingest** | **YES by construction** | entropy coding, motion compensation and residual reconstruction all scale with how much information the frame carries |
| **the LIVE change gate** | **YES** | it is a content decision by definition; measured at 0.8 % fire rate on MOT17 and 95.8 % on a still scene |

So the detector is the **control arm** — it should stay flat across the whole
ladder, and if it does not, the -0.009 result was wrong. Decode and the gate are
where a dispatch could hide.

---

## 1. The corpus — real content, because synthetic misprices this exact thing

`codec-measurement` §9 and §18 are unusually blunt here: a synthetic clip's
stage shares can be off by **2-3x**, and a per-item cost calibrated on synthetic
content was wrong by **46x** when it met real footage. Entropy-scaling is
precisely the quantity that goes wrong, because synthetic content is thin on
exactly the axis being tested (coefficient density).

So the ladder is the **Xiph derf collection** — the same clips the
content-adaptive-dispatch skill's own case studies use, which makes its
thresholds comparable:

| clip | character | expected entropy |
|---|---|---|
| `akiyo` | talking head, static background | lowest |
| `container` | slow pan, mostly still water | low |
| `foreman` | handheld, moderate motion + detail | mid |
| `bus` | fast translation, moderate detail | high |
| `stefan` | fast irregular motion, tennis | high |
| `mobile` | dense texture, complex multi-object motion | highest |

Plus **MOT17-09** already in the tree as a real-surveillance control point.

**Entropy is MEASURED, not assumed.** Each clip is encoded once with fixed
settings and its **bits per frame** becomes the x-axis — a deterministic number
that needs no quiet box, and the honest proxy for "how much information is
here". Ordering the clips by intuition and then reporting a trend against that
ordering would be circular.

---

## 2. What is measured, per clip

| quantity | why |
|---|---|
| **bits/frame** | the entropy axis, deterministic |
| **decode ms/frame** (rff) | the suspect |
| **decode cpu/frame + cores** | §2a/§2b — a stage that serialises on hard content looks fine on wall alone at low load |
| **detect ms/frame** (Diana) | the CONTROL; must stay flat |
| **LIVE gate fire rate** | the gate whose failure mode is known to be a cliff |
| **frames decoded** | §4 work parity — a clip that decodes fewer frames voids its row |

## 3. What an inflection point looks like

The dispatch signal is **not** "hard content is slower" — that is expected and
uninteresting. It is **cost rising faster than entropy**:

* Fit `cost = a + b * bits_per_frame` across the ladder.
* **Linear** (cost/bit roughly constant) → the stage scales with the work it is
  given. No gate to build.
* **Super-linear** (cost/bit rising with entropy) → something is degrading:
  a cache falling out, a fast path declining, a fallback engaging. **That is the
  dispatch gate**, and the clip where cost/bit turns is where its threshold sits.
* **A CLIFF** — flat then a step — is the strongest signal, and means a discrete
  path change, not a gradual one.

Report **cost per bit**, not cost. It is the derivative that carries the answer,
and plotting raw cost against entropy will show a rising line for a perfectly
healthy stage.

## 4. Stop rules

1. **The control must stay flat.** If Diana's detect time tracks entropy, stop
   and re-open the -0.009 finding before interpreting anything else.
2. **Work parity per clip.** Different frame counts void the comparison (§4).
3. **A trend needs >= 4 points on the ladder**, not two ends. Two points always
   fit a line.
4. **No building in this pass.** Output is a table and a verdict per stage.

---

## 5. Results

*(filled in by execution below)*

# EXECUTED — 2026-08-06

Corpus from the Xiph derf collection, encoded once with fixed x264 settings.
Entropy is **measured** as bits/frame, not assumed from clip names.

| clip | bits/frame | frames |
|---|---:|---:|
| akiyo | 7,407 | 164 |
| container | 15,259 | 164 |
| foreman | 17,605 | 164 |
| bus | 35,848 | 150 |
| stefan | 36,525 | 90 |
| mobile | 56,734 | 164 |

**7.7x entropy span.** `corpora/clips/entropy-ladder/` (gitignored, generated),
`crates/ffai-media/examples/entropy_decode.rs`.

---

## FINDING 1 — CABAC decode is broken, and it fails SILENTLY

A correctness defect, not a performance one, and the literal answer to "works on
simple content, breaks on harder content".

Feature isolation on akiyo, one variable at a time:

| variant | frames decoded |
|---|---:|
| CAVLC, no B-frames | **164 / 164** |
| CAVLC + B-frames | **164 / 164** |
| High profile 8x8 transform, CAVLC | **164 / 164** |
| **CABAC, no B-frames** | **49 / 164** |
| **x264 DEFAULT (High: CABAC + 8x8 + B-frames)** | **0 / 164** |

B-frames are fine. The 8x8 transform is fine. **CABAC is the breaking feature**
— and CABAC is the entropy coder, so the failure sits exactly on the axis under
test.

Across the ladder with CABAC forced on:

| clip | bits/frame | written | CAVLC | CABAC | CABAC % |
|---|---:|---:|---:|---:|---:|
| akiyo | 7,407 | 164 | 164 | 49 | **30 %** |
| container | 15,259 | 164 | 164 | 14 | 9 % |
| foreman | 17,605 | 164 | 164 | 14 | 9 % |
| stefan | 36,525 | 90 | 90 | 8 | 9 % |
| bus | 35,848 | 150 | 150 | 11 | 7 % |
| mobile | 56,734 | 164 | 164 | 13 | 8 % |

**Broken on 6 of 6 clips, and worse as entropy rises** — the simplest content
gets 30 % of the way through, everything harder stops at 7-9 %.

**The failure mode is the serious part.** `sample_frames` returns `Ok` with a
short frame list and no error. A caller decoding a normal x264 file gets **zero
frames and no diagnostic** — and x264's default profile is High, so that is the
common case, not a corner.

**Action:** upstream bug report against `rff-codec-h264`; and `ffai-media`
should REFUSE rather than silently truncate — a decode that ends early on a
stream whose container declares N frames is an error, not a result.

## FINDING 2 — CAVLC decode is SUB-linear in entropy. No gate to build.

| clip | bits/frame | ms/frame | **us/kbit** |
|---|---:|---:|---:|
| akiyo | 7,407 | 1.150 | 155.2 |
| container | 15,259 | 1.486 | 97.4 |
| foreman | 17,605 | 1.895 | 107.6 |
| stefan | 36,525 | 2.261 | 61.9 |
| bus | 35,848 | 2.496 | 69.6 |
| mobile | 56,734 | 2.931 | 51.7 |

Cost rises **2.5x** for **7.7x** the bits, so cost/kbit **falls** 155 -> 52.
Sub-linear and healthy: decode is dominated by fixed per-frame, per-macroblock
work rather than by coefficient count.

**A dispatch gate needs cost/bit RISING.** This does the opposite.

*(The first pass labelled a FALLING cost/bit "SUPER-LINEAR: an inflection
exists". The classifier had the direction backwards.)*

## FINDING 3 — the control holds: detection is content-independent

| clip | bits/frame | detect p50 | within-clip spread |
|---|---:|---:|---:|
| akiyo | 7,407 | 46.98 | 47 % |
| container | 15,259 | 46.64 | 44 % |
| foreman | 17,605 | 45.67 | 47 % |
| bus | 35,848 | 44.41 | 49 % |
| stefan | 36,525 | 46.23 | 52 % |
| mobile | 56,734 | 45.55 | 48 % |

**1.06x across a 7.7x entropy span**, and more entropy is nominally FASTER —
the same no-relationship signature as the -0.009 correlation with detection
count. The 6 % trend sits an order of magnitude below the 44-52 % within-clip
spread.

**The first run of this arm said 1.36x and "not flat".** It ran the clips
sequentially, so the numbers tracked run order — slow, slow, fast, fast, slow —
not entropy. Re-run round-robin with the order reversed every rep, the spread
collapsed to 1.06x. §3 again, and this campaign has now been caught by it four
times.

## FINDING 4 — the LIVE gate has a clean entropy cliff

| clip | bits/frame | gated |
|---|---:|---:|
| akiyo | 7,407 | **39 %** |
| container | 15,259 | **38 %** |
| foreman | 17,605 | 0 % |
| bus | 35,848 | 0 % |
| stefan | 36,525 | 0 % |
| mobile | 56,734 | 0 % |

The gate fires on the two lowest-entropy clips and **not at all above ~17
kbit/frame**. Correct behaviour — it is supposed to refuse when content moves —
but the cliff LOCATION is a usable deployment number: **above roughly
17 kbit/frame the gate is pure overhead**, a per-frame pixel diff that never
pays. It joins the 0.8 %-on-MOT17 figure as a second measured bound.

---

## Verdict

| stage | entropy behaviour | dispatch gate? |
|---|---|---|
| CAVLC decode | sub-linear, cost/bit falls 3x | no |
| **CABAC decode** | **broken at every entropy, worse as it rises** | **correctness defect, not a gate** |
| Diana detection | flat, 1.06x over 7.7x | no — control confirmed |
| LIVE gate | clean cliff at ~17 kbit/frame | already a gate; now bounded |

**No missed dispatch gate.** The search found something more valuable: a silent
decoder defect on the most common encoder settings in the world, which only
content variation could have exposed — every prior video test in this project
used clips produced by the one encoder path that happens to work.
