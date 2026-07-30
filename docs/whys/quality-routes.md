# WHYS — the two routes to a quality claim

**Question:** Mercury matches whisper.cpp on accuracy because it *is* Whisper —
same weights, and at matched decoding the implementations agree to within
noise. So the honest standing has been "line-ball", never "better". What
would make an accuracy claim defensible?

Two levers were named in the gap inventory and neither had been built:
**beam search** (every reference ships `beam_size=5`; we have been greedy
since M1) and **larger model sizes** (we stopped at base). Both are explored
here.

---

## Route A — beam search

### What was built

`decode_beam` in `asr/decoder.rs`, behind `FFAI_BEAM_SIZE` (default 1 =
greedy, so every existing ledger line stays valid). It reuses
`apply_logit_filters` verbatim, so the timestamp grammar, suppression lists
and the no-speech gate are bit-for-bit the greedy ones — the search changes
which hypothesis wins, nothing about what is legal.

Two implementation points worth keeping:

**Per-beam KV cache snapshots.** Each hypothesis owns a
`text_decoder::DecoderState`. The naive alternative — reset and re-feed each
beam's whole prefix every step — is O(steps²·beams), roughly a hundredfold
more decoder work at beam 5. The snapshot is *shallow*: a candle `Tensor` is
a handle onto refcounted storage and the caches are only ever replaced (by
`Tensor::cat`), never written in place, so snapshots share storage with the
live cache and stay valid as it advances. The cross-attention cache is
identical across beams — same audio — so sharing it is free.

**Length normalization at selection only.** Beams are *expanded* on raw
cumulative log-probability and the winner is chosen on `logprob / len^alpha`
(`FFAI_LENGTH_PENALTY`, default 1.0). Ranking mid-search on the normalized
score biases toward whichever beam is shortest at that step.

Beam search runs at temperature 0 only; above it the fallback ladder is
deliberately *sampling* to escape a degenerate result, and openai-whisper
switches to plain sampling there for the same reason.

### Measured — and this one clears the bar

| corpus | greedy | beam 5 | improved / worsened | z |
|---|---:|---:|---|---:|
| test-clean | 6.36 % | **5.76 %** | 19 / 9 | +1.89 |
| test-other | 13.87 % | **13.12 %** | 25 / 15 | +1.58 |
| **pooled** | | | **44 / 24** | **+2.43** |

CER moves the same way (2.25→2.05, 6.52→6.22; pooled 56/34, **z = +2.32**).

Neither corpus alone reaches |z| > 2, and reporting either alone would be
the "same-direction movement on a second corpus is weak evidence" error this
project has already made once. **Pooled, both metrics clear the bar.** That
is a real, directional, per-clip-verified quality improvement — categorically
better evidence than VAD's z = 0.00 or adaptive context's z = −0.45, both of
which moved a corpus aggregate while the distribution said nothing happened.

**The cost is the headline caveat: ~5× slower.** tiny.en runs 21.6–33.8 ×RT
greedy and 3.6–6.0 ×RT at beam 5. Beam search spends the entire speed surplus
the adaptive-context campaign just won, and then some.

### Trap hit on the way

Four *identical* readings — WER, CER **and throughput** — for beam 1 vs beam
5. That is not a null result, it is the stale-binary signature
(`codec-memory-copies` §4): the example's rebuild had died in a `link.exe`
1104 lock failure while a previous run held the exe, so the binary predated
the beam code. Rebuilt (08:32 → 08:52 mtime) the arms separated immediately.
*Three identical A/Bs in a row means "is my binary rebuilding", not "the
change does nothing".*

---

## Route B — model size

Manifests and registrations for `small.en` (244M) and `medium.en` (769M).
Above medium the family is multilingual-only, which needs the language
detection we do not have, so medium.en is the largest *honestly usable* size
today. The loader is fully config-driven, so this was manifests plus
registration — no model code.

### The ladder (test-clean, greedy, same harness, 200 clips)

| model | WER % | CER % | ×RT |
|---|---:|---:|---:|
| tiny.en | 6.39 | 2.31 | 19.9 |
| base.en | 5.16 | 2.10 | 8.3 |
| **small.en** | **3.05** | **0.88** | 3.2 |

small.en **more than halves** tiny.en's error rate. It is by far the biggest
accuracy lever in this document — beam search's 0.6 pp against small.en's
3.3 pp — and it costs ~6× throughput.

### The size-matched comparison — and the harness bug that nearly shipped

`examples/matched_size.rs` runs both implementations over the same clips at
the same size, scoring both through `ffai_bench::metrics`. test-clean, 200
clips, 1502.8 s audio, both greedy, path-matched:

| implementation | WER % | CER % | ×RT |
|---|---:|---:|---:|
| **mercury** (small.en) | **3.05** | **0.88** | **4.2** |
| whisper.cpp (small.en) | 3.38 | 1.16 | 3.7 |

**Ahead on quality and speed at identical weights** — WER −0.33 pp, CER
−0.28 pp (24 % relative), throughput 1.14×. Direction consistent but not
conclusive alone: 16 better / 8 worse / 176 tied, **z = +1.63**, under the
|z| > 2 bar. Report as "ahead, not yet significant", the same discipline beam
search was held to. (60-clip subset, also path-matched: 2.20 vs 2.45, same
direction.)

**THE INSTRUMENT LIED FIRST, AND ITS MAGNITUDE GAVE IT AWAY.** The first
version of this harness paired transcripts to ground truth **by index**,
assuming the reference emits results in the order the batch file listed them.
On the beam-5 run that assumption broke and it reported:

```
mercury          2.36 % WER
whisper.cpp     20.79 % WER      <- a mature reference, 6x worse than itself
per-clip: mercury better 10 / worse 1, z = +2.71 SIGNIFICANT
```

An 18 pp lead with a significant sign test — a spectacular, publishable,
entirely fictitious result. It was caught by **implausible magnitude**, not
by inspection: whisper.cpp does not decode six times worse at its own default
beam setting. Two probes settled it in minutes — running the reference
directly produced *perfect* transcripts, and re-scoring the same 60
transcripts **matched by path** gave **3.06 %**.

Fixed to pair by path, and to hard-error rather than silently truncate when a
clip is missing from either side. *Rules reconfirmed: never pair two arms by
position; and a result too good to be true is the probe asking for help.*
This is the same class as the PSNR-by-timestamp misalignment in
`codec-analyzer` — the metric, not the system, was wrong.

The 200-clip greedy figures had come from that same by-index harness, so they
could not be trusted either — they were re-run path-matched rather than
quoted on faith. They **reproduced exactly** (3.05 / 0.88 / z = +1.63), so
the greedy pairing had happened to be correct and only the beam-5 run was
scrambled. That is the right resolution: a number you cannot trust is
re-measured, not argued about, and re-measuring cost one run.

### The comparison this does NOT license

Quoting our small.en (3.05 %) against whisper.cpp's tiny.en (7.58 %) would
price the *weights*, not the implementation — precisely the mismatched-
reference error `corpora/references.toml` exists to prevent, and the same
error that made the harness's own verdict line read `quality FAIL` for months
by judging tiny.en against a 74M beam-search model. So
`whisper-cpp-small-greedy-t24` was added as a size-matched reference (and
`whisper-cpp-tiny-beam5-t24` for the separate "each tool at its own defaults"
question), and `examples/matched_size.rs` runs both implementations over the
same clips scoring both through `ffai_bench::metrics`, which normalizes each
side identically.

---

## What can now be claimed

Two different claims, and they should not be blurred:

1. **Product claim (Route B).** Mercury transcribes at 3.05 % WER on
   test-clean with small.en — less than half its own tiny.en error rate — at
   4.2 ×RT. This is a claim about what the toolkit can *do*, and it is the
   one a user actually cares about.
2. **Implementation claim (size-matched).** At the same weights, Mercury
   beats whisper.cpp on WER (3.05 vs 3.38), CER (0.88 vs 1.16) **and**
   throughput (4.2 vs 3.3 ×RT). Direction consistent, per-clip z = +1.63 —
   ahead, not yet significant.
3. **Decoding claim (Route A).** Beam search is a pooled-significant
   improvement over greedy (WER z = +2.43, CER z = +2.32) at ~5× the cost.

Neither is "our model is better than Whisper" — we run Whisper, and the
accuracy ceiling is Whisper's. What changed is that two claims which were
previously unavailable now have numbers: the toolkit reaches an accuracy tier
it could not before, and at matched weights the runtime is ahead on both axes
rather than line-ball on one.

**What would settle claim 2**: it needs the |z| > 2 bar, which means either
test-other at small.en as a second corpus to pool (the same move that carried
beam search over the line), or more clips. That is the next measurement, not
a new feature.
