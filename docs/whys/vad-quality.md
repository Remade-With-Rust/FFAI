# WHY did VAD change WER? — a descent that refuted its own premise

**Date:** 2026-07-28
**Probe:** `cargo run --release -p ffai-mercury --example vad_why -- <corpus>`
**Status:** closed. The mechanism does not exist; the effect is a tail artifact.

---

## The symptom

VAD was built as a speed feature: skip silence, skip the encoder pass. It was
predicted to do nothing on LibriSpeech, where clips are continuous read speech
with no silence to skip. The ledger disagreed:

| corpus | WER off → on | CER off → on |
|---|---|---|
| test-clean | 7.99 → **6.79** (15 % rel.) | 3.27 → **2.74** |
| test-other | 16.79 → **16.43** (2.1 % rel.) | 8.34 → **8.07** |

Both corpora moved the same way. The proposed mechanism — *Whisper hallucinates
into silence-padded context, and VAD removes the silence* — was stated
confidently before any of what follows was measured.

---

## WHY 1 — is the gain broad, or a few clips?

200 clips per corpus, per-clip WER in both arms.

| corpus | improved | worsened | unchanged | top-5 share of net gain |
|---|---:|---:|---:|---:|
| test-clean | 17 | 10 | 173 | 89 % |
| test-other | 21 | **28** | 151 | **946 %** |

The median clip is untouched in both. On test-other, **more clips get worse
than better**, and the top-5 share exceeds 100 % — the five best clips
contribute more than the entire net, so the remaining 195 are net *negative*.

**Combined sign test: 38 improved, 38 worsened. z = 0.00.** This campaign
refuses findings below |z| > 2 (§6.25 refused z = +2.3). There is no
directional effect to explain.

---

## WHY 2 — does the gain track the silence removed?

If trimming silence were the mechanism, clips with more trimmed would gain
more. Pearson correlation between seconds trimmed and WER gain:

| corpus | total trimmed | lead only | tail only |
|---|---:|---:|---:|
| test-clean | −0.090 | −0.035 | −0.103 |
| test-other | −0.093 | −0.009 | −0.132 |

Zero on both corpora, and faintly *negative* — the opposite sign to the
hypothesis. Mean trim was 0.20 s lead / 0.19 s tail against a ~7 s clip.

**The trimming mechanism is refuted.**

---

## WHY 3 — did the windowing change?

Window count is identical in both arms for 400 of 400 clips; one clip exceeds
30 s. Not a segmentation-count effect, and not a confound either.

---

## WHY 4 — what changed in the text?

Not hallucinations disappearing. Different recognitions, in both directions:

```
super volus            -> superfluous            (better)
All-turing is Demeter  -> altering his demeanor  (better)
A pause and suit       -> A pause ensued         (better)
You are a worthy leech -> Your wirty leech       (WORSE)
Miles McLaren          -> Mal's McLaren          (WORSE)
```

These are the outputs of a decoder given slightly different input, not the
outputs of a decoder that has stopped hallucinating.

---

## WHY 5 — hallucinated continuations?

Clips emitting more than two words beyond ground truth: **off 0, on 1** on
test-clean. The no-VAD arm was not over-generating at all, so there was no
hallucination for VAD to suppress.

**The hallucination mechanism is refuted.**

---

## WHY 6 — is the instrument sound?

Yes, and it is what makes the rest interpretable. Both arms run in one process
over identical clips with identical ground truth; WER is deterministic under
greedy decode; the only difference between arms is the sample range handed to
the encoder. The measurement is a paired comparison with no timing component,
so machine state cannot enter it.

---

## ANSWER

**VAD does not improve quality. It perturbs the decode.**

Every clip is padded to Whisper's fixed 30 s context. VAD changes where inside
that window the speech sits — by 0.2 s at each end. That shift changes the mel
frames, which changes the encoder output, which occasionally changes a token
choice. On 324 of 400 clips it changes nothing. On the remaining 76 it re-rolls,
**38 up and 38 down**.

Corpus WER moved because WER is a length-weighted aggregate dominated by a
handful of high-delta clips, and on these two corpora those clips landed
favourably. That is not a mechanism. It is a sample.

---

## What this cost, and what caught it

The corpus-level numbers were treated as **independent replication** — two
corpora, same direction, therefore real. That reasoning was wrong and it was
made twice: once when the test-clean result arrived, and again when test-other
agreed. Both times the aggregate was believed before the distribution was
looked at.

Aggregate agreement across corpora is not replication of a mechanism. Two
corpora can both move positive while the per-clip sign test is exactly even,
and here they did.

This is the **fourth** instance in this campaign of the same failure —
open-campaign.md already logs three under "a single favourable sample promoted
to a finding". The first three were caught by arithmetic on reported counts.
This one needed a per-clip decomposition, because the aggregate was
reproducible: re-running it would have produced 6.79 every time.

---

## What survives

VAD stays on by default. Its justification was never the WER number and does
not depend on it:

- **4.2× on audio with trailing silence** (3036 ms → 725 ms), transcript
  byte-identical. A mechanism, not a delta.
- **Silent input produces nothing**, without an encoder pass.
- **Streaming**: the demo's sliding window ran 7 chunks to produce 2 lines;
  five encoder passes bought nothing.

What must **not** be claimed: that VAD improves WER. The shipped configuration
does score 6.79 / 16.43, and those are honest ledger numbers for that
configuration — but attributing them to a quality mechanism is exactly the
inference this descent refutes, and they should not be expected to hold on
other data.

---

## Open, downstream of this

- The perturbation is directionally neutral **at threshold 0.5 with 150 ms
  padding**. Whether some setting is genuinely better is a separate question
  this descent does not answer, and it needs a paired test with |z| > 2 rather
  than a corpus aggregate.
- 28 regressions on test-other are a real cost being paid for the speed win.
  Worth knowing whether padding more (less perturbation) keeps the speed and
  drops the regressions.
