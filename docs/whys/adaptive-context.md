# WHYS — the side-by-side, the residual gap, and the padding lever

**Mission:** side-by-side Mercury vs whisper.cpp; localize where we are
slower; find a lever that is faster AND higher quality.
Method: `codec-six-whys-unknowns` + `codec-content-adaptive-dispatch`.

## D6 first — is the earlier "1.60× slower encoder" real?

`compare_stages` (30 clips, one whisper.cpp invocation then one Mercury pass)
read encode **1.60× slower, 117 % of the total gap**. But the arms ran
sequentially on a box whose op-level spread is 28–49 %, and single-instrument
distributions disagreed:

| instrument | encoder ms/window |
|---|---|
| Mercury `profile_dump`, 3 idle runs | **175–179** (stable) |
| whisper.cpp `whisper-bench -t 24`, 3 runs | **144–160** |
| `compare_stages`'s Mercury arm, hot box | 307 |

Real encoder gap: **~1.15×**, consistent with OPEN.md's 1.13–1.15×. The
1.60× was position bias — the reference ran first on a cold box. Decode and
sampling we WIN (0.88×/token, 0.62×). The whole residual gap is the encoder.

## The lever — the encoder is O(n) and mostly encodes silence

10 clean clips = 65.4 s of audio became 10 × 30 s = 300 s of encode: **~78 %
of encoder work is the fixed 30 s padding** (§6.4 measured 69 % and O(n)
scaling; both confirmed). §6.4's variable-length window was PRUNED at 268 %
WER — but that prune predates the repetition guard, the temperature ladder,
and the seek loop. A refutation expires when its baseline moves.

**Adaptive context** (`FFAI_ADAPTIVE_CTX`): encode each window at
`bucket(window_secs + 1 s)` — 5 s steps, 10 s floor — with the timestamp
grammar masked at the encoded extent, and **escalation, not acceptance, on
guard failure**: a short-context decode that trips the confidence or
repetition guard is re-decoded at the full 30 s context, which is
byte-for-byte the shipped path. The §6.4 failure mode (decoder repetition on
truncated context) is therefore caught per-window and repaired with today's
exact behaviour, not shipped.

Paired smoke (same process, back-to-back, 10 clean clips):
wall **5.23 s → 1.95 s**, encoder 331 → 67 ms/window, decoder ALSO faster
(6.35 → 4.47 ms/token — cross-attention reads 500 keys, not 1500).

## Quality half A — prefix conditioning: REFUTED on this corpus, kept opt-in

The long-form 10.55 % WER decomposed (longform_why, first run ever): alone
9.20 % vs in-context 13.16 %, **z = +1.50 (below the bar)** — and the whole
context penalty is two utterances that go 0.00 → 1.00 (a 3.3 s and a 9.9 s
utterance swallowed whole). Root cause, reproduced at longform-02 82.3 s: a
window resuming mid-sentence decodes the sentence tail with no memory of its
head, overshoots the tail's end-timestamp by +2.3 s, and the short utterance
under the overshoot vanishes — the overshoot also shrinks the coverage-repair
hole to 0.97 s, 30 ms under its 1 s floor.

`<|startofprev|>` conditioning (`FFAI_PREV_CONTEXT=on`, openai-whisper's
`condition_on_previous_text`) was the named mechanism for exactly this.
Measured: **in-context WER 13.16 % → 36.90 %, 31 worse / 4 better,
z = +4.56 — SIGNIFICANT in the wrong direction.** On a corpus of concatenated
UNRELATED utterances, conditioning on the previous (unrelated) book makes the
model skip spliced content far more often. Kept as an opt-in knob with this
number attached; do not default it on without a coherent-long-form corpus
that could ever vote the other way.

## Quality half B — the repair pass stops trusting stretched spans

Both drops hid behind a segment whose claimed span its own word count cannot
cover ("Their masters said Mrs. Neverband." claiming 12.4 s = 0.5 words/s).
For COVERAGE ACCOUNTING ONLY, each segment's claim is clipped to
`start + words/1.0 + 1 s`; repair windows start 2 s before the hole because a
clipped claim's edge is fuzzy by construction. Measured on longform-04: the
swallowed 9.9 s utterance is re-decoded and recovered.

Residual known: the 3.3 s drop on longform-02 leaves a 0.97 s hole, still
under the 1 s floor. Catching it needs word-level coverage (the aligner), not
a lower floor — recorded as open work, not tuned to the example.

## The dispatch iterations (each driven by a measured failure)

1. **v1, confidence bar −1.0:** test-clean neutral; test-other WER
   13.66 → 15.29 %, 41 worse / 23 improved, z = −2.25. FAIL. One mover read
   WER 1.04 — insertions, a pipeline smell, not a model one.
2. **The duplication bug (7975-280063-0002):** the short-context decode
   transcribed the whole window correctly but closed its segment at 1.2 s of
   7.0 s of speech; the seek loop trusted that, resumed at 1.2 s, and
   re-decoded the same speech into a duplicate. Fix: **geometry guards** —
   escalate when the decode stops > 1 s short of the VAD speech extent, or
   when any segment claims a span beyond `words/1.0 + 2 s`. → WER z = −1.92,
   CER z = −2.19. Still FAIL, but now the residue was genuine model
   degradation on hard speakers, not plumbing.
3. **Confidence routing (bar −0.5, `FFAI_CTX_LOGPROB`):** accept the short
   context only when the model's own mean logprob clears a stricter bar —
   the −1.0 bar asks "is this salvageable", the wrong question for trusting
   a reduced context. Quality neutral (z = −1.06 / −0.97) but 61 % of noisy
   windows escalated and the tax made test-other 6-8 % SLOWER.
4. **Early abort** (`early_abort_logprob`, bar − 0.05 from token 14): a
   decode headed for rejection stops within ~14 tokens instead of running to
   completion. Other −8 % → −4 %.
5. **The 15 s attempt cap:** past the 15 s bucket the attempt risks the most
   encode to save the least. Other → neutral (27.4/28.4 → 27.1/27.2 ×RT,
   inside noise); clean holds 1.46-1.48×.

**Routing signals pruned, with instruments (do not rebuild without new
data):** (a) VAD energy contrast (p90 − p10 dB) does NOT separate the
corpora — clean p50 35.6 vs other p50 35.5; test-other is hard SPEAKERS on a
clean channel, not a noisy channel. (b) Nor does it predict escalation
within a corpus — escalated p50 33.1 vs kept 36.2, full overlap
(`ctx_route_probe`). The only signal that knows a window is hard is the
model's own logprob, which is inherently post-decode; a genuinely
pre-decode difficulty signal remains open work and would delete the
remaining escalation tax.

## Final gates (all PASS, 2026-07-29)

- **Quality, shipped config** (`ab_clips`, per-clip): clean 9/11/180,
  z = −0.45; other 12/17/171, z = −0.93; CER both |z| ≤ 0.69. Aggregates
  within the documented run spread. Long-form: 4/4 clips byte-identical
  (30 s windows never trigger the short context).
- **Speed, ABAB paired 30 clips:** clean 35.2/35.4 → 52.2/51.0 ×RT
  (**1.47×**); other 27.4/28.4 → 27.1/27.2 (neutral).
- **Side-by-side vs whisper.cpp** (`compare_stages`, 2 runs, their
  flash-attention default, matched greedy, no `-nt`): Mercury wins EVERY
  stage — encode **1.98× / 2.02×**, decode 1.08×/1.19× (0.85×/0.77× per
  token), mel 1.35×/1.48×, sample 1.73×/1.95×. Totals 9.74 → 6.54 s and
  8.38 → 5.24 s: **~1.5-1.6× faster end-to-end** on the clean 30-clip set.
- **Conditional annotations (Q1) gated: NULL.** 197-199 of 200 clips
  unchanged on both corpora. The 0.22 pp cost that motivated it was
  measured at the 7.99 %-era baseline and no longer exists. Default stays
  `None`; the knob stays for regression checks.
- **Long-form quality** (repair hardening, measured by the same
  `ffai_bench::metrics` code on the same clips): **WER 10.55 % → 7.19 %,
  CER 6.86 % → 2.90 %**. longform_why's context penalty fell from +3.96 pp
  to +2.55 pp (11 worse / 5 better / 40 tied, z = +1.50 — the residue is
  diffuse and below the significance bar; the catastrophic-drop class is
  what the fix removed).

## Spending the surplus on the last 0.42 pp — TWO REFUTATIONS, then stop

Long-form is the only corpus still behind (6.89 % vs whisper.cpp's 6.47 %).
Two ways to spend the new speed surplus on it were built and measured; both
lost, so the remaining gap is called architectural rather than swung at a
third time.

**(1) Lower the repair hole floor 1.0 → 0.5 s. REVERTED.** The target — the
3.3 s "YOU DO ME A GREAT HONOUR" at longform-02 82.3 s — leaves **no hole at
any floor**: the following segment abuts the overshooting one exactly
(84.64/84.64), so the utterance is absorbed between two contiguous,
*individually plausible* spans. Span-based coverage is structurally blind to
that. Meanwhile 0.5 s false-fired 2-3 times per short-clip corpus where
1.0 s fires zero. Kept from the attempt: the repair keep-filter now requires
segments to **overlap** the hole rather than merely precede its end, which
bounds what any future false fire can inject.

**(2) Narrower packing windows (the surplus spent on coverage). REFUTED,
4 points, monotone on BOTH axes.** Fewer utterances per window should mean
fewer absorptions, and adaptive context makes a 15 s window ~half the encode
of a 30 s one — so this was affordable in a way it was not before:

| `vad_chunk_secs` | WER % | CER % | ×RT |
|---|---:|---:|---:|
| **30 (shipped)** | **7.19** | **2.90** | **45.0** |
| 20 | 9.42 | 4.91 | 34.4 |
| 15 | 9.67 | 4.42 | 31.6 |
| 10 | 9.68 | 4.97 | 23.4 |

Worse AND slower at every step — more windows means more encoder passes, and
less context per pass is off-distribution, the same mechanism §6.4 found. The
speed surplus cannot be converted into long-form quality this way.

**The bridge, named:** the failure class is an utterance absorbed between two
contiguous plausible spans, which no span-level accounting can see. The fix is
**word-level coverage** — run the CTC aligner over each window's audio and
compare its word timings against the emitted segments, so a swallowed
utterance shows up as a run of unaligned audio rather than as a hole that
isn't there. That is a real piece of work, not a threshold, and it is the
next long-form brick.

## Residual known issues

- The 3.3 s drop on longform-02 leaves a 0.97 s hole, under the repair's
  1 s floor; word-level coverage (the aligner) is the honest fix.
- The escalation tax on hard-speaker content is ~0-3 % — deleting it needs
  a pre-decode difficulty signal (both energy-based candidates refuted).
- `FFAI_PREV_CONTEXT` stays opt-in-off; re-evaluate only on a COHERENT
  long-form corpus (podcast/lecture, one speaker), where the mechanism has
  a chance to vote the other way.
