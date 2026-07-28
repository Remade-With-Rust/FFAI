# Mercury-X Mission Plan

**Component:** the WhisperX enhancement layer inside `ffai-mercury`
**Scope:** VAD segmentation · word-level timestamps · speaker diarization
**Shape:** **opt-in flags on the existing engine — not a fork, not a second engine**
**Parent:** [mercury-mission-plan.md](mercury-mission-plan.md) §3.2, which
reserved this layer. This file is the execution plan for it.
**Prime directive:** unchanged — pure Rust end to end, every claim traceable
to a line in `bench/ledger.jsonl`, and a skipped gate is never a pass.

---

## 1. Why this, why now

Two findings from 2026-07-27/28 pushed this up the queue, and both came from
the side-by-side demo rather than from the corpus.

**Whisper hallucinates on silence.** On a silent chunk the model emitted
`you`. We fixed the symptom with a no-speech gate (§6.31 in the parent plan):
read `P(<|nospeech|>)` at the first decode position and drop the window above
0.6. It works — silence now produces nothing. But it is a **downstream**
fix. We still run the full encoder and a decode pass over the silence and
then throw the result away.

**The demo pays for that on every tick.** A 10 s sliding window at a 1 s tick
processed 7 chunks to produce 2 useful lines. Five encoder passes bought
nothing.

VAD is the upstream fix for both: don't hand silence to Whisper at all.
And it is the *only* thing that unlocks the speed lever we actually need —
see §3.

There is a third motive. The parent plan's speed gate has been **FAIL at
~1.12× behind whisper.cpp** through five milestones of kernel work, and the
remaining kernel wins are single-digit percentages (§6.8, §6.11, §6.25 are
all pruned or marginal). Batched encoder inference is a different order of
lever, and it is downstream of VAD.

---

## 2. Upstream analysis — what WhisperX actually is

Reviewed at `github.com/m-bain/whisperX`, 2026-07-28.

**It is not a Whisper reimplementation.** It is a pipeline wrapped around
one, and the wrapper is the contribution. Bain et al., Interspeech 2023.

| Stage | What it does | Backend upstream |
|---|---|---|
| 1. VAD | segment audio into speech regions before ASR | pyannote **or Silero** |
| 2. ASR | batched transcription, `without_timestamps=True` | faster-whisper (CTranslate2) |
| 3. Align | force-align transcript to audio for word times | wav2vec2-CTC per language |
| 4. Diarize | speaker labels | pyannote (HF-gated) |

**Verified facts** (each checked against the repo, not recalled):

- **License: BSD-2-Clause.** Permissive, compatible with our MIT/Apache-2.0
  dual licence. We are reading it as a *specification*, not vendoring it —
  our implementation is independent and in Rust. Attribution to the paper
  and repo goes in the module docs regardless.
- **`whisperx/vads/` contains `vad.py`, `pyannote.py`, and `silero.py`.**
  Silero is in mainline behind an abstract VAD base class. The README's TODO
  ("Allow silero-vad as alternative VAD option") is **stale** — worth
  recording, because that TODO is what most secondary sources still repeat,
  and it is why several third-party `whisperX-silero` forks exist.
- **`merge_chunks(segments, chunk_size, onset, offset)`** closes a chunk when
  `seg.end - curr_start > chunk_size`, emitting `{start, end, segments[]}`.
  Speech regions are packed into windows up to `chunk_size`, not cut at it.
- **Silero binding** loads `snakers4/silero-vad` via `torch.hub`, passes
  `threshold = vad_onset` and `max_speech_duration_s = chunk_size`, and
  requires 16 kHz. `min_silence_duration_ms` and `min_speech_duration_ms`
  are **commented out** upstream — a knob they chose not to expose.
- **Alignment** is: load per-language phoneme CTC model → extract emissions
  → align text to emissions by **trellis search** → backtrace the optimal
  path → emit word (optionally character) times. 5 languages via torchaudio
  pipelines, ~37 more via HuggingFace.
- **Diarization requires a HuggingFace token** and acceptance of a gated
  model licence.

**Not verified — do not plan around these until checked:**

- Exact numeric defaults for `vad_onset`, `vad_offset`, `chunk_size`,
  `batch_size`. They are not in `utils.py`; they live in the CLI parser or
  the pipeline class, which I did not open. Widely-quoted values
  (0.500 / 0.363 / 30) are **plausible but unconfirmed** and must be read
  from source before we match or deliberately diverge.
- The batching implementation's memory behaviour. One secondary source
  reports `batch_size=16` alongside pyannote exhausting 24 GB of VRAM,
  which if true is a caution about stage coupling, not about the idea.

---

## 3. What we adopt, and what we reject

**Adopt: VAD-first segmentation.** This is the load-bearing idea. Boundaries
fall on silence instead of on a 30 s grid, which fixes three things at once —
no hallucination on empty windows, no words cut mid-utterance at a boundary,
and independent segments that can be batched.

**Adopt: Silero as the default VAD, not pyannote.** Silero is small,
permissively licensed, and does not require a HuggingFace token or licence
acceptance. Pyannote is gated, which sits badly with principle 4 ("weights
are data, fetched from manifests that surface each model's own licence") —
a model you cannot fetch without accepting terms in a browser is not data we
can put in a manifest. The parent plan already called this correctly in
§3.2 ("Silero-VAD-class, ported to candle"); the upstream repo has since
converged on the same option being available.

**Adopt: CTC forced alignment for word timestamps.** Whisper's own timestamp
tokens are coarse and drift. A phoneme CTC model plus Viterbi/trellis
alignment is the standard answer and is well within candle's reach.

**Reject: the batching-shape coupling.** Upstream batches by packing VAD
segments to `chunk_size` and running with `without_timestamps=True`. We
should batch, but our batching must not silently change decode behaviour the
way `without_timestamps` does — that is the same class of defect as §6.17
(`-nt` handing the reference 23 % less work). Batching is a **speed** change
and must be proven output-identical.

**Defer: diarization.** Heaviest stage, needs speaker-embedding + clustering
models and a diarization corpus (DER scoring), and it is the one with the
worst licence story upstream. It is M-X3, and it is allowed to slip.

**Explicitly out of scope:** translation, sentence segmentation via an NLTK
equivalent, and multi-language alignment beyond English at first. English
alignment gates first; languages are a widening, not a milestone.

---

## 4. Flag surface — the user-facing contract

**Stages are off by default and switched on per call**, with one earned
exception. The layer is *added capability*, not a behaviour change to the
existing path — except where a stage is measured to improve the existing path,
in which case defaulting it off would be withholding a win to preserve a rule.

**VAD is on by default as of 2026-07-28, on speed evidence only.** 2.2–4.2× on
audio with trailing silence at a byte-identical transcript, an empty result on
silence without an encoder pass, and a live sliding window that stops paying
for chunks containing nothing.

Corpus WER also moves with it on (test-clean 7.99 → 6.79, test-other
16.79 → 16.43) and that is **not** a quality win. The per-clip decomposition
is 38 improved / 38 worsened over 400 clips — a sign test of z = 0.00 — and
`corr(silence trimmed, WER gain)` is −0.09 on both corpora, the opposite sign
to the mechanism that was proposed. VAD perturbs where speech sits inside the
fixed 30 s context and re-rolls the decode on ~19 % of clips, half each way.
The aggregate moved because WER is dominated by a handful of high-delta clips.
Full descent in [whys/vad-quality.md](whys/vad-quality.md); the transferable
rule went into the `codec-tune-quality` skill.

`--no-vad` restores the fixed-grid behaviour. Alignment and diarization remain
opt-in — they add models and change the output's shape, and neither has earned
anything yet.

```sh
ffai asr -i talk.wav                                  # unchanged, today's behaviour
ffai asr -i talk.wav --vad                            # + VAD segmentation
ffai asr -i talk.wav --word-timestamps                # + forced alignment (implies --vad)
ffai asr -i talk.wav --diarize                        # + speaker labels (implies --vad)
ffai asr -i talk.wav --word-timestamps --diarize      # full WhisperX mode
ffai asr -i talk.wav --vad --vad-threshold 0.4        # tuning, for people who need it
ffai asr -i talk.wav --no-vad                         # explicit off, defeats any implication
```

Library surface — `AsrOptions` already carries `word_timestamps` and
`diarize` from Phase 0; `vad` joins them:

```rust
pub struct AsrOptions {
    // ... existing
    /// Segment on speech before transcribing. Off by default.
    pub vad: bool,
    pub vad_threshold: f32,     // default from the ported model, not invented
    pub vad_chunk_secs: f32,    // pack speech regions up to this, default 30.0
    pub word_timestamps: bool,  // implies vad
    pub diarize: bool,          // implies vad
}
```

Rules that make the flags honest:

1. **Off is genuinely off.** `--vad` absent means not a byte of VAD code
   runs and no VAD model is fetched. No lazy "it's cheap so always on".
2. **Implication is explicit and reported.** `--word-timestamps` turning on
   VAD is stated in the run header, not silent. `--no-vad` wins over an
   implication and errors if combined with a stage that requires it, rather
   than quietly degrading.
3. **A missing model is an error, not a silent downgrade.** If the VAD
   weights are not cached and cannot be fetched, the run fails with the
   manifest name. Silently transcribing without the requested stage is how
   a benchmark lies.
4. **The engine trait does not change.** `AsrEngine` is untouched; this is
   all inside `whisper-candle`.
5. **Interaction with the no-speech gate is defence in depth.** The §6.31
   gate stays on when VAD is on. VAD should make it near-unreachable; if the
   gate keeps firing with VAD enabled, VAD is mis-tuned and we want to know.
   Instrument the count, do not remove the gate.

---

## 5. Architecture

```
audio ──▶ vad.rs ──▶ segments ──▶ batch ──▶ encoder ──▶ decoder ──▶ text
             │                                                       │
             └──────────────── align.rs ◀────────────────────────────┘
                                  │
                              diarize.rs
```

Three new modules, each an independently callable function with its own
contract — the parent plan's §2 design rule, unchanged:

| Module | Contract |
|---|---|
| `asr/vad.rs` | `AudioBuffer → Vec<TimedSegment<()>>` |
| `asr/align.rs` | `(AudioBuffer, Transcript) → Transcript` with per-word `TimedSegment`s |
| `asr/diarize.rs` | `(AudioBuffer, Transcript) → Transcript` with `speaker` labels |

Each is testable without the others. `vad.rs` in particular must be usable
standalone — it is independently valuable (it is a VAD, people want those)
and that keeps its tests honest.

---

## 6. Milestones and exit gates

The four-gate discipline applies unchanged: **correctness / quality / speed /
footprint**, and a skipped gate is never a pass. Each milestone appends a
ledger line.

### M-X0 — the corpora, before any code

This layer cannot be gated on what we already have. LibriSpeech test-clean
and test-other are continuous read speech: **they cannot fail a VAD, cannot
score a word timestamp, and contain one speaker.** Building the feature
first and discovering that afterwards is precisely the trap that hid the
silence bug for five milestones.

Required before M-X1 starts:

1. **A silence/noise corpus.** Digital silence, room tone, music, a cough,
   non-speech noise. Ground truth: *empty transcript*. This is the corpus
   that would have caught `you` on day one, and it is cheap to build.
2. **A word-alignment corpus.** LibriSpeech has published forced alignments;
   pin a subset with hashes the way the existing corpora are pinned.
   Metric: median absolute word-boundary error, and % of boundaries within
   a 100 ms collar.
3. **A long-form corpus.** VAD's advantage over a 30 s grid only appears on
   audio long enough to have structure. Read-speech clips of ~8 s cannot
   show it.

Exit: three corpus TOMLs with hashes, and every existing reference scored on
them so we know the bar before we build.

### M-X1 — VAD segmentation

Port a Silero-VAD-class model to candle. Implement `merge_chunks`-equivalent
packing (close a window when adding the next region would exceed
`vad_chunk_secs`, do not cut regions).

**Exit gates:**
- correctness: silence corpus produces empty transcripts, 100 %
- quality: WER on test-clean and test-other **unchanged or better** with
  `--vad` on. VAD that costs accuracy on continuous speech is mis-tuned.
- speed: measurable win on the long-form corpus (silence is no longer
  encoded). Report ×RT with and without.
- footprint: the VAD model's resident cost stated, not waved at.
- **Ablation required:** the no-speech gate's fire count with VAD on. Near
  zero is the expected result and is the evidence VAD is doing its job.

### M-X2 — batched encoder — **PRUNED 2026-07-28, before implementation**

> Pruned on arithmetic, not on effort. Two independent reasons, either
> sufficient: **(1)** every corpus clip produces exactly **one** 30 s window,
> so the applicable batch size is 1 and there is nothing to batch; **(2)** the
> encoder's dominant matmuls already run at **84–94 % of compute peak** at
> batch 1 on matrices that are already large (`1500 × 384 @ 384 × 1536`), so
> raising arithmetic intensity cannot help — the ceiling is ~1.11× on part of
> one stage.
>
> WhisperX's batching number comes from GPU occupancy, where batch 1 leaves
> the device idle. It does not transfer to a CPU already at 90 % of peak.
>
> The stage that *would* pay on CPU is the **decoder** — every op is a GEMV
> streaming full weights per token, ~80 MB per token on the vocabulary
> projection — but that needs ≥ 2 windows too, and lockstep decoding across
> windows of different lengths is real machinery in exactly the path where the
> byte-identical gate is hardest to hold.
>
> Full descent, including the proposed redefinition (M-X2′: throughput across
> *files* rather than windows) and why it should wait for a workload that
> needs it: [whys/mx2-batching.md](whys/mx2-batching.md).
>
> Cost to find out: four profiler runs and a window count. The alternative was
> extending four hand-written AVX2 kernels — `conv1d_gemm` and the
> transposed-K attention path both silently assume batch 1 while the encoder's
> signature advertises a batch dimension — for a measured prize of ~0.

<details>
<summary>Original specification (kept for the record)</summary>

#### M-X2 — batched encoder, on top of VAD

The speed milestone. Independent segments → batch the encoder.

**Exit gates:**
- correctness: **transcripts byte-identical** to unbatched on the full
  corpus. Not "close" — identical. Batching is an arithmetic reordering and
  anything else is a bug.
- speed: this is the milestone's whole point; it must move ×RT materially or
  it is pruned like §6.8.
- footprint: batch memory is peak-dominant by nature; report both steady and
  peak and state the batch size.
- Null arm required (the parent plan's standing rule): a batch size of 1
  must reproduce the unbatched number.

</details>

### M-X3 — word-level timestamps

wav2vec2-CTC alignment, English first. Trellis forward pass + backtrace.

**Exit gates:**
- correctness: every word in the transcript receives a monotonic,
  non-overlapping interval inside its segment.
- quality: median boundary error and 100 ms-collar rate vs the alignment
  corpus, **compared against WhisperX itself** — it is the declared bar for
  this capability in the parent plan §3.3.
- speed: alignment is a second model over the same audio; its cost is
  reported separately so the flag's price is visible.

### M-X4 — diarization

Speaker embeddings + clustering. Scored as **DER** against a diarization
corpus. Allowed to slip; gated the same way.

---

## 7. Risks, honestly

**The licence trap.** Diarization upstream depends on gated weights. If our
chosen embedding model has the same problem, the manifest principle says we
do not ship it as a default — we surface the licence and make the user
fetch it knowingly. Check this *before* M-X4 design, not during.

**Porting a VAD is not free.** Silero is small but it is a real model with
its own preprocessing. Budget for the port to be its own piece of work with
its own correctness gate against the reference implementation's segment
boundaries, not a footnote to M-X1.

**Batching can lie.** `without_timestamps=True` upstream exists to make
batching possible. If we find ourselves changing decode behaviour to enable a
batch, that is the §6.17 defect reappearing — a speed win bought with less
work. The byte-identical gate in M-X2 exists to make that impossible to ship
by accident.

**VAD can cost WER.** Aggressive thresholds clip word onsets. The M-X1
quality gate is "unchanged or better", and if tuning cannot reach it, the
honest outcome is that `--vad` ships as opt-in-with-a-caveat rather than
being quietly defaulted on.

**Our own blind spot repeats.** Everything in §6 of the parent plan that
went wrong went wrong because the corpus could not see it. M-X0 exists for
that reason and must not be skipped to get to the interesting part.

---

## 8. Relationship to today's decisions

- The **no-speech gate** (parent §6.31) stays. VAD makes it near-redundant;
  redundant safety on a hallucination is worth keeping.
- **`suppress_non_speech: false`** — we match whisper.cpp and annotate
  non-speech events, at a measured cost of 0.22 pp WER on test-clean
  (7.77 → 7.99). VAD interacts with this: segments that are pure noise
  should never reach the decoder, so the annotation path should fire less
  often once `--vad` is on. Whether that recovers part of the 0.22 pp is a
  **question M-X1 can answer**, and it is the first candidate for the
  "optimize quality" work queued alongside this plan.
- The **CER gap** (3.27 vs whisper.cpp 2.87, ~14 % relative, test-clean
  only, unexplained since §6.7) is *not* addressed by this layer. It is
  tracked separately and should not be folded into these milestones.

---

## 9. Attribution

WhisperX — Bain, Huh, Han, Zisserman, *WhisperX: Time-Accurate Speech
Transcription of Long-Form Audio*, Interspeech 2023.
`github.com/m-bain/whisperX`, BSD-2-Clause.

FFai's implementation is independent and in Rust. Where we match an upstream
algorithm or default, the module docs say so and name the source; where we
diverge, they say why and carry the measurement.
