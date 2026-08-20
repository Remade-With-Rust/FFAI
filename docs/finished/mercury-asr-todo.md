# Mercury ASR — what is missing

**Audited 2026-07-29**, after Mercury-X M-X1…M-X5 completed and Phase E gated
them. Nothing here is scheduled: this is the list of gaps, ranked by what
their absence actually costs, so the next decision is made from a written
inventory rather than memory.

**Current posture: no new features.** Testing what exists takes priority —
the audit that produced this list also found `audio_encoder.rs` had **zero
tests** while containing a live batch bug, which is the argument for that
posture in one sentence.

---

## What is complete, and gated

| Capability | Gate |
|---|---|
| Greedy decode, timestamp grammar, suppression lists, temperature fallback | WER 6.79 / 16.43 — ahead of whisper.cpp on both holdouts |
| No-speech gate (silence yields nothing) | silence corpus **8/8 empty** |
| VAD segmentation, on by default | ↑ same corpus; 2.2–4.2× on trailing silence |
| Batched encoder | bit-exact vs unbatched; batch-stride regression tested |
| Word timestamps (CTC forced alignment) | containment **100 %**, 1105 words, multi-window |
| Diarization, batch | **DER 4.21 %** |
| Diarization, streaming / online | **DER 5.68 %** vs 53.58 % without |
| Output: text, SRT, VTT + word tags, JSON + words + speakers | 10 tests |
| Quantization: q8_0 decoder; q8_0 alignment (opt-in) | steady 509 → 266 MiB, quality identical |

---

## 1. Multilingual — the gap that changes what the tool *is*

**Status: effectively absent.** Whisper's headline capability is 99
languages; Mercury does English.

- Only `.en` engines are registered: `whisper-candle` (tiny.en) and
  `whisper-candle-base` (base.en), plus their q8_0 variants.
- `models/whisper-tiny.toml` (multilingual) and `models/whisper-large-v3.toml`
  exist as manifests with **no engine using them**.
- **There is no language detection at all.** `AsrOptions::language` is
  documented as "force a language instead of auto-detecting", and the
  auto-detect half was never built. On a multilingual model with
  `language: None`, the match in `whisper_candle.rs` falls through to
  `_ => None` and emits no language token — undefined behaviour for Whisper,
  not a graceful default.

**Work:** register a multilingual engine; implement detection (Whisper does it
by reading the language-token distribution at the first decode position — the
same position the no-speech gate already reads, so the hook exists); decide
what `language: None` means on an `.en` model versus a multilingual one.

**Gate needed:** a multilingual corpus. Common Voice or FLEURS, pinned and
hashed like the rest. None of the five current corpora contains a non-English
word.

## 2. `--translate` is unreachable from the CLI

`DecodeConfig::translate` works and `AsrOptions::translate` carries it, but
`crates/ffai-cli/src/main.rs` hardcodes `translate: false`. The library can do
it; the binary cannot ask.

Moot until multilingual lands — translation on an `.en` model is a no-op — but
it is a one-line gap between two layers that should not survive the fix.

## 3. Beam search

**Greedy only.** `ROADMAP.md` Phase 1 still lists it ⬜, and
`decoder.rs`'s header comment still says "Beam search and temperature fallback
land in M2" — temperature fallback landed, beam search did not.

Every reference we benchmark ships `beam_size=5` by default. We pin them to
greedy so the comparison measures implementations rather than decoding
strategies, which is honest — but it also means Mercury cannot compete on
*absolute* quality with anything running its own defaults. Typically worth
0.5–1.5 pp WER.

## 4. Model sizes stop at base

No small / medium / large path. `whisper-large-v3.toml` sits unregistered. A
user who wants accuracy over speed has nowhere to go.

Cheap in code — the engine is already parameterised by manifest name — and
expensive in bench time, since every size needs its own gate line.

## 5. Prefix conditioning — cheapest, and possibly answers an open question

Whisper carries context across 30 s windows using `<|startofprev|>`. Mercury
does not. `tk.prev` appears in exactly one place in the decoder: **the
suppression list**. We suppress the token Whisper uses for coherence and never
use it for anything else.

This is a concrete mechanism for the long-form WER left unexplained in Phase E
(**10.55 % on long-form against 6.79 % on clips**). Unlike the VAD hypothesis
— which was refuted at z = 0.00 — this one names a specific missing input
rather than a vague context effect.

**RESOLVED 2026-07-29 — built, gated, and the gate said no** (full descent:
[whys/adaptive-context.md](whys/adaptive-context.md)). `longform_why` ran
first, as this note asked, and it was right to insist: the long-form penalty
was NOT diffuse context confusion — it was two utterances swallowed whole
(0.00 → 1.00 WER) behind stretched segment timestamps. Prefix conditioning
was then built anyway (`FFAI_PREV_CONTEXT=on`, exact openai token budget,
reset-on-high-temperature) and measured: in-context WER **13.16 % → 36.90 %**
(31 worse / 4 better, z = +4.56) — on a corpus of concatenated UNRELATED
utterances, conditioning on the previous book makes the model skip spliced
content. It ships opt-in-off with that number attached. What actually fixed
the long-form gap was hardening the coverage repair against lying spans:
**long-form WER 10.55 % → 7.19 %**. Re-evaluate conditioning only on a
coherent-long-form corpus.

Also unbuilt: `initial_prompt`, the caller-supplied hint openai-whisper uses
to bias vocabulary and style.

## 6. Smaller gaps

- **Segment confidence is always `None`.** Words carry confidence; segments
  have the field and never populate it — a value that looks available and
  never is.
- **No partial / provisional results.** `persist_speakers` gives streaming
  diarization, but each `transcribe` call is independent. Real streaming ASR
  emits provisional text that is later revised.
- **GPU untested.** `cuda` / `metal` feature flags exist and forward to
  candle; nothing is gated on either.
- **Word timestamps are English-only** — the alignment model is per-language,
  so this is downstream of §1.

---

## Testing debt — the current priority

- **`audio_encoder.rs` had zero tests** and contained a live batch bug: it
  indexed `src[c * l_in..]` with no batch stride while its signature promised
  `(batch, n_mels, frames)`, silently returning item 0's features for every
  item. Fixed in M-X2 and **now covered** by two tests that assert on content
  rather than shape — a shape assertion passed throughout the bug's life.
- **Audit the other hot modules the same way.** The question is not "is there
  a test" but "would a test have caught *that* bug" — the batch bug produced
  correct shapes and wrong contents, which is the class most tests miss.
- **Verification precision.** Word timing is gated at utterance granularity,
  not milliseconds; the diarization corpus has no speaker overlap or natural
  turn-taking. Both gate *regression*, not *readiness*.
- **The q8_0 timing anomaly is unexplained.** Per file, q8_0 and f32 are at
  parity; four files in one process shows 2.9×. Recorded in `aligner.rs` and
  still open.
