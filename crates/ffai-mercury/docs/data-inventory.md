# ffai-mercury — data inventory and flow map

Satisfies **C-01** (data inventory) and **C-02** (data-flow map) of the hardening
process. Written because the audit found this crate derives **voiceprints**, which are
special-category data under GDPR Art 9 — a fact nothing in the crate previously recorded.

**Reviewed**: 2026-08-15 · **Next review**: 2026-11-15, or on any change to the diarizer,
the speaker module, or what the crate persists.

> **Role note.** `ffai-mercury` is a **library**. Under GDPR it is neither controller nor
> processor on its own; the application embedding it takes those roles. This inventory
> exists so that application can meet its Art 30 duty without reverse-engineering us, and
> so it knows a voiceprint is being produced at all.

## 1. Data classes

| # | Data | Class | Where it enters | Lifetime in this crate |
|---|---|---|---|---|
| D1 | Speech audio (PCM samples) | **Personal data** — content is often confidential | Caller passes it to `transcribe` | In memory for the call; framed, mel-transformed, discarded |
| D2 | Transcript text + word timings | **Personal data** | Derived from D1 | Returned to caller; not persisted |
| D3 | **Speaker embeddings / voiceprints** | **GDPR Art 9 special category** — biometric data when used to identify a person | Derived from D1 by `asr/speaker.rs`, `asr/diarize.rs` | In memory for the call; returned as diarization labels |
| D4 | Speaker labels ("speaker 1") | Pseudonymous; personal in combination | Derived from D3 | Returned to caller |
| D5 | Input text for synthesis | Personal data if it contains any | Caller passes it to the TTS path | In memory; phonemized, synthesized |
| D6 | Synthesized waveform | Personal data if D5 was | Derived from D5 | Returned to caller |
| D7 | Model weights, ONNX graphs, configs, lexicons | Not personal | Read from the model cache | Mapped/parsed for the process lifetime |

**D3 is the one that changes obligations.** A voiceprint capable of distinguishing
speakers is biometric data; when used to *identify* a person it is Art 9 special category,
which requires an Art 9(2) condition (usually explicit consent) on top of an Art 6 lawful
basis. Diarization within a single recording is a weaker case than matching a speaker
across recordings — but the crate produces the embedding either way, and the embedding is
the sensitive artefact.

## 2. Flow map

```
caller ──D1 audio──► ffai-mercury ──► mel/features ──► encoder ──► decoder ──D2 transcript──► caller
                          │
                          ├──► VAD ──► diarizer ──D3 embeddings──► clustering ──D4 labels──► caller
                          │
caller ──D5 text───►      ├──► phonemize ──► VITS ──D6 waveform──► caller
                          │
model cache ──D7──────────┘   (read-only; no verification performed here — see threat-model.md)
```

**Egress from this crate: none.** No sockets are opened; nothing is written to disk;
nothing is sent to a third party. Every data class either returns to the caller or is
dropped when the call ends. Weight downloading — the only network activity in the stack —
belongs to `ffai-models`.

**Therefore**: no subprocessor relationship arises from this crate (C-10), no
transmission-security control applies to it (C-03), and it stores nothing at rest (C-04).
Those obligations attach to whatever embeds it.

## 3. What the embedding application must do

This crate cannot discharge these; they are listed so the integrator does not miss them.

| Obligation | Why it lands on you, not us |
|---|---|
| Lawful basis (Art 6) **and** an Art 9(2) condition for D3 | You have the relationship with the data subject |
| Transparency: tell people a voiceprint is derived | You own the notice; most people do not expect diarization to produce biometrics |
| Retention and erasure (Art 17) reaching **embeddings**, not only recordings | You decide what is stored; deleting the audio does not delete a voiceprint you kept |
| Access control over D1–D4 at rest | We hold nothing at rest |
| Records of processing (Art 30) | You are the controller/processor |
| DPIA where diarization identifies people at scale | Art 35 threshold is about your processing, not our code |

## 3a. Retention inside the process (C-06)

Nothing is written to disk, but "not persisted" is not the same as "not retained".
One structure holds personal data beyond the call that produced it:

| Structure | Holds | Bound | Lifetime | Erasure |
|---|---|---|---|---|
| `Diarizer.cache` | **D3 speaker embeddings** (voiceprints), keyed by a hash of the raw audio window | `EMBED_CACHE_CAP` = 512 entries, ~0.4 MB | the `Diarizer` instance | drop the `Diarizer`, or call `Diarizer::clear_embed_cache()` |

Everything else — audio buffers, mel frames, transcripts, synthesized waveforms — is
owned by the call and dropped when it returns.

**The cache exists for speed**, not for recall: streaming re-embeds a sliding buffer and
~12 of ~13 windows per tick are byte-identical to the previous tick's. Content keying
makes a hit numerically identical to a recompute.

**An audit finding, fixed 2026-08-15.** `clear_embed_cache_counters()` was documented as
"Drop every cached embedding" and only ever reset two atomic counters. Nothing called it
yet, so it was a trap rather than an active bug — but it had two faces: a harness using it
to time a cold arm would have silently measured a warm cache, and anyone calling it to
purge biometric data would have believed an erasure that never happened.
`Diarizer::clear_embed_cache()` now performs it, `embed_cache_len()` lets an auditor see
what is held, and both are regression-tested.

**For the integrator (Art 17).** Erasing a recording does not erase a voiceprint derived
from it. If you hold a `Diarizer` across requests, clear its cache when you honour an
erasure request — or construct one per request and let the drop do it.

## 4. Open items in this crate

| Item | Gate |
|---|---|
| Log hygiene unreviewed — no evidence that D1/D2/D3 never reach a log sink | C-08, H-20 |
| ~~Retention within the process unreviewed~~ — **closed 2026-08-15**: audited, documented in §3a, and the one retaining structure now has a working erasure path | C-06 |
| No API-level signal that D3 is special-category; the type is an ordinary embedding | C-01 (partly closed by this document) |
