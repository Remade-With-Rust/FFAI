# Argus — launch plan

**Created 2026-08-21.** Argus is FFai's VLM component: image/video → language.
It is a registered stub today (66 lines, returns `NotImplemented`). This is the
plan to make it real.

Carmenta is the template and the warning. It reached 0.116 text on
OmniDocBench by one discipline: **the reference implementation's own evaluator
is the only scoreboard, and our own metric lied to us for a year before we
learned that** (§8.173 — a scorer biased 2.8× in our favour produced four
shipped mechanisms that were artifacts). Every number in this document that
comes from a vendor's published table is a **claim**, not a measurement, until
it is reproduced locally. They are marked as such.

---

## 0. What Argus is FOR — decided by measurement, not ambition

The §57 loss ledger attributed every point of Carmenta's remaining error from
the evaluator's own per-block records. Two entries are VLM-shaped:

| Carmenta's remaining loss | share | VLM-addressable? |
|---|---:|---|
| inline-math blocks | 36 % | partly — the reference answer is a *specialist* formula model, not a general VLM |
| recognition substitutions | 20 % | yes — a language model repairs `tne` → `the` from context; a CTC head cannot |
| assembly/matcher, unread CJK, detection, harness | 44 % | no — pipeline work, already named and priced |

So Argus has **two distinct missions**, and conflating them is the first way
this goes wrong:

- **A — the product mission (the roadmap's Phase 4).** "Describe this image /
  this video." Captioning, VQA, a chaptered timed track from a video. Nothing
  else in the toolkit does this: Carmenta reads text that *exists* in an image
  and correctly returns nothing for a photo of a dog; Diana returns boxes and
  labels, not description. This is what completes FFai's pitch.
- **B — the repair mission.** A VLM tier that rescoring Carmenta's output could
  attack ~20 % of its remaining loss.

**Mission A is the one to build.** Mission B is a *hypothesis* until a probe
prices it (§4, gate B1) — and Carmenta's cheaper, named pipeline levers
(server-tier recognizer, vertical CJK, matcher alignment) come first regardless.
Do not justify Argus by mission B without that probe.

---

## 1. The champions, the corpus, and the scoreboard

### 1.1 The scoreboard: VLMEvalKit is our OmniDocBench

The direct analog exists and it is the right one to adopt:
**[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** (OpenCompass) —
one-command evaluation, 220+ models and 80+ benchmarks implemented, and it is
the harness that produces the public **OpenVLM Leaderboard**. Same shape as
`omnidocbench-eval`: their code, their prompts, their metrics, their answer
extraction.

> **This matters more than the model choice.** VLM benchmarks are mostly
> multiple-choice or short-answer, which means *answer extraction* — parsing
> "B" or "the third one" out of free text — is part of the metric. Every
> implementation does it differently, and a home-grown extractor is exactly
> the 2.8×-biased scorer again. **Run their harness or the number is fiction.**

The OpenVLM core leaderboard is eight benchmarks, and it is a good spine:

| benchmark | what it measures | why it's in the core set |
|---|---|---|
| **MMBench v1.1** | all-round capability, EN + CN | CircularEval — rotates answer order to kill position bias |
| **MMStar** | vision-*dependent* questions | curated to remove items answerable without the image |
| **MMMU** | college-level, 30 subjects | the "MMLU of VLMs" — reasoning, not perception |
| **MathVista** | visual math | the closest general benchmark to our math gap |
| **HallusionBench** | hallucination & illusion | the failure mode a small VLM has most of |
| **AI2D** | diagram understanding | structured-image reasoning |
| **OCRBench** | text in images, 0–1000 | **our home turf — Carmenta's strength should show here** |
| **MMVet** | integrated capability | ⚠ GPT-judged — costs API calls and is not reproducible offline |

**Add for FFai specifically**, beyond the core eight:

| benchmark | why we need it |
|---|---|
| **DocVQA** (ANLS) | document VQA — the Carmenta↔Argus overlap, and mission B's proving ground |
| **ChartQA**, **InfoVQA** | structured visuals; where OCR + reasoning must combine |
| **TextVQA** | scene text — Carmenta's photo front, asked as a question |
| **OCRBench v2** | the expanded OCR suite ([site](https://99franklin.github.io/ocrbench_v2/)) |
| **Video-MME / MVBench** | video understanding — `describe_video` has no other gate |
| **COCO Captions (CIDEr)** | plain captioning; the one metric that is purely generative |

> **Gate 1.1 — stand up the harness FIRST, and score a REFERENCE model
> through it before Argus produces a single token.** Carmenta's §32 lesson:
> we scored the reference implementation under our own harness and found the
> "residual gap" was the harness. If VLMEvalKit + a known model does not
> reproduce that model's published row, the harness is wrong and every Argus
> number after it would be wrong too.

### 1.2 The champions — two tiers, and only one of them is a target

⚠ **All figures below are vendor/leaderboard CLAIMS gathered 2026-08-21 and
must be re-pinned at build time.** They set direction, not targets.

**Tier 1 — the frontier (the bar, not the goal).** GPT-5-class, Gemini 3,
Claude Opus 4.x. Closed, enormous, API-only. Argus will not beat these and
should never claim to. They are the "PaddleOCR-VL" of this campaign: the
number you cite to be honest about where you stand.

**Tier 2 — open weights (the real competitive field).**

| model | claimed MMMU | claimed OCRBench | note |
|---|---:|---:|---|
| Llama 4 Maverick | ~73.4 | — | leaderboard leader among open weights |
| Qwen2.5-VL-72B | ~70.2 | ~888 | strongest OCR of the open tier |
| InternVL3-78B | ~72.2 | — | strongest MIT-licensed |
| **InternVL3.5-8B** | **~73.4** | **~840** | ⚠ 8B claiming 78B-class MMMU — verify before believing |

**Tier 3 — the portable tier, and this is where Argus actually lives.**
FFai is CPU-first, pure Rust, no CUDA requirement. A 72B model is not a
candidate; a 256M–4B model is.

| model | size | claimed scores | why it's a candidate |
|---|---|---|---|
| **SmolVLM2** | 256M / 500M / 2.2B | 256M: OCRBench 52.6, DocVQA 58.3, ChartQA 55.6 | explicitly designed for the memory floor; Apache-2.0; the 256M runs in <1 GB |
| **Qwen2.5-VL-3B / Qwen3-VL-2B** | 2–3B | strong OCR lineage | shares the family whose OCR we already trust (PP-OCRv5 ≠ Qwen, but the VL line is OCR-heavy) |
| **InternVL3.5-2B** | 2B | family claims strong small-tier | MIT |
| **Moondream** | ~1.8B | runs in 2 GB | built for tiny deployment |

> **Gate 1.2 — the port-cost triage.** Before choosing, answer for each
> candidate: (a) is the vision tower a plain ViT/SigLIP we can express in
> candle, or something exotic? (b) is the connector an MLP (easy) or a
> Q-Former/resampler (harder)? (c) is the LLM a Llama/Qwen-shaped decoder
> candle already has reference implementations for? (d) is dynamic-resolution
> tiling required for its claimed scores? **Pick the model whose architecture
> is the cheapest honest port, not the one with the best row.** That is exactly
> how `mobiledet-svtr` was chosen over a 3B VLM and it is why Carmenta shipped.

### 1.3 The corpus discipline (inherited, non-negotiable)

- `corpora/argus-*.toml` following the existing corpus format; `ffai-bench`
  already accepts `task = "vlm"` (`corpus.rs:70`, `reference.rs:37`) — the
  harness slot exists, unused.
- References declared in `corpora/references.toml` with **decode config pinned
  explicitly**, the way every ASR reference is. Sampling temperature, top-p,
  max tokens: unpinned, we would be comparing decoding strategies while
  pretending to compare models.
- Every claim traceable to a ledger line. No exceptions.

---

## 2. Content and context types a VLM requires

This is the full input/output surface. FFai's `VlmEngine` trait models a
**fraction** of it — noted per row, because each gap is either a trait change
(breaking) or a deliberate scope cut.

### 2.1 Visual inputs

| type | what it means | FFai today |
|---|---|---|
| **Single image** | one `ImageBuffer` | ✅ `describe_image` |
| **Native / arbitrary aspect ratio** | modern VLMs do *not* square-crop; they preserve aspect and pad to patch multiples | ⚠ preprocessing must be written; Carmenta's `svtr_input` is the closest precedent |
| **Dynamic resolution / tiling** ("AnyRes") | a high-res page is split into tiles + a global thumbnail, each encoded, tokens concatenated. **This is what makes small VLMs readable on documents** | ❌ not built |
| **Multi-image** | several images in one prompt ("compare these") | ❌ trait takes one |
| **Interleaved image-text** | `text <img> text <img> text` — the native format of modern VLM chat | ❌ trait has no representation |
| **Video as frames** | sampled frames + timestamps | ✅ `describe_video(&[VideoFrame])`, and `VideoFrame` already carries `timestamp` |
| **Video with temporal encoding** | the model must know frames are *ordered in time*, not a bag of images (M-RoPE's time axis) | ❌ not modelled |
| **Region / grounding input** | "what is in this box" — a box or point as part of the prompt | ❌ |

### 2.2 Textual / control context

| type | what it means | FFai today |
|---|---|---|
| **User prompt** | the instruction | ✅ `VlmOptions.prompt` |
| **System prompt** | role/behaviour framing | ❌ |
| **Chat template** | every instruct model has a *specific* turn format (`<\|im_start\|>`…); getting it wrong silently degrades output with no error | ❌ — **the highest-risk silent failure in the whole build** |
| **Special vision tokens** | `<image>`, `<\|vision_start\|>`… inserted at the exact position image tokens occupy | ❌ |
| **Conversation history** | multi-turn | ❌ |
| **Generation config** | temperature, top-p, top-k, repetition penalty, seed | ⚠ only `max_new_tokens` |
| **Stop conditions** | EOS token, stop strings, max length | ❌ |

### 2.3 Outputs

| type | what it means | FFai today |
|---|---|---|
| **Free text** | a caption or answer | ✅ `String` |
| **Timed segments** | video → chaptered track | ✅ `Vec<TimedSegment<String>>` |
| **Structured / JSON** | schema-constrained extraction — the highest-value output for a *tooling* product | ❌ needs constrained decoding |
| **Grounded output** | text + boxes ("the dog `[x,y,w,h]`") | ❌ |
| **Token-level confidence** | logprobs — Carmenta uses recognition confidence for its verifier; the analog is valuable | ❌ |
| **Streaming** | tokens as produced | ❌ |

> **Gate 2 — decide the trait surface ONCE, before implementing.**
> `VlmOptions` currently has two fields. Every gap above is either added now
> (cheap, pre-1.0) or becomes a breaking change later. Recommendation: extend
> `VlmOptions` with system prompt, sampling config and stop conditions; add a
> multi-image/interleaved input type; leave grounding and streaming out of v1
> **explicitly and in writing**, so their absence is a decision and not an
> oversight.

---

## 3. The encode / decode function map

This is the complete call graph of a VLM forward pass, annotated with **what
FFai already has** — which is more than it looks.

### 3.1 ENCODE — image → tokens

```
ImageBuffer
  │
  ├─ 3.1.1  PREPROCESS
  │         resize (aspect-preserving) · pad to patch multiple · normalize
  │         (mean/std) · RGB↔BGR · planar layout · dtype cast
  │         ✅ PRECEDENT: carmenta::svtr::svtr_input, carmenta::image
  │
  ├─ 3.1.2  TILE (dynamic resolution)
  │         choose tile grid from aspect ratio · crop tiles · build global
  │         thumbnail · record tile positions
  │         ❌ NEW — and load-bearing for document/OCR scores
  │
  ├─ 3.1.3  PATCH EMBED
  │         conv2d(k=patch, s=patch) → flatten → [n_patches, d]
  │         ✅ PRECEDENT: carmenta::svtr conv + reshape
  │
  ├─ 3.1.4  VISION POSITION ENCODING
  │         learned abs · OR interpolated 2-D · OR 2-D RoPE
  │         ⚠ PARTIAL — 1-D only in existing code
  │
  ├─ 3.1.5  VISION TRANSFORMER (×N)
  │         LayerNorm → MHSA(q,k,v,proj) → residual
  │         LayerNorm → MLP(fc1, act, fc2) → residual
  │         ✅ PRECEDENT: carmenta::svtr::{attention, enc_block, ln, linear}
  │            — SVTR is a real ViT-shaped encoder already running on candle
  │
  └─ 3.1.6  CONNECTOR / PROJECTOR → LLM embedding space
            MLP (easiest) · pixel-shuffle/token-merge (cuts token count 4×)
            · Q-Former / perceiver resampler (hardest)
            ❌ NEW — but the MLP variant is ~50 lines
```

```
Text prompt
  │
  ├─ 3.2.1  CHAT TEMPLATE  — exact turn markers for the chosen model  ❌ NEW
  ├─ 3.2.2  TOKENIZE       — BPE/SentencePiece
  │                          ✅ HAVE: `tokenizers` crate, used by Mercury ASR
  ├─ 3.2.3  SPECIAL TOKENS — <image> placeholders at image positions  ❌ NEW
  └─ 3.2.4  EMBED          — token ids → vectors                      ✅ candle
```

```
  3.3  SEQUENCE ASSEMBLY
       splice image-token blocks into the text embedding sequence at the
       placeholder positions · build attention mask · assign positions
       (M-RoPE: separate time/height/width axes for video)
       ❌ NEW — this is the actual "multimodal" step and the one most likely
       to be silently wrong (wrong offset = plausible but degraded output)
```

### 3.2 DECODE — tokens → text

```
  3.4.1  PREFILL           full forward over the assembled sequence,
                           populate KV cache
                           ✅ PRECEDENT: Mercury Whisper decoder holds and
                              mutates a KV cache across steps
  3.4.2  DECODE STEP       one token at a time, cache-appended
                           ✅ PRECEDENT: same
  3.4.3  LOGIT PROCESSING  temperature · top-k · top-p · min-p ·
                           repetition/frequency penalty                ⚠ EXTEND
  3.4.4  SAMPLE            greedy or stochastic (seeded — determinism is a
                           house requirement; Mercury advertises byte-stable
                           output and Carmenta gates on byte-identity)  ⚠ EXTEND
  3.4.5  CONSTRAINED DECODE  grammar / JSON-schema masking — what makes a
                           small model emit valid structured output     ❌ NEW
  3.4.6  STOP              EOS · stop strings · max_new_tokens          ❌ NEW
  3.4.7  DETOKENIZE        ids → text, incremental for streaming
                           ✅ `tokenizers`
  3.4.8  ASSEMBLE          video: per-window captions → TimedSegment    ✅ types exist
```

### 3.3 The reuse ledger — what this build actually costs

| already in-tree | where |
|---|---|
| candle tensor spine, safetensors loading | workspace-wide |
| ViT-shaped encoder blocks (attention/LN/MLP/conv) | `carmenta::svtr` |
| image preprocessing + crop/resize/normalize | `carmenta::image`, `svtr_input` |
| autoregressive decoder with KV cache | `mercury::asr::whisper_candle` |
| BPE tokenizer + detokenize | `tokenizers`, used by Mercury |
| ONNX graph import (arch JSON + safetensors) | `carmenta::onnx_graph` |
| engine registry, `VlmEngine` trait, `VideoFrame`, `TimedSegment` | `ffai-core` |
| bench harness with a `"vlm"` task slot | `ffai-bench` |

| genuinely new | rough weight |
|---|---|
| dynamic-resolution tiling | medium — but decides document scores |
| connector/projector (MLP form) | small |
| chat template + special-token splicing | small code, **high silent-failure risk** |
| M-RoPE / 3-D positions for video | medium |
| sampling stack + stop conditions | small |
| constrained (JSON) decoding | medium — defer to v2 unless the product needs it |
| sequence assembly | small code, **highest silent-failure risk** |

**The honest read: Argus is a connector-and-plumbing build, not a
from-scratch-model build** — provided the model choice (Gate 1.2) picks an
architecture whose two halves we already have precedent for.

> **The alternative that must be priced, not assumed away:** the house
> doctrine (`rusty-coding-requirements`) names **mistral.rs** as the LLM
> serving engine on top of candle, through a `ChatEngine` seam, and
> `ffai-argus/Cargo.toml` already reserves a `mistralrs-backend` feature. If
> mistral.rs supports the chosen VLM, most of §3.2–§3.4 is *already written by
> someone else* and Argus becomes: preprocess → call the engine → assemble.
> **Gate 3: check mistral.rs's VLM support for the Gate-1.2 shortlist BEFORE
> writing any decoder code.** Porting a decoder we could have called is the
> most expensive possible mistake in this plan.

---

## 4. Sequencing, with kill gates

| # | step | gate that must pass before the next step |
|---|---|---|
| **0** | stand up VLMEvalKit; score a published reference model | its published row reproduces. **If not, stop — the scoreboard is broken** |
| **1** | port-cost triage of the Tier-3 shortlist (§1.2) + mistral.rs support check (Gate 3) | one model chosen, with its port cost written down and its connector type named |
| **2** | trait surface decision (§2 Gate 2) | `VlmOptions`/input types settled; v1 exclusions written down explicitly |
| **3** | **vision tower only**, oracle-matched against the reference runtime on fixed input | tensor match to tolerance, the way DocLayout/SLANet/FormulaNet were (§47). **A mismatched tower cannot be debugged later through generated text** |
| **4** | connector + sequence assembly + chat template | a known image+prompt reproduces the reference implementation's output tokens |
| **5** | decode loop (or mistral.rs call) | greedy decode matches reference greedy decode |
| **6** | `describe_image` end-to-end; score on the core suite | a real row on real benchmarks, published with its config |
| **7** | `describe_video`: frame sampling → windowed captions → timed track | Video-MME/MVBench row; frame-sampling policy stated |
| **B1** | *(mission B probe, independent)* take Carmenta's worst substitution pages, ask the chosen VLM to repair, score on OmniDocBench | ≥ the pipeline levers' prize, or mission B is closed and Argus stays a product feature |

---

## 5. What this build must NOT repeat

- **No home-grown scorer.** The single most expensive lesson of the Carmenta
  campaign. Answer extraction is part of the metric; write none of it.
- **No claim without the reference under the same harness.** Published rows are
  claims; a reproduced row is a measurement.
- **Oracle-match the tower before generating text.** Every silent defect the
  §45–§47 model ports had was found by tensor comparison, not by looking at
  output. Generated text hides numerical error behind plausibility.
- **Pin the decode config, ours and theirs.** Unpinned sampling compares
  strategies, not implementations (`references.toml` already says this for ASR).
- **State the C/C++ position honestly.** candle's CUDA path and the
  `tokenizers`/`onig_sys` build-time C dependency already exist in this
  workspace; the README rule is that we never claim "no C in the tree" for a
  candle build. Whatever Argus adds inherits that disclosure.
- **Do not let mission B justify the build without the B1 probe.** A VLM that
  helps documents is a hypothesis; the ledger says 44 % of Carmenta's remaining
  loss is cheaper pipeline work either way.
