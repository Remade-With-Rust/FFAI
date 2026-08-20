# Mercury TTS Mission Plan

**Component:** Mercury — FFai's voice component (`ffai-mercury`)
**Task:** TTS (text → speech)
**Status:** Phase 0 stubs registered · this plan takes TTS from stub to `stable`
**Mirror target:** [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) — every
capability it ships, in pure Rust, measured against it at every milestone
**Prime directive:** pure Rust end-to-end, measured against the non-Rust world
standard by `ffai bench` at every milestone. No claim without a ledger line.

This mission supersedes §4.2 of the
[finished Mercury plan](finished/mercury-mission-plan.md) (the kokoro-candle /
any-tts engine ladder). Piper is the better first target for the same reason
Whisper was: a small, CPU-first, massively deployed model with an installable
reference implementation to answer to. Kokoro and the any-tts tier remain on
the ROADMAP watchlist as later engines behind the same trait.

---

## 1. Mission

Complete Mercury as a voice component: ASR shipped and gated; this mission
ships its counterpart, and with it FFai's whole audio/speech surface.

1. **TTS that is a fully functional tool on day one:** text in, natural speech
   out, voice-selectable, streaming-capable — `ffai tts "hello" -o hello.wav`
   works the day the engine registers as `experimental`.
2. **A capability mirror of piper1-gpl**, not a port of it: same voices, same
   controls, same output quality — from our own phonemizer, our own VITS
   inference on candle, and zero GPL or C/C++ in the shipped crates.
3. Every milestone exits through the analyzer: `ffai bench tts` compares our
   audio to piper1-gpl's on pinned text corpora, scored by the same frozen
   judge, and the result — win or loss — is appended to `bench/ledger.jsonl`.

**Success =** round-trip WER within the 5 % relative band of piper1-gpl's own
audio through the same frozen judge, at **≥ 1.05× its warm synthesis
throughput** (the stated mission goal: a measured 5 % speed win over the
M-T0 baseline's 26.8× — i.e. ≥ 28.2× warm on the pinned corpus), in safe
Rust — with determinism piper itself does not offer (§3.3), and every claim
traceable to a ledger line reproducible from the public repo.

---

## 2. Design rule: independent functions, composed pipelines

Same law as ASR (finished plan §2): a **toolbox of independently callable
functions**, each with its own contract and oracle. The CLI composes them; so
can any embedder. Nothing is welded together.

```
ffai-mercury
└── tts/
    ├── normalize.rs   text normalization: numbers, dates, abbreviations → words
    ├── phonemize.rs   G2P: sentence → espeak-compatible IPA phoneme string (en-US first)
    ├── phoneme_ids.rs voice's phoneme_id_map: IPA string → interleaved id tensor
    ├── vits.rs        VITS inference on candle: ids → waveform
    │                    text encoder · duration predictor · flow · HiFi-GAN generator
    ├── chunk.rs       sentence segmentation, prosody-safe splits, concatenation, silence gaps
    ├── voice.rs       voice pack loading: .onnx weights + .onnx.json config via ffai-models
    └── piper_candle.rs  PiperCandle: composes the above → AudioBuffer
```

Contracts that make the functions independent:

- Every stage consumes/produces `ffai-core` types (`AudioBuffer`) or plain
  candle tensors / plain strings — no stage knows its neighbors.
- Each stage has **its own oracle** (§6): phonemizer vs espeak-ng's output on a
  pinned sentence set; VITS vs piper's own synthesis from *identical phoneme
  ids* at zero noise; round-trip WER only after both stage oracles pass, so a
  quality regression is attributable to one box in minutes.
- `normalize` and `phonemize` are engine-agnostic infrastructure: a future
  kokoro-candle or any-tts engine reuses them unchanged.

## 3. What we are mirroring: piper1-gpl anatomy

### 3.1 The reference

piper1-gpl (Open Home Foundation; the maintained successor of rhasspy/piper)
is the world's default fast-local-TTS: VITS models exported to ONNX, run on
onnxruntime, phonemized by an **embedded espeak-ng** — which is why the
project is GPL-3.0. Deployed in Home Assistant, NVDA, and LocalAI; installable
as `pip install piper-tts`; ships a CLI, an HTTP server, and a C API.

| piper1-gpl capability | Mercury mirror |
|---|---|
| espeak-ng phonemization | our own pure-Rust G2P emitting espeak-compatible IPA (§3.2) |
| VITS via onnxruntime | VITS inference on candle, weights loaded from the same voice files |
| voices from `rhasspy/piper-voices` (HF, ungated) | same voices, declared as `ffai-models` manifests, licenses surfaced |
| quality tiers x_low / low / medium / high (16–22.05 kHz) | same tiers, same files |
| `--length-scale`, `--noise-scale`, `--noise-w` | `TtsOptions`: `speed` (= 1/length_scale), `noise_scale`, `noise_w` |
| multi-speaker voices (`--speaker N`) | `TtsOptions::speaker` |
| sentence-by-sentence synthesis, `--sentence-silence` | `chunk.rs` + `sentence_silence_s` |
| raw streaming output to stdout | streaming synthesis, time-to-first-audio measured |
| HTTP server | **not mirrored** — out of scope for Mercury; FFai is a library + CLI |

### 3.2 The licensing wall, and how it shapes the design

This is the pyannote story again (README: "licences shaped the design").
espeak-ng is GPL-3.0 C. Linking it — even behind a feature flag — makes the
binary GPL and breaks both FFai principles at once (pure Rust, MIT/Apache).
piper1-gpl chose to embed it and accept GPL; **we cannot and do not.**

So the phonemizer is ours: `phonemize.rs`, a dictionary + letter-to-sound
G2P for en-US emitting the espeak IPA phoneme inventory the voices were
trained on. espeak-ng participates only as an **out-of-process test oracle** —
a subprocess in the bench harness dumping reference phonemes for pinned
sentences, exactly as openai-whisper's Python dumps the mel fixture. Nothing
GPL is linked, vendored, or shipped.

This is also the mission's biggest risk (§8): the voices were trained on
espeak's *exact* phoneme sequences, so every phonemization divergence is a
potential pronunciation error. M-T1 exists to measure that gap before any
synthesis code is written, and the phonemizer gets its own gate that isolates
it from synthesis quality (§6.2).

Scope consequence, recorded now as the ASR inventory records its multilingual
gap: piper is 40+ languages *because* espeak is; our G2P starts at **en-US
only**. Additional languages are additional G2P work, one gate each. This is
the honest cost of the license line, stated up front.

### 3.3 Determinism — one place we exceed the mirror

Piper's exported ONNX graphs sample noise *inside the graph*, so piper's own
output is not reproducible run-to-run. Ours is: all sampling flows through a
fixed-seed xorshift (the temperature-fallback precedent), so
`ffai tts --seed 42` is **byte-stable**, and `noise_scale = noise_w = 0` is a
fully deterministic mode used by the acoustic oracle (§6.3). Testing, caching,
and byte-identical A/B gating all hang off this.

## 4. TTS specification

### 4.1 Fully functional means

```
ffai tts "Hello from FFai." -o hello.wav                        # just works, default voice
ffai tts -t script.txt -o narration.wav --voice en_US-lessac-medium --speed 1.2
ffai tts "..." -o out.wav --noise-scale 0.667 --noise-w 0.8 --seed 42
ffai tts --list-voices                                          # tiers + licenses, like `ffai engines`
```

- Long-form input: sentence segmentation, prosody-safe chunking, seamless
  concatenation, configurable inter-sentence silence.
- Voice selection from `ffai models`-managed packs; every manifest carries the
  voice's *own* license from its MODEL_CARD — the piper-voices repo warns some
  voices are restricted (see §8), so surfacing is mandatory, not cosmetic.
- Streaming synthesis: audio starts before the full text is processed;
  time-to-first-audio is a measured, ledger-recorded number.
- Deterministic mode (§3.3) for testing and caching.

`TtsOptions` grows from `{voice, speed}` to carry `noise_scale`, `noise_w`,
`speaker`, `seed`, `sentence_silence_s` — the same trait-growth path
`AsrOptions` took. The `TtsEngine` trait itself does not change.

### 4.2 The engine: `piper-candle`

VITS has four stages, and all four run on candle: transformer text encoder →
stochastic duration predictor (flows) → residual-coupling flow decoder →
HiFi-GAN generator. Weights come from the voice's own `.onnx` — extracted
from the ONNX initializers into safetensors by a converter in
`corpora/refs/` (the `dump_whisper_mel.py` precedent: reference tooling may
be Python; shipped code may not). The `.onnx.json` config supplies
`phoneme_id_map`, sample rate, inference defaults, and speaker count, and is
consumed as-is.

Where the boundary sits, stated the way §6.2 of the finished plan stated it:
the phonemizer, id mapping, chunking, and decode-time controls are ours
outright and are where quality and speed live; candle primitives (conv,
attention) are shared infrastructure until the profiler says otherwise. The
ASR campaign's ending is this mission's starting posture — expect to own hot
stages eventually, and let measurement pick which.

Registered as `piper-candle`, status `experimental` until §6's gates pass.
Bring-up voice: `en_US-lessac-medium` (the de facto piper default, 22.05 kHz),
with one `x_low` voice added at M-T4 to prove the tier sweep.

### 4.3 World-standard reference (non-Rust)

One bar, by design — the user-facing comparison is Piper1, full stop.

| Reference | Stack | Why it's the bar |
|---|---|---|
| piper1-gpl | Python CLI · onnxruntime (C++) · embedded espeak-ng (C) | the fast-local-TTS standard; same voices, same model family — so every gap is *implementation*, not model |

Declared in `corpora/references.toml` like every ASR reference; version
recorded per ledger line. openai-whisper-class "absolute quality" confusion
cannot recur here: same weights on both sides means matched-model comparison
is the *only* comparison, which is the cleanest bench Mercury has ever had.

### 4.4 Measuring TTS quality without ears

Unchanged from finished-plan §4.4, now made concrete:

- **Round-trip intelligibility (primary, automated):** synthesize the pinned
  holdout texts → transcribe with a **frozen third-party judge** — whisper.cpp
  v1.9.1, the binary already in `.whispercpp/`, pinned by version and argv,
  **never whisper-candle** (no self-grading, benchmarking.md principle 6) —
  → WER/CER between input text and round-trip transcript, both sides through
  the same `EnglishTextNormalizer`. Ours vs piper's audio, same judge, same
  texts.
- **Speed:** ×-realtime synthesis (audio seconds per wall second), warm and
  end-to-end, best-of-N; time-to-first-audio for streaming; load time
  recorded separately. All benchmarking.md timing rules apply verbatim.
- **Footprint:** steady resident memory, sampled identically on both sides;
  peak recorded beside it, judged on steady (the README's ASR precedent).
- **Naturalness (secondary):** UTMOS-class MOS predictor when practical;
  human spot-listens logged in ledger notes — never claimed as measurements.

Known bluntness, recorded now: a competent judge may transcribe both sides
near-perfectly, saturating round-trip WER into a tie. A tie *is* the pass
criterion (we match the reference), but it cannot distinguish "as good" from
"better", so no "better than Piper" quality claim is available from this
instrument — only parity claims plus speed/footprint/determinism wins. If a
naturalness claim is ever wanted, that is a new instrument, gated then.

## 5. Analyzer integration (`ffai-bench`)

`ffai bench tts` is the M-T0 deliverable, built on the same spine (gates,
best-of-N, hashed corpora, ledger) with a TTS-shaped adapter contract:

```
ffai bench tts --corpus corpora/harvard-sentences-v1.toml
ffai bench tts --corpus corpora/harvard-sentences-v1.toml --baseline-only
```

- **Corpus manifests pin text, not audio:** each line's SHA-256, train/holdout
  split, same abort-on-drift rule. First corpus: Harvard sentences (public
  domain, phonetically balanced) — enough sentences for a real holdout, none
  of the 11-clip smoke-corpus trap (§7).
- **Reference adapter contract** (documented in benchmarking.md when built):
  batch invocation, one process per corpus; reads `{filelist}` of texts,
  writes WAVs to an output dir, emits JSONL — `{"load_secs": …}` then per line
  `{"text_id": …, "wav": …, "synth_secs": …}` with `synth_secs` excluding
  model load. Warm/e2e both recorded, the M0 lesson pre-paid.
- **Pinning:** voice file hash, `length/noise/noise_w` values, speaker id,
  thread count, judge version + argv — anything that changes the work, in the
  ledger line (benchmarking.md: "pin it, or you're measuring the wrong
  thing"). Our engine runs the voice config's own defaults, exactly as piper
  does, so neither side gets a tuned advantage.
- **Four gates, skipped-is-not-passed:** correctness = stage oracles + seeded
  byte-stability; quality = round-trip WER band vs piper's audio; speed = ×RT
  warm; footprint = steady RSS. `verdict: claimable` requires all four.

## 6. Milestones and exit gates

Every milestone exits through all four gates on the holdout split — a skipped
gate blocks exit. Results are appended below their milestone as they land,
including prunes, exactly as the finished plan accumulated §6.1–§6.25.

| # | Deliverable | Exit gate (ledger-recorded) |
|---|---|---|
| **M-T0** ✅ | Baselines: text corpus pinned; `ffai bench tts` vertical live; piper1-gpl installed as a reference; espeak-ng phoneme fixtures dumped | piper's round-trip WER / ×RT / TTFA / RSS on the board via `--baseline-only`; judge frozen and pinned — **see §6.1** |
| **M-T1** ✅ | Phonemizer: `normalize.rs` + `phonemize.rs` (en-US) + `lexicon.rs` | phoneme oracle vs espeak-ng fixtures on holdout sentences; **substitution gate**: our phonemes fed through *piper's own runtime* score round-trip WER within the 5 % band of espeak's phonemes through the same runtime — **PASS, see §6.2** |
| **M-T2** ✅ | VITS on candle: weights from voice `.onnx`, all four stages, `ffai tts` speaks end-to-end | zero-noise acoustic oracle: ours vs piper from **identical phoneme ids**, waveform/mel within tolerance; end-to-end round-trip WER within band on holdout — **both PASS, see §6.3** |
| **M-T3** ◐ | Full core: determinism, all knobs, long-form chunking, multi-speaker, WAV egress via ffai-media | all four gates vs piper1-gpl; seeded runs byte-stable; speed gate target ×RT ≥ piper warm — **speed campaign §6.4: 3.2× → 16.4×, gate still open** |
| **M-T4** | Streaming + breadth: streaming synthesis, tier sweep (x_low + medium), `--list-voices` with licenses | time-to-first-audio < 500 ms on reference hardware; every added voice re-runs the gates; licenses surfaced from MODEL_CARDs |
| **M-T5** | TTS `stable`: docs, README status section written from the ledger, `mercury` lib example | every public claim maps to a ledger line id; README states the caveats the way the ASR section does |

Sequencing notes. M-T1 and the M-T2 weight-extraction work share no code and
can run in parallel — but the **substitution gate blocks M-T2 exit**: until
our phonemes are proven through piper's runtime, an end-to-end WER miss
cannot be attributed to one box. And M-T0 comes first without exception; the
ASR campaign never once regretted having baselines on the board before
bring-up, and twice caught the harness lying because references were watched
from day one.

### 6.1 M-T0 result — the baseline is on the board

**Corpus.** `harvard-sentences` v1: 200 sentences (lists 1–20 of IEEE
297-1969 Appendix C), 1,561 words, 134 holdout / 66 train on the same
every-3rd-is-train rule as the LibriSpeech corpora. Texts are **committed**
(public domain, ~25 KB — a TTS corpus whose inputs can vanish with a webpage
is not pinned in any useful sense) and every file is SHA-256-pinned in
[`corpora/harvard-sentences-v1.toml`](../corpora/harvard-sentences-v1.toml);
regenerate deterministically with `prepare_harvard`. Manifest fingerprint
`47cfa0604493`.

**Baseline** (ledger `bench-tts-1785331739`, best-of-3, Windows x86_64 /
Intel Raptor Lake, CPU only, voice defaults noise 0.667 / length 1.0 /
noise_w 0.8):

| Implementation | round-trip WER % | CER % | ×RT warm | ×RT e2e | TTFA p50/p95 | load s | peak MiB | clips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| piper1-gpl 1.6.0, en_US-lessac-medium | 6.45 | 2.10 | 26.8 | 21.6 | 80 / 92 ms | 2.17 | 239 | 134/134 |

**These are the numbers piper-candle must answer to**: round-trip WER within
the 5 % relative band of 6.45 % through the same judge, at ≥ 26.8× warm
synthesis, under 239 MiB peak — plus the two capabilities the baseline
cannot show, seeded determinism and streaming TTFA.

Read the row with its instruments in view:

- **6.45 % is judge floor + piper error, inseparable.** whisper.cpp
  base.en/beam5 scores 5.96 % transcribing *real human speech* on
  LibriSpeech test-clean, so most of 6.45 % is plausibly the judge, not
  piper. This is why §4.4 limits the instrument to parity claims — the
  number is a bar, not a measurement of piper alone.
- **The judge is base.en/beam5, deliberately stronger than the plan's
  wording.** §6.1's spec said "the same binary and flags the ASR bench
  already carries"; the ASR bench's greedy tiny.en carries a ~7.5 % floor,
  which would have drowned the comparison. Same pinned binary (whisper.cpp
  v1.9.1), better model, beam 5: a lower shared floor leaves more resolution
  between implementations. Deviation from spec, recorded as such.
- **TTFA p50 equals per-sentence synth time (80 ms)** because piper emits
  one chunk per sentence and the corpus is single-sentence — for this corpus
  TTFA ≈ full synthesis, and the M-T4 streaming gate will need long-form
  texts to mean anything. Known and accepted at M-T0.
- **piper's audio is not reproducible run to run** (noise sampled in-graph,
  no seed exposed), so the WER above belongs to the audio of the kept
  (fastest) run. Cross-seed variance is unmeasured as of this line; treat
  sub-point deltas as noise (§8 risk, now observed in the instrument design).

**One adapter defect found and fixed before it could lie:** onnxruntime's
first inference carries ~3.3 s of session/kernel setup against ~0.07 s
steady-state per sentence. The first version of `piper_ref.py` let that land
in the first clip's `synth_secs`, quietly taxing warm throughput by ~15 % on
a 134-clip corpus; it is now an untimed warm-up folded into `load_secs` —
the same class of fix M0/M1 made on both sides of the ASR harness, caught
here by reading the adapter's own per-clip output rather than the aggregate.

**Also landed with this milestone, ahead of schedule:**

- **Phoneme fixtures for M-T1 are already dumped and pinned:**
  [`corpora/fixtures/harvard-espeak-phonemes-v1.jsonl`](../corpora/fixtures/harvard-espeak-phonemes-v1.jsonl)
  — all 200 sentences through piper's own embedded espeak-ng (the exact
  phonemizer the voices were trained on), carrying both the IPA phoneme
  sequences and the voice's literal model-input id sequences (BOS/EOS/pad
  interleaving included). M-T1's oracle and substitution gate both read from
  this file.
- **The harness-side resampler** ([`crates/ffai-bench/src/resample.rs`](../crates/ffai-bench/src/resample.rs)):
  every implementation's audio reaches the judge as 16 kHz mono through the
  same windowed-sinc code path, tested on content (tone frequency/amplitude
  survival, alias rejection) rather than shape — the audio_encoder.rs lesson
  applied on day one.
- The TTS vertical shares the ASR/OCR gate machinery (`fill_gates`,
  `QualityMetric::Wer`, the `decode`-key matched-reference logic), so the
  M-T2 quality gate needs no new harness code — only an engine whose config
  declares its voice.

### 6.1.1 M-T0 specification (as planned)

- **Corpus:** `harvard-sentences-v1.toml` — public-domain Harvard sentences,
  hash-pinned per line, deterministic regeneration script, train/holdout
  split sized so a handful of words cannot move WER by half a point (the
  §6.7 reversal is the design input here: 134 clips, not 11).
- **References:** piper1-gpl in `.venv-bench` (`pip install piper-tts`),
  adapter `corpora/refs/piper_ref.py` under the §5 contract; version, voice
  hash, and argv per ledger line.
- **Judge:** whisper.cpp v1.9.1 pinned — the *same binary and flags* the ASR
  bench already carries, minus nothing added: the `-nt` lesson (§6.17 of the
  finished plan) says the judge's argv is part of the measurement and goes in
  the ledger.
- **Oracle fixtures:** `corpora/refs/dump_espeak_phonemes.py` runs espeak-ng
  as a subprocess over the corpus and pins its IPA output; regenerable by
  anyone, nothing GPL in the repo.
- Exit: one `--baseline-only` ledger line with piper's numbers on the board.

### 6.2 M-T1 result — the pure-Rust phonemizer clears the substitution gate

**The gate** (`examples/tts_substitution_gate.rs`, 134 holdout sentences,
both arms rendered by piper's own runtime at zero noise — deterministic, so
no noise sample can flatter either arm — judged by the frozen
whisper.cpp-base judge):

| Arm | round-trip WER % | CER % |
|---|---:|---:|
| piper ← espeak-ng phonemes | 4.51 | 1.71 |
| piper ← **Mercury phonemes** | **4.69** | **1.75** |

**PASS**: 4.69 % against the band limit of 4.51 × 1.05 = 4.73 %. Stated at
its true width: the margin is 0.18 pp of WER ≈ **two net word flips across
1,050 words**, and both runs are deterministic, so this is a stable
measurement rather than a coin flip — but it is a pass at the rim, not the
middle. Only three sentences separate the arms, and one of them favors ours.

**What the phonemizer is.** `tts/lexicon.rs` (CMUdict, BSD, fetched via
`models/cmudict.toml` — never vendored) + `tts/phonemize.rs`: ~40
espeak-convention mapping rules (stress before the stressed vowel, the CLOTH
set, flapping, glottal syllabic-n, linking ɹ/ʲ, spelling-conditioned
reductions), a ~100-entry closed-class function-word table, 13 glued
collocations (`on the` → ɔnðə, `out of` → ˌaʊɾəv), and ~35 lexical
exceptions. espeak-ng was consulted only as an out-of-process oracle over
pinned corpora; no GPL source or data ships.

**Phoneme oracle, with the tuning history stated honestly:**

| Set | Sentences | Exact match | PER |
|---|---:|---:|---:|
| train (corpus lists 1–20) | 66 | 98.5 % | 0.04 % |
| tune (lists 21–40, disjoint from the corpus) | 200 | 87.0 % | 0.37 % |
| **holdout** | **134** | **83.6 %** | **0.49 %** |

The first holdout read (57.5 % / 1.79 %) exposed classes the train split
never contained — the overfitting the mission plan predicted. Rather than
tune on holdout (the §6.23 sin), a **tune set** was generated from Harvard
lists 21–40 through the same espeak dump: free oracle data, disjoint from
the corpus, unlimited. Holdout was read three times in total (initial ·
frozen-mapping · after two bug fixes), each read declared; every rule change
between reads was validated on train + tune first. The two bugs the failing
gate isolated were both general: ER0-elision deleting `around`'s first
syllable, and flapping over-suppressed after r-colored vowels (`dirty`).

**Two observations banked for later milestones:**

- **Zero-noise audio judges far better than piper's defaults** — the espeak
  arm scores 4.51 % here against 6.45 % at default `noise_scale` 0.667
  (M-T0). Piper's shipped knobs trade ~2 pp of judged intelligibility for
  naturalness. When M-T3 picks `piper-candle`'s defaults, that trade is a
  measured decision, not an inheritance.
- Phonemization cost is negligible (all 134 sentences in milliseconds,
  single-threaded) — the ≥ 1.05× speed goal will be won or lost in M-T2's
  synthesis loop, not here.

**Residual gap, named**: the ~16 % of holdout sentences that still diverge
are dominated by espeak's lexical stress choices on common verbs and its
unstressed-vowel lexicon (ə/ɪ/ᵻ) — one character each, mostly inaudible to
the judge. `phoneme_ids.rs` (IPA → the voice's id map) moved to M-T2 where
its consumer lives; piper's `phonemes_to_ids` filled that seat in this gate.

### 6.2.1 M-T1 specification (as planned) — the phonemizer gate that isolates the risk

The substitution gate is the load-bearing instrument of this mission. Feeding
**our** phoneme ids into **piper's own onnxruntime** and scoring round-trip
WER against espeak-phonemized piper measures exactly one thing: how much
speech quality our G2P costs, with synthesis held constant. It converts "the
phonemizer seems fine" into a number *before* any candle code exists, and it
ranks every G2P defect by what it costs where it matters — some phoneme
divergences are inaudible, some are not, and raw phoneme-error-rate cannot
tell them apart.

Build order inside M-T1: espeak-oracle fixtures first, dictionary + LTS core
second, normalize.rs third (numbers/dates/abbreviations — scored by the same
gate, since espeak normalizes internally and our fixtures capture its output).

### 6.3 M-T2 result — VITS on candle, oracle-exact, and the speed bill

**Mercury synthesizes.** `ffai tts "..." -o out.wav` runs the full VITS
stack in Rust on candle — text encoder (relative-position attention),
stochastic duration predictor (DDS convs + rational-quadratic spline flows,
reverse mode), residual coupling flow, and piper's compact HiFi-GAN — on
weights extracted from the voice's own `.onnx`
(`corpora/refs/dump_piper_weights.py` → named safetensors + a geometry JSON;
`models/piper-vits-lessac-medium.toml`).

**The acoustic oracle** (`examples/vits_oracle.rs`, three fixture sentences,
zero noise, every stage fed the REFERENCE's input so errors cannot hide or
compound):

| Stage | vs onnxruntime | Verdict |
|---|---|---|
| text encoder (m_p / logs_p) | max\|Δ\| ≤ 3.3e-6 | PASS |
| duration predictor (w_ceil) | **integer-EXACT**, all 275 phonemes | PASS |
| flow reverse (dec_in) | max\|Δ\| ≤ 2.4e-5 | PASS |
| HiFi-GAN (audio) | max\|Δ\| ≤ 1.9e-6 | PASS |
| **end-to-end waveform** | **max\|Δ\| ≤ 3.2e-5**, rel-RMS ≤ 1.7e-5 | PASS |

The spline flows — the gnarliest translation — are validated by w_ceil being
integer-exact: any drift in the spline inverse changes a duration somewhere.

**How the archaeology was done, for the next voice conversion:** ONNX export
mangles names but node paths keep the module tree, so every weight is renamed
by the op that consumes it. Three export traps found and handled, each of
which would have been a silent wrong-output bug: the phoneme embedding is
exported under the name `sid`; the converter's first size filter ate the
[2,1] ElementwiseAffine parameters (caught by a loader that errors by NAME,
never defaults); and `exp(-logs)` was constant-folded, so the "missing" logs
parameter is a folded constant. The graph also confirmed two VITS inference
quirks before any Rust existed: `dp.flows.1` never runs, and only the total
length is clamped, not per-phoneme durations.

**The bench** (ledger `bench-tts-1785339497`, 134 holdout, matched voice
defaults, best-of-3; our engine deterministic at seed 0, piper sampling
in-graph):

| | round-trip WER % | CER % | ×RT warm | ×RT e2e | load s | clips |
|---|---:|---:|---:|---:|---:|---:|
| **piper-candle** (pure Rust) | **6.00** | 2.46 | 3.2 | 3.1 | 0.77 | 134/134 |
| piper1-gpl (onnxruntime) | 6.12 | 2.18 | 18.2 | 14.7 | 3.20 | 134/134 |

**Quality gate PASS** — 6.00 % against piper's 6.12 %, first place in the
matched field. Read it as parity, not victory: piper's audio differs every
run (in-graph noise), so its 6.12 % is one sample of a distribution the
M-T0 run drew at 6.45 %. The claim the band supports is "indistinguishable
through the judge", which is the claim M-T2 needed.

**Speed gate FAIL, 5.7× behind (3.2× vs 18.2× warm)** — the honest state of
a correctness-first bring-up, the same posture M1 recorded at 4× behind
before the M2 campaign closed it to 1.2×. Nothing has been profiled or
optimized; candle ops are used scalar-clear (the spline runs an element
loop), and the mission goal of **≥ 1.05× piper (≥ ~19× warm here)** is
M-T3's campaign, to be fought with the codec-optimize discipline: profile
first, then eliminate redundancy, then kernels where the profiler points.
Load is already 4.2× faster than piper's (0.77 s vs 3.20 s).

**Footprint SKIP**: the harness does not yet sample the in-process TTS
engine's memory (the ASR-side sampler was not wired into `run_tts_engine`);
M-T3 carries it. A skipped gate is not a pass, and the verdict correctly
reads "not claimable yet".

### 6.3.1 M-T2 specification (as planned) — the acoustic oracle

At `noise_scale = noise_w = 0` both implementations are deterministic
functions of the phoneme ids. Same ids + same weights ⇒ the waveforms must
match to tolerance (exact op-order differences accumulate; the mel-oracle
precedent of max-abs-Δ over a fixture, tolerance set from measurement, not
hope). This is the byte-identical-gate analogue for TTS: every subsequent
speed change is checked against it, and a synthesis bug can never hide behind
"VITS is stochastic". The noise path is then gated separately by seeded
byte-stability plus round-trip WER at the voice's default noise settings.

### 6.4 M-T3 speed campaign, round one — 3.2× → 16.4× warm

The M-T2 bill was 5.7×. One profiled campaign paid most of it (ledger
`bench-tts-1785343718`; every brick gated on the stage oracles staying green
and the 106-test suite):

| | round-trip WER % | ×RT warm | steady MiB | peak MiB | load s |
|---|---:|---:|---:|---:|---:|
| **piper-candle** | **6.00** | 16.4 | **172** | **213** | **0.34** |
| piper1-gpl | 6.20 | **31.6** | 241 | 241 | 1.78 |

**Three of four gates now PASS.** Quality (6.00 % — the same number on
every bench this campaign, because the engine is seeded; piper's has drawn
6.45/6.12/6.20), correctness 134/134, and **footprint, first pass: 0.71× of
piper's steady memory** with the engine sampler newly wired (the M-T2 SKIP
closed). Speed remains FAIL and the milestone stays open.

**The campaign, brick by brick** (per-stage instrument:
`examples/profile_tts.rs`; decoder sub-profile via `FFAI_PROFILE=1`):

| Brick | Result |
|---|---|
| Profile: decoder 82.7 %; candle conv1d DEGRADES with length (99 → 27 GF/s) as its working set leaves cache; im2col+matmul no better (40 MB gathers) | the campaign's map |
| Cache-blocked direct conv (t-tiles × co-chunks, rayon grid) | **KEPT** — 101–149 GF/s at every shape; first cut was time-tiles only and ran 3× SLOWER than candle at short lengths (2 tasks on 24 cores) — the 2-D task grid fixed it |
| Phase-decomposed ConvTranspose (k/s taps per output, contiguous per phase) | **KEPT** — strided-write first version rewritten; ups now ~12 ms/sentence |
| Flat decoder: whole HiFi-GAN chain in one Vec domain, single-pass activations | **KEPT** — killed per-op tensor↔Vec copies; 4.4× → 8.6× overall |
| AVX2+FMA micro-kernel (4 co × 32 lanes, 16 ymm accumulators) | **KEPT** — conv ~68 → ~44 ms/sentence; autovec had ceilinged at ~226 GF/s |
| Rayon glue (branchless leaky, parallel adds) | **KEPT** — glue 46 → 16 ms |
| Route flow.* through the direct kernels | **PRUNED** — 1.4× SLOWER at the flow's short lengths (L≈190: rayon overhead beats the work); reverted, lesson recorded at the call site |

**On the gap ratio, the standing caution applies verbatim:** piper's own
warm ×RT has read 18.2×, 26.8× and 31.6× across this campaign's ledger lines
— it swings ~1.7× with machine load from the parallel workstream. Our own
throughput (3.2 → 16.4×, same corpus, same harness) is the progress signal;
the ratio (currently 1.9× behind the reference's best recorded run) is the
standing. **The ≥ 1.05× goal is open**, and the next levers are named from
the profile, not guessed: the flow's WN as a flat fused-gate kernel
(~37 → ~18 ms/sentence), text-encoder/duration-predictor de-plumbing
(~45 ms of candle small-op tax), the ups micro-kernel, and the conv tail.
Remaining M-T3 scope besides speed: byte-stability test for seeded runs,
CLI knobs (noise/seed/speaker), long-form chunking, and the M-T4 items
beyond.

### 6.5 M-T3 round two — routing settled by measurement; the functional core lands

**Speed bricks** (ledger `bench-tts-1785345156`):

| Brick | Result |
|---|---|
| Size-adaptive conv path: serial under a 4M-MAC work threshold — the FIX for round one's pruned flow routing, not an exception list | **KEPT** |
| Depthwise kernel for the dp's separable convs | **KEPT** — duration predictor 395 → 254 ms per 20 sentences |
| Universal routing (enc_p/flow through the direct kernels) | **PRUNED** — enc_p regressed 1.35× (serial 1×1s lose to candle's threaded matmul); flow flat. Routing settled per stage: dec.* + dp.* direct, enc_p/flow candle |

Best quiet-machine reading this round: **15.3× warm** (profile instrument;
the bench itself drew 13.9× vs piper's 25.1× on a loaded machine — both
sides depressed together, ratio ~1.8× either way). Speed gate still FAIL,
still open.

**The M-T3 functional core is now complete:**

- **Byte-stability, proven at two layers**: same seed → identical samples
  in-process AND identical WAV file hashes from the CLI; different seed →
  different audio. Piper can offer neither guarantee. (Checked by
  `vits_oracle` as part of ALL STAGES PASS.)
- **Knobs**: `TtsOptions` and `ffai tts` now carry `--speed`,
  `--noise-scale`, `--noise-w`, `--seed`, `--sentence-silence`; noise knobs
  default to the voice's own values so matched benches stay matched.
- **Long-form chunking** (`chunk.rs`): sentence segmentation with an
  abbreviation guard and lowercase-continuation rule, per-sentence
  synthesis, configurable silence gaps. `ffai tts` on multi-sentence text
  just works.
- Multi-speaker moves to M-T4 with the tier sweep: the converted voice is
  single-speaker, so the knob would be untestable here (a gate that cannot
  fail is not a gate).

**Quality note worth keeping**: across four benches piper has drawn
6.45/6.12/6.20/5.77 % from its in-graph noise while Mercury reads 6.00 %
every time. The parity claim holds run over run; the reproducibility is
itself the differentiator.

**Still open for M-T3 exit**: the speed gate alone. Next levers, from the
profile: flow's WN as a flat fused-gate kernel, text-encoder de-plumbing
(now the second-largest stage), the upsampler micro-kernel, conv tail.

### 6.6 M-T3 round three — a wash caught by pairing, a fusion banked, and the ratio signal

Fought under heavy load from the parallel workstream (both arms' absolute
×RT depressed ~20–30 % run to run), which shaped the method: interleaved
paired A/B for verdicts, same-run ratios for standing.

| Brick | Result |
|---|---|
| **Flat fused-gate flow** (`FlatFlow`: whole coupling stack in Vec domain, one-pass tanh×sigmoid gates) | **WASH — not shipped as default.** Oracle-exact, but the interleaved paired A/B read 0.99×, 4/15 rounds: this stage's cost is its convolutions, not op dispatch. Kept behind `FFAI_FLAT_FLOW=1` pending a quiet-machine retest; the §6.19-ASR lesson (a plausible de-plumbing story, refuted by the pair test) repeating on the TTS side |
| **conv-into fusion**: the resblock's `y += conv(leaky(y))` accumulates in the conv's write pass | **KEPT** — strictly one allocation and one full memory pass less per conv (6 per stage); oracle-exact |
| A test-suite flake fixed: both phonemizer tests shared one temp-file path (write/remove race that ordering luck had hidden) | unique per-call paths |

Ledger `bench-tts-1785347261`: 15.0× vs piper's 23.9× on the loaded box —
and the number that matters, the **same-run gap ratio**, has narrowed every
round: **5.7× (M-T2) → 1.93× → 1.81× → 1.59×.** Quality PASS (ours 6.00 %
as always; piper's fifth draw 6.05 %), footprint PASS, correctness 134/134.

**Toward the beat-piper target (~24–32× warm):** the per-sentence budget
says the remaining fight is dec conv (~44 ms, already ~350 GF/s — needs
ORT-class kernels to shrink further), enc/dp/flow op-tax (~55 ms combined —
needs flat rewrites that the flow experiment says must fuse ARITHMETIC, not
just dispatch), and glue (~16 ms). All of it wants a quiet machine; paired
A/B is mandatory equipment from here.

### 6.7 M-T3 round four — encoder fusions land; the flow refuses a third attack; one ledger line disowned

| Brick | Result |
|---|---|
| FFN pads folded into conv1d's own padding parameter (the explicit Pad ops were an ONNX export artifact) | **KEPT** — two fewer allocs+copies per layer, oracle-exact |
| Fused QKV projection: per-layer weights concatenated at load, one conv dispatch instead of three | **KEPT** — arithmetic identical, m_p oracle unchanged at ≤ 3.8e-6 |
| conv-into fusion first clean reading (quiet machine) | decoder 1122 ms per 20 sentences, its best yet |
| Flow v2: convs as fat GEMMs (1×1s as direct matmuls, k5 via L2-resident im2col), biases fused into gate/add passes | **PRUNED as default — 0/15 rounds, 0.83×.** The flow has now refuted THREE attacks (interface routing 1.4× worse, de-plumbing 0.99×, GEMM-shaping 0.83×): candle's conv1d is genuinely strong at [384,192,k5,L≈190], and this stage will move only with a purpose-built short-length kernel. Opt-in retained (`FFAI_FLAT_FLOW=1`), verdicts at the call site |

**One ledger line is explicitly disowned: `bench-tts-1785352384`.** Piper
collapsed to 5.5× warm in that run (load 6.57 s, TTFA p50 394 ms — the
machine was crushed during its segment) and simultaneously drew its
luckiest WER yet (5.41 %). Its speed "PASS" was earned against a crippled
reference and its quality "FAIL" against a lucky draw; neither is a
finding. It stays in the append-only ledger with this paragraph as its
reading instructions — the `-nt` lesson cuts in both directions.

**The fair round-four line** (`bench-tts-1785352631`, piper healthy):

| | WER % | ×RT warm | steady MiB | load s |
|---|---:|---:|---:|---:|
| **piper-candle** | **6.00** | 18.2 | **206** | **0.26** |
| piper1-gpl | 5.85 (7th draw; range 5.41–6.45) | **24.9** | 239 | 1.94 |

Quality PASS, footprint PASS, correctness PASS, speed FAIL — and the
same-run gap ratio has now narrowed **five rounds straight: 5.7× → 1.93× →
1.81× → 1.59× → 1.37×**, with our own warm throughput at 18.2×, a 5.7×
gain over the M-T2 bring-up.

**Harness note for M-T4**: piper's WER spread across seven draws (5.41 to
6.45, ±0.5 around our fixed 6.00) says single-draw quality gating against a
stochastic reference is at the edge of its resolution; the §8 mitigation
(average the reference over N seeds, record the range) should be built
before the next quality claim is published.

### 6.8 M-T3 round five — the ⅓-fold, the multi-seed harness, and where the campaign rests

**Speed** (`bench-tts-1785354825`): warm **19.1×**, the campaign's best —
the cumulative arc is **3.2× → 19.1× (6.0×)**. The round's brick: the
resblock average (`xs/3`) FOLDED into the next conv's weights — leaky relu
is positively homogeneous, so `next(leaky(x/3)) == next_scaled(leaky(x))`
exactly; three full multi-MB passes disappear, oracle unchanged at ≤ 2e-6.
Same-run gap 1.53× against a healthy piper at 29.3× (the reference's own
swing dominates round-to-round ratio motion at this scale; our warm number
is the signal).

**The multi-seed harness landed and earned its keep immediately** (the §6.7
prerequisite): the reference's quality is now the MEAN of all best-of-N
draws with the range in the ledger notes. First run: piper's three draws
spanned **4.06–6.61 %** — a ±1.3 pp spread that six single-draw benches had
been sampling blind. The quality gate consequently read FAIL by 0.05 pp
against a 3-draw mean whose own standard error is ~±0.75 pp: **a statistical
tie, now visible as one.** Standing recommendation, recorded: quality runs
against this reference use `--runs 5+`, and a variance-aware band is M-T4
harness work.

**Honest closure on the two levers left un-pulled:**

- **The flow short-length kernel is not winnable with today's toolbox.**
  Three designs measured and refuted (§6.6, §6.7); candle's conv1d at
  [384,192,k5,L≈190] is the floor until someone writes a genuinely better
  packed short-L kernel. Parked with its evidence, not abandoned silently.
- **Decoder conv weight repacking (quad-major for the AVX2 kernel) is the
  named next brick of the ORT-class campaign** — the kernel's 4 scattered
  weight loads per (ci,j) become one contiguous load. Unbuilt: it is kernel
  surgery that deserves a quiet machine and paired verdicts from the start,
  not the tail of a long session.

**Where M-T3 rests**: correctness PASS, footprint PASS (0.92×), quality a
statistical tie the harness now measures properly, speed FAIL at 19.1× vs
25–31×. The milestone stays open on the speed gate alone, with the ledger
carrying nine TTS lines, every kept brick oracle-gated, every pruned brick
recorded with its refutation.

### 6.9 Function-by-function against piper — every loss named

The §6.14-ASR treatment, TTS edition: ORT's per-node profiler
(`corpora/refs/profile_piper_stages.py`) puts piper's stage budget beside
`profile_tts`'s, same 20 sentences, both warm, quiet machine. Mercury walls
20.3× this session; piper's node-time totals 21.9× (its wall ~25–31×).

| Stage (ms / 20 sentences) | Mercury | piper (ORT) | Verdict |
|---|---:|---:|---|
| decoder | 1019 | 780 | behind 1.31× — but split it: |
| — upsamplers (ConvTranspose) | **~186** | 468 | **we WIN 2.5×** (phase decomposition) |
| — plain convs | ~540 | 241 | **we LOSE 2.2×** |
| flow | 551 | **207** (conv 136) | **we LOSE 2.7× — the biggest loss** |
| text encoder | **390** | 409 | parity, slight win |
| duration predictor | **252** | 522 | **we WIN 2.1×** (scalar spline beats their ScatterND graph) |
| total | 2212 (wall) | 2020 (node) | |

**Three findings that redirect the campaign:**

1. **The flow verdict of §6.8 is OVERTURNED.** Three designs lost to
   candle's conv1d at [384,192,k5,L≈190] — but ORT runs the same convs
   2.7× faster, so candle's floor was never the real floor. The flow
   kernel is winnable; the target is now a measured 136 ms, not a hope.
2. **The decoder's plain convs have the same 2.2× headroom** despite the
   AVX2 kernel — ORT's packed-weight MLAS convs are the standing proof
   that quad-major weight repacking (§6.8's named next brick) pays.
3. **Two stages are already ahead of the reference**: our upsampler
   (2.5×) and duration predictor (2.1×). The wins are real and banked;
   the campaign's remaining ~47 ms/sentence of losses sit in exactly two
   kernels, both plain convs, both with an existence proof of the target.

**The arithmetic of the win**: closing both conv gaps to ORT's numbers is
~32 ms/sentence, taking 110.6 → ~78 ms ≈ **28× warm — past piper's typical
wall** — before any further ideas. That is the next campaign, opening with
weight repacking under paired A/B on a quiet machine.

### 6.10 The three §6.9 verdicts, hammered

**Verdict 2 — packed weights: LANDED, +50 % at every shape.** Quad-major
repacking ([co/4][ci][j][4]) turned the AVX2 kernel's four scattered weight
loads per (ci,j) into one streamed cache line: 149/118/101 → **224/191/153
GF/s** (load-invariant same-process ratios vs candle improved 1.4–1.6×).
One trap found by the flow: per-call packing is free where work dwarfs
weights (decoder) and was HALF the runtime where it doesn't (flow: 1.5 MB
of weights vs 70 M MACs) — packing now happens once at construction.

**Verdict 1 — the flow: four attacks, four refutations, each closer.**
v4 (packed kernel + fused gates) progressed 0.58× → 0.72× (pack-once) →
0.74× (8-lane tail strips) → **0.84×, 0/15** — candle keeps the stage. Two
sub-verdicts worth their ledger ink: the Padé rational tanh (the §6.10-ASR
GELU trick) was tried in the gates and **REVERTED — its 1.8e-4 error
compounds through 16 stacked coupled gates into 7e-2 at dec_in**, a stage
oracle FAIL; and rayon-parallel exact gates were kept. The ORT existence
proof (136 ms) still stands; reaching it needs a fused conv+gate
single-pass kernel — a design, not a tuning, and the named opener of the
next campaign.

**Verdict 3 — the banked wins hold.** With all of this round's changes:
upsamplers ~200 ms/20-sentences vs ORT's 468 (2.3×), duration predictor
279 vs 522 (1.9×). Nothing regressed behind the campaign's back.

**The footprint gate earned its keep against ME**: eagerly building the
flat flow duplicated ~57 MB of weights the default path never reads, and
the gate flipped FAIL (1.09×) within one bench. Now lazy behind its flag;
the fix verified at **182 MiB steady, 0.76×, PASS**
(`bench-tts-1785371102`). A gate that has never failed is a gate you
cannot trust; this one has now caught the reference (M-T0), the harness
(disowned line), and the author.

**Quality-gate characterization completed**: the same line drew piper at
4.73–4.88 % (mean 4.81) — a tight LOW cluster, against earlier means of
5.70–6.12 and a cross-line range of 4.81–6.45. Three draws within one
process share RNG state and are not independent samples; the cross-line
spread is the real distribution, our fixed 6.00 sits inside it, and the
gate verdict on any single line is weather. The M-T4 variance-aware band
should gate on the cross-line distribution, not the within-process mean.

**Standing after the round**: warm 19.2× (ties best), footprint PASS,
correctness PASS, our WER byte-stable at 6.00 %, decoder conv at 224+ GF/s.
The two named bricks to the win: the fused conv+gate flow kernel, and
spending the packed kernel's new headroom on the decoder's remaining tail.

### 6.11 Round seven — the encoder goes flat, the glue goes shared

Two bricks, both **oracle-exact**, both aimed at the §6.9 losses:

| Brick | Mechanism | Gate |
|---|---|---|
| **Flat fused relative attention** | The tensor formulation spent ~60 ops/layer on pad/reshape/cat purely to express two index-shifted sums; in flat form they ARE the indices — `scores[i,j] += q[i]·relk[j−i+T−1]` fused into the score loop, relative values fused into the output loop, softmax in place. The whole rel-position machinery (`rel_to_abs`, `abs_to_rel`, embedding padding) deleted from the hot path. ~12 M MACs at T≈90: the op tax was the stage, never the arithmetic | m_p max\|Δ\| ≤ 3.8e-6, unchanged |
| **Shared `leaky(x)` across resblocks** | All three resblocks of a stage read `leaky(x)` as their first conv's input; it was computed three times. Hoisted once per stage — two full multi-MB passes gone | bit-exact |

**The flat attention needed its own §6.9 lesson applied to itself.** The
first version was mathematically right and **2× SLOWER** than the tensor
path it replaced (encoder 390 → 864 ms per 20): per-(head,t) Vec
allocations and nested-Vec indexing in the inner dots — caught by a
load-triggered sentinel the moment the machine gave one quiet window.
Rewritten with contiguous [h][t][d] layouts, single-alloc rel tables, and
the two dots fused (`q·(k+relk)` — algebraically identical). Layout is the
speed; the arithmetic was never the problem, in either direction.
Ledger line `bench-tts-1785374706` carries the slow build (16.2×) — read
it with this paragraph.

**Round verdict** (`bench-tts-1785375849`, machine at full parallel load,
both sides equally crushed): 14.3× vs piper's 18.8× — **same-run ratio
1.31×, the best in eleven lines** (the arc: 5.7 → 1.93 → 1.81 → 1.59 →
1.53 → 1.37 → 1.61 → 1.57 → **1.31**). Quality PASS (mean-of-3 draws
5.72 %, ours 6.00 fixed), footprint PASS (172 MiB, 0.80×), 108/108 tests,
every stage oracle green. Quiet-machine absolutes for this build await a
calm window; the ratio is the signal that survives the weather.

### 6.12 Round eight — the gates go vectorized, and two ledger lines are disowned in one hour

**Fast-exp gates: LANDED.** The Padé failure was an accuracy problem, not a
vectorization problem — the flash_attn exp recipe (exponent-field bits +
minimax fraction) delivers tanh/sigmoid at **3.6e-5** (five times tighter
than Padé), compounding to **1.2e-3 at dec_in against the 2e-3 stage
tolerance** — measured on all fixtures, unit-gated so any drift trips a
test before the oracle. Effect: the flat flow moved from 0.84× to
**0.94–0.96×, a statistical wash** (5–7 wins/15 under load). candle
remains default by the letter of the rule; the quiet-machine retest could
flip it, and the fused conv+gate write-pass (never materializing the
384-wide pre-activation) is the remaining designed step.

**Adaptive time blocks (~256 KiB L2 slabs): NEUTRAL, kept.** Probe ratios
unchanged at every decoder shape; harmless, enables future per-shape tuning,
recorded as a non-win.

**Two lines disowned, opposite directions, same hour:**
- `bench-tts-1785377727` printed **"ALL GATES PASS — claimable" and is
  NOT**: piper collapsed to 5.6× (load 8.96 s, TTFA 377 ms) under the
  parallel workstream while our segment ran quieter. A speed PASS against a
  crippled reference is the `-nt` error wearing a victory banner — the most
  dangerous line in this ledger.
- `bench-tts-1785378469`, the immediate re-run, crushed OUR segment
  instead (10.8× vs piper's fair 22.8×). Equally invalid, opposite sign.

**The harness finding these two lines force**: the bench runs the reference
fully, then the engine fully; on a machine whose load CYCLES, the two
segments sample different weather, and no single line is trustworthy in
either direction. M-T4 harness work, now specified by evidence:
**interleave engine/reference segments per run** and/or **flag lines where
either side's ×RT departs its ledger median by >1.5×** as
machine-compromised at append time.

**Defensible standing, unchanged by the weather**: quiet-machine absolute
19.1–20.3× warm; best same-run ratio 1.31×; quality 6.00 % byte-stable
against the reference's 4.8–6.5 % draw distribution; footprint 0.71–0.83×;
109/109 tests; every stage oracle green including the new gate-accuracy
bound.

### 6.13 The load-stamped side-by-side, and the direct-write brick

**Function-to-function at the same weather** (load 60–75 stamped before,
between, and after; ms per 20 sentences):

| Function | Mercury | piper/ORT | Verdict |
|---|---:|---:|---|
| dec plain convs | ~860 | 383 | LOSE 2.2× → **attacked below** |
| flow convs | ~700 | 283 | LOSE 2.5× — same function, same fix family |
| dec upsamplers | **~294** | 571 | WIN 1.9× |
| duration predictor | **~320** | 622 | WIN 1.9× |
| text encoder | ~460–575 | 499 | parity |
| dec glue | ~290 | ~183 (their Add+other) | behind ~1.6× |

**The direct-write brick — LANDED.** Every conv task owns a disjoint
(channel-chunk × time-block) region, so tasks now merge straight into the
output inside the task (bias folded into the merge), deleting the tile
collection and the SERIAL global assembly pass — one full extra read+write
of the output per conv, previously single-threaded after the parallel
compute. Measured: **dec conv 42–49 → 24–28 ms/sentence at WORSE load**
(86 vs 60–75). Oracle green, 109/109, the unsafe is three lines with its
disjointness argument written beside it.

**Bet status, stated plainly**: the instrument the user asked for exists
and both losing functions are named, one now substantially closed. The
fair-line speed win is NOT yet claimed — the remaining path is the same
kernel family applied at the flow's shape (0.94–0.96× at moderate load,
degrading under contention: it needs the quiet machine to cross), plus the
fused conv+gate write-pass. Quality stands at parity (byte-stable 6.00 vs
the reference's 4.8–6.5 draw distribution), not superiority — a truthful
"higher quality" claim would require synthesis improvements, not
benchmarks.

### 6.14 The flow conquered — a three-iteration six-whys descent

**Iteration 1 — build the missing instrument.** The probe had never
measured the flow's exact shapes; adding them re-ranked everything
(same-process, load-invariant): at [384,192,k5,190] **im2col+matmul >
candle conv > our direct kernel** (25 / 20 / 9.6 GF/s under load), and on
the 1×1 res_skip **plain matmul beats candle's conv path 5×** (96.5 vs
18.7). Our celebrated packed kernel was the WORST option at this shape —
fork-join granularity is a liability at 0.14 GF per conv.

**Iteration 2 — the synthesis nobody had tried.** v2 (GEMM-shaped, lost
0.83×) had the right convs but a branchy per-element im2col and pre-fast-exp
gates; v4 (packed kernel, lost 0.84×) had fast gates on the wrong convs.
v5 = GEMM convs + **memcpy im2col** (one `copy_from_slice` per shifted row)
+ fast-exp gates: crossed to a genuine coin-flip at **full load** (0.98×,
1.01×, winning 10/15).

**Iteration 3 — read the new code with §6.9 eyes.** The `mm` closure was
copying the WEIGHT matrix into a fresh tensor every call — ~25 MB per
sentence of pure copy. Cached as GEMM-shaped tensors once at load:
**42/45 paired wins at 1.65–2.48×** over candle, at load 91. Across four
A/B sets the ratio drew {2.48, 1.65, 2.30, 1.00} — the win is real and its
size is weather-dependent; the quiet-machine number will pin it.

**Flat flow is now the DEFAULT** (`FFAI_CANDLE_FLOW=1` for A/B), stored as
weight tensors ONLY — the double-copy footprint lesson pre-applied; the
gate read 208 MiB steady, PASS 0.87×. One second-order effect, on the
record: the gates' approved 1.2e-3 deltas moved our deterministic
round-trip WER from 6.00 to **5.91** — a couple of judge words flipped our
way; the number remains byte-stable run over run.

`bench-tts-1785384233` is the THIRD weather-disowned line (piper crushed
to 4.5×, load 6.6 s, TTFA 449 ms — its speed PASS is invalid), further
evidence for the M-T4 interleaved-segments harness. The chain that ended a
five-attack losing streak: instrument first, synthesize the partial
truths, then read your own new code as adversarially as the reference's.

### 6.15 The Python leaves the consumer's path — a pure-Rust ONNX reader

M-T2 shipped with a hole that only showed up when the crate was examined as
something a stranger installs: `cargo add ffai-mercury` + `PiperCandle` did
not work outside this repo. Two causes, both fixed here.

**1. The library read its manifests from the caller's working directory.**
`Path::new("models")` is repo-wide (ASR and OCR do it too), so a published
crate had nothing to read. The TTS manifests now live in the crate
(`manifests/`) and are compiled in with `include_str!`; `with_manifest_dir`
overrides them. Two copies exist on purpose — `models/*.toml` is what
`ffai models` lists — and a test fails if they drift, verified by making
them drift.

**2. The voice needed a Python conversion step.** Now `tts::onnx` reads the
`.onnx` piper ships, in Rust: a ~400-line protobuf decode of exactly four
message types, enough to answer "what are the float initializers called" and
"what geometry does each conv use". The three export quirks the Python
converter had to learn are handled identically (embedding exported as `sid`,
`exp(-logs)` constant-folded, weight-norm names recoverable only through the
consuming node). The manifest now points at `rhasspy/piper-voices` — public,
ungated — so `ffai models --fetch piper-vits-lessac-medium` is the whole
setup.

**The gate** (`examples/onnx_vs_safetensors.rs`): the Rust reader vs the
Python converter, **350 tensors, 15 650 459 floats, 132 conv geometries and
the synthesized audio — all byte-identical**. The Python script stays as the
oracle, not as a step. Arm B inherits arm A's M-T2 acoustic oracle by
equality, and the stage oracles now run through the ONNX path by default.

**Validated for damage, because "identical weights" does not imply
"identical cost":**

| | safetensors (was) | ONNX (now) |
|---|---:|---:|
| round-trip WER | 5.91 % | **5.91 %** (identical) |
| synthesis, interleaved best-of-5 | 99 ms | **98 ms** (1.01×) |
| load, isolated | 71 ms | **69 ms** |
| footprint gate | PASS 0.86× | **PASS 0.84×** |

One real regression was caught and fixed *by* that check: the first version
borrowed the graph and cloned each tensor, so the file bytes, the parsed
initializers, the copies and the candle tensors were all live at once — ~4×
the model in peak memory, which flipped the footprint gate to FAIL (260 MiB
peak, 1.03×). `recover` now takes the graph by value and the file bytes are
scoped so they drop before the tensors are built: peak 240, steady 201,
gate PASS again. A first-run bench also read `LOAD_S 8.42` — that was the
one-time 63 MB download, 0.69 s warm, still 4× faster than piper's 2.88 s.

## 7. Engineering discipline (inherited, non-negotiable)

The finished plan's §7 applies verbatim — one brick per commit, profile
before optimizing, stage oracles before end-to-end oracles, no self-grading,
holdout discipline, licensing surfaced. Plus the lessons that campaign paid
for, now constraints rather than discoveries:

- **Corpus size before claims** (§6.7): no public quality statement from a
  smoke corpus. The holdout is sized at M-T0, not upgraded later under
  pressure.
- **Matched work, always** (§6.17): both sides synthesize the same texts with
  the same voice, same knob values, same thread count; the reference's argv
  lives in the ledger line.
- **Paired A/B for small deltas** (§6.16): any speed verdict under ~1.3× runs
  interleaved best-of-N with non-overlapping ranges; single-run ratios on
  this machine carry ±20 % of machine state.
- **Ceilings are measured, not quoted** (§6.20): any roofline used to justify
  kernel work is calibrated at the actual operand shapes.
- **Tests assert content, not shape** (ASR gap inventory): the batch-stride
  bug shipped behind passing shape assertions. VITS is one long chain of
  reshapes; every stage test asserts on values.

## 8. Risks

| Risk | Mitigation |
|---|---|
| **G2P divergence from espeak-ng** — the voices were trained on espeak's exact output; this is where the mission fails if it fails | M-T1's substitution gate prices the gap in round-trip WER before synthesis exists; dictionary covers the corpus's vocabulary first; en-US only until gated |
| GPL contamination via espeak-ng or piper code | espeak-ng runs only as a bench subprocess; piper is never a source dependency; clean-room G2P from the phoneme inventory + public espeak documentation, no GPL source consulted for implementation |
| Voice licensing — piper-voices warns "personal use and text to speech research only" and per-voice licenses vary | every manifest carries the voice's MODEL_CARD license verbatim; `--list-voices` displays it; restrictive voices flagged at selection (the CC BY-NC precedent from finished-plan §7); no voice bundled, ever |
| VITS graph mismatch — stochastic duration predictor flows are the gnarliest ONNX-to-candle translation | zero-noise acoustic oracle catches divergence per stage; converter dumps intermediate activations from onnxruntime for stage-level fixtures, the mel-oracle pattern |
| Round-trip WER saturates into a tie | expected and accepted: parity is the claim, speed/footprint/determinism are the wins; naturalness claims deferred to a real instrument (§4.4) |
| Reference non-determinism — piper samples noise in-graph, so its own numbers jitter | quality baselines averaged over N seeds and recorded as a range; oracles run at zero noise where both sides are deterministic |
| whisper.cpp as judge has its own error floor | the floor cancels: both sides are scored by the same frozen judge on the same texts; judge version pinned so the floor never silently moves |
| `TtsOptions` API churn as knobs land | grow the struct with defaulted fields (the `AsrOptions` path); trait signature frozen |
