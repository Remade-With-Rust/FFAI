# Benchmarking FFai

How `ffai bench` produces numbers we're willing to publish, and how to
reproduce them. The discipline is inherited from Prometheus, the private
refinery built for `remade_ffmpeg_rs`; this document is the public contract.

## The one call

```sh
ffai bench asr --corpus corpora/librispeech-test-clean-v1.toml
ffai bench asr --corpus corpora/librispeech-test-clean-v1.toml --baseline-only
```

`--baseline-only` measures just the external references — useful before our
engine exists, so the targets are on the board from day one.

## Principles

1. **Same data, same code, same clock.** Every implementation reads the same
   16 kHz mono WAV files from the same holdout split, is scored by the same
   metric code, and is timed by the same harness.
2. **The corpus is pinned.** Each clip's SHA-256 lives in the manifest, and a
   run aborts if the bytes on disk have drifted. The manifest's own hash is
   recorded in every ledger line, so no result silently moves onto different
   data.
3. **Holdout discipline.** Claims are measured on `holdout` clips only.
   `train` clips exist so tuning has somewhere legal to happen.
4. **A skipped gate is never a pass.** Four gates — correctness, quality,
   speed, footprint — and `verdict: claimable` requires all four to pass.
5. **Losses are recorded.** The ledger is append-only. A result that didn't
   go our way stays in the file.
6. **No self-grading.** Quality is judged against external ground truth or a
   frozen third-party implementation, never against another FFai engine.

## Timing: why two numbers, always

Invoking a Python reference once per clip would put interpreter startup and
model load inside every timed run — several seconds of overhead against a
few seconds of audio. That measurement would show our Rust as
spectacularly faster and the claim would be worthless.

So ASR references are invoked **once for the whole corpus** and report
per-clip transcription time themselves. Two numbers come out, and both are
recorded:

| Metric | Meaning | Who cares |
|---|---|---|
| `xRT_WARM` | media seconds per second of processing, model already loaded | steady-state throughput; what implementations publish and what a server experiences |
| `xRT_E2E` | media seconds per wall second for the whole corpus run, including one model load | what a CLI user experiences |

`LOAD_S` records the model load separately. Quoting only the flattering
number is how benchmarks lie; the ledger keeps both so it can't happen by
accident.

**Model downloads must not be timed.** Warm every reference's cache once
before a measured run (see the reproduce steps below).

**Keep the machine quiet.** Best-of-N takes the minimum precisely because it
is the run least perturbed by other load, but nothing rescues a benchmark
run alongside a compile or a large download. Don't start other heavy work
while a measured run is in flight.

## Configuration: pin it, or you're measuring the wrong thing

Implementations ship different defaults. openai-whisper decodes **greedily**
by default; faster-whisper defaults to **beam_size=5**. Benchmarking one
against the other as-shipped compares decoding strategies while claiming to
compare implementations — and beam search is both slower and more accurate,
so the result is wrong in *both* dimensions at once.

We hit this during the first M0 run: faster-whisper appeared *slower* than
openai-whisper, which contradicts everything known about it. The cause was
the unpinned default, not the implementation.

So every ASR reference in [`corpora/references.toml`](../corpora/references.toml)
pins `--beam-size 5` explicitly, and the exact argv is recorded in each
ledger line's `command` field. When we add our own engine, it gets pinned to
the same decode configuration.

The general rule: **anything that changes the work being done must be
identical across implementations and recorded in the ledger.** Model size,
beam size, compute type, temperature schedule, thread count.

## Scoring: normalization

Raw WER between a LibriSpeech reference (`MISTER QUILTER`, `TWENTY THREE`)
and a Whisper hypothesis (`Mr. Quilter`, `23`) is mostly noise about
formatting. Both sides pass through a port of Whisper's
`EnglishTextNormalizer` ([`crates/ffai-bench/src/normalize.rs`](../crates/ffai-bench/src/normalize.rs))
— identically, for every implementation, so it advantages no one.

Parity status is documented honestly in that module: the replacer table,
filler removal, symbol handling, and spelled-number → digit conversion are
implemented; the ~1,700-entry British/American spelling map and Unicode NFKD
diacritic handling are **not yet** at bit-parity with openai-whisper. Until
they are, treat cross-implementation comparisons from this harness as sound
and absolute agreement with published WER figures as approximate.

## Reproducing the ASR baseline

Prerequisites: Rust, `tar`, `ffmpeg`, and Python 3.9+.

```sh
# 1. Reference implementations, isolated from your system Python
python -m venv .venv-bench
.venv-bench/Scripts/pip install faster-whisper                    # Windows
.venv-bench/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-bench/Scripts/pip install openai-whisper
#   (on unix: .venv-bench/bin/pip ...)

# 2. Corpus — fetched from the canonical source, never committed
curl -LO https://www.openslr.org/resources/12/test-clean.tar.gz
cargo run -p ffai-bench --example prepare_librispeech -- \
    --archive test-clean.tar.gz \
    --out corpora/clips/librispeech-test-clean \
    --manifest corpora/librispeech-test-clean-v1.toml \
    --count 16

# 3. Warm every model cache OUTSIDE the timed run — ours and the references'
cargo run -p ffai-cli -- models --fetch whisper-tiny-en
.venv-bench/Scripts/python corpora/refs/faster_whisper_ref.py \
    --batch one-clip.txt --model tiny.en

# 4. Measure — RELEASE BUILD ONLY
cargo build --release -p ffai-cli
./target/release/ffai bench asr \
    --corpus corpora/librispeech-test-clean-v1.toml --runs 3
```

**Never benchmark a debug build.** Rust debug builds run 10–100× slower than
release, and the references are optimized C++ and PyTorch. A debug-build
comparison isn't a conservative estimate, it's a meaningless one.

Step 2 is deterministic: the same archive and arguments regenerate a
byte-identical manifest, so a published benchmark can be re-derived from the
public source rather than trusted.

## Adding a reference

No code required — declare it in [`corpora/references.toml`](../corpora/references.toml):

```toml
[[reference]]
name = "my-asr-tool"
task = "asr"
batch_command = ["my-tool", "--files", "{filelist}", "--json"]
version_command = ["my-tool", "--version"]
```

`{filelist}` is replaced with a temp file holding one audio path per line.
The adapter prints JSONL to stdout — one object per clip:

```json
{"load_secs": 1.83}
{"path": "clip1.wav", "text": "the transcript", "transcribe_secs": 0.42}
```

`transcribe_secs` must exclude model load. See
[`corpora/refs/`](../corpora/refs/) for working adapters. Simple tools whose
startup is negligible (tesseract) may use single-file `command` mode with
`{input}` instead.

## The OCR vertical (Carmenta M-C0)

Same spine, three task-scoped differences, all visible in the code rather
than folklore:

1. **The media unit is the page.** `xRT` becomes pages/second (warm and
   end-to-end, same two-number rule). The ledger schema is unchanged;
   `media_secs` carries the page count and the task field scopes its meaning.
2. **Scoring is `Mode::Ocr`**: whitespace runs collapse to a single space and
   *nothing else* is normalized. Reading case, digits, and punctuation is
   OCR's job — the ASR normalizer would score the task's hardest parts as
   free. The quality gate verdicts on **CER** (OCR's ranking metric), with
   WER recorded beside it.
3. **Per-page latency percentiles land in the notes** of every reference run
   (p50/p95 from the adapter's own per-image timings). That is the number a
   LIVE streaming loop experiences, and it is on the board from M-C0 so the
   M-C2 latency gate has baselines waiting. Caveat recorded where it
   belongs: tesseract has no server mode, so its per-page times include
   ~10–30 ms of process spawn; the in-memory references don't pay that. See
   `corpora/refs/tesseract_ref.py`.

### Reproducing the OCR baseline

Prerequisites: Rust, `curl`, and the `.venv-bench` from the ASR section.

```sh
# 1. Reference implementations (CPU)
.venv-bench/Scripts/pip install easyocr --extra-index-url https://download.pytorch.org/whl/cpu
.venv-bench/Scripts/pip install paddlepaddle paddleocr

# 2. Tesseract — portable, no admin: extract the official installer.
#    (Any tesseract 5.x on PATH also works; adjust references.toml.)
#    7-Zip extracted via `msiexec /a` administrative install, then:
curl -sLO https://github.com/tesseract-ocr/tesseract/releases/download/5.5.3/tesseract-ocr-w64-setup-5.5.3.20260724.exe
.tools-bench/7zip/Files/7-Zip/7z.exe x tesseract-ocr-w64-setup-*.exe -o.tools-bench/tesseract -y
curl -sL -o .tools-bench/tesseract/tessdata/eng.traineddata \
    https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata

# 3. Corpora — synthetic, deterministic, license-free (fonts fetched, never
#    committed; the manifest pins the generated bytes by SHA-256)
curl -sL https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip -o corpora/fonts/dejavu.zip
# extract DejaVuSans.ttf, DejaVuSerif.ttf, DejaVuSansMono.ttf into corpora/fonts/
cargo run -p ffai-bench --example prepare_carmenta_synth -- --fonts corpora/fonts --out corpora

# 4. Warm every model cache OUTSIDE the timed run (EasyOCR and PaddleOCR
#    download weights on first use)
echo corpora/clips/carmenta-render/page-00.png > warm.txt
.venv-bench/Scripts/python corpora/refs/easyocr_ref.py --batch warm.txt
.venv-bench/Scripts/python corpora/refs/paddleocr_ref.py --batch warm.txt

# 5. Measure — RELEASE BUILD ONLY
cargo build --release -p ffai-cli
./target/release/ffai bench ocr --corpus corpora/carmenta-render-v1.toml --baseline-only
./target/release/ffai bench ocr --corpus corpora/carmenta-frames-v1.toml --baseline-only
```

Configuration pinning carries over unchanged: tesseract's `--psm`/`--oem`
are this vertical's `beam_size` — explicit in references.toml, recorded in
every ledger line. PaddleOCR's document-preprocessing stages (orientation,
unwarping) are explicitly disabled so every reference does the same work:
bare detection + recognition.

The synthetic corpora are the smoke/oracle tier: exact ground truth, zero
license friction, regenerable from a fixed seed. They cannot support public
claims about real-world documents — audited public ground-truth corpora are
the claims tier and land per the Carmenta mission plan (§6.2).

## The detection vertical (Diana M-D0)

Same spine, four task-scoped differences:

1. **The media unit is the image.** `xRT` becomes images/second (warm and
   end-to-end, same two-number rule); `media_secs` carries the image count
   and the task field scopes its meaning, exactly as pages do for OCR.
2. **Scoring is mAP, not edit distance.** `crates/ffai-bench/src/detect.rs`
   implements the COCO matching rule — confidence-ranked greedy assignment
   per class at IoU 0.50:0.05:0.95, maxDets 100, 101-point interpolated AP
   averaged over classes with ground truth. The ledger gains two fields
   (`map50`, `map5095`); both are recorded so neither is quoted alone. The
   quality gate verdicts on **`1 − mAP@0.5`** — the gate machinery is
   written lower-is-better, so the mAP is folded into a miss-rate rather
   than growing a second comparison direction.
3. **The scorer is cross-validated before it is trusted.**
   `tools/diana_validate_scorer.py` scores the same detections through
   pycocotools and through the Rust scorer and fails if they differ by more
   than 0.005 absolute. On the M-D0 dump they agree to four decimals
   (0.7014 / 0.5377 both ways). This is the Carmenta instrument-defect
   lesson applied in advance: a new scorer on a new corpus is cross-checked
   against a known-good implementation *before* any number it produces goes
   on the board, not after a contradiction appears.
4. **The wire format is JSON boxes, not text.** Adapters return
   `[[x0,y0,x1,y1,cls,conf], ...]` in original-image pixels in the `text`
   field of the standard batch JSONL, and ground truth is one
   `{"width","height","objects"}` JSON per image. The batch/timing/memory
   contract is otherwise reused unchanged.

### Reproducing the detection baseline

Prerequisites: Rust and Python 3.11+. The detection stack gets its **own**
venv — `ultralytics` pulls a torch version that would disturb `.venv-bench`.

```sh
# 1. Reference implementations (CPU)
python -m venv .venv-diana
.venv-diana/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-diana/Scripts/pip install ultralytics onnxruntime pycocotools

# 2. Weights — AGPL-3.0, fetched ungated, NEVER vendored (see
#    docs/diana-mission-plan.md §7.1). Ultralytics downloads on first use.
cd corpora/cache && ../../.venv-diana/Scripts/python -c \
    "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26s.pt')" && cd ../..
.venv-diana/Scripts/python tools/diana_export_onnx.py   # the ORT deployment bar

# 3. Corpus — COCO val2017 subset, license-filtered and crowd-free.
#    Images are re-encoded losslessly to PNG so every implementation reads
#    IDENTICAL pixels rather than each running its own JPEG decoder.
curl -o corpora/cache/annotations_trainval2017.zip \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
python tools/diana_coco_corpus.py

# 4. Validate the scorer against pycocotools BEFORE measuring anything
.venv-diana/Scripts/python corpora/refs/ultralytics_ref.py --batch holdout.txt \
    --model corpora/cache/yolo26n.pt --imgsz 640 --conf 0.001 --max-dets 100 \
    --rect off > dets.jsonl
.venv-diana/Scripts/python tools/diana_validate_scorer.py \
    --corpus corpora/diana-coco-v2.toml --dets dets.jsonl

# 5. Measure — RELEASE BUILD ONLY (use tools/rebuild.sh: it kills stale
#    processes and fails if any .rs is newer than the binary)
bash tools/rebuild.sh
./target/release/ffai bench detect --corpus corpora/diana-coco-v2.toml --baseline-only

# ...or, linking only the two crates the measurement needs (useful when
# another component crate is mid-edit — the CLI links all of them):
cargo run --release -p ffai-diana --example bench_detect -- \
    corpora/diana-coco-v2.toml --runs 3
```

Configuration pinning carries over unchanged, and this vertical found its
own instance of the defect the rule exists for. The decode triple
`--imgsz 640 --conf 0.001 --max-dets 100` is straightforward — mAP needs
the low-confidence tail and maxDets must match the scorer's truncation —
but the knob that actually bit is **`--rect`, this vertical's `beam_size`**.
Ultralytics' `predict()` defaults `rect=True`, letterboxing each image to
the smallest multiple-of-32 *rectangle* (a 586×640 image is fed as
640×608), while the ONNX export is fixed 640×640 square. Left unpinned, the
`.pt` and ORT rows of the same tier ran different input geometry and their
mAP disagreed by 1.5–1.8 pp in inconsistent directions. `references.toml`
now pins `--rect off` on both sides for the matched comparison and carries
the official rectangular default as its own declared variant with its own
config key — the same shape as the ASR references' explicit greedy/beam
variants.

YOLO26 is natively end-to-end, so there is no NMS knob to pin — the ONNX
export emits final detections `[1, 300, 6]` and
`corpora/refs/yolo_ort_ref.py` **refuses** a raw-head export rather than
supplying NMS glue, which would put work belonging to the engine under test
inside a reference adapter.

The corpus is license-filtered at build time: COCO *annotations* are
CC-BY-4.0, but the images carry individual Flickr licenses, so
`tools/diana_coco_corpus.py` admits only the Attribution / Attribution-
ShareAlike / no-known-copyright / US-Government classes and records the
license per clip. It also excludes every image containing an `iscrowd=1`
annotation, because pycocotools treats crowd regions as ignore-zones and the
proxy scorer deliberately implements no ignore logic — the corpus excludes
the cases where that would matter instead of silently mis-scoring them.

## TTS: the round-trip bench

```sh
ffai bench tts --corpus corpora/harvard-sentences-v1.toml --baseline-only
```

The TTS vertical inverts the data flow — the corpus pins **text**, and audio
is what implementations produce — but keeps every principle above. What
changes, and why:

- **Quality is round-trip intelligibility.** Every implementation's audio is
  transcribed by a single **frozen third-party judge** (declared as
  `task = "tts-judge"` in `references.toml` — currently whisper.cpp
  base.en/beam5) and scored as WER/CER between the input text and the
  transcript. The judge's own error floor is shared noise on both sides of
  any comparison; it never cancels absolutely, so round-trip WER supports
  *parity* claims against a reference, not absolute quality claims. Exactly
  one judge may be declared, and never an FFai engine (principle 6).
- **The harness owns the judge's input format.** Generated WAVs are loaded,
  downmixed and resampled to 16 kHz mono by `ffai-bench`'s own windowed-sinc
  resampler — the same code path for every implementation, so no score
  depends on a native sample rate or an external tool's resampler.
- **Generated audio is measured, not trusted:** `media_secs` (and therefore
  ×RT) comes from the harness loading the WAVs of the kept run.
- **Time-to-first-audio** percentiles (p50/p95, adapter-timed, warm) ride in
  the notes — the latency a streaming caller experiences.
- **The judge pass is untimed** and runs after the synthesis batch.

TTS adapter contract (`batch_command` with `{filelist}` of text paths and
`{outdir}` for WAVs; see [`corpora/refs/piper_ref.py`](../corpora/refs/piper_ref.py)):

```json
{"load_secs": 2.17}
{"voice": "en_US-lessac-medium", "voice_sha256": "5efe09...", "noise_scale": 0.667}
{"path": "hvd-01-01.txt", "wav": "out/hvd-01-01.wav", "synth_secs": 0.07,
 "ttfa_secs": 0.07, "audio_secs": 2.21}
```

`synth_secs` excludes model load **and** WAV writing; any one-time inference
warm-up (onnxruntime's first-call kernel setup measured at ~3.3 s vs ~0.07 s
steady) must be folded into `load_secs`, not the first clip. A line with a
`voice` key is carried into the ledger notes verbatim — voice files are not
corpus-pinned, so their hash rides in the record instead.

Reproducing the TTS baseline:

```sh
# 1. piper1-gpl + its voice (ungated, hash echoed into every ledger line)
.venv-bench/Scripts/pip install piper-tts
.venv-bench/Scripts/python -m piper.download_voices en_US-lessac-medium --data-dir .piper-voices

# 2. Corpus — Harvard sentences (public domain; texts ARE committed, and the
#    manifest pins them; regenerate from the canonical page if you prefer)
curl -sLO https://www.cs.columbia.edu/~hgs/audio/harvard.html
cargo run -p ffai-bench --example prepare_harvard -- --source harvard.html

# 3. Judge model (whisper.cpp base.en)
curl -sL -o .whispercpp/ggml-base.en.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# 4. Measure — RELEASE BUILD ONLY
cargo build --release -p ffai-cli
./target/release/ffai bench tts --corpus corpora/harvard-sentences-v1.toml --baseline-only --runs 3
```

One honesty note specific to piper: it samples noise inside its ONNX graph
with no seed control, so its audio — and its round-trip WER — varies run to
run. The harness scores the audio of the run whose wall clock it kept; treat
sub-point WER deltas as noise until measured across seeds.

## The ledger

Every run appends one JSON line to `bench/ledger.jsonl` containing the
corpus fingerprint, every implementation's metrics, reference versions, the
environment (OS, arch, rustc, CPU), and the full gate report.

**Every performance or quality claim FFai makes in public should be
traceable to a ledger line.** That is the whole point of the apparatus: not
to make us look good, but to make us checkable.
