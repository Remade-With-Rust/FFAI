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

# 3. Warm every model cache OUTSIDE the timed run
.venv-bench/Scripts/python corpora/refs/faster_whisper_ref.py \
    --batch one-clip.txt --model tiny.en

# 4. Measure
cargo run -p ffai-cli -- bench asr \
    --corpus corpora/librispeech-test-clean-v1.toml \
    --baseline-only --runs 3
```

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

## The ledger

Every run appends one JSON line to `bench/ledger.jsonl` containing the
corpus fingerprint, every implementation's metrics, reference versions, the
environment (OS, arch, rustc, CPU), and the full gate report.

**Every performance or quality claim FFai makes in public should be
traceable to a ledger line.** That is the whole point of the apparatus: not
to make us look good, but to make us checkable.
