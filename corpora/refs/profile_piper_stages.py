"""Stage-level profile of piper's OWN runtime via onnxruntime's per-node
profiler — the TTS counterpart of the ASR campaign's compare_stages: put the
reference's stage budget beside Mercury's so every loss has a name.

    profile_piper_stages.py --model <voice.onnx> --fixtures <espeak jsonl> --n 20

Aggregates node durations by module prefix (/enc_p, /dp, /flow, /dec) over
the same holdout sentences Mercury's profile_tts uses, warm (first run
discarded), best-of-3 per sentence.
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import onnxruntime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    rows = []
    for line in Path(args.fixtures).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            rows.append(obj["phoneme_ids"][0])
    # profile_tts uses the first 20 HOLDOUT sentences; the fixture file is all
    # 200 in corpus order — close enough for a stage-share comparison, and the
    # totals are reported per second of audio, not per sentence.
    rows = rows[: args.n]

    so = onnxruntime.SessionOptions()
    so.enable_profiling = True
    sess = onnxruntime.InferenceSession(
        args.model, sess_options=so, providers=["CPUExecutionProvider"]
    )

    def run(ids):
        feeds = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }
        return sess.run(["output"], feeds)[0]

    # Warm up (session setup, first-call kernel init), then 3 passes.
    audio_samples = 0
    for ids in rows:
        audio_samples = run(rows[0]).size
        break
    audio_secs = 0.0
    for _ in range(3):
        audio_secs = 0.0
        for ids in rows:
            out = run(ids)
            audio_secs += out.size / 22050.0

    profile_path = sess.end_profiling()
    stages = collections.Counter()
    op_kinds = collections.Counter()
    total_us = 0
    for ev in json.loads(Path(profile_path).read_text(encoding="utf-8")):
        if ev.get("cat") != "Node" or "dur" not in ev:
            continue
        dur = ev["dur"]
        total_us += dur
        name = ev.get("name", "")
        stage = "other"
        for prefix in ("enc_p", "dp", "flow", "dec"):
            if f"/{prefix}/" in name or name.startswith(f"{prefix}/"):
                stage = prefix
                break
        stages[stage] += dur
        op = ev.get("args", {}).get("op_name", "?")
        op_kinds[f"{stage}:{op}"] += dur

    # The profile covers warmup + 3 passes over n sentences; report per-pass.
    passes = 4.0
    print(f"piper/ort stage budget ({len(rows)} sentences, {audio_secs:.1f}s audio/pass, node time /pass):")
    for stage, us in stages.most_common():
        print(f"  {stage:<8} {us/1000/passes:8.1f} ms  {100.0*us/total_us:5.1f} %")
    print(f"  {'total':<8} {total_us/1000/passes:8.1f} ms  -> {audio_secs/(total_us/1e6/passes):.1f}x realtime (node-time)")
    print("\ntop ops:")
    for key, us in op_kinds.most_common(12):
        print(f"  {key:<28} {us/1000/passes:8.1f} ms")
    Path(profile_path).unlink()


if __name__ == "__main__":
    main()
