"""Per-stage twin of `profile_piper_matched.py` — the reference's stage budget
on the SAME phoneme ids and the SAME thread count our engine achieves.

Fixes two defects in `profile_piper_stages.py`: it profiled different sentences
than our probe, and it left ORT's thread count at a default (~15.6 effective)
while our engine reaches ~8.2, so its per-stage ms were both a different
workload and a different machine allocation.

The printed total is cross-checked against the independently measured wall
clock; if they disagree the stage split is not trustworthy.

    python corpora/refs/profile_piper_stages_matched.py \
        --model <voice.onnx> --ids ids.txt --threads 8
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument(
        "--unprofiled-wall",
        type=float,
        default=0.0,
        help="ms/pass for the same config with profiling OFF; enables the per-node tax correction",
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.ids).read_text().splitlines() if line.strip()]

    so = onnxruntime.SessionOptions()
    so.enable_profiling = True
    if args.threads:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
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

    # Profiling cannot be reset mid-session, so the warmup pass is inside the
    # profile and the divisor must include it.
    for ids in rows:
        run(ids)
    audio_secs = 0.0
    w0 = time.perf_counter()
    for _ in range(args.passes):
        audio_secs = 0.0
        for ids in rows:
            out = run(ids)
            audio_secs += out.size / 22050.0
    measured_wall_ms = (time.perf_counter() - w0) * 1000.0 / args.passes
    divisor = float(args.passes + 1)

    profile_path = sess.end_profiling()
    stages = collections.Counter()
    counts = collections.Counter()
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
        counts[stage] += 1
        op = ev.get("args", {}).get("op_name", "?")
        op_kinds[f"{stage}:{op}"] += dur
    Path(profile_path).unlink()

    node_ms = total_us / 1000.0 / divisor
    # Enabling the profiler makes the run materially slower, so the raw node
    # durations are NOT comparable to an un-profiled engine's stage timings.
    # The overhead is per-node, so charge it per node rather than scaling every
    # stage uniformly: a stage of many tiny nodes absorbs far more of it than a
    # stage of few large ones. `--unprofiled-wall` is the same config measured
    # with profiling off (from profile_piper_matched.py).
    total_nodes = sum(counts.values()) / divisor
    print(f"threads={args.threads} rows={len(rows)} {audio_secs:.1f}s audio/pass")
    ovh_ms = 0.0
    if args.unprofiled_wall > 0:
        ovh_ms = measured_wall_ms - args.unprofiled_wall
        per_node_us = ovh_ms * 1000.0 / total_nodes
        print(
            f"  profiler tax: {measured_wall_ms:.1f} profiled - {args.unprofiled_wall:.1f} "
            f"unprofiled = {ovh_ms:.1f} ms over {total_nodes:.0f} nodes "
            f"= {per_node_us:.2f} us/node"
        )
    print(f"  {'stage':<8} {'raw ms':>9} {'nodes':>8} {'corrected':>10}")
    for stage, us in stages.most_common():
        n = counts[stage] / divisor
        raw = us / 1000 / divisor
        corr = raw - (n * ovh_ms / total_nodes) if ovh_ms else raw
        print(f"  {stage:<8} {raw:9.1f} {n:8.0f} {corr:10.1f}")
    print(f"  {'NODE SUM':<8} {node_ms:9.1f} {total_nodes:8.0f} {node_ms - ovh_ms:10.1f}")
    print(
        f"  cross-check: node sum / profiled wall = {node_ms / measured_wall_ms:.2f} "
        f"({'OK' if 0.8 < node_ms / measured_wall_ms < 1.2 else 'MISMATCH - split not trustworthy'})"
    )
    print("\ntop ops:")
    for key, us in op_kinds.most_common(14):
        print(f"  {key:<30} {us / 1000 / divisor:8.1f} ms")


if __name__ == "__main__":
    main()
