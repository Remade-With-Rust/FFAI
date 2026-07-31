"""Matched-footing speed twin of `pin_probe.rs`.

The existing `profile_piper_stages.py` profiles the first 20 sentences in
CORPUS order while our probe runs the first 20 HOLDOUT sentences. Those are
different texts, so they synthesize different amounts of audio, and the raw
"ms per 20 sentences" figures this campaign compared are not commensurable.

This script removes that defect: it consumes the EXACT phoneme-id rows our
engine ran (dumped by `FFAI_DUMP_IDS=... pin_probe`), so both arms drive the
same graph with the same sequence lengths and emit the same audio.

It also reports what the campaign never recorded for the reference: CPU time
alongside wall time. The cpu/wall ratio is the reference's effective thread
count, and a wall-clock comparison between an 8-thread arm and a 1-thread arm
is not a comparison of engines.

    python corpora/refs/profile_piper_matched.py \
        --model <voice.onnx> --ids ids.txt [--threads N]
"""

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime


def cpu_secs() -> float:
    """Process CPU time (kernel + user) in seconds, via GetProcessTimes.

    time.process_time() omits kernel time on some platforms; the Win32 call is
    the same instrument the Rust probe uses, which keeps the two comparable.
    """
    k = ctypes.c_ulonglong()
    u = ctypes.c_ulonglong()
    c = ctypes.c_ulonglong()
    e = ctypes.c_ulonglong()
    h = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.kernel32.GetProcessTimes(
        ctypes.c_void_p(h),
        ctypes.byref(c),
        ctypes.byref(e),
        ctypes.byref(k),
        ctypes.byref(u),
    )
    if not ok:
        return 0.0
    return (k.value + u.value) * 1e-7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True, help="one JSON int array per line")
    ap.add_argument("--threads", type=int, default=0, help="intra_op threads; 0 = ORT default")
    ap.add_argument("--passes", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.ids).read_text().splitlines() if line.strip()]

    so = onnxruntime.SessionOptions()
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
            # Same scales the Rust probe uses: noise 0.667, length 1.0, noise_w 0.8.
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }
        return sess.run(["output"], feeds)[0]

    # Warm: session setup and first-call kernel init are one-time costs.
    for ids in rows:
        run(ids)

    best_wall = float("inf")
    best_cpu = float("inf")
    audio_secs = 0.0
    for _ in range(args.passes):
        audio_secs = 0.0
        w0 = time.perf_counter()
        c0 = cpu_secs()
        for ids in rows:
            out = run(ids)
            audio_secs += out.size / 22050.0
        wall = time.perf_counter() - w0
        cpu = cpu_secs() - c0
        best_wall = min(best_wall, wall)
        best_cpu = min(best_cpu, cpu)

    print(f"threads={args.threads or 'default'} rows={len(rows)}")
    print(
        f"  {audio_secs:.1f}s audio, best-of-{args.passes}: "
        f"{best_wall * 1000:.1f} ms wall  {best_cpu * 1000:.1f} ms cpu  "
        f"-> {audio_secs / best_wall:.2f}x realtime  (cpu/wall {best_cpu / best_wall:.2f})"
    )


if __name__ == "__main__":
    main()
