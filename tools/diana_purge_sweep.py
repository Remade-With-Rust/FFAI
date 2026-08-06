#!/usr/bin/env python3
"""What does each mimalloc purge delay actually cost, and buy?

    python tools/diana_purge_sweep.py --frames corpora/clips/mot17-09/img1

`MIMALLOC_PURGE_DELAY=-1` was measured to remove Diana's page-fault spikes
completely — 152/frame normal and 1,257 on slow frames, down to a flat 15 — for
+37.8 MiB of steady RSS. That is the extreme of a knob, not the knob: the value
is a DELAY IN MILLISECONDS before mimalloc returns an idle page to the OS, and
-1 means never.

The interesting question is therefore not "purge or not" but where on the curve
the faults have already collapsed while the retention has not yet arrived. A
frame takes ~60 ms; a delay comfortably longer than one frame should keep pages
across the gap between frames, which is where they are being lost, without
retaining everything forever.

Three quantities per setting, and they are not equally trustworthy:

* **page faults/frame** — a COUNT. One run, immune to the box.
* **steady and peak RSS** — a SIZE. Also deterministic enough to read directly,
  and the one the footprint gate scores.
* **wall p50** — a CLOCK, on a box at 66 % foreign load. Reported, not trusted;
  the whole reason this is a trade is that the clock side measured 0.3 %.

The gate this feeds: ffai-bench scores steady RSS against ONNX Runtime's
161 MiB, and Diana currently passes by 1 MiB. Any row whose steady RSS rises by
more than that margin fails a gate the project currently holds.
"""

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
FFAI = ROOT / "target" / "release" / ("ffai.exe" if sys.platform == "win32" else "ffai")
ORT_STEADY_MIB = 161.0


def run(delay, paths, engine, conf, warm):
    env = dict(os.environ)
    if delay is not None:
        env["MIMALLOC_PURGE_DELAY"] = str(delay)
    else:
        env.pop("MIMALLOC_PURGE_DELAY", None)
    p = subprocess.Popen(
        [str(FFAI), "detect", "--serve", "--engine", engine, "--conf", str(conf)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        cwd=str(ROOT), env=env)
    assert json.loads(p.stdout.readline()).get("ready")
    ps = psutil.Process(p.pid)

    ms, faults, peak, dets = [], [], 0, 0
    f_prev = ps.memory_info().num_page_faults
    for i, q in enumerate(paths):
        p.stdin.write(f"{q.as_posix()}\n")
        p.stdin.flush()
        r = json.loads(p.stdout.readline())
        mi = ps.memory_info()
        if i >= warm:
            ms.append(r["ms"])
            faults.append(mi.num_page_faults - f_prev)
            peak = max(peak, mi.rss)
            dets += r.get("n", 0)
        f_prev = mi.num_page_faults
    steady = ps.memory_info().rss
    p.stdin.close()
    p.wait(timeout=10)
    return {
        "p50": st.median(ms), "mean_p50": (sum(ms) / len(ms)) / st.median(ms),
        "faults": st.median(faults), "faults_max": max(faults),
        "steady": steady / 1048576, "peak": peak / 1048576, "dets": dets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--engine", default="yolo26n")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--delays", default="default,50,250,1000,5000,-1")
    args = ap.parse_args()

    paths = sorted(q for q in args.frames.iterdir()
                   if q.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.n]
    delays = [None if d == "default" else int(d) for d in args.delays.split(",")]

    print(f"corpus {args.frames}, {len(paths)} frames ({args.warm} dropped as warm-up)")
    print(f"footprint gate: ORT steady = {ORT_STEADY_MIB:.0f} MiB; Diana currently passes by ~1 MiB\n")
    print(f"  {'purge delay':>12} {'faults/fr':>10} {'worst':>8} {'steady MiB':>11} "
          f"{'peak MiB':>9} {'wall p50':>9} {'mean/p50':>9}")

    base = None
    for d in delays:
        r = run(d, paths, args.engine, args.conf, args.warm)
        if base is None:
            base = r
        label = "default" if d is None else str(d)
        print(f"  {label:>12} {r['faults']:10.0f} {r['faults_max']:8.0f} {r['steady']:11.1f} "
              f"{r['peak']:9.1f} {r['p50']:9.1f} {r['mean_p50']:9.2f}")
        r["_label"] = label
        if d is not None:
            dsteady = r["steady"] - base["steady"]
            verdict = "FAILS the footprint gate" if dsteady > 1.0 else "within gate headroom"
            print(f"  {'':>12} -> steady {dsteady:+.1f} MiB vs default, "
                  f"latency {100*(r['p50']/base['p50']-1):+.1f}%   {verdict}")
        if r["dets"] != base["dets"]:
            print(f"  *** detections changed ({r['dets']} vs {base['dets']}) — not a pure knob ***")

    print("\n  faults/frame is a COUNT (one run, immune to the box).")
    print("  steady/peak MiB is a SIZE (deterministic, and what the gate scores).")
    print("  wall p50 is a CLOCK on a box at 66% foreign load — reported, not trusted.")


if __name__ == "__main__":
    main()
