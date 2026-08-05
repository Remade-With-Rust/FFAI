#!/usr/bin/env python3
"""D6 probe: the per-frame latency SERIES, not its mean.

    python tools/diana_tail_probe.py --frames corpora/clips/mot17-09/img1 --n 200

A mean/median gap says a distribution has a tail. It does not say WHERE the tail
is, and the two candidates want opposite fixes:

* **front-loaded** — the first frames are slow and everything after is flat.
  That is warm-up, and if the harness warmed one engine and not the other it is
  a harness defect, not an engine property.
* **spread** — slow frames occur throughout. That is a real tail: allocator
  purge cycles, thread-pool parking, page faults, scheduler eviction.

So this prints the series, not a summary: the first 12 frames alone, then the
distribution of everything after them, then the worst offenders with their frame
index. If the worst indices are all small, the answer is warm-up.

Deterministic counts first (codec-measurement, order of instruments): frames in,
detections out, and frames whose latency exceeds 2x the steady median. Those are
one-run numbers no scheduler can move.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
FFAI = ROOT / "target" / "release" / ("ffai.exe" if sys.platform == "win32" else "ffai")


def run_diana(paths, engine, conf):
    cmd = [str(FFAI), "detect", "--serve", "--engine", engine, "--conf", str(conf)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1, cwd=str(ROOT))
    assert json.loads(proc.stdout.readline()).get("ready")
    out = []
    for p in paths:
        proc.stdin.write(f"{p.as_posix()}\n")
        proc.stdin.flush()
        r = json.loads(proc.stdout.readline())
        out.append((r["ms"], r["n"], r.get("decode_ms",0.0)))
    proc.stdin.close()
    proc.wait(timeout=10)
    return out


def run_ultra(paths, weights, conf):
    from ultralytics import YOLO
    model = YOLO(weights)
    out = []
    for p in paths:
        t = time.perf_counter()
        res = model.predict(str(p), verbose=False, conf=conf)[0]
        out.append(((time.perf_counter() - t) * 1000.0,
                    0 if res.boxes is None else len(res.boxes), 0.0))
    return out


def report(name, series, warm=12):
    ms = [m for m, _, _ in series]
    dets = sum(n for _, n, _ in series)
    dec = [d for _, _, d in series]
    head, tail = ms[:warm], ms[warm:]
    steady = median(tail) if tail else median(ms)
    over = [(i + warm, v) for i, v in enumerate(tail) if v > 2 * steady]

    print(f"\n=== {name} ===")
    print(f"  frames {len(ms)}   detections {dets}   (deterministic: same every run)")
    if any(dec): print(f"  decode share: p50 {median(dec[warm:]):.1f} ms of {median(ms[warm:]):.1f} ms total = {100*median(dec[warm:])/median(ms[warm:]):.0f}%")
    print(f"  first {warm}: " + " ".join(f"{v:.0f}" for v in head))
    if not tail:
        return steady, 0
    s = sorted(tail)
    q = lambda f: s[min(len(s) - 1, int(len(s) * f))]
    print(f"  after warm-up  n={len(tail)}")
    print(f"    min {s[0]:6.1f}   p10 {q(.10):6.1f}   p50 {steady:6.1f}   "
          f"p90 {q(.90):6.1f}   p99 {q(.99):6.1f}   max {s[-1]:6.1f}")
    print(f"    mean {sum(tail)/len(tail):6.1f}   mean/p50 {(sum(tail)/len(tail))/steady:5.2f}")
    print(f"    COUNT frames > 2x steady median: {len(over)}/{len(tail)}"
          + (f"   at {[i for i, _ in over[:12]]}" if over else ""))
    print(f"  whole series   mean {sum(ms)/len(ms):6.1f}   p50 {median(ms):6.1f}"
          f"   mean/p50 {(sum(ms)/len(ms))/median(ms):5.2f}   <-- what the viewer reported")
    return steady, len(over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--engine", default="yolo26n")
    ap.add_argument("--weights", default="corpora/cache/yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warm", type=int, default=12)
    ap.add_argument("--skip-ultra", action="store_true")
    args = ap.parse_args()

    paths = sorted(p for p in args.frames.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.n]
    print(f"{len(paths)} frames from {args.frames}")

    d_steady, d_over = report(f"DIANA  {args.engine}", run_diana(paths, args.engine, args.conf), args.warm)
    if not args.skip_ultra:
        u_steady, u_over = report("ULTRALYTICS", run_ultra(paths, args.weights, args.conf), args.warm)
        print(f"\n  steady-state ratio (ultra/diana): {u_steady/d_steady:.2f}x"
              f"  -- {'ahead' if u_steady >= d_steady else 'behind'}")
        print(f"  outlier counts: diana {d_over}, ultralytics {u_over}")


if __name__ == "__main__":
    main()
