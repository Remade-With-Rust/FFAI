#!/usr/bin/env python3
"""Is the latency tail OURS, or the BOX's? Decided by coincidence, not by size.

    python tools/diana_tail_paired.py --frames corpora/clips/mot17-09/img1 --n 150

The question this answers cannot be answered by comparing two tail SIZES on a
loaded machine, because the load is what makes tails. It can be answered by
asking WHERE the slow frames fall.

Both engines process frame i back to back — Diana, then Ultralytics, then on to
frame i+1 — so within a frame they see the same machine, the same neighbouring
build, the same scheduler. Then:

* **If the spikes COINCIDE**, the cause is outside both engines. A stall that
  hits Diana on frame 71 and Ultralytics on frame 71 is not a property of
  either one.
* **If the spikes are INDEPENDENT**, each engine has its own tail, and the
  sizes are worth comparing.

The statistic is the overlap of the two outlier sets against what independence
would predict. With a Diana outliers and b Ultralytics outliers over n frames,
independent spiking predicts a*b/n coincidences; the ratio of observed to
expected is the answer, and it needs no quiet box.

Page faults are counted alongside, because an allocator-driven tail is the one
mechanism that WOULD be Diana's own — mimalloc retaining and re-faulting is
what the campaign's largest measured effect (1.64x) was about. A per-frame
fault count that is flat while latency spikes rules it out; that is a COUNT,
and codec-measurement's order of instruments puts it above any clock here.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import psutil

ROOT = Path(__file__).resolve().parents[1]
FFAI = ROOT / "target" / "release" / ("ffai.exe" if sys.platform == "win32" else "ffai")
PY = Path(sys.executable)


def spawn(kind, engine, weights, conf):
    if kind == "diana":
        cmd = [str(FFAI), "detect", "--serve", "--engine", engine, "--conf", str(conf)]
    else:
        cmd = [str(PY), str(ROOT / "tools" / "diana_ultra_serve.py"), weights, str(conf)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1, cwd=str(ROOT))
    assert json.loads(p.stdout.readline()).get("ready")
    return p


def faults(ps):
    try:
        return ps.memory_info().num_page_faults
    except Exception:
        return 0


def outliers(series, k=2.0):
    m = median(series)
    return {i for i, v in enumerate(series) if v > k * m}, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--engine", default="yolo26n")
    ap.add_argument("--weights", default="corpora/cache/yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--k", type=float, default=2.0, help="outlier = k x that arm's own median")
    args = ap.parse_args()

    paths = sorted(q for q in args.frames.iterdir()
                   if q.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.n]

    d = spawn("diana", args.engine, args.weights, args.conf)
    u = spawn("ultra", args.engine, args.weights, args.conf)
    dps = psutil.Process(d.pid)

    dm, um, df = [], [], []
    f_prev = faults(dps)
    for p in paths:
        d.stdin.write(f"{p.as_posix()}\n"); d.stdin.flush()
        rd = json.loads(d.stdout.readline())
        f_now = faults(dps)
        u.stdin.write(f"{p.as_posix()}\n"); u.stdin.flush()
        ru = json.loads(u.stdout.readline())
        dm.append(rd["ms"]); um.append(ru["ms"]); df.append(f_now - f_prev)
        f_prev = f_now

    for p in (d, u):
        p.stdin.close(); p.wait(timeout=10)

    dm, um, df = dm[args.warm:], um[args.warm:], df[args.warm:]
    n = len(dm)
    do, dmed = outliers(dm, args.k)
    uo, umed = outliers(um, args.k)
    both = do & uo
    exp = len(do) * len(uo) / n if n else 0

    print(f"\n  frames {n} (first {args.warm} dropped as warm-up), interleaved per frame")
    print(f"  medians   diana {dmed:6.1f} ms   ultralytics {umed:6.1f} ms   "
          f"ratio {umed/dmed:.2f}x")
    print(f"  mean/p50  diana {sum(dm)/n/dmed:5.2f}        ultralytics {sum(um)/n/umed:5.2f}"
          f"   <- the 'heavier tail' statistic")

    print(f"\n  outliers (> {args.k}x that arm's OWN median)")
    print(f"    diana        {len(do):3}  at {sorted(do)[:14]}")
    print(f"    ultralytics  {len(uo):3}  at {sorted(uo)[:14]}")
    print(f"    COINCIDENT   {len(both):3}  at {sorted(both)[:14]}")
    print(f"\n    expected if independent: {exp:.1f}")
    if exp > 0:
        r = len(both) / exp
        print(f"    observed / expected    : {r:.1f}x")
        if r >= 3:
            print("    => the spikes COINCIDE far beyond chance. The tail is the BOX,")
            print("       not either engine. Comparing tail sizes here measures the load.")
        elif r <= 1.5:
            print("    => spikes are ~independent. Each engine has its own tail;")
            print("       the sizes above are comparable.")
        else:
            print("    => partly shared. Neither reading is safe on this box.")

    if any(df):
        slow = [df[i] for i in do] or [0]
        norm = [df[i] for i in range(n) if i not in do] or [0]
        print(f"\n  page faults/frame (diana)  normal {median(norm):.0f}   on slow frames {median(slow):.0f}")
        print("    a flat count across slow frames rules out an allocator-driven tail")


if __name__ == "__main__":
    main()
