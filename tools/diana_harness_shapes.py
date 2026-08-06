#!/usr/bin/env python3
"""Does the HARNESS SHAPE change the verdict? Five arrangements, same work.

    python tools/diana_harness_shapes.py --reps 6

One arrangement — per-frame alternation, the side-by-side viewer's design — was
found to cost Diana 43 % while costing Ultralytics nothing, a 1.46x swing in the
reported ratio from the harness alone. That was discovered by accident. This
looks for the rest of them on purpose.

Every arrangement below does IDENTICAL work: the same frames, the same models,
the same conf. Only the ORDER and CO-RESIDENCY change. If the ratio moves
between them, the difference is the harness, not the engines — and whichever
number gets published needs to name its arrangement.

  solo      each engine alone, whole pass, ABBA at the pass level
  alternate per frame: diana, ultra, diana, ultra ... (the viewer)
  blockwise all of diana, then all of ultra (what §3 forbids)
  resident  solo, but the other engine's process is loaded and idle
  shuffled  solo, frames in random order - kills any sequential-frame locality
"""

import argparse
import glob
import json
import random
import statistics as st
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFAI = str(ROOT / "target" / "release" / "ffai.exe")


def dproc():
    p = subprocess.Popen(
        [FFAI, "detect", "--serve", "--engine", "yolo26n", "--conf", "0.25"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        cwd=str(ROOT))
    json.loads(p.stdout.readline())
    return p


def dframe(p, f):
    p.stdin.write(f.as_posix() + "\n")
    p.stdin.flush()
    return json.loads(p.stdout.readline())["ms"]


def close(p):
    try:
        p.stdin.close()
        p.wait(timeout=15)
    except Exception:
        p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=ROOT / "corpora/clips/mot17-09/img1")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--warm", type=int, default=8)
    args = ap.parse_args()

    frames = [Path(f) for f in sorted(glob.glob(str(args.frames / "*.jpg")))[: args.n]]
    from ultralytics import YOLO

    m = YOLO(str(ROOT / "corpora/cache/yolo26n.pt"))
    m.predict(str(frames[0]), verbose=False, conf=0.25)

    def uframe(f):
        t = time.perf_counter()
        m.predict(str(f), verbose=False, conf=0.25)
        return (time.perf_counter() - t) * 1000

    def solo(order=None):
        fr = order or frames
        p = dproc()
        d = [dframe(p, f) for f in fr]
        close(p)
        u = [uframe(f) for f in fr]
        return st.median(d[args.warm:]), st.median(u[args.warm:])

    def alternate():
        p = dproc()
        d, u = [], []
        for f in frames:
            d.append(dframe(p, f))
            u.append(uframe(f))
        close(p)
        return st.median(d[args.warm:]), st.median(u[args.warm:])

    def blockwise():
        p = dproc()
        d = [dframe(p, f) for f in frames]
        close(p)
        u = [uframe(f) for f in frames]
        return st.median(d[args.warm:]), st.median(u[args.warm:])

    def resident():
        # Diana's process stays alive and IDLE while Ultralytics runs, and vice
        # versa - isolating "the other engine is loaded" from "the other engine
        # is running".
        p = dproc()
        d = [dframe(p, f) for f in frames]
        u = [uframe(f) for f in frames]   # p still alive, idle
        close(p)
        return st.median(d[args.warm:]), st.median(u[args.warm:])

    def shuffled():
        fr = frames[:]
        random.Random(7).shuffle(fr)
        return solo(fr)

    shapes = [("solo", solo), ("alternate", alternate), ("blockwise", blockwise),
              ("resident", resident), ("shuffled", shuffled)]

    acc = {n: [] for n, _ in shapes}
    for rep in range(args.reps):
        order = shapes if rep % 2 == 0 else shapes[::-1]
        for name, fn in order:
            d, u = fn()
            acc[name].append((d, u))

    print(f"  {args.reps} reps x {len(frames)} frames, shape order alternated each rep\n")
    print(f"  {'arrangement':<12}{'diana ms':>10}{'ultra ms':>10}{'ratio':>9}"
          f"{'vs solo':>10}{'d-spread':>10}")
    base = None
    for name, _ in shapes:
        d = st.median([x[0] for x in acc[name]])
        u = st.median([x[1] for x in acc[name]])
        r = u / d
        if base is None:
            base = r
        sp = (max(x[0] for x in acc[name]) - min(x[0] for x in acc[name])) / d
        print(f"  {name:<12}{d:>10.2f}{u:>10.2f}{r:>8.3f}x{r/base:>9.2f}x{sp:>9.0%}")

    print("\n  ratio = ultralytics/diana; >1 means Diana faster.")
    print("  'vs solo' is how much the ARRANGEMENT alone moves the verdict.")
    worst = max(abs(1 - st.median([x[1] for x in acc[n]]) / st.median([x[0] for x in acc[n]]) / base)
                for n, _ in shapes)
    print(f"  largest harness-induced swing: {worst:.0%}")


if __name__ == "__main__":
    main()
