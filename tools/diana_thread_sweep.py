#!/usr/bin/env python3
"""Thread-width sweep: can Diana spend its CPU headroom to buy wall latency?

    python tools/diana_thread_sweep.py --frames corpora/clips/mot17-09/img1

The descent that closed docs/whys/diana-1080p-and-tail.md ended on a fact
rather than a fix: Diana uses **3.25x less CPU than Ultralytics at a wall ratio
near 1.0**. Ultralytics is converting more cores into the same wall clock. A
user feels wall, so unspent CPU is unspent latency.

Thread width was refuted at 4-6 workers earlier in this campaign.
`codec-measurement` §11: **a refutation expires when its baseline moves**, and
it has — epilogue fusion, JIT decode, and the decode-timing fix all landed
since. This re-runs it rather than inheriting the old answer.

Both metrics are reported per width because they answer different questions:

* **wall ms/frame** is the product question — what a user waits.
* **cpu ms/frame** is the cost question — what a server pays. It also reveals
  the parallel tax: if wall falls 20 % while CPU rises 80 %, that width is
  buying latency at a price a batch workload should refuse.
* **cpu/wall** is the achieved parallelism, and it is the honest check on
  whether the extra workers did anything at all.

Reps are interleaved across widths (round-robin, not width-blocked) so a drift
in the neighbouring build lands on every width equally instead of on the last
one measured.
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


def spawn(threads, engine, conf):
    env = dict(os.environ)
    env["FFAI_DIANA_THREADS"] = str(threads)
    p = subprocess.Popen(
        [str(FFAI), "detect", "--serve", "--engine", engine, "--conf", str(conf)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        cwd=str(ROOT), env=env)
    assert json.loads(p.stdout.readline()).get("ready")
    return p


def tree_cpu(ps):
    t = ps.cpu_times()
    total = t.user + t.system
    for c in ps.children(recursive=True):
        try:
            u = c.cpu_times()
            total += u.user + u.system
        except psutil.Error:
            pass
    return total


def one_pass(p, paths):
    ps = psutil.Process(p.pid)
    c0, w0 = tree_cpu(ps), time.perf_counter()
    n, dets = 0, 0
    for q in paths:
        p.stdin.write(f"{q.as_posix()}\n")
        p.stdin.flush()
        r = json.loads(p.stdout.readline())
        dets += r.get("n", 0)
        n += 1
    wall = time.perf_counter() - w0
    cpu = tree_cpu(ps) - c0
    return cpu / n * 1000, wall / n * 1000, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--engine", default="yolo26n")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--widths", default="1,2,4,6,8,12")
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",")]
    paths = sorted(q for q in args.frames.iterdir()
                   if q.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.n]

    print(f"method: round-robin interleaved across widths, CPU time via process tree, "
          f"model load excluded, {len(paths)} frames x {args.reps} reps")
    print(f"corpus: {args.frames}   cores: {psutil.cpu_count()}\n")

    cpu = {w: [] for w in widths}
    wall = {w: [] for w in widths}
    dets = {w: set() for w in widths}

    # ONE PROCESS ALIVE AT A TIME. Spawning every width up front and
    # round-robining between them looks tidier and is void: six live thread
    # pools contend with each other, and the first attempt read a 105.8 %
    # within-arm spread with a non-monotonic curve (1->156, 4->77, 6->167).
    # A width's cost is only its own when nothing else of ours is resident.
    # Model load stays outside the timed region either way, because `spawn`
    # waits for the `ready` line before `one_pass` starts its clock.
    for rep in range(args.reps):
        order = widths if rep % 2 == 0 else list(reversed(widths))
        for w in order:
            p = spawn(w, args.engine, args.conf)
            c, wl, d = one_pass(p, paths)
            try:
                p.stdin.close()
                p.wait(timeout=10)
            except Exception:
                p.kill()
            cpu[w].append(c)
            wall[w].append(wl)
            dets[w].add(d)

    all_dets = set().union(*dets.values())
    print(f"  WORK PARITY: detections per pass across every width = {sorted(all_dets)}")
    if len(all_dets) != 1:
        print("  *** widths disagree on output — thread count is changing RESULTS, stop ***")

    base_w = min(wall, key=lambda w: st.median(wall[w]))
    print(f"\n  {'threads':>8} {'wall ms':>9} {'cpu ms':>9} {'cpu/wall':>9} "
          f"{'vs 1-thread wall':>17}")
    w1 = st.median(wall[widths[0]])
    for w in widths:
        mw, mc = st.median(wall[w]), st.median(cpu[w])
        print(f"  {w:8} {mw:9.1f} {mc:9.1f} {mc/mw:9.2f} {w1/mw:16.2f}x")

    best = st.median(wall[base_w])
    cur = st.median(wall[4]) if 4 in wall else None
    print(f"\n  lowest wall at {base_w} threads: {best:.1f} ms/frame")
    if cur and base_w != 4:
        gain = cur / best
        print(f"  against the shipped default of 4: {gain:.3f}x")
        spread = (max(wall[4]) - min(wall[4])) / st.median(wall[4])
        print(f"  within-arm spread at 4 threads: {spread:.1%}")
        print(f"  -> a {gain:.3f}x claim needs to clear that spread AND the null-arm")
        print(f"     floor (10.4% CPU on this box) before it is a result.")
    elif base_w == 4:
        print("  the shipped default of 4 is still the lowest-wall width.")


if __name__ == "__main__":
    main()
