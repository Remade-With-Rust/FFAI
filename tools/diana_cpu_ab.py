#!/usr/bin/env python3
"""ABBA-interleaved CPU-time A/B: Diana against Ultralytics, on a box that will not hold still.

    python tools/diana_cpu_ab.py --frames corpora/clips/mot17-09/img1 --n 60 --reps 6

WHY CPU TIME. `codec-measurement` §2: affinity restricts you, it does not
reserve the core. Elapsed wall counts the time you spent DESCHEDULED, so on a
box running someone else's `cargo test --release --workspace` the wall clock is
measuring their build as much as your engine. CPU time does not accrue
off-core. Measured on that skill's own campaign, wall spread 0.78-1.50 against
CPU 0.950-1.089 for the same binaries.

WHY SUBPROCESSES FOR BOTH. Both engines run behind the same stdin/stdout
protocol in their own child process, so each child's CPU time is that engine's
work and nothing else. Running one in the driver would have charged it with the
harness. Model load is excluded: each child prints `ready` when loaded and CPU
is sampled after that line arrives.

WHY THE COARSE CLOCK IS FINE. GetProcessTimes updates on the ~15.6 ms timer
tick, far too coarse for one 40 ms frame. It is measured across N frames and
divided, so at N=60 the granularity contributes ~0.26 ms/frame. Never read a
single frame off this instrument.

ABBA. Arm order flips every rep, so "the second one is warmer" and any monotone
drift in the neighbouring build cancel instead of landing on one arm.

THE NULL ARM. `--null` runs Diana against Diana. Whatever it reports as a
difference is this harness's floor on this box in this session, and any real
delta smaller than it is not a result. Run it first; §15's corollary is that
the floor is not stationary.
"""

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

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
    hello = json.loads(p.stdout.readline())
    assert hello.get("ready"), hello
    return p


def tree_cpu(ps):
    """CPU seconds for a process AND its children.

    A venv's `Scripts/python.exe` on Windows can be a trampoline that execs the
    real interpreter as a child, so reading the spawned pid alone returned
    exactly 0.000 s while the arm was plainly doing 84 ms/frame of work.
    codec-measurement §7: an impossible number is the instrument asking for
    help. Summing the tree is the fix; the assert below is so it cannot fail
    silently a second time.
    """
    total = ps.cpu_times()
    total = total.user + total.system
    for c in ps.children(recursive=True):
        try:
            t = c.cpu_times()
            total += t.user + t.system
        except psutil.Error:
            pass
    return total


def measure(p, paths):
    """CPU seconds and wall seconds consumed by `p` across `paths`, plus counts."""
    ps = psutil.Process(p.pid)
    c0 = tree_cpu(ps)
    w0 = time.perf_counter()
    n_det, n_frames = 0, 0
    for path in paths:
        p.stdin.write(f"{path.as_posix()}\n")
        p.stdin.flush()
        r = json.loads(p.stdout.readline())
        n_det += r.get("n", 0)
        n_frames += 1
    wall = time.perf_counter() - w0
    cpu = tree_cpu(ps) - c0
    assert cpu > 0, ("CPU time read as zero across a pass that plainly did work - "
                     "the instrument is watching the wrong process")
    return cpu, wall, n_frames, n_det


def close(p):
    try:
        p.stdin.close()
        p.wait(timeout=10)
    except Exception:
        p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--engine", default="yolo26n")
    ap.add_argument("--weights", default="corpora/cache/yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--null", action="store_true", help="diana vs diana: the harness floor")
    args = ap.parse_args()

    paths = sorted(q for q in args.frames.iterdir()
                   if q.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.n]
    kinds = ("diana", "diana") if args.null else ("diana", "ultra")
    label = ("diana-A", "diana-B") if args.null else ("diana", "ultralytics")

    print(f"method: ABBA-interleaved, CPU time via GetProcessTimes, both arms in "
          f"child processes, model load excluded, {len(paths)} frames x {args.reps} reps")
    print(f"corpus: {args.frames}  ({'NULL ARM' if args.null else 'cross-implementation'})\n")

    a, b = spawn(kinds[0], args.engine, args.weights, args.conf), \
           spawn(kinds[1], args.engine, args.weights, args.conf)
    procs = {label[0]: a, label[1]: b}

    rows = {label[0]: [], label[1]: []}
    walls = {label[0]: [], label[1]: []}
    dets = {label[0]: set(), label[1]: set()}

    for rep in range(args.reps):
        order = [label[0], label[1]] if rep % 2 == 0 else [label[1], label[0]]
        for name in order:
            cpu, wall, nf, nd = measure(procs[name], paths)
            rows[name].append(cpu / nf * 1000.0)
            walls[name].append(wall / nf * 1000.0)
            dets[name].add(nd)
        print(f"  rep {rep+1}  " + "   ".join(
            f"{n} cpu {rows[n][-1]:6.1f} wall {walls[n][-1]:6.1f}" for n in label))

    for p in procs.values():
        close(p)

    print("\n  WORK PARITY (codec-measurement §4) — detections per pass, must be constant")
    for n in label:
        print(f"    {n:12} {sorted(dets[n])}")
    if not args.null and len(dets[label[0]] | dets[label[1]]) > 2:
        print("    NOTE: counts differ between arms; that is expected for two engines,")
        print("    what matters is that each arm is CONSTANT across its own reps.")
    for n in label:
        if len(dets[n]) != 1:
            print(f"    *** {n} is NOT deterministic across reps — comparison is void ***")

    print(f"\n  {'arm':12} {'cpu ms/frame':>14} {'wall ms/frame':>15}")
    for n in label:
        print(f"  {n:12} {st.median(rows[n]):14.1f} {st.median(walls[n]):15.1f}")

    ra = st.median(rows[label[0]])
    rb = st.median(rows[label[1]])
    wa = st.median(walls[label[0]])
    wb = st.median(walls[label[1]])
    print(f"\n  CPU  ratio ({label[1]}/{label[0]}): {rb/ra:.3f}x")
    print(f"  WALL ratio ({label[1]}/{label[0]}): {wb/wa:.3f}x")
    spread = lambda v: (max(v) - min(v)) / st.median(v)
    print(f"  within-arm spread   cpu {spread(rows[label[0]]):.1%} / {spread(rows[label[1]]):.1%}"
          f"   wall {spread(walls[label[0]]):.1%} / {spread(walls[label[1]]):.1%}")
    if args.null:
        print("\n  ^ this IS the floor. Any cross-implementation delta smaller than the")
        print("    CPU ratio's distance from 1.000 is not a result.")


if __name__ == "__main__":
    main()
