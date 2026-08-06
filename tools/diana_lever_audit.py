#!/usr/bin/env python3
"""Audit every settled lever against the axes it was never judged on.

    python tools/diana_lever_audit.py            # the sweep
    python tools/diana_lever_audit.py --reps 8   # deeper

Every keep/revert decision in the Diana campaign was made on WALL TIME, on one
machine, at one thread count. That is one cell of a four-cell table, and the
other three each produced a result the first one missed (see
docs/diana-dispatch-audit-plan.md). This runs the other three.

MANDATORY COLUMNS, and each exists because its absence cost something:

* **wall ms/frame** — the shipped objective.
* **cpu ms/frame** — codec-measurement §2b. Less machine burned is a RESULT,
  not a consolation. An arm slower in wall on less CPU is the arm a
  throughput-bound host should pick.
* **cores busy** (`cpu/wall`) — §2a. Two arms at different parallelism are not
  comparable and NOTHING ELSE IN THE OUTPUT SAYS SO. Its absence let a lever be
  declared dead that was 1.12x faster once fanned out.
* **detections per pass** — §4 work parity. A count that differs between reps
  of the SAME arm voids that arm; a count that differs between arms is expected
  (they are different code) but must be constant within each.
* **per-rep spread** — so a 25 % spread is never read as a verdict.

ABBA-interleaved (§3): the arm order flips every rep, so drift lands on both
arms equally instead of on whichever ran second.

Every lever here is a DISABLE toggle — it turns a shipped optimisation off. So
`wall x` and `cpu x` below 1.00 mean the shipped default is winning that axis.
The interesting rows are the ones where the two axes DISAGREE.
"""

import argparse
import json
import os
import statistics as st
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
FFAI = ROOT / "target" / "release" / "ffai.exe"

# (label, env, prior, POLARITY, note) — ordered by expected yield, per the plan.
#
# POLARITY IS NOT COSMETIC. Most of these env vars DISABLE a shipped
# optimisation ("off"), but three ENABLE an alternative that was refuted
# ("alt"): DIRECT, NESTED_PAR and MIMALLOC_PURGE_DELAY. The raw ratios are the
# same either way, but the VERDICT inverts, and the first run of this sweep
# labelled all ten as "off" and read three of them backwards.
LEVERS = [
    ("no zero-copy SliceOp", {"FFAI_DIANA_NO_ZEROCOPY": "1"}, "validation", "off",
     "pure work removal; disabling MUST cost on both axes or the harness is wrong"),
    ("im2col zero-fill back", {"FFAI_DIANA_ZEROFILL": "1"}, "high", "off",
     "3.1-9.4 ms/img of redundant writes"),
    ("candle grouped depthwise", {"FFAI_DIANA_NO_DWCONV": "1"}, "high", "off",
     "different algorithm; CPU/parallelism profile unknown"),
    ("nested parallelism", {"FFAI_DIANA_NESTED_PAR": "1"}, "high", "alt",
     "refuted on a 2.32x CPU tax, never on cores-busy"),
    ("direct convolution", {"FFAI_DIANA_DIRECT": "1"}, "high", "alt",
     "0.95x CPU at 0.86x wall; re-test for under-parallelisation"),
    ("candle pointwise", {"FFAI_DIANA_NO_PW": "1"}, "medium", "off",
     "dispatched by KIND, never within kind"),
    ("candle stride-2", {"FFAI_DIANA_NO_S2": "1"}, "medium", "off", ""),
    ("candle dense 3x3", {"FFAI_DIANA_NO_CONV3": "1"}, "medium", "off", ""),
    ("scalar SiLU (no AVX2)", {"FFAI_DIANA_NO_AVX2": "1"}, "pruned", "off",
     "SiLU is 1.0 % of detect - ceiling is 1 %, measured only because it is free"),
    ("mimalloc purge off", {"MIMALLOC_PURGE_DELAY": "-1"}, "non-toggle", "alt",
     "page faults are KERNEL cpu; only wall was ever measured"),
]

TOGGLE_KEYS = sorted({k for _, e, _, _, _ in LEVERS for k in e})


def spawn(env_extra, engine):
    env = dict(os.environ)
    for k in TOGGLE_KEYS:
        env.pop(k, None)
    env.update(env_extra)
    p = subprocess.Popen(
        [str(FFAI), "detect", "--serve", "--engine", engine],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        bufsize=1, cwd=str(ROOT), env=env)
    json.loads(p.stdout.readline())  # ready
    return p


def measure(p, frames):
    """CPU, wall and detection count for one pass. Model load already excluded."""
    ps = psutil.Process(p.pid)
    c0 = ps.cpu_times()
    w0 = time.perf_counter()
    n_det = 0
    for f in frames:
        p.stdin.write(f.as_posix() + "\n")
        p.stdin.flush()
        n_det += json.loads(p.stdout.readline())["n"]
    wall = time.perf_counter() - w0
    c1 = ps.cpu_times()
    cpu = (c1.user - c0.user) + (c1.system - c0.system)
    assert cpu > 0, "CPU read as zero over a pass that did work"
    return cpu, wall, n_det


def close(p):
    try:
        p.stdin.close()
        p.wait(timeout=15)
    except Exception:
        p.kill()


def classify(wx, cx, cores_ratio, pol):
    """The five buckets from the plan. 'Inconclusive' is deliberately absent.

    SEMANTICS, and they invert with POLARITY — the first version got this wrong
    twice, once by flipping the ratio and once by ignoring polarity entirely:

    `wx`/`cx` are always the ARM over the shipped default. For a `pol="off"`
    lever the arm has the optimisation turned OFF, so `wx > 1` means the
    optimisation is WINNING wall. For `pol="alt"` the arm is a refuted
    ALTERNATIVE, so `wx < 1` means that alternative is winning and the shipped
    default is the loser.
    """
    if pol == "off":
        wins_wall, wins_cpu = wx > 1.02, cx > 1.03
        loses_wall, loses_cpu = wx < 0.98, cx < 0.97
    else:
        # The arm IS the alternative; the default wins when the arm is worse.
        wins_wall, wins_cpu = wx < 0.98, cx < 0.97
        loses_wall, loses_cpu = wx > 1.02, cx > 1.03
        wins_wall, wins_cpu, loses_wall, loses_cpu = (
            not (wx < 0.98), not (cx < 0.97), wx < 0.98, cx < 0.97)

    if loses_wall and loses_cpu:
        return "MISSED WIN - flip the default"
    if wins_wall and loses_cpu:
        return "TRADE - default buys wall with CPU"
    if loses_wall and wins_cpu:
        return "TRADE - default buys CPU with wall"
    if loses_wall and cores_ratio > 1.25:
        return "AXIS-B SUSPECT - default under-parallelised"
    if wins_wall and wins_cpu:
        return "confirmed on both axes"
    return "no effect either axis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path,
                    default=ROOT / "corpora/clips/mot17-09/img1")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--engine", default="yolo26n")
    ap.add_argument("--only", default=None, help="substring filter on the label")
    args = ap.parse_args()

    frames = sorted(q for q in args.frames.iterdir()
                    if q.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.n]

    print(f"method: ABBA-interleaved, {args.reps} reps x {len(frames)} frames, "
          f"CPU via GetProcessTimes, both arms in child processes, model load excluded")
    print(f"corpus: {args.frames}\n")
    print(f"  {'lever (DISABLED)':<26}{'wall x':>8}{'cpu x':>7}{'cores':>7}"
          f"{'spread':>8}  {'pol':>4}  verdict")

    rows = []
    for label, env, prior, pol, note in LEVERS:
        if args.only and args.only.lower() not in label.lower():
            continue
        base, arm = [], []
        dets = {"base": set(), "arm": set()}
        for rep in range(args.reps):
            a = spawn({}, args.engine)
            b = spawn(env, args.engine)
            order = [("base", a), ("arm", b)] if rep % 2 == 0 else [("arm", b), ("base", a)]
            got = {}
            for name, proc in order:
                got[name] = measure(proc, frames)
            base.append(got["base"])
            arm.append(got["arm"])
            dets["base"].add(got["base"][2])
            dets["arm"].add(got["arm"][2])
            close(a)
            close(b)

        bw = st.median([x[1] for x in base]) / len(frames) * 1000
        bc = st.median([x[0] for x in base]) / len(frames) * 1000
        aw = st.median([x[1] for x in arm]) / len(frames) * 1000
        ac = st.median([x[0] for x in arm]) / len(frames) * 1000
        wx, cx = aw / bw, ac / bc
        cores_b, cores_a = bc / bw, ac / aw
        wl = [x[1] for x in arm]
        spread = (max(wl) - min(wl)) / st.median(wl)
        parity = "ok" if len(dets["arm"]) == 1 and len(dets["base"]) == 1 else "VOID"
        v = classify(wx, cx, cores_a / cores_b, pol) if parity == "ok" else "work parity VOID"
        rows.append((label, prior, wx, cx, cores_a, cores_b, spread, parity, v, note))
        print(f"  {label:<26}{wx:>8.2f}{cx:>7.2f}{cores_a:>7.2f}{spread:>7.0%}  "
              f"{pol:>4}  {v}")

    print(f"\n  baseline: wall {bw:.2f} ms/f, cpu {bc:.2f} ms/f, cores {cores_b:.2f}")
    print("\n  wall x / cpu x are the DISABLED arm over the shipped default.")
    print("  <1.00 means the shipped default wins that axis.")
    print("  >1.00 means DISABLING costs more, i.e. the shipped default wins that axis.")
    print("  AXIS-B SUSPECT = the default runs on far fewer cores than the arm beating it.")

    flagged = [r for r in rows if "TRADE" in r[8] or "SUSPECT" in r[8] or "MISSED" in r[8]]
    if flagged:
        print(f"\n  {len(flagged)} row(s) need follow-up:")
        for r in flagged:
            print(f"    {r[0]:<26} {r[8]}")
            if r[9]:
                print(f"      {r[9]}")
    else:
        print("\n  no rows flagged - every lever is settled on both axes.")


if __name__ == "__main__":
    main()
