#!/usr/bin/env python3
"""Score Diana's tracker on MOT17 — MOTA, IDF1, ID-switches.

    python tools/diana_mot_track_bench.py --seq mot17-09
    python tools/diana_mot_track_bench.py --all

AP50 cannot see a tracker. It scores boxes, and a tracker's whole job is the
IDENTITY attached to them — swap every id in a sequence and AP50 does not move
one point. This campaign has scored MOT17 on AP50 for weeks while discarding
column 2 of `gt.txt`, which is the track id.

The metrics that can see it:

* **MOTA** = 1 - (FN + FP + IDSW) / GT. Overall accuracy, dominated by the
  detector.
* **IDF1** = identity F1 over the best one-to-one mapping between predicted and
  ground-truth trajectories. This is the tracker's own score — it punishes a
  track that fragments or swaps even when every box is right.
* **IDSW** = raw identity switches.
* **MT/ML** = trajectories tracked for >=80 % / <=20 % of their life.

MOT conventions, matching the detection bench already in this repo: ground-truth
rows with `conf=0` are ignored regions and dropped, and only `class=1`
(pedestrian) is scored.

**PASS `--classes 0` TO THE DETECTOR.** MOT17 ground truth is pedestrians only,
and Diana is an 80-class COCO detector — so without the filter every car, bus
and traffic light is scored as a predicted pedestrian and charged as a false
positive. This campaign ran for multiple sessions without it. The contamination
was 13.8-47.3 % of detections depending on the sequence, and removing it is
worth **+1.54 pp IDF1 and +5.38 pp MOTA** overall:

    MOT17-13, end to end       IDF1     MOTA       FP
      no class filter         21.32    -6.22     2759
      --classes 0             25.02   +16.28       99

A NEGATIVE MOTA is the tell, and it sat in the results for weeks: MOTA is
1 - (FN+FP+IDSW)/GT, so going below zero means more errors than ground-truth
boxes. That is not a hard sequence, it is a broken comparison, and it should
have been chased the first time it appeared (codec-measurement s7 — an
impossible number is the instrument asking for help).

The reference was always run as `model.track(..., classes=[0])`, so the two
sides were never doing the same job.

**No thresholds are tuned here.** The tracking plan's stop rule is that the
first number must be honest, because a tracker has four knobs and fitting all of
them to one corpus is trivial.
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFAI = str(ROOT / "target" / "release" / "ffai.exe")


def load_gt(p: Path):
    """frame -> [(id, x, y, w, h)], MOT conventions applied."""
    out = defaultdict(list)
    for line in p.read_text().splitlines():
        f = line.split(",")
        if len(f) < 9:
            continue
        conf, cls = float(f[6]), int(float(f[7]))
        if conf == 0 or cls != 1:
            continue
        out[int(f[0])].append((int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5])))
    return out


def load_pred(p: Path):
    out = defaultdict(list)
    for line in p.read_text().splitlines():
        f = line.split(",")
        if len(f) < 7:
            continue
        out[int(f[0])].append((int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5])))
    return out


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / (aw * ah + bw * bh - i)


def greedy_match(gt, pr, thr=0.5):
    """Highest-IoU-first matching within a frame. Standard for CLEAR-MOT."""
    pairs, used_g, used_p = [], set(), set()
    cands = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pr):
            v = iou(g[1:], p[1:])
            if v >= thr:
                cands.append((v, gi, pi))
    for v, gi, pi in sorted(cands, reverse=True):
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi))
    return pairs


def score(gt, pred):
    frames = sorted(set(gt) | set(pred))
    fp = fn = idsw = n_gt = 0
    last_match = {}          # gt id -> pred id, for IDSW
    # For IDF1: how often each (gt id, pred id) pair co-occurred.
    co = defaultdict(int)
    gt_count = defaultdict(int)
    pr_count = defaultdict(int)
    tracked = defaultdict(int)

    for f in frames:
        g, p = gt.get(f, []), pred.get(f, [])
        n_gt += len(g)
        for x in g:
            gt_count[x[0]] += 1
        for x in p:
            pr_count[x[0]] += 1
        pairs = greedy_match(g, p)
        for gi, pi in pairs:
            gid, pid = g[gi][0], p[pi][0]
            co[(gid, pid)] += 1
            tracked[gid] += 1
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid
        fp += len(p) - len(pairs)
        fn += len(g) - len(pairs)

    mota = 1.0 - (fn + fp + idsw) / max(n_gt, 1)

    # IDF1 over the best one-to-one gt<->pred trajectory mapping. Greedy on
    # co-occurrence is the standard approximation and is what py-motmetrics
    # falls back to; the exact version is a global assignment.
    idtp = 0
    ug, up = set(), set()
    for (gid, pid), n in sorted(co.items(), key=lambda kv: -kv[1]):
        if gid in ug or pid in up:
            continue
        ug.add(gid)
        up.add(pid)
        idtp += n
    idfp = sum(pr_count.values()) - idtp
    idfn = sum(gt_count.values()) - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    mt = sum(1 for g, n in gt_count.items() if tracked.get(g, 0) / n >= 0.8)
    ml = sum(1 for g, n in gt_count.items() if tracked.get(g, 0) / n <= 0.2)
    # The raw identity counts, not just the ratio. Pooling a metric across
    # sequences has to sum the COUNTS and divide once: averaging seven
    # per-sequence IDF1 percentages weights a 525-frame clip the same as a
    # 1050-frame one, which silently changes what "overall IDF1" means
    # depending on which sequences happen to be in the set.
    return dict(mota=mota, idf1=idf1, idsw=idsw, fp=fp, fn=fn, gt=n_gt,
                mt=mt, ml=ml, traj=len(gt_count),
                idtp=idtp, idfp=idfp, idfn=idfn)


def run(seq, engine, conf, out):
    frames = ROOT / "corpora" / "clips" / seq / "img1"
    subprocess.run(
        [FFAI, "detect", "--serve", "--engine", engine, "--conf", str(conf)],
        cwd=str(ROOT), check=False, capture_output=True,
    )
    # Frame directory -> the CLI streams video, so feed it the extracted frames
    # through the same tracker via --live's directory form.
    raise SystemExit("use --pred to score an existing MOT file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="mot17-09")
    ap.add_argument("--pred", type=Path, required=True,
                    help="MOT-format predictions from `ffai detect --track -o`")
    args = ap.parse_args()

    gt_p = ROOT / "corpora" / "clips" / args.seq / "gt.txt"
    if not gt_p.exists():
        sys.exit(f"no ground truth at {gt_p}")
    r = score(load_gt(gt_p), load_pred(args.pred))

    print(f"  sequence      {args.seq}")
    print(f"  GT boxes      {r['gt']}   trajectories {r['traj']}")
    print()
    print(f"  MOTA          {100*r['mota']:6.2f} %")
    print(f"  IDF1          {100*r['idf1']:6.2f} %")
    print(f"  ID switches   {r['idsw']:6}")
    print(f"  FP / FN       {r['fp']} / {r['fn']}")
    print(f"  MT / ML       {r['mt']} / {r['ml']}  of {r['traj']}")
    print()
    print("  MOTA is dominated by the DETECTOR (FP+FN); IDF1 and IDSW are the")
    print("  TRACKER's own score. Published ByteTrack on MOT17 sits near MOTA 80 /")
    print("  IDF1 77 with a tuned detector - that is standing, not a target.")


if __name__ == "__main__":
    main()
