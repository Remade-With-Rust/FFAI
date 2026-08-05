"""Side-by-side on MOT17: Diana (gated and ungated) against Ultralytics.
The question nothing measured so far can answer: **what does the LIVE gate
COST?** Every prior number — 12.58x, 95.8 % skip, the delta fix — measures how
much work the gate skips. A gated frame serves the PREVIOUS frame's boxes, so
on a scene with people walking through it those boxes go stale. Without ground
truth a stale box and a fresh one look identical, which is exactly why the
synthetic corpus could not price it.
MOT17-09 is a fixed camera with 525 annotated frames, so the price is
measurable: score gated and ungated against the same GT and read the
difference.
Work parity, deliberately: all three arms are handed the SAME extracted frames
from disk. Neither engine pays for a decode the other does not — the asymmetry
that voided a benchmark earlier in this project.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON = 0  # COCO class id
def load_gt(path):
    """MOT gt.txt -> {frame: [(x0,y0,x1,y1), ...]} for evaluated pedestrians.
    MOT17 marks non-evaluated rows with conf=0 and uses class 1 for
    'pedestrian'. Scoring the other classes (people on vehicles, static
    persons, distractors) against a COCO 'person' detector would grade the
    class taxonomy rather than the detector, so they are dropped — the
    convention every published MOT detection baseline follows.
    """
    gt = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 9:
                continue
            fr, _id, x, y, w, h, conf, cls, vis = (
                int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                float(p[4]), float(p[5]), float(p[6]), int(p[7]), float(p[8]),
            )
            if conf == 0 or cls != 1:
                continue
            gt[fr].append((x, y, x + w, y + h))
    return gt
def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
def average_precision(dets, gt, thr=0.5):
    """Single-class AP at a fixed IoU, greedy matching by descending score.
    Plain area-under-PR by the all-points rule, which is what COCO's AP50
    reduces to for one class and is enough to compare two arms of the SAME
    detector against each other — which is the question here.
    """
    flat = [(f, s, b) for f, boxes in dets.items() for (s, b) in boxes]
    flat.sort(key=lambda t: -t[1])
    npos = sum(len(v) for v in gt.values())
    if npos == 0:
        return 0.0, 0, 0
    used = defaultdict(set)
    tp, fp = [], []
    for fr, _s, box in flat:
        best, bi = 0.0, -1
        for i, g in enumerate(gt.get(fr, [])):
            if i in used[fr]:
                continue
            v = iou(box, g)
            if v > best:
                best, bi = v, i
        if best >= thr and bi >= 0:
            used[fr].add(bi)
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    ctp = ctf = 0
    prev_r, ap = 0.0, 0.0
    for t, f in zip(tp, fp):
        ctp += t; ctf += f
        r = ctp / npos
        p = ctp / (ctp + ctf)
        ap += (r - prev_r) * p
        prev_r = r
    return ap, sum(tp), npos
def run_diana(frames_dir, out_path, gated, extra=()):
    exe = os.path.join(ROOT, "target", "release", "ffai.exe")
    cmd = [exe, "detect", "-i", frames_dir, "--conf", "0.25", "-o", out_path]
    if gated:
        cmd.append("--live")
    cmd += list(extra)
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t
    if r.returncode != 0:
        raise SystemExit(f"diana failed: {r.stderr[-500:]}")
    return wall, r.stdout.strip()
def parse_diana(path):
    """stem TAB class TAB conf TAB x0 TAB y0 TAB x1 TAB y1 -> {frame: [(score, box)]}"""
    out = defaultdict(list)
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            q = line.rstrip().split("	")
            if len(q) < 7 or q[1] != "person":
                continue
            out[int(q[0])].append(
                (float(q[2]), (float(q[3]), float(q[4]), float(q[5]), float(q[6])))
            )
    return out
def run_ultralytics(frames_dir, model):
    """Same frames from disk, so neither engine pays a decode the other does not."""
    from ultralytics import YOLO
    net = YOLO(model)
    dets = defaultdict(list)
    n = 0
    for f in sorted(os.listdir(frames_dir)):
        fr = int(os.path.splitext(f)[0])
        r = net.predict(
            os.path.join(frames_dir, f), conf=0.25, device="cpu",
            classes=[PERSON], verbose=False,
        )[0]
        n += 1
        b = r.boxes
        if b is not None and len(b) > 0:
            xy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            for k in range(len(cf)):
                dets[fr].append((float(cf[k]), tuple(float(v) for v in xy[k])))
    return dets, n
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="corpora/clips/mot17-09")
    ap.add_argument("--model", default="corpora/cache/yolo26n.pt")
    args = ap.parse_args()
    seq = os.path.join(ROOT, args.seq)
    frames = os.path.join(seq, "img1")
    gt = load_gt(os.path.join(seq, "gt.txt"))
    n_frames = len(os.listdir(frames))
    print(f"MOT17-09: {n_frames} frames, {sum(len(v) for v in gt.values())} evaluated pedestrians")
    print("all three arms are fed the SAME extracted frames from disk")
    rows = []
    # UNGATED: change_fraction 0 means EVERY frame exceeds the threshold, so
    # nothing is ever gated. Same code path; gating is the only variable.
    #
    # The first version used 2.0 — "must change 200%" — reasoning that an
    # impossible threshold disables the gate. It does the OPPOSITE: nothing
    # crosses it, so every frame counts as UNCHANGED and all of them gate. The
    # arms ran swapped and the labels lied. The log line each arm prints is
    # what caught it.
    o = os.path.join(seq, "diana_ungated.tsv")
    wall, log = run_diana(frames, o, gated=True, extra=["--change-fraction", "0.0"])
    rows.append(("diana ungated", parse_diana(o), wall, log))
    o = os.path.join(seq, "diana_gated.tsv")
    wall, log = run_diana(frames, o, gated=True)
    rows.append(("diana LIVE gated", parse_diana(o), wall, log))
    t = time.perf_counter()
    ud, un = run_ultralytics(frames, os.path.join(ROOT, args.model))
    rows.append(("ultralytics", ud, time.perf_counter() - t, f"{un} frames"))
    print(f"{'arm':<20} {'AP50':>8} {'recall':>8} {'dets':>7} {'wall s':>8} {'fps':>7}")
    results = {}
    for name, dets, wall, log in rows:
        a, tp, npos = average_precision(dets, gt, 0.5)
        nd = sum(len(v) for v in dets.values())
        results[name] = {"ap50": a, "tp": tp, "npos": npos, "dets": nd, "wall": wall}
        print(f"{name:<20} {a*100:7.2f}% {tp/npos*100:7.2f}% {nd:>7} {wall:8.1f} {n_frames/wall:7.2f}")
        if log:
            first = log.splitlines()[0] if log.splitlines() else ""
            if first:
                print(f"    {first}")
    ug = results["diana ungated"]["ap50"]
    ga = results["diana LIVE gated"]["ap50"]
    wu = results["diana ungated"]["wall"]
    wg = results["diana LIVE gated"]["wall"]
    print()
    print(f"  THE GATE'S PRICE : AP50 {ug*100:.2f}% -> {ga*100:.2f}%  = {(ga-ug)*100:+.2f} pp")
    print(f"  THE GATE'S SAVING: {wu:.1f}s -> {wg:.1f}s  = {wu/max(wg,1e-9):.2f}x")
    with open(os.path.join(seq, "bench.json"), "w") as f:
        json.dump(results, f, indent=1)
if __name__ == "__main__":
    main()