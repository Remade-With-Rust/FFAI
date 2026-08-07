#!/usr/bin/env python3
"""Headful side-by-side: Diana (pure Rust) vs Ultralytics, on the same frames.

    python tools/diana_sbs_viewer.py --frames corpora/clips/mot17-09/img1
    python tools/diana_sbs_viewer.py --video clip.mp4 --live

Left pane is Diana, right is Ultralytics, both fed the identical frame in the
identical order. Overlays carry per-frame latency, a rolling frame rate, the
running median, and the object count; the footer carries the cumulative
comparison and how far the two engines' boxes agree.

WHAT THIS IS AND IS NOT A MEASUREMENT OF
========================================
It is a demonstration. `bench/ledger.jsonl` is the measurement, and where the
two disagree the ledger wins. Three things are different here:

* **The engines alternate; they never run concurrently.** Diana is sent a
  frame and its reply is read before Ultralytics is called, so each has the
  whole machine while the other is idle. Running both at once would have them
  contending for the same cores and the two latencies would each be measuring
  the other. This is the one fairness property the viewer does preserve, and
  it is why the numbers on screen are worth reading at all.
* **No min-of-N, no warm-up discipline beyond one frame.** The bench takes a
  p50 over a hash-pinned corpus after warm-up. Here every frame counts,
  including whatever the OS was doing during it. Expect more spread.
* **Drawing and imshow are outside both timed regions**, but they still cost
  wall clock, so the on-screen FPS is lower than either engine's own rate.
  The per-engine `fps` figures are derived from that engine's latency alone;
  the footer's `wall` is the real end-to-end rate including display.

Both engines decode inside their own timed region — Diana in `--serve`,
Ultralytics in `predict(path)`. That sentence was FALSE when this file was
first written: `--serve` had `load_image` sitting above `Instant::now()`, so
Diana's number excluded a decode that Ultralytics' included — 8.4 ms and 16 %
of the frame at 1080p, handed to us for free. It is fixed, and `--serve` now
reports `ms` / `detect_ms` / `decode_ms` split so neither half can hide again.
The claim had been written from intent rather than from the code.

The wall-clock numbers this viewer displays are also the WEAKER instrument.
On a loaded box the null arm — Diana against Diana — reads a 10.4 % floor with
up to 47 % within-arm spread, which is wider than most differences worth
caring about. `tools/diana_cpu_ab.py` is the admissible comparison (ABBA, CPU
time, both arms in child processes); this one is for watching, not deciding.
See docs/whys/diana-1080p-and-tail.md.

Keys: q quit · space pause · s save a screenshot · [ ] step while paused
"""

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from statistics import median

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FFAI = ROOT / "target" / "release" / ("ffai.exe" if sys.platform == "win32" else "ffai")

DIANA_BGR = (90, 220, 90)
ULTRA_BGR = (80, 160, 250)
WARN_BGR = (60, 200, 255)
BG = (28, 26, 24)


def palette(class_id):
    """Stable per-class colour so the two panes tint the same class alike."""
    rng = np.random.default_rng(class_id * 9781 + 17)
    c = rng.integers(70, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def agreement(da, db, thr=0.5):
    """How many of Diana's boxes have a same-class Ultralytics box over `thr`.

    Greedy and order-dependent, which is fine for a status line and would not
    be for a score — mAP is computed by `tools/diana_mot_bench.py`, not here.
    """
    used, hit = set(), 0
    for d in da:
        best, best_j = thr, None
        for j, u in enumerate(db):
            if j in used or u["class"] != d["class"]:
                continue
            v = iou((d["x0"], d["y0"], d["x1"], d["y1"]), (u["x0"], u["y0"], u["x1"], u["y1"]))
            if v >= best:
                best, best_j = v, j
        if best_j is not None:
            used.add(best_j)
            hit += 1
    return hit


def draw_boxes(img, dets, thickness=2):
    for d in dets:
        c = palette(d["class"])
        p0 = (int(d["x0"]), int(d["y0"]))
        p1 = (int(d["x1"]), int(d["y1"]))
        cv2.rectangle(img, p0, p1, c, thickness)
        label = f"{d['name']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(p0[1], th + 4)
        cv2.rectangle(img, (p0[0], ly - th - 4), (p0[0] + tw + 6, ly + 2), c, -1)
        cv2.putText(img, label, (p0[0] + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)


def panel(img, title, colour, ms, hist, n, badge=None):
    """Header strip: title, this frame's latency, the running median, count."""
    h, w = img.shape[:2]
    strip = np.full((78, w, 3), BG, np.uint8)
    cv2.rectangle(strip, (0, 0), (w, 4), colour, -1)
    cv2.putText(strip, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour, 2, cv2.LINE_AA)

    p50 = median(hist) if hist else 0.0
    fps = 1000.0 / ms if ms > 0 else 0.0
    txt = f"{ms:6.1f} ms   p50 {p50:5.1f}   {fps:5.1f} fps   {n} objects"
    cv2.putText(strip, txt, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (225, 225, 225), 1, cv2.LINE_AA)

    if badge:
        (tw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(strip, (w - tw - 26, 14), (w - 10, 46), WARN_BGR, -1)
        cv2.putText(strip, badge, (w - tw - 18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([strip, img])


def footer(w, i, total, dh, uh, agree, nd, wall_fps, live):
    bar = np.full((92, w, 3), BG, np.uint8)
    cv2.rectangle(bar, (0, 0), (w, 3), (70, 70, 70), -1)

    dm = median(dh) if dh else 0.0
    um = median(uh) if uh else 0.0
    if dm > 0 and um > 0:
        ratio = um / dm
        verdict = f"{ratio:.2f}x {'AHEAD' if ratio >= 1 else 'BEHIND'}"
        vc = DIANA_BGR if ratio >= 1 else WARN_BGR
    else:
        verdict, vc = "--", (200, 200, 200)

    cv2.putText(bar, f"frame {i}/{total}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(bar, f"median  diana {dm:5.1f} ms   ultralytics {um:5.1f} ms", (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(bar, verdict, (640, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, vc, 2, cv2.LINE_AA)
    cv2.putText(bar, f"box agreement {agree}/{nd}", (830, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(bar, f"wall {wall_fps:4.1f} fps", (1040, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)

    note = ("engines alternate, never concurrent - one at a time on the whole machine.  "
            "demo, not the benchmark: bench/ledger.jsonl is authoritative.")
    cv2.putText(bar, note, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    if live:
        cv2.putText(bar, "LIVE gate on: a gated frame reuses the previous boxes at zero model cost.",
                    (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WARN_BGR, 1, cv2.LINE_AA)
    else:
        cv2.putText(bar, "q quit   space pause   s screenshot", (12, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return bar


def frames_from_video(path, out_dir, stride):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    paths, i, kept = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            p = out_dir / f"{kept:06d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(p)
            kept += 1
        i += 1
    cap.release()
    print(f"extracted {kept} frames from {path.name} (every {stride})")
    return paths


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", type=Path, help="directory of frames, sorted by name")
    src.add_argument("--video", type=Path, help="a video file; frames are extracted first")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame from --video")
    ap.add_argument("--engine", default="yolo26n", help="Diana engine name")
    ap.add_argument("--weights", default="yolo26n.pt", help="Ultralytics checkpoint (AGPL, yours)")
    ap.add_argument("--conf", type=float, default=0.25)
    # Applied to BOTH engines or neither. Diana is an 80-class COCO detector and
    # so is the reference, but this campaign's MOT17 numbers were scored against
    # pedestrian-only ground truth while Diana was still emitting cars, buses
    # and traffic lights — 13.8-47.3 % of detections depending on the sequence.
    # Watching that side by side is the whole point of a viewer, so it is a flag
    # rather than a default, and it goes to both sides at once.
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="restrict BOTH engines to these class ids, e.g. --classes 0 for people")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames")
    ap.add_argument("--height", type=int, default=520, help="pane height in pixels")
    ap.add_argument("--live", action="store_true", help="put Diana's change gate in the loop")
    ap.add_argument("--fps", type=float, default=0.0, help="cap playback to this rate")
    ap.add_argument("--record", type=Path, help="also write the composed view to this .mp4")
    # Raised by DEFAULT. The spikes this viewer used to show were the SCHEDULER,
    # not either engine. Measured over 200 frames, identical work, identical CPU
    # time, only the priority class changed:
    #
    #   priority   p50     p99     max    cpu_s
    #   Normal    64.0   384.2   408.1     29.5
    #   High      64.7   100.5   113.4     30.0
    #
    # Same median, same CPU, a 3.8x better tail. On a box 56 % busy with
    # editors, browsers and neighbouring cargo builds, a 64 ms frame loses its
    # timeslice and lands at 400 ms — and a viewer that shows that is reporting
    # the desktop, not the engine. Raised on BOTH sides or it is a thumb on the
    # scale.
    ap.add_argument("--no-priority", action="store_true",
                    help="do NOT raise both engines to High (shows raw scheduler noise)")
    args = ap.parse_args()

    if not FFAI.exists():
        sys.exit(f"{FFAI} not found - cargo build -p ffai-cli --release")

    if args.video:
        paths = frames_from_video(args.video, ROOT / "corpora" / "clips" / f"_sbs_{args.video.stem}", args.stride)
    else:
        paths = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        sys.exit("no frames found")
    if args.limit:
        paths = paths[: args.limit]

    from ultralytics import YOLO

    print(f"loading ultralytics {args.weights} ...")
    model = YOLO(args.weights)
    ucls = args.classes if args.classes else None
    model.predict(str(paths[0]), verbose=False, conf=args.conf, classes=ucls)  # warm up outside the loop

    cmd = [str(FFAI), "detect", "--serve", "--engine", args.engine, "--conf", str(args.conf)]
    if args.classes:
        # repeatable flag, not a space-separated list
        for c in args.classes:
            cmd += ["--classes", str(c)]
    if args.live:
        cmd.append("--live")
    print(f"starting diana: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1, cwd=str(ROOT))
    hello = json.loads(proc.stdout.readline())
    assert hello.get("ready"), hello
    print("diana ready (model load is outside every timed region)")

    if not args.no_priority:
        # BOTH sides, or the raise is a thumb on the scale. Ultralytics runs
        # inside THIS process, so this process is the other arm.
        try:
            import psutil
            psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
            psutil.Process(proc.pid).nice(psutil.HIGH_PRIORITY_CLASS)
            print("both engines raised to High priority (--no-priority to disable)")
        except Exception as e:
            print(f"could not raise priority ({e}); tail will show scheduler noise")

    dh, uh = deque(maxlen=300), deque(maxlen=300)
    writer = None
    paused = False
    i = 0
    t_wall = time.perf_counter()
    wall_hist = deque(maxlen=30)

    win = "Diana (pure Rust)  vs  Ultralytics"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while i < len(paths):
        p = paths[i]
        frame = cv2.imread(str(p))
        if frame is None:
            i += 1
            continue

        # --- Diana. Its reply is read to completion before Ultralytics is
        # --- called, so the two never contend for cores.
        proc.stdin.write(f"{p.as_posix()}\n")
        proc.stdin.flush()
        rec = json.loads(proc.stdout.readline())
        if "error" in rec:
            print("diana:", rec["error"])
            i += 1
            continue
        d_ms, d_dets, gated = rec["ms"], rec["detections"], rec["gated"]

        # --- Ultralytics, now that Diana is idle.
        t = time.perf_counter()
        res = model.predict(str(p), verbose=False, conf=args.conf, classes=ucls)[0]
        u_ms = (time.perf_counter() - t) * 1000.0
        names = res.names
        u_dets = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            u_dets = [
                {"x0": float(b[0]), "y0": float(b[1]), "x1": float(b[2]), "y1": float(b[3]),
                 "class": int(c), "name": names.get(int(c), "?"), "conf": float(s)}
                for b, c, s in zip(xyxy, cls, cf)
            ]

        # A gated frame ran no model, so folding its ~0 ms into the median
        # would report a latency the engine never achieved on a frame it
        # actually processed. Counted in the badge, kept out of the statistic.
        if not gated:
            dh.append(d_ms)
        uh.append(u_ms)

        scale = args.height / frame.shape[0]
        size = (int(frame.shape[1] * scale), args.height)
        left, right = cv2.resize(frame, size), cv2.resize(frame, size)

        def scaled(dets):
            return [{**d, "x0": d["x0"] * scale, "y0": d["y0"] * scale,
                     "x1": d["x1"] * scale, "y1": d["y1"] * scale} for d in dets]

        draw_boxes(left, scaled(d_dets))
        draw_boxes(right, scaled(u_dets))

        lp = panel(left, "DIANA - pure Rust", DIANA_BGR, d_ms, dh, len(d_dets),
                   badge="GATED" if gated else None)
        rp = panel(right, "ULTRALYTICS - PyTorch", ULTRA_BGR, u_ms, uh, len(u_dets))
        gap = np.full((lp.shape[0], 4, 3), BG, np.uint8)
        top = np.hstack([lp, gap, rp])

        now = time.perf_counter()
        wall_hist.append(now - t_wall)
        t_wall = now
        wall_fps = 1.0 / (sum(wall_hist) / len(wall_hist)) if wall_hist else 0.0

        view = np.vstack([top, footer(top.shape[1], i + 1, len(paths), dh, uh,
                                      agreement(d_dets, u_dets), len(d_dets), wall_fps, args.live)])

        if args.record and writer is None:
            writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                     20.0, (view.shape[1], view.shape[0]))
        if writer is not None:
            writer.write(view)

        cv2.imshow(win, view)
        delay = 1 if args.fps <= 0 else max(1, int(1000 / args.fps))
        k = cv2.waitKey(0 if paused else delay) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("s"):
            out = ROOT / f"diana_sbs_{i:06d}.png"
            cv2.imwrite(str(out), view)
            print("wrote", out)
        if paused and k == ord("["):
            i = max(0, i - 2)
        i += 1

    proc.stdin.close()
    proc.wait(timeout=5)
    if writer is not None:
        writer.release()
        print("wrote", args.record)
    cv2.destroyAllWindows()

    if dh and uh:
        dm, um = median(dh), median(uh)
        print(f"\n{'':>14}{'median':>10}{'mean':>10}{'frames':>9}")
        print(f"{'diana':>14}{dm:9.1f}m{sum(dh)/len(dh):9.1f}m{len(dh):9}")
        print(f"{'ultralytics':>14}{um:9.1f}m{sum(uh)/len(uh):9.1f}m{len(uh):9}")
        print(f"\n  {um/dm:.2f}x {'ahead' if um >= dm else 'behind'} on the median of this run.")
        print("  One run, no warm-up discipline, display in the loop.")
        print("  bench/ledger.jsonl is the measurement; this is a demo.")


if __name__ == "__main__":
    main()
