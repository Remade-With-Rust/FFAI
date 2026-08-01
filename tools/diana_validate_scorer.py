"""Cross-validate ffai-bench's mAP proxy against pycocotools (Diana M-D0).

**Why this exists.** `crates/ffai-bench/src/detect.rs` implements COCO-style
mAP from scratch — a new scorer, on a new corpus, produced by the same hand
that wrote the thing it will eventually judge. Carmenta shipped two instrument
defects that each produced impossible numbers (a holdout misalignment scoring
every receipt against a different receipt's GT, and a baseline config
mismatch), and the lesson recorded there is: cross-validate a new scorer
against a known-good implementation BEFORE trusting anything it says.

This script scores the SAME detections JSONL two ways — through pycocotools'
COCOeval and through the Rust scorer (via `cargo run --example
score_detect`) — and reports the delta. The tolerance is 0.005 absolute on
both mAP@0.5 and mAP@0.5:0.95, which is far tighter than any claim the M-D0
board makes and loose enough for the one documented difference between the
implementations (pycocotools' area-range and crowd-ignore machinery, which
the corpus is built to keep inert — see tools/diana_coco_corpus.py).

Usage (from the repo root, after a bench run has produced a dump):
    .venv-diana/Scripts/python.exe tools/diana_validate_scorer.py \
        --corpus corpora/diana-coco-v1.toml --dets dets-yolo26n.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOLERANCE = 0.005


def parse_manifest(path: Path):
    """Minimal TOML read: the [[clips]] array, holdout entries only."""
    clips, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "[[clips]]":
            current = {}
            clips.append(current)
        elif current is not None and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            current[k.strip()] = v.strip().strip('"')
    return [c for c in clips if c.get("split") == "holdout"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dets", required=True, help="adapter JSONL dump")
    args = ap.parse_args()

    corpus_path = ROOT / args.corpus if not Path(args.corpus).is_absolute() else Path(args.corpus)
    dets_path = ROOT / args.dets if not Path(args.dets).is_absolute() else Path(args.dets)
    holdout = parse_manifest(corpus_path)
    base = corpus_path.parent

    by_path = {}
    for line in dets_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        if "path" in obj and "text" in obj:
            by_path[obj["path"].replace("\\", "/")] = json.loads(obj["text"])

    # Build COCO-format GT + detections over the holdout only.
    images, annotations, coco_dets = [], [], []
    ann_id = 1
    classes = sorted(
        {
            int(o[4])
            for clip in holdout
            for o in json.loads((base / clip["ground_truth"]).read_text(encoding="utf-8"))[
                "objects"
            ]
        }
    )
    for img_id, clip in enumerate(holdout, start=1):
        gt = json.loads((base / clip["ground_truth"]).read_text(encoding="utf-8"))
        images.append({"id": img_id, "width": gt["width"], "height": gt["height"]})
        for x0, y0, x1, y1, cls in gt["objects"]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        key = str((base / clip["path"]).as_posix())
        rows = by_path.get(key)
        if rows is None:
            match = [v for k, v in by_path.items() if k.endswith(clip["path"])]
            rows = match[0] if match else []
        # maxDets=100 by confidence, matching the Rust scorer's truncation.
        for x0, y0, x1, y1, cls, conf in sorted(rows, key=lambda r: -r[5])[:100]:
            coco_dets.append(
                {
                    "image_id": img_id,
                    "category_id": int(cls),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": conf,
                }
            )

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_json = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c, "name": str(c)} for c in classes],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_json
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(coco_dets) if coco_dets else None
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ref_5095, ref_50 = float(ev.stats[0]), float(ev.stats[1])

    print("running the Rust scorer over identical inputs ...")
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "-p",
            "ffai-bench",
            "--example",
            "score_detect",
            "--",
            str(corpus_path),
            str(dets_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"score_detect failed:\n{proc.stderr}")
    m = re.search(r"mAP50 ([0-9.]+)\s+mAP50-95 ([0-9.]+)", proc.stdout)
    if not m:
        sys.exit(f"could not parse score_detect output:\n{proc.stdout}")
    ours_50, ours_5095 = float(m.group(1)), float(m.group(2))

    print()
    print(f"{'metric':<12} {'pycocotools':>12} {'ffai-bench':>12} {'delta':>10}")
    ok = True
    for label, ref, ours in (
        ("mAP@0.5", ref_50, ours_50),
        ("mAP@0.5:0.95", ref_5095, ours_5095),
    ):
        delta = ours - ref
        print(f"{label:<12} {ref:>12.4f} {ours:>12.4f} {delta:>+10.4f}")
        ok = ok and abs(delta) <= TOLERANCE
    print()
    if ok:
        print(f"PASS — both within {TOLERANCE} absolute of pycocotools")
    else:
        sys.exit(f"FAIL — scorer disagrees with pycocotools by more than {TOLERANCE}")


if __name__ == "__main__":
    main()
