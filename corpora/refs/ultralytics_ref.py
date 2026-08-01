"""Batch adapter: official Ultralytics Python inference, for `ffai bench detect`.

The parity oracle of the Diana mission (docs/diana-mission-plan.md section 5):
the official PyTorch path running the official checkpoint, invoked once over
the whole corpus per the batch contract in crates/ffai-bench/src/reference.rs.

stdout JSONL:
  {"load_secs": s}                      once, after import + model load + one
                                        untimed warm inference
  {"path", "text", "transcribe_secs"}   per image, where `text` is a JSON
                                        detections payload
                                        [[x0,y0,x1,y1,cls,conf], ...] in
                                        ORIGINAL image pixels, parsed by
                                        crates/ffai-bench/src/detect.rs

Every knob that changes the work is an explicit required-or-defaulted
argument so the exact configuration lands in the ledger's argv line.
`--conf 0.001 --max-dets 100` are the COCO-eval settings: mAP needs the
low-confidence tail, and maxDets=100 matches the scorer's truncation.

**`--rect` is this vertical's `beam_size`, and it is not optional.**
Ultralytics' `predict()` defaults `rect=True` (engine/model.py), which
letterboxes each image to the smallest multiple-of-32 RECTANGLE — a
586x640 image is fed as 640x608, not 640x640. The ONNX export is fixed
square. Left unpinned, the .pt and ORT rows of the same tier silently run
different input geometry and their mAP disagrees by 1.5-1.8 pp in
inconsistent directions (measured, M-D0). This flag makes the geometry an
explicit argv choice on both sides, exactly as beam_size was pinned across
the ASR references after the same class of defect.
"""

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="file with one image path per line")
    ap.add_argument("--model", required=True, help="path to the .pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-dets", type=int, default=100)
    ap.add_argument(
        "--rect",
        required=True,
        choices=("on", "off"),
        help="on = Ultralytics' own default (multiple-of-32 rectangle); "
        "off = square imgsz x imgsz, matching the ONNX export",
    )
    args = ap.parse_args()
    rect = args.rect == "on"

    # utf-8-sig: a filelist written by PowerShell redirection carries a BOM,
    # which would otherwise reach the loader as part of the first path.
    with open(args.batch, encoding="utf-8-sig") as f:
        paths = [line.strip() for line in f if line.strip()]

    t0 = time.perf_counter()
    from ultralytics import YOLO

    model = YOLO(args.model)
    if paths:  # one untimed warm inference, per the adapter contract
        model.predict(
            paths[0],
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_dets,
            rect=rect,
            device="cpu",
            verbose=False,
        )
    print(json.dumps({"load_secs": time.perf_counter() - t0}), flush=True)

    for path in paths:
        t = time.perf_counter()
        result = model.predict(
            path,
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_dets,
            rect=rect,
            device="cpu",
            verbose=False,
        )[0]
        secs = time.perf_counter() - t

        rows = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            order = conf.argsort()[::-1][: args.max_dets]
            for i in order:
                rows.append(
                    [
                        round(float(xyxy[i][0]), 2),
                        round(float(xyxy[i][1]), 2),
                        round(float(xyxy[i][2]), 2),
                        round(float(xyxy[i][3]), 2),
                        int(cls[i]),
                        round(float(conf[i]), 5),
                    ]
                )
        print(
            json.dumps(
                {"path": path, "text": json.dumps(rows), "transcribe_secs": secs}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
