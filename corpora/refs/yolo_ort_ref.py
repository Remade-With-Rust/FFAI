"""Batch adapter: ONNX Runtime CPU over the exported YOLO26 model, for
`ffai bench detect`.

The deployment bar of the Diana mission (docs/diana-mission-plan.md section
5) — the path most edge users actually take: `yolo export format=onnx`, then
onnxruntime. The export is produced once, offline, by
`tools/diana_export_onnx.py`; this adapter refuses anything that is not the
natively end-to-end YOLO26 export (output `[1, N, 6]` — x0,y0,x1,y1,conf,cls
in letterboxed-input pixels). A raw-head export would need NMS glue that
belongs to the engine under test, not to a reference adapter — fail closed
rather than approximate.

Preprocessing reproduces Ultralytics' LetterBox: scale to fit, center-pad
with 114, /255, CHW float32. Detections are mapped back through the inverse
letterbox to ORIGINAL image pixels. Per-image timing includes preprocessing —
the official predictor's per-image timing does too, so the comparison is
like-for-like.

stdout JSONL: same contract as ultralytics_ref.py.
"""

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="file with one image path per line")
    ap.add_argument("--model", required=True, help="path to the exported .onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-dets", type=int, default=100)
    args = ap.parse_args()

    # utf-8-sig: a filelist written by PowerShell redirection carries a BOM,
    # which would otherwise reach the image loader as part of the first path.
    with open(args.batch, encoding="utf-8-sig") as f:
        paths = [line.strip() for line in f if line.strip()]

    t0 = time.perf_counter()
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    (out_meta,) = sess.get_outputs()
    if len(out_meta.shape) != 3 or out_meta.shape[2] != 6:
        raise SystemExit(
            f"model output {out_meta.shape} is not the end-to-end [1,N,6] export — "
            "re-export with tools/diana_export_onnx.py"
        )
    input_name = sess.get_inputs()[0].name
    size = args.imgsz

    def letterbox(img: "Image.Image"):
        w0, h0 = img.size
        r = min(size / w0, size / h0)
        nw, nh = round(w0 * r), round(h0 * r)
        left = round((size - nw) / 2 - 0.1)
        top = round((size - nh) / 2 - 0.1)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
        x = np.asarray(canvas, dtype=np.float32) / 255.0
        return x.transpose(2, 0, 1)[None], r, left, top, w0, h0

    def run(path: str):
        img = Image.open(path).convert("RGB")
        x, r, left, top, w0, h0 = letterbox(img)
        (y,) = sess.run(None, {input_name: x})
        rows = []
        dets = y[0]
        dets = dets[dets[:, 4].argsort()[::-1]]
        for x0, y0, x1, y1, conf, cls in dets:
            if conf < args.conf:
                continue
            rows.append(
                [
                    round(max(0.0, min(float((x0 - left) / r), w0)), 2),
                    round(max(0.0, min(float((y0 - top) / r), h0)), 2),
                    round(max(0.0, min(float((x1 - left) / r), w0)), 2),
                    round(max(0.0, min(float((y1 - top) / r), h0)), 2),
                    int(cls),
                    round(float(conf), 5),
                ]
            )
            if len(rows) >= args.max_dets:
                break
        return rows

    if paths:  # one untimed warm inference, per the adapter contract
        run(paths[0])
    print(json.dumps({"load_secs": time.perf_counter() - t0}), flush=True)

    for path in paths:
        t = time.perf_counter()
        rows = run(path)
        secs = time.perf_counter() - t
        print(
            json.dumps(
                {"path": path, "text": json.dumps(rows), "transcribe_secs": secs}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
