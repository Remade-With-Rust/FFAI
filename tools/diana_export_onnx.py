"""Export the YOLO26 checkpoints in corpora/cache to ONNX (Diana M-D0).

One-time, offline, re-runnable — the export the ONNX Runtime reference
adapter (corpora/refs/yolo_ort_ref.py) consumes. YOLO26 is natively
end-to-end, so the default export already emits final detections
[1, 300, 6] (x0,y0,x1,y1,conf,cls in letterboxed-input pixels); the script
verifies that shape and refuses to leave a raw-head export on disk, because
the adapter would then need NMS glue that belongs to an engine, not a
reference.

Usage (from the repo root):
    .venv-diana/Scripts/python.exe tools/diana_export_onnx.py
"""

from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO

CACHE = Path(__file__).resolve().parent.parent / "corpora" / "cache"

for name in ("yolo26n", "yolo26s"):
    pt = CACHE / f"{name}.pt"
    if not pt.exists():
        raise SystemExit(f"missing {pt} — fetch it first (see diana-mission-plan.md §7.1)")
    out = YOLO(str(pt)).export(format="onnx", imgsz=640, device="cpu")
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    (meta,) = sess.get_outputs()
    if len(meta.shape) != 3 or meta.shape[2] != 6:
        Path(out).unlink()
        raise SystemExit(f"{name}: output {meta.shape} is not the end-to-end [1,N,6] export")
    print(f"{name}: {out} output {meta.shape} OK")
