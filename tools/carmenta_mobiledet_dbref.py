"""Run paddle's OWN DBPostProcess on the pinned probability map, so the Rust
reimplementation is checked against the reference rather than against my
reading of it.

`mobiledet.rs` reimplements the whole postprocess — connected components,
minimum-area rectangle by rotating calipers, the Vatti unclip that undoes DB's
training-time shrink. Reading `processors.py` is how that gets nearly right;
this is how it gets *verified*. Boxes are reduced to axis-aligned bounds
because that is what the port emits (see `boxes_from_probability` for why).

Separate from `carmenta_mobiledet_oracle.py` because importing paddlex drags in
torch, and torch's DLLs will not load in a process that has already initialised
the paddle inference runtime.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

FIX = Path(__file__).resolve().parent.parent / "corpora" / "refs" / "fixtures"
THRESH, BOX_THRESH, UNCLIP = 0.3, 0.6, 1.5


def main():
    from paddlex.inference.models.text_detection.processors import DBPostProcess

    prob = load_file(str(FIX / "mobiledet_oracle_prob.safetensors"))["prob"].astype(np.float32)
    pred = prob[0, 0]
    h, w = pred.shape

    db = DBPostProcess(thresh=THRESH, box_thresh=BOX_THRESH, unclip_ratio=UNCLIP, score_mode="fast")
    quads, scores = db.boxes_from_bitmap(pred, pred > THRESH, w, h, BOX_THRESH, UNCLIP)

    boxes = sorted(
        ([int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max()),
          round(float(s), 4)] for q, s in zip(quads, scores)),
        key=lambda b: (b[1], b[0]))
    (FIX / "mobiledet_oracle_boxes.json").write_text(json.dumps({
        "thresh": THRESH, "box_thresh": BOX_THRESH, "unclip_ratio": UNCLIP,
        "map": [h, w],
        "note": "axis-aligned bounds of paddle DBPostProcess quads on the pinned crop",
        "boxes": boxes,
    }, indent=1), encoding="utf-8")
    print(f"paddle DBPostProcess: {len(boxes)} boxes on the {w}x{h} crop")
    for b in boxes[:8]:
        print("   ", b)


if __name__ == "__main__":
    main()
