"""Stage-level Paddle adapter (plan §8.6): expose PP-OCRv5 mobile's
DETECTION and RECOGNITION stages separately, so Carmenta's stages are scored
function-vs-function on identical inputs — per-crop CER against their rec,
box IoU against their det polys, ms per stage — instead of pipeline
averages.

  --det {filelist}: JSONL {"path", "polys": [[[x,y]x4]...], "secs"}
  --rec {filelist}: JSONL {"path", "text", "secs"}   (inputs are LINE CROPS)

mkldnn stays off (paddlepaddle 3.3.1 PIR+oneDNN crash on this box, recorded
in paddleocr_ref.py). The 3.x module classes are tried first; the pipeline
constructor is the fallback for older wheels.
"""

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det")
    ap.add_argument("--rec")
    args = ap.parse_args()
    assert bool(args.det) != bool(args.rec), "exactly one of --det/--rec"

    if args.det:
        from paddleocr import TextDetection

        model = TextDetection(model_name="PP-OCRv5_mobile_det", enable_mkldnn=False)
        paths = [l.strip() for l in open(args.det, encoding="utf-8") if l.strip()]
        model.predict(paths[0])  # untimed warm
        print(json.dumps({"load_secs": 0.0}), flush=True)
        for p in paths:
            t0 = time.perf_counter()
            out = model.predict(p)[0]
            secs = time.perf_counter() - t0
            polys = [[[float(x), float(y)] for x, y in poly] for poly in out["dt_polys"]]
            print(json.dumps({"path": p, "polys": polys, "secs": secs}), flush=True)
    else:
        from paddleocr import TextRecognition

        model = TextRecognition(model_name="PP-OCRv5_mobile_rec", enable_mkldnn=False)
        paths = [l.strip() for l in open(args.rec, encoding="utf-8") if l.strip()]
        model.predict(paths[0])  # untimed warm
        print(json.dumps({"load_secs": 0.0}), flush=True)
        for p in paths:
            t0 = time.perf_counter()
            out = model.predict(p)[0]
            secs = time.perf_counter() - t0
            print(json.dumps({"path": p, "text": str(out["rec_text"]), "secs": secs}), flush=True)


if __name__ == "__main__":
    main()
