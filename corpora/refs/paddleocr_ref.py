"""Batch adapter for PaddleOCR (Python + Paddle C++ core) — the accuracy bar
most OCR products actually use.

Contract (crates/ffai-bench/src/reference.rs): read a filelist, emit JSONL —
one {"load_secs"} line, then {"path", "text", "transcribe_secs"} per image.

Pipeline is pinned to BARE det+rec: document-orientation classify, unwarping,
and textline-orientation are all disabled explicitly, because the work being
compared is text detection + recognition, and PaddleOCR 3.x turns doc
preprocessing on by default. Leaving a reference's extra stages on while our
engine doesn't run them is the `-nt`-flag class of unfairness, in reverse.

Handles both the 3.x API (`predict`, result carries `rec_texts`) and the
2.x API (`ocr(path, cls=False)`), reporting whichever the installed version
speaks; the version lands in the ledger via version_command.
"""

import argparse
import json
import time


def extract_texts(result) -> list:
    """Pull recognized strings out of either API's result shape."""
    texts = []
    for item in result or []:
        # 3.x: OCRResult behaves like a dict with 'rec_texts'.
        try:
            rec = item["rec_texts"]
            texts.extend(str(t) for t in rec)
            continue
        except (TypeError, KeyError, IndexError):
            pass
        # 2.x: list of [box, (text, confidence)] per page.
        try:
            for line in item:
                texts.append(str(line[1][0]))
        except (TypeError, IndexError):
            pass
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--det-model", default=None, help="pin the detection model by name")
    ap.add_argument("--rec-model", default=None, help="pin the recognition model by name")
    # MEASURED CONSTRAINT, recorded 2026-07-29: paddlepaddle 3.3.1's PIR +
    # oneDNN executor crashes on this bench box (ConvertPirAttribute2Runtime-
    # Attribute, onednn_instruction.cc:118) for every model tried, so mkldnn
    # off is the only configuration that RUNS. That handicaps Paddle's CPU
    # speed and the ledger argv says so — revisit on each paddlepaddle
    # upgrade rather than letting the workaround fossilize.
    ap.add_argument("--mkldnn", choices=["on", "off"], required=True)
    args = ap.parse_args()

    with open(args.batch, encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    t0 = time.perf_counter()
    from paddleocr import PaddleOCR  # noqa: PLC0415 — inside the timed load on purpose

    kwargs = {
        "lang": args.lang,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": args.mkldnn == "on",
    }
    if args.det_model:
        kwargs["text_detection_model_name"] = args.det_model
    if args.rec_model:
        kwargs["text_recognition_model_name"] = args.rec_model

    try:  # 3.x
        ocr = PaddleOCR(**kwargs)
        api = "predict"
    except TypeError:  # 2.x signature
        ocr = PaddleOCR(lang=args.lang, use_angle_cls=False, show_log=False)
        api = "ocr"

    def run(path):
        if api == "predict":
            return ocr.predict(path)
        return ocr.ocr(path, cls=False)

    # One untimed warm inference — first-call lazy init stays out of frame 0.
    if paths:
        run(paths[0])
    print(json.dumps({"load_secs": time.perf_counter() - t0}), flush=True)

    for path in paths:
        t1 = time.perf_counter()
        result = run(path)
        elapsed = time.perf_counter() - t1
        text = "\n".join(extract_texts(result))
        print(json.dumps({"path": path, "text": text, "transcribe_secs": elapsed}), flush=True)


if __name__ == "__main__":
    main()
