"""Batch adapter for PP-StructureV3 — the DOCUMENT-tier bar.

The line-level references (tesseract, easyocr, paddleocr) answer "did you read
the characters". A document reference answers a different question: did you
read them IN THE RIGHT ORDER, having first worked out that the page has two
columns, a running header, and a footer that is not part of the prose. M-C3's
exit gate names PP-Structure for exactly that reason, and until now nothing
implemented it.

Contract (crates/ffai-bench/src/reference.rs): read a filelist, emit JSONL —
one {"load_secs"} line, then {"path", "text", "transcribe_secs"} per image.

Emitting text in READING ORDER is the whole point, so the adapter takes
PP-Structure's ordered parsing result rather than re-sorting boxes itself. If
it ever falls back to raster order the comparison quietly becomes a line-level
one, so the fallback is loud.

Handicaps are pinned and stated, as with the paddleocr adapter: oneDNN is off
because paddlepaddle 3.3.1's PIR executor crashes on this box (§7.1), and the
model set is the mobile tier so the comparison stays at a matched size.
"""

import argparse
import json
import time


def texts_in_reading_order(result) -> list:
    """Pull PP-Structure's ordered text out, without re-sorting it ourselves.

    Shape, read off the object rather than guessed (`.tools-bench/ppstruct_shape.py`):
    a page result is dict-like and carries `parsing_res_list`, a list of
    `LayoutBlock` OBJECTS — attribute access, not subscript — each with
    `.content` and a `.label` such as `header` / `text` / `paragraph_title`.
    The list is already in reading order, which is the thing being compared.
    """
    out = []
    for page in result or []:
        try:
            blocks = page["parsing_res_list"]
        except (TypeError, KeyError, IndexError):
            blocks = None
        if blocks:
            for b in blocks:
                t = getattr(b, "content", None)
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
            continue
        # Loud, not silent: falling back to unordered text turns a
        # document-tier comparison into a line-level one.
        print("WARNING: no ordered parsing result; falling back to rec_texts", flush=True)
        try:
            out.extend(str(t) for t in page["overall_ocr_res"]["rec_texts"])
        except (TypeError, KeyError, IndexError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--mkldnn", default="off")
    args = ap.parse_args()

    t0 = time.perf_counter()
    from paddleocr import PPStructureV3

    kwargs = {
        "lang": args.lang,
        # Matched-tier: the mobile det+rec pair the other adapter pins.
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": "PP-OCRv5_mobile_rec",
        # Off for the same reason the OCR adapter turns them off — the work
        # being compared is layout + read, not document preprocessing.
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        # Tables and formulas are separate milestones (M-C3's table work and
        # M-C5); leaving them on would compare stages we do not run.
        "use_table_recognition": False,
        "use_formula_recognition": False,
        "use_seal_recognition": False,
    }
    if args.mkldnn == "off":
        kwargs["enable_mkldnn"] = False
    pipe = PPStructureV3(**kwargs)
    print(json.dumps({"load_secs": time.perf_counter() - t0}), flush=True)

    with open(args.batch, encoding="utf-8") as fh:
        paths = [ln.strip() for ln in fh if ln.strip()]

    for p in paths:
        t = time.perf_counter()
        try:
            res = pipe.predict(p)
            text = "\n".join(texts_in_reading_order(res))
        except Exception as e:  # a reference that dies must not kill the run
            print(f"WARNING: {p}: {e}", flush=True)
            text = ""
        print(json.dumps({"path": p, "text": text,
                          "transcribe_secs": time.perf_counter() - t}), flush=True)


if __name__ == "__main__":
    main()
