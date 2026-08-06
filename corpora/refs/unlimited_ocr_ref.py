"""Batch adapter for Baidu's Unlimited-OCR — the best-in-class document bar.

A 3B Mixture-of-Experts VLM (~500M active) released 2026-06-22 under MIT,
continue-trained from DeepSeek-OCR, which replaces decoder attention with
Reference Sliding Window Attention so the KV cache stays flat across a whole
document. It holds the OmniDocBench v1.5/v1.6 record. It is, on paper, the
thing to beat.

Contract (crates/ffai-bench/src/reference.rs): read a filelist, emit JSONL —
one {"load_secs"} line, then {"path", "text", "transcribe_secs"} per image.

## This is a CROSS-TIER comparison and the ledger must say so

Every other reference here is a conventional detector+recogniser measured on
CPU. This is a 3B generative model that wants a GPU. Against Carmenta's 4.7 MB
detector the quality comparison is meaningful and the speed/footprint
comparison is not apples-to-apples in either direction — it is the
`tesseract spawn tax` and `paddle mkldnn off` situation again, one tier up, and
it rides in the ledger notes rather than a footnote nobody reads.

Runs in its OWN virtualenv (`.venv-unlimited`). `.venv-bench` pins a CPU-only
torch that EasyOCR's reference depends on, and upgrading it in place to get
CUDA would put every existing OCR comparison at risk to add one new one.

Output is Markdown-ish document parsing, so the adapter strips the markup the
model emits for structure — headings, list bullets, table pipes — because the
corpora score plain text and leaving `##` in would charge the model for
formatting we never asked our own engines to produce.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROMPT = "<image>document parsing."


def strip_markup(md: str) -> str:
    """Markdown structure -> plain lines, preserving READING ORDER.

    Deliberately conservative: it removes markers, never reorders or merges.
    The whole reason to use a document-tier reference is that the order it
    produces is the thing under test.
    """
    out = []
    for raw in (md or "").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("```") or set(ln) <= set("-|: "):  # fences, table rules
            continue
        ln = re.sub(r"^#{1,6}\s*", "", ln)          # headings
        ln = re.sub(r"^[-*+]\s+", "- ", ln)          # bullets -> our own marker
        ln = re.sub(r"^\d+\.\s+", "", ln)            # ordered list numbers
        ln = ln.replace("**", "").replace("__", "")  # bold
        ln = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", ln)  # links/images
        ln = ln.strip("| ").replace("|", " ")        # table cells
        ln = re.sub(r"\s{2,}", " ", ln).strip()
        if ln:
            out.append(ln)
    return "\n".join(out)


def main():
    # The model PRINTS its own decoded output during `infer`, and Windows
    # defaults stdout to cp1252. One `ş`, `☆`, `−` or `○` on the page raises
    # UnicodeEncodeError inside the model, the except below catches it, and the
    # page is scored as EMPTY TEXT — i.e. 100 % CER.
    #
    # On the 236-page holdout that silently destroyed 59 of the reference's
    # pages (25 %) and would have handed us a 26 pp win that did not exist. The
    # reconfigure is here rather than in the caller's environment because a
    # reference must not depend on how it is invoked to report honest numbers.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--model", default="baidu/Unlimited-OCR")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base-size", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--max-length", type=int, default=32768)
    args = ap.parse_args()

    t0 = time.perf_counter()
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        # Stated, not silent: a CPU run is a different measurement entirely.
        print(f"WARNING: running on CPU (cuda available={torch.cuda.is_available()})",
              flush=True)
        model = model.float()
    print(json.dumps({"load_secs": time.perf_counter() - t0}), flush=True)

    outdir = Path(".tools-bench/unlimited_out")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(args.batch, encoding="utf-8") as fh:
        paths = [ln.strip() for ln in fh if ln.strip()]

    for i, p in enumerate(paths):
        t = time.perf_counter()
        # `infer` returns None and WRITES `result.md` into output_path — checked
        # with .tools-bench/unlimited_shape.py rather than assumed, after the
        # first adapter took its return value and scored a 3 GB model at 100 %
        # CER on empty strings. Each image gets its own directory so a failed
        # page cannot silently inherit the previous page's result.
        page_dir = outdir / f"{i:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        md = page_dir / "result.md"
        if md.exists():
            md.unlink()
        try:
            model.infer(
                tok, prompt=PROMPT, image_file=p, output_path=str(page_dir),
                base_size=args.base_size, image_size=args.image_size, crop_mode=True,
                max_length=args.max_length,
                no_repeat_ngram_size=35, ngram_window=128,
                save_results=True,
            )
            raw = md.read_text(encoding="utf-8", errors="replace") if md.exists() else ""
            if not raw:
                print(f"WARNING: {p}: no result.md written", flush=True)
            text = strip_markup(raw)
        except Exception as e:  # a reference that dies must not kill the run
            print(f"WARNING: {p}: {type(e).__name__}: {e}", flush=True)
            text = ""
        print(json.dumps({"path": p, "text": text,
                          "transcribe_secs": time.perf_counter() - t}), flush=True)


if __name__ == "__main__":
    main()
