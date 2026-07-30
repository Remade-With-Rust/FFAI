"""carmenta-cord-v1: the REAL-photo claims corpus (plan §6.2 claims tier).

CORD-v2 (naver-clova-ix, CC-BY-4.0, ungated — audit §7.1): photographed
receipts with word-level ground truth. This takes the VALIDATION split's
first N images, writes PNGs + flat-text GT (all words of every valid_line
in reading order), and pins a hash manifest. Real lighting, paper warp,
camera noise — the distribution the synthetic tier cannot test.

GT construction: CORD's `ground_truth.valid_line[].words[].text`, lines
joined with newlines. Mode::Ocr scoring collapses whitespace, so line
placement differences don't score; receipts are single-column, so reading
order is fair. Claims stay holdout-only: first 15 clips are train (tuning),
the remaining 45 holdout.

Run: .venv-bench/Scripts/python tools/carmenta_cord_corpus.py
"""

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "corpora" / "clips" / "carmenta-cord"
URL = ("https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/"
       "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet")
N = 60


def main():
    CLIPS.mkdir(parents=True, exist_ok=True)
    pf = CLIPS / "_validation.parquet"
    if not pf.exists():
        print("downloading validation parquet ...", flush=True)
        urllib.request.urlretrieve(URL, pf)
    table = pq.read_table(pf).slice(0, N)
    images = table.column("image").to_pylist()
    gts = table.column("ground_truth").to_pylist()

    man = ['name = "carmenta-cord"', "version = 1", 'task = "ocr"']
    kept = 0
    for i, (img, gt) in enumerate(zip(images, gts)):
        parsed = json.loads(gt)
        lines = []
        for line in parsed.get("valid_line", []):
            words = [w.get("text", "") for w in line.get("words", [])]
            text = " ".join(w for w in words if w)
            if text:
                lines.append(text)
        if not lines:
            continue
        # Parquet embeds the original JPEG/PNG bytes; keep them verbatim
        # (transcoding real photos would alter the pixels we pin).
        blob = img["bytes"]
        ext = "png" if blob[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
        name = f"cord-{kept:03}.{ext}"
        (CLIPS / name).write_bytes(blob)
        (CLIPS / f"cord-{kept:03}.txt").write_text("\n".join(lines), encoding="utf-8")
        sha = hashlib.sha256(blob).hexdigest()
        split = "train" if kept < 15 else "holdout"
        man += ["", "[[clips]]", f'id = "cord-{kept:03}"',
                f'path = "clips/carmenta-cord/{name}"',
                f'ground_truth = "clips/carmenta-cord/cord-{kept:03}.txt"',
                'class = "photo"', f'split = "{split}"',
                'license = "CC-BY-4.0 (CORD-v2, naver-clova-ix)"', f'sha256 = "{sha}"']
        kept += 1
    (REPO / "corpora" / "carmenta-cord-v1.toml").write_text("\n".join(man), encoding="utf-8")
    print(f"kept {kept} receipts -> corpora/carmenta-cord-v1.toml")


if __name__ == "__main__":
    main()
