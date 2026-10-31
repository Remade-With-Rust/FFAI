"""Stage the mobiledet-svtr weights for a versioned public host. Uploads NOTHING.

Why: the `ppocrv5-mobile-det` and `ppocrv5-mobile-rec` manifests carry SHA-256
but no `hf_repo`, so `fetch` cannot download them and a crates.io user has no
way to get the document engine's weights (PdfCraft, 2026-10-30, finding #12).
`ffai-models` already downloads from a Hugging Face repo named in a manifest's
`hf_repo`, by file name, verifying each file against its pinned hash; what is
missing is the repo. Publishing weights is an owner's decision, so this stops at
a staging directory and prints the two commands that finish the job.

    python -I tools/carmenta_stage_weights.py OUT_DIR [HF_REPO]

OUT_DIR gets exactly the files the engine loads, each verified against the
crate's embedded manifest, plus a model card and the upstream licence notice.
HF_REPO (default `remade-with-rust/carmenta-ppocrv5-mobile`) is only used in the
printed commands and manifest lines.

The weights are PaddlePaddle's PP-OCRv5 mobile detector and recognizer
(Apache-2.0), converted by `carmenta_mobiledet_prepare.py` +
`carmenta_mobiledet_fuse.py` and `carmenta_svtr_prepare.py`. Redistribution
under Apache-2.0 requires the licence and attribution, which the card carries.
"""
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

MANIFESTS = Path("crates/ffai-carmenta/models")
CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "ffai" / "models"
# model -> the files the engine actually loads (engine.rs `fetch` calls).
FILES = {
    "ppocrv5-mobile-det": ["det-fused.safetensors"],
    "ppocrv5-mobile-rec": ["rec.safetensors", "charset.txt"],
}

CARD = """---
license: apache-2.0
tags: [ocr, text-detection, text-recognition, safetensors]
---

# Carmenta: PP-OCRv5 mobile (detection + SVTR recognition), safetensors

Weights for the `mobiledet-svtr` document engine of
[ffai-carmenta](https://crates.io/crates/ffai-carmenta), the pure-Rust OCR
engine of [FFAI](https://github.com/Remade-With-Rust/FFAI).

| file | model | sha256 |
|---|---|---|
{rows}

These are **PaddlePaddle's PP-OCRv5 mobile** text detector (DBNet / PP-LCNetV3)
and text recognizer (SVTR, 18 383 characters + blank), released by PaddleOCR
under the Apache License 2.0. They were converted to safetensors (the detector
with its re-parameterised branches and BatchNorms fused into plain
convolutions) and checked against PaddlePaddle's own exported programs. No
weights were retrained. Copyright for the original models remains with the
PaddlePaddle Authors; see `LICENSE`.

Use: `ffai_carmenta::engine::CraftCrnn::offline(RecStage::Svtr,
DetStage::MobileDet, root)` with `root/<model>/<file>`, or let the `fetch`
feature download them by the hashes pinned in the crate's manifests.
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned(model: str) -> dict:
    """file name -> sha256 from the crate's embedded manifest."""
    text = (MANIFESTS / f"{model}.toml").read_text(encoding="utf8")
    return dict(re.findall(r'name = "([^"]+)"\s*\nsha256 = "([0-9a-f]{64})"', text))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    repo = sys.argv[2] if len(sys.argv) > 2 else "remade-with-rust/carmenta-ppocrv5-mobile"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, files in FILES.items():
        pins = pinned(model)
        for name in files:
            src = CACHE / model / name
            want = pins.get(name)
            if want is None:
                sys.exit(f"{model}.toml pins no `{name}`")
            got = sha256(src)
            if got != want:
                sys.exit(f"{src}: sha256 {got} does not match the manifest's {want}; refusing to stage")
            shutil.copy2(src, out / name)
            rows.append(f"| `{name}` | {model} | `{want}` |")
            print(f"staged {name} ({src.stat().st_size:,} B), sha256 verified")
    (out / "README.md").write_text(CARD.format(rows="\n".join(rows)), encoding="utf8")
    (out / "LICENSE").write_text(
        "The model weights in this repository are derived from PaddleOCR's PP-OCRv5\n"
        "mobile detection and recognition models, Copyright (c) PaddlePaddle Authors,\n"
        "licensed under the Apache License, Version 2.0:\n"
        "https://www.apache.org/licenses/LICENSE-2.0\n"
        "Source: https://github.com/PaddlePaddle/PaddleOCR\n",
        encoding="utf8",
    )
    print(f"\nStaged in {out}. Nothing was uploaded. To publish (the owner's decision):\n")
    print(f"  huggingface-cli repo create {repo.split('/', 1)[1]} --organization {repo.split('/', 1)[0]} --type model")
    print(f"  huggingface-cli upload {repo} {out} . --commit-message \"PP-OCRv5 mobile det+rec for ffai-carmenta\"")
    print("\nthen add this line under `license = ...` in BOTH manifests")
    print("(crates/ffai-carmenta/models/ppocrv5-mobile-{det,rec}.toml and the copies in models/):\n")
    print(f'  hf_repo = "{repo}"')


if __name__ == "__main__":
    main()
