"""Goal-3 prep: PARSeq-tiny (baudm/parseq, Apache-2.0) — the audit-cleared
successor to the g2 CRNN whose sentence-period ceiling holds the M-C1
quality gate open.

This is the de-risk slice, not the port: download the release checkpoint
(direct GitHub asset, no account), convert to safetensors into the FFai
cache, and dump every parameter's shape to a JSON map so the candle port is
written against facts instead of a paper figure — the exact lesson of the
CRAFT upconv widths. The forward-pass oracle fixture requires the strhub
model code and is the port session's first step, recorded as such.

Run: .venv-bench/Scripts/python tools/carmenta_parseq_prepare.py
"""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import torch

CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "ffai" / "models" / "parseq-tiny"
REPO = Path(__file__).resolve().parent.parent

# 94-char charset from strhub's `charset_train` for the released English
# models: digits, lowercase, uppercase, punctuation — everything Mode::Ocr
# scores, which is why this lineage is the successor.
CHARSET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    api = "https://api.github.com/repos/baudm/parseq/releases"
    with urllib.request.urlopen(api) as r:
        releases = json.load(r)
    asset = next(
        a
        for rel in releases
        for a in rel.get("assets", [])
        if a["name"].startswith("parseq_tiny") and a["name"].endswith(".pt")
    )
    print(f"asset: {asset['name']} ({asset['size']/1e6:.1f} MB)")
    pt = CACHE / asset["name"]
    if not pt.exists():
        urllib.request.urlretrieve(asset["browser_download_url"], pt)

    state = torch.load(pt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

    shapes = {k: list(v.shape) for k, v in state.items()}
    (REPO / "corpora/refs/fixtures/parseq_tiny_shapes.json").write_text(
        json.dumps({"charset": CHARSET, "shapes": shapes}, indent=1), encoding="utf-8"
    )
    print(f"{len(shapes)} tensors; shape map -> corpora/refs/fixtures/parseq_tiny_shapes.json")

    from safetensors.torch import save_file
    st = CACHE / "parseq-tiny.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(st))
    print(f"wrote {st} sha256={hashlib.sha256(st.read_bytes()).hexdigest()}")
    print("DONE — forward oracle fixture is the port session's first step (needs strhub code)")


if __name__ == "__main__":
    main()
