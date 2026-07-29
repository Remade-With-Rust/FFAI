"""Convert EasyOCR's english_g2 CRNN to safetensors + dump a stage oracle.

The recognition pivot recorded in the M-C1 log: trocr-small-printed is
case-collapsed (SROIE-trained; the oracle fixture proved it before any port
work), so recognition is the EasyOCR g2 CRNN — mixed-case 96-char charset,
Apache-2.0, weights already fetchable ungated from JaidedAI's GitHub
releases, and the same lineage as our CRAFT detector.

The oracle here isolates the NET, not the pipeline: we preprocess a line
crop OURSELVES (grey -> h=64 bilinear -> (x/255 - 0.5)/0.5), dump that exact
tensor, run EasyOCR's torch model on it, and record the CTC-greedy ids +
text. The Rust side must reproduce the ids from the same tensor bytes.

Run: .venv-bench/Scripts/python tools/carmenta_crnn_prepare.py
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "ffai" / "models"
FIXTURES = REPO / "corpora" / "refs" / "fixtures"
PTH = Path.home() / ".EasyOCR" / "model" / "english_g2.pth"

CHARSET = (
    "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


def ctc_greedy(logits, charset):
    ids = logits.argmax(-1).tolist()
    out, prev = [], 0
    for i in ids:
        if i != 0 and i != prev:
            out.append(charset[i - 1])
        prev = i
    return ids, "".join(out)


def main():
    from easyocr.model.vgg_model import Model

    state = torch.load(PTH, map_location="cpu")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model = Model(input_channel=1, output_channel=256, hidden_size=256, num_class=len(CHARSET) + 1)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("english_g2 loaded strict=True", flush=True)

    from safetensors.torch import save_file
    out_dir = CACHE / "crnn-english-g2"
    out_dir.mkdir(parents=True, exist_ok=True)
    st = out_dir / "crnn.safetensors"
    save_file(state, str(st))
    print(f"wrote {st} sha256={hashlib.sha256(st.read_bytes()).hexdigest()}", flush=True)

    # ---- stage oracle: our preprocessing, their net ----
    from PIL import Image
    img = Image.open(FIXTURES / "trocr_line.png").convert("L")
    w, h = img.size
    new_w = max(8, round(w * 64 / h))
    img = img.resize((new_w, 64), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x.astype("<f4").tofile(FIXTURES / f"crnn_input_1x1x64x{new_w}_f32.bin")

    with torch.no_grad():
        logits = model(torch.from_numpy(x)[None, None], None)[0]
    ids, text = ctc_greedy(logits, CHARSET)
    (FIXTURES / "crnn_fixture.json").write_text(
        json.dumps({"input": f"crnn_input_1x1x64x{new_w}_f32.bin", "width": new_w,
                    "ids": ids, "text": text}),
        encoding="utf-8",
    )
    print(f"crnn oracle text: {text!r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
