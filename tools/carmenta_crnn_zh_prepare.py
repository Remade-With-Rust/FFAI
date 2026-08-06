"""Convert EasyOCR's zh_sim_g2 CRNN to safetensors + dump a stage oracle (§8.143).

§8.142 measured 11 holdout pages of 236 as **51 % of the entire competitive gap**
against Unlimited-OCR, for one reason: they contain Chinese, and our recognizer
is `english_g2` with a 96-character Latin charset. A CTC head that has no Chinese
class cannot emit one — every CJK character is a guaranteed substitution. That is
not a threshold to tune, it is an alphabet we do not have.

**The port is small because the architecture is identical.** Both checkpoints
carry the same 44 tensors under the same names; only the CTC head differs:

    module.Prediction.weight    en (97, 256)  ->  zh (6719, 256)
    module.Prediction.bias      en (97,)      ->  zh (6719,)

So `crnn.rs` runs it unchanged. What ships is a different safetensors and a
bigger charset, nothing else.

**And the charset covers Latin.** EasyOCR builds `zh_sim_g2` from `symbols` plus
`characters`, and the latter contains the full ASCII range — `ABC..XYZ`,
`abc..xyz`, digits and punctuation — followed by 6 614 CJK characters from
`character/ch_sim_char.txt`. 6 718 + 1 CTC blank = 6 719, matching the head
exactly. So this model can read English pages too, which is what makes a single
runtime switch possible instead of a routing problem.

**Whether it reads English AS WELL is the open question, and the reason this
ships OFF.** A 6 719-class head is normally weaker on Latin than a 97-class
specialist, and trading a point of English CER for 1.02 pp of CJK would be a bad
bargain. `FFAI_REC_LANG=zh` selects it; the default stays `english_g2`, which
also keeps the English model as the oracle — the same shape as
`--features mimalloc` and `FFAI_CONV3X3=0`.

The charset is written next to the weights rather than compiled in: 6 718
characters is 20 KB of UTF-8, and a generated `const` that large is a merge
hazard for no benefit.

Run: .venv-bench/Scripts/python tools/carmenta_crnn_zh_prepare.py
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "ffai" / "models"
FIXTURES = REPO / "corpora" / "refs" / "fixtures"
PTH = Path.home() / ".EasyOCR" / "model" / "zh_sim_g2.pth"
OUT_DIR = CACHE / "crnn-zh-sim-g2"


def easyocr_charset() -> str:
    """The exact `zh_sim_g2` charset, read from EasyOCR rather than retyped.

    Taken from the imported module, not by parsing `config.py`: the field
    contains escaped quotes and backslashes, and a regex over the source got it
    wrong on the first attempt. Importing gets the string Python already built.

    The `characters` field is COMPLETE — 95 ASCII, 6 614 CJK and 9 other, 6 718
    total, which with the CTC blank is the 6 719 classes the head has. Nothing
    needs concatenating, and the check below refuses to proceed if that ever
    stops being true: an off-by-one here silently shifts EVERY decoded
    character, which is §8.113's failure in a form far harder to notice.
    """
    from easyocr.config import recognition_models

    return recognition_models["gen2"]["zh_sim_g2"]["characters"]


def ctc_greedy(logits, charset):
    ids = logits.argmax(-1).tolist()
    out, prev = [], 0
    for i in ids:
        if i != 0 and i != prev:
            out.append(charset[i - 1])
        prev = i
    return ids, "".join(out)


def main():
    if not PTH.exists():
        sys.exit(f"  missing {PTH} — fetch zh_sim_g2.zip from the EasyOCR v1.3 release")
    charset = easyocr_charset()
    state = torch.load(PTH, map_location="cpu", weights_only=True)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    n_class = state["Prediction.weight"].shape[0]
    print(f"  charset {len(charset)} chars   head {n_class} classes")
    if len(charset) + 1 != n_class:
        sys.exit(f"  MISMATCH: charset+blank = {len(charset) + 1} but the head has {n_class}. "
                 "Decoding would be shifted on every character.")

    from easyocr.model.vgg_model import Model
    model = Model(input_channel=1, output_channel=256, hidden_size=256, num_class=n_class)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("  zh_sim_g2 loaded strict=True")

    from safetensors.torch import save_file
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    st = OUT_DIR / "crnn.safetensors"
    save_file(state, str(st))
    (OUT_DIR / "charset.txt").write_text(charset, encoding="utf-8")
    print(f"  wrote {st}")
    print(f"    sha256={hashlib.sha256(st.read_bytes()).hexdigest()}")
    print(f"  wrote {OUT_DIR / 'charset.txt'} ({len(charset)} chars)")

    # ---- stage oracle: OUR preprocessing, THEIR net ----------------------
    # Same contract as the English tool: we preprocess the crop ourselves, dump
    # the exact tensor, and record the ids the torch model produces from those
    # bytes. The Rust side must reproduce the ids, not merely the text — a text
    # match with different ids means the charset is misaligned somewhere that
    # happens not to show on this line.
    from PIL import Image
    img = Image.open(FIXTURES / "trocr_line.png").convert("L")
    w, h = img.size
    new_w = max(8, round(w * 64 / h))
    img = img.resize((new_w, 64), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x.astype("<f4").tofile(FIXTURES / f"crnn_zh_input_1x1x64x{new_w}_f32.bin")
    with torch.no_grad():
        logits = model(torch.from_numpy(x)[None, None], None)[0]
    ids, text = ctc_greedy(logits, charset)
    (FIXTURES / "crnn_zh_fixture.json").write_text(
        json.dumps({"input": f"crnn_zh_input_1x1x64x{new_w}_f32.bin", "width": new_w,
                    "ids": ids, "text": text, "n_class": n_class,
                    "charset_len": len(charset)}, ensure_ascii=False),
        encoding="utf-8")
    print(f"  zh oracle text: {text!r}")
    print("DONE")


if __name__ == "__main__":
    main()
