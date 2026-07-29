"""PARSeq-tiny forward oracle: OUR preprocessing, THEIR model (torch.hub).

Dumps the exact input tensor bytes + the reference's AR-greedy ids/text
(refine_iters=0 so both sides run the same pure AR path; refinement is a
recorded later brick), plus the hparams the port must not guess (decoder
head count, tokenizer specials).

Run: .venv-bench/Scripts/python tools/carmenta_parseq_oracle.py
"""

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "corpora" / "refs" / "fixtures"


def main():
    model = torch.hub.load("baudm/parseq", "parseq_tiny", pretrained=True, trust_repo=True).eval()
    inner = getattr(model, "model", model)  # newer strhub nests the module under .model
    for m in (model, inner):
        if hasattr(m, "refine_iters"):
            m.refine_iters = 0
    layer = inner.decoder.layers[0]
    info = {
        "dec_heads": layer.self_attn.num_heads,
        "enc_heads": inner.encoder.blocks[0].attn.num_heads,
        "max_label_length": getattr(model, "max_label_length", getattr(inner, "max_label_length", None)),
        "charset_itos": list(model.tokenizer._itos),
        "bos_id": model.tokenizer.bos_id,
        "eos_id": model.tokenizer.eos_id,
        "pad_id": model.tokenizer.pad_id,
    }
    print(json.dumps({k: v for k, v in info.items() if k != "charset_itos"}))

    from PIL import Image
    img = Image.open(FIX / "trocr_line.png").convert("RGB")
    # A single word crop reads best for a scene-text recognizer; take the
    # first ~"Clock" region, then the standard 32x128 bicubic resize.
    w, h = img.size
    word = img.crop((0, 0, min(w, int(h * 3.2)), h)).resize((128, 32), Image.BICUBIC)
    x = np.asarray(word, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5  # (32,128,3)
    chw = np.transpose(x, (2, 0, 1)).copy()
    chw.astype("<f4").tofile(FIX / "parseq_input_1x3x32x128_f32.bin")

    with torch.no_grad():
        logits = model(torch.from_numpy(chw)[None])  # (1, T, 95)
    ids = logits.argmax(-1)[0].tolist()
    text = model.tokenizer.decode(logits)[0][0]
    (FIX / "parseq_fixture.json").write_text(
        json.dumps({**info, "ids": ids, "text": text, "input": "parseq_input_1x3x32x128_f32.bin"}),
        encoding="utf-8",
    )
    print(f"oracle ids={ids[:12]}... text={text!r}")
    print("DONE")


if __name__ == "__main__":
    main()
