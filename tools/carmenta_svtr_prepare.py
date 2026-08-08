"""PP-OCRv5_mobile_rec (SVTR) -> shape map + safetensors + charset (§8.165).

Same route as `carmenta_mobiledet_prepare.py` took for the DETECTOR half of this
release: paddle loads the inference program, we dump every parameter, and the
candle port is written against CHECKPOINT FACTS rather than a paper (the CRAFT
widths lesson). Nothing here touches the shipped engine — it writes model files
and a shape map, and the port is a new module behind a new engine name.

Justified by the §8.165 ceiling probe BEFORE any Rust: with detection held
constant and only the recognizer swapped, SVTR reads our hardest stylized pages
at 40.12 % against CRNN's 44.13 % (-4.01 pp) AND the clean controls at 2.66 %
against 3.20 % (-0.53 pp). It wins on both halves, so the 18,383-class head's
confusability risk did not materialise.

Refuses to write unless charset+1 == head classes — the same guard
`carmenta_crnn_zh_prepare.py` uses, which is what catches a charset/model
mismatch before it becomes a silent accuracy loss.
"""
import hashlib, json, os
from pathlib import Path

CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "ffai" / "models" / "ppocrv5-mobile-rec"
ROOT = Path.home() / ".paddlex" / "official_models" / "PP-OCRv5_mobile_rec"
FIX = Path("corpora/refs/fixtures")


def main():
    import numpy as np
    import paddle
    import yaml

    assert ROOT.exists(), f"{ROOT} missing; run the paddleocr adapter once to fetch it"
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    prog, _, _ = paddle.static.load_inference_model(str(ROOT / "inference"), exe)
    scope = paddle.static.global_scope()
    state = {p.name: np.array(scope.find_var(p.name).get_tensor())
             for p in prog.global_block().all_parameters()}
    shapes = {k: list(v.shape) for k, v in state.items()}
    FIX.mkdir(parents=True, exist_ok=True)
    (FIX / "ppocrv5_mobile_rec_shapes.json").write_text(json.dumps(shapes, indent=0), encoding="utf-8")
    print(f"  {len(shapes)} tensors shape-mapped")

    cfg = yaml.safe_load((ROOT / "inference.yml").read_text(encoding="utf-8"))
    charset = cfg["PostProcess"]["character_dict"]
    # CTC head: [in, n_class]. PaddleOCR's CTCLabelDecode builds its label list
    # as  [blank] + character_dict + [' ']  when `use_space_char` is set, which
    # this release does implicitly — the dict itself carries no space (verified)
    # yet the head is charset+2. So index 0 is blank, 1..N are the dict in order,
    # and N+1 is a literal space. Getting this wrong shifts EVERY character index
    # and yields fluent garbage, which is why it is asserted rather than assumed.
    head = [v for k, v in shapes.items() if len(v) == 2 and max(v) > 10000]
    n_class = max(max(v) for v in head) if head else -1
    assert n_class == len(charset) + 2, (
        f"charset/head mismatch: {len(charset)} chars + blank + space != {n_class} classes. "
        "Refusing to write — this is the silent-accuracy-loss guard.")
    assert " " not in charset, "dict already carries a space; the +2 accounting would double it"
    print(f"  charset {len(charset)} + blank + space == head {n_class}  OK")

    CACHE.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v) for k, v in state.items()}, str(CACHE / "rec.safetensors"))
    (CACHE / "charset.txt").write_text("\n".join(charset), encoding="utf-8")
    sha = hashlib.sha256((CACHE / "rec.safetensors").read_bytes()).hexdigest()
    print(f"  wrote {CACHE / 'rec.safetensors'}  sha256={sha}")
    print(f"  wrote {CACHE / 'charset.txt'}")
    print("  preprocessing (for the port): " + json.dumps(cfg.get("PreProcess", {}))[:200])


if __name__ == "__main__":
    main()
