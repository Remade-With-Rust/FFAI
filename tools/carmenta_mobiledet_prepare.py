"""#5 prep: PP-OCRv5_mobile_det -> shape map + safetensors (the CRAFT-widths
lesson: port against checkpoint facts). Uses the model PaddleOCR already
fetched during M-C0 warm; paddle loads the inference program, we dump every
parameter. The candle port (PP-LCNetV3 + DBNet head) is the campaign brick.
"""
import json, hashlib, os
from pathlib import Path
CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "ffai" / "models" / "ppocrv5-mobile-det"
def main():
    import paddle
    from paddle.inference import Config
    # locate the fetched model dir
    roots = [Path.home() / ".paddlex" / "official_models" / "PP-OCRv5_mobile_det"]
    root = next((r for r in roots if r.exists()), None)
    assert root, "PP-OCRv5_mobile_det not in paddlex cache; run the paddleocr adapter once"
    prog = root / "inference.json"
    params = root / "inference.pdiparams"
    import numpy as np
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    prog, _, _ = paddle.static.load_inference_model(str(root / "inference"), exe)
    scope = paddle.static.global_scope()
    state = {p.name: np.array(scope.find_var(p.name).get_tensor()) for p in prog.global_block().all_parameters()}
    CACHE.mkdir(parents=True, exist_ok=True)
    shapes = {k: list(v.shape) for k, v in state.items()}
    Path("corpora/refs/fixtures/ppocrv5_mobile_det_shapes.json").write_text(
        json.dumps(shapes, indent=0), encoding="utf-8")
    print(f"{len(shapes)} tensors shape-mapped")
    import numpy as np
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v) for k, v in state.items()}, str(CACHE / "det.safetensors"))
    sha = hashlib.sha256((CACHE / "det.safetensors").read_bytes()).hexdigest()
    print(f"wrote {CACHE/'det.safetensors'} sha256={sha}")
    print("DONE")
main()
