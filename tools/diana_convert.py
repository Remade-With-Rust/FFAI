"""Convert an official YOLO26 `.pt` to safetensors + manifest (Diana M-D1).

This is the §7 product artifact, not a footnote: offline, pinned,
re-runnable from a fresh clone, and the only thing standing between an
AGPL-3.0 checkpoint the user fetches themselves and a pure-Rust binary that
loads no Python. Nothing here is redistributed — the tool ships, the weights
do not.

What it does, in the order §7 specifies:

1. Load the `.pt` in a pinned environment and take `state_dict()`.
2. **Drop the training-only one2many head, loudly.** `Detect.forward` at
   inference runs `preds["one2one"]` when `end2end` — the `cv2`/`cv3`
   branch exists only for the training loss. That is 120 of the head's 240
   tensors and ~30% of the checkpoint's parameters. Porting them would be
   the "plausible-but-wrong" failure Mercury hit when hidden states were
   argmaxed as logits, so the drop is explicit, counted, and recorded in
   the manifest rather than left as an accident of which keys got read.
3. **Fold BatchNorm into the preceding convolution.** Every Conv in this
   graph is conv->bn->act with no branch between them, so folding is exact
   in inference mode and removes 4 tensors per conv. The manifest records
   `fused: true` — §7 step 6 requires the fused/unfused decision to be
   stated, never inferred.
4. Emit safetensors + a machine-readable manifest: architecture id, strides,
   nc, imgsz, end2end, conversion-map version, and sha256 of BOTH the source
   checkpoint and the output.

Usage:
    .venv-diana/Scripts/python.exe tools/diana_convert.py --model yolo26n
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "corpora" / "cache"
MODELS = ROOT / "models"

# Where `ffai-models` looks for a manually-placed model, mirroring
# `ffai_models::cache_dir()` — FFAI_CACHE if set, else the platform cache
# dir. Same convention as tools/carmenta_mobiledet_prepare.py. The weights
# are AGPL-3.0 and have no hf_repo, so manual placement is the ONLY route:
# there is nothing for FFai to download on the user's behalf, by design.
def model_cache(name: str) -> Path:
    root = os.environ.get("FFAI_CACHE")
    if root:
        return Path(root) / "models" / name
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if not local:
        local = str(Path.home() / ".cache")
    return Path(local) / "ffai" / "models" / name

# Bump when the key mapping or fusion policy changes. A manifest carrying an
# older version is not silently accepted by the loader.
CONVERSION_MAP_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_bn(
    w: torch.Tensor,
    b: torch.Tensor | None,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """conv(w,b) -> bn(gamma,beta,mean,var,eps) collapsed into one conv.

    scale = gamma / sqrt(var + eps)
    w' = w * scale[:, None, None, None]
    b' = (b - mean) * scale + beta
    Exact in eval mode; this is arithmetic, not approximation.
    """
    scale = gamma / torch.sqrt(var + eps)
    w_out = w * scale.reshape(-1, 1, 1, 1)
    b_in = torch.zeros_like(mean) if b is None else b
    b_out = (b_in - mean) * scale + beta
    return w_out, b_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", help="checkpoint stem in corpora/cache")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    ckpt = CACHE / f"{args.model}.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt} — fetch it first (docs/diana-mission-plan.md §7.1)")

    net = YOLO(str(ckpt)).model
    net.eval()
    head = net.model[-1]
    sd = net.state_dict()
    src_total = len(sd)

    # Which task is this checkpoint? Read it off the head, never off the
    # filename — `-depth` is a naming convention and conventions get broken,
    # whereas the module type is what the weights actually are.
    task = "depth" if type(head).__name__ == "Depth" else "detect"

    # --- step 2: drop the training-only one2many head -------------------
    #
    # Detect only. The Depth head has no one2many branch — there is nothing
    # to drop, and matching `.cv2.`/`.cv3.` against it would silently delete
    # real weights if the head ever grew a submodule with those names.
    dropped = [] if task == "depth" else [k for k in sd if ".cv2." in k or ".cv3." in k]
    dropped = [k for k in dropped if k.startswith(f"model.{len(net.model) - 1}.")]
    dropped_params = int(sum(sd[k].numel() for k in dropped))
    kept = {k: v for k, v in sd.items() if k not in set(dropped)}

    # --- step 3: fold BN into the preceding conv ------------------------
    eps_by_bn: dict[str, float] = {}
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            eps_by_bn[name] = float(m.eps)

    tensors: dict[str, torch.Tensor] = {}
    folded = 0
    conv_prefixes = sorted(
        {k[: -len(".conv.weight")] for k in kept if k.endswith(".conv.weight")}
    )
    consumed: set[str] = set()
    for pre in conv_prefixes:
        wk, bk = f"{pre}.conv.weight", f"{pre}.conv.bias"
        bn = f"{pre}.bn"
        if f"{bn}.weight" not in kept:
            continue
        eps = eps_by_bn.get(bn, 1e-3)
        w, b = fold_bn(
            kept[wk],
            kept.get(bk),
            kept[f"{bn}.weight"],
            kept[f"{bn}.bias"],
            kept[f"{bn}.running_mean"],
            kept[f"{bn}.running_var"],
            eps,
        )
        tensors[f"{pre}.weight"] = w.contiguous()
        tensors[f"{pre}.bias"] = b.contiguous()
        consumed |= {wk, bk, f"{bn}.weight", f"{bn}.bias", f"{bn}.running_mean",
                     f"{bn}.running_var", f"{bn}.num_batches_tracked"}
        folded += 1

    # everything not consumed by a fold, carried through verbatim
    passthrough = 0
    for k, v in kept.items():
        if k in consumed:
            continue
        if k.endswith("num_batches_tracked"):
            continue
        tensors[k] = v.contiguous()
        passthrough += 1

    MODELS.mkdir(exist_ok=True)
    out_st = CACHE / f"{args.model}-diana.safetensors"
    save_file({k: v.to(torch.float32) for k, v in tensors.items()}, str(out_st))

    manifest = {
        "architecture": "yolo26",
        "scale": net.yaml.get("scale"),
        "task": task,
        "imgsz": args.imgsz,
        "letterbox": "square-center-pad-114",
        "inference_branch": "one2one",
        "fused": True,
        "fusion": "conv+bn folded at conversion (exact in eval mode)",
        "conversion_map_version": CONVERSION_MAP_VERSION,
        "source_file": ckpt.name,
        "source_sha256": sha256_file(ckpt),
        "output_file": out_st.name,
        "output_sha256": sha256_file(out_st),
        "tensor_count": len(tensors),
        "param_count": int(sum(v.numel() for v in tensors.values())),
        "source_tensor_count": src_total,
        "dropped_one2many_tensors": len(dropped),
        "dropped_one2many_params": dropped_params,
    }
    if task == "detect":
        manifest.update(
            {
                "nc": int(head.nc),
                "reg_max": int(head.reg_max),
                "end2end": bool(head.end2end),
                "max_det": int(head.max_det),
                "strides": [float(s) for s in head.stride],
                "class_names": [net.names[i] for i in sorted(net.names)],
            }
        )
    else:
        # Depth carries its own scalars. `cal_a`/`cal_b` are LEARNED and
        # per-tier — `depth^cal_a * exp(cal_b)` scales every pixel, so they
        # belong in the manifest beside the weights, not in the engine.
        #
        # `refine_used` records that the head stores three refine stages and
        # runs two: the reference loop is `range(nl - 2, -1, -1)`. The third
        # is dead weight in every released checkpoint and is emitted anyway,
        # so the manifest round-trips against the source.
        proj_in = [int(m.conv.in_channels) for m in head.proj]
        manifest.update(
            {
                "head_channels": int(head.proj[0].conv.out_channels),
                "proj_in_channels": proj_in,
                "cal_a": float(head.cal_a),
                "cal_b": float(head.cal_b),
                "logit_clamp": [-4.0, 5.0],
                "output_stride": 4,
                "output_units": "metres",
                "refine_stages": len(head.refine),
                "refine_used": len(head.refine) - 1,
            }
        )
    man_path = MODELS / f"{args.model}-diana.json"
    man_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # Place both artifacts where ffai-models resolves them, so the engine
    # finds the weights without a network round trip it could never make.
    dest = model_cache(f"{args.model}-diana")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_st, dest / out_st.name)
    shutil.copyfile(man_path, dest / man_path.name)

    print(f"wrote {out_st.relative_to(ROOT)} ({out_st.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {man_path.relative_to(ROOT)}")
    print(f"placed in model cache: {dest}")
    print(f"  safetensors sha256  : {manifest['output_sha256']}")
    print(f"  source tensors      : {src_total}")
    print(f"  task                : {task}")
    print(f"  dropped (one2many)  : {len(dropped)}  ({dropped_params:,} params)")
    print(f"  conv+bn folded      : {folded}")
    print(f"  passthrough         : {passthrough}")
    print(f"  output tensors      : {len(tensors)}  ({manifest['param_count']:,} params)")


if __name__ == "__main__":
    main()
