"""Probe the real YOLO26n architecture from the checkpoint (Diana M-D1).

**Port from the checkpoint, never from a paper figure.** Carmenta lost time
twice to this: CRAFT's repo constructor widths did not match its shipped
weights, and PARSeq's checkpoint carried dec_heads=6 / eos=0 / bos=95 that no
diagram mentioned. Everything the Rust port needs to be correct is read off
the artifact here and written to a JSON the port and its oracles both consume.

Emits `corpora/refs/fixtures/yolo26n_arch.json`:
  - yaml:      the parsed model YAML (from/repeats/module/args per layer)
  - modules:   every leaf module, its type and its constructor-relevant attrs
  - tensors:   every state_dict key -> shape + dtype
  - forward:   per-layer input/output shapes on a fixed 640x640 input
  - head:      Detect-head specifics (nc, reg_max, strides, end2end flags)

Usage:
    .venv-diana/Scripts/python.exe tools/diana_probe_arch.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from ultralytics import YOLO

import sys

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "yolo26n"
CKPT = ROOT / "corpora" / "cache" / f"{MODEL}.pt"
OUT = ROOT / "corpora" / "refs" / "fixtures" / f"{MODEL}_arch.json"

model = YOLO(str(CKPT))
net = model.model  # DetectionModel
net.eval()

report: dict = {"checkpoint": CKPT.name}

# ---- 1. the YAML: the authoritative layer graph -------------------------
yaml = getattr(net, "yaml", {})
report["yaml"] = {
    "nc": yaml.get("nc"),
    "scale": yaml.get("scale"),
    "scales": yaml.get("scales"),
    "backbone": yaml.get("backbone"),
    "head": yaml.get("head"),
}

# ---- 2. every module in graph order, with the attrs a port needs --------
INTERESTING = (
    "c1 c2 c_ k s p g d n e shortcut c3k nh ne num_heads head_dim key_dim "
    "kernel_size stride padding groups dilation in_channels out_channels "
    "eps momentum scale_factor mode nc reg_max no end2end max_det"
).split()

modules = []
for name, m in net.named_modules():
    if name == "":
        continue
    entry = {"name": name, "type": type(m).__name__}
    attrs = {}
    for a in INTERESTING:
        if hasattr(m, a):
            v = getattr(m, a)
            if isinstance(v, (int, float, bool, str)):
                attrs[a] = v
            elif isinstance(v, (tuple, list)) and all(
                isinstance(x, (int, float, bool, str)) for x in v
            ):
                attrs[a] = list(v)
            elif isinstance(v, torch.Tensor) and v.numel() <= 8:
                attrs[a] = v.flatten().tolist()
    if attrs:
        entry["attrs"] = attrs
    modules.append(entry)
report["modules"] = modules

# ---- 2b. activations, read DIRECTLY off each Conv ----------------------
# TRAP, and it would have produced a silently wrong port. Ultralytics sets
# `Conv.default_act = nn.SiLU()` as a CLASS attribute, so every Conv that
# takes the default shares ONE module instance. `named_modules()` dedupes by
# object identity, so the shared SiLU is listed at its FIRST occurrence and
# omitted everywhere else — making `model.9.cv2` look like it has no
# activation at all, when it has SiLU and only `model.9.cv1` has the
# Identity (i.e. genuinely none). Same shape as the torchvision inplace-ReLU
# trap Carmenta hit on CRAFT: the module tree does not say what the forward
# pass does. Every Conv's activation is therefore recorded explicitly.
#
# Second trap, found while porting the head: match on isinstance, NOT on
# `type(m).__name__ == "Conv"`. `DWConv` SUBCLASSES `Conv`, so a name test
# silently omitted all 12 depthwise convolutions in the classification
# head — they carry SiLU, and a port reading this fixture would have built
# them activation-free.
from ultralytics.nn.modules.conv import Conv as UConv  # noqa: E402

acts = {}
kinds = {}
for name, m in net.named_modules():
    if isinstance(m, UConv) and hasattr(m, "act"):
        acts[name] = type(m.act).__name__
        kinds[name] = type(m).__name__
report["conv_activations"] = acts
report["conv_kinds"] = kinds
report["activation_kinds"] = sorted(set(acts.values()))

# ---- 3. every tensor: name -> shape, dtype ------------------------------
sd = net.state_dict()
report["tensors"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sd.items()}
report["tensor_count"] = len(sd)
report["param_count"] = int(sum(v.numel() for v in sd.values()))

# ---- 4. per-layer forward shapes on a fixed input -----------------------
# The graph is a DAG with `from` indices, so shapes are how a port verifies
# it wired the skips correctly before any numbers are compared.
shapes: list = []
hooks = []


def make_hook(idx: int, mtype: str):
    def hook(module, inputs, output):
        def shp(x):
            if isinstance(x, torch.Tensor):
                return list(x.shape)
            if isinstance(x, (list, tuple)):
                return [shp(i) for i in x]
            return str(type(x).__name__)

        shapes.append(
            {
                "i": idx,
                "type": mtype,
                "in": [shp(i) for i in inputs],
                "out": shp(output),
            }
        )

    return hook


for i, layer in enumerate(net.model):
    hooks.append(layer.register_forward_hook(make_hook(i, type(layer).__name__)))

x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    out = net(x)
for h in hooks:
    h.remove()
report["forward"] = shapes


def describe(o):
    if isinstance(o, torch.Tensor):
        return list(o.shape)
    if isinstance(o, (list, tuple)):
        return [describe(i) for i in o]
    return str(type(o).__name__)


report["model_output"] = describe(out)

# ---- 5. the Detect head's own facts ------------------------------------
head = net.model[-1]
report["head"] = {
    "type": type(head).__name__,
    **{
        a: (list(getattr(head, a).flatten().tolist()) if isinstance(getattr(head, a), torch.Tensor) else getattr(head, a))
        for a in ("nc", "reg_max", "no", "end2end", "max_det", "stride")
        if hasattr(head, a)
    },
}
report["names_count"] = len(getattr(net, "names", {}) or {})

# ---- 6. flags that change the forward but live nowhere in the YAML ------
# SPPF's 4-arg form `[1024, 5, 3, True]` and C3k2's `[1024, True, 0.5, True]`
# carry booleans a paper figure does not show. Read the resolved objects.
extras = {}
for i, layer in enumerate(net.model):
    t = type(layer).__name__
    if t in ("SPPF", "C3k2", "C2PSA", "C2f"):
        e = {"type": t}
        for a in ("n", "add", "shortcut", "c", "c1", "c2"):
            if hasattr(layer, a):
                v = getattr(layer, a)
                if isinstance(v, (int, float, bool, str)):
                    e[a] = v
        if hasattr(layer, "m"):
            e["m_len"] = len(layer.m) if hasattr(layer.m, "__len__") else 1
            inner = layer.m[0] if hasattr(layer.m, "__getitem__") else layer.m
            e["m_type"] = type(inner).__name__
            if hasattr(inner, "add"):
                e["m_add"] = bool(inner.add)
        extras[f"model.{i}"] = e
report["block_flags"] = extras

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

print(f"wrote {OUT.relative_to(ROOT)}")
print(f"  scale        : {report['yaml']['scale']}  nc={report['yaml']['nc']}")
print(f"  tensors      : {report['tensor_count']}  params={report['param_count']:,}")
print(f"  layers       : {len(shapes)}")
print(f"  model output : {report['model_output']}")
print(f"  head         : {report['head']}")
