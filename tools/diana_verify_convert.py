"""Verify the converted safetensors reproduce the reference (Diana M-D1).

**Prove the port offline, for free, before writing any Rust.** This is a
complete YOLO26n forward pass built ONLY from `yolo26n-diana.safetensors`
and `torch.nn.functional` — no ultralytics module classes, no `.pt`. If it
reproduces the oracle dump layer for layer, then three things are proven at
once, cheaply:

  1. the conversion's key map is complete and correct;
  2. the conv+bn fold is exact, and dropping the one2many head is safe;
  3. the architecture as *understood here* — the block wiring, the
     attention layout, the two-stage top-k — is the architecture the
     checkpoint actually implements.

That last one is the point. Everything below is the specification the Rust
port transliterates, and it has been checked against the reference rather
than read off a diagram. Carmenta lost a session to a port built from a
repo's own constructor calls that did not match its shipped checkpoint;
this file is that lesson spent in advance.

Usage:
    .venv-diana/Scripts/python.exe tools/diana_verify_convert.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
ST = ROOT / "corpora" / "cache" / "yolo26n-diana.safetensors"
MAN = ROOT / "models" / "yolo26n-diana.json"
FIX = ROOT / "corpora" / "refs" / "fixtures"

W: dict[str, torch.Tensor] = {}
MAN_J: dict = {}


# ---- primitives --------------------------------------------------------
def conv(x, pre, stride=1, pad=None, groups=1, act=True):
    """Fused conv (+bias) with optional SiLU. `pre` is the module prefix."""
    w = W[f"{pre}.weight"]
    b = W[f"{pre}.bias"]
    k = w.shape[-1]
    if pad is None:
        pad = k // 2
    y = F.conv2d(x, w, b, stride=stride, padding=pad, groups=groups)
    return F.silu(y) if act else y


def bneck(x, pre, add=True):
    y = conv(conv(x, f"{pre}.cv1"), f"{pre}.cv2")
    return x + y if add else y


def c3k(x, pre, n_inner=2):
    """C3k = C3 with k=3 bottlenecks: cv3(cat(m(cv1(x)), cv2(x)))."""
    y = conv(x, f"{pre}.cv1")
    for i in range(n_inner):
        y = bneck(y, f"{pre}.m.{i}", add=True)
    return conv(torch.cat([y, conv(x, f"{pre}.cv2")], 1), f"{pre}.cv3")


def attention(x, pre, num_heads, key_dim, head_dim):
    B, C, H, Wd = x.shape
    N = H * Wd
    qkv = conv(x, f"{pre}.qkv", act=False)
    q, k, v = qkv.view(B, num_heads, key_dim * 2 + head_dim, N).split(
        [key_dim, key_dim, head_dim], dim=2
    )
    scale = key_dim**-0.5
    attn = (q * scale).transpose(-2, -1) @ k
    attn = attn.softmax(dim=-1)
    out = (v @ attn.transpose(-2, -1)).view(B, C, H, Wd) + conv(
        v.reshape(B, C, H, Wd), f"{pre}.pe", groups=C, act=False
    )
    return conv(out, f"{pre}.proj", act=False)


def psablock(x, pre, num_heads, key_dim, head_dim, add=True):
    a = attention(x, f"{pre}.attn", num_heads, key_dim, head_dim)
    x = x + a if add else a
    f = conv(conv(x, f"{pre}.ffn.0"), f"{pre}.ffn.1", act=False)
    return x + f if add else f


def c3k2(x, pre, c, n=1, kind="bottleneck", heads=2):
    """C2f-shaped: split cv1 in two, chain the inner module, concat, cv2."""
    y = list(conv(x, f"{pre}.cv1").chunk(2, 1))
    for i in range(n):
        last = y[-1]
        if kind == "bottleneck":
            nxt = bneck(last, f"{pre}.m.{i}", add=True)
        elif kind == "c3k":
            nxt = c3k(last, f"{pre}.m.{i}")
        elif kind == "attn":
            nxt = bneck(last, f"{pre}.m.{i}.0", add=True)
            nxt = psablock(nxt, f"{pre}.m.{i}.1", heads, 32, 64, add=True)
        else:
            raise ValueError(kind)
        y.append(nxt)
    return conv(torch.cat(y, 1), f"{pre}.cv2")


def sppf(x, pre, n=3, k=5, add=True):
    """cv1 has act=False; the residual is added AFTER cv2 when c1 == c2."""
    y = [conv(x, f"{pre}.cv1", act=False)]
    for _ in range(n):
        y.append(F.max_pool2d(y[-1], kernel_size=k, stride=1, padding=k // 2))
    out = conv(torch.cat(y, 1), f"{pre}.cv2")
    return out + x if add else out


def c2psa(x, pre, c, heads=2):
    a, b = conv(x, f"{pre}.cv1").split((c, c), dim=1)
    b = psablock(b, f"{pre}.m.0", heads, 32, 64, add=True)
    return conv(torch.cat([a, b], 1), f"{pre}.cv2")


# ---- the graph ---------------------------------------------------------
def forward(x: torch.Tensor) -> tuple[dict[int, torch.Tensor], dict]:
    L: dict[int, torch.Tensor] = {}
    L[0] = conv(x, "model.0", stride=2)
    L[1] = conv(L[0], "model.1", stride=2)
    L[2] = c3k2(L[1], "model.2", c=16, kind="bottleneck")
    L[3] = conv(L[2], "model.3", stride=2)
    L[4] = c3k2(L[3], "model.4", c=32, kind="bottleneck")
    L[5] = conv(L[4], "model.5", stride=2)
    L[6] = c3k2(L[5], "model.6", c=64, kind="c3k")
    L[7] = conv(L[6], "model.7", stride=2)
    L[8] = c3k2(L[7], "model.8", c=128, kind="c3k")
    L[9] = sppf(L[8], "model.9")
    L[10] = c2psa(L[9], "model.10", c=128)
    L[11] = F.interpolate(L[10], scale_factor=2.0, mode="nearest")
    L[12] = torch.cat([L[11], L[6]], 1)
    L[13] = c3k2(L[12], "model.13", c=64, kind="c3k")
    L[14] = F.interpolate(L[13], scale_factor=2.0, mode="nearest")
    L[15] = torch.cat([L[14], L[4]], 1)
    L[16] = c3k2(L[15], "model.16", c=32, kind="c3k")
    L[17] = conv(L[16], "model.17", stride=2)
    L[18] = torch.cat([L[17], L[13]], 1)
    L[19] = c3k2(L[18], "model.19", c=64, kind="c3k")
    L[20] = conv(L[19], "model.20", stride=2)
    L[21] = torch.cat([L[20], L[10]], 1)
    L[22] = c3k2(L[21], "model.22", c=128, kind="attn", heads=2)

    # ---- head: the ONE2ONE branch only -------------------------------
    feats = [L[16], L[19], L[22]]
    boxes, scores = [], []
    for i, f in enumerate(feats):
        b = conv(conv(f, f"model.23.one2one_cv2.{i}.0"), f"model.23.one2one_cv2.{i}.1")
        b = F.conv2d(b, W[f"model.23.one2one_cv2.{i}.2.weight"],
                     W[f"model.23.one2one_cv2.{i}.2.bias"])
        c = f
        for j in (0, 1):
            cin = c.shape[1]
            c = conv(c, f"model.23.one2one_cv3.{i}.{j}.0", groups=cin, act=True)
            c = conv(c, f"model.23.one2one_cv3.{i}.{j}.1")
        c = F.conv2d(c, W[f"model.23.one2one_cv3.{i}.2.weight"],
                     W[f"model.23.one2one_cv3.{i}.2.bias"])
        boxes.append(b)
        scores.append(c)

    B = feats[0].shape[0]
    nc = MAN_J["nc"]
    strides = MAN_J["strides"]
    box_cat = torch.cat([b.view(B, 4, -1) for b in boxes], 2)
    score_cat = torch.cat([s.view(B, nc, -1) for s in scores], 2)

    # anchors: cell centres (+0.5), per level, and the matching stride vector
    apts, svec = [], []
    for f, s in zip(feats, strides):
        h, w = f.shape[2], f.shape[3]
        sx = torch.arange(w, dtype=torch.float32) + 0.5
        sy = torch.arange(h, dtype=torch.float32) + 0.5
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        apts.append(torch.stack([gx, gy], -1).view(-1, 2))
        svec.append(torch.full((h * w, 1), float(s)))
    anchors = torch.cat(apts).transpose(0, 1)  # (2, A)
    stride_t = torch.cat(svec).transpose(0, 1)  # (1, A)

    # reg_max == 1 -> dfl is Identity; end2end -> xyxy, not xywh
    lt, rb = box_cat.chunk(2, 1)
    x1y1 = anchors.unsqueeze(0) - lt
    x2y2 = anchors.unsqueeze(0) + rb
    dbox = torch.cat([x1y1, x2y2], 1) * stride_t
    decoded = torch.cat([dbox, score_cat.sigmoid()], 1)  # (B, 4+nc, A)

    # two-stage top-k, exactly as Detect.postprocess does it
    preds = decoded.permute(0, 2, 1)  # (B, A, 4+nc)
    bx, sc = preds.split([4, nc], dim=-1)
    k = min(MAN_J["max_det"], sc.shape[1])
    ori = sc.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
    gathered = sc.gather(dim=1, index=ori.expand(-1, -1, nc))
    flat, index = gathered.flatten(1).topk(k)
    idx = ori.gather(dim=1, index=(index // nc).unsqueeze(-1))
    cls = (index % nc)[..., None].float()
    bx = bx.gather(dim=1, index=idx.expand(-1, -1, 4))
    final = torch.cat([bx, flat[..., None], cls], dim=-1)

    return L, {
        "head_boxes": box_cat,
        "head_scores": score_cat,
        "anchors": anchors,
        "strides": stride_t,
        "decoded": decoded,
        "final": final,
    }


def main() -> None:
    global W, MAN_J
    if not ST.exists():
        raise SystemExit(f"missing {ST} — run tools/diana_convert.py first")
    W = load_file(str(ST))
    MAN_J = json.loads(MAN.read_text(encoding="utf-8"))
    print(f"loaded {len(W)} tensors from {ST.name} (map v{MAN_J['conversion_map_version']})")

    worst_overall = 0.0
    failures = []
    for kind in ("synth", "photo"):
        npz = FIX / f"yolo26n_oracle_{kind}.npz"
        if not npz.exists():
            print(f"skip {kind}: {npz.name} absent (run tools/diana_oracle_dump.py)")
            continue
        ref = np.load(npz)
        x = torch.from_numpy(ref["input"])
        with torch.no_grad():
            L, head = forward(x)

        print(f"\n--- {kind} ---")
        print(f"{'tensor':<16}{'max-abs':>12}{'rel':>12}   shape")
        for i in sorted(L):
            key = f"layer_{i:02d}"
            if key not in ref:
                continue
            got = L[i].numpy()
            want = ref[key]
            d = float(np.abs(got - want).max())
            scale = float(np.abs(want).max()) or 1.0
            worst_overall = max(worst_overall, d / scale)
            if d / scale > 1e-4:
                failures.append((kind, key, d))
            print(f"{key:<16}{d:>12.2e}{d / scale:>12.2e}   {list(got.shape)}")

        for name, t in head.items():
            if name not in ref:
                continue
            got = t.numpy()
            want = ref[name]
            if got.shape != want.shape:
                print(f"{name:<16}{'SHAPE':>12}   {list(got.shape)} vs {list(want.shape)}")
                failures.append((kind, name, float("nan")))
                continue
            d = float(np.abs(got - want).max())
            scale = float(np.abs(want).max()) or 1.0
            worst_overall = max(worst_overall, d / scale)
            if d / scale > 1e-4:
                failures.append((kind, name, d))
            print(f"{name:<16}{d:>12.2e}{d / scale:>12.2e}   {list(got.shape)}")

        # the detections that actually matter
        gf, wf = head["final"].numpy()[0], ref["final"][0]
        keep = wf[:, 4] > 0.25
        if keep.any():
            box_d = float(np.abs(gf[keep][:, :4] - wf[keep][:, :4]).max())
            cls_same = int((gf[keep][:, 5] == wf[keep][:, 5]).sum())
            print(
                f"confident dets: {int(keep.sum())} · max box delta {box_d:.3e} px · "
                f"classes identical {cls_same}/{int(keep.sum())}"
            )

    print()
    if failures:
        for kind, key, d in failures:
            print(f"FAIL {kind}/{key}: {d:.3e}")
        raise SystemExit(f"{len(failures)} tensor(s) outside 1e-4 relative")
    print(f"PASS — every tensor within 1e-4 relative (worst {worst_overall:.2e})")


if __name__ == "__main__":
    main()
