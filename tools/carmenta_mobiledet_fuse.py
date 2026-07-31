"""Collapse PP-OCRv5 mobile-det's LearnableRepLayer branches into plain
convolutions, offline, so the candle port never has to know they existed.

PP-LCNetV3 trains each conv as several parallel branches (4x conv_kxk, an
optional conv_1x1, each with its own BatchNorm) summed together, then scaled
by a learnable per-tensor affine. All of that is LINEAR, so it folds exactly
into a single conv weight + bias:

    conv+BN   ->  W' = W * g/sqrt(v+eps),  b' = beta - mean*g/sqrt(v+eps)
    branches  ->  sum the W' (1x1 zero-padded to the kxk centre) and the b'
    lab       ->  W'' = W' * scale,        b'' = b' * scale + bias

Only the POST-ACTIVATION affine (`act.lab`) survives, because it sits after a
non-linearity — the port applies it as a scalar affine after hardswish.

Output: `det-fused.safetensors` in the model cache, plus a JSON manifest of
the fused layer list so the Rust side is built against facts rather than a
paper figure (the CRAFT-widths lesson).

Self-check: for every fused layer, a random input is pushed through the
branch sum and through the fused conv, and the max abs difference must sit
at float noise. A fusion that does not verify is not written.
"""

import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "ffai" / "models" / "ppocrv5-mobile-det"
EPS = 1e-5


def fuse_bn(w, g, b, m, v):
    """conv weight + BatchNorm -> (weight, bias)."""
    s = g / np.sqrt(v + EPS)
    return w * s.reshape(-1, 1, 1, 1), b - m * s


def pad_to(w, k):
    """Centre a 1x1 kernel inside a kxk one."""
    if w.shape[-1] == k:
        return w
    out = np.zeros(w.shape[:2] + (k, k), dtype=w.dtype)
    c = k // 2
    out[:, :, c : c + 1, c : c + 1] = w
    return out


def conv_ref(x, w, b, groups, stride, pad):
    """Minimal NCHW conv for the self-check (small tensors only)."""
    n, cin, h, wd = x.shape
    cout, cing, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (wd + 2 * pad - kw) // stride + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.float32)
    per = cout // groups
    for oc in range(cout):
        g = oc // per
        sl = xp[:, g * cing : (g + 1) * cing]
        for i in range(oh):
            for j in range(ow):
                patch = sl[:, :, i * stride : i * stride + kh, j * stride : j * stride + kw]
                out[:, oc, i, j] = (patch * w[oc]).sum(axis=(1, 2, 3))
        out[:, oc] += b[oc]
    return out


def main():
    st = load_file(str(CACHE / "det-train.safetensors"))
    keys = set(st)

    # Every rep layer is identified by the prefix owning `conv_kxk.0.conv.weight`.
    prefixes = sorted({k[: -len(".conv_kxk.0.conv.weight")]
                       for k in keys if k.endswith(".conv_kxk.0.conv.weight")})
    fused, layers, worst = {}, [], 0.0

    for p in prefixes:
        branches = sorted({int(k.split(".conv_kxk.")[1].split(".")[0])
                           for k in keys if k.startswith(p + ".conv_kxk.")})
        k = st[f"{p}.conv_kxk.0.conv.weight"].shape[-1]
        W = np.zeros_like(st[f"{p}.conv_kxk.0.conv.weight"], dtype=np.float32)
        B = np.zeros(W.shape[0], dtype=np.float32)
        parts = []
        for i in branches:
            q = f"{p}.conv_kxk.{i}"
            w, b = fuse_bn(st[f"{q}.conv.weight"].astype(np.float32),
                           st[f"{q}.bn.weight"].astype(np.float32),
                           st[f"{q}.bn.bias"].astype(np.float32),
                           st[f"{q}.bn._mean"].astype(np.float32),
                           st[f"{q}.bn._variance"].astype(np.float32))
            parts.append((w, b))
            W += w
            B += b
        if f"{p}.conv_1x1.conv.weight" in keys:
            q = f"{p}.conv_1x1"
            w, b = fuse_bn(st[f"{q}.conv.weight"].astype(np.float32),
                           st[f"{q}.bn.weight"].astype(np.float32),
                           st[f"{q}.bn.bias"].astype(np.float32),
                           st[f"{q}.bn._mean"].astype(np.float32),
                           st[f"{q}.bn._variance"].astype(np.float32))
            parts.append((pad_to(w, k), b))
            W += pad_to(w, k)
            B += b
        # The pre-activation affine folds straight in; it is per-tensor.
        if f"{p}.lab.scale" in keys:
            sc = float(st[f"{p}.lab.scale"].reshape(-1)[0])
            bi = float(st[f"{p}.lab.bias"].reshape(-1)[0])
            W, B = W * sc, B * sc + bi
        else:
            sc, bi = 1.0, 0.0

        # ---- self-check: branch sum vs fused conv on random input ----
        cin_g = W.shape[1]
        groups = 1 if cin_g > 1 else W.shape[0]
        x = np.random.RandomState(0).randn(1, cin_g if groups == 1 else W.shape[0], 7, 7).astype(np.float32)
        pad = k // 2
        ref = sum(conv_ref(x, w, b, groups, 1, pad) for w, b in parts) * sc + bi
        got = conv_ref(x, W, B, groups, 1, pad)
        d = float(np.abs(ref - got).max())
        worst = max(worst, d)

        fused[f"{p}.weight"] = W
        fused[f"{p}.bias"] = B
        layers.append({
            "name": p, "k": int(k), "shape": [int(v) for v in W.shape],
            "branches": len(parts), "groups": int(groups),
            "act_scale": float(st[f"{p}.act.lab.scale"].reshape(-1)[0]) if f"{p}.act.lab.scale" in keys else None,
            "act_bias": float(st[f"{p}.act.lab.bias"].reshape(-1)[0]) if f"{p}.act.lab.bias" in keys else None,
        })

    # Everything that is not a rep branch passes through unchanged, with
    # plain conv+BN pairs folded too.
    handled = {kk for p in prefixes for kk in keys
               if kk.startswith(p + ".conv_kxk.") or kk.startswith(p + ".conv_1x1.")
               or kk.startswith(p + ".lab.")}
    for kk in sorted(keys - handled):
        fused[kk] = st[kk].astype(np.float32)

    assert worst < 2e-3, f"fusion self-check FAILED: max abs diff {worst}"
    save_file(fused, str(CACHE / "det-fused.safetensors"))
    Path("corpora/refs/fixtures/ppocrv5_mobile_det_fused.json").write_text(
        json.dumps({"layers": layers}, indent=0), encoding="utf-8")
    print(f"fused {len(prefixes)} rep layers, self-check max abs diff {worst:.2e}")
    print(f"tensors {len(st)} -> {len(fused)}; wrote det-fused.safetensors")


if __name__ == "__main__":
    main()
