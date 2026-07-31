"""A complete numpy forward of PP-OCRv5 mobile-det, built from the TRAIN
checkpoint and gated on paddle's own exported program.

Why a numpy reference at all, when the exported program is right there: the
exported program is a black box with op-indexed names, so it can tell us
whether we are right but never *where* we are wrong. This reference is
transparent — once it agrees with the oracle end-to-end, every intermediate it
holds is trustworthy, and those intermediates become the per-stage fixtures the
candle port is gated on. It is the instrument, and the oracle calibrates it.

Two details of the det variant are NOT derivable from any source available on
this box (PaddleOCR's `det_pp_lcnet_v3` is not vendored in the venv, and the
classification variant in paddlex disagrees with PaddleOCR's own necks):

  * the hardsigmoid slope inside the squeeze-excitation blocks — paddlex uses
    paddle's 1/6 default, PaddleOCR's SEModule hard-codes 0.2;
  * whether RSELayer applies its residual shortcut.

Rather than pick the plausible one, this script SEARCHES the small space and
reports which combination reproduces the oracle. A guess that happens to be
wrong here would degrade every box quietly, which is the failure mode that cost
this campaign a day on PARSeq's missing sqrt(d).
"""

import itertools
import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ["LOCALAPPDATA"]) / "ffai" / "models" / "ppocrv5-mobile-det"
FIX = REPO / "corpora" / "refs" / "fixtures"
EPS = 1e-5

# (name, depthwise kernel, stride). Strides are not guessed: a LearnableRepLayer
# carries an `identity` BatchNorm branch iff in==out AND stride==1, and a
# depthwise conv always has in==out — so the 19 identity branches in the
# checkpoint pin the stride of every depthwise conv exactly.
BLOCKS = [
    ("blocks2.0", 3, 1),
    ("blocks3.0", 3, 2), ("blocks3.1", 3, 1),
    ("blocks4.0", 3, 2), ("blocks4.1", 3, 1),
    ("blocks5.0", 3, 2), ("blocks5.1", 5, 1), ("blocks5.2", 5, 1),
    ("blocks5.3", 5, 1), ("blocks5.4", 5, 1),
    ("blocks6.0", 5, 2), ("blocks6.1", 5, 1), ("blocks6.2", 5, 1),
    ("blocks6.3", 5, 1),
]
# Taps feeding the neck: the end of stages 3, 4, 5, 6 (strides 4, 8, 16, 32).
TAPS = {"blocks3.1": 0, "blocks4.1": 1, "blocks5.4": 2, "blocks6.3": 3}


# ---------------------------------------------------------------- primitives

def conv2d(x, w, b=None, stride=1, pad=0, groups=1):
    n, c, h, wd = x.shape
    o, cg, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (wd + 2 * pad - kw) // stride + 1
    if groups == c and cg == 1:  # depthwise: no gather, just shifted adds
        acc = np.zeros((n, c, oh, ow), np.float32)
        for ky in range(kh):
            for kx in range(kw):
                acc += (xp[:, :, ky:ky + oh * stride:stride, kx:kx + ow * stride:stride]
                        * w[:, 0, ky, kx].reshape(1, c, 1, 1))
    else:
        og = o // groups
        acc = np.empty((n, o, oh, ow), np.float32)
        for g in range(groups):
            xs = xp[:, g * cg:(g + 1) * cg]
            ws = w[g * og:(g + 1) * og]
            part = np.zeros((n, og, oh, ow), np.float32)
            for ky in range(kh):
                for kx in range(kw):
                    part += np.einsum(
                        "nchw,oc->nohw",
                        xs[:, :, ky:ky + oh * stride:stride, kx:kx + ow * stride:stride],
                        ws[:, :, ky, kx], optimize=True)
            acc[:, g * og:(g + 1) * og] = part
    if b is not None:
        acc = acc + b.reshape(1, -1, 1, 1)
    return acc.astype(np.float32)


def conv_transpose2x2(x, w, b):
    """Stride 2, kernel 2, no padding — so the taps never overlap and this is a
    per-pixel matmul into a 2x2 block rather than a scatter-add."""
    n, c, h, wd = x.shape
    _, o, _, _ = w.shape
    out = np.empty((n, o, h * 2, wd * 2), np.float32)
    for ky in range(2):
        for kx in range(2):
            out[:, :, ky::2, kx::2] = np.einsum(
                "nchw,co->nohw", x, w[:, :, ky, kx], optimize=True)
    return (out + b.reshape(1, -1, 1, 1)).astype(np.float32)


def bn(x, st, p):
    s = st[f"{p}.weight"] / np.sqrt(st[f"{p}._variance"] + EPS)
    return x * s.reshape(1, -1, 1, 1) + (st[f"{p}.bias"] - st[f"{p}._mean"] * s).reshape(1, -1, 1, 1)


def hardswish(x):
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def hardsigmoid(x, slope):
    return np.clip(slope * x + 0.5, 0.0, 1.0)


def upsample2(x, f):
    return np.repeat(np.repeat(x, f, axis=2), f, axis=3)


# ------------------------------------------------------------------- modules

def rep_layer(x, st, p, stride, k, groups, fused=False):
    """LearnableRepLayer: identity-BN + 1x1 + N x kxk, summed, then `lab`, then
    the activation IFF stride != 2 (the `act` parameters exist either way, which
    is a trap — they are saved for stride-2 layers and must not be applied)."""
    if fused:
        out = conv2d(x, st[f"{p}.weight"], st[f"{p}.bias"], stride, k // 2, groups)
        if f"{p}.act.lab.scale" in st:  # absent by construction when stride == 2
            out = hardswish(out) * float(st[f"{p}.act.lab.scale"].reshape(-1)[0]) \
                + float(st[f"{p}.act.lab.bias"].reshape(-1)[0])
        return out
    out = None
    if f"{p}.identity.weight" in st:
        out = bn(x, st, f"{p}.identity")
    if f"{p}.conv_1x1.conv.weight" in st:
        r = bn(conv2d(x, st[f"{p}.conv_1x1.conv.weight"], None, stride, 0, groups),
               st, f"{p}.conv_1x1.bn")
        out = r if out is None else out + r
    i = 0
    while f"{p}.conv_kxk.{i}.conv.weight" in st:
        q = f"{p}.conv_kxk.{i}"
        r = bn(conv2d(x, st[f"{q}.conv.weight"], None, stride, k // 2, groups), st, f"{q}.bn")
        out = r if out is None else out + r
        i += 1
    out = out * float(st[f"{p}.lab.scale"]) + float(st[f"{p}.lab.bias"])
    if stride != 2:
        out = hardswish(out)
        out = out * float(st[f"{p}.act.lab.scale"]) + float(st[f"{p}.act.lab.bias"])
    return out


def se(x, st, p, slope):
    o = x.mean(axis=(2, 3), keepdims=True)
    o = conv2d(o, st[f"{p}.conv1.weight"], st[f"{p}.conv1.bias"])
    o = np.maximum(o, 0.0)
    o = conv2d(o, st[f"{p}.conv2.weight"], st[f"{p}.conv2.bias"])
    return x * hardsigmoid(o, slope)


def rse_layer(x, st, p, k, shortcut, slope):
    x = conv2d(x, st[f"{p}.in_conv.weight"], None, 1, k // 2)
    s = se(x, st, f"{p}.se_block", slope)
    return x + s if shortcut else s


def forward(x, st, cfg, keep=None, fused=False):
    """One forward for both checkpoints. `fused=True` reads the collapsed
    weights the Rust port loads, so the fusion is checked on exactly the graph
    the port will run — not on a parallel implementation of it."""
    bb_slope, neck_slope, shortcut = cfg
    # ---- backbone
    if fused:
        h = conv2d(x, st["backbone.conv1.weight"], st["backbone.conv1.bias"], 2, 1)
    else:
        h = bn(conv2d(x, st["backbone.conv1.conv.weight"], None, 2, 1), st, "backbone.conv1.bn")
    taps = [None] * 4
    for name, k, stride in BLOCKS:
        p = f"backbone.{name}"
        c = h.shape[1]
        h = rep_layer(h, st, f"{p}.dw_conv", stride, k, c, fused)
        if f"{p}.se.conv1.weight" in st:
            h = se(h, st, f"{p}.se", bb_slope)
        h = rep_layer(h, st, f"{p}.pw_conv", 1, 1, 1, fused)
        if name in TAPS:
            i = TAPS[name]
            taps[i] = conv2d(h, st[f"backbone.layer_list.{i}.weight"],
                             st[f"backbone.layer_list.{i}.bias"])
            if keep is not None:
                keep[f"tap{i}"] = taps[i]
    # ---- neck (RSEFPN): top-down add, then per-level 3x3, then concat at 1/4
    ins = [rse_layer(taps[i], st, f"neck.ins_conv.{i}", 1, shortcut, neck_slope)
           for i in range(4)]
    outs = [None] * 4
    outs[3] = ins[3]
    for i in (2, 1, 0):
        outs[i] = ins[i] + upsample2(outs[i + 1], 2)
    ps = [rse_layer(outs[i], st, f"neck.inp_conv.{i}", 3, shortcut, neck_slope)
          for i in range(4)]
    fuse = np.concatenate(
        [upsample2(ps[3], 8), upsample2(ps[2], 4), upsample2(ps[1], 2), ps[0]], axis=1)
    if keep is not None:
        keep["fuse"] = fuse
    # ---- head (binarize branch only; `thresh` is training-only)
    p = "head.binarize"
    if fused:
        y = np.maximum(conv2d(fuse, st[f"{p}.conv1.weight"], st[f"{p}.conv1.bias"], 1, 1), 0)
        y = np.maximum(conv_transpose2x2(y, st[f"{p}.conv2.weight"], st[f"{p}.conv2.bias"]), 0)
    else:
        y = np.maximum(bn(conv2d(fuse, st[f"{p}.conv1.weight"], None, 1, 1), st, f"{p}.conv_bn1"), 0)
        y = np.maximum(bn(conv_transpose2x2(y, st[f"{p}.conv2.weight"], st[f"{p}.conv2.bias"]),
                          st, f"{p}.conv_bn2"), 0)
    y = conv_transpose2x2(y, st[f"{p}.conv3.weight"], st[f"{p}.conv3.bias"])
    if keep is not None:
        keep["logit"] = y
    return 1.0 / (1.0 + np.exp(-y))


# ---------------------------------------------------------------------- main

def main():
    st = {k: v.astype(np.float32) for k, v in load_file(str(CACHE / "det-train.safetensors")).items()}
    ref = np.load(FIX / "mobiledet_probmap.npz")
    x, want = ref["noise_input"], ref["noise_prob"]

    def logit(p):
        return np.log(np.clip(p, 1e-12, 1) / np.clip(1 - p, 1e-12, None))

    best = None
    print("searching the two undocumented det-variant choices, scored in logit space:")
    for cfg in itertools.product((1 / 6, 0.2), (1 / 6, 0.2), (True, False)):
        got = forward(x, st, cfg)
        d = float(np.abs(logit(got) - logit(want)).max())
        print(f"  bb_slope={cfg[0]:.4f} neck_slope={cfg[1]:.4f} shortcut={cfg[2]!s:5s}"
              f"  max|dlogit| {d:.4e}")
        if best is None or d < best[1]:
            best = (cfg, d)
    cfg, d = best
    print(f"\nBEST {cfg}  max|dlogit| {d:.3e}")
    if d > 1e-2:
        print("REFERENCE DOES NOT MATCH THE ORACLE — do not port against it.")
        return 1

    # Re-run on the real page and keep the per-stage activations the port needs.
    keep = {}
    got = forward(ref["page_input"], st, cfg, keep=keep)
    want_p = ref["page_prob"]
    dp = float(np.abs(got - want_p).max())
    agree = float(((got > 0.3) == (want_p > 0.3)).mean())
    print(f"page: max|dprob| {dp:.3e}   binarised agreement at 0.3: {agree * 100:.4f}%")

    np.savez_compressed(FIX / "mobiledet_stages.npz",
                        **{k: v.astype(np.float32) for k, v in keep.items()})
    (FIX / "mobiledet_ref.json").write_text(json.dumps({
        "bb_se_hardsigmoid_slope": cfg[0], "neck_se_hardsigmoid_slope": cfg[1],
        "rse_shortcut": cfg[2], "noise_max_abs_dlogit": d,
        "page_max_abs_dprob": dp, "page_binarised_agreement": agree,
        "stages": {k: list(v.shape) for k, v in keep.items()},
    }, indent=1), encoding="utf-8")
    print("wrote mobiledet_stages.npz (" + ", ".join(f"{k}{list(v.shape)}" for k, v in keep.items()) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
