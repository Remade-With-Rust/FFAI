"""Collapse PP-OCRv5 mobile-det into plain convolutions, offline, so the candle
port never has to know about rep branches or BatchNorm.

PP-LCNetV3 trains each conv as several parallel branches — an identity
BatchNorm, an optional 1x1, and N kxk convs, each with its own BN — summed and
then scaled by a learnable per-tensor affine. All of that is LINEAR, so it folds
exactly into one conv weight + bias:

    conv+BN   ->  W' = W * g/sqrt(v+eps),  b' = beta - mean*g/sqrt(v+eps)
    identity  ->  a centred kernel holding the BN's per-channel scale
    branches  ->  sum the W' (1x1 zero-padded to the kxk centre) and the b'
    lab       ->  W'' = W' * scale,        b'' = b' * scale + bias

Only the POST-activation affine (`act.lab`) survives, because it sits after a
non-linearity. It is emitted ONLY for stride-1 layers: `LearnableRepLayer`
constructs `act` unconditionally but applies it `if self.stride != 2`, so the
checkpoint carries parameters for stride-2 layers that must never be used.
Dropping them here makes that unrepresentable downstream rather than a comment
somebody has to obey.

## On the self-check

The previous version of this script verified the branch sum against the fused
conv and reported 4.88e-04 — while silently dropping all 19 identity branches,
because a check built from the parts it collected cannot fail on a part it never
collected. The lesson is not "test harder", it is that a self-consistency check
is not evidence. So verification now runs the WHOLE fused model and compares it
against paddle's own exported program, via `carmenta_mobiledet_ref`. Nothing is
written unless that agrees.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carmenta_mobiledet_ref as ref  # noqa: E402

CACHE = Path(os.environ["LOCALAPPDATA"]) / "ffai" / "models" / "ppocrv5-mobile-det"
FIX = Path(__file__).resolve().parent.parent / "corpora" / "refs" / "fixtures"
EPS = 1e-5


def bn_scale_shift(st, p):
    s = st[f"{p}.weight"] / np.sqrt(st[f"{p}._variance"] + EPS)
    return s, st[f"{p}.bias"] - st[f"{p}._mean"] * s


def fuse_conv_bn(st, conv, bnp):
    s, t = bn_scale_shift(st, bnp)
    return st[conv] * s.reshape(-1, 1, 1, 1), t.copy()


def centre(w, k):
    """Zero-pad a 1x1 kernel into the middle of a kxk one."""
    if w.shape[-1] == k:
        return w
    out = np.zeros(w.shape[:2] + (k, k), np.float32)
    c = k // 2
    out[:, :, c:c + 1, c:c + 1] = w
    return out


def identity_kernel(scale, k, groups, out_ch):
    """The identity BatchNorm branch as a conv kernel: a centred delta carrying
    the per-channel scale. Depthwise (groups==out) gets a [C,1,k,k] delta; a
    dense layer gets a diagonal."""
    c = k // 2
    if groups == out_ch:
        w = np.zeros((out_ch, 1, k, k), np.float32)
        w[:, 0, c, c] = scale
    else:
        w = np.zeros((out_ch, out_ch, k, k), np.float32)
        w[np.arange(out_ch), np.arange(out_ch), c, c] = scale
    return w


def fuse_rep(st, p, k, groups, out_ch):
    """One LearnableRepLayer -> (weight, bias, n_branches)."""
    # Shape from the checkpoint, never inferred: a pointwise conv may widen
    # (16 -> 32), so in_channels is not out_channels.
    in_ch = st[f"{p}.conv_kxk.0.conv.weight"].shape[1]
    W = np.zeros((out_ch, in_ch, k, k), np.float32)
    B = np.zeros(out_ch, np.float32)
    n = 0
    if f"{p}.identity.weight" in st:
        s, t = bn_scale_shift(st, f"{p}.identity")
        W += identity_kernel(s, k, groups, out_ch)
        B += t
        n += 1
    if f"{p}.conv_1x1.conv.weight" in st:
        w, b = fuse_conv_bn(st, f"{p}.conv_1x1.conv.weight", f"{p}.conv_1x1.bn")
        W += centre(w, k)
        B += b
        n += 1
    i = 0
    while f"{p}.conv_kxk.{i}.conv.weight" in st:
        w, b = fuse_conv_bn(st, f"{p}.conv_kxk.{i}.conv.weight", f"{p}.conv_kxk.{i}.bn")
        W += w
        B += b
        n += 1
        i += 1
    sc = float(st[f"{p}.lab.scale"].reshape(-1)[0])
    bi = float(st[f"{p}.lab.bias"].reshape(-1)[0])
    return W * sc, B * sc + bi, n


def main():
    st = {k: v.astype(np.float32) for k, v in load_file(str(CACHE / "det-train.safetensors")).items()}
    out, layers = {}, []

    # ---- stem: conv + BN, no activation
    w, b = fuse_conv_bn(st, "backbone.conv1.conv.weight", "backbone.conv1.bn")
    out["backbone.conv1.weight"], out["backbone.conv1.bias"] = w, b

    # ---- backbone blocks
    for name, k, stride in ref.BLOCKS:
        p = f"backbone.{name}"
        ch = st[f"{p}.dw_conv.conv_kxk.0.conv.weight"].shape[0]
        pw_out = st[f"{p}.pw_conv.conv_kxk.0.conv.weight"].shape[0]
        for tag, kk, groups, oc, s in (("dw_conv", k, ch, ch, stride),
                                       ("pw_conv", 1, 1, pw_out, 1)):
            q = f"{p}.{tag}"
            W, B, n = fuse_rep(st, q, kk, groups, oc)
            out[f"{q}.weight"], out[f"{q}.bias"] = W, B
            # Emitted only when it is actually applied; see module docstring.
            if s != 2:
                out[f"{q}.act.lab.scale"] = st[f"{q}.act.lab.scale"].reshape(1)
                out[f"{q}.act.lab.bias"] = st[f"{q}.act.lab.bias"].reshape(1)
            layers.append({"name": q, "k": kk, "stride": s, "groups": groups,
                           "shape": [int(v) for v in W.shape], "branches": n,
                           "post_act": s != 2})
        for c in (1, 2):  # squeeze-excitation, present on blocks6.0 / blocks6.1
            if f"{p}.se.conv{c}.weight" in st:
                out[f"{p}.se.conv{c}.weight"] = st[f"{p}.se.conv{c}.weight"]
                out[f"{p}.se.conv{c}.bias"] = st[f"{p}.se.conv{c}.bias"]

    # ---- taps, neck: plain convs already (RSELayer's in_conv is bias-free, no BN)
    for i in range(4):
        for t in ("weight", "bias"):
            out[f"backbone.layer_list.{i}.{t}"] = st[f"backbone.layer_list.{i}.{t}"]
        for m in ("ins_conv", "inp_conv"):
            out[f"neck.{m}.{i}.in_conv.weight"] = st[f"neck.{m}.{i}.in_conv.weight"]
            for c in (1, 2):
                for t in ("weight", "bias"):
                    out[f"neck.{m}.{i}.se_block.conv{c}.{t}"] = st[f"neck.{m}.{i}.se_block.conv{c}.{t}"]

    # ---- head, binarize branch only; `thresh` is training-only and dropped
    p = "head.binarize"
    w, b = fuse_conv_bn(st, f"{p}.conv1.weight", f"{p}.conv_bn1")
    out[f"{p}.conv1.weight"], out[f"{p}.conv1.bias"] = w, b
    # conv2 is TRANSPOSED: weight is [in, out, kh, kw], so its BN scales axis 1.
    s, t = bn_scale_shift(st, f"{p}.conv_bn2")
    out[f"{p}.conv2.weight"] = st[f"{p}.conv2.weight"] * s.reshape(1, -1, 1, 1)
    out[f"{p}.conv2.bias"] = st[f"{p}.conv2.bias"] * s + t
    out[f"{p}.conv3.weight"] = st[f"{p}.conv3.weight"]
    out[f"{p}.conv3.bias"] = st[f"{p}.conv3.bias"]

    # ---- verification against paddle's exported program, not against ourselves
    cfg = json.loads((FIX / "mobiledet_ref.json").read_text(encoding="utf-8"))
    conf = (cfg["bb_se_hardsigmoid_slope"], cfg["neck_se_hardsigmoid_slope"], cfg["rse_shortcut"])
    oracle = np.load(FIX / "mobiledet_probmap.npz")

    # Two checks, because neither alone is sound. Against the UNFUSED reference
    # the comparison is on raw pre-sigmoid logits — exact, and unaffected by the
    # page's saturation to 0/1, where probability-space error is meaningless and
    # logit-space error is an artifact of clipping. Against PADDLE the
    # comparison is on the probability map and its binarisation, which is what
    # the postprocess actually consumes.
    worst_logit, worst_prob, worst_agree = 0.0, 0.0, 1.0
    for tag in ("noise", "page"):
        x, want = oracle[f"{tag}_input"], oracle[f"{tag}_prob"]
        kf, ku = {}, {}
        got = ref.forward(x, out, conf, keep=kf, fused=True)
        ref.forward(x, st, conf, keep=ku, fused=False)
        dl = float(np.abs(kf["logit"] - ku["logit"]).max())
        dp = float(np.abs(got - want).max())
        agree = float(((got > 0.3) == (want > 0.3)).mean())
        worst_logit, worst_prob = max(worst_logit, dl), max(worst_prob, dp)
        worst_agree = min(worst_agree, agree)
        print(f"  {tag:6s}  fused-vs-unfused max|dlogit| {dl:.3e}   "
              f"vs paddle max|dprob| {dp:.3e}  binarised {agree * 100:.4f}%")
    if worst_logit > 1e-2 or worst_prob > 5e-3 or worst_agree < 1.0:
        print(f"FUSION REJECTED: dlogit {worst_logit:.3e} dprob {worst_prob:.3e} "
              f"agreement {worst_agree * 100:.4f}%")
        return 1
    worst = worst_logit

    save_file(out, str(CACHE / "det-fused.safetensors"))
    (FIX / "ppocrv5_mobile_det_fused.json").write_text(json.dumps({
        "layers": layers, "tensors": len(out), "source_tensors": len(st),
        "max_abs_dlogit_vs_paddle": worst,
        "bb_se_hardsigmoid_slope": conf[0], "neck_se_hardsigmoid_slope": conf[1],
        "rse_shortcut": conf[2],
    }, indent=1), encoding="utf-8")
    print(f"fused {len(layers)} rep layers; tensors {len(st)} -> {len(out)}; "
          f"verified to {worst:.2e} logit against paddle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
