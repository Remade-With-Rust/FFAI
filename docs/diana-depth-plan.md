# Diana — Monocular Depth Estimation

**Component:** Diana (`ffai-diana`)
**Task:** `depth` — dense per-pixel metric depth from a single image
**Status:** mapped from the real checkpoint; implementation in progress
**Reference:** Ultralytics 8.4.113, `yolo26{n,s,m,l,x}-depth.pt`

---

## 0. The scoping answer, from the checkpoint rather than the docs

> "I believe it's just a new function call with existing weights."

**Half right, and the half that is right is the expensive half.**

**Wrong on weights.** Depth needs its own checkpoints — `yolo26n-depth.pt`,
12.4 MB, `task: "depth"`, **6.36 M params against detect's 2.57 M**. Detection
weights cannot produce depth; there is no head to run.

**Right on architecture, completely.** Probed side by side:

| | detect | depth |
|---|---|---|
| backbone | 11 layers | **byte-identical YAML** |
| neck | 13 layers | **12 of 13 identical** |
| final layer | `[[16,19,22], 1, 'Detect', ['nc']]` | `[[16,19,22], 1, 'Depth', [256,'log']]` |

**Only the last layer differs**, and it taps the same P3/P4/P5 outputs the
detect head does. Diana's backbone and neck — the bulk of the graph, already
oracle-gated across five tiers and two geometries — are reused unchanged. This
is a **head-only build**, which is the best case available.

---

## 1. The `Depth` head, specified

From `Depth.forward` in Ultralytics 8.4.113 and the nano checkpoint's own
modules:

```python
feats = [self.proj[i](x[i]) for i in range(3)]      # P3,P4,P5 -> 256ch each
out = feats[-1]                                      # start at P5
for i in (1, 0):
    out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
    out = out + feats[i]
    out = self.refine[i](out)
out = self.head(out)                                 # (B,1,H/4,W/4)
depth = torch.exp(out.clamp(-4.0, 5.0))
depth = depth.pow(self.cal_a) * self.cal_b.exp()     # eval only
```

Nano shapes, read off the checkpoint:

| block | layers |
|---|---|
| `proj` | 3 × Conv1x1 + BN + SiLU — 64→256, 128→256, 256→256 |
| `refine[i]` | 2 × Conv3x3 + BN + SiLU, 256→256 |
| `head` | Conv3x3 256→128 · **ConvTranspose2d 128→128 k2 s2** · Conv3x3 128→64 · Conv2d 64→1 k1 |
| calibration | `cal_a = 1.0`, `cal_b = -0.19384765625` → `exp(cal_b) = 0.8237834` |

Output is `(1, 1, H/4, W/4)` in **metres**, unbounded by construction —
`exp` of a clamped logit rather than a scaled sigmoid, which is what lets one
model span indoor and outdoor scenes.

### `refine[2]` is never executed

The loop runs `i ∈ {1, 0}`, so `refine[2]` is dead weight in every released
checkpoint — 1.2 M of the nano model's 6.36 M parameters. Noted because a
strict loader must still *accept* those tensors, and because a converter that
silently drops them would produce a manifest that no longer round-trips.
**Load them, never run them**, and record the fact rather than discovering it
twice.

---

## 2. What Diana does not have yet

Everything except two primitives is already in the crate.

| need | status |
|---|---|
| Conv1x1 + BN + SiLU | **have** — `ConvAct` / `ConvKind::Pointwise` |
| Conv3x3 + BN + SiLU | **have** — `ConvKind::Dense3x3` |
| bare Conv2d 1x1 with bias | **have** — `ConvAct` with `act = false` |
| residual add, tensor plumbing | **have** |
| **bilinear ×2, `align_corners=True`** | **NEW** |
| **ConvTranspose2d, k=2 s=2 p=0** | **NEW** |
| `exp` / `clamp` / `pow` elementwise | trivial; `exp` already exists in `silu` |

**Bilinear with `align_corners=True`** is the fiddly one: the comment in
Ultralytics' source says it is "baked into the released depth weights", so the
sampling convention is not a free choice. `align_corners=True` maps output
pixel `i` to input coordinate `i·(in−1)/(out−1)`, which is *not* the
half-pixel convention used elsewhere in the codebase. Getting it wrong shifts
the whole depth map by half a pixel per pyramid level and will not fail
loudly.

**ConvTranspose2d k=2 s=2 p=0** is the easy one: stride equals kernel, so
output tiles never overlap and it is exactly "each input pixel becomes a
weighted 2×2 block". No accumulation, no gradient-style scatter.

---

## 3. Build order, each step gated

1. **Converter** — teach `tools/diana_convert.py` the `Depth` head: emit
   `proj`, `refine`, `head`, and the two calibration scalars into safetensors
   plus a manifest that declares `task = "depth"`. Fails closed on any shape
   mismatch, as the detect path already does.
   *Gate:* round-trip — every tensor in the `.pt` appears in the manifest with
   matching shape, `refine.2` included.

2. **`bilinear2x_align_corners`** — its own module, its own test against a
   hand-computed 3×3 → 6×6 case, before it is wired to anything.
   *Gate:* matches PyTorch's `F.interpolate(..., align_corners=True)` to 1e-6
   on a fixture dumped from the reference.

3. **`convtranspose2x`** — same treatment.
   *Gate:* matches `nn.ConvTranspose2d(k=2,s=2)` on a dumped fixture.

4. **`DepthHead`** — assemble, load strict.
   *Gate:* full-graph oracle against a reference depth map dumped from
   Ultralytics for a tracked fixture image, per-pixel, with a stated bound.

5. **Engine + task surface** — `DepthEngine` trait in `ffai-core`, a
   `Yolo26Depth` engine, `ffai depth <image>` in the CLI, and depth output
   written as a 16-bit PNG or raw f32.
   *Gate:* the existing determinism test shape — same input, byte-identical
   output, at any thread count.

6. **Bench** — `ffai bench depth` with the four gates. The reference is
   `ultralytics ... task=depth`; the quality metric is standard monocular
   depth error (AbsRel / δ<1.25) rather than mAP, so `ffai-bench` needs a new
   scorer.
   *Gate:* a `bench/ledger.jsonl` line. **No claim before that line exists.**

---

## 4. What this does not get for free

* **Five tiers, but not five oracles.** The backbone/neck reuse is exact, so
  `s/m/l/x` should follow `n` with no code change — but "should" is not
  measured, and the c3k promotion caught exactly that assumption once already.
  Each tier gets its own oracle line.
* **Latency is unknown and probably worse than detect.** The head is 3.8 M
  parameters against the detect head's ~0.2 M, and it runs two 256-channel
  refine stages at P4 and P3 resolution — the largest feature maps in the
  graph. Every optimisation in `docs/whys/diana-latency.md` was measured on
  the detect head's shape and none of it transfers by assumption.
* **Weights stay AGPL.** `-depth.pt` checkpoints carry the same licence as
  the detect ones, so the same rule applies: converted offline by the user,
  never vendored, never redistributed.


---

## The bench, scoped — and it needs no ground truth

Step 6 assumed `ffai bench depth` required a ground-truth corpus (NYU/KITTI)
and an AbsRel / delta<1.25 scorer. Scoping it says otherwise, and the reason
matters for what the gate would be FOR.

**Quality is already measured, and more strongly than GT would measure it.**
The oracle compares our depth map against Ultralytics' **per pixel**, at all
five tiers, worst relative error **7.4e-6**. A ground-truth metric would grade
Ultralytics' WEIGHTS against nature; this grades our PORT against Ultralytics,
which is the question a reimplementation has to answer. AbsRel against NYU
would go up or down with the model, not with us.

So a depth bench would add **speed and footprint**, both of which need no
ground truth at all — the same reference adapter, the same corpus, timed.

**Not built.** It is a few hours of harness work: `Task::Depth` through
`ffai-bench`, a depth reference adapter, and a scorer whose quality column is
reference-agreement rather than mAP. Worth doing, but it buys a ledger LINE
for numbers we do not currently claim, where the same hours spent on
documentation made two shipped features discoverable.

**Until it exists, depth carries no speed or memory claim** — only the
correctness one, which is stated everywhere depth is mentioned.
