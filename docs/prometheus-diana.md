# Prometheus for Diana — assessment and Stage 1 deployment

**Verdict: yes, and for one specific function.**

## Why it fits

Prometheus's first keeper was MP3's `x^(3/4)` — "the hot transcendental of
the quantize stage", located at 62.7 % of encode and strength-reduced
bit-exactly. Diana has the same shape of problem, and three measurements
taken during the latency campaign say so rather than an analogy:

1. **`silu::exp_fast` is 30.7 % of serial detect** — the largest single line
   item, after three rounds of hand optimisation (magic-number rounding, a
   double-write removal, an explicit AVX2 kernel).
2. **It is COMPUTE-bound, not memory-bound.** Measured 1.12-1.61 Gelem/s
   against a ~2.4 Gelem/s ceiling implied by its own op count. The only
   remaining lever is FEWER OPERATIONS.
3. **Fewer operations is an ACCURACY trade, and this oracle is tight.** A
   single FMA fusion — one rounding instead of two — breached it by 2 %.
   Hand-guessing a shorter polynomial is not safe here.

Point 3 is what earns Prometheus its place over a hand-fitted minimax
polynomial: `prom-prove` bounds the error with SMT over the interval the
harvest shows is actually used, instead of over a defensive one.

## Stage 1 — harvested (this commit)

`--features prometheus-telemetry`, off by default, zero-cost when off (hook
compiles to an empty body; verified absent from the default rlib), additive
only. 14,279,920 activation inputs from 12 real corpus images:

| percentile | value |
|---|---:|
| p1 | -3.67 |
| p50 | **-0.08** |
| p99 | +4.61 |
| p99.998 span | **[-21.0, +25.4]** |
| **what the kernel defends** | **[-180, +180]** |

**99 % of activations lie within +/-5. The kernel is sized for a range 8x
wider than the network produces**, and only 111 samples in 14.3 M fell
outside +/-40.

## Stage 2 — the target spec for `prom-distill`

The naive read is "shorten the polynomial", and it is wrong: range reduction
means the Horner always evaluates on `f in [-0.5, 0.5]` whatever the input,
so the harvested range does not shorten it.

What the harvest DOES say is that the composition is the wrong shape.
`silu(x) = x / (1 + exp(-x))` is near-trivial outside the measured band —
approximately `0` below -10 and approximately `x` above +10 — so the whole
`exp` + reciprocal composition is being paid to compute something that only
varies interestingly on a narrow interval.

**Target:** discover a direct closed form for `silu` over the harvested
distribution, minimising operation count subject to a proved error bound.

* **Inputs:** `x`, weighted by `corpora/refs/fixtures/silu_activation_dist.csv`
* **Output:** `silu(x)`
* **Current cost:** ~20 ops — clamp (2), mul (1), magic round (2), sub (1),
  degree-5 Horner (10), exponent-field write (3), divide (1)
* **Error budget:** the full-graph oracle's `head_boxes` bound, which a 2 %
  overshoot already breached — so the SMT bound must be tighter than the
  tolerance, not equal to it
* **Gates:** five-tier oracle, determinism, `detect_parity` at n/m/l/x, and
  the paired A/B harness with its null arm

## Not targets, and why

* `candle_nn::ops::sigmoid` in the head — 672 K scores/image, twice, but
  decode is only 2.1 % of serial. Worth ~1 %. Second in line.
* `powf(-0.5)` in attention — evaluated once per forward. Nothing.
* The two-stage top-k — a SEARCH, not a formula. Wrong tool.

## Coupling

One-way, per Prometheus's own rule: the refinery may depend on Diana; Diana
never references the refinery. What comes back is generated formula code —
small, self-contained, provenance-commented — landed behind a gate like every
other brick.
