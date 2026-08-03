# Figure vs gutter: a staged test plan

**Question.** Can we spend compute — of which we have a lot spare — to tell a
*figure* from a *column gutter*, and does that convert into CER?

**Why it might work.** At ordering time Carmenta sees only boxes, so a column
gutter and a figure are the same thing: an absence of boxes. The figure's
whitespace is typically ~19 line-heights against a gutter's ~2, so it wins the
valley contest, the page is cut into horizontal bands, and each band emits
left-column and right-column text interleaved (`omni-0038`: `[0,5,1,5,1,5…]`).
Pages with figures carry **11.13 pp** of ordering slack against **4.77 pp**
without (§8.60).

**Why it might not.** §8.61 measured the ink discriminator separating cleanly
(bimodal, empty valley between 0.004 and 0.036) but firing on only **~1.8
candidate bands per page**, with `find_gutters` invoked on **6 of 12** pages at
all. Against a **~0.5 pp run-to-run CER variance** (§8.53) the expected effect
may sit under the noise floor of the instrument that would certify it.

**The budget that makes this worth revisiting.** Measured against the competitor
rather than against ourselves:

| | today | headroom before we lose an axis |
|---|---:|---|
| memory | 373–448 MiB | **8745 MiB** (Unlimited-OCR) — ~20x |
| pages/s | 0.15 | **0.01** (Unlimited-OCR) — ~15x |

So a heavier discriminator is affordable. Cost is not the constraint; **evidence
is**.

---

## Stage 0 — Ceiling. Is the prize big enough to chase?

**Cost:** ~30 min, no code.

Simulate *perfect* figure/gutter discrimination and measure the CER it buys.
Figures are not annotated in OmniDocBench, so derive figure area as page area
covered by **no annotated text region**, above a size floor, bounded above and
below by text. Reject any candidate gutter overlapping it, then re-order and
score.

```
.tools-bench/figure_ceiling.py holdout
```

**Report:** shipped CER, oracle-figure-rejection CER, delta, bootstrap CI.

**KILL GATE:** if the delta is **< 1.0 pp** (twice the noise floor), stop. The
mechanism is real but too small to certify, and no implementation can beat its
own ceiling. Record and close.

---

## Stage 1 — Does the signal separate on the bands that MATTER?

**Cost:** ~20 min, no plumbing.

§8.61 measured ink over *all* box-free bands. That is the wrong population. The
only bands that matter are the ones `find_gutters` **accepts** — the rest are
already rejected on width or margin. Dump accepted gutters via the existing
`FFAI_COL_DEBUG`, sample ink in exactly those, and split by whether the page has
figure captions.

**Report:** ink distribution for accepted gutters on figure pages vs non-figure
pages; the separation threshold and its false-positive rate.

**KILL GATE:** if accepted gutters on figure pages are **not** inkier than those
on clean pages, the discriminator cannot act where it needs to. Stop.

---

## Stage 2 — Minimal plumbing, behind a switch

**Cost:** ~2 h. Only reached if Stages 0 and 1 both pass.

`find_gutters` takes boxes and `page_w`; it has no pixels. Rather than threading
the image through `order_reading` into every recursion node — a wide API change
across `boxes.rs` and all callers — compute an **ink column profile ONCE per
page** at detection time (a `Vec<f32>`, one mean-ink value per x, ~7 KB for a
1700 px page) and pass that single slice down. The detector already has the
decoded image in hand there, so the sample is nearly free and the recursion
carries a slice instead of an image.

```rust
// off by default; the previous behaviour is the default path
FFAI_INK_GUTTER=1        // enable
FFAI_INK_GUTTER_THR=0.02 // reject a gutter whose mean ink exceeds this
```

**Design constraints, from things that already went wrong:**

* **Default OFF.** §8.54 shipped an "obviously correct" gutter change that
  measured worse; a switch means the next one costs nothing to abandon.
* **Node-scale, not page-scale.** §8.39 voided a whole testbed by applying
  page-scale units inside a node. Sample the profile over the NODE's y-extent,
  not the page's.
* **No new tuned constant without its own sweep.** The threshold comes from
  Stage 1's measured valley, not from taste.

---

## Stage 3 — Trial, on the metric that ships

**Cost:** ~1 h of runs.

1. **Train first**, with a paired sign test AND a bootstrap CI. §8.44 wasted a
   holdout run on a train margin that was never significant.
2. **Holdout decides.** Five ordering variants won on train and lost on holdout.
3. **CER, never inversions.** §8.66: inversion rate does not predict CER — a
   ranker at 5.26 % inversions produced *worse* CER than random swaps at
   15.61 %. Proxies localise defects; they must not accept changes.
4. **Run in isolation.** Concurrent load corrupted a speed gate and produced a
   false byte-diff on `omni-0130` earlier today.
5. **Check the page count.** Four probes this session returned plausible numbers
   while silently measuring nothing. If dumps are missing, the comparison is
   void.

**ACCEPT:** holdout CER improves with a bootstrap CI excluding zero, and the
page-level count is not negative.
**REJECT:** anything else. Keep the switch, default OFF, record the numbers.

---

## What this is not

It is **not** the layout model. §8.64 measured that region-based ordering must be
**near-oracle** to beat line-level ordering — a learned ranker at 5.26 % region
inversions scored **30.63 %** against shipped **20.27 %**, because a misplaced
region relocates a whole block. This plan buys one specific discrimination
cheaply; it does not buy layout understanding, and it should not be described as
if it did.

Expected value, stated honestly before starting: **Stage 0 is the most likely
place this dies.** The mechanism is real and the frequency is thin.
