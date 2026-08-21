//! CRAFT postprocess: region/affinity maps → word boxes → text lines.
//!
//! v1 keeps axis-aligned boxes (polygons arrive when a corpus can fail
//! them). The connected-component walk follows the reference algorithm's
//! shape: binarize region|affinity, label components, keep components whose
//! peak region score clears `text_threshold`. The reference's dilation-based
//! box expansion is replaced by a proportional pad at crop time (engine.rs)
//! — same purpose (CRAFT scores hug character cores), simpler machinery,
//! gated by end-to-end CER like every other postprocess choice.

/// One detected text region, in MAP coordinates (half input resolution).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetBox {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize, // exclusive
    pub y1: usize, // exclusive
    pub score: f32,
}

pub const TEXT_THRESHOLD: f32 = 0.7;
pub const LINK_THRESHOLD: f32 = 0.4;
pub const LOW_TEXT: f32 = 0.4;

/// Env-sweepable overrides for the det-stage campaign (photo text peaks
/// below the reference defaults; the sweep decides, per-corpus CER gates).
pub fn text_threshold() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FFAI_DET_TEXT_THR").ok().and_then(|v| v.parse().ok()).unwrap_or(TEXT_THRESHOLD))
}
pub fn low_text() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FFAI_DET_LOW").ok().and_then(|v| v.parse().ok()).unwrap_or(LOW_TEXT))
}
/// Components smaller than this many map pixels are noise.
const MIN_AREA: usize = 10;

/// Extract word-level boxes from the two maps (row-major, `w` × `h`).
#[must_use] 
pub fn extract_boxes(region: &[f32], affinity: &[f32], w: usize, h: usize) -> Vec<DetBox> {
    let mut mask: Vec<bool> =
        (0..w * h).map(|i| region[i] >= low_text() || affinity[i] >= LINK_THRESHOLD).collect();
    let mut boxes = Vec::new();
    let mut stack = Vec::new();

    for start in 0..w * h {
        if !mask[start] {
            continue;
        }
        // BFS one component, tracking bbox + peak region score + area.
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        let (mut peak, mut area) = (0f32, 0usize);
        stack.push(start);
        mask[start] = false;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
            peak = peak.max(region[i]);
            area += 1;
            if x > 0 && mask[i - 1] {
                mask[i - 1] = false;
                stack.push(i - 1);
            }
            if x + 1 < w && mask[i + 1] {
                mask[i + 1] = false;
                stack.push(i + 1);
            }
            if y > 0 && mask[i - w] {
                mask[i - w] = false;
                stack.push(i - w);
            }
            if y + 1 < h && mask[i + w] {
                mask[i + w] = false;
                stack.push(i + w);
            }
        }
        if area >= MIN_AREA && peak >= text_threshold() {
            boxes.push(DetBox { x0, y0, x1, y1, score: peak });
        }
    }
    boxes
}

/// Group word boxes into lines by vertical overlap, then order both levels
/// for reading: lines top-to-bottom, boxes left-to-right within a line.
/// Returns each line's member boxes; the caller unions them per line.
#[must_use] 
pub fn group_lines(mut boxes: Vec<DetBox>) -> Vec<Vec<DetBox>> {
    boxes.sort_by_key(|b| b.y0 + b.y1); // by vertical center ×2
    let mut lines: Vec<Vec<DetBox>> = Vec::new();
    for b in boxes {
        // Both constants here are hand-set and fire on EVERY box: the overlap
        // fraction that merges a box into a line, and how far back the search
        // looks. Env-overridable so their ceilings can be checked without a
        // rebuild -- §8.56's rule, applied to the last unexamined pair.
        let frac = env_f32("FFAI_LINE_OVERLAP", 0.5);
        let back = std::env::var("FFAI_LINE_BACK").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(3usize);
        // §1.4: X-ADJACENCY. The join test above is vertical overlap ALONE, so a
        // box at the right margin joins a line built entirely at the left margin
        // whenever they share a y-band — which is how a two-column page becomes
        // one line per row across both columns. `FFAI_LINE_XGAP` requires the
        // box to sit within N line-heights of the line's current extent.
        //
        // Measured in LINE HEIGHTS, not pixels or page fractions: the quantity
        // that decides whether two words belong to the same line is the space
        // between them relative to the type size, and that is scale-free across
        // a 640 px frame and a 2400 px receipt.
        //
        // DEFAULT 0.0 = OFF, so the shipped path is byte-identical until a gate
        // says otherwise. This condition was built once before (§42), swept over
        // four tolerances, and returned numbers identical to the last digit —
        // because it was measured on `mobiledet-svtr`, where `group_lines` is
        // NEVER CALLED. It is only reachable from the CRAFT path (engine.rs:370)
        // and Composed's orphan boxes (engine.rs:517), so it must be gated on
        // the LIVE and photo corpora, never on OmniDocBench.
        let xgap = env_f32("FFAI_LINE_XGAP", 0.0);
        let joined = lines.iter_mut().rev().take(back).find(|line| {
            let (ly0, ly1) = line_span(line);
            let inter = b.y1.min(ly1).saturating_sub(b.y0.max(ly0));
            let min_h = (b.y1 - b.y0).min(ly1 - ly0).max(1);
            if inter as f32 <= min_h as f32 * frac {
                return false;
            }
            if xgap <= 0.0 {
                return true;
            }
            let lb = line_bbox(line);
            let gap = if b.x0 >= lb.x1 {
                (b.x0 - lb.x1) as f32
            } else if lb.x0 >= b.x1 {
                (lb.x0 - b.x1) as f32
            } else {
                0.0 // overlapping in x — already the same column
            };
            gap <= xgap * (ly1 - ly0).max(1) as f32
        });
        match joined {
            Some(line) => line.push(b),
            None => lines.push(vec![b]),
        }
    }
    for line in &mut lines {
        line.sort_by_key(|b| b.x0);
    }
    // Order lines by vertical CENTER, not top edge: a line whose tallest box
    // carries an ascender/quote sorts above a neighbour under a top-edge key
    // even when its text sits lower — measured as an insert/delete swap pair
    // on the train split.
    lines.sort_by_key(|l| {
        let (y0, y1) = line_span(l);
        y0 + y1
    });
    lines
}

fn line_span(line: &[DetBox]) -> (usize, usize) {
    let y0 = line.iter().map(|b| b.y0).min().unwrap_or(0);
    let y1 = line.iter().map(|b| b.y1).max().unwrap_or(0);
    (y0, y1)
}

/// Union bbox of a line's boxes.
pub fn line_bbox(line: &[DetBox]) -> DetBox {
    DetBox {
        x0: line.iter().map(|b| b.x0).min().unwrap_or(0),
        y0: line.iter().map(|b| b.y0).min().unwrap_or(0),
        x1: line.iter().map(|b| b.x1).max().unwrap_or(0),
        y1: line.iter().map(|b| b.y1).max().unwrap_or(0),
        score: line.iter().map(|b| b.score).fold(0.0, f32::max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_words_one_line_group_and_order() {
        // Two boxes on one row (out of x order), one on a lower row.
        let boxes = vec![
            DetBox { x0: 40, y0: 10, x1: 60, y1: 20, score: 0.9 },
            DetBox { x0: 5, y0: 11, x1: 25, y1: 21, score: 0.9 },
            DetBox { x0: 5, y0: 40, x1: 30, y1: 50, score: 0.9 },
        ];
        let lines = group_lines(boxes);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0][0].x0, 5, "left box first in the line");
        assert_eq!(lines[0][1].x0, 40);
        assert_eq!(lines[1][0].y0, 40, "lower line second");
    }

    #[test]
    fn components_below_thresholds_are_dropped() {
        // 20x10 map: one strong 5x2 blob, one blob that never clears
        // text_threshold, one tiny high blob under MIN_AREA.
        let (w, h) = (20usize, 10usize);
        let mut region = vec![0f32; w * h];
        for y in 2..4 {
            for x in 2..7 {
                region[y * w + x] = 0.9;
            }
        }
        for y in 6..8 {
            for x in 2..7 {
                region[y * w + x] = 0.5; // above LOW_TEXT, below TEXT_THRESHOLD
            }
        }
        region[9 * w + 19] = 0.99; // area 1 < MIN_AREA
        let affinity = vec![0f32; w * h];
        let boxes = extract_boxes(&region, &affinity, w, h);
        assert_eq!(boxes.len(), 1, "{boxes:?}");
        assert_eq!((boxes[0].x0, boxes[0].y0, boxes[0].x1, boxes[0].y1), (2, 2, 7, 4));
    }
}

/// Split one line's bbox into word segments by column projection: a run of
/// map columns whose peak region score stays below `LOW_TEXT` for more than
/// ~35% of the line height is a word gap. Exists for word-level recognizers
/// (PARSeq): CRAFT's affinity links characters ACROSS word gaps at tight
/// rendering spacing, so connected components are line-level here — measured
/// as one box per line on the render corpus.
#[must_use] 
pub fn split_words(region: &[f32], affinity: &[f32], w: usize, line: &DetBox) -> Vec<DetBox> {
    let height = line.y1.saturating_sub(line.y0).max(1);
    let min_gap = ((height as f32) * 0.35).max(1.0) as usize;
    // Peak of max(region, affinity): region is per-CHARACTER gaussians, so
    // it dips between letters of one word; affinity exists to fill exactly
    // those gaps. Splitting on region alone cut words mid-glyph (measured:
    // "October" -> "Oo"+"ccober").
    let col_peak: Vec<f32> = (line.x0..line.x1)
        .map(|x| {
            (line.y0..line.y1)
                .map(|y| region[y * w + x].max(affinity[y * w + x]))
                .fold(0f32, f32::max)
        })
        .collect();
    let mut words = Vec::new();
    let (mut start, mut gap) = (None::<usize>, 0usize);
    for (i, &p) in col_peak.iter().enumerate() {
        if p >= LOW_TEXT {
            if start.is_none() {
                start = Some(i);
            }
            gap = 0;
        } else if let Some(s) = start {
            gap += 1;
            if gap >= min_gap {
                words.push(DetBox { x0: line.x0 + s, y0: line.y0, x1: line.x0 + i + 1 - gap, y1: line.y1, score: line.score });
                start = None;
                gap = 0;
            }
        }
    }
    if let Some(s) = start {
        words.push(DetBox { x0: line.x0 + s, y0: line.y0, x1: line.x1, y1: line.y1, score: line.score });
    }
    if words.is_empty() { vec![*line] } else { words }
}

// ---------------------------------------------------------------------------
// Column-aware reading order
// ---------------------------------------------------------------------------
//
// §8.27 measured a 71-point CER penalty on two-column pages, and the
// token-order probe attributed **55.75 of it to ordering alone** — we read the
// words correctly and emit them interleaved, alternating between columns line
// by line. The fix does not need a document model.
//
// LIVE's `calibrate_bands` already finds horizontal text bands by projecting
// detected line boxes onto the y-axis. A column gutter is the same operation on
// the OTHER axis: a vertical strip that no line box crosses. The boxes are
// already computed, so this costs a histogram over them — against a 3B
// document VLM at ~103 s per page.

/// A line box wider than this fraction of the page is treated as SPANNING —
/// a title, rule, or full-width footer — rather than as column content. Without
/// it a single full-width heading fills the gutter and hides the columns.
const SPAN_FRAC: f32 = 0.60;
/// A gutter narrower than this fraction of the page is inter-word whitespace.
const GUTTER_MIN_FRAC: f32 = 0.025;
/// Gutters touching either margin are margins, not gutters.
const MARGIN_FRAC: f32 = 0.08;

/// Interior vertical gaps that NO column-content box crosses.
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub(crate) fn find_gutters(lines: &[Vec<DetBox>], page_w: usize) -> Vec<(usize, usize)> {
    if page_w == 0 {
        return Vec::new();
    }
    // PAGE-RELATIVE THRESHOLDS ARE DELIBERATE, AND THE OBVIOUS "FIX" IS WORSE.
    //
    // Every fraction here scales with `page_w`, while `xy_cut_pernode` calls
    // this on node SUBSETS. That looks plainly inconsistent, and the reasoning
    // for making it node-relative is sound: on a half-page node a heading
    // spanning the whole node measures under `page_w * SPAN_FRAC`, so it is
    // projected rather than skipped and vetoes every gutter in that node —
    // reintroducing at depth the veto §8.28 added `is_spanning` to prevent.
    //
    // Measured, node-relative thresholds are SIGNIFICANTLY WORSE (§8.54):
    // holdout CER 20.27 % -> 20.66 %, 95 % CI [-0.70, -0.12] excluding zero,
    // 30 pages worse against 8 better. A page-relative fraction is STRICTER
    // when applied to a smaller node, and that strictness is load-bearing: it
    // is the only thing suppressing spurious gutters deep in the recursion,
    // where a handful of lines can leave an accidental vertical band. Relaxing
    // it finds more columns and most of them are not real.
    //
    // So the inconsistency is a depth-dependent brake that happens to be spelt
    // as a page fraction. If it is ever made explicit, it needs to stay a brake
    // — a node-relative minimum with a floor, not a pure proportion.
    let (nx0, nx1) = lines.iter().fold((usize::MAX, 0usize), |a, l| {
        let b = line_bbox(l);
        (a.0.min(b.x0), a.1.max(b.x1))
    });
    if nx1 <= nx0 {
        return Vec::new();
    }
    let _node_w = nx1 - nx0;
    let span_w = (page_w as f32 * SPAN_FRAC) as usize;
    // COUNT crossings, do not just mark occupancy. A single centred element —
    // a page number sitting in the gutter — must not veto a column break, and
    // on this corpus exactly that happened: eroding opened gaps at (800,826)
    // and (873,894) with the footer occupying the 47 px between them.
    let mut crossings = vec![0u32; page_w];
    let mut n_lines = 0u32;
    for line in lines {
        let b = line_bbox(line);
        if b.x1.saturating_sub(b.x0) >= span_w {
            continue; // spanning element: it legitimately crosses any gutter
        }
        // ERODE before projecting. DBNet's boxes arrive unclipped by ~1.5x
        // (see `mobiledet::UNCLIP_LINE`), which widens each line by roughly
        // 0.7x its own height on every side — enough to close a 60 px gutter
        // completely. Measured: before eroding, the free-run scan over a
        // two-column page returned only the two margins, no interior gap at
        // all. The erosion is proportional to box HEIGHT because that is what
        // the unclip distance scales with, and capped as a fraction of width
        // so a short line cannot erode to nothing.
        let h = b.y1.saturating_sub(b.y0) as f32;
        let w = b.x1.saturating_sub(b.x0) as f32;
        let erode = (h * 0.6).min(w * 0.25) as usize;
        let (ex0, ex1) = (b.x0 + erode, b.x1.saturating_sub(erode));
        n_lines += 1;
        for x in ex0.min(page_w)..ex1.min(page_w) {
            crossings[x] += 1;
        }
    }
    // A column of body text is crossed by every one of its lines; a stray
    // centred caption crosses by one. Tolerate the stray.
    // ...but only once there are enough lines for "outlier" to mean anything.
    // A unit test caught this: with three lines, a tolerance of 1 erased an
    // entire single-line column and the gutter merged into the right margin.
    let tol = if n_lines >= 12 { (n_lines / 20).max(1) } else { 0 };
    let occupied: Vec<bool> = crossings.iter().map(|&c| c > tol).collect();
    if std::env::var("FFAI_COL_DEBUG").is_ok() {
        let mut runs = Vec::new();
        let (mut r, mut i) = (None::<usize>, 0usize);
        while i <= page_w {
            let free = i < page_w && !occupied[i];
            match (free, r) {
                (true, None) => r = Some(i),
                (false, Some(s0)) => { runs.push((s0, i)); r = None; }
                _ => {}
            }
            i += 1;
        }
        eprintln!("cols: page_w={page_w} span_w={span_w} free-runs={runs:?}");
    }
    // Env-overridable so the ceiling can be checked without a rebuild. These two
    // are the only find_gutters constants never swept in any campaign, and the
    // grid path they gate is where §8.47's remaining 7.42 pp of ordering slack
    // lives now that the axis rule is exonerated (§8.55, 25/25 correct).
    let (min_w, margin) = (
        (page_w as f32 * env_f32("FFAI_GUTTER_MIN", GUTTER_MIN_FRAC)) as usize,
        (page_w as f32 * env_f32("FFAI_MARGIN", MARGIN_FRAC)) as usize,
    );
    let mut out = Vec::new();
    // Sweep the NODE's extent, not the page's. Outside it every column is free,
    // and those voids are not gutters — they were previously rejected only
    // because they happened to touch x=0 or x=page_w and tripped the margin
    // test, which is an accident of position rather than a reason.
    let (lo, hi) = (0usize, page_w);
    let (mut run, mut x) = (None::<usize>, lo);
    while x <= hi {
        let free = x < hi && !occupied[x];
        match (free, run) {
            (true, None) => run = Some(x),
            (false, Some(s)) => {
                // Interior only: a run reaching either margin is the margin.
                if x - s >= min_w && s > lo + margin && x < hi - margin {
                    out.push((s, x));
                }
                run = None;
            }
            _ => {}
        }
        x += 1;
    }
    out
}

/// Reading order for a page that may be laid out in columns.
///
/// One level of XY-cut, which is all a column layout needs: spanning elements
/// (title, running header, footer) cut the page into horizontal stripes, and
/// each stripe is read column by column. Returns the lines reordered.
///
/// With no gutter this is exactly the previous behaviour — top-to-bottom — so
/// single-column pages cannot regress.
#[must_use] 
pub fn order_reading(lines: Vec<Vec<DetBox>>, page_w: usize) -> Vec<Vec<DetBox>> {
    // FFAI_ORDER selects the strategy so all three can be A/B'd on ONE binary.
    // They could not be before: the recursive cut replaced the one-level cut in
    // place while a sibling session swapped the image decoders in the same
    // window, so the arms were compared across different binaries — which is
    // how a 29.61 % "baseline" came to be set against a 33.02 % result that was
    // really measured against a 74.39 % one.
    match std::env::var("FFAI_ORDER").as_deref() {
        Ok("raster") => sorted_by_y(lines),
        Ok("onelevel") => order_one_level(lines, page_w),
        Ok("adaptive") => adaptive_cut(lines, page_w),
        Ok("vfirst") => xy_cut_vfirst(lines, page_w, 0),
        Ok("vtop") => xy_cut_vtop(lines, page_w, 0),
        Ok("hybrid") => hybrid_order(lines, page_w),
        // DEFAULT is per-node routing (§8.53). It reads 20.27 % CER on the
        // OmniDocBench holdout against the plain recursive cut's 24.77 %, and
        // 21.67 % against 28.74 % on train — both bootstrap intervals exclude
        // zero. `xycut` keeps the previous default reachable for A/B.
        Ok("xycut") => xy_cut(lines, page_w, 0),
        Ok("noselect") => xy_cut_pernode(lines, page_w, 0),
        Ok("cost") => xy_cut_cost(lines, page_w, 0),
        Ok("span") => xy_cut_span(lines, page_w, 0),
        _ => {
            // §51/§52: the v2 objective is the DEFAULT selection since 0.9.0 —
            // +0.0035 ReadOrder with the CI excluding zero, text exactly
            // neutral, confirmed through the full engine on all 1651 pages.
            // `FFAI_ORDER_SELECT=1` restores the 0.8.x reset-score selection.
            let out = if std::env::var("FFAI_ORDER_SELECT").as_deref() == Ok("1") {
                order_by_selection(lines, page_w)
            } else {
                order_by_selection_v2(lines, page_w)
            };
            // §8.156 SPARSE-PAGE GATE, default ON. `FFAI_ORDER_GATE=0` disables.
            // Measured through the engine on 236 holdout pages, one binary and
            // two env settings with the arms interleaved: macro 20.459 % ->
            // 19.897 % (+0.562 pp, 95 % CI [+0.191, +1.009]), 9 pages better and
            // 2 worse. MICRO moves only +0.069 pp — the pages this rescues are
            // small, so character-weighting barely sees them. Macro decides here
            // (§8.119, the metric `ffai-bench` reports), and the two losses are
            // a CJK page we cannot read either way and a 344-character contents
            // page where +3.5 pp is twelve characters.
            if std::env::var("FFAI_ORDER_GATE").as_deref() != Ok("0")
                && sparse_scatter(&out) > env_f32("FFAI_ORDER_GATE_T", ORDER_GATE_T)
            {
                return xy_cut(out, page_w, 0);
            }
            out
        }
    }
}

/// Mean vertical step between consecutive emitted lines, over the ink extent.
///
/// §8.153 decomposed ordering cost into 1.90 pp of SELECTOR REGRET and 2.90 pp
/// of POOL CEILING, and the split is not uniform: on SPARSE pages regret (0.90)
/// exceeds ceiling (0.58), so a different strategy can help, while on dense
/// pages ceiling (2.32) buries regret (1.00) and nothing in the pool does. This
/// separates those populations — pages where it fires have a median of 18 lines
/// and 0.159 coverage against 107 and 0.386 for the rest.
///
/// Deliberately normalised by the INK EXTENT rather than page height: the
/// equivalent page-height form measures identically (holdout +0.493 pp either
/// way, 45 of 52 pages shared) and `order_reading` is not given `page_h`.
/// Changing a shipped signature for a gate that has not yet earned its default
/// is the wrong order of operations.
///
/// Order-dependent by construction — it reads the sequence `order_by_selection`
/// just produced, which is why the gate runs after it and not instead of it.
fn sparse_scatter(ord: &[Vec<DetBox>]) -> f32 {
    if ord.len() < 2 {
        return 0.0;
    }
    let ys: Vec<f32> = ord.iter().map(|l| line_bbox(l).y0 as f32).collect();
    let step: f32 =
        ys.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / (ys.len() - 1) as f32;
    let top = ord.iter().map(|l| line_bbox(l).y0).min().unwrap_or(0) as f32;
    let bot = ord.iter().map(|l| line_bbox(l).y1).max().unwrap_or(0) as f32;
    step / (bot - top).max(1.0)
}

/// Fitted on the 79 TRAIN pages (+0.596 pp), judged once on 224 holdout
/// (+0.493 pp, 95 % CI [+0.13, +0.94], 7 pages better and 2 worse, worst single
/// page -3.5 pp). The same gate also wins with `cost` (+0.346) and `vtop`
/// (+0.304), so it is not one lucky rule/strategy pairing.
const ORDER_GATE_T: f32 = 0.0525;

/// Cut only pages that HAVE column structure; read the rest top-to-bottom.
///
/// §8.37 measured a sign flip, on annotated regions so detection is not in the
/// question: recursive cutting halves raster's inversions on newspapers
/// (29.65 % -> 13.42 %) and LOSES to it on slides (0.14 % -> 2.50 %) and exam
/// papers (0.00 % -> 0.35 %). A page with no columns has no structure for the
/// recursion to find, so every cut it makes is invented, and a slide read
/// top-to-bottom is simply correct.
///
/// The separator costs nothing extra because it is the projection the cut
/// already computes: no vertical valley at the top level means no columns.
/// Note this tests the WHOLE page once — it does not disable the recursion's
/// own per-node choices, which are what handle a headline above two columns.
/// Emit one row of cut telemetry per recursion node, for Prometheus to distill.
///
/// The thresholds this recursion turns on — `H_GAP_MIN`, `V_GAP_MIN`,
/// `SPAN_FRAC`, the erosion `min(0.6h, 0.25w)` — were all set by hand and never
/// derived. That is precisely the "human-guessed heuristic" Prometheus exists to
/// replace, and §8.41 showed the axis choice is where it goes wrong: a figure's
/// horizontal whitespace (~19 line-heights) outbids a real column gutter (~2).
///
/// Traced from the SHIPPED path rather than a reimplementation. §8.39 voided an
/// entire testbed that mirrored this function in Python and diverged from it on
/// 10 pages out of 12, so telemetry that is not the real code is not telemetry.
///
/// The node's line boxes go out with the features because the LABEL — which
/// axis the true reading order wanted — needs the region annotations, which live
/// outside this crate. The probe joins them.
fn trace_node(lines: &[Vec<DetBox>], page_w: usize, depth: usize) {
    use std::io::Write;
    let Ok(path) = std::env::var("FFAI_CUT_TRACE") else { return };
    let mut hs: Vec<usize> =
        lines.iter().map(|l| { let b = line_bbox(l); b.y1.saturating_sub(b.y0) }).collect();
    hs.sort_unstable();
    let lh = hs[hs.len() / 2].max(1) as f32;
    let h = best_gap(lines, lh, page_w, Axis::Horizontal);
    let v = best_gap(lines, lh, page_w, Axis::Vertical);
    let spans = u8::from(lines.iter().any(|l| is_spanning(&line_bbox(l), page_w)));
    let (x0, y0, x1, y1) = lines.iter().fold((usize::MAX, usize::MAX, 0, 0), |a, l| {
        let b = line_bbox(l);
        (a.0.min(b.x0), a.1.min(b.y0), a.2.max(b.x1), a.3.max(b.y1))
    });
    let boxes: Vec<String> = lines
        .iter()
        .map(|l| { let b = line_bbox(l); format!("{},{},{},{}", b.x0, b.y0, b.x1, b.y1) })
        .collect();
    let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    // The page TAG is emitted because matching nodes to pages by `page_w`
    // does not work — many pages share a width, so the join silently attached
    // the wrong regions and the labeller discarded 313 of 315 nodes as
    // unlabellable. A join key that has to be guessed is not a join key.
    let tag = std::env::var("FFAI_TRACE_TAG").unwrap_or_default();
    let _ = writeln!(
        f,
        "{tag}	{depth}	{}	{lh}	{}	{}	{}	{}	{spans}	{page_w}	{x0},{y0},{x1},{y1}	{}",
        lines.len(),
        h.map_or(-1.0, |g| g.1),
        v.map_or(-1.0, |g| g.1),
        h.map_or(-1, |g| g.0 as i64),
        v.map_or(-1, |g| g.0 as i64),
        boxes.join(";")
    );
}

/// Band at the SPANNING ELEMENT, not at the widest gap.
///
/// A 2-column -> 1-column -> 2-column page is the shape every remaining failure
/// has (§8.74's renders). The 1-column part is a real full-width ELEMENT — a
/// masthead, a heading, a wide caption — and that is where the band boundary
/// belongs.
///
/// Every previous cut chose the widest VALLEY instead, which is why a figure
/// wins: a figure is a wide gap with NO TEXT, a masthead is a wide element WITH
/// text, and a whitespace projection cannot tell them apart. Measuring the
/// element rather than the hole makes them different objects, not different
/// sizes — which is why §8.68's threshold shift could never work and this might.
///
/// So: if the node contains a spanning ELEMENT, cut immediately ABOVE it
/// (isolating what precedes), which walks the page down through its real
/// structural breaks. With none the node is a uniform grid and takes the
/// vertical cut.
///
/// **The element grouping is load-bearing** (§8.76). `is_spanning` tests a
/// LINE, and a full-width paragraph is one element but N full-width lines —
/// median 0 spanning regions per page against a max of 35 spanning lines. Cut
/// per line, each cut costs a recursion level, and on 9 % of pages the budget
/// is exhausted peeling single lines off the top; the node then falls through
/// to `sorted_by_y`, i.e. raster, the worst ordering measured in this campaign.
/// That defect — not the idea — is what §8.75 measured at +6.42 pp.
fn xy_cut_span(lines: Vec<Vec<DetBox>>, page_w: usize, depth: usize) -> Vec<Vec<DetBox>> {
    if lines.len() < 2 || depth >= MAX_CUT_DEPTH {
        return sorted_by_y(lines);
    }
    let mut hs: Vec<usize> =
        lines.iter().map(|l| { let b = line_bbox(l); b.y1.saturating_sub(b.y0) }).collect();
    hs.sort_unstable();
    let lh = hs[hs.len() / 2].max(1) as f32;

    let top = lines.iter().map(|l| line_bbox(l).y0).min().unwrap_or(0);
    let cut = element_tops(&lines, page_w, lh)
        .into_iter()
        .find(|&y| y > top + (lh as usize))
        .map(|y| (Axis::Horizontal, y))
        .or_else(|| best_gap(&lines, lh, page_w, Axis::Vertical).map(|g| (Axis::Vertical, g.0)))
        .or_else(|| best_gap(&lines, lh, page_w, Axis::Horizontal).map(|g| (Axis::Horizontal, g.0)));

    let Some((axis, at)) = cut else { return sorted_by_y(lines) };
    let (mut near, mut far) = (Vec::new(), Vec::new());
    for l in lines {
        let b = line_bbox(&l);
        let key = match axis {
            Axis::Horizontal => b.y0.midpoint(b.y1),
            Axis::Vertical => b.x0.midpoint(b.x1),
        };
        if key < at { near.push(l) } else { far.push(l) }
    }
    if near.is_empty() || far.is_empty() {
        return sorted_by_y(if near.is_empty() { far } else { near });
    }
    let mut out = xy_cut_span(near, page_w, depth + 1);
    out.extend(xy_cut_span(far, page_w, depth + 1));
    out
}

/// Choose the cut by what it BREAKS, not by how wide its whitespace is.
///
/// §8.68 refuted shifting the valley-width floors: raising `H_GAP_MIN` so a
/// figure's whitespace stops qualifying costs 5.3 pp, because the horizontal cut
/// is doing necessary work on headers, footers and section breaks. The floor is
/// global; the damage is local. A cut needs to pay for the harm IT does.
///
/// The cost here is column continuity. A horizontal cut through a live
/// multi-column block severs every column that spans it — those lines continue
/// on the far side in the same x-band, and separating them is exactly the
/// `[0,5,1,5,…]` interleave. A horizontal cut under a masthead severs nothing,
/// because no column crosses it. So: count the x-bands present on BOTH sides of
/// a candidate cut, and prefer the cut that severs fewer.
///
/// This is not a threshold and adds no tuned constant — it is a comparison
/// between two candidates that the geometry already produced.
fn xy_cut_cost(lines: Vec<Vec<DetBox>>, page_w: usize, depth: usize) -> Vec<Vec<DetBox>> {
    if lines.len() < 2 || depth >= MAX_CUT_DEPTH {
        return sorted_by_y(lines);
    }
    let mut hs: Vec<usize> =
        lines.iter().map(|l| { let b = line_bbox(l); b.y1.saturating_sub(b.y0) }).collect();
    hs.sort_unstable();
    let lh = hs[hs.len() / 2].max(1) as f32;
    let h = best_gap(&lines, lh, page_w, Axis::Horizontal);
    let v = best_gap(&lines, lh, page_w, Axis::Vertical);

    // How many distinct x-bands appear on both sides of a horizontal cut at `at`?
    // Bands are quantised to a twentieth of the page so a column is one band.
    let severed = |at: usize| -> usize {
        let band = |b: &DetBox| (b.x0.midpoint(b.x1) * 20 / page_w.max(1)) as u32;
        let (mut above, mut below) = (Vec::new(), Vec::new());
        for l in &lines {
            let b = line_bbox(l);
            if b.y0.midpoint(b.y1) < at { above.push(band(&b)) } else { below.push(band(&b)) }
        }
        above.sort_unstable();
        above.dedup();
        below.iter().filter(|x| above.binary_search(x).is_ok()).count().min(above.len())
    };

    let cut = match (h, v) {
        (Some(a), Some(b)) => {
            // A horizontal cut that severs more than one column band is cutting
            // through live columns; prefer the vertical one that separates them.
            if severed(a.0) > 1 { Some((Axis::Vertical, b.0)) } else { Some((Axis::Horizontal, a.0)) }
        }
        (Some(a), None) => Some((Axis::Horizontal, a.0)),
        (None, Some(b)) => Some((Axis::Vertical, b.0)),
        (None, None) => None,
    };
    let Some((axis, at)) = cut else { return sorted_by_y(lines) };

    let (mut near, mut far) = (Vec::new(), Vec::new());
    for l in lines {
        let b = line_bbox(&l);
        let key = match axis {
            Axis::Horizontal => b.y0.midpoint(b.y1),
            Axis::Vertical => b.x0.midpoint(b.x1),
        };
        if key < at { near.push(l) } else { far.push(l) }
    }
    if near.is_empty() || far.is_empty() {
        return sorted_by_y(if near.is_empty() { far } else { near });
    }
    let mut out = xy_cut_cost(near, page_w, depth + 1);
    out.extend(xy_cut_cost(far, page_w, depth + 1));
    out
}

/// Order the page SEVERAL ways and keep the most column-coherent result.
///
/// Every ordering strategy this campaign built wins on some layouts and loses on
/// others — five of them were refuted individually (§8.44, §8.45) yet each was
/// the best available choice on part of the corpus. The problem was never that
/// one rule is right; it is that no rule is right everywhere and nothing chose
/// between them at run time.
///
/// This chooses, using a criterion computable from the OUTPUT alone — no ground
/// truth, no model. A page read down its columns moves rightward and resets left
/// once per column change, so leftward resets ≈ columns − 1. A page read ACROSS
/// its columns oscillates on nearly every line. Fewest resets wins.
///
/// Measured on the 236-page `OmniDocBench` holdout, against `pernode` alone:
///
/// | ordering | CER |
/// |---|---|
/// | `xycut` | 24.77 % |
/// | `vfirst` | 23.25 % |
/// | `pernode` | 20.27 % |
/// | **selection** | **18.87 %** |
///
/// −1.40 pp, 95 % CI [+0.48, +2.52] excluding zero, 23 pages better against 15
/// worse — a positive page count, which `hybrid` and the inversion-judged
/// variants never had. Robust across the reset threshold: every value from 0.03
/// to 0.30 beats `pernode` (−1.32 to −2.03 pp), so the RULE carries it, not the
/// constant.
///
/// Cost is negligible: detection and recognition run once and are shared; only
/// the ordering repeats, over boxes already in hand.
fn order_by_selection(lines: Vec<Vec<DetBox>>, page_w: usize) -> Vec<Vec<DetBox>> {
    if lines.len() < 6 || page_w == 0 {
        return xy_cut_pernode(lines, page_w, 0);
    }
    let eps = env_f32("FFAI_ORDER_SELECT_EPS", 0.08);
    // Each candidate consumes the lines, so they are cloned per attempt. A page
    // is a few hundred small boxes; this is far below the recognition cost that
    // has already been paid by the time ordering runs.
    let candidates = [
        xy_cut_pernode(lines.clone(), page_w, 0),
        xy_cut(lines.clone(), page_w, 0),
        xy_cut_vfirst(lines.clone(), page_w, 0),
        // `order_one_level` — the explicit column grid — was tried in this pool
        // and is NOT kept. Holdout 18.88 % -> 18.65 %, but the CI spans zero
        // ([-0.90, +1.44]) and the page count is NEGATIVE: 3 better, 4 worse,
        // on only 7 pages changed. It is almost never the most column-coherent
        // candidate, so the reset score rarely picks it. -0.23 pp also sits
        // under the ~0.5 pp run-to-run variance of §8.53. Reachable as
        // `FFAI_ORDER=onelevel`.
        // `xy_cut_cost` is deliberately NOT in this pool. Measured on holdout it
        // is 24.73 % standalone against `pernode`'s 20.27 % — significantly
        // worse, CI [-7.99, -1.23] — and adding it as a fourth candidate moved
        // the selection from 18.88 % to 19.66 %. **More candidates is not free:**
        // the reset score prefers column-coherent output, and a wrong ordering
        // can be more column-coherent than a right one. A candidate earns its
        // place by being best somewhere, not by existing.
        //
        // `xy_cut_span` — element-level banding — was tried here as a fourth
        // candidate and is NOT kept (§8.78). The pool does not require a
        // candidate to be good on average, only to be best SOMEWHERE, and this
        // one is not: 18.63 % -> 19.16 %, CI [-1.34, -0.04] excluding zero, and
        // the page count is decisive at **1 better, 16 worse**. It is the third
        // candidate to fail this way after `xy_cut_cost` and `order_one_level`,
        // and always by the same mechanism: the reset score rewards
        // column-coherent output, and a wrong ordering can be more
        // column-coherent than a right one. Reachable as `FFAI_ORDER=span`.
    ];
    let score = |ord: &[Vec<DetBox>]| -> f32 {
        let xs: Vec<f32> = ord
            .iter()
            .map(|l| { let b = line_bbox(l); b.x0.midpoint(b.x1) as f32 / page_w as f32 })
            .collect();
        if xs.len() < 2 {
            return 0.0;
        }
        let resets = xs.windows(2).filter(|w| w[1] < w[0] - eps).count();
        resets as f32 / (xs.len() - 1) as f32
    };
    candidates
        .into_iter()
        .map(|c| { let s = score(&c); (s, c) })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)).map_or_else(Vec::new, |(_, c)| c)
}

/// §51 selection objective: `wreset + 0.5·yback + 2·scat`, lower is better.
///
/// The reset score's documented failure — "a wrong ordering can be more
/// column-coherent than a right one" — is that it counts leftward jumps and
/// sees nothing else. This prices three failure modes at once: `wreset` weighs
/// leftward jumps by their SIZE (a false column reset across the page costs
/// more than a ragged margin), `yback` charges orderings that move UP the page
/// (the signature of a wrong cut emitting a right column before a left one has
/// finished), and `scat` (the §8.156 sparse-scatter) charges sequences that
/// leap vertically. Weights fixed offline on the 1469-page census
/// (`stage34_pool.py`): the optimum is FLAT across 1–2× on every term, so
/// these are round numbers from a plateau, not a knife-edge fit.
fn order_objective(ord: &[Vec<DetBox>], page_w: usize) -> f32 {
    if ord.len() < 2 {
        return 0.0;
    }
    let n1 = (ord.len() - 1) as f32;
    let bb: Vec<DetBox> = ord.iter().map(|l| line_bbox(l)).collect();
    let xs: Vec<f32> =
        bb.iter().map(|b| (b.x0 + b.x1) as f32 / 2.0 / page_w.max(1) as f32).collect();
    let wreset: f32 = xs.windows(2).map(|w| (w[0] - w[1]).max(0.0)).sum::<f32>() / n1;
    let mut hs: Vec<f32> = bb.iter().map(|b| (b.y1 - b.y0) as f32).collect();
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_h = if hs.len() % 2 == 1 {
        hs[hs.len() / 2]
    } else {
        (hs[hs.len() / 2 - 1] + hs[hs.len() / 2]) / 2.0
    }
    .max(1.0);
    let ys: Vec<f32> = bb.iter().map(|b| (b.y0 + b.y1) as f32 / 2.0).collect();
    let yback =
        ys.windows(2).filter(|w| w[1] < w[0] - med_h).count() as f32 / n1;
    wreset + 0.5 * yback + 2.0 * sparse_scatter(ord)
}

/// §51 selection v2 — the same three candidates scored by `order_objective`.
/// A RASTER challenger exists behind `FFAI_ORDER_V2_MARGIN`, REFUTED as a
/// default.
///
/// §1.1's note-page sign-flip (raster beats ordering there, 0.940 vs 0.785
/// contiguity) argued for raster in the menu, and on the CONTIGUITY PROXY a
/// margin-guarded challenger won (+0.004 mean). The evaluator refused it: on
/// the 170 pages the challenger took at margin 0.04, Text^Edit read **−0.0185
/// (EN −0.0375), CI excluding zero** — a raster-assembled multi-column page
/// keeps its blocks index-compact (what contiguity sees) while interleaving
/// them across columns (what block matching sees). The objective swap alone,
/// same instrument, read order +0.0106 / text +0.0004. So the default margin
/// is beyond reach and the challenger fires only when explicitly asked —
/// salvage, if any, is a sparse-page-only condition, not a lower bar.
///
/// Reached only via `FFAI_ORDER_SELECT=2`; the §8.156 sparse gate still runs
/// on whatever this returns.
fn order_by_selection_v2(lines: Vec<Vec<DetBox>>, page_w: usize) -> Vec<Vec<DetBox>> {
    if lines.len() < 6 || page_w == 0 {
        return xy_cut_pernode(lines, page_w, 0);
    }
    let incumbents = [
        xy_cut_pernode(lines.clone(), page_w, 0),
        xy_cut(lines.clone(), page_w, 0),
        xy_cut_vfirst(lines.clone(), page_w, 0),
    ];
    let challenger = sorted_by_y(lines);
    let (inc_obj, inc) = incumbents
        .into_iter()
        .map(|c| (order_objective(&c, page_w), c))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .expect("three candidates");
    let margin = env_f32("FFAI_ORDER_V2_MARGIN", f32::INFINITY);
    if order_objective(&challenger, page_w) < inc_obj - margin {
        challenger
    } else {
        inc
    }
}

/// Route between grid and recursion at every NODE, not once per page.
///
/// §8.44 measured four ordering variants and none transferred to holdout. The
/// closest, `hybrid`, routed per PAGE on the presence of a spanning element and
/// improved the aggregate (4.44 % -> 4.25 %) while making MORE pages worse than
/// better (33 against 25, z = −1.05). The diagnosis there is the reason to try
/// this: a single page routinely contains BOTH a uniform column grid and a
/// structure change — a figure inside a two-column paper is exactly that — so a
/// page-level decision is answering a question that has two different answers
/// on the same page.
///
/// The recursion already isolates those parts. Below a horizontal cut, a band
/// either has clean columns or it does not. So the same discriminator is asked
/// at every node instead of once at the top: a node with no page-wide element
/// and real gutters is a column grid, and gets split into ALL its columns at
/// once (column-major, then recursively ordered within each). Anything else
/// falls through to the ordinary larger-valley cut, which is what handles a
/// masthead over columns and was never the problem there.
///
/// Splitting into all columns at once matters — a binary vertical cut on a
/// three-column band leaves two columns fused, and the next cut has to find
/// them again from a smaller sample.
fn xy_cut_pernode(lines: Vec<Vec<DetBox>>, page_w: usize, depth: usize) -> Vec<Vec<DetBox>> {
    if lines.len() < 2 || depth >= MAX_CUT_DEPTH {
        return sorted_by_y(lines);
    }
    // `any()` looks fragile — ONE spurious wide line abandons the column grid
    // for the whole node, and on `omni-0069` three gutter-merged lines out of
    // ninety cost 56.3 pp (§8.85). Replacing it with a FRACTION was measured
    // and is REFUTED: tolerance 0.01 costs +2.45 pp and 0.50 costs +3.24 pp,
    // every value worse. The strictness is load-bearing for the reason §8.29
    // built it — a genuine spanning headline sliced by a vertical cut costs far
    // more than the rare merged line saves. The defect is upstream, in line
    // grouping, not here.
    let spans = lines.iter().any(|l| is_spanning(&line_bbox(l), page_w));
    if !spans {
        let gutters = find_gutters(&lines, page_w);
        if !gutters.is_empty() {
            let mut edges: Vec<usize> =
                gutters.iter().map(|g| g.0.midpoint(g.1)).collect();
            edges.push(page_w);
            let mut cols: Vec<Vec<Vec<DetBox>>> = vec![Vec::new(); edges.len()];
            for l in lines {
                let b = line_bbox(&l);
                let c = edges.iter().position(|&e| b.x0.midpoint(b.x1) < e).unwrap_or(0);
                cols[c].push(l);
            }
            // A grid that puts everything in one column is not a grid; falling
            // through would recurse forever on the same set.
            if cols.iter().filter(|c| !c.is_empty()).count() > 1 {
                let mut out = Vec::new();
                for c in cols {
                    if !c.is_empty() {
                        out.extend(xy_cut_pernode(c, page_w, depth + 1));
                    }
                }
                return out;
            }
            let mut flat: Vec<Vec<DetBox>> = Vec::new();
            for c in cols {
                flat.extend(c);
            }
            return sorted_by_y(flat);
        }
    }
    xy_cut(lines, page_w, depth)
}

/// Dispatch between the two architectures on the property that separates them.
///
/// Measured on TRAIN, inversions over real detected lines, both architectures
/// against the same baseline:
///
/// | cell | recursive cut | one-level grid |
/// |---|---|---|
/// | `academic_literature` | 12.01 % | **4.51 %** |
/// | magazine | 6.07 % | **4.06 %** |
/// | newspaper | **5.25 %** | 9.13 % |
/// | book | **3.24 %** | 6.51 % |
///
/// They fail on opposite layouts, and the reason is structural rather than
/// incidental. The recursive cut compares whitespace valleys, so a figure's
/// horizontal gap (~19 line-heights on `omni-0038`) outbids a column gutter
/// (~2) and the page is read across its columns. The one-level grid computes
/// ONE gutter set for the whole page, so it cannot be fooled that way — and
/// cannot describe a page whose column structure changes down it, which is what
/// a masthead over three columns is.
///
/// So the discriminator is whether the page has a page-wide element at all. A
/// spanning box means the layout changes vertically and the recursion is the
/// only one of the two that can follow it; no spanning box means a uniform
/// grid, which the projection describes exactly and the recursion keeps
/// mis-cutting. This is the same test §8.29 already uses to protect headlines
/// from vertical cuts, reused as a routing decision instead of a veto.
fn hybrid_order(lines: Vec<Vec<DetBox>>, page_w: usize) -> Vec<Vec<DetBox>> {
    if lines.iter().any(|l| is_spanning(&line_bbox(l), page_w)) {
        xy_cut(lines, page_w, 0)
    } else {
        order_one_level(lines, page_w)
    }
}

/// Find the page's columns FIRST, then cut normally inside them.
///
/// §8.41's `vfirst` preferred the column cut at every node and lost on holdout
/// (4.44 % -> 4.65 %), with `colorful_textbook` regressing 8.06 % -> 10.24 %.
/// The diagnosis behind it was still right — `omni-0038` interleaves because a
/// figure's horizontal gap (~19 line-heights) outbids a real column gutter
/// (~2), so the page is sliced into bands and each band orders its own left and
/// right lines correctly — but the remedy was too broad. Inside a thin band a
/// vertical gap is word spacing, not a column, and preferring it there is what
/// cost `colorful_textbook`.
///
/// Column structure is a property of the PAGE, not of every subregion. So the
/// preference applies at depth 0 only: find the columns, then let the ordinary
/// larger-valley rule work inside each one, where it was never the problem.
/// The spanning exception from §8.29 still holds — a headline across the top
/// must be separated before the page is split down the middle, or the partition
/// assigns it to whichever side its centre lands on.
fn xy_cut_vtop(lines: Vec<Vec<DetBox>>, page_w: usize, depth: usize) -> Vec<Vec<DetBox>> {
    if lines.len() < 2 || depth >= MAX_CUT_DEPTH {
        return sorted_by_y(lines);
    }
    let mut hs: Vec<usize> =
        lines.iter().map(|l| { let b = line_bbox(l); b.y1.saturating_sub(b.y0) }).collect();
    hs.sort_unstable();
    let lh = hs[hs.len() / 2].max(1) as f32;

    let h = best_gap(&lines, lh, page_w, Axis::Horizontal);
    let v = best_gap(&lines, lh, page_w, Axis::Vertical);
    let prefer_v = depth == 0 && !lines.iter().any(|l| is_spanning(&line_bbox(l), page_w));
    let cut = match (h, v) {
        (Some(a), Some(b)) => Some(if prefer_v {
            (Axis::Vertical, b.0)
        } else if a.1 >= b.1 {
            (Axis::Horizontal, a.0)
        } else {
            (Axis::Vertical, b.0)
        }),
        (Some(a), None) => Some((Axis::Horizontal, a.0)),
        (None, Some(b)) => Some((Axis::Vertical, b.0)),
        (None, None) => None,
    };
    let Some((axis, at)) = cut else { return sorted_by_y(lines) };

    let (mut near, mut far) = (Vec::new(), Vec::new());
    for l in lines {
        let b = line_bbox(&l);
        let key = match axis {
            Axis::Horizontal => b.y0.midpoint(b.y1),
            Axis::Vertical => b.x0.midpoint(b.x1),
        };
        if key < at { near.push(l) } else { far.push(l) }
    }
    if near.is_empty() || far.is_empty() {
        return sorted_by_y(if near.is_empty() { far } else { near });
    }
    let mut out = xy_cut_vtop(near, page_w, depth + 1);
    out.extend(xy_cut_vtop(far, page_w, depth + 1));
    out
}

/// Prefer the COLUMN cut over the larger valley, unless something spans.
///
/// §8.40 made `academic_literature` the worst ordering cell (12.01 %, double
/// newspaper) and §8.41 found the mechanism: `xy_cut` takes whichever valley is
/// wider, and a two-column paper with figures between its captions has
/// horizontal gaps of ~19 line-heights against a ~2 line-height gutter. So it
/// cuts horizontally first, into thin bands; each band then orders its own left
/// and right lines correctly, and the page comes out interleaved
/// left-right-left-right. That is `omni-0038`'s emitted sequence exactly.
///
/// A vertical valley is not the same KIND of evidence as a horizontal one. Any
/// gap `best_gap` returns on the vertical axis is a band no box crosses, i.e. a
/// gutter running the full height of this node — structure. A horizontal gap is
/// just whitespace, and whitespace is wider on pages with figures in them.
/// Comparing their widths compares incomparable things.
///
/// The exception is what §8.29 built horizontal-first for: when a headline
/// spans the node, cutting vertically first slices it in half, because the
/// partition assigns it to whichever side its centre lands on. So a spanning
/// box still forces the horizontal cut, and only then do the bands below find
/// their columns.
fn xy_cut_vfirst(lines: Vec<Vec<DetBox>>, page_w: usize, depth: usize) -> Vec<Vec<DetBox>> {
    if lines.len() < 2 || depth >= MAX_CUT_DEPTH {
        return sorted_by_y(lines);
    }
    let mut hs: Vec<usize> =
        lines.iter().map(|l| { let b = line_bbox(l); b.y1.saturating_sub(b.y0) }).collect();
    hs.sort_unstable();
    let lh = hs[hs.len() / 2].max(1) as f32;

    let spans_something = lines.iter().any(|l| is_spanning(&line_bbox(l), page_w));
    let h = best_gap(&lines, lh, page_w, Axis::Horizontal);
    let v = best_gap(&lines, lh, page_w, Axis::Vertical);
    let cut = match (h, v) {
        (Some(a), Some(b)) => {
            if spans_something {
                Some((Axis::Horizontal, a.0))
            } else {
                Some((Axis::Vertical, b.0))
            }
        }
        (Some(a), None) => Some((Axis::Horizontal, a.0)),
        (None, Some(b)) => Some((Axis::Vertical, b.0)),
        (None, None) => None,
    };
    let Some((axis, at)) = cut else { return sorted_by_y(lines) };

    let (mut near, mut far) = (Vec::new(), Vec::new());
    for l in lines {
        let b = line_bbox(&l);
        let key = match axis {
            Axis::Horizontal => b.y0.midpoint(b.y1),
            Axis::Vertical => b.x0.midpoint(b.x1),
        };
        if key < at { near.push(l) } else { far.push(l) }
    }
    if near.is_empty() || far.is_empty() {
        return sorted_by_y(if near.is_empty() { far } else { near });
    }
    let mut out = xy_cut_vfirst(near, page_w, depth + 1);
    out.extend(xy_cut_vfirst(far, page_w, depth + 1));
    out
}

fn adaptive_cut(lines: Vec<Vec<DetBox>>, page_w: usize) -> Vec<Vec<DetBox>> {
    if lines.len() < 2 {
        return sorted_by_y(lines);
    }
    let mut hs: Vec<usize> =
        lines.iter().map(|l| { let b = line_bbox(l); b.y1.saturating_sub(b.y0) }).collect();
    hs.sort_unstable();
    let lh = hs[hs.len() / 2].max(1) as f32;
    if best_gap(&lines, lh, page_w, Axis::Vertical).is_none() {
        return sorted_by_y(lines);
    }
    xy_cut(lines, page_w, 0)
}

fn order_one_level(mut lines: Vec<Vec<DetBox>>, page_w: usize) -> Vec<Vec<DetBox>> {
    lines.sort_by_key(|l| line_bbox(l).y0);
    let gutters = find_gutters(&lines, page_w);
    if gutters.is_empty() {
        return lines;
    }

    // Column band edges, from the gutters' midpoints.
    let mut edges: Vec<usize> = gutters.iter().map(|g| g.0.midpoint(g.1)).collect();
    edges.push(page_w);
    let column_of = |b: &DetBox| edges.iter().position(|&e| b.x0.midpoint(b.x1) < e).unwrap_or(0);
    // Spanning is decided by WIDTH, using the same threshold that excluded the
    // box from the projection. Testing "crosses a gutter" on the RAW box is
    // inconsistent with a gutter computed from ERODED boxes — every unclipped
    // column line then reads as spanning, every line flushes the stripe, and
    // the output stays interleaved while the gutter looks correctly found.
    let span_w = (page_w as f32 * SPAN_FRAC) as usize;
    let spans = |b: &DetBox| b.x1.saturating_sub(b.x0) >= span_w;

    let mut out: Vec<Vec<DetBox>> = Vec::with_capacity(lines.len());
    let mut stripe: Vec<Vec<DetBox>> = Vec::new();
    // A stripe is flushed column-major whenever a spanning element closes it.
    let flush = |stripe: &mut Vec<Vec<DetBox>>, out: &mut Vec<Vec<DetBox>>| {
        stripe.sort_by_key(|l| {
            let b = line_bbox(l);
            (column_of(&b), b.y0)
        });
        out.append(stripe);
    };
    for line in lines {
        if spans(&line_bbox(&line)) {
            flush(&mut stripe, &mut out);
            out.push(line);
        } else {
            stripe.push(line);
        }
    }
    flush(&mut stripe, &mut out);
    out
}


/// Recursive XY-cut.
///
/// §8.28's one-level cut computed ONE set of gutters for the whole page and let
/// spanning elements separate stripes. That is right for a report and wrong for
/// a newspaper: measured on `OmniDocBench`, newspapers score 34.51 % as-is and
/// **13.11 % order-free — 21.40 pp of pure ordering**, and they are the only
/// source where order dominates. A newspaper is not one column grid; it is a
/// headline over a 3-column block beside a boxed sidebar over a 2-column block,
/// and a single page-wide projection cannot describe it.
///
/// So the cut recurses, and at each node it compares the widest HORIZONTAL
/// valley against the widest VERTICAL one and cuts along whichever is larger.
/// That ordering matters and is not arbitrary: cutting a newspaper vertically
/// first would slice through its headline, while cutting horizontally first
/// isolates the headline band and lets each body band find its own columns.
///
/// Both valleys are measured in units of the local median line height, which is
/// what makes them comparable — and is why inter-line leading (~1x) never beats
/// a real column gutter or a section break.
fn xy_cut(lines: Vec<Vec<DetBox>>, page_w: usize, depth: usize) -> Vec<Vec<DetBox>> {
    if lines.len() < 2 || depth >= MAX_CUT_DEPTH {
        return sorted_by_y(lines);
    }
    let heights: Vec<usize> =
        lines.iter().map(|l| { let b = line_bbox(l); b.y1.saturating_sub(b.y0) }).collect();
    let mut hs = heights.clone();
    hs.sort_unstable();
    let lh = hs[hs.len() / 2].max(1) as f32;

    trace_node(&lines, page_w, depth);
    let h = best_gap(&lines, lh, page_w, Axis::Horizontal);
    let v = best_gap(&lines, lh, page_w, Axis::Vertical);
    // Prefer the larger valley; ties go to horizontal, which keeps a headline
    // whole instead of splitting it down the middle.
    // FFAI_ORDER=hfirst forces the horizontal cut whenever BOTH valleys exist.
    //
    // This is what the Prometheus harvest distilled to (§8.55). Of 1178 traced
    // `xy_cut` nodes, only 25 have a live axis choice AND a signal in the true
    // reading order — and ALL 25 prefer horizontal, while the shipped
    // larger-valley rule picks horizontal on only 18. The discovered formula is
    // therefore a constant, which is a real answer rather than a failed fit:
    // once `pernode` routes column layouts to the grid path, the nodes still
    // reaching `xy_cut` are the spanning ones, where §8.29's headline argument
    // says horizontal-first is correct — and the data agrees, unanimously.
    let force_h = std::env::var("FFAI_ORDER").as_deref() == Ok("hfirst");
    let cut = match (h, v) {
        (Some(a), Some(b)) => Some(if force_h || a.1 >= b.1 {
            (Axis::Horizontal, a.0)
        } else {
            (Axis::Vertical, b.0)
        }),
        (Some(a), None) => Some((Axis::Horizontal, a.0)),
        (None, Some(b)) => Some((Axis::Vertical, b.0)),
        (None, None) => None,
    };
    // A page-spanning element — a headline, a rule, a full-width caption — is a
    // structural separator whatever the valley around it measures. §8.28's
    // one-level cut got this right by construction and the valley-only version
    // regressed it: a title with ordinary leading above and below produced no
    // horizontal valley, so the title was sorted into a COLUMN. Split on it
    // explicitly, topmost first, and recurse either side.
    if let Some(i) = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| is_spanning(&line_bbox(l), page_w))
        .min_by_key(|(_, l)| line_bbox(l).y0)
        .map(|(i, _)| i)
    {
        let sep = line_bbox(&lines[i]);
        let (mut above, mut below, mut sep_lines) = (Vec::new(), Vec::new(), Vec::new());
        for (j, l) in lines.into_iter().enumerate() {
            let b = line_bbox(&l);
            if j == i {
                sep_lines.push(l);
            } else if b.y0.midpoint(b.y1) < sep.y0.midpoint(sep.y1) {
                above.push(l);
            } else {
                below.push(l);
            }
        }
        if !above.is_empty() || !below.is_empty() {
            let mut out = xy_cut(above, page_w, depth + 1);
            out.append(&mut sep_lines);
            out.extend(xy_cut(below, page_w, depth + 1));
            return out;
        }
        return sorted_by_y(sep_lines);
    }

    let Some((axis, at)) = cut else { return sorted_by_y(lines) };

    let (mut near, mut far) = (Vec::new(), Vec::new());
    for l in lines {
        let b = line_bbox(&l);
        let key = match axis {
            Axis::Horizontal => b.y0.midpoint(b.y1),
            Axis::Vertical => b.x0.midpoint(b.x1),
        };
        if key < at { near.push(l) } else { far.push(l) }
    }
    // A cut that does not actually divide would recurse forever.
    if near.is_empty() || far.is_empty() {
        return sorted_by_y(if near.is_empty() { far } else { near });
    }
    let mut out = xy_cut(near, page_w, depth + 1);
    out.extend(xy_cut(far, page_w, depth + 1));
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Axis {
    Horizontal,
    Vertical,
}

/// Widest empty band on `axis`, as `(cut position, width in line-heights)`.
///
/// Vertical scans use ERODED boxes and skip page-spanning ones, for the reasons
/// §8.28 measured: `DBNet`'s 1.5x unclip closes a real gutter outright, and one
/// centred page number must not veto a column break. Horizontal scans use the
/// boxes as they are — a line's vertical extent is not inflated the same way,
/// and a heading that spans the page is exactly what a horizontal cut wants to
/// separate rather than ignore.
fn best_gap(lines: &[Vec<DetBox>], line_h: f32, page_w: usize, axis: Axis) -> Option<(usize, f32)> {
    let span_lo = |b: &DetBox| if axis == Axis::Horizontal { b.y0 } else { b.x0 };
    let span_hi = |b: &DetBox| if axis == Axis::Horizontal { b.y1 } else { b.x1 };

    // Extent of the whole set on this axis, so margins are never "gaps".
    let (mut lo, mut hi) = (usize::MAX, 0usize);
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(lines.len());
    // Spanning is judged against the MEDIAN width of this set, not the widest
    // member: with `widest` every box in a uniform two-column block counts as
    // spanning, every box is skipped, and the gutter disappears. A unit test
    // caught exactly that.
    let _ = median_width(lines);
    for l in lines {
        let b = line_bbox(l);
        if axis == Axis::Vertical {
            // Page-spanning elements legitimately cross every gutter.
            if is_spanning(&b, page_w) {
                continue;
            }
            let h = b.y1.saturating_sub(b.y0) as f32;
            let w = b.x1.saturating_sub(b.x0) as f32;
            let e = (h * 0.6).min(w * 0.25) as usize;
            spans.push((b.x0 + e, b.x1.saturating_sub(e)));
        } else {
            spans.push((span_lo(&b), span_hi(&b)));
        }
        lo = lo.min(span_lo(&b));
        hi = hi.max(span_hi(&b));
    }
    if spans.len() < 2 || hi <= lo {
        return None;
    }
    spans.sort_unstable();

    // Sweep for the widest interval covered by nothing.
    let (mut best, mut reach) = (None::<(usize, f32)>, spans[0].1);
    for &(s, e) in &spans[1..] {
        if s > reach {
            let width = (s - reach) as f32 / line_h;
            if best.is_none_or(|(_, w)| width > w) {
                best = Some((reach.midpoint(s), width));
            }
        }
        reach = reach.max(e);
    }
    // The ASYMMETRY between these two floors is the axis bias itself, and it
    // has never been swept. §8.41 diagnosed the defect as a figure's horizontal
    // whitespace outbidding a real column gutter; raising the horizontal floor
    // is the most direct expression of a fix. Env-overridable per §8.56's rule.
    let floor = if axis == Axis::Horizontal {
        env_f32("FFAI_H_GAP_MIN", H_GAP_MIN)
    } else {
        env_f32("FFAI_V_GAP_MIN", V_GAP_MIN)
    };
    best.filter(|&(_, w)| w >= floor)
}

fn median_width(lines: &[Vec<DetBox>]) -> usize {
    let mut w: Vec<usize> =
        lines.iter().map(|l| { let b = line_bbox(l); b.x1.saturating_sub(b.x0) }).collect();
    if w.is_empty() {
        return 0;
    }
    w.sort_unstable();
    w[w.len() / 2]
}

/// Spanning is judged against the PAGE, not against the set.
///
/// Judging it against the set's MEDIAN line width measured 29.61 % -> 35.46 %
/// overall, and academic literature 34.50 % -> 73.49 %: a page thick with short
/// lines (references, captions, equation labels) has a small median, so
/// ordinary body lines clear 1.8x it, every one forces a split, and the reading
/// order shatters. The page width cannot be dragged down that way.
fn is_spanning(b: &DetBox, page_w: usize) -> bool {
    page_w > 0 && (b.x1.saturating_sub(b.x0)) as f32 >= page_w as f32 * SPAN_FRAC
}

/// The top edge of each spanning ELEMENT, in reading order down the node.
///
/// A spanning element is a contiguous run of full-width lines: a masthead, a
/// heading, a full-width paragraph. Only the FIRST line of each run is a
/// candidate band boundary — the interior lines of a paragraph are not
/// structural breaks, and treating them as such is the §8.76 defect that made
/// this whole approach look refuted.
///
/// A run ends at the first line that does not span, or at a vertical gap wider
/// than ordinary leading (`H_GAP_MIN`, the same yardstick the valley cuts use),
/// so a full-width paragraph followed by a full-width heading stays two
/// elements rather than merging into one.
fn element_tops(lines: &[Vec<DetBox>], page_w: usize, lh: f32) -> Vec<usize> {
    let mut by_y: Vec<(usize, usize, bool)> = lines
        .iter()
        .map(|l| { let b = line_bbox(l); (b.y0, b.y1, is_spanning(&b, page_w)) })
        .collect();
    by_y.sort_unstable();

    let lead = (lh * H_GAP_MIN) as usize;
    let mut tops = Vec::new();
    let mut open: Option<usize> = None; // y1 of the previous line in the run
    for &(y0, y1, spanning) in &by_y {
        if !spanning {
            open = None;
            continue;
        }
        match open {
            Some(prev_y1) if y0 <= prev_y1.saturating_add(lead) => {}
            _ => tops.push(y0),
        }
        open = Some(y1.max(open.unwrap_or(y1)));
    }
    tops
}

/// Split a detected box at a white corridor visible in the SOURCE PIXELS.
///
/// §8.85-§8.87: `DBNet` occasionally emits one component spanning two columns, so
/// its text comes out interleaved — `"...Knowledge   except that the range..."`
/// — which no reordering can repair, and the wide box then suppresses the column
/// grid for the whole page. `omni-0069` loses 68 pp to three such lines.
///
/// Five triggers were refused (§8.86, §8.87) and every one of them asked the
/// wrong source. In particular the probability map was tried and rejected
/// because it reads **1.000 at the true gutter** — but that hallucination IS the
/// bug, so the map is the last thing that can arbitrate it. The paper cannot
/// lie: a merge has real whitespace between the columns, a masthead has glyphs.
///
/// Measured on holdout, widest internal white corridor in line-heights:
///
/// | | n | median |
/// |---|---:|---:|
/// | gutter merges | 20 | **1.14** |
/// | legitimate spans | 22 588 | **0.33** |
///
/// A 3.5x separation, against the 1.05-vs-1.03 that box HEIGHT gave. At 0.8x it
/// catches 17 of 20 merges and touches 1.9 % of everything else; at 1.0x, 15 of
/// 20 and 0.7 %.
/// Split a TALL-NARROW box into upright glyph cells — vertical CJK typesetting.
///
/// §43. Detection is not the defect on these pages: on `page-942ac90d` DBNet's
/// probability map reads 0.242 inside the GT text against 0.002 outside, and
/// `boxes_from_probability` passes 16 of 19 components with a median shape of
/// **19 x 1115**. Those boxes are the text columns, found correctly. The loss is
/// in recognition: `svtr_input` scales every crop to height 48 and derives the
/// width from the aspect, so a 1115-pixel column collapses to the 8-px floor and
/// the page emits ONE character.
///
/// **Rotation was tried first and REFUTED.** PaddleOCR turns a crop 90° when
/// height/width exceeds ~1.5, and that rule is for genuinely ROTATED lines. In
/// classical CJK vertical typesetting the glyphs stay UPRIGHT and stack
/// downward, so rotating the column lays every glyph on its side. Measured: 1
/// character became 43, of which nearly all were rotation-symmetric forms
/// (`一`, `。`, `十`, `日`, `）`) against 719 GT characters. The output was the
/// hypothesis refuting itself.
///
/// The transform that matches the typesetting is to CUT the column into cells
/// about as tall as it is wide. Each cell is one upright glyph, which the
/// recognizer already handles — it is just a very short line.
///
/// ## The threshold is 4.0, and it was chosen by blast radius, not by recovery
///
/// A vertical column and a tall glyph are not close: the columns measure
/// **aspect ~59** (19x1115), while the tallest single glyphs in the corpus —
/// digits like `27x69 '4'` and `72x171 '2'` in books — sit at **2.4-2.6**.
/// Counting every box the engine currently emits, by threshold:
///
/// | thr | boxes split OUTSIDE vertical pages | pages hit |
/// |---|---:|---:|
/// | 2.0 | **199** | 92 |
/// | 3.0 | 1 | 1 |
/// | **4.0** | **0** | **0** |
///
/// At 2.0 the split would slice 199 tall digits into meaningless strips. At 4.0
/// nothing the engine currently emits is touched anywhere in the corpus, and
/// recovery is unchanged — 270 / 72 / 446 characters on the three vertical pages
/// against 270 / 72 / 464 at 2.0. So the safe threshold costs 18 characters on
/// one page and removes every measured regression.
///
/// DEFAULT ON at 4.0. `FFAI_VSPLIT_ASPECT=0` disables it, and the disabled path
/// is byte-identical to the pre-§43 engine, so it stays the oracle.
/// §53 — merge WORD FRAGMENTS back into lines on the DBNet path.
///
/// Large-font slide and exam text defeats DBNet's line forming: detection
/// emits one box per WORD, the orderer shuffles the word soup, and Text^Edit
/// pays twice — scrambled within-block sequence and broken block matching.
/// Pages fragmented this way read 3–7× worse than clean pages of the same
/// segment (en PPT 0.276 vs 0.070).
///
/// The merge: cluster boxes into row bands by y-centre, then join x-adjacent
/// neighbours when the horizontal gap is at most `k`× the pair's shorter
/// height AND at least one partner is word-shaped (width < 5× height). The
/// `wordish` condition is load-bearing: an ungated merge priced **−0.0213
/// text** on the oracle (the matcher merges but cannot split — a wrong merge
/// is irrecoverable), while the gated dispatch priced **+0.0079** (CI
/// excluding zero). Two long lines are never joined; gutters are wider than
/// a line height almost everywhere, and this runs BEFORE
/// `split_at_white_corridor`, so a rare gutter-jumping merge is re-split by
/// the corridor guard.
///
/// Merging happens BEFORE recognition, so the recognizer reads the merged
/// crop as one line with full context — strictly better than concatenating
/// separately-recognized words, which is what the +0.0079 floor measured.
///
/// `FFAI_WORD_MERGE` is the gap multiplier; unset/0 = off (byte-identical),
/// `1` = the swept default.
pub fn merge_word_fragments(boxes: Vec<DetBox>) -> Vec<DetBox> {
    // DEFAULT ON since 0.9.0 (§55): +0.0097 text AND +0.0044 order on the full
    // benchmark, both CIs excluding zero, no losing content class. The page
    // gate below is what makes that safe — see it for the ungated numbers.
    // `FFAI_WORD_MERGE=0` restores the 0.8.x behaviour exactly.
    let k = match std::env::var("FFAI_WORD_MERGE").ok().and_then(|v| v.parse::<f32>().ok()) {
        Some(v) if v <= 0.0 => return boxes,
        Some(v) => v,
        None => 1.0,
    };
    if boxes.len() < 2 {
        return boxes;
    }
    let mut hs: Vec<usize> = boxes.iter().map(|b| b.y1.saturating_sub(b.y0)).collect();
    hs.sort_unstable();
    let med_h = hs[hs.len() / 2].max(1) as f32;

    struct Row {
        yc: f32,
        h: f32,
        items: Vec<DetBox>,
    }
    let mut sorted: Vec<DetBox> = boxes;
    sorted.sort_by_key(|b| b.y0 + b.y1); // by y-centre
    let mut rows: Vec<Row> = Vec::new();
    for b in sorted {
        let yc = (b.y0 + b.y1) as f32 / 2.0;
        let h = (b.y1 - b.y0) as f32;
        match rows
            .iter_mut()
            .find(|r| (yc - r.yc).abs() <= 0.5 * med_h.min(r.h))
        {
            Some(r) => {
                r.items.push(b);
                let n = r.items.len() as f32;
                r.yc += (yc - r.yc) / n;
                r.h = r.items.iter().map(|x| (x.y1 - x.y0) as f32).sum::<f32>() / n;
            }
            None => rows.push(Row { yc, h, items: vec![b] }),
        }
    }

    let mergeable = |a: &DetBox, b: &DetBox| {
        let gap = b.x0 as f32 - a.x1 as f32;
        let (ah, bh) = ((a.y1 - a.y0) as f32, (b.y1 - b.y0) as f32);
        let wordish =
            ((a.x1 - a.x0) as f32) < 5.0 * ah || ((b.x1 - b.x0) as f32) < 5.0 * bh;
        gap <= k * ah.min(bh) && wordish
    };

    let mut n_boxes = 0usize;
    for r in &mut rows {
        r.items.sort_by_key(|b| b.x0);
        n_boxes += r.items.len();
    }

    // THE PAGE GATE, and it is load-bearing three times over: ungated the
    // merge priced −0.0213 text; wordish-only −0.0006; gated on this signal
    // +0.0095 text AND +0.0071 order, both CIs excluding zero. The signal is
    // the merge's own reach — the fraction of boxes adjacent to a partner
    // under the merge conditions. Fragmented big-font pages read 0.25+;
    // clean pages 0.02 (multi-column rows fail the gap, full lines fail
    // `wordish`). Plateau across 0.10–0.25; 0.20 is its middle.
    let mut in_chain = 0usize;
    for r in &rows {
        let mut prev_linked = false;
        for w in r.items.windows(2) {
            if mergeable(&w[0], &w[1]) {
                in_chain += usize::from(!prev_linked) + 1;
                prev_linked = true;
            } else {
                prev_linked = false;
            }
        }
    }
    if (in_chain as f32) < env_f32("FFAI_WORD_MERGE_MIN", 0.20) * n_boxes as f32 {
        let mut out: Vec<DetBox> = rows.into_iter().flat_map(|r| r.items).collect();
        out.sort_by_key(|b| (b.y0, b.x0));
        return out;
    }

    let mut out = Vec::with_capacity(n_boxes);
    for r in rows {
        let mut it = r.items.into_iter();
        let mut cur = it.next().expect("row has at least one box");
        for nxt in it {
            if mergeable(&cur, &nxt) {
                cur = DetBox {
                    x0: cur.x0,
                    y0: cur.y0.min(nxt.y0),
                    x1: cur.x1.max(nxt.x1),
                    y1: cur.y1.max(nxt.y1),
                    score: cur.score.min(nxt.score),
                };
            } else {
                out.push(cur);
                cur = nxt;
            }
        }
        out.push(cur);
    }
    out
}

pub fn split_vertical_columns(boxes: Vec<DetBox>) -> Vec<DetBox> {
    let thr = env_f32("FFAI_VSPLIT_ASPECT", 4.0);
    if thr <= 0.0 {
        return boxes;
    }
    let mut out = Vec::with_capacity(boxes.len());
    for b in boxes {
        let (bw, bh) = (b.x1.saturating_sub(b.x0), b.y1.saturating_sub(b.y0));
        if bw == 0 || (bh as f32) <= thr * bw as f32 {
            out.push(b);
            continue;
        }
        // cells about as tall as the column is wide: one glyph each
        let n = ((bh as f32 / bw as f32).round() as usize).max(1);
        for i in 0..n {
            let y0 = b.y0 + bh * i / n;
            let y1 = b.y0 + bh * (i + 1) / n;
            if y1 > y0 + 1 {
                out.push(DetBox { x0: b.x0, y0, x1: b.x1, y1, score: b.score });
            }
        }
    }
    out
}

pub fn split_at_white_corridor(
    boxes: Vec<DetBox>,
    gray: &[f32],
    w: usize,
    h: usize,
) -> Vec<DetBox> {
    let mult = env_f32("FFAI_WHITE_SPLIT", 1.0);
    if mult <= 0.0 || boxes.is_empty() {
        return boxes;
    }
    let ink = env_f32("FFAI_WHITE_INK", 200.0);
    let mut hs: Vec<usize> = boxes.iter().map(|b| b.y1.saturating_sub(b.y0)).collect();
    hs.sort_unstable();
    let lh = hs[hs.len() / 2].max(1);
    let min_run = ((lh as f32) * mult).round().max(2.0) as usize;

    // ONLY boxes wide enough to do the damage. The harm a merge causes is not
    // the interleaved text alone — it is that a box spanning `SPAN_FRAC` of the
    // page makes `is_spanning` fire, which makes `xy_cut_pernode` abandon the
    // column grid for the WHOLE page and fall to raster. A narrow box carrying a
    // white corridor cannot do that, so splitting it is all risk and no reward.
    // This is derived from the damage mechanism, not fitted: the threshold is
    // `SPAN_FRAC` itself.
    let span_w = (w as f32 * SPAN_FRAC) as usize;
    let wide_only = env_f32("FFAI_WHITE_WIDE", 0.0) > 0.0;

    // PASS 1 proposes every cut; the page-level GATE below decides whether any
    // are applied. Splitting first and judging after is impossible — each gate
    // variable is a statistic OVER the proposed cuts, so all must exist before
    // any is acted on.
    let mut proposed: Vec<(usize, usize)> = Vec::new(); // (box index, cut x)
    for (bi, b) in boxes.iter().enumerate() {
        let (x0, x1) = (b.x0.min(w), b.x1.min(w));
        let (y0, y1) = (b.y0.min(h), b.y1.min(h));
        if wide_only && x1.saturating_sub(x0) < span_w {
            continue;
        }
        if x1.saturating_sub(x0) < min_run * 3 || y1 <= y0 {
            continue;
        }
        // A column is white when its DARKEST pixel is still light: one glyph
        // stroke anywhere down the box disqualifies it.
        let white: Vec<bool> = (x0..x1)
            .map(|x| (y0..y1).map(|y| gray[y * w + x]).fold(255.0, f32::min) > ink)
            .collect();
        // Interior only — the margins either side are white by construction.
        let (lo, hi) = (white.len() / 10, white.len() * 9 / 10);
        let (mut run, mut i) = (0usize, 0usize);
        while i <= white.len() {
            if i < white.len() && white[i] && i >= lo && i < hi {
                run += 1;
            } else {
                if run >= min_run {
                    let c = x0 + i - run / 2;
                    if c > b.x0 && c < b.x1 {
                        proposed.push((bi, c));
                    }
                }
                run = 0;
            }
            i += 1;
        }
    }
    if proposed.is_empty() {
        return boxes;
    }

    // PASS 2 — the gate. Per cut: how far the white corridor runs VERTICALLY
    // (a real gutter runs the column; a word space dies at the line above and
    // below), and what fraction of vertically nearby boxes decline to cross it.
    let mut exts: Vec<f32> = Vec::with_capacity(proposed.len());
    let mut resps: Vec<f32> = Vec::with_capacity(proposed.len());
    for &(bi, cx) in &proposed {
        let b = boxes[bi];
        let cy = b.y0.midpoint(b.y1).min(h.saturating_sub(1));
        let col_white = |y: usize| -> bool {
            let (a, z) = (cx.saturating_sub(2), (cx + 3).min(w));
            (a..z).all(|x| gray[y * w + x] > ink)
        };
        let (mut up, mut y) = (0usize, cy);
        while y > 0 && col_white(y) {
            y -= 1;
            up += 1;
        }
        let (mut dn, mut y) = (0usize, cy);
        while y + 1 < h && col_white(y) {
            y += 1;
            dn += 1;
        }
        exts.push((up + dn) as f32 / lh as f32);

        let near: Vec<&DetBox> = boxes
            .iter()
            .filter(|o| {
                let oc = o.y0.midpoint(o.y1) as i64;
                (oc - cy as i64).unsigned_abs() < (12 * lh) as u64
            })
            .collect();
        if near.is_empty() {
            resps.push(1.0);
        } else {
            let crossing = near.iter().filter(|o| o.x0 + 2 < cx && o.x1 > cx + 2).count();
            resps.push(1.0 - crossing as f32 / near.len() as f32);
        }
    }
    let med = |v: &mut Vec<f32>| -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };
    let ext_min = exts.iter().copied().fold(f32::INFINITY, f32::min);
    let ext_med = med(&mut exts.clone());
    let resp_med = med(&mut resps.clone());
    let aspect = h as f32 / w.max(1) as f32;
    let area_mp = (w as f32 * h as f32) / 1.0e6;

    // The gate itself. Two clauses: an unambiguous corridor with near-total
    // agreement from its neighbours, or a merely good corridor backed by page
    // evidence — a large sheet or a very deep minimum — on a page whose shape is
    // not the tall dense format the splitter fails on.
    let pass = (ext_med > env_f32("FFAI_G_EXTMED_HI", 40.0)
        && resp_med > env_f32("FFAI_G_RESP_HI", 0.95))
        || (ext_med > env_f32("FFAI_G_EXTMED_LO", 17.0)
            && resp_med >= env_f32("FFAI_G_RESP_LO", 0.91)
            && (area_mp > env_f32("FFAI_G_AREA", 20.0)
                || ext_min > env_f32("FFAI_G_EXTMIN", 10.0))
            && aspect <= env_f32("FFAI_G_ASPECT", 1.51));
    if std::env::var("FFAI_SPLIT_DEBUG").is_ok() {
        eprintln!(
            "gate cuts={} ext_med={ext_med:.2} ext_min={ext_min:.2} resp_med={resp_med:.3} aspect={aspect:.2} area_mp={area_mp:.1} -> {}",
            proposed.len(),
            if pass { "SPLIT" } else { "blocked" }
        );
    }
    if !pass {
        return boxes;
    }

    let mut out = Vec::with_capacity(boxes.len() + proposed.len());
    for (bi, b) in boxes.iter().enumerate() {
        let mut left = b.x0;
        for &(pi, c) in proposed.iter().filter(|(pi, _)| *pi == bi) {
            let _ = pi;
            if c > left && c < b.x1 {
                out.push(DetBox { x0: left, x1: c, ..*b });
                left = c;
            }
        }
        out.push(DetBox { x0: left, ..*b });
    }
    out
}

fn sorted_by_y(mut lines: Vec<Vec<DetBox>>) -> Vec<Vec<DetBox>> {
    lines.sort_by_key(|l| { let b = line_bbox(l); (b.y0, b.x0) });
    lines
}

/// A horizontal valley must beat ordinary leading, which is ~1 line height by
/// construction; a column gutter is judged against the same yardstick but can
/// be narrower than a section break and still be real.
const H_GAP_MIN: f32 = 1.35;
const V_GAP_MIN: f32 = 0.55;
/// Depth bound: a page that keeps splitting is a page whose structure we have
/// already lost, and unbounded recursion on adversarial input is a hazard.
const MAX_CUT_DEPTH: usize = 12;
/// How much wider than the median line a box must be to count as spanning.
// Nothing reads this today - the span branch it was fitted for is not wired in.
// Kept because the value is a measured result, not a guess; delete it only
// together with the paragraph above.
#[allow(dead_code)]
const SPAN_RATIO: f32 = 1.8;


#[cfg(test)]
mod column_tests {
    use super::*;

    fn bx(x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<DetBox> {
        vec![DetBox { x0, y0, x1, y1, score: 1.0 }]
    }

    #[test]
    fn single_column_order_is_unchanged() {
        let lines = vec![bx(100, 10, 900, 40), bx(100, 60, 900, 90), bx(100, 110, 900, 140)];
        let got = order_reading(lines, 1000);
        assert_eq!(got.iter().map(|l| l[0].y0).collect::<Vec<_>>(), vec![10, 60, 110]);
    }

    #[test]
    fn two_columns_read_left_then_right() {
        // Interleaved by y, as the detector emits them.
        let lines = vec![
            bx(100, 10, 450, 40),   // L1
            bx(550, 12, 900, 42),   // R1
            bx(100, 60, 450, 90),   // L2
            bx(550, 62, 900, 92),   // R2
        ];
        let got = order_reading(lines, 1000);
        let xs: Vec<usize> = got.iter().map(|l| l[0].x0).collect();
        assert_eq!(xs, vec![100, 100, 550, 550], "left column must precede right");
    }

    #[test]
    fn spanning_title_separates_stripes() {
        let lines = vec![
            bx(100, 10, 900, 40),   // full-width title
            bx(100, 60, 450, 90),   // L1
            bx(550, 62, 900, 92),   // R1
            bx(100, 110, 450, 140), // L2
        ];
        let got = order_reading(lines, 1000);
        let ys: Vec<usize> = got.iter().map(|l| l[0].y0).collect();
        assert_eq!(ys, vec![10, 60, 110, 62], "title first, then left column, then right");
    }
}
