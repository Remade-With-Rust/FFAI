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
pub fn group_lines(mut boxes: Vec<DetBox>) -> Vec<Vec<DetBox>> {
    boxes.sort_by_key(|b| b.y0 + b.y1); // by vertical center ×2
    let mut lines: Vec<Vec<DetBox>> = Vec::new();
    for b in boxes {
        let joined = lines.iter_mut().rev().take(3).find(|line| {
            let (ly0, ly1) = line_span(line);
            let inter = b.y1.min(ly1).saturating_sub(b.y0.max(ly0));
            let min_h = (b.y1 - b.y0).min(ly1 - ly0).max(1);
            inter * 2 > min_h // > 50% vertical overlap with the line band
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
fn find_gutters(lines: &[Vec<DetBox>], page_w: usize) -> Vec<(usize, usize)> {
    if page_w == 0 {
        return Vec::new();
    }
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
    let (min_w, margin) =
        ((page_w as f32 * GUTTER_MIN_FRAC) as usize, (page_w as f32 * MARGIN_FRAC) as usize);
    let mut out = Vec::new();
    let (mut run, mut x) = (None::<usize>, 0usize);
    while x <= page_w {
        let free = x < page_w && !occupied[x];
        match (free, run) {
            (true, None) => run = Some(x),
            (false, Some(s)) => {
                // Interior only: a run reaching either margin is the margin.
                if x - s >= min_w && s > margin && x < page_w - margin {
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
pub fn order_reading(mut lines: Vec<Vec<DetBox>>, page_w: usize) -> Vec<Vec<DetBox>> {
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
    let mut flush = |stripe: &mut Vec<Vec<DetBox>>, out: &mut Vec<Vec<DetBox>>| {
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
