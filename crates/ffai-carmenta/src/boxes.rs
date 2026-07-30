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
