//! Word boxes for a line-level recognizer, from where its decoder emitted.
//!
//! The document engines recognise whole LINES, so `OcrLine::words` used to be
//! empty and a caller drawing a word-level text layer had to re-split the line
//! itself. PdfCraft's attempt with [`crate::image::split_ink_words`] matched
//! the word count on 38 of 472 lines: a fixed gap rule cannot tell a word space
//! from the space inside "r n" at one size and "W A" at another.
//!
//! The recognizer already knows. CTC walks the crop left to right one
//! timestep per column band, so each emitted character comes with the frames
//! that emitted it, and the decoded SPACES say which characters form a word.
//! That fixes how many words there are and roughly where each one sits; the
//! pixels then say exactly where its ink is:
//!
//! 1. between two neighbouring words, the cut is the widest ink-free column
//!    run between the last frame of one and the first frame of the next (the
//!    midpoint when there is none);
//! 2. inside its cuts, a word's box is snapped to its ink — the first and last
//!    ink columns, and the band of ink rows carrying the most ink, which keeps
//!    the neighbouring lines that a DB line box (unclipped to ~2.5x the ink
//!    height) reaches into out of it.
//!
//! The words' texts are the line's text split at its spaces, so joining them
//! with single spaces gives the line back.

use ffai_core::types::{BoundingBox, OcrWord};

/// One decoded character run and the CTC frames (inclusive) that emitted it.
#[derive(Debug, Clone, PartialEq)]
pub struct Emit {
    pub text: String,
    pub t0: usize,
    pub t1: usize,
    /// Probability of the emitting class at its first frame.
    pub p: f32,
}

/// Ink mask over a crop of the grayscale image: a pixel is ink when it is
/// further from the crop's median (its paper) than 40 % of the crop's 98th
/// percentile deviation, and at least 10 grey levels — so light-on-dark and
/// grey-on-grey lines work as well as black on white.
struct Ink {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    on: Vec<bool>,
}

impl Ink {
    fn new(gray: &[f32], img_w: usize, (x0, y0, x1, y1): (usize, usize, usize, usize)) -> Option<Self> {
        let (w, h) = (x1.checked_sub(x0)?, y1.checked_sub(y0)?);
        if w == 0 || h == 0 {
            return None;
        }
        let px = |x: usize, y: usize| gray[(y0 + y) * img_w + x0 + x];
        let mut sample: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).step_by(7).map(|(x, y)| px(x, y)).collect();
        sample.sort_by(f32::total_cmp);
        let paper = sample[sample.len() / 2];
        let mut dev: Vec<f32> = sample.iter().map(|v| (v - paper).abs()).collect();
        dev.sort_by(f32::total_cmp);
        let p98 = dev[(dev.len() * 98 / 100).min(dev.len() - 1)];
        let thr = (0.4 * p98).max(10.0);
        let on = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| (px(x, y) - paper).abs() > thr).collect();
        Some(Self { x0, y0, w, h, on })
    }

    fn at(&self, x: usize, y: usize) -> bool {
        self.on[y * self.w + x]
    }

    /// Whether column `x` (crop coordinates) has ink in rows `ys`.
    fn col(&self, x: usize, ys: std::ops::Range<usize>) -> bool {
        ys.into_iter().any(|y| self.at(x, y))
    }
}

/// Split `emits` (one line's decode) into words and box each one on the page.
///
/// `crop` is the rectangle the recognizer read, in image pixels, `t` its
/// number of timesteps, and `fold` the same per-string normalisation applied
/// to the line's text, so word texts agree with it.
#[must_use]
pub fn boxes(
    emits: &[Emit],
    t: usize,
    gray: &[f32],
    img_w: usize,
    crop: (usize, usize, usize, usize),
    fold: impl Fn(&mut String),
) -> Vec<OcrWord> {
    // Group emissions into words at the spaces.
    let mut groups: Vec<&[Emit]> = Vec::new();
    let mut start = None;
    for (i, e) in emits.iter().enumerate() {
        let space = e.text.chars().all(char::is_whitespace);
        match (space, start) {
            (true, Some(s)) => {
                groups.push(&emits[s..i]);
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(s) = start {
        groups.push(&emits[s..]);
    }
    if groups.is_empty() || t == 0 {
        return Vec::new();
    }
    let (cx0, cy0, cx1, cy1) = crop;
    let cw = (cx1 - cx0) as f32;
    // Frame edges in crop columns.
    let left = |f: usize| ((f as f32 / t as f32) * cw) as usize;
    let right = |f: usize| ((((f + 1) as f32 / t as f32) * cw).ceil() as usize).min(cx1 - cx0);
    let ink = Ink::new(gray, img_w, crop);

    // Cuts between neighbouring words.
    let mut cuts = vec![0usize];
    for pair in groups.windows(2) {
        let a = right(pair[0].last().map_or(0, |e| e.t1));
        let b = left(pair[1].first().map_or(0, |e| e.t0)).max(a);
        let cut = ink
            .as_ref()
            .and_then(|ink| {
                // Widest ink-free run of columns in [a, b).
                let (mut best, mut run_start, mut best_len) = (None, None, 0usize);
                for x in a..=b.min(ink.w) {
                    let free = x < b.min(ink.w) && !ink.col(x, 0..ink.h);
                    match (free, run_start) {
                        (true, None) => run_start = Some(x),
                        (false, Some(s)) => {
                            if x - s > best_len {
                                best_len = x - s;
                                best = Some((s + x) / 2);
                            }
                            run_start = None;
                        }
                        _ => {}
                    }
                }
                best
            })
            .unwrap_or((a + b) / 2);
        cuts.push(cut.max(*cuts.last().unwrap_or(&0)));
    }
    cuts.push(cx1 - cx0);

    groups
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let mut text: String = g.iter().map(|e| e.text.as_str()).collect();
            fold(&mut text);
            let conf = g.iter().map(|e| e.p).sum::<f32>() / g.len() as f32;
            let (lo, hi) = (cuts[i], cuts[i + 1].max(cuts[i] + 1));
            let bbox = ink.as_ref().and_then(|ink| snap(ink, lo, hi)).unwrap_or(BoundingBox {
                x: (cx0 + lo) as f32,
                y: cy0 as f32,
                width: (hi - lo) as f32,
                height: (cy1 - cy0) as f32,
            });
            OcrWord { text, bbox: Some(bbox), confidence: Some(conf) }
        })
        .collect()
}

/// The ink box of crop columns `[lo, hi)`: the band of ink rows carrying the
/// most ink (gaps up to a tenth of the crop height bridged, so an accent stays
/// with its letter), then the first and last ink columns in that band.
///
/// Most ink, not the densest row: the DB line box reaches into the lines above
/// and below, and a neighbour's full row of descenders can out-count any one
/// row of this word, but never this word's whole band.
fn snap(ink: &Ink, lo: usize, hi: usize) -> Option<BoundingBox> {
    let hi = hi.min(ink.w);
    if lo >= hi {
        return None;
    }
    let rows: Vec<usize> = (0..ink.h).map(|y| (lo..hi).filter(|&x| ink.at(x, y)).count()).collect();
    let tol = (ink.h / 10).max(1);
    // (top, bottom, mass) of each gap-bridged band.
    let mut bands: Vec<(usize, usize, usize)> = Vec::new();
    let mut cur: Option<(usize, usize, usize)> = None;
    for (y, &n) in rows.iter().enumerate() {
        if n == 0 {
            continue;
        }
        cur = match cur {
            Some((t, b, m)) if y - b <= tol + 1 => Some((t, y, m + n)),
            Some(band) => {
                bands.push(band);
                Some((y, y, n))
            }
            None => Some((y, y, n)),
        };
    }
    bands.extend(cur);
    let &(top, bot, _) = bands.iter().max_by_key(|b| b.2)?;
    let cols: Vec<usize> = (lo..hi).filter(|&x| ink.col(x, top..bot + 1)).collect();
    let (&x0, &x1) = (cols.first()?, cols.last()?);
    Some(BoundingBox {
        x: (ink.x0 + x0) as f32,
        y: (ink.y0 + top) as f32,
        width: (x1 + 1 - x0) as f32,
        height: (bot + 1 - top) as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(text: &str, t0: usize, t1: usize) -> Emit {
        Emit { text: text.into(), t0, t1, p: 0.9 }
    }

    /// Two ink blobs on white with a wide gap; the decoder put its space in
    /// the gap. Each word's box must be exactly its blob.
    #[test]
    fn words_snap_to_their_ink() {
        let (w, h) = (100usize, 30usize);
        let mut gray = vec![255f32; w * h];
        for y in 10..20 {
            for x in (10..30).chain(60..90) {
                gray[y * w + x] = 0.0;
            }
        }
        // 10 timesteps over the 100-px crop: "ab" in frames 1-2, space 4-5,
        // "cd" in frames 6-8.
        let emits = [e("a", 1, 1), e("b", 2, 2), e(" ", 4, 5), e("c", 6, 7), e("d", 8, 8)];
        let words = boxes(&emits, 10, &gray, w, (0, 0, w, h), |_| {});
        assert_eq!(words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), ["ab", "cd"]);
        let b0 = words[0].bbox.unwrap();
        let b1 = words[1].bbox.unwrap();
        assert_eq!((b0.x, b0.y, b0.width, b0.height), (10.0, 10.0, 20.0, 10.0));
        assert_eq!((b1.x, b1.y, b1.width, b1.height), (60.0, 10.0, 30.0, 10.0));
    }

    /// A neighbouring line's descenders inside the crop's top rows stay out.
    #[test]
    fn a_neighbouring_line_is_not_in_the_box() {
        let (w, h) = (60usize, 40usize);
        let mut gray = vec![255f32; w * h];
        for y in 0..3 {
            for x in 5..55 {
                gray[y * w + x] = 0.0; // the line above
            }
        }
        for y in 15..28 {
            for x in 10..50 {
                gray[y * w + x] = 0.0;
            }
        }
        let words = boxes(&[e("word", 1, 8)], 10, &gray, w, (0, 0, w, h), |_| {});
        let b = words[0].bbox.unwrap();
        assert_eq!((b.x, b.y, b.width, b.height), (10.0, 15.0, 40.0, 13.0));
    }
}
