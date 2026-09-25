//! Checkboxes on forms: where they are and whether they are marked
//! (docs/plans/commercial-gaps.md, gaps 4 and 14).
//!
//! Read as text, a checkbox comes back as a character — `@`, `0`, `D`, `B` —
//! at the confidence of text, so a caller cannot tell a ticked box from a
//! letter, and a form whose one value field is a set of ticks is unreadable.
//! This finds the boxes directly, from the pixels, with no model:
//!
//! 1. **A frame is four sides.** A candidate starts at a horizontal dark run
//!    of checkbox size (the top edge). It is kept when a bottom edge sits
//!    roughly one side-length below it and the left and right edges are each
//!    at least [`SIDE_COVERAGE`] covered. Testing the frame's SIDES, rather
//!    than a connected blob's outline, is what keeps a tick that overflows
//!    the box from hiding the box.
//! 2. **Marked is ink inside.** The interior, inset past the frame's own
//!    thickness, is marked when its dark fraction reaches [`MARK_FILL`] — an
//!    X, a tick and a solid fill all qualify; an empty box does not.
//!
//! Size is bounded (`min_side..=max_side`), so a table's cells and a page's
//! border are not checkboxes. Square character grids (one-letter-per-box
//! fields) will be found as boxes; a caller can drop them by position.

use ffai_core::types::{BoundingBox, ImageBuffer};

/// Fraction of a frame side that must be dark for the side to count.
pub const SIDE_COVERAGE: f32 = 0.85;
/// Dark fraction of the interior at which a box counts as marked.
pub const MARK_FILL: f32 = 0.06;
/// Dark fraction of each corner square for the corner to count as sharp.
pub const CORNER_FILL: f32 = 0.5;
/// Largest dark fraction allowed along the line just OUTSIDE each side.
pub const OUTSIDE_INK: f32 = 0.25;

/// One checkbox.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Checkbox {
    /// The frame, in image pixels.
    pub bbox: BoundingBox,
    /// Whether the interior carries a mark.
    pub checked: bool,
    /// Dark fraction of the interior, the evidence for `checked`.
    pub fill: f32,
}

/// Size bounds for [`find`], in pixels.
#[derive(Debug, Clone, Copy)]
pub struct CheckboxParams {
    pub min_side: usize,
    pub max_side: usize,
}

impl Default for CheckboxParams {
    /// 10-150 px: a 3-12 mm box at 300 ppi, down to a small box at 150 ppi.
    fn default() -> Self {
        Self {
            min_side: 10,
            max_side: 150,
        }
    }
}

/// Every checkbox on the page, top to bottom, left to right.
#[must_use]
pub fn find(img: &ImageBuffer, p: CheckboxParams) -> Vec<Checkbox> {
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let lum = crate::rules::luma(img);
    let (t, bg) = crate::rules::otsu(&lum);
    if bg <= t {
        return Vec::new();
    }
    let dark: Vec<bool> = lum.iter().map(|&v| v <= t).collect();
    let at = |x: usize, y: usize| dark[y * w + x];

    // Fraction of the row segment [x0, x1) at `y` that is dark.
    let row_cov = |y: usize, x0: usize, x1: usize| {
        (x0..x1).filter(|&x| at(x, y)).count() as f32 / (x1 - x0).max(1) as f32
    };
    // A vertical side at columns [c0, c1) over rows [y0, y1): the fraction of
    // rows with any dark pixel in the band (a side has thickness).
    let col_cov = |c0: usize, c1: usize, y0: usize, y1: usize| {
        (y0..y1).filter(|&y| (c0..c1).any(|x| at(x, y))).count() as f32 / (y1 - y0).max(1) as f32
    };

    let mut found: Vec<Checkbox> = Vec::new();
    for y in 0..h {
        let mut x = 0;
        while x < w {
            if !at(x, y) {
                x += 1;
                continue;
            }
            let x0 = x;
            while x < w && at(x, y) {
                x += 1;
            }
            let x1 = x;
            let side = x1 - x0;
            if side < p.min_side || side > p.max_side {
                continue;
            }
            // Already inside a found box: a thick top edge produces a run on
            // each of its rows.
            if found.iter().any(|b| contains(&b.bbox, x0, y)) {
                continue;
            }
            if let Some(cb) = frame_at(x0, x1, y, w, h, &row_cov, &col_cov, &at) {
                found.push(cb);
            }
        }
    }
    found
}

/// Is there a square frame whose top edge is the run `[x0, x1)` at row `y`?
#[allow(clippy::too_many_arguments)]
fn frame_at(
    x0: usize,
    x1: usize,
    y: usize,
    w: usize,
    h: usize,
    row_cov: &dyn Fn(usize, usize, usize) -> f32,
    col_cov: &dyn Fn(usize, usize, usize, usize) -> f32,
    at: &dyn Fn(usize, usize) -> bool,
) -> Option<Checkbox> {
    let side = x1 - x0;
    // Bottom edge: a well-covered row between 0.8 and 1.25 side-lengths down.
    let lo = y + side * 4 / 5;
    let hi = (y + side * 5 / 4).min(h - 1);
    if lo >= h {
        return None;
    }
    let bottom = (lo..=hi).find(|&yy| row_cov(yy, x0, x1) >= SIDE_COVERAGE)?;
    // The bottom edge may be several rows thick: take its last row.
    let mut y1 = bottom;
    while y1 + 1 < h && y1 + 1 <= hi && row_cov(y1 + 1, x0, x1) >= SIDE_COVERAGE {
        y1 += 1;
    }
    let height = y1 + 1 - y;
    // A mark that overflows the box (a tick flicked past its top-right
    // corner) joins the top edge and stretches the run past the frame. The
    // frame's right side is the rightmost column that is a full vertical
    // line over the frame's height: trim back to it.
    let full = |c: usize| col_cov(c, c + 1, y, y1 + 1) >= SIDE_COVERAGE;
    let x1 = (x0 + side * 2 / 3..x1)
        .rev()
        .find(|&c| full(c))
        .map_or(x1, |c| c + 1);
    let side = x1 - x0;
    // Side thickness allowance: a 2-4 px pen at 300 ppi, proportionally less
    // on a small box. Taken from the TRIMMED side.
    let band = (side / 8).clamp(2, 6);
    // Near-square only: a wide rectangle is a field, not a box.
    if (height as f32) < side as f32 * 0.8 || (height as f32) > side as f32 * 1.25 {
        return None;
    }
    let left = col_cov(x0, (x0 + band).min(w), y, y1 + 1);
    let right = col_cov(x1.saturating_sub(band), x1, y, y1 + 1);
    if left < SIDE_COVERAGE || right < SIDE_COVERAGE {
        return None;
    }
    // A printed box has SHARP corners; `o`, `e`, `a`, `0`, `8` have rounded
    // ones, and `D` and `B` round on the right. Each corner's band x band
    // square must be mostly ink — an L of two frame sides meeting.
    let corner = |cx: usize, cy: usize| {
        let (cx0, cy0) = (
            cx.min(w.saturating_sub(band)),
            cy.min(h.saturating_sub(band)),
        );
        let dark = (cy0..cy0 + band)
            .flat_map(|yy| (cx0..cx0 + band).map(move |xx| (xx, yy)))
            .filter(|&(xx, yy)| at(xx, yy))
            .count();
        dark as f32 / (band * band) as f32 >= CORNER_FILL
    };
    let (xr, yb) = (x1.saturating_sub(band), (y1 + 1).saturating_sub(band));
    if !(corner(x0, y) && corner(xr, y) && corner(x0, yb) && corner(xr, yb)) {
        return None;
    }
    // ...but a run taken from a round letter's own contour ALSO has dark ends,
    // so the corner test alone passes `o`, `d`, `g`, `R`. What a letter cannot
    // fake is CLEAR SPACE just outside the frame: a box's top edge is its
    // topmost ink and its sides its outermost, while a curve bulges past any
    // run cut from it. Measured over the middle of each side, so a tick
    // overflowing one corner does not disqualify its box.
    let clear = |dark: usize, n: usize| (dark as f32) <= OUTSIDE_INK * n.max(1) as f32;
    let (my0, my1) = (y + band, (y1 + 1).saturating_sub(band));
    let (mx0, mx1) = (x0 + band, x1.saturating_sub(band));
    let outside_left = x0 == 0
        || clear(
            (my0..my1).filter(|&yy| at(x0 - 1, yy)).count(),
            my1.saturating_sub(my0),
        );
    let outside_right = x1 >= w
        || clear(
            (my0..my1).filter(|&yy| at(x1, yy)).count(),
            my1.saturating_sub(my0),
        );
    let outside_top = y == 0
        || clear(
            (mx0..mx1).filter(|&xx| at(xx, y - 1)).count(),
            mx1.saturating_sub(mx0),
        );
    let outside_bottom = y1 + 1 >= h
        || clear(
            (mx0..mx1).filter(|&xx| at(xx, y1 + 1)).count(),
            mx1.saturating_sub(mx0),
        );
    if !(outside_left && outside_right && outside_top && outside_bottom) {
        return None;
    }
    // Interior, inset past the frame. Measure the frame's real thickness from
    // the top edge so a heavy pen does not read as a mark.
    let mut thick = 1;
    while thick < band * 2 && y + thick < y1 && row_cov(y + thick, x0, x1) >= SIDE_COVERAGE {
        thick += 1;
    }
    let inset = thick + 2;
    let (ix0, ix1, iy0, iy1) = (
        x0 + inset,
        x1.saturating_sub(inset),
        y + inset,
        (y1 + 1).saturating_sub(inset),
    );
    if ix1 <= ix0 || iy1 <= iy0 {
        return None;
    }
    let interior = (ix1 - ix0) * (iy1 - iy0);
    let ink = (iy0..iy1)
        .map(|yy| (ix0..ix1).filter(|&xx| at(xx, yy)).count())
        .sum::<usize>();
    let fill = ink as f32 / interior as f32;
    Some(Checkbox {
        bbox: BoundingBox {
            x: x0 as f32,
            y: y as f32,
            width: side as f32,
            height: height as f32,
        },
        checked: fill >= MARK_FILL,
        fill,
    })
}

fn contains(b: &BoundingBox, x: usize, y: usize) -> bool {
    let (x, y) = (x as f32, y as f32);
    x >= b.x - 2.0 && x <= b.x + b.width + 2.0 && y >= b.y - 2.0 && y <= b.y + b.height + 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffai_core::types::PixelFormat;

    struct Page {
        w: usize,
        d: Vec<u8>,
    }
    impl Page {
        fn new(w: usize, h: usize) -> Self {
            Self {
                w,
                d: vec![255; w * h],
            }
        }
        fn fill(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
            for y in y0..y1 {
                self.d[y * self.w + x0..y * self.w + x1].fill(0);
            }
        }
        fn frame(&mut self, x: usize, y: usize, s: usize, t: usize) {
            self.fill(x, y, x + s, y + t);
            self.fill(x, y + s - t, x + s, y + s);
            self.fill(x, y, x + t, y + s);
            self.fill(x + s - t, y, x + s, y + s);
        }
        fn img(self) -> ImageBuffer {
            let h = self.d.len() / self.w;
            ImageBuffer {
                width: self.w as u32,
                height: h as u32,
                format: PixelFormat::Gray8,
                data: self.d,
            }
        }
    }

    #[test]
    fn an_empty_and_a_marked_box_are_found_and_told_apart() {
        let mut p = Page::new(300, 100);
        p.frame(20, 20, 40, 3); // empty
        p.frame(120, 20, 40, 3);
        for i in 0..28 {
            // an X inside the second
            p.fill(126 + i, 26 + i, 128 + i, 28 + i);
            p.fill(152 - i, 26 + i, 154 - i, 28 + i);
        }
        let found = find(&p.img(), CheckboxParams::default());
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(!found[0].checked, "{:?}", found[0]);
        assert!(found[1].checked, "{:?}", found[1]);
    }

    #[test]
    fn a_tick_overflowing_the_box_does_not_hide_it() {
        let mut p = Page::new(200, 120);
        p.frame(40, 40, 40, 3);
        // a stroke from inside the box up and out past its top-right corner
        for i in 0..50 {
            p.fill(50 + i, 70 - i, 53 + i, 73 - i);
        }
        let found = find(&p.img(), CheckboxParams::default());
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].checked);
    }

    #[test]
    fn letters_rules_and_wide_fields_are_not_boxes() {
        let mut p = Page::new(400, 120);
        p.fill(10, 100, 390, 103); // an underline rule
        p.fill(20, 20, 60, 24); // a 'T' top bar...
        p.fill(38, 20, 42, 60); // ...and its stem
        p.frame(100, 20, 40, 3); // a box...
        p.fill(100, 20, 140, 60); // ...filled solid: a redaction square, still a box
        p.fill(200, 20, 390, 23); // a wide field frame 190x40: not square
        p.fill(200, 57, 390, 60);
        p.fill(200, 20, 203, 60);
        p.fill(387, 20, 390, 60);
        let found = find(&p.img(), CheckboxParams::default());
        assert_eq!(found.len(), 1, "only the square: {found:?}");
        assert!(found[0].checked, "a solid square reads as marked");
    }
}
