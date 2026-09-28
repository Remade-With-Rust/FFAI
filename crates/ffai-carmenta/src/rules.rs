//! Remove horizontal form rules before reading a page
//! (docs/plans/commercial-gaps.md, gap 12).
//!
//! A typed value sitting on a form's field line, or a line a filer struck
//! through, fuses the rule into the glyphs: `1` gains a foot and reads as `T`
//! or `I`, `0` reads as `U`, and the recognizer is confident about it. The
//! rule carries no text, so erasing it before detection and recognition gives
//! the models the glyphs alone.
//!
//! What counts as a rule pixel, and why each condition is there:
//!
//! * **dark**, by an Otsu threshold over the page;
//! * **in a long horizontal dark run** (at least [`RuleParams::min_len`]) —
//!   text breaks between words and inside most glyphs, a rule does not;
//! * **vertically thin** — the dark run through it, top to bottom, is at most
//!   [`RuleParams::max_thick`] pixels. A glyph stroke that crosses the rule, or
//!   stands on it, makes its column's vertical run tall, so those pixels are
//!   KEPT and the stroke survives intact. A filled box (a redaction patch) is
//!   tall everywhere and is never touched, which matters: erasing a redaction
//!   would be the opposite of what a privacy-honouring caller wants.
//!
//! Erased pixels take the page's background value (the mean of the light
//! class). Vertical rules are untouched.

use ffai_core::types::{ImageBuffer, PixelFormat};

/// Thresholds for [`remove_horizontal_rules`].
#[derive(Debug, Clone, Copy)]
pub struct RuleParams {
    /// Minimum horizontal dark run, in pixels, to be considered a rule.
    pub min_len: usize,
    /// Maximum vertical thickness, in pixels, of a pixel that is erased.
    pub max_thick: usize,
}

impl RuleParams {
    /// Defaults scaled to the page: a rule is at least 1/16 of the width
    /// (and never under 60 px), and at most 8 px thick. A 300 ppi scan draws a
    /// field line 2-4 px thick; glyph stems are an x-height tall.
    #[must_use]
    pub fn for_width(width: usize) -> Self {
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
        Self {
            min_len: env("FFAI_RULE_MIN_LEN").unwrap_or((width / 16).max(60)),
            max_thick: env("FFAI_RULE_MAX_THICK").unwrap_or(8),
        }
    }
}

/// What [`remove_horizontal_rules`] did, for instruments and tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuleStats {
    /// Pixels erased.
    pub erased: usize,
    /// Distinct rows that had at least one pixel erased.
    pub rows: usize,
}

pub(crate) fn luma(img: &ImageBuffer) -> Vec<u8> {
    match img.format {
        PixelFormat::Gray8 => img.data.clone(),
        PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
            let bpp = img.format.bytes_per_pixel();
            img.data
                .chunks_exact(bpp)
                .map(|p| {
                    ((77 * u32::from(p[0]) + 150 * u32::from(p[1]) + 29 * u32::from(p[2])) >> 8)
                        as u8
                })
                .collect()
        }
    }
}

/// Otsu threshold and the mean of the light class.
pub(crate) fn otsu(lum: &[u8]) -> (u8, u8) {
    let mut hist = [0u64; 256];
    for &v in lum {
        hist[v as usize] += 1;
    }
    let total = lum.len() as f64;
    let sum_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let (mut w_b, mut sum_b, mut best, mut t) = (0f64, 0f64, -1f64, 128u8);
    for (i, &c) in hist.iter().enumerate() {
        w_b += c as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += i as f64 * c as f64;
        let (m_b, m_f) = (sum_b / w_b, (sum_all - sum_b) / w_f);
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best {
            best = between;
            t = i as u8;
        }
    }
    let light: (f64, f64) = hist
        .iter()
        .enumerate()
        .skip(t as usize + 1)
        .fold((0.0, 0.0), |(s, n), (i, &c)| {
            (s + i as f64 * c as f64, n + c as f64)
        });
    let bg = if light.1 > 0.0 {
        (light.0 / light.1).round() as u8
    } else {
        255
    };
    (t, bg)
}

/// Erase horizontal rules, returning the cleaned image and what was erased.
/// The input is unchanged; the output has the same size and format.
#[must_use]
pub fn remove_horizontal_rules(img: &ImageBuffer, p: RuleParams) -> (ImageBuffer, RuleStats) {
    let (w, h) = (img.width as usize, img.height as usize);
    let mut out = img.clone();
    if w == 0 || h == 0 {
        return (out, RuleStats::default());
    }
    let lum = luma(img);
    let (t, bg) = otsu(&lum);
    // A page with no contrast has nothing to separate.
    if bg <= t {
        return (out, RuleStats::default());
    }
    let dark: Vec<bool> = lum.iter().map(|&v| v <= t).collect();

    // Vertical dark-run length through every dark pixel, column by column.
    let mut vrun = vec![0u16; w * h];
    for x in 0..w {
        let mut y = 0;
        while y < h {
            if !dark[y * w + x] {
                y += 1;
                continue;
            }
            let start = y;
            while y < h && dark[y * w + x] {
                y += 1;
            }
            let len = (y - start).min(u16::MAX as usize) as u16;
            for yy in start..y {
                vrun[yy * w + x] = len;
            }
        }
    }

    let bpp = img.format.bytes_per_pixel();
    let mut stats = RuleStats::default();
    for y in 0..h {
        let row = &dark[y * w..(y + 1) * w];
        let mut touched = false;
        let mut x = 0;
        while x < w {
            if !row[x] {
                x += 1;
                continue;
            }
            let start = x;
            while x < w && row[x] {
                x += 1;
            }
            if x - start < p.min_len {
                continue;
            }
            for xx in start..x {
                if (vrun[y * w + xx] as usize) <= p.max_thick {
                    let o = (y * w + xx) * bpp;
                    out.data[o..o + bpp.min(3)].fill(bg);
                    stats.erased += 1;
                    touched = true;
                }
            }
        }
        stats.rows += usize::from(touched);
    }
    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(w: usize, h: usize) -> Vec<u8> {
        vec![255; w * h]
    }

    fn fill(buf: &mut [u8], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) {
        for y in y0..y1 {
            buf[y * w + x0..y * w + x1].fill(0);
        }
    }

    fn gray(data: Vec<u8>, w: usize, h: usize) -> ImageBuffer {
        ImageBuffer {
            width: w as u32,
            height: h as u32,
            format: PixelFormat::Gray8,
            data,
        }
    }

    const P: RuleParams = RuleParams {
        min_len: 60,
        max_thick: 8,
    };

    #[test]
    fn a_rule_is_erased_and_a_stem_standing_on_it_survives() {
        let (w, h) = (300, 60);
        let mut d = page(w, h);
        fill(&mut d, w, 20, 40, 280, 43); // the rule: 3 px thick, 260 long
        fill(&mut d, w, 100, 10, 104, 40); // a stem standing on it
        let (out, stats) = remove_horizontal_rules(&gray(d, w, h), P);
        let at = |x: usize, y: usize| out.data[y * w + x];
        assert_eq!(at(50, 41), 255, "rule pixel away from the stem is erased");
        assert_eq!(at(102, 20), 0, "the stem is untouched");
        assert_eq!(at(102, 41), 0, "the stem's column through the rule is kept");
        assert_eq!(stats.rows, 3);
    }

    #[test]
    fn a_filled_box_such_as_a_redaction_is_never_touched() {
        let (w, h) = (300, 80);
        let mut d = page(w, h);
        fill(&mut d, w, 20, 10, 280, 60); // 50 px tall black patch
        let img = gray(d, w, h);
        let (out, stats) = remove_horizontal_rules(&img, P);
        assert_eq!(stats.erased, 0);
        assert_eq!(out.data, img.data);
    }

    #[test]
    fn short_dark_runs_such_as_glyph_bars_are_not_rules() {
        let (w, h) = (300, 40);
        let mut d = page(w, h);
        fill(&mut d, w, 20, 10, 60, 13); // a 40 px bar: a 'T' top, not a rule
        let (_, stats) = remove_horizontal_rules(&gray(d, w, h), P);
        assert_eq!(stats.erased, 0);
    }

    #[test]
    fn rgb_input_keeps_its_format_and_size() {
        let (w, h) = (200, 30);
        let mut d = vec![255u8; w * h * 3];
        for x in 10..190 {
            for c in 0..3 {
                d[(20 * w + x) * 3 + c] = 0;
            }
        }
        let img = ImageBuffer {
            width: w as u32,
            height: h as u32,
            format: PixelFormat::Rgb8,
            data: d,
        };
        let (out, stats) = remove_horizontal_rules(&img, P);
        assert_eq!(
            (out.width, out.height, out.format),
            (img.width, img.height, img.format)
        );
        assert_eq!(stats.erased, 180);
        assert!(out.data.iter().all(|&v| v == 255));
    }
}
