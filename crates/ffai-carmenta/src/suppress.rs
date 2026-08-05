//! Body-text filter: drop lines the document-text task does not ask for (§8.106).
//!
//! §8.105 measured the whole competitive gap as SCOPE rather than accuracy. Of
//! the 26.19 pp of non-editorial error, substitutions — the recognizer actually
//! misreading characters — are **2.36 pp**; insertions are 17.98 pp, and 14.6 %
//! of our output characters land in regions the benchmark does not annotate:
//! equations, code blocks, table cells, running headers, page numbers. An oracle
//! that suppresses them takes the holdout from **18.88 % to 11.92 %**, moving
//! Carmenta from 5.99 pp behind Unlimited-OCR to 0.97 pp ahead.
//!
//! ## Why this is a MODE, not a default
//!
//! A benchmark that scores against `text_block`/`title`/`figure_caption` defines
//! the task as "extract the body text". A tool that also emits the table cells
//! is answering a different, often more useful question — so suppression is
//! **opt-in**. `FFAI_BODY_ONLY=1` selects the benchmark's scope; the default
//! emits everything the page contains, which is what a document-to-text user
//! usually wants.
//!
//! This is not benchmark-fitting: Unlimited-OCR emits TYPED regions, so its
//! harness filters to the scored classes by construction. Running body-only is
//! the like-for-like comparison, not a thumb on the scale.
//!
//! ## What the rule is, and what it cost to learn it
//!
//! Thirteen runtime features per line were harvested (geometry + recognised text
//! + CTC confidence; nothing from the annotation but the label), fitted on the
//! TRAIN split and judged once on holdout:
//!
//! | model | holdout | share of the 6.96 pp prize |
//! |---|---:|---:|
//! | hand rule (`nchars <= 7`) | -0.57 pp | 8 % |
//! | logistic regression | -1.00 pp | 14 % |
//! | **decision tree, depth 3** | **-1.38 pp** | **20 %** |
//! | gradient boosting, 200 trees | -2.16 pp | 31 % |
//!
//! The depth-3 tree keeps 64 % of the boosted model's win in four comparisons
//! and needs no model artifact, so it is what ships. The boosted figure is
//! recorded as the price of the next step, not left as a rumour.
//!
//! **The other 69 % is not reachable from these features.** The orphans the
//! rules miss carry five times the characters of the ones they catch, and they
//! are geometrically IDENTICAL to body text — author affiliations and long
//! captions sitting at `same_left = 37`, `nn_gap = 0.88`, ordinary pitch. They
//! differ from a paragraph only in what they MEAN, which is the same wall
//! §8.81 and §8.87 hit from other directions.

use ffai_core::types::OcrLine;

/// Per-line features, all computable from what the engine already has.
struct Feat {
    /// Total characters in this line's PARENT RUN — the contiguous block of
    /// left-aligned lines it belongs to. The strongest single signal at
    /// **52 % importance**, more than double the next (§8.108): a line is
    /// non-body mostly because of the BLOCK it sits in, not its own shape.
    /// This is what carries the block context down to a per-line decision.
    run_chars: f32,
    /// Line width divided by the page's OWN 90th-percentile line width — i.e.
    /// how wide this line is against a FULL COLUMN LINE on this page.
    ///
    /// Not width-over-page-width, which is confounded by column count: the same
    /// physical body line reads ~0.85 on a single-column page, ~0.45 on two
    /// columns and ~0.30 on three (§8.109). Normalised this way body lines
    /// cluster at 0.97 — they ARE the p90 width — and non-body at 0.085, with
    /// ZERO overlap on train.
    ///
    /// In-corpus the two are equivalent for a tree (-1.59 vs -1.63 pp), because
    /// a tree carves the space per page-population and does not need one global
    /// cutoff. This form ships for TRANSFER: fitted on single-column pages, a
    /// page-width threshold makes multi-column pages WORSE (+0.26 pp), while
    /// this one still helps (-0.33 pp). Production meets layouts the fit never
    /// saw.
    w_p90: f32,
    /// Vertical centre as a fraction of page height; catches running headers.
    y_rel: f32,
    /// Distance to the nearest other line centre, in line heights. A displayed
    /// equation or page number sits in whitespace; a body line has a neighbour
    /// about one pitch away.
    nn_gap: f32,
    /// CTC confidence, as reported by the recognizer.
    conf: f32,
    /// Lines in the parent run, and how many other lines share this line's left
    /// edge. Both are low for the population the width branch alone kept:
    /// mid-width lines in tiny isolated blocks — code fragments, table rows,
    /// email addresses, figure text (§8.112).
    run_lines: usize,
    same_left: usize,
    /// This line carries the syntax of a reference-list entry: a parenthesised
    /// four-digit year, `(1999)` or `(2011a)`.
    year_paren: bool,
    /// How many lines on the PAGE carry it. Load-bearing, and the whole reason
    /// this branch is safe: a bibliography is a BLOCK, so the pages that own one
    /// show 8+ hits, while an in-text citation ("as shown by Smith (2019)")
    /// appears once or twice on a page of ordinary annotated prose. Without the
    /// count the rule fires on both and every observed miss is the second kind
    /// (§8.116).
    page_year_hits: usize,
}

/// The fitted depth-4 tree, transcribed. Thresholds come from the TRAIN split
/// and were judged once on holdout at -1.63 pp; they are not re-tunable here
/// without re-running that fit.
fn is_non_body(f: &Feat) -> bool {
    // The WIDTH branches. `w_p90 <= 0.198` means narrower than a fifth of a full
    // column line on this page.
    let by_width = if f.w_p90 <= 0.198 {
        if f.run_chars <= 59.5 {
            // Small block: a label or table cell, unless it sits tight against a
            // neighbour, which makes it part of real running text.
            f.nn_gap > 0.0129
        } else {
            // Large block: only the very narrow ones are non-body.
            f.w_p90 <= 0.0874
        }
    } else {
        // Full-width lines are body — except in the top margin, a running header.
        f.y_rel <= 0.0533
    };

    // THE ISOLATED-FRAGMENT branch (§8.112). The width rule above guards the
    // wide case only by top margin, and 89 % of what it missed landed there:
    // mid-width lines sitting in tiny, poorly-aligned blocks. Body text lives in
    // blocks of ~1600 characters aligned with ~47 neighbours; these sit in 3-line
    // blocks with fewer than 5.
    //
    // Fitted against CHARACTER-weighted net gain rather than accuracy — the
    // distinction that made three earlier rules lose — and it generalises:
    // -0.26 pp on train, -0.27 pp on holdout, which is the agreement a rule that
    // merely fits one split never shows.
    let isolated_fragment = f.run_lines <= 3 && f.w_p90 < 0.35 && f.same_left < 5;

    // THE BIBLIOGRAPHY branch (§8.116). §8.115 showed the residual is 85 %
    // reference lists and boilerplate, geometrically identical to body text —
    // they differ only in what the words ARE. This is the one textual signal
    // that survived: not a fitted threshold but a REGEX for citation syntax,
    // which has no capacity to overfit the way a numeric cutoff does.
    //
    // The count is what makes it shippable. Ungated it nets +4,407 characters
    // with 8 misses; gated at 4 it nets +6,165 with 98 % precision, and the
    // gate sits on a FLAT PLATEAU from 4 to 8 — a tuned constant has an edge,
    // this has none.
    //
    // Provisional, and the reason is recorded rather than buried: the win lives
    // on 4 pages of 316, so the corpus can show the downside is bounded
    // (-160 characters total) but cannot prove the upside generalises. It ships
    // on a bounded-risk argument, not a validated one.
    let bibliography = f.year_paren && f.page_year_hits >= 4;

    by_width || isolated_fragment || bibliography
}

/// `(` `19|20` `dd` optional lowercase suffix `)` — citation-year syntax.
///
/// Hand-rolled rather than a `regex` dependency: the pattern is fixed and this
/// runs once per line. Byte scanning is safe on UTF-8 because every byte
/// compared here is ASCII, and multi-byte sequences have the high bit set.
fn has_year_paren(s: &str) -> bool {
    let b = s.as_bytes();
    for i in 0..b.len() {
        if b[i] != b'(' {
            continue;
        }
        let mut j = i + 1;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if j + 3 >= b.len() {
            continue;
        }
        if !((b[j] == b'1' && b[j + 1] == b'9') || (b[j] == b'2' && b[j + 1] == b'0')) {
            continue;
        }
        if !(b[j + 2].is_ascii_digit() && b[j + 3].is_ascii_digit()) {
            continue;
        }
        let mut k = j + 4;
        if k < b.len() && b[k].is_ascii_lowercase() {
            k += 1;
        }
        while k < b.len() && b[k].is_ascii_whitespace() {
            k += 1;
        }
        if k < b.len() && b[k] == b')' {
            return true;
        }
    }
    false
}

/// Group lines into contiguous runs: cluster by LEFT EDGE first, then split on
/// vertical pitch.
///
/// Left edge first is load-bearing. Sorting by `y` interleaves a two-column
/// page as L,R,L,R and breaks every run at every line — which read as a
/// meaningless 100 % run purity until it was caught (§8.106), the same
/// units-mismatch that voided §8.39 and the oracle-crop plan.
fn run_stats_per_line(boxes: &[(f32, f32, f32, f32)], text_len: &[usize], pw: f32, ph: f32)
    -> (Vec<f32>, Vec<usize>)
{
    let mut cols: Vec<(f32, Vec<usize>)> = Vec::new();
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        (boxes[a].0 / pw).partial_cmp(&(boxes[b].0 / pw)).unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in order {
        let l = boxes[i].0 / pw;
        match cols.iter_mut().find(|(k, _)| (k - l).abs() < 0.02) {
            Some((_, v)) => v.push(i),
            None => cols.push((l, vec![i])),
        }
    }
    let mut out = vec![0f32; boxes.len()];
    let mut nlines = vec![1usize; boxes.len()];
    for (_, mut v) in cols {
        v.sort_by(|&a, &b| {
            boxes[a].1.partial_cmp(&boxes[b].1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut run: Vec<usize> = vec![v[0]];
        let flush = |run: &Vec<usize>, out: &mut Vec<f32>, nl: &mut Vec<usize>| {
            let total: usize = run.iter().map(|&i| text_len[i]).sum();
            for &i in run {
                out[i] = total as f32;
                nl[i] = run.len();
            }
        };
        for w in v.windows(2) {
            let (a, b) = (w[0], w[1]);
            if (boxes[b].1 - boxes[a].1) / ph < 0.035 {
                run.push(b);
            } else {
                flush(&run, &mut out, &mut nlines);
                run = vec![b];
            }
        }
        flush(&run, &mut out, &mut nlines);
    }
    (out, nlines)
}

/// Drop non-body lines when `FFAI_BODY_ONLY` is set; otherwise return `lines`
/// untouched.
///
/// Takes the page size because every feature is page-relative — an absolute
/// threshold would be meaningless across the 1600..5723 px pages in the corpus.
pub fn body_only(lines: Vec<OcrLine>, page_w: f32, page_h: f32) -> Vec<OcrLine> {
    if std::env::var("FFAI_BODY_ONLY").as_deref() != Ok("1") || lines.len() < 3 {
        return lines;
    }
    let (pw, ph) = (page_w.max(1.0), page_h.max(1.0));
    let boxes: Vec<(f32, f32, f32, f32)> = lines
        .iter()
        .map(|l| l.bbox.as_ref().map(|b| (b.x, b.y, b.width, b.height)).unwrap_or((0.0, 0.0, pw, 1.0)))
        .collect();
    let text_len: Vec<usize> = lines.iter().map(|l| l.text.trim().chars().count()).collect();
    let (run_chars, run_lines) = run_stats_per_line(&boxes, &text_len, pw, ph);
    let lefts: Vec<f32> = boxes.iter().map(|b| b.0).collect();

    // The page's own 90th-percentile line width: what a FULL COLUMN LINE looks
    // like here, whatever the column count.
    let mut ws: Vec<f32> = boxes.iter().map(|b| b.2 / pw).collect();
    ws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // `((n-1)*9)/10`, matching the index the thresholds were fitted against.
    // The first form here was `(n*9)/10` with a divisibility fudge, which is
    // off by one at n = 73, 137, 255 — ordinary page line counts — so the
    // shipped filter measured widths against a different denominator than the
    // fit did. It cost 0.35 pp and was caught only because the offline
    // simulation and the engine disagreed (§8.113).
    let p90 = ws[((ws.len() - 1) * 9) / 10].max(1e-6);

    let mut hs: Vec<f32> = boxes.iter().map(|b| b.3).collect();
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lh = hs.get(hs.len() / 2).copied().unwrap_or(1.0).max(1.0);
    let cys: Vec<f32> = boxes.iter().map(|b| b.1 + b.3 / 2.0).collect();

    // Counted over EVERY line on the page, not just the ones the other branches
    // keep. The two populations were measured and give identical results here,
    // but the whole-page count is the one that cannot drift as the branches
    // above change — and counting over a subset is exactly the units mismatch
    // that voided §8.39 and §8.106.
    let years: Vec<bool> = lines.iter().map(|l| has_year_paren(&l.text)).collect();
    let page_year_hits = years.iter().filter(|&&y| y).count();

    let keep: Vec<bool> = (0..lines.len())
        .map(|i| {
            let (x, y, w, h) = boxes[i];
            let cy = y + h / 2.0;
            let nn = cys
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, c)| (c - cy).abs())
                .fold(f32::INFINITY, f32::min);
            !is_non_body(&Feat {
                run_chars: run_chars[i],
                run_lines: run_lines[i],
                same_left: lefts.iter().filter(|&&lx| (lx - x).abs() <= lh / 2.0).count()
                    .saturating_sub(1),
                w_p90: (w / pw) / p90,
                y_rel: cy / ph,
                nn_gap: if nn.is_finite() { nn / lh } else { 20.0 },
                conf: lines[i].confidence.unwrap_or(1.0),
                year_paren: years[i],
                page_year_hits,
            })
        })
        .collect();
    lines.into_iter().zip(keep).filter_map(|(l, k)| k.then_some(l)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each branch of the depth-4 tree, at a value the fit chose.
    #[test]
    fn tree_branches() {
        // small block, narrow, isolated -> label or table cell
        assert!(is_non_body(&Feat { run_chars: 20.0, w_p90: 0.10, y_rel: 0.5, nn_gap: 0.5, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, page_year_hits: 0 }));
        // small block, narrow, but tight against a neighbour -> body
        assert!(!is_non_body(&Feat { run_chars: 20.0, w_p90: 0.10, y_rel: 0.5, nn_gap: 0.005, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, page_year_hits: 0 }));
        // small block, wide, top margin -> running header
        assert!(is_non_body(&Feat { run_chars: 20.0, w_p90: 0.9, y_rel: 0.02, nn_gap: 0.5, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, page_year_hits: 0 }));
        // small block, wide, mid-page, confident -> body
        assert!(!is_non_body(&Feat { run_chars: 20.0, w_p90: 0.9, y_rel: 0.5, nn_gap: 0.5, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, page_year_hits: 0 }));
        // large block, low confidence, mid-sized -> non-body
        assert!(is_non_body(&Feat { run_chars: 100.0, w_p90: 0.05, y_rel: 0.5, nn_gap: 0.5, conf: 0.5, run_lines: 40, same_left: 50, year_paren: false, page_year_hits: 0 }));
        // genuinely long prose, even at low confidence -> body
        assert!(!is_non_body(&Feat { run_chars: 900.0, w_p90: 0.9, y_rel: 0.5, nn_gap: 0.5, conf: 0.5, run_lines: 40, same_left: 50, year_paren: false, page_year_hits: 0 }));
        // large, confident, top margin -> running header
        assert!(is_non_body(&Feat { run_chars: 900.0, w_p90: 0.9, y_rel: 0.02, nn_gap: 0.5, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, page_year_hits: 0 }));
    }

    /// The isolated-fragment branch (§8.112): a mid-width line the width rules
    /// keep, but which sits in a tiny, poorly-aligned block — a code fragment,
    /// a table row, an email address, figure text.
    #[test]
    fn isolated_fragment_branch() {
        let frag = |run_lines, same_left, w_p90| Feat {
            run_chars: 300.0,
            w_p90,
            y_rel: 0.5,
            nn_gap: 0.5,
            conf: 0.99,
            run_lines,
            same_left,
            year_paren: false,
            page_year_hits: 0,
        };
        // three-line block, mid width, almost nothing shares its left edge
        assert!(is_non_body(&frag(3, 2, 0.30)));
        // the same shape, but aligned with a column of body text
        assert!(!is_non_body(&frag(3, 40, 0.30)));
        // the same shape, but part of a long block
        assert!(!is_non_body(&frag(20, 2, 0.30)));
        // isolated and short-blocked, but a full column line
        assert!(!is_non_body(&frag(3, 2, 0.90)));
    }

    /// The citation-year scanner, including the shapes that must NOT match.
    #[test]
    fn year_paren_syntax() {
        for s in [
            "Meinhart CD, Wereley ST, Santiago JG (1999) PIV measurements",
            "Block; C R (1984). Is crime seasonal? Chicago",
            "Smith J (2011a) A study of things",
            "trailing spaces inside ( 2003 ) still count",
        ] {
            assert!(has_year_paren(s), "should match: {s}");
        }
        for s in [
            "no parens here 1999 at all",
            "(199) too few digits",
            "(19999) too many",          // the ')' is not where a year ends
            "(1850) is not 19xx or 20xx",
            "a page range (277-282) is not a year",
            "unterminated (1999 and then nothing",
            "(2011ab) two suffix letters",
            "",
        ] {
            assert!(!has_year_paren(s), "should NOT match: {s}");
        }
        // Multi-byte input must not panic or false-match on byte boundaries.
        assert!(!has_year_paren("café — naïve (résumé)"));
        assert!(has_year_paren("Müller H (2007) Über etwas"));
    }

    /// The bibliography branch (§8.116). It exists BECAUSE of the second case:
    /// every observed miss of the ungated rule was an in-text citation on a page
    /// of ordinary annotated prose, and the page-wide count separates those from
    /// a reference list, which is a dense block.
    #[test]
    fn bibliography_branch() {
        // Body-shaped in every geometric respect — this is the point: the
        // branch has to fire on something the other two provably keep.
        let bib = |year_paren, page_year_hits| Feat {
            run_chars: 1600.0,
            w_p90: 0.98,
            y_rel: 0.5,
            nn_gap: 0.9,
            conf: 0.99,
            run_lines: 40,
            same_left: 34,
            year_paren,
            page_year_hits,
        };
        // A reference entry on a page dense with them.
        assert!(is_non_body(&bib(true, 12)));
        // The SAME line on a page with only a couple of in-text citations.
        assert!(!is_non_body(&bib(true, 2)));
        // Exactly at the gate, and one below it.
        assert!(is_non_body(&bib(true, 4)));
        assert!(!is_non_body(&bib(true, 3)));
        // Body text on a bibliography page is untouched — the line must carry
        // the syntax itself, the page count alone is never enough.
        assert!(!is_non_body(&bib(false, 40)));
    }

    /// Off by default: a filter that silently deletes output must be asked for.
    #[test]
    fn off_unless_requested() {
        let mk = |w: f32| OcrLine {
            text: "x".into(),
            words: Vec::new(),
            bbox: Some(ffai_core::types::BoundingBox { x: 0.0, y: 0.0, width: w, height: 10.0 }),
            confidence: Some(0.99),
        };
        let lines = vec![mk(5.0), mk(5.0), mk(900.0)];
        assert_eq!(body_only(lines, 1000.0, 1000.0).len(), 3);
    }
}
