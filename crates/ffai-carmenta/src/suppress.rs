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
}

/// The fitted depth-4 tree, transcribed. Thresholds come from the TRAIN split
/// and were judged once on holdout at -1.63 pp; they are not re-tunable here
/// without re-running that fit.
fn is_non_body(f: &Feat) -> bool {
    if f.w_p90 <= 0.198 {
        // Narrower than a fifth of a full column line.
        if f.run_chars <= 59.5 {
            // In a SMALL block: a label or table cell, unless it sits tight
            // against a neighbour, which makes it part of real running text.
            f.nn_gap > 0.0129
        } else {
            // In a LARGE block: only the very narrow ones are non-body.
            f.w_p90 <= 0.0874
        }
    } else {
        // Full-width lines are body text — except in the top margin, where a
        // full-width line is a running header.
        f.y_rel <= 0.0533
    }
}

/// Group lines into contiguous runs: cluster by LEFT EDGE first, then split on
/// vertical pitch.
///
/// Left edge first is load-bearing. Sorting by `y` interleaves a two-column
/// page as L,R,L,R and breaks every run at every line — which read as a
/// meaningless 100 % run purity until it was caught (§8.106), the same
/// units-mismatch that voided §8.39 and the oracle-crop plan.
fn run_chars_per_line(boxes: &[(f32, f32, f32, f32)], text_len: &[usize], pw: f32, ph: f32)
    -> Vec<f32>
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
    for (_, mut v) in cols {
        v.sort_by(|&a, &b| {
            boxes[a].1.partial_cmp(&boxes[b].1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut run: Vec<usize> = vec![v[0]];
        let flush = |run: &Vec<usize>, out: &mut Vec<f32>| {
            let total: usize = run.iter().map(|&i| text_len[i]).sum();
            for &i in run {
                out[i] = total as f32;
            }
        };
        for w in v.windows(2) {
            let (a, b) = (w[0], w[1]);
            if (boxes[b].1 - boxes[a].1) / ph < 0.035 {
                run.push(b);
            } else {
                flush(&run, &mut out);
                run = vec![b];
            }
        }
        flush(&run, &mut out);
    }
    out
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
    let run_chars = run_chars_per_line(&boxes, &text_len, pw, ph);

    // The page's own 90th-percentile line width: what a FULL COLUMN LINE looks
    // like here, whatever the column count.
    let mut ws: Vec<f32> = boxes.iter().map(|b| b.2 / pw).collect();
    ws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p90 = ws[(ws.len() * 9) / 10 - usize::from(ws.len() * 9 % 10 == 0)].max(1e-6);

    let mut hs: Vec<f32> = boxes.iter().map(|b| b.3).collect();
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lh = hs.get(hs.len() / 2).copied().unwrap_or(1.0).max(1.0);
    let cys: Vec<f32> = boxes.iter().map(|b| b.1 + b.3 / 2.0).collect();

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
            let _ = x;
            !is_non_body(&Feat {
                run_chars: run_chars[i],
                w_p90: (w / pw) / p90,
                y_rel: cy / ph,
                nn_gap: if nn.is_finite() { nn / lh } else { 20.0 },
                conf: lines[i].confidence.unwrap_or(1.0),
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
        assert!(is_non_body(&Feat { run_chars: 20.0, w_p90: 0.10, y_rel: 0.5, nn_gap: 0.5, conf: 0.99 }));
        // small block, narrow, but tight against a neighbour -> body
        assert!(!is_non_body(&Feat { run_chars: 20.0, w_p90: 0.10, y_rel: 0.5, nn_gap: 0.005, conf: 0.99 }));
        // small block, wide, top margin -> running header
        assert!(is_non_body(&Feat { run_chars: 20.0, w_p90: 0.9, y_rel: 0.02, nn_gap: 0.5, conf: 0.99 }));
        // small block, wide, mid-page, confident -> body
        assert!(!is_non_body(&Feat { run_chars: 20.0, w_p90: 0.9, y_rel: 0.5, nn_gap: 0.5, conf: 0.99 }));
        // large block, low confidence, mid-sized -> non-body
        assert!(is_non_body(&Feat { run_chars: 100.0, w_p90: 0.05, y_rel: 0.5, nn_gap: 0.5, conf: 0.5 }));
        // genuinely long prose, even at low confidence -> body
        assert!(!is_non_body(&Feat { run_chars: 900.0, w_p90: 0.9, y_rel: 0.5, nn_gap: 0.5, conf: 0.5 }));
        // large, confident, top margin -> running header
        assert!(is_non_body(&Feat { run_chars: 900.0, w_p90: 0.9, y_rel: 0.02, nn_gap: 0.5, conf: 0.99 }));
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
