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
    /// Line width as a fraction of the page — the single strongest signal
    /// (37 % of the boosted model's importance). Table cells, page numbers and
    /// figure labels are narrow; body lines span their column.
    w_rel: f32,
    /// Vertical centre as a fraction of page height; catches running headers.
    y_rel: f32,
    /// How many OTHER lines start at the same left edge (within half a line
    /// height). Body text in a column shares one with dozens of neighbours; an
    /// equation, a table cell or a caption does not.
    same_left: usize,
    /// CTC confidence, as reported by the recognizer.
    conf: f32,
}

/// The fitted depth-3 tree, transcribed. Thresholds come from the TRAIN split
/// and were judged once on holdout at -1.38 pp; they are not re-tunable here
/// without re-running that fit.
fn is_non_body(f: &Feat) -> bool {
    if f.w_rel <= 0.079 {
        // Narrow: non-body unless it is one of many lines sharing a left edge
        // AND wide enough to be a real (short) body line.
        f.same_left <= 20 || f.w_rel <= 0.026
    } else {
        // Wide: body, unless it sits in the top margin (running header) or the
        // recognizer had little confidence in it.
        f.y_rel <= 0.053 || f.conf <= 0.924
    }
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
    // Left edges are needed by every line's `same_left`, so gather once.
    let lefts: Vec<f32> = lines.iter().filter_map(|l| l.bbox.as_ref().map(|b| b.x)).collect();
    let mut hs: Vec<f32> = lines.iter().filter_map(|l| l.bbox.as_ref().map(|b| b.height)).collect();
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lh = hs.get(hs.len() / 2).copied().unwrap_or(1.0).max(1.0);

    lines
        .into_iter()
        .filter(|l| {
            let Some(b) = l.bbox.as_ref() else { return true };
            let same_left =
                lefts.iter().filter(|&&x| (x - b.x).abs() <= lh / 2.0).count().saturating_sub(1);
            !is_non_body(&Feat {
                w_rel: b.width / pw,
                y_rel: (b.y + b.height / 2.0) / ph,
                same_left,
                conf: l.confidence.unwrap_or(1.0),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tree's four branches, each exercised at a value the fit chose.
    #[test]
    fn tree_branches() {
        // narrow + few neighbours sharing a left edge -> non-body
        assert!(is_non_body(&Feat { w_rel: 0.05, y_rel: 0.5, same_left: 3, conf: 0.99 }));
        // narrow but many neighbours and wide enough -> body (a short body line)
        assert!(!is_non_body(&Feat { w_rel: 0.05, y_rel: 0.5, same_left: 40, conf: 0.99 }));
        // very narrow stays non-body even among neighbours
        assert!(is_non_body(&Feat { w_rel: 0.02, y_rel: 0.5, same_left: 40, conf: 0.99 }));
        // wide in the top margin -> running header
        assert!(is_non_body(&Feat { w_rel: 0.4, y_rel: 0.03, same_left: 40, conf: 0.99 }));
        // wide, mid-page, confident -> body
        assert!(!is_non_body(&Feat { w_rel: 0.4, y_rel: 0.5, same_left: 40, conf: 0.99 }));
        // wide but the recognizer was unsure -> non-body
        assert!(is_non_body(&Feat { w_rel: 0.4, y_rel: 0.5, same_left: 40, conf: 0.5 }));
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
