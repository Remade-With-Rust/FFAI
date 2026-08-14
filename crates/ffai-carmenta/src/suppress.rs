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
    /// This line follows a "References" heading in READING ORDER (§8.130).
    /// Style-independent by construction: it catches numbered and parenthesised
    /// references identically, which is what §8.127 proved no syntax rule could
    /// do — the three worst pages share no citation format at all.
    after_refs: bool,
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

    // THE SECTION-SCOPE branch (§8.130). Everything after a references heading
    // is outside the document body whatever citation style it uses. Found by
    // RENDERING the failing pages rather than by searching features: on
    // `omni-0039` the heading is annotated and all 76 lines after it are not.
    //
    // Measured incrementally on the corpus at precision 1.000 over 218 lines and
    // six pages, +1.995 pp holdout macro, and — alone among every candidate this
    // campaign tested — it improves the ordinary pages too (+0.436 pp body).
    by_width || isolated_fragment || bibliography || f.after_refs
}

/// Column index per line, by GUTTER rather than by left edge (§8.130).
///
/// A left edge cannot tell an indent from a column. Clustering `x` split
/// `omni-0055`'s indented numbered notes into a phantom column that sorted after
/// the real one, which swept 40 annotated lines in with its references.
///
/// A column boundary is a vertical band of the page that no line crosses, so the
/// occupancy is built from each line's FULL EXTENT and the empty bands are the
/// gutters. Nearly-empty rather than exactly empty: a running header spans the
/// middle of `omni-0055`, and one crossing line is enough to make a strict test
/// report a single column.
fn column_of(boxes: &[(f32, f32, f32, f32)], pw: f32) -> Vec<usize> {
    const BINS: usize = 400;
    const MIN_GUTTER: f32 = 0.015;
    const SPAN_TOL: f32 = 0.03;
    if boxes.is_empty() {
        return Vec::new();
    }
    let mut occ = [0usize; BINS];
    for &(x, _, w, _) in boxes {
        let a = ((BINS as f32 * x / pw) as isize).clamp(0, BINS as isize - 1) as usize;
        let b = ((BINS as f32 * (x + w) / pw) as isize).clamp(0, BINS as isize - 1) as usize;
        for o in occ.iter_mut().take(b + 1).skip(a) {
            *o += 1;
        }
    }
    let lo = occ.iter().position(|&v| v > 0).unwrap_or(0);
    let hi = occ.iter().rposition(|&v| v > 0).unwrap_or(BINS - 1);
    let thresh = ((SPAN_TOL * boxes.len() as f32) as usize).max(1);

    let mut gutters: Vec<f32> = Vec::new();
    let mut i = lo;
    while i <= hi {
        if occ[i] <= thresh {
            let mut j = i;
            while j <= hi && occ[j] <= thresh {
                j += 1;
            }
            if (j - i) as f32 / BINS as f32 >= MIN_GUTTER {
                gutters.push((i + j) as f32 / 2.0 / BINS as f32);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    boxes
        .iter()
        .map(|&(x, _, w, _)| {
            let c = (x + w / 2.0) / pw;
            gutters.iter().filter(|&&g| c > g).count()
        })
        .collect()
}

/// Is this line a heading that ends the document body?
///
/// Matched on the WHOLE line, trimmed of trailing punctuation — a paragraph
/// mentioning "references" must not fire it.
fn is_refs_heading(s: &str) -> bool {
    let t: String = s
        .trim()
        .trim_end_matches([':', '.', ' '])
        .to_ascii_lowercase();
    matches!(
        t.as_str(),
        "references"
            | "reference list"
            | "bibliography"
            | "works cited"
            | "literature cited"
            | "citations"
    )
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


// ---------------------------------------------------------------------------
// The BLOCK branch (§8.131-§8.135)
// ---------------------------------------------------------------------------
//
// Everything above decides one LINE at a time. §8.131 measured that the wrong
// unit: group the page in 2D and **98 % of characters sit in blocks that are
// >= 90 % one class**, so a block-level decision loses almost nothing to a
// line-level one — and a block carries features a line cannot have (its area,
// aspect, isolation, and how consistent its lines are with each other).
//
// The grouping is deliberately NOT `run_stats_per_line`'s. That clusters by LEFT
// EDGE and then splits on vertical pitch, which is right for a prose column and
// cannot form a figure's block: a chart's title, axis labels and tick values
// share no left edge. Here two lines join when they OVERLAP HORIZONTALLY and sit
// within two line heights vertically, so a caption joins its plot.
//
// Blocks are formed from the lines the branches above KEEP, and the page's p90
// and line height are recomputed on that surviving population — the page the
// filter leaves behind is the page this branch sees. Getting that wrong is what
// made §8.133 report a 91 % collapse: it scored rules against the geometry of
// the ORIGINAL block, and a ten-line block whose filter removed eight is a
// two-line fragment with entirely different width, height, area and aspect.

/// Union-find over 2D adjacency. Returns one group of line indices per block.
fn group_blocks(idx: &[usize], boxes: &[(f32, f32, f32, f32)], lh: f32) -> Vec<Vec<usize>> {
    const V_GAP: f32 = 2.0;
    const H_OVERLAP: f32 = 0.15;
    let n = idx.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut Vec<usize>, mut a: usize) -> usize {
        while p[a] != a {
            p[a] = p[p[a]];
            a = p[a];
        }
        a
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let (ax, ay, aw, ah) = boxes[idx[i]];
            let (bx, by, bw, bh) = boxes[idx[j]];
            let ov = (ax + aw).min(bx + bw) - ax.max(bx);
            if ov < H_OVERLAP * aw.min(bw) {
                continue;
            }
            if ((ay + ah / 2.0) - (by + bh / 2.0)).abs() / lh <= V_GAP {
                let (ra, rb) = (find(&mut parent, i), find(&mut parent, j));
                parent[ra] = rb;
            }
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = Default::default();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(idx[i]);
    }
    groups.into_values().collect()
}

/// The three rules, fitted on train and judged once on holdout at **+0.891 pp
/// macro** — holdout ABOVE train (+0.573), which is the opposite of the overfit
/// signature that refused a fourth rule. 431 blocks over **160 pages, 110 of
/// them gaining**; every line-level lever this campaign refused sat on one to
/// three pages.
///
/// What they are, physically: isolated single-line strips, thin isolated bands,
/// and narrow low-confidence blocks — captions, labels, header strips, axis text.
fn block_is_non_body(f: &BlockFeat) -> bool {
    // A: an isolated strip more than six times wider than tall.
    let strip = f.area_frac < 0.004 && f.isolation > 1.2 && f.aspect > 6.0;
    // B: a thin isolated band whose lines agree on width.
    let band = f.blk_h < 0.015 && f.isolation > 1.2 && f.w_cv < 0.2;
    // C: a narrow block the recognizer was not sure about.
    let narrow = f.w_med < 0.30 && f.blk_w < 0.40 && f.conf < 0.96;
    strip || band || narrow
}

struct BlockFeat {
    area_frac: f32,
    aspect: f32,
    isolation: f32,
    blk_h: f32,
    blk_w: f32,
    w_cv: f32,
    w_med: f32,
    conf: f32,
}

/// Population standard deviation over mean — scale-free irregularity, and the
/// population form because a block IS its lines, not a sample of them.
fn cv(v: &[f32]) -> f32 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = v.iter().sum::<f32>() / v.len() as f32;
    if m == 0.0 {
        return 0.0;
    }
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32;
    var.sqrt() / m
}

/// The TRUE median: the mean of the two middle values when the count is even.
///
/// Not `v[len/2]`. The block rules were fitted with Python's `statistics.median`,
/// which averages, and taking the upper-middle instead silently changes `w_med`
/// and the page's line height on every block with an even number of lines. It
/// cost a 463 pp disagreement on `omni-0245` — which was first misdiagnosed as a
/// stale dump (§8.136) before the dumps were refreshed and came back
/// byte-identical (§8.137).
fn median(v: &mut Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
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

    // Section scope: order the page by column then by `y`, find a references
    // heading, and mark everything after it (§8.130).
    let cols = column_of(&boxes, pw);
    let mut order: Vec<usize> = (0..lines.len()).collect();
    order.sort_by(|&a, &b| {
        (cols[a], boxes[a].1)
            .partial_cmp(&(cols[b], boxes[b].1))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut after_refs = vec![false; lines.len()];
    if let Some(pos) = order.iter().position(|&i| is_refs_heading(&lines[i].text)) {
        let head = order[pos];
        let n_cols = cols.iter().collect::<std::collections::HashSet<_>>().len();
        // ABSTAIN when the layout contradicts the detection: a section heading
        // in a genuinely single-column page starts at the left margin, so one
        // sitting mid-page means there are columns we did not find and ordering
        // by `y` would interleave them. `omni-0060` is exactly this — heading at
        // x = 841 of 1653 with one column reported — and marking it cost 34
        // annotated lines. Abstaining costs that page's references; guessing
        // costs another page's body (§8.101).
        if !(n_cols == 1 && boxes[head].0 > 0.25 * pw) {
            for &i in &order[pos + 1..] {
                after_refs[i] = true;
            }
        }
    }

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
                after_refs: after_refs[i],
                page_year_hits,
            })
        })
        .collect();
    // ---- SECOND PASS: the block branch (§8.136) --------------------------
    // Runs on the lines the branches above KEEP, with the page statistics
    // recomputed on that surviving population.
    let mut keep = keep;
    let surv: Vec<usize> = (0..lines.len()).filter(|&i| keep[i]).collect();
    if surv.len() >= 2 {
        let mut hs2: Vec<f32> = surv.iter().map(|&i| boxes[i].3).collect();
        let lh2 = median(&mut hs2).max(1.0);
        let mut ws2: Vec<f32> = surv.iter().map(|&i| boxes[i].2 / pw).collect();
        ws2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p90_2 = ws2[((ws2.len() - 1) * 9) / 10].max(1e-6);

        let blocks = group_blocks(&surv, &boxes, lh2);
        let bb: Vec<(f32, f32, f32, f32)> = blocks
            .iter()
            .map(|g| {
                let x0 = g.iter().map(|&i| boxes[i].0).fold(f32::INFINITY, f32::min);
                let y0 = g.iter().map(|&i| boxes[i].1).fold(f32::INFINITY, f32::min);
                let x1 = g.iter().map(|&i| boxes[i].0 + boxes[i].2).fold(f32::NEG_INFINITY, f32::max);
                let y1 = g.iter().map(|&i| boxes[i].1 + boxes[i].3).fold(f32::NEG_INFINITY, f32::max);
                (x0, y0, x1, y1)
            })
            .collect();

        for (k, g) in blocks.iter().enumerate() {
            let (x0, y0, x1, y1) = bb[k];
            // Distance to the nearest OTHER block, in line heights: zero when
            // they overlap on an axis, so a block inside a column reads ~0 and a
            // caption stranded in whitespace reads high.
            let isolation = bb
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != k)
                .map(|(_, d)| {
                    let dx = (x0 - d.2).max(d.0 - x1).max(0.0);
                    let dy = (y0 - d.3).max(d.1 - y1).max(0.0);
                    (dx * dx + dy * dy).sqrt() / lh2
                })
                .fold(f32::INFINITY, f32::min);
            let mut wl: Vec<f32> = g.iter().map(|&i| boxes[i].2 / pw).collect();
            let w_cv = cv(&wl);
            let w_med = median(&mut wl) / p90_2;
            let conf = g.iter().map(|&i| lines[i].confidence.unwrap_or(1.0)).sum::<f32>()
                / g.len() as f32;
            let f = BlockFeat {
                area_frac: ((x1 - x0) * (y1 - y0)) / (pw * ph),
                aspect: (x1 - x0) / (y1 - y0).max(1.0),
                isolation: if isolation.is_finite() { isolation.min(40.0) } else { 40.0 },
                blk_h: (y1 - y0) / ph,
                blk_w: (x1 - x0) / pw,
                w_cv,
                w_med,
                conf,
            };
            if block_is_non_body(&f) {
                for &i in g {
                    keep[i] = false;
                }
            }
        }
    }

    lines.into_iter().zip(keep).filter_map(|(l, k)| k.then_some(l)).collect()
}

// ---------------------------------------------------------------------------
// §8.157 — the axis-price reorder, for tall dense multi-column pages.
//
// §8.156 closed the SPARSE half of the ordering problem (+0.562 pp, engine
// measured). 4.30 pp of the 4.80 pp remains, 3.04 pp of it on 116 dense 3+
// column pages that gate is disjoint from by construction.
//
// §8.155 built a recursive cut whose axis decision is explicitly PRICED — a
// gutter is wide in CHARACTERS, a band gap is tall in LINES — and refuted it as
// a default: it never beats `order_by_selection` on either split, at any alpha.
// But its failure is not uniform. It RESCUES 24 dense pages at a mean of
// +16.6 pp and WRECKS 21 at -18.2 pp, and they cancel to +0.057 pp. Fired only
// where it helps it is worth **+1.773 pp corpus macro**.
//
// So this is not a better ordering. It is a second ordering that is right on a
// population the shipped one is wrong on, and the whole value is in the guard.
//
// WHY IT LIVES HERE AND NOT IN `order_reading`. The guard needs `body_frac`
// (post-suppression) and the cut needs `med_cw = width / chars` (post-
// recognition). Neither exists at `engine.rs:400` where ordering runs. §8.156's
// gate is pure geometry and drops into `order_reading`; this one cannot.
//
// The guard's geometry is computed on the FULL line population, before
// suppression deletes any of it — §8.153's D6a found gutters over 111 lines are
// not gutters over 80, and that defect voided an entire measurement harness.

/// Page statistics for the §8.157 guard, taken BEFORE suppression.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProbeStats {
    n_col: usize,
    cover: f32,
    aspect: f32,
    n_all: usize,
}

/// Column count by left-edge clustering; a column needs >= 4 lines, so a centred
/// caption is not one. Coverage is text-box area over page area.
pub fn probe_stats(lines: &[OcrLine], page_w: f32, page_h: f32) -> ProbeStats {
    let (pw, ph) = (page_w.max(1.0), page_h.max(1.0));
    let bx: Vec<(f32, f32, f32, f32)> = lines
        .iter()
        .filter_map(|l| l.bbox.as_ref().map(|b| (b.x, b.y, b.width, b.height)))
        .collect();
    if bx.is_empty() {
        return ProbeStats::default();
    }
    let mut lefts: Vec<f32> = bx.iter().map(|b| b.0 / pw).collect();
    lefts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (mut n_col, mut run) = (0usize, 1usize);
    for i in 1..lefts.len() {
        if lefts[i] - lefts[i - 1] < 0.03 {
            run += 1;
        } else {
            if run >= 4 {
                n_col += 1;
            }
            run = 1;
        }
    }
    if run >= 4 {
        n_col += 1;
    }
    ProbeStats {
        n_col,
        cover: bx.iter().map(|b| b.2 * b.3).sum::<f32>() / (pw * ph),
        aspect: ph / pw,
        n_all: lines.len(),
    }
}

/// Fitted on 44 train / judged on 224 holdout dense 3+ column pages: holdout
/// +0.228 pp macro, 95 % CI [+0.059, +0.438], **7 rescues and 0 wrecks**.
///
/// The column condition is LOAD-BEARING, not scoping. Without it the rule fires
/// on `omni-0063` — a tall dense TWO-column page the shipped order reads at
/// 5.8 % — and takes it to 73.4 %, which alone flips the corpus result to
/// -0.053 pp with a CI spanning zero.
fn probe_gate_fires(st: &ProbeStats, n_body: usize) -> bool {
    let body_frac = if st.n_all == 0 { 0.0 } else { n_body as f32 / st.n_all as f32 };
    st.n_col >= 3 && st.cover >= 0.18 && st.aspect > 1.6 && body_frac > 0.85
}

/// Maximal interior sub-ranges of `[lo, hi)` that no interval covers.
fn free_runs(mut iv: Vec<(f32, f32)>, lo: f32, hi: f32, min_w: f32) -> Vec<(f32, f32)> {
    if iv.is_empty() {
        return Vec::new();
    }
    iv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (mut out, mut cur) = (Vec::new(), lo);
    for (a, b) in iv {
        if a - cur >= min_w && a > lo {
            out.push((cur, a));
        }
        cur = cur.max(b);
    }
    if hi - cur >= min_w && cur > lo {
        out.push((cur, hi));
    }
    // Strictly interior: a run touching either edge is the margin, not a gutter.
    out.retain(|r| r.0 > lo && r.1 < hi);
    out
}

/// One line as the cut sees it: box, character count, and its index in the input.
#[derive(Clone, Copy)]
struct Item {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    chars: f32,
    idx: usize,
}

const PROBE_SPAN_FRAC: f32 = 0.50;
const PROBE_GUTTER_MIN: f32 = 0.015;
const PROBE_HGAP_LINES: f32 = 0.8;
/// The axis price. A vertical cut wins when the gutter, measured in CHARACTER
/// widths, beats ALPHA times the band gap measured in LINE heights. Swept on
/// train over 0, 0.5, 1, 2, 4, 8, 16, 64 and infinity; 4.0 was best (§8.155).
const PROBE_ALPHA: f32 = 4.0;

fn probe_cut(it: &[Item], bx: (f32, f32, f32, f32), depth: usize) -> Vec<usize> {
    let by_y = |v: &[Item]| {
        let mut s: Vec<Item> = v.to_vec();
        s.sort_by(|a, b| (a.y, a.x).partial_cmp(&(b.y, b.x)).unwrap_or(std::cmp::Ordering::Equal));
        s.into_iter().map(|i| i.idx).collect::<Vec<_>>()
    };
    if it.len() < 2 || depth >= 8 {
        return by_y(it);
    }
    let (x0, y0, x1, y1) = bx;
    let nw = (x1 - x0).max(1.0);
    let med_h = median(&mut it.iter().map(|i| i.h).collect()).max(1.0);
    let med_cw = median(&mut it.iter().map(|i| i.w / i.chars.max(1.0)).collect()).max(1.0);

    // Vertical gutters. Spanning lines are excluded from the projection but not
    // from the node — a headline legitimately crosses any gutter. Boxes are
    // ERODED first: detection arrives unclipped by ~1.5x, enough to close a real
    // gutter completely (see `boxes::find_gutters`).
    let span_w = nw * PROBE_SPAN_FRAC;
    let xs: Vec<(f32, f32)> = it
        .iter()
        .filter(|i| i.w < span_w)
        .map(|i| {
            let er = (i.h * 0.6).min(i.w * 0.25);
            (i.x + er, i.x + i.w - er)
        })
        .collect();
    let gut = free_runs(xs, x0, x1, (nw * PROBE_GUTTER_MIN).max(med_cw * 0.8));
    let ys: Vec<(f32, f32)> = it.iter().map(|i| (i.y, i.y + i.h)).collect();
    let hgap = free_runs(ys, y0, y1, med_h * PROBE_HGAP_LINES);

    let best_g = gut.iter().map(|g| g.1 - g.0).fold(0.0f32, f32::max) / med_cw;
    let best_h = hgap.iter().map(|g| g.1 - g.0).fold(0.0f32, f32::max) / med_h;

    if best_g > 0.0 && (best_h == 0.0 || best_g > PROBE_ALPHA * best_h) {
        let mut edges: Vec<f32> = gut.iter().map(|g| g.0 + (g.1 - g.0) / 2.0).collect();
        edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut cols: Vec<Vec<Item>> = vec![Vec::new(); edges.len() + 1];
        for i in it {
            let c = i.x + i.w / 2.0;
            cols[edges.iter().filter(|e| c > **e).count()].push(*i);
        }
        if cols.iter().filter(|c| !c.is_empty()).count() >= 2 {
            let mut bounds = vec![x0];
            bounds.extend(edges);
            bounds.push(x1);
            let mut out = Vec::new();
            for (k, c) in cols.iter().enumerate() {
                if !c.is_empty() {
                    out.extend(probe_cut(c, (bounds[k], y0, bounds[k + 1], y1), depth + 1));
                }
            }
            return out;
        }
    }
    if best_h > 0.0 {
        let e = hgap
            .iter()
            .max_by(|a, b| (a.1 - a.0).partial_cmp(&(b.1 - b.0)).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or((0.0, 0.0));
        let mid = e.0 + (e.1 - e.0) / 2.0;
        let (top, bot): (Vec<Item>, Vec<Item>) =
            it.iter().partition(|i| i.y + i.h / 2.0 < mid);
        if !top.is_empty() && !bot.is_empty() {
            let mut out = probe_cut(&top, (x0, y0, x1, mid), depth + 1);
            out.extend(probe_cut(&bot, (x0, mid, x1, y1), depth + 1));
            return out;
        }
    }
    by_y(it)
}

/// Re-order body lines with the §8.157 axis-price cut, if the guard fires.
///
/// DEFAULT ON. `FFAI_ORDER_PROBE=0` disables.
///
/// Engine A/B over 236 holdout pages, one binary and two env settings with the
/// arms interleaved, measured on top of the shipped §8.156 gate:
///
/// | | MACRO | MICRO |
/// |---|---|---|
/// | off | 19.897 % | 14.318 % |
/// | on | **19.666 %** | **13.648 %** |
/// | gain | +0.231 pp | +0.670 pp |
///
/// 95 % CI [+0.069, +0.436], excludes zero. 11 pages changed, 9 better, 2 worse.
///
/// MICRO moves nearly 3x macro — the mirror of §8.156, which fires on small pages
/// and reads +0.562 macro against +0.069 micro. This one fires on LARGE dense
/// pages, so character-weighting sees it and page-weighting barely does. The two
/// gates cover opposite halves of the ordering problem: combined they are
/// +0.793 pp macro and +0.738 pp micro.
pub fn probe_reorder(
    lines: Vec<OcrLine>,
    st: &ProbeStats,
    lowconf: f32,
    page_w: f32,
    page_h: f32,
) -> Vec<OcrLine> {
    if std::env::var("FFAI_ORDER_PROBE").as_deref() == Ok("0") {
        return lines;
    }
    // FFAI_VERIFY_LOWCONF overrides the per-model constant so §8.171's
    // calibration can be swept from ONE binary. Env-reading stays in this
    // wrapper; `probe_apply` is pure so tests need not mutate process env.
    let lowconf = env_f32_opt("FFAI_VERIFY_LOWCONF").unwrap_or(lowconf);
    probe_apply(lines, st, lowconf, page_w, page_h)
}

/// A float from the environment, or `None` when unset or unparseable.
fn env_f32_opt(key: &str) -> Option<f32> {
    std::env::var(key).ok()?.parse().ok()
}

/// The guard and the reorder, without the env check — so tests exercise the
/// behaviour without mutating process env, which in Rust 2024 is `unsafe` and
/// races every other test in the binary.
fn probe_apply(
    lines: Vec<OcrLine>,
    st: &ProbeStats,
    lowconf: f32,
    page_w: f32,
    page_h: f32,
) -> Vec<OcrLine> {
    // W1 (plan §11): the Great Gate feature tap. `probe_decide` returns early on
    // four paths, so the emit lives HERE, outside it, and fires for every page —
    // tapping at the decision point would harvest only pages that reached it and
    // silently narrow the denominator (§3 rule 10). Disabled, this is one cached
    // bool and no allocation; `harvest_inert_when_unset` pins that.
    if !crate::harvest::enabled() {
        return probe_decide(lines, st, lowconf, page_w, page_h, None);
    }
    let mut row = crate::harvest::Row::new();
    let out = probe_decide(lines, st, lowconf, page_w, page_h, Some(&mut row));
    row.emit();
    out
}

/// Columns filled only once the decision is actually reached, in emit order.
/// Early exits fill them as MISSING via `Row::missing` so every row carries
/// every column — see `harvest_row_width_is_constant`.
const DECISION_COLS: &[&str] = &[
    "probe_valid", "gate_fires", "verifier_blind",
    "join_shipped", "join_probe", "join_margin",
    "numseq_shipped", "numseq_probe", "numseq_margin",
    "frac_moved", "accepted",
];

/// The ordering decision itself. `row`, when present, is filled as the decision
/// proceeds; it never changes what is returned.
fn probe_decide(
    lines: Vec<OcrLine>,
    st: &ProbeStats,
    lowconf: f32,
    page_w: f32,
    page_h: f32,
    mut row: Option<&mut crate::harvest::Row>,
) -> Vec<OcrLine> {
    // Page-level features, available on EVERY path including the early guards.
    if let Some(r) = row.as_deref_mut() {
        let n = lines.len().max(1) as f64;
        let confs: Vec<f32> = lines.iter().filter_map(|l| l.confidence).collect();
        let mut sorted = confs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.is_empty() { f64::NAN } else { sorted[sorted.len() / 2] as f64 };
        let heights: f64 = lines.iter().filter_map(|l| l.bbox.as_ref()).map(|b| b.height as f64).sum();
        let chars: f64 = lines.iter().map(|l| l.text.chars().count() as f64).sum();
        r.put("n_col", st.n_col as f64);
        r.put("cover", st.cover as f64);
        r.put("aspect", st.aspect as f64);
        r.put("n_all", st.n_all as f64);
        r.put("n_lines", lines.len() as f64);
        r.put("body_frac", lines.len() as f64 / st.n_all.max(1) as f64);
        r.put("page_w", page_w as f64);
        r.put("page_h", page_h as f64);
        r.put("page_aspect", (page_h / page_w.max(1.0)) as f64);
        r.put("conf_median", median);
        r.put("conf_mean", if confs.is_empty() { f64::NAN } else { confs.iter().map(|&c| c as f64).sum::<f64>() / confs.len() as f64 });
        r.put("conf_frac_low", confs.iter().filter(|&&c| c < lowconf).count() as f64 / confs.len().max(1) as f64);
        r.put("conf_frac_low_096", confs.iter().filter(|&&c| c < 0.96).count() as f64 / confs.len().max(1) as f64);
        r.put("mean_line_h", heights / n);
        r.put("total_chars", chars);
        r.put("mean_chars", chars / n);
        r.put("lowconf_thresh", lowconf as f64);
        // FLOAT-INTERRUPTION PROXIES (§13). 67.9 % of English reading-order
        // error is on pages carrying a figure, table or isolated equation, and
        // our detector emits no such regions — so the signal has to be
        // geometric. A float shows up as a tall band with no text, and as lines
        // that stop short of the page's usual measure. Whether any of these
        // actually separates the population is the question the harvest asks;
        // they are emitted because a signal absent from the CSV can never be
        // searched for.
        let mut bx: Vec<(f32, f32, f32, f32)> = lines
            .iter()
            .filter_map(|l| l.bbox.as_ref().map(|b| (b.y, b.height, b.x, b.width)))
            .collect();
        bx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut max_gap = 0.0f32;
        let mut bands = 1usize;
        let med_h = if bx.is_empty() { 1.0 } else { bx[bx.len() / 2].1.max(1.0) };
        for w in bx.windows(2) {
            let gap = w[1].0 - (w[0].0 + w[0].1);
            if gap > max_gap {
                max_gap = gap;
            }
            if gap > 2.0 * med_h {
                bands += 1;
            }
        }
        let widths: Vec<f32> = bx.iter().map(|b| b.3).collect();
        let wmax = widths.iter().cloned().fold(0.0f32, f32::max).max(1.0);
        let wmean = if widths.is_empty() { 0.0 } else { widths.iter().sum::<f32>() / widths.len() as f32 };
        let wsd = if widths.len() < 2 { 0.0 } else {
            (widths.iter().map(|w| (w - wmean).powi(2)).sum::<f32>() / widths.len() as f32).sqrt()
        };
        let rights: Vec<f32> = bx.iter().map(|b| b.2 + b.3).collect();
        let rmean = if rights.is_empty() { 0.0 } else { rights.iter().sum::<f32>() / rights.len() as f32 };
        let rsd = if rights.len() < 2 { 0.0 } else {
            (rights.iter().map(|r| (r - rmean).powi(2)).sum::<f32>() / rights.len() as f32).sqrt()
        };
        r.put("max_vgap", (max_gap / page_h.max(1.0)) as f64);
        r.put("n_bands", bands as f64);
        r.put("width_cv", if wmean > 0.0 { (wsd / wmean) as f64 } else { f64::NAN });
        r.put("frac_short", widths.iter().filter(|&&w| w < 0.6 * wmax).count() as f64
            / widths.len().max(1) as f64);
        r.put("right_cv", if rmean > 0.0 { (rsd / rmean) as f64 } else { f64::NAN });
        r.put("max_line_w_frac", (wmax / page_w.max(1.0)) as f64);
    }
    // §8.160 widens §8.157: the probe order is computed for EVERY dense
    // multi-column page, and accepted on either of two grounds:
    //   1. the §8.157 aspect guard (engine-verified, +0.231 pp macro), or
    //   2. the TEXT VERIFIER — the probe's join fluency strictly beats the
    //      shipped order's, or ties it while its numbered-item sequence is
    //      strictly more monotone.
    // Every ordering score before the verifier was geometric and all ten
    // failed the rescue/wreck question (ceiling AUC 0.643) for one reason: a
    // wrong ordering can be more geometrically tidy than a right one. Reading
    // order is a LINGUISTIC property; scored in text space the same question
    // reads AUC 0.750, and the three catastrophic wrecks (-69.0, -47.6, -45.5)
    // all sit at negative margin — their shipped text already flows, and
    // breaking it is visible. `FFAI_ORDER_VERIFY=0` disables ground 2.
    // §13 HARVEST ARM, off unless asked. `FFAI_ORDER_PROBE_ALL=1` bypasses the
    // column/coverage guard so the probe is COMPUTED on every page with enough
    // lines to order. Without it a gate search over 1- and 2-column pages is
    // impossible: the probe never runs there, so an ON/OFF harvest reads gain 0
    // on every page and the calculator has nothing to find. 67.9 % of English
    // reading-order error lives in that excluded population.
    //
    // This is a MEASUREMENT arm, not a proposal. Forcing the probe everywhere
    // is expected to LOSE overall — the guard exists because §8.157 measured
    // that it should. The harvest asks a narrower question: on which pages, by
    // which decision-time signal, would it have won? A rule that answers it
    // becomes the gate's new axis; if none does, the guard stays and we have
    // refuted the lever with data instead of extending it on a hunch.
    let probe_all = std::env::var("FFAI_ORDER_PROBE_ALL").as_deref() == Ok("1");
    // Pages the SHIPPED guard rejects. Under the harvest arm these are reordered
    // UNCONDITIONALLY — the verifier's opinion is not what we are asking about.
    // We want the ORACLE gain: what reordering is worth on this page, so the
    // calculator can search for the signal that predicts it. Pages the guard
    // ADMITS keep the normal accept logic, so the arm changes exactly one
    // population and the 3+ column result stays comparable.
    let bypassed = probe_all && (st.n_col < 3 || st.cover < 0.18);
    if lines.len() < 3 || (!probe_all && (st.n_col < 3 || st.cover < 0.18)) {
        if let Some(r) = row.as_deref_mut() {
            r.put_bool("reached_probe", false);
            r.missing(DECISION_COLS);
        }
        return lines;
    }
    if let Some(r) = row.as_deref_mut() {
        r.put_bool("reached_probe", true);
    }
    let it: Vec<Item> = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, l)| {
            l.bbox.as_ref().map(|b| Item {
                x: b.x,
                y: b.y,
                w: b.width,
                h: b.height,
                chars: l.text.chars().count() as f32,
                idx,
            })
        })
        .collect();
    if it.len() != lines.len() {
        if let Some(r) = row.as_deref_mut() {
            r.put_bool("probe_valid", false);
            r.missing(&DECISION_COLS[1..]);
        }
        return lines; // a line without a box: cannot order what cannot be placed
    }
    let order = probe_cut(&it, (0.0, 0.0, page_w.max(1.0), page_h.max(1.0)), 0);
    if order.len() != lines.len() {
        if let Some(r) = row.as_deref_mut() {
            r.put_bool("probe_valid", false);
            r.missing(&DECISION_COLS[1..]);
        }
        return lines; // not a permutation — refuse rather than drop output
    }
    // The verifier's own signals, computed here whether or not the verifier is
    // the branch that decides — a rule search needs the margin on pages the
    // §8.157 guard admitted too, or it can only ever learn about the pages the
    // verifier already handles.
    if let Some(r) = row.as_deref_mut() {
        let shipped: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
        let probed: Vec<&str> = order.iter().map(|&i| lines[i].text.as_str()).collect();
        let (js, jp) = (join_fluency(&shipped), join_fluency(&probed));
        let (ns, np) = (num_seq_monotone(&shipped), num_seq_monotone(&probed));
        let moved = order.iter().enumerate().filter(|&(i, &o)| i != o).count();
        r.put_bool("probe_valid", true);
        r.put_bool("gate_fires", probe_gate_fires(st, lines.len()));
        r.put_bool("verifier_blind", verifier_blind(&lines, lowconf));
        r.put("join_shipped", js as f64);
        r.put("join_probe", jp as f64);
        r.put("join_margin", (jp - js) as f64);
        r.put("numseq_shipped", ns as f64);
        r.put("numseq_probe", np as f64);
        r.put("numseq_margin", (np - ns) as f64);
        r.put("frac_moved", moved as f64 / lines.len().max(1) as f64);
    }
    // §31: ground 1 (the §8.157 geometric guard) BYPASSES the text verifier, and
    // one of its four terms is now dead. `body_frac = n_body / n_all` measured
    // how much SUPPRESSION removed, and §19 turned suppression off for the
    // benchmark config — so `n_body == n_all`, `body_frac` is always 1.0, and
    // `body_frac > 0.85` is always true. The guard fires more often than it was
    // ever fitted to, force-accepting reorders on pages §22 proved the verifier
    // judges well (disabling the verifier costs 0.0083 text).
    //
    // `FFAI_ORDER_GUARD=0` drops ground 1 so every page is decided by the
    // verifier. Default unchanged; this is an A/B-able arm, not a silent edit.
    let guard_on = std::env::var("FFAI_ORDER_GUARD").as_deref() != Ok("0");
    let accept = if bypassed {
        true // §13 harvest arm: oracle gain on the excluded population
    } else if guard_on && probe_gate_fires(st, lines.len()) {
        true // ground 1: the shipped §8.157 guard
    } else if std::env::var("FFAI_ORDER_VERIFY").as_deref() == Ok("0") {
        false
    } else if verifier_blind(&lines, lowconf) {
        // The verifier must not vote on text it cannot read. `omni-0080`'s CJK
        // columns reach the recognizer as garbage and its "evidence" was one
        // net join; 85.3 % of its lines sit under 0.96 against a population
        // median of 5.0 %, and no rescued page exceeds 8.3 %. The rule was
        // pre-registered before that harvest ran, reusing the §8.135 block
        // rules' existing low-confidence line rather than a new constant.
        // NOTE this is competence-gating, not order-gating: per-line confidence
        // is permutation-invariant, so it can never CHOOSE between orders —
        // §8.160 refuted that use by argument — but it can say whether the
        // strings the verifier reads mean anything.
        false
    } else {
        // ground 2: the text verifier, on the SAME recognized strings.
        let shipped: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
        let probed: Vec<&str> = order.iter().map(|&i| lines[i].text.as_str()).collect();
        let jm = join_fluency(&probed) - join_fluency(&shipped);
        if jm > 1e-9 {
            true
        } else if jm.abs() <= 1e-9 {
            num_seq_monotone(&probed) > num_seq_monotone(&shipped) + 1e-9
        } else {
            false
        }
    };
    if let Some(r) = row.as_deref_mut() {
        r.put_bool("accepted", accept);
    }
    if !accept {
        return lines;
    }
    let mut slot: Vec<Option<OcrLine>> = lines.into_iter().map(Some).collect();
    order.into_iter().filter_map(|i| slot[i].take()).collect()
}

/// §8.135's block-rule low-confidence line, reused rather than a new constant.
/// CRNN'S SCALE. §8.171: this number is a property of a RECOGNIZER'S CONFIDENCE
/// DISTRIBUTION, not of documents, and carrying it across models is a silent
/// bug. SVTR's median line confidence is 0.9689 against CRNN's 0.9892, so on
/// `omni-0003` the same page puts 27.8 % of lines under this line against
/// CRNN's 6.7 % — it trips the abstain under SVTR only, the verifier never
/// runs, and the page loses its row-major fix (5.1 -> 53.1). That single page
/// was the entire apparent deficit in the first conflated A/B.
pub const VERIFY_LOWCONF_CRNN: f32 = 0.96;
/// SVTR's scale, calibrated in §8.171 over all 236 holdout pages to reproduce
/// CRNN's abstain RATE rather than its numeric value: 18 pages against CRNN's
/// 19 (8.1 %). Carrying 0.96 across would have abstained on 66 pages — 28 %,
/// 3.5x the rate the verifier was measured under — silencing the reordering on
/// 47 pages that should keep it.
///
/// HONEST LIMIT, so it is not over-trusted: the rate matches but the PAGE SETS
/// only half overlap (9 shared, 9 SVTR-only, 10 CRNN-only, Jaccard 0.321). That
/// is defensible — competence is a property of a (model, page) pair, and a
/// better recognizer should find different pages legible — and 0.946 is also
/// the peak of page agreement, but it is weaker than "identical behaviour".
/// The full-config A/B, not this calibration, is what accepts the number.
///
/// FRAGILE: the constant sits on a steep part of the curve (0.946 -> 18 pages,
/// 0.956 -> 52). A future recognizer wants its own harvest, not this value; the
/// durable fix is a scale-free rule, which is open work.
pub const VERIFY_LOWCONF_SVTR: f32 = 0.946;
/// Abstain when more than this fraction of lines sits under the low-confidence
/// line. Scale-free (it is a fraction of the page), so it is NOT per-model.
const VERIFY_ABSTAIN_FRAC: f32 = 0.25;

/// True when too much of the page's text is unreliable for a TEXT verifier's
/// vote to mean anything (§8.160). Engine re-scored with this abstain:
/// +0.987 pp macro, 95 % CI [+0.179, +1.988], excludes zero.
fn verifier_blind(lines: &[OcrLine], lowconf: f32) -> bool {
    if lines.is_empty() {
        return true;
    }
    let low = lines
        .iter()
        .filter(|l| l.confidence.unwrap_or(0.0) < lowconf)
        .count();
    low as f32 / lines.len() as f32 > VERIFY_ABSTAIN_FRAC
}

/// Join fluency of a reading order (§8.160). Higher = the text flows better.
///
/// Per consecutive pair: +2 when a hyphenated fragment is completed by a
/// lowercase (or digit) start; +1 when a mid-sentence line is continued
/// lowercase; -1 when a mid-sentence line is followed by a capital (a broken
/// join). Crude on purpose — no dictionary, no language model. Crude measured
/// AUC 0.750 on the rescue/wreck question; refinement is upside, not rescue.
/// True for a character in a script with no letter case and no inter-word
/// spacing — CJK ideographs, kana, and the CJK punctuation block.
fn is_cjk(c: char) -> bool {
    let o = c as u32;
    (0x4E00..=0x9FFF).contains(&o)       // unified ideographs
        || (0x3400..=0x4DBF).contains(&o) // extension A
        || (0x3040..=0x309F).contains(&o) // hiragana
        || (0x30A0..=0x30FF).contains(&o) // katakana
        || (0xF900..=0xFAFF).contains(&o) // compatibility ideographs
        || (0x3000..=0x303F).contains(&o) // CJK punctuation 。、「」etc
        || (0xFF00..=0xFFEF).contains(&o) // fullwidth forms ，．！？
}

/// Is this passage predominantly CJK? Decided on the TEXT, which is free — we
/// already recognised it — rather than on a language-detector pass over the
/// image. §15: script routing needs DETECTION, not a detector.
fn mostly_cjk(texts: &[&str]) -> bool {
    let (mut cjk, mut total) = (0usize, 0usize);
    for t in texts {
        for c in t.chars() {
            if c.is_whitespace() {
                continue;
            }
            total += 1;
            if is_cjk(c) {
                cjk += 1;
            }
        }
    }
    total >= 8 && cjk * 4 >= total * 3 // >= 75 % of non-space characters
}

fn join_fluency(texts: &[&str]) -> f32 {
    // `FFAI_CJK_FLUENCY=0` forces the old Latin-only behaviour, so the CJK arm
    // can be A/B'd from ONE binary (§3 rule: one binary, two env settings —
    // §8.53 was lost comparing arms across builds). Default ON: the Latin arm is
    // measurably blind on CJK, so leaving it as the default would mean shipping
    // a verifier we have proven cannot work on 54 % of the benchmark.
    if mostly_cjk(texts) && std::env::var("FFAI_CJK_FLUENCY").as_deref() != Ok("0") {
        return join_fluency_cjk(texts);
    }
    join_fluency_latin(texts)
}

/// §15: the CJK arm. `join_fluency_latin` scores on letter CASE and Latin
/// terminators, and CJK has NEITHER — `join_fluency_is_blind_on_cjk` measures it
/// returning ~0 for every candidate ordering, so the §8.160 verifier could never
/// rank two orders on 54 % of the benchmark.
///
/// The Latin arm asks: does line B continue line A's sentence, or start a new
/// one? The same question in CJK is answered by different evidence:
///   * a line ending in a SENTENCE TERMINATOR (。！？) is a clean break, so a
///     following line neither confirms nor denies flow — score 0, as Latin does
///     after a full stop;
///   * a line ending in a NON-TERMINAL character mid-sentence should be
///     continued, and a continuation in CJK simply means the next line starts
///     with more text rather than with a closing bracket or a comma;
///   * a line ending in an OPENING bracket 「（ or a comma-class mark 、，
///     is explicitly unfinished — the strongest continuation evidence there is.
///
/// Sign convention matches the Latin arm: positive = this order flows.
fn join_fluency_cjk(texts: &[&str]) -> f32 {
    const SENT_END: &[char] = &['\u{3002}', '\u{FF01}', '\u{FF1F}', '.', '!', '?'];
    const COMMA: &[char] = &['\u{3001}', '\u{FF0C}', '\u{FF1B}', '\u{FF1A}', ',', ';', ':'];
    const OPEN: &[char] = &['\u{300C}', '\u{300E}', '\u{FF08}', '\u{3010}', '\u{FF3B}', '('];
    const CLOSE: &[char] = &['\u{300D}', '\u{300F}', '\u{FF09}', '\u{3011}', '\u{FF3D}', ')'];
    let (mut s, mut pairs) = (0.0f32, 0u32);
    for w in texts.windows(2) {
        let a = w[0].trim_end();
        let b = w[1].trim_start();
        let (Some(last), Some(first)) = (a.chars().next_back(), b.chars().next()) else {
            continue;
        };
        pairs += 1;
        if COMMA.contains(&last) || OPEN.contains(&last) {
            // Unambiguously mid-clause: the next line must continue it.
            s += 2.0;
        } else if SENT_END.contains(&last) {
            // A finished sentence. A following CLOSING mark is stranded — that
            // is the one arrangement a clean break cannot explain.
            if CLOSE.contains(&first) {
                s -= 1.0;
            }
        } else {
            // Mid-sentence with no punctuation cue, which is the common CJK
            // case. Continuing with ordinary text is coherent; opening a new
            // quotation or bracket right after an unterminated line is not.
            if OPEN.contains(&first) {
                s -= 1.0;
            } else if is_cjk(first) || first.is_alphanumeric() {
                s += 1.0;
            }
        }
    }
    s / (pairs.max(1) as f32)
}

fn join_fluency_latin(texts: &[&str]) -> f32 {
    const TERM: &[char] = &['.', '!', '?', ':', ';', '"', '\'', '\u{201d}', '\u{2019}', ')'];
    let (mut s, mut pairs) = (0.0f32, 0u32);
    for w in texts.windows(2) {
        let a = w[0].trim_end();
        let b = w[1].trim_start();
        let (Some(last), Some(first)) = (a.chars().next_back(), b.chars().next()) else {
            continue;
        };
        pairs += 1;
        if last == '-' && (first.is_lowercase() || first.is_ascii_digit()) {
            s += 2.0;
        } else if !TERM.contains(&last) && !last.is_ascii_digit() {
            if first.is_lowercase() {
                s += 1.0;
            } else if first.is_uppercase() {
                s -= 1.0;
            }
        }
    }
    s / (pairs.max(1) as f32)
}

/// Monotonicity of leading item numbers ("55.", "(12)", "3]") — §8.160.
///
/// Self-contained numbered units are invisible to `join_fluency` (every
/// inter-item join scores 0 in any order), but their numbers carry the
/// canonical order directly: `omni-0003` reads 55,58,61,56.. shipped and
/// 55,56,57,58.. probed, and the reference is monotone. 0.5 = no evidence.
fn num_seq_monotone(texts: &[&str]) -> f32 {
    let mut nums: Vec<u32> = Vec::new();
    for t in texts {
        let s = t.trim_start().trim_start_matches('(');
        let d: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        if d.is_empty() || d.len() > 4 {
            continue;
        }
        let rest = &s[d.len()..];
        if matches!(rest.chars().next(), Some('.') | Some(')') | Some(']')) {
            nums.push(d.parse().unwrap_or(0));
        }
    }
    if nums.len() < 3 {
        return 0.5;
    }
    nums.windows(2).filter(|w| w[1] > w[0]).count() as f32 / (nums.len() - 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each branch of the depth-4 tree, at a value the fit chose.
    #[test]
    fn tree_branches() {
        // small block, narrow, isolated -> label or table cell
        assert!(is_non_body(&Feat { run_chars: 20.0, w_p90: 0.10, y_rel: 0.5, nn_gap: 0.5, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, after_refs: false, page_year_hits: 0 }));
        // small block, narrow, but tight against a neighbour -> body
        assert!(!is_non_body(&Feat { run_chars: 20.0, w_p90: 0.10, y_rel: 0.5, nn_gap: 0.005, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, after_refs: false, page_year_hits: 0 }));
        // small block, wide, top margin -> running header
        assert!(is_non_body(&Feat { run_chars: 20.0, w_p90: 0.9, y_rel: 0.02, nn_gap: 0.5, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, after_refs: false, page_year_hits: 0 }));
        // small block, wide, mid-page, confident -> body
        assert!(!is_non_body(&Feat { run_chars: 20.0, w_p90: 0.9, y_rel: 0.5, nn_gap: 0.5, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, after_refs: false, page_year_hits: 0 }));
        // large block, low confidence, mid-sized -> non-body
        assert!(is_non_body(&Feat { run_chars: 100.0, w_p90: 0.05, y_rel: 0.5, nn_gap: 0.5, conf: 0.5, run_lines: 40, same_left: 50, year_paren: false, after_refs: false, page_year_hits: 0 }));
        // genuinely long prose, even at low confidence -> body
        assert!(!is_non_body(&Feat { run_chars: 900.0, w_p90: 0.9, y_rel: 0.5, nn_gap: 0.5, conf: 0.5, run_lines: 40, same_left: 50, year_paren: false, after_refs: false, page_year_hits: 0 }));
        // large, confident, top margin -> running header
        assert!(is_non_body(&Feat { run_chars: 900.0, w_p90: 0.9, y_rel: 0.02, nn_gap: 0.5, conf: 0.99, run_lines: 40, same_left: 50, year_paren: false, after_refs: false, page_year_hits: 0 }));
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
            after_refs: false,
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
            after_refs: false,
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

    /// Gutter columns (§8.130): a left EDGE cannot tell an indent from a column.
    #[test]
    fn columns_by_gutter() {
        // Two columns of width 400 with a 200px gutter, on a 1000px page.
        let mut b: Vec<(f32, f32, f32, f32)> = Vec::new();
        for i in 0..10 {
            b.push((50.0, i as f32 * 20.0, 350.0, 15.0));
            b.push((600.0, i as f32 * 20.0, 350.0, 15.0));
        }
        let c = column_of(&b, 1000.0);
        assert_eq!(c.iter().filter(|&&v| v == 0).count(), 10);
        assert_eq!(c.iter().filter(|&&v| v == 1).count(), 10);

        // An INDENT inside one column must not become a column. Same left block,
        // every other line pushed right by 40px — no gutter is created.
        let mut d: Vec<(f32, f32, f32, f32)> = Vec::new();
        for i in 0..10 {
            let x = if i % 2 == 0 { 50.0 } else { 90.0 };
            d.push((x, i as f32 * 20.0, 350.0, 15.0));
        }
        assert_eq!(column_of(&d, 1000.0).iter().collect::<std::collections::HashSet<_>>().len(), 1);

        // ONE spanning line (a running header) must not fill the gutter.
        let mut e = b.clone();
        e.push((50.0, -30.0, 900.0, 15.0));
        let ce = column_of(&e, 1000.0);
        assert_eq!(ce.iter().collect::<std::collections::HashSet<_>>().len(), 2);
    }

    /// The heading matcher fires on a heading and not on prose mentioning one.
    #[test]
    fn refs_heading_syntax() {
        for s in ["References", "REFERENCES", "  references  ", "Bibliography",
                  "Works Cited", "References:", "Reference List", "Literature Cited"] {
            assert!(is_refs_heading(s), "should match: {s}");
        }
        for s in ["See the references at the end", "References to prior work are",
                  "Reference", "", "1. References and notes on the method"] {
            assert!(!is_refs_heading(s), "should NOT match: {s}");
        }
    }

    /// The section-scope branch, and the ABSTENTION that makes it safe.
    #[test]
    fn after_refs_branch() {
        let f = |after_refs| Feat {
            run_chars: 1600.0, w_p90: 0.98, y_rel: 0.5, nn_gap: 0.9, conf: 0.99,
            run_lines: 40, same_left: 34, year_paren: false, after_refs,
            page_year_hits: 0,
        };
        // Body-shaped in every geometric respect — the point of the branch.
        assert!(is_non_body(&f(true)));
        assert!(!is_non_body(&f(false)));
    }

    /// 2D grouping (§8.136): a caption joins its plot, and two columns do not
    /// join each other. This is what `run_stats_per_line` cannot do.
    #[test]
    fn blocks_group_in_2d() {
        // One column of five stacked lines -> one block.
        let col: Vec<(f32, f32, f32, f32)> =
            (0..5).map(|i| (50.0, i as f32 * 20.0, 300.0, 14.0)).collect();
        let idx: Vec<usize> = (0..5).collect();
        assert_eq!(group_blocks(&idx, &col, 14.0).len(), 1);

        // Two side-by-side columns that never overlap horizontally -> two blocks.
        let mut two = col.clone();
        two.extend((0..5).map(|i| (500.0, i as f32 * 20.0, 300.0, 14.0)));
        let idx2: Vec<usize> = (0..10).collect();
        assert_eq!(group_blocks(&idx2, &two, 14.0).len(), 2);

        // A caption 1.5 line heights under a plot label, overlapping it
        // horizontally, JOINS it — the case left-edge grouping gets wrong,
        // because the two share no left edge.
        let fig = vec![
            (100.0, 0.0, 400.0, 14.0),   // a chart title
            (260.0, 21.0, 80.0, 14.0),   // an axis label under it, indented
        ];
        assert_eq!(group_blocks(&[0, 1], &fig, 14.0).len(), 1);

        // The same pair pushed far apart vertically stays separate.
        let far = vec![(100.0, 0.0, 400.0, 14.0), (260.0, 300.0, 80.0, 14.0)];
        assert_eq!(group_blocks(&[0, 1], &far, 14.0).len(), 2);
    }

    /// The three fitted block rules, each at a value the fit chose (§8.135).
    #[test]
    fn block_branch_rules() {
        let base = BlockFeat {
            area_frac: 0.05, aspect: 2.0, isolation: 0.5, blk_h: 0.10,
            blk_w: 0.60, w_cv: 0.30, w_med: 0.95, conf: 0.99,
        };
        // A body paragraph: large, not isolated, full-column width.
        assert!(!block_is_non_body(&base));
        // A: an isolated strip six times wider than tall.
        assert!(block_is_non_body(&BlockFeat {
            area_frac: 0.002, isolation: 2.0, aspect: 8.0, ..base
        }));
        // ...but not if it sits inside a column.
        assert!(!block_is_non_body(&BlockFeat {
            area_frac: 0.002, isolation: 0.5, aspect: 8.0, ..base
        }));
        // B: a thin isolated band whose lines agree on width.
        assert!(block_is_non_body(&BlockFeat {
            blk_h: 0.010, isolation: 2.0, w_cv: 0.05, ..base
        }));
        // C: narrow and low-confidence.
        assert!(block_is_non_body(&BlockFeat {
            w_med: 0.20, blk_w: 0.30, conf: 0.90, ..base
        }));
        // ...and confidence alone is not enough on a full-width block.
        assert!(!block_is_non_body(&BlockFeat { conf: 0.90, ..base }));
    }

    /// `cv` is the POPULATION form — a block IS its lines, not a sample of them.
    #[test]
    fn cv_is_population_form() {
        assert_eq!(cv(&[]), 0.0);
        assert_eq!(cv(&[5.0]), 0.0);
        assert_eq!(cv(&[3.0, 3.0, 3.0]), 0.0);
        // mean 2, population variance 1, so cv = 1/2.
        assert!((cv(&[1.0, 3.0]) - 0.5).abs() < 1e-6);
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

    /// THE COLUMN CONDITION IS LOAD-BEARING, NOT SCOPING (§8.157).
    ///
    /// `omni-0063` is a tall, dense, high-body-fraction TWO-column page that the
    /// shipped order reads at 5.8 %. It satisfies every other clause of the
    /// guard. Drop `n_col >= 3` and the probe fires on it and takes it to
    /// 73.4 % — one page, and the corpus result flips from +0.228 pp with a CI
    /// excluding zero to -0.053 pp with a CI spanning it.
    #[test]
    fn probe_guard_requires_three_columns() {
        let omni_0063 = ProbeStats { n_col: 2, cover: 0.564, aspect: 2.0, n_all: 100 };
        assert!(!probe_gate_fires(&omni_0063, 90), "must NOT fire on a 2-column page");
        let three = ProbeStats { n_col: 3, ..omni_0063 };
        assert!(probe_gate_fires(&three, 90), "must fire once there are 3 columns");
    }

    /// Every other clause must also be able to veto on its own.
    #[test]
    fn probe_guard_clauses_all_bite() {
        let ok = ProbeStats { n_col: 3, cover: 0.30, aspect: 1.8, n_all: 100 };
        assert!(probe_gate_fires(&ok, 90));
        assert!(!probe_gate_fires(&ProbeStats { cover: 0.17, ..ok }, 90), "sparse page");
        assert!(!probe_gate_fires(&ProbeStats { aspect: 1.5, ..ok }, 90), "not tall enough");
        assert!(!probe_gate_fires(&ok, 80), "body_frac 0.80 is under the 0.85 bar");
    }

    /// The reorder must be a PERMUTATION. A stage that silently drops a line
    /// would read as an ordering win while deleting output — the failure mode
    /// §8.153's harness hit twice.
    #[test]
    fn probe_reorder_keeps_every_line() {
        let mk = |x: f32, y: f32, t: &str| OcrLine {
            text: t.into(),
            words: Vec::new(),
            bbox: Some(ffai_core::types::BoundingBox { x, y, width: 200.0, height: 12.0 }),
            confidence: Some(0.99),
        };
        // Two columns of five lines each, on a tall page.
        let mut lines = Vec::new();
        for i in 0..5 {
            lines.push(mk(40.0, 100.0 + i as f32 * 20.0, "left"));
            lines.push(mk(560.0, 100.0 + i as f32 * 20.0, "right"));
        }
        let n = lines.len();
        let st = ProbeStats { n_col: 3, cover: 0.30, aspect: 1.8, n_all: n };
        let out = probe_apply(lines.clone(), &st, VERIFY_LOWCONF_CRNN, 800.0, 1600.0);
        assert_eq!(out.len(), n, "reorder dropped or duplicated a line");
        let mut got: Vec<&str> = out.iter().map(|l| l.text.as_str()).collect();
        got.sort_unstable();
        assert_eq!(got.iter().filter(|t| **t == "left").count(), 5);
        assert_eq!(got.iter().filter(|t| **t == "right").count(), 5);
    }

    /// The §8.160 verifier's parts, pinned on constructed text.
    #[test]
    fn join_fluency_prefers_the_flowing_order() {
        // Correct read: hyphen completes, sentence continues.
        let good = ["the leg was in an elong-", "ated position and the", "subject could rest."];
        // Interleaved read: same lines, joins broken.
        let bad = ["the leg was in an elong-", "subject could rest.", "ated position and the"];
        let g: Vec<&str> = good.to_vec();
        let b: Vec<&str> = bad.to_vec();
        assert!(join_fluency(&g) > join_fluency(&b));
    }

    #[test]
    fn num_seq_prefers_ascending_items() {
        let asc = ["55. first item?", "56. second item?", "57. third item?", "58. fourth?"];
        let col = ["55. first item?", "57. third item?", "56. second item?", "58. fourth?"];
        let a: Vec<&str> = asc.to_vec();
        let c: Vec<&str> = col.to_vec();
        assert!(num_seq_monotone(&a) > num_seq_monotone(&c));
        // fewer than 3 numbers = no evidence, not a preference
        let few: Vec<&str> = vec!["1. one?", "no number", "also none"];
        assert_eq!(num_seq_monotone(&few), 0.5);
    }

    /// The abstain: identical geometry and text, only CONFIDENCE differs — the
    /// verifier reorders the confident page and must leave the garbage one
    /// alone. Pins the §8.160 competence gate (`omni-0080`, 85.3 % low-conf).
    #[test]
    fn verifier_abstains_on_unreliable_text() {
        let mk = |x: f32, y: f32, t: &str, conf: f32| OcrLine {
            text: t.into(),
            words: Vec::new(),
            bbox: Some(ffai_core::types::BoundingBox { x, y, width: 200.0, height: 12.0 }),
            confidence: Some(conf),
        };
        // Three columns; shipped (input) order interleaves them so the probe's
        // column-major read strictly improves the joins.
        let build = |conf: f32| {
            // Column starts are CAPITALISED, so the row-major interleave
            // breaks every hyphen join (uppercase after "-" scores -1) while
            // the column-major read completes them (+2). Uniformly lowercase
            // text ties the margins and the verifier correctly refuses.
            let texts = [["Alpha beta gam-", "ma delta and the", "value keeps going"],
                         ["Second column al-", "so continues in a", "flowing manner too"],
                         ["Third column runs-", "on with more of", "the same body text"]];
            let mut v = Vec::new();
            for row in 0..3 {
                for col in 0..3 {
                    // pitch 14 on height-12 lines: no horizontal band gap, so
                    // the cut goes vertical and reads column-major.
                    v.push(mk(40.0 + col as f32 * 270.0, 100.0 + row as f32 * 14.0,
                              texts[col][row], conf));
                }
            }
            v
        };
        let st = ProbeStats { n_col: 3, cover: 0.30, aspect: 1.2, n_all: 9 };
        let confident = probe_apply(build(0.99), &st, VERIFY_LOWCONF_CRNN, 850.0, 1020.0);
        let garbage = probe_apply(build(0.50), &st, VERIFY_LOWCONF_CRNN, 850.0, 1020.0);
        let ys = |v: &[OcrLine]| v.iter().map(|l| l.bbox.as_ref().unwrap().y as i32).collect::<Vec<_>>();
        let input = vec![100, 100, 100, 114, 114, 114, 128, 128, 128];
        assert_ne!(ys(&confident), input,
                   "confident text: verifier must accept the column-major reorder");
        assert_eq!(ys(&garbage), input,
                   "unreliable text: verifier must abstain and leave the order alone");
    }

    /// Ground 2 must NOT accept when the text is indifferent: a guard-failing
    /// page whose margins are all zero passes through byte-for-byte.
    #[test]
    fn verifier_rejects_when_text_is_indifferent() {
        let mk = |x: f32, y: f32| OcrLine {
            text: "self contained.".into(), // terminal '.' -> every join scores 0
            words: Vec::new(),
            bbox: Some(ffai_core::types::BoundingBox { x, y, width: 150.0, height: 12.0 }),
            confidence: Some(0.99),
        };
        let mut lines = Vec::new();
        for i in 0..4 {
            for c in 0..3 {
                lines.push(mk(40.0 + c as f32 * 260.0, 100.0 + i as f32 * 30.0));
            }
        }
        let first_y = lines[0].bbox.as_ref().unwrap().y;
        // aspect 1.2 fails the §8.157 guard; verifier must then refuse too.
        let st = ProbeStats { n_col: 3, cover: 0.30, aspect: 1.2, n_all: lines.len() };
        let out = probe_apply(lines, &st, VERIFY_LOWCONF_CRNN, 800.0, 960.0);
        assert_eq!(out[0].bbox.as_ref().unwrap().y, first_y, "order must be untouched");
    }

    /// The Great Gate tap must not change what the engine returns (plan §11 W1).
    /// A telemetry tap that perturbs output would corrupt the very harvest it
    /// exists to produce, and every gate fitted on it after. Exercises BOTH the
    /// tapped and untapped paths through `probe_decide` on the same input and
    /// asserts the orders are identical — including a page that returns early,
    /// since the tap deliberately fires on those too.
    #[test]
    fn harvest_tap_does_not_change_output() {
        let mk = |x: f32, y: f32, t: &str| OcrLine {
            text: t.into(),
            words: Vec::new(),
            bbox: Some(ffai_core::types::BoundingBox { x, y, width: 180.0, height: 12.0 }),
            confidence: Some(0.97),
        };
        for (st, lines) in [
            // reaches the decision
            (ProbeStats { n_col: 3, cover: 0.30, aspect: 1.8, n_all: 9 },
             (0..9).map(|i| mk((i % 3) as f32 * 260.0, (i / 3) as f32 * 14.0 + 100.0, "item text"))
                 .collect::<Vec<_>>()),
            // returns early on the guard — the tap must still be inert here
            (ProbeStats { n_col: 1, cover: 0.05, aspect: 1.0, n_all: 3 },
             vec![mk(0.0, 300.0, "a"), mk(0.0, 100.0, "b"), mk(0.0, 200.0, "c")]),
        ] {
            let plain = probe_decide(lines.clone(), &st, VERIFY_LOWCONF_CRNN, 800.0, 1600.0, None);
            let mut row = crate::harvest::Row::new();
            let tapped =
                probe_decide(lines, &st, VERIFY_LOWCONF_CRNN, 800.0, 1600.0, Some(&mut row));
            let ys = |v: &[OcrLine]| {
                v.iter().map(|l| (l.bbox.as_ref().unwrap().x as i32,
                                  l.bbox.as_ref().unwrap().y as i32)).collect::<Vec<_>>()
            };
            assert_eq!(ys(&plain), ys(&tapped), "the harvest tap changed the ordering");
            assert_eq!(plain.len(), tapped.len(), "the harvest tap changed the line count");
        }
    }

    /// §15: `join_fluency` is STRUCTURALLY INERT on CJK, measured not argued.
    ///
    /// Its terminator set is Latin punctuation and it scores on letter CASE.
    /// Chinese uses `。！？：；` and CJK characters have no case, so neither the
    /// lowercase nor the uppercase branch can fire and the score is ~0 for every
    /// candidate ordering. The §8.160 verifier — our largest ordering win — can
    /// therefore never rank two orders on a Chinese page, on 54 % of the
    /// benchmark. This test pins that so the CJK fix has a failing baseline, and
    /// so nobody "simplifies" the Latin path later without noticing it was
    /// already blind here.
    ///
    /// The English control in the same test is what makes it evidence: the same
    /// function separates a good order from a bad one when the text is Latin.
    #[test]
    fn join_fluency_separates_cjk_orders() {
        // The BASELINE this replaces: `join_fluency_latin` returns ~0 on both of
        // these, measured before the fix — its terminator set is Latin and it
        // scores on letter case, which CJK does not have. The verifier could
        // therefore never rank two orders on 54 % of the benchmark (§15).
        let good = ["本文提出了一种新的方法", "用于解决图像识别中的", "长期存在的问题。"];
        let bad = ["长期存在的问题。", "本文提出了一种新的方法", "用于解决图像识别中的"];
        let (lg, lb) = (join_fluency_latin(&good.to_vec()), join_fluency_latin(&bad.to_vec()));
        assert!((lg - lb).abs() < 1e-6,
                "the Latin arm is supposed to be blind here — that is the bug: {lg} vs {lb}");

        // THE FIX: routed by script, the flowing order now scores strictly higher.
        let (fg, fb) = (join_fluency(&good.to_vec()), join_fluency(&bad.to_vec()));
        assert!(fg > fb, "CJK arm must prefer the flowing order: {fg} vs {fb}");

        // CONTROL: Latin text must still take the Latin arm and still separate.
        let eg = ["this paper proposes a new", "method for solving the", "long-standing problem."];
        let eb = ["long-standing problem.", "this paper proposes a new", "method for solving the"];
        let (eg_s, eb_s) = (join_fluency(&eg.to_vec()), join_fluency(&eb.to_vec()));
        assert!(eg_s > eb_s, "Latin routing regressed: {eg_s} vs {eb_s}");
    }

    /// The script router must not hijack Latin pages that merely quote CJK, and
    /// must not fall back to Latin on CJK pages carrying Latin numerals or
    /// citations — both are common in academic text and getting either wrong
    /// silently sends a page down the arm that cannot read it.
    #[test]
    fn script_router_picks_the_right_arm() {
        let latin_with_cjk_quote = ["the term (中文) appears here", "and the sentence continues"];
        assert!(!mostly_cjk(&latin_with_cjk_quote.to_vec()),
                "a Latin line quoting CJK must stay on the Latin arm");

        let cjk_with_numerals = ["本文提出了一种新的方法 2024", "用于解决图像识别问题"];
        assert!(mostly_cjk(&cjk_with_numerals.to_vec()),
                "CJK with Latin numerals must still take the CJK arm");

        // Too little evidence to route on: stay with the Latin default.
        assert!(!mostly_cjk(&["中文".to_string().as_str()].to_vec()),
                "a two-character fragment is not enough evidence to switch arms");
    }

    /// EVERY harvest row carries EVERY column, whichever path the page takes.
    ///
    /// The header is written from whichever row lands first, so a single
    /// early-returning page at the head of a run would misalign every full row
    /// after it and the calculator would fit rules to columns shifted under
    /// their names — a silent, total corruption of the harvest. Found by
    /// inspection after a three-page smoke test happened to hit only the full
    /// path; this is the test that would have caught it.
    #[test]
    fn harvest_row_width_is_constant() {
        let mk = |x: f32, y: f32| OcrLine {
            text: "item 1 text".into(),
            words: Vec::new(),
            bbox: Some(ffai_core::types::BoundingBox { x, y, width: 180.0, height: 12.0 }),
            confidence: Some(0.97),
        };
        let dense: Vec<OcrLine> =
            (0..9).map(|i| mk((i % 3) as f32 * 260.0, (i / 3) as f32 * 14.0 + 100.0)).collect();
        let mut widths = Vec::new();
        for (st, lines) in [
            // full path: reaches the decision
            (ProbeStats { n_col: 3, cover: 0.30, aspect: 1.8, n_all: 9 }, dense.clone()),
            // early exit on the column guard
            (ProbeStats { n_col: 1, cover: 0.30, aspect: 1.8, n_all: 9 }, dense.clone()),
            // early exit on the coverage guard
            (ProbeStats { n_col: 3, cover: 0.01, aspect: 1.8, n_all: 9 }, dense.clone()),
            // early exit on too few lines
            (ProbeStats { n_col: 3, cover: 0.30, aspect: 1.8, n_all: 2 },
             vec![mk(0.0, 100.0), mk(0.0, 120.0)]),
        ] {
            let mut row = crate::harvest::Row::new();
            let _ = probe_decide(lines, &st, VERIFY_LOWCONF_CRNN, 800.0, 1600.0, Some(&mut row));
            widths.push(row.len());
        }
        assert!(widths.windows(2).all(|w| w[0] == w[1]),
                "harvest rows differ in width across paths: {widths:?} — the CSV would be ragged");
    }

    /// A page the guard rejects passes through byte-for-byte, whatever the
    /// geometry looks like — the reorder must be inert outside its population.
    #[test]
    fn probe_reorder_inert_when_guard_rejects() {
        let mk = |y: f32| OcrLine {
            text: "x".into(),
            words: Vec::new(),
            bbox: Some(ffai_core::types::BoundingBox { x: 0.0, y, width: 100.0, height: 10.0 }),
            confidence: Some(0.99),
        };
        let lines = vec![mk(300.0), mk(100.0), mk(200.0)];
        let two_col = ProbeStats { n_col: 2, cover: 0.30, aspect: 1.8, n_all: 3 };
        let out = probe_apply(lines, &two_col, VERIFY_LOWCONF_CRNN, 800.0, 1600.0);
        assert_eq!(out[0].bbox.as_ref().unwrap().y, 300.0, "order must be untouched");
    }
}
