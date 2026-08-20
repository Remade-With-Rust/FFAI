//! Region routing — the layer that turns three working models into a PIPELINE.
//!
//! §47 established that layout, table and formula all execute correctly and
//! match `onnxruntime` token-for-token. None of that reached the benchmark,
//! because nothing decided which model a region belongs to: the engine ran
//! detection and recognition over the whole page and the other three modules
//! were reachable only from their probe binaries.
//!
//! This is the missing decision. Layout classifies the page into regions; a
//! `table` region goes to SLANet, a `formula` region to PP-FormulaNet, and
//! everything else stays on the detect+recognise path that §33–§43 tuned.
//!
//! ## Why it merges instead of replacing
//!
//! The routing runs AFTER the existing pipeline has produced its lines, and
//! splices into that sequence rather than rebuilding it. Reading order was the
//! single hardest thing this campaign earned (§29–§43, most of it refutations),
//! and a router that re-sorts the page from scratch would silently discard it.
//! A region's output is therefore inserted at the position of the FIRST line it
//! absorbs, so the surviving text keeps exactly the order it already had.
//!
//! ## Off by default
//!
//! `FFAI_ROUTE=1` opts in. Standing law from §8.106 and the revert-if-unproven
//! rule: a change that deletes or rewrites output must be ASKED for until a
//! gate says it pays. It also means the banked 0.1278/0.2336 baseline stays
//! reproducible from the same binary.

use crate::doclayout::{DocLayout, Region};
use crate::formula::FormulaModel;
use crate::table::TableModel;
use ffai_core::error::Result;
use ffai_core::types::{BoundingBox, ImageBuffer, OcrLine};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Layout score floor. PicoDet emits a score per class per anchor; below this a
/// region is not confident enough to REDIRECT a page away from the text path,
/// which is the asymmetric risk here — a missed table still gets read as text,
/// a false table replaces real text with an empty grid.
fn score_thr() -> f32 {
    env_f32("FFAI_LAYOUT_SCORE", 0.45)
}
fn iou_thr() -> f32 {
    env_f32("FFAI_LAYOUT_IOU", 0.40)
}
/// A text line counts as belonging to a region when this much of it is inside.
/// Centre-only containment mis-assigns a line that straddles a table edge.
fn cover_thr() -> f32 {
    env_f32("FFAI_ROUTE_COVER", 0.60)
}

/// GUARD 2 — the bar to REDIRECT is higher than the bar to DETECT.
///
/// Deciding a region exists and deciding to overwrite a page's text with a
/// model's rendering of it are different claims on the same score. Measured on
/// the 305-page subset, routing everything layout would report cost text
/// **+0.0347** (CI excluding zero, 50 pages hurt) and the single worst case was
/// a Chinese newspaper whose column grid scored as a `table`: the region
/// absorbed the page, SLANet returned a near-empty grid, and 6970 bytes of
/// correctly-read prose became 2041 bytes of `<td></td>` — 0.0144 to 1.0000.
fn route_thr() -> f32 {
    env_f32("FFAI_ROUTE_SCORE", 0.60)
}

/// GUARD 1 — refuse a rendering that carries LESS TEXT than it replaces.
///
/// `route.rs` already fell back when a model ERRORED. It had no check for a
/// model that succeeded BADLY, which is the case that actually costs: an empty
/// table skeleton and a truncated equation are both well-formed output. The
/// asymmetry this module's own header states — "a missed table still gets read
/// as text, a false table replaces real text with an empty grid" — is only
/// enforced if something counts the characters.
///
/// Compared on CONTENT, not markup: cell text for a table, the LaTeX body for a
/// formula. LaTeX is normally LONGER than the plain reading of the same
/// equation (`\frac{1}{2}` against `1/2`), so a short one is already suspect.
fn retain(kept: usize, had: usize) -> bool {
    // nothing was there to lose — an isolated equation the text detector
    // missed entirely is exactly the §40 case this stage exists for
    if had == 0 {
        return true;
    }
    kept as f32 >= env_f32("FFAI_ROUTE_RETAIN", 0.60) * had as f32
}

/// GUARD 3 (§51) — a region that absorbs most of the PAGE is not a table.
///
/// The §50 hit list showed the failure the retain guard cannot see: a false
/// `table` over a newspaper column grid or a textbook TOC absorbs nearly every
/// line, the cells re-read fine (so `retain` passes), and the whole page's
/// text exits the evaluator's text pool as markup — 0.01 → 0.90 and
/// 0.24 → 1.00. A true table is a MINORITY of a page's characters. Swept
/// offline on the 457 routed pages: at 0.60 the cap reverts 21 pages — the two
/// catastrophes plus whole-page-table slides that score identically either
/// way — and costs zero order win; tighter caps add churn and recover nothing.
fn absorb(had: usize, page_total: usize) -> bool {
    page_total == 0 || (had as f32) <= env_f32("FFAI_ROUTE_ABSORB", 0.60) * page_total as f32
}

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

pub fn enabled() -> bool {
    std::env::var("FFAI_ROUTE").map(|v| v != "0").unwrap_or(false)
}

fn fixtures() -> PathBuf {
    std::env::var("FFAI_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new("corpora/refs/fixtures").to_path_buf())
}

/// The three ONNX-backed stages, loaded once. Kept separate from the engine's
/// `Models` because these are lazy in a different sense: a corpus with no
/// tables never pays SLANet's load, and the router is opt-in besides.
pub struct Router {
    pub layout: DocLayout,
    table: OnceLock<Option<TableModel>>,
    formula: OnceLock<Option<FormulaModel>>,
    dir: PathBuf,
}

impl Router {
    pub fn new() -> Result<Self> {
        let dir = fixtures();
        let layout = DocLayout::new(
            &dir.join("doclayout_s_arch.json"),
            &dir.join("doclayout_s.safetensors"),
        )?;
        Ok(Self { layout, table: OnceLock::new(), formula: OnceLock::new(), dir })
    }

    fn table(&self) -> Option<&TableModel> {
        self.table
            .get_or_init(|| {
                TableModel::new(
                    &self.dir.join("slanet_plus_arch.json"),
                    &self.dir.join("slanet_plus.safetensors"),
                )
                .ok()
            })
            .as_ref()
    }

    fn formula(&self) -> Option<&FormulaModel> {
        self.formula
            .get_or_init(|| {
                FormulaModel::new(
                    &self.dir.join("formulanet_arch.json"),
                    &self.dir.join("formulanet.safetensors"),
                    &self.dir.join("formulanet_vocab.json"),
                )
                .ok()
            })
            .as_ref()
    }
}

/// Reads one page-coordinate box with whatever recognizer the engine is
/// configured with. Supplied by `engine.rs` because that is the only place that
/// knows which one ran — the same reason `VERIFY_LOWCONF` is passed in (§8.171).
pub type RecFn<'a> = &'a dyn Fn(usize, usize, usize, usize) -> Option<String>;

fn crop(img: &ImageBuffer, x0: usize, y0: usize, x1: usize, y1: usize) -> ImageBuffer {
    let bpp = img.format.bytes_per_pixel();
    let (w, h) = (img.width as usize, img.height as usize);
    let (x0, y0) = (x0.min(w.saturating_sub(1)), y0.min(h.saturating_sub(1)));
    let (x1, y1) = (x1.min(w).max(x0 + 1), y1.min(h).max(y0 + 1));
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut data = vec![0u8; cw * ch * bpp];
    for y in 0..ch {
        let s = ((y0 + y) * w + x0) * bpp;
        data[y * cw * bpp..(y + 1) * cw * bpp].copy_from_slice(&img.data[s..s + cw * bpp]);
    }
    ImageBuffer { width: cw as u32, height: ch as u32, format: img.format, data }
}

/// Fraction of `l` that lies inside `r`.
fn covered(l: &BoundingBox, r: &Region) -> f32 {
    let (lx1, ly1) = (l.x + l.width, l.y + l.height);
    let ix = (lx1.min(r.x1) - l.x.max(r.x0)).max(0.0);
    let iy = (ly1.min(r.y1) - l.y.max(r.y0)).max(0.0);
    let a = (l.width * l.height).max(1e-6);
    ix * iy / a
}

/// Route one page: replace the lines inside `table`/`formula` regions with the
/// output of the model that owns them, in place.
pub fn apply(
    r: &Router,
    img: &ImageBuffer,
    lines: Vec<OcrLine>,
    rec: RecFn<'_>,
) -> Result<Vec<OcrLine>> {
    let regions = r.layout.detect(img, score_thr(), iou_thr())?;
    let dbg = std::env::var("FFAI_ROUTE_DEBUG").is_ok();
    if dbg {
        eprintln!("route: {} regions", regions.len());
        for g in &regions {
            eprintln!(
                "   {:<16} {:.2}  [{:.0},{:.0},{:.0},{:.0}]",
                g.label(), g.score, g.x0, g.y0, g.x1, g.y1
            );
        }
    }

    // Only the two classes we have a model for redirect anything. Every other
    // label — paragraph_title, figure, header — is still TEXT to us, and
    // pretending otherwise would delete content we currently read correctly.
    let targets: Vec<&Region> = regions
        .iter()
        .filter(|g| (g.is_table() || g.routes_to_latex()) && g.score >= route_thr())
        .collect();
    if targets.is_empty() {
        return Ok(lines);
    }

    // Which region, if any, absorbs each line. A line can only be claimed once;
    // the region covering the most of it wins, so nested boxes (a formula
    // number inside a formula band) do not double-claim.
    let mut owner: Vec<Option<usize>> = vec![None; lines.len()];
    for (li, l) in lines.iter().enumerate() {
        let Some(bb) = l.bbox.as_ref() else { continue };
        let mut best = (cover_thr(), usize::MAX);
        for (ti, t) in targets.iter().enumerate() {
            let c = covered(bb, t);
            if c > best.0 {
                best = (c, ti);
            }
        }
        if best.1 != usize::MAX {
            owner[li] = Some(best.1);
        }
    }

    // Region output, computed once per region.
    let page_chars: usize = lines.iter().map(|l| l.text.chars().count()).sum();
    let mut rendered: Vec<Option<String>> = Vec::with_capacity(targets.len());
    for (ti, t) in targets.iter().enumerate() {
        // The absorbed lines, in the order the pipeline already put them —
        // this is what fills table cells and what a failed model falls back to.
        let inside: Vec<&OcrLine> = lines
            .iter()
            .enumerate()
            .filter(|(li, _)| owner[*li] == Some(ti))
            .map(|(_, l)| l)
            .collect();
        let c = crop(img, t.x0 as usize, t.y0 as usize, t.x1 as usize, t.y1 as usize);
        // What we would DESTROY by accepting this region's rendering.
        let had: usize = inside.iter().map(|l| l.text.chars().count()).sum();
        let out = if !absorb(had, page_chars) {
            if dbg {
                eprintln!("   -> {} {}: ABSORB-CAPPED ({had} of {page_chars} page chars)",
                    t.label(), ti);
            }
            None
        } else if t.is_table() {
            r.table().and_then(|m| m.recognize(&c).ok()).and_then(|s| {
                // Cell text comes from reading each predicted cell; the
                // structure model recovers the GRID, the recognizer supplies
                // the CONTENT. Neither can do the other's job — SLANet emits
                // no characters at all.
                let texts = bind_cells(&s, &inside, t, rec);
                let kept: usize = texts.iter().map(|c| c.chars().count()).sum();
                retain(kept, had).then(|| s.to_html(&texts))
            })
        } else {
            r.formula().and_then(|m| m.recognize(&c).ok()).and_then(|s| {
                let s = s.trim().to_string();
                // An empty or degenerate decode must NOT delete the text we
                // already had; falling through to `None` restores it below.
                if s.is_empty() || !retain(s.chars().count(), had) {
                    None
                } else {
                    Some(format!("$${s}$$"))
                }
            })
        };
        if dbg {
            eprintln!(
                "   -> {} {}: {} lines absorbed, {}",
                t.label(),
                ti,
                inside.len(),
                out.as_deref().map(|s| format!("{} chars", s.len()))
                    .unwrap_or_else(|| format!("REJECTED, kept {had} chars of text"))
            );
        }
        rendered.push(out);
    }

    // Splice: each region's output takes the slot of the FIRST line it absorbed,
    // and the rest of that region's lines drop out. Lines no region claimed keep
    // their position and their order exactly.
    let mut out: Vec<OcrLine> = Vec::with_capacity(lines.len());
    let mut emitted = vec![false; targets.len()];
    for (li, l) in lines.into_iter().enumerate() {
        match owner[li] {
            None => out.push(l),
            Some(ti) => match &rendered[ti] {
                // model failed: keep the original text rather than lose it
                None => out.push(l),
                Some(text) => {
                    if !emitted[ti] {
                        emitted[ti] = true;
                        let t = targets[ti];
                        out.push(OcrLine {
                            text: text.clone(),
                            words: Vec::new(),
                            bbox: Some(BoundingBox {
                                x: t.x0,
                                y: t.y0,
                                width: t.x1 - t.x0,
                                height: t.y1 - t.y0,
                            }),
                            confidence: Some(t.score),
                        });
                    }
                }
            },
        }
    }
    // A region that absorbed NOTHING still has content — an isolated equation
    // the text detector missed entirely is exactly the §40 case this whole
    // stage exists for. Append those by vertical position.
    for (ti, t) in targets.iter().enumerate() {
        if emitted[ti] {
            continue;
        }
        let Some(text) = &rendered[ti] else { continue };
        let at = out
            .iter()
            .position(|l| l.bbox.as_ref().is_some_and(|b| b.y > t.y0))
            .unwrap_or(out.len());
        out.insert(
            at,
            OcrLine {
                text: text.clone(),
                words: Vec::new(),
                bbox: Some(BoundingBox {
                    x: t.x0,
                    y: t.y0,
                    width: t.x1 - t.x0,
                    height: t.y1 - t.y0,
                }),
                confidence: Some(t.score),
            },
        );
    }
    Ok(out)
}

/// Assign each recognised line to the predicted cell it sits in, and return one
/// string per cell in cell order.
///
/// Cell boxes arrive in CROP coordinates, so the region origin is added back
/// before comparing against page-coordinate line boxes — the class of mistake
/// that makes a table come out empty while every part in isolation looks right.
fn bind_cells(
    s: &crate::table::TableStructure,
    inside: &[&OcrLine],
    t: &Region,
    rec: RecFn<'_>,
) -> Vec<String> {
    // READ EACH CELL, do not redistribute lines into cells.
    //
    // The two box sets do not correspond and cannot be made to. The detector
    // emits TEXT LINES and SLANet emits CELLS: one detected line routinely
    // spans a whole table row, and a narrow column of figures comes back as a
    // single tall box. Assigning lines to cells by proximity produced exactly
    // that corruption on the RWKV table — the header row landed empty and
    // "2048 2560 4096 6144", four separate rows of the table, was deposited in
    // one cell. No distance metric fixes that, because the information the
    // binding needs was destroyed upstream when four rows became one box.
    //
    // The cell box is itself a crop rectangle, so the fix is to read it.
    let mut texts = Vec::with_capacity(s.cells.len());
    for c in &s.cells {
        // Predicted cell boxes hug the text; the recognizers were tuned on
        // crops with a little context (`PAD_X`/`PAD_Y` exist for the same
        // reason). Scaled by cell HEIGHT so a tall cell and a short one get
        // proportionate context.
        let ch = (c.y1 - c.y0).max(1.0);
        let (px, py) = (ch * env_f32("FFAI_CELL_PAD_X", 0.0), ch * env_f32("FFAI_CELL_PAD_Y", 0.0));
        let (x0, y0) = (
            (c.x0 + t.x0 - px).max(0.0) as usize,
            (c.y0 + t.y0 - py).max(0.0) as usize,
        );
        let (x1, y1) = (
            (c.x1 + t.x0 + px).max(0.0) as usize,
            (c.y1 + t.y0 + py).max(0.0) as usize,
        );
        // A degenerate or sub-glyph cell is genuinely empty, not worth a crop.
        let ok = x1 > x0 + 2 && y1 > y0 + 2;
        texts.push(if ok { rec(x0, y0, x1, y1).unwrap_or_default() } else { String::new() });
    }
    // If per-cell reading produced nothing at all — no recognizer available on
    // this path, or every crop degenerate — fall back to the lines we already
    // have rather than emit an empty grid, which would DELETE content the
    // pipeline read correctly.
    if texts.iter().all(|s| s.trim().is_empty()) && !inside.is_empty() {
        return vec![inside.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ")]
            .into_iter()
            .chain(std::iter::repeat(String::new()))
            .take(s.cells.len().max(1))
            .collect();
    }
    texts
}
