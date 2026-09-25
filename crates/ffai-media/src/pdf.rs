//! PDF pages as a viewer shows them — for OCR of scanned documents
//! (docs/plans/commercial-gaps.md, gaps 1, 2 and 11).
//!
//! A scanned PDF page is an image `XObject` with things drawn on top of it.
//! Extracting the image alone is **unsafe**: a published filing hid a home
//! address under a black patch, a viewer showed the patch, and reading the
//! extracted scan returned the address at high confidence, because the patch
//! was a separate object drawn over the scan (gap 11). So this module never
//! returns a raw scan. It returns the scan **composited with everything drawn
//! after it**:
//!
//! * filled paths (a black box, a white-out), painted in their fill colour
//!   over the bounding box of the path — over-painting a curved shape's
//!   corners is the fail-safe direction;
//! * images drawn after the scan (a patch pasted in a later save), painted
//!   black — an overlay may hide a stamp, it must never reveal what it covers;
//! * form `XObjects` drawn after the scan, walked recursively;
//! * `Redact`, `Square`, `Circle`, `Polygon` and `Stamp` annotations, painted
//!   black over their rectangle.
//!
//! A page whose drawing cannot be walked returns **no image** and a note — the
//! caller cannot honour redactions it cannot see, so it is not handed pixels.
//!
//! The composited image is then turned upright the way a viewer shows it,
//! from the image's placement matrix and the page's `/Rotate`.
//!
//! Decoders: DCT (JPEG, via `rusty_jpeg`), CCITT Group 3 one-dimensional and
//! Group 4 (via the pure-Rust `fax` crate), and Flate/LZW/raw samples at 1 or
//! 8 bits in gray or RGB, including 1-bit stencil masks. JBIG2 and JPEG 2000
//! are reported per page as unsupported rather than guessed at.
//!
//! Every decoded image is bounded by [`MAX_PIXELS`] before anything is
//! allocated for it, and decompression is capped at the size the declared
//! dimensions justify, so a hostile PDF cannot drive an unbounded allocation.
//!
//! Numbers read from the file (dimensions, component counts, parameters) go
//! through checked conversions. The casts that remain carry their own
//! `allow` and the reason the value is already bounded.

// Page geometry is f64 affine arithmetic on coordinates a PDF writer chose. A
// fused multiply-add rounds differently from every other reader of the same
// file, and nothing here needs the extra precision.
#![allow(clippy::suboptimal_flops)]

use lopdf::{Dictionary, Document, Object, ObjectId};

use ffai_core::error::{Error, Result};
use ffai_core::types::{ImageBuffer, PixelFormat};

/// Largest image this module will decode, in pixels: 100 MP, a 600 ppi
/// tabloid page with room to spare.
pub const MAX_PIXELS: usize = 100_000_000;

/// Deepest form `XObject` nesting walked before a page is refused.
const MAX_FORM_DEPTH: usize = 8;

/// One page of a PDF.
#[derive(Debug, Clone)]
pub struct PdfPage {
    /// Zero-based page index.
    pub index: usize,
    /// The page's content draws text (a born-digital page, or a scan with an
    /// OCR text layer). A caller may prefer the text layer for such a page.
    pub has_text_layer: bool,
    /// The page's scan as a viewer shows it: overlays and redactions painted
    /// in, turned upright. `None` when the page has no image, or when it has
    /// one that could not be composited safely — see `note`.
    pub image: Option<ImageBuffer>,
    /// Why `image` is `None`, or what was approximated.
    pub note: Option<String>,
    /// The page's text layer, when it has one (`lopdf`'s extractor: text in
    /// content-stream order, no positions). **Read `redaction_risk` before
    /// using it.**
    pub text: Option<String>,
    /// The text layer may contain text a viewer does NOT show. A black box
    /// drawn over born-digital text hides it on screen and leaves it in the
    /// content stream, where a text extractor reads straight through it — the
    /// text-layer twin of the scan incident (gap 11). Set when the page carries
    /// a `Redact`/`Square`/`Circle`/`Polygon`/`Stamp` annotation, or a dark
    /// filled shape is drawn AFTER text was shown. A caller handling private
    /// data should OCR the rendered `image` instead, or withhold the page.
    pub redaction_risk: bool,
}

/// Every page of `pdf`, in order. Errors only when the file itself cannot be
/// parsed; a page that cannot be rendered safely carries a note instead.
pub fn pdf_pages(pdf: &[u8]) -> Result<Vec<PdfPage>> {
    let doc = Document::load_mem(pdf).map_err(|e| Error::Media(format!("PDF parse: {e}")))?;
    Ok(doc
        .get_pages()
        .into_iter()
        .enumerate()
        .map(|(index, (_, id))| page(&doc, index, id))
        .collect())
}

fn page(doc: &Document, index: usize, id: ObjectId) -> PdfPage {
    let mut out = PdfPage {
        index,
        has_text_layer: false,
        image: None,
        note: None,
        text: None,
        redaction_risk: false,
    };
    let walked = match walk_page(doc, id) {
        Ok(w) => w,
        Err(e) => {
            // Unwalkable: nothing can be said about what covers the text
            // either, so the risk is the fail-safe answer.
            out.redaction_risk = true;
            out.note = Some(format!("drawing could not be walked, image withheld: {e}"));
            return out;
        }
    };
    out.has_text_layer = walked.text;
    let covering_annotations = annotation_overlays(doc, id).map_or(true, |a| !a.is_empty());
    out.redaction_risk = walked.dark_fill_after_text || covering_annotations;
    if walked.text {
        // lopdf numbers pages from 1.
        let number = u32::try_from(index + 1).unwrap_or(u32::MAX);
        out.text = doc
            .extract_text_with_limit(&[number], MAX_FORM_CONTENT)
            .ok();
    }
    let Some(scan) = walked.scan else {
        out.note = Some("no image on this page".into());
        return out;
    };
    let mut image = match decode_image_xobject(doc, scan.id) {
        Ok(img) => img,
        Err(e) => {
            out.note = Some(format!("scan not decoded: {e}"));
            return out;
        }
    };
    let mut overlays = walked.overlays;
    match annotation_overlays(doc, id) {
        Ok(a) => overlays.extend(a),
        Err(e) => {
            out.note = Some(format!(
                "annotations could not be read, image withheld: {e}"
            ));
            return out;
        }
    }
    paint(&mut image, &scan.ctm, &overlays);
    let (turns, mirrored) = upright_turns(&scan.ctm, page_rotate(doc, id));
    if mirrored {
        image = flip_horizontal(&image);
    }
    out.image = Some(rotate_cw(&image, turns));
    out
}

// ---------------------------------------------------------------- drawing walk

/// A 2-D affine map `[a b c d e f]`: (x, y) -> (a x + c y + e, b x + d y + f).
type Matrix = [f64; 6];

const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// `m` applied first, then `n`.
fn concat(m: Matrix, n: Matrix) -> Matrix {
    [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]
}

fn apply(m: &Matrix, x: f64, y: f64) -> (f64, f64) {
    (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
}

/// A region to paint, in page space, and the gray (0 black .. 1 white).
#[derive(Debug, Clone)]
struct Overlay {
    corners: Vec<(f64, f64)>,
    gray: f64,
}

struct Scan {
    id: ObjectId,
    ctm: Matrix,
    area: f64,
}

struct Walked {
    scan: Option<Scan>,
    overlays: Vec<Overlay>,
    text: bool,
    /// A dark filled shape was drawn after some text was shown: it may be
    /// hiding that text on screen while leaving it in the text layer.
    dark_fill_after_text: bool,
}

/// A fill at or below this gray level counts as a box that could hide text.
const DARK_FILL: f64 = 0.2;

fn real(o: &Object) -> Option<f64> {
    match o {
        // A coordinate or colour beyond 2^52 is not a page; precision there
        // is irrelevant.
        #[allow(clippy::cast_precision_loss)]
        Object::Integer(i) => Some(*i as f64),
        Object::Real(r) => Some(f64::from(*r)),
        _ => None,
    }
}

fn gray_of(values: &[f64]) -> Option<f64> {
    match *values {
        [g] => Some(g),
        [r, g, b] => Some(0.3 * r + 0.59 * g + 0.11 * b),
        [c, m, y, k] => Some(1.0 - (0.3 * c + 0.59 * m + 0.11 * y + k).min(1.0)),
        _ => None,
    }
}

fn walk_page(doc: &Document, page_id: ObjectId) -> std::result::Result<Walked, String> {
    let content = doc
        .get_and_decode_page_content(page_id)
        .map_err(|e| e.to_string())?;
    let xobjects = page_xobjects(doc, page_id);
    let mut w = Walked {
        scan: None,
        overlays: Vec::new(),
        text: false,
        dark_fill_after_text: false,
    };
    walk(doc, &content.operations, &xobjects, IDENTITY, &mut w, 0)?;
    Ok(w)
}

/// The page's `XObject` dictionary, own resources first, then inherited.
fn page_xobjects(doc: &Document, page_id: ObjectId) -> Vec<Dictionary> {
    let Ok((own, inherited)) = doc.get_page_resources(page_id) else {
        return Vec::new();
    };
    own.into_iter()
        .cloned()
        .chain(
            inherited
                .iter()
                .filter_map(|id| doc.get_dictionary(*id).ok().cloned()),
        )
        .filter_map(|r| xobject_dict(doc, &r))
        .collect()
}

fn xobject_dict(doc: &Document, resources: &Dictionary) -> Option<Dictionary> {
    let x = resources.get(b"XObject").ok()?;
    doc.dereference(x)
        .ok()
        .and_then(|(_, o)| o.as_dict().ok().cloned())
}

fn resolve_xobject(dicts: &[Dictionary], name: &[u8]) -> Option<ObjectId> {
    dicts
        .iter()
        .find_map(|d| d.get(name).ok()?.as_reference().ok())
}

/// Largest form `XObject` content stream decompressed, in bytes. A content
/// stream is drawing operators; 64 MiB of them is far past any real page, and
/// the cap is what stops a small hostile file inflating without bound.
const MAX_FORM_CONTENT: usize = 64 << 20;

/// `Do`: an image becomes the scan or an overlay; a form is walked.
fn draw_xobject(
    doc: &Document,
    xobjects: &[Dictionary],
    name: &[u8],
    ctm: Matrix,
    w: &mut Walked,
    depth: usize,
) -> std::result::Result<(), String> {
    let Some(id) = resolve_xobject(xobjects, name) else {
        return Ok(());
    };
    let Ok(stream) = doc.get_object(id).and_then(Object::as_stream) else {
        return Ok(());
    };
    match stream
        .dict
        .get(b"Subtype")
        .ok()
        .and_then(|t| t.as_name().ok())
    {
        Some(b"Image") => {
            let area = (ctm[0] * ctm[3] - ctm[1] * ctm[2]).abs();
            let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                .map(|(x, y)| apply(&ctm, x, y))
                .to_vec();
            match &w.scan {
                // A bigger image replaces the scan; everything drawn before it
                // is underneath and irrelevant.
                Some(s) if area <= s.area => w.overlays.push(Overlay { corners, gray: 0.0 }),
                _ => {
                    w.scan = Some(Scan { id, ctm, area });
                    w.overlays.clear();
                }
            }
        }
        Some(b"Form") => {
            let matrix = stream
                .dict
                .get(b"Matrix")
                .ok()
                .and_then(|m| m.as_array().ok())
                .map(|a| a.iter().filter_map(real).collect::<Vec<_>>())
                .filter(|v| v.len() == 6)
                .map_or(IDENTITY, |v| [v[0], v[1], v[2], v[3], v[4], v[5]]);
            let content = if stream.dict.get(b"Filter").is_ok() {
                stream
                    .decompressed_content_with_limit(MAX_FORM_CONTENT)
                    .map_err(|e| format!("form XObject content: {e}"))?
            } else {
                stream.content.clone()
            };
            let ops = lopdf::content::Content::decode(&content)
                .map_err(|e| format!("form XObject decode: {e}"))?
                .operations;
            let mut inner = xobjects.to_vec();
            if let Some(r) = stream
                .dict
                .get(b"Resources")
                .ok()
                .and_then(|r| doc.dereference(r).ok())
                .and_then(|(_, r)| r.as_dict().ok())
                .and_then(|r| xobject_dict(doc, r))
            {
                inner.insert(0, r);
            }
            walk(doc, &ops, &inner, concat(matrix, ctm), w, depth + 1)?;
        }
        _ => {}
    }
    Ok(())
}

fn walk(
    doc: &Document,
    ops: &[lopdf::content::Operation],
    xobjects: &[Dictionary],
    base: Matrix,
    w: &mut Walked,
    depth: usize,
) -> std::result::Result<(), String> {
    if depth > MAX_FORM_DEPTH {
        return Err(format!("form XObjects nested deeper than {MAX_FORM_DEPTH}"));
    }
    let mut ctm = base;
    let mut fill = 0.0f64;
    let mut stack: Vec<(Matrix, f64)> = Vec::new();
    let mut path: Vec<(f64, f64)> = Vec::new();
    for op in ops {
        let args: Vec<f64> = op.operands.iter().filter_map(real).collect();
        match op.operator.as_str() {
            "q" => stack.push((ctm, fill)),
            "Q" => (ctm, fill) = stack.pop().unwrap_or((ctm, fill)),
            "cm" if args.len() == 6 => {
                ctm = concat([args[0], args[1], args[2], args[3], args[4], args[5]], ctm);
            }
            "g" | "rg" | "k" | "sc" | "scn" => fill = gray_of(&args).unwrap_or(fill),
            "re" if args.len() == 4 => {
                let (x, y, rw, rh) = (args[0], args[1], args[2], args[3]);
                for (px, py) in [(x, y), (x + rw, y), (x + rw, y + rh), (x, y + rh)] {
                    path.push(apply(&ctm, px, py));
                }
            }
            // Every point a path visits, control points included: a curve lies
            // inside the hull of its control points, so the bounding box of
            // these always contains the filled shape.
            "m" | "l" | "c" | "v" | "y" => {
                for pair in args.chunks_exact(2) {
                    path.push(apply(&ctm, pair[0], pair[1]));
                }
            }
            "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" => {
                if w.text && fill <= DARK_FILL && !path.is_empty() {
                    w.dark_fill_after_text = true;
                }
                if w.scan.is_some() && !path.is_empty() {
                    w.overlays.push(Overlay {
                        corners: std::mem::take(&mut path),
                        gray: fill,
                    });
                }
                path.clear();
            }
            "n" | "S" | "s" | "W" | "W*" => {
                if matches!(op.operator.as_str(), "n" | "S" | "s") {
                    path.clear();
                }
            }
            "Tj" | "TJ" | "'" | "\"" => w.text = true,
            "Do" => {
                if let Some(name) = op.operands.first().and_then(|o| o.as_name().ok()) {
                    draw_xobject(doc, xobjects, name, ctm, w, depth)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Opaque annotations drawn over the page, painted black.
///
/// `/Annots` is resolved here rather than through lopdf's
/// `get_page_annotations`, which follows only indirect references: an
/// annotation stored inline in the array would be skipped, and a skipped
/// redaction is the one failure this module exists to prevent.
fn annotation_overlays(
    doc: &Document,
    page_id: ObjectId,
) -> std::result::Result<Vec<Overlay>, String> {
    let page = doc.get_dictionary(page_id).map_err(|e| e.to_string())?;
    let Ok(annots) = page.get(b"Annots") else {
        return Ok(Vec::new());
    };
    let annots = doc.dereference(annots).map_err(|e| e.to_string())?.1;
    let annots = annots
        .as_array()
        .map_err(|_| "/Annots is not an array".to_string())?;
    let mut out = Vec::new();
    for item in annots {
        let a = doc
            .dereference(item)
            .map_err(|e| e.to_string())?
            .1
            .as_dict()
            .map_err(|_| "an annotation is not a dictionary".to_string())?;
        let sub = a.get(b"Subtype").ok().and_then(|s| s.as_name().ok());
        if !matches!(
            sub,
            Some(b"Redact" | b"Square" | b"Circle" | b"Polygon" | b"Stamp")
        ) {
            continue;
        }
        let Some(rect) = a
            .get(b"Rect")
            .ok()
            .and_then(|r| doc.dereference(r).ok())
            .and_then(|(_, r)| r.as_array().ok())
            .map(|v| v.iter().filter_map(real).collect::<Vec<_>>())
            .filter(|v| v.len() == 4)
        else {
            continue;
        };
        let (x0, y0, x1, y1) = (rect[0], rect[1], rect[2], rect[3]);
        out.push(Overlay {
            corners: vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            gray: 0.0,
        });
    }
    Ok(out)
}

fn page_rotate(doc: &Document, page_id: ObjectId) -> i64 {
    // /Rotate is inheritable: walk up through /Parent.
    let mut id = Some(page_id);
    for _ in 0..32 {
        let Some(dict) = id.and_then(|i| doc.get_dictionary(i).ok()) else {
            break;
        };
        if let Ok(r) = dict.get(b"Rotate").and_then(Object::as_i64) {
            return r;
        }
        id = dict.get(b"Parent").ok().and_then(|p| p.as_reference().ok());
    }
    0
}

// ------------------------------------------------------------- painting

/// Paint every overlay onto the scan's pixels. `ctm` maps the image's unit
/// square into page space; its inverse takes an overlay back to pixels.
fn paint(image: &mut ImageBuffer, ctm: &Matrix, overlays: &[Overlay]) {
    let [a, b, c, d, e, f] = *ctm;
    let det = a * d - b * c;
    if det.abs() < f64::EPSILON {
        return;
    }
    let (w, h) = (f64::from(image.width), f64::from(image.height));
    // Page space -> unit square -> pixels. The image's first row is the TOP
    // of the unit square (v = 1).
    let to_px = |(x, y): (f64, f64)| {
        let (u, v) = (
            (d * (x - e) - c * (y - f)) / det,
            (a * (y - f) - b * (x - e)) / det,
        );
        (u * w, (1.0 - v) * h)
    };
    let bpp = image.format.bytes_per_pixel();
    let stride = image.width as usize * bpp;
    for o in overlays {
        let pts: Vec<(f64, f64)> = o.corners.iter().copied().map(to_px).collect();
        // Clamped to [0, image side] first, so the cast cannot truncate or
        // lose a sign. A NaN edge (a degenerate overlay) casts to 0.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let clamp = |v: f64, max: f64| v.clamp(0.0, max) as usize;
        let x0 = clamp(pts.iter().map(|p| p.0).fold(f64::MAX, f64::min).floor(), w);
        let x1 = clamp(pts.iter().map(|p| p.0).fold(f64::MIN, f64::max).ceil(), w);
        let y0 = clamp(pts.iter().map(|p| p.1).fold(f64::MAX, f64::min).floor(), h);
        let y1 = clamp(pts.iter().map(|p| p.1).fold(f64::MIN, f64::max).ceil(), h);
        // In [0, 255] after the clamp.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v = (o.gray.clamp(0.0, 1.0) * 255.0).round() as u8;
        for row in y0..y1 {
            let s = row * stride;
            image.data[s + x0 * bpp..s + x1 * bpp].fill(v);
        }
    }
}

// ------------------------------------------------------------- orientation

/// Clockwise quarter turns that show the image the way a viewer does, and
/// whether its placement is mirrored. Viewer space is page space with y
/// flipped, then rotated clockwise by `/Rotate`.
fn upright_turns(ctm: &Matrix, rotate: i64) -> (u8, bool) {
    // Where the image's +column (right) and +row (down) directions point in
    // viewer space. Image rows run from v = 1 down to v = 0.
    let right = (ctm[0], -ctm[1]);
    let down = (-ctm[2], ctm[3]);
    let mirrored = right.0 * down.1 - right.1 * down.0 < 0.0;
    let right = if mirrored {
        (-right.0, -right.1)
    } else {
        right
    };
    // Dominant direction of the image's right-hand axis, as clockwise turns.
    let placement = if right.0.abs() >= right.1.abs() {
        if right.0 >= 0.0 { 0 } else { 2 }
    } else if right.1 >= 0.0 {
        1
    } else {
        3
    };
    // rem_euclid(360) / 90 is 0..=3; a non-multiple of 90 rounds down.
    let page = u8::try_from(rotate.rem_euclid(360) / 90).unwrap_or(0);
    ((placement + page) % 4, mirrored)
}

fn rotate_cw(img: &ImageBuffer, turns: u8) -> ImageBuffer {
    let turns = turns % 4;
    if turns == 0 {
        return img.clone();
    }
    let (w, h) = (img.width as usize, img.height as usize);
    let bpp = img.format.bytes_per_pixel();
    let (width, height) = if turns == 2 {
        (img.width, img.height)
    } else {
        (img.height, img.width)
    };
    let nw = width as usize;
    let mut out = vec![0u8; w * h * bpp];
    for y in 0..h {
        for x in 0..w {
            let (nx, ny) = match turns {
                1 => (h - 1 - y, x),
                2 => (w - 1 - x, h - 1 - y),
                _ => (y, w - 1 - x),
            };
            let (s, d) = ((y * w + x) * bpp, (ny * nw + nx) * bpp);
            out[d..d + bpp].copy_from_slice(&img.data[s..s + bpp]);
        }
    }
    ImageBuffer {
        width,
        height,
        format: img.format,
        data: out,
    }
}

fn flip_horizontal(img: &ImageBuffer) -> ImageBuffer {
    let (w, bpp) = (img.width as usize, img.format.bytes_per_pixel());
    let mut out = img.clone();
    for row in out.data.chunks_exact_mut(w * bpp) {
        let src = row.to_vec();
        for x in 0..w {
            row[x * bpp..(x + 1) * bpp].copy_from_slice(&src[(w - 1 - x) * bpp..(w - x) * bpp]);
        }
    }
    out
}

// ------------------------------------------------------------- decoding

const fn int(o: &Object) -> Option<i64> {
    match o {
        Object::Integer(i) => Some(*i),
        // A float where an integer belongs: saturating truncation, and every
        // caller range-checks the result before using it as a size.
        #[allow(clippy::cast_possible_truncation)]
        Object::Real(r) => Some(*r as i64),
        _ => None,
    }
}

/// A positive integer entry, as a `usize` — or an error naming the key.
fn dim(dict: &Dictionary, key: &[u8]) -> std::result::Result<usize, String> {
    let k = String::from_utf8_lossy(key);
    let v = dict
        .get(key)
        .ok()
        .and_then(int)
        .ok_or_else(|| format!("missing /{k}"))?;
    usize::try_from(v)
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("/{k} {v} is not a positive size"))
}

fn filters(dict: &Dictionary) -> Vec<Vec<u8>> {
    match dict.get(b"Filter") {
        Ok(Object::Name(n)) => vec![n.clone()],
        Ok(Object::Array(a)) => a
            .iter()
            .filter_map(|o| o.as_name().ok().map(<[u8]>::to_vec))
            .collect(),
        _ => Vec::new(),
    }
}

/// `/DecodeParms` for the LAST filter (the one that produced the samples).
fn decode_parms(doc: &Document, dict: &Dictionary) -> Option<Dictionary> {
    let p = dict.get(b"DecodeParms").ok()?;
    let p = doc.dereference(p).ok()?.1;
    match p {
        Object::Dictionary(d) => Some(d.clone()),
        Object::Array(a) => a.iter().rev().find_map(|o| {
            doc.dereference(o)
                .ok()
                .and_then(|(_, o)| o.as_dict().ok().cloned())
        }),
        _ => None,
    }
}

/// Components per sample of the image's colour space.
fn components(doc: &Document, dict: &Dictionary) -> std::result::Result<usize, String> {
    let Ok(cs) = dict.get(b"ColorSpace") else {
        return Ok(1);
    };
    let cs = doc.dereference(cs).map_err(|e| e.to_string())?.1;
    let name = match cs {
        Object::Name(n) => n.clone(),
        Object::Array(a) => match a.first() {
            Some(Object::Name(n)) if n == b"ICCBased" => {
                let n = a
                    .get(1)
                    .and_then(|s| doc.dereference(s).ok())
                    .and_then(|(_, s)| s.as_stream().ok())
                    .and_then(|s| s.dict.get(b"N").ok().and_then(int))
                    .unwrap_or(3);
                // /N comes from the file and feeds the row-size arithmetic:
                // only the three sizes a colour space can have are accepted.
                return match n {
                    1 => Ok(1),
                    3 => Ok(3),
                    4 => Ok(4),
                    other => Err(format!("ICC colour space with {other} components")),
                };
            }
            Some(Object::Name(n)) => n.clone(),
            _ => return Err("unrecognised colour space".into()),
        },
        _ => return Err("unrecognised colour space".into()),
    };
    match name.as_slice() {
        b"DeviceGray" | b"CalGray" | b"G" => Ok(1),
        b"DeviceRGB" | b"CalRGB" | b"RGB" => Ok(3),
        b"DeviceCMYK" | b"CMYK" => Ok(4),
        other => Err(format!(
            "colour space {} not supported",
            String::from_utf8_lossy(other)
        )),
    }
}

fn decode_image_xobject(doc: &Document, id: ObjectId) -> std::result::Result<ImageBuffer, String> {
    let stream = doc
        .get_object(id)
        .and_then(Object::as_stream)
        .map_err(|e| e.to_string())?;
    let dict = &stream.dict;
    let (w, h) = (dim(dict, b"Width")?, dim(dict, b"Height")?);
    let pixels = w.checked_mul(h).ok_or("image dimensions overflow")?;
    if pixels > MAX_PIXELS {
        return Err(format!("{w}x{h} exceeds the {MAX_PIXELS}-pixel limit"));
    }
    // Both sides are now at most MAX_PIXELS, which fits a u32.
    let (w32, h32) = (
        u32::try_from(w).map_err(|_| "width out of range")?,
        u32::try_from(h).map_err(|_| "height out of range")?,
    );
    let mask = matches!(dict.get(b"ImageMask"), Ok(Object::Boolean(true)));
    let invert = dict
        .get(b"Decode")
        .ok()
        .and_then(|d| d.as_array().ok())
        .and_then(|a| a.first())
        .and_then(real)
        .is_some_and(|first| first > 0.5);
    let fs = filters(dict);
    let last = fs.last().map(Vec::as_slice);

    let gray = |data: Vec<u8>| ImageBuffer {
        width: w32,
        height: h32,
        format: PixelFormat::Gray8,
        data,
    };

    match last {
        Some(b"DCTDecode") => {
            let img = crate::decode_image(&stream.content).map_err(|e| e.to_string())?;
            if (img.width as usize, img.height as usize) != (w, h) {
                return Err("JPEG dimensions disagree with /Width and /Height".into());
            }
            Ok(img)
        }
        Some(b"CCITTFaxDecode") => {
            let parms = decode_parms(doc, dict);
            let get = |k: &[u8]| parms.as_ref().and_then(|p| p.get(k).ok()).and_then(int);
            let k = get(b"K").unwrap_or(0);
            // /Columns sizes the decoder's line buffers; it must be a real
            // row width, not whatever the file says.
            let columns = match get(b"Columns") {
                None => w,
                Some(c) => usize::try_from(c)
                    .ok()
                    .filter(|c| (1..=MAX_CCITT_COLUMNS).contains(c))
                    .ok_or_else(|| format!("CCITT /Columns {c} out of range"))?,
            };
            let black_is_1 = matches!(
                parms.as_ref().and_then(|p| p.get(b"BlackIs1").ok()),
                Some(Object::Boolean(true))
            );
            let mut bits = ccitt(&stream.content, k, columns, w, h)?;
            // `ccitt` yields 0 for black. BlackIs1 and a [1 0] /Decode each
            // flip what a sample means; two flips cancel.
            if black_is_1 ^ invert {
                for v in &mut bits {
                    *v = 255 - *v;
                }
            }
            Ok(gray(bits))
        }
        Some(b"JBIG2Decode") => Err("JBIG2 is not supported yet".into()),
        Some(b"JPXDecode") => Err("JPEG 2000 is not supported".into()),
        _ => {
            let bpc = match dict
                .get(b"BitsPerComponent")
                .ok()
                .and_then(int)
                .unwrap_or(1)
            {
                1 => 1usize,
                8 => 8,
                other => return Err(format!("{other} bits per component is not supported")),
            };
            let comps = if mask { 1 } else { components(doc, dict)? };
            let row_bytes = (w * comps * bpc).div_ceil(8);
            let expected = row_bytes.checked_mul(h).ok_or("image size overflows")?;
            let raw = if fs.is_empty() {
                stream.content.clone()
            } else {
                stream
                    .decompressed_content_with_limit(expected + 1024)
                    .map_err(|e| format!("decompress: {e}"))?
            };
            if raw.len() < expected {
                return Err(format!("{} bytes of samples, {expected} needed", raw.len()));
            }
            samples_to_image(&raw, (w, w32), (h, h32), comps, bpc, invert)
        }
    }
}

/// Widest CCITT row accepted: 100 000 pixels, far past a 600 ppi tabloid.
const MAX_CCITT_COLUMNS: usize = 100_000;

/// Raw samples to an 8-bit image. `invert_one_bit`: for 1-bit data, whether a
/// 1 bit is dark. By default a 1 bit is white for BOTH a 1-bit gray image and
/// a stencil mask (a mask paints where the sample is 0), so the mask flag does
/// not change the mapping; only a `[1 0]` /Decode flips it.
fn samples_to_image(
    raw: &[u8],
    (w, width): (usize, u32),
    (h, height): (usize, u32),
    comps: usize,
    bpc: usize,
    invert_one_bit: bool,
) -> std::result::Result<ImageBuffer, String> {
    let row_bytes = (w * comps * bpc).div_ceil(8);
    let image = |format, data| {
        Ok(ImageBuffer {
            width,
            height,
            format,
            data,
        })
    };
    match (comps, bpc) {
        (1, 1) => {
            let mut data = Vec::with_capacity(w * h);
            for y in 0..h {
                let row = &raw[y * row_bytes..(y + 1) * row_bytes];
                for x in 0..w {
                    let bit = (row[x / 8] >> (7 - (x % 8))) & 1 == 1;
                    // Default gray: 1 is white. For a stencil mask, a 0 bit is
                    // painted (black) — also "1 is white". An invert flips both.
                    let white = bit != invert_one_bit;
                    data.push(if white { 255 } else { 0 });
                }
            }
            image(PixelFormat::Gray8, data)
        }
        (1, 8) => image(
            PixelFormat::Gray8,
            (0..h)
                .flat_map(|y| raw[y * row_bytes..y * row_bytes + w].iter().copied())
                .collect(),
        ),
        (3, 8) => image(
            PixelFormat::Rgb8,
            (0..h)
                .flat_map(|y| raw[y * row_bytes..y * row_bytes + 3 * w].iter().copied())
                .collect(),
        ),
        (4, 8) => {
            // CMYK to RGB, naive: enough for OCR, which reads luminance.
            let mut data = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                for px in raw[y * row_bytes..y * row_bytes + 4 * w].chunks_exact(4) {
                    let k = 255 - u16::from(px[3]);
                    for c in &px[..3] {
                        // (255 - c) * k / 255 is at most 255.
                        data.push(u8::try_from((255 - u16::from(*c)) * k / 255).unwrap_or(u8::MAX));
                    }
                }
            }
            image(PixelFormat::Rgb8, data)
        }
        (c, b) => Err(format!(
            "{c} components at {b} bits per component is not supported"
        )),
    }
}

/// CCITT Group 3 (1-D) or Group 4 to 8-bit gray, 0 for black.
fn ccitt(
    data: &[u8],
    k: i64,
    columns: usize,
    w: usize,
    h: usize,
) -> std::result::Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(w * h);
    let mut line_cb = |transitions: &[u32]| {
        if out.len() >= w * h {
            return;
        }
        let mut line = vec![255u8; w];
        let (mut black, mut x) = (false, 0usize);
        for &t in transitions {
            let t = (t as usize).min(w);
            if black {
                line[x..t].fill(0);
            }
            x = t;
            black = !black;
        }
        if black {
            line[x.min(w)..].fill(0);
        }
        out.extend_from_slice(&line);
    };
    let columns = u32::try_from(columns).map_err(|_| "CCITT /Columns out of range")?;
    let rows = u32::try_from(h).map_err(|_| "image height out of range")?;
    let ok = match k.cmp(&0) {
        std::cmp::Ordering::Less => {
            fax::decoder::decode_g4(data.iter().copied(), columns, Some(rows), &mut line_cb)
        }
        std::cmp::Ordering::Equal => fax::decoder::decode_g3(data.iter().copied(), &mut line_cb),
        std::cmp::Ordering::Greater => {
            return Err("CCITT Group 3 two-dimensional (K > 0) is not supported yet".into());
        }
    };
    if ok.is_none() && out.is_empty() {
        return Err("CCITT data did not decode".into());
    }
    out.resize(w * h, 255);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{Stream, dictionary};

    /// A one-page PDF: `content` drawn with `images` as named XObjects, and
    /// any extra page-dictionary entries (`/Rotate`, `/Annots`).
    fn pdf(content: &[u8], images: Vec<(&str, Stream)>, extra: Vec<(&str, Object)>) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let mut xobjects = Dictionary::new();
        for (name, stream) in images {
            let id = doc.add_object(stream);
            xobjects.set(name, Object::Reference(id));
        }
        let content_id = doc.add_object(Stream::new(Dictionary::new(), content.to_vec()));
        let pages_id = doc.new_object_id();
        let mut page = dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Resources" => dictionary! { "XObject" => xobjects },
        };
        for (k, v) in extra {
            page.set(k, v);
        }
        let page_id = doc.add_object(page);
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![page_id.into()],
                "Count" => 1,
            }),
        );
        let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog);
        let mut out = Vec::new();
        doc.save_to(&mut out).unwrap();
        out
    }

    /// An uncompressed 8-bit gray image XObject.
    fn gray_image(w: usize, h: usize, value: impl Fn(usize, usize) -> u8) -> Stream {
        let data: Vec<u8> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| value(x, y))
            .collect();
        Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image",
                "Width" => w as i64, "Height" => h as i64,
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8,
            },
            data,
        )
    }

    fn only_page(bytes: &[u8]) -> PdfPage {
        let mut pages = pdf_pages(bytes).unwrap();
        assert_eq!(pages.len(), 1);
        pages.remove(0)
    }

    fn at(img: &ImageBuffer, x: usize, y: usize) -> u8 {
        img.data[y * img.width as usize + x]
    }

    /// Gap 11, the incident: a patch image pasted over the scan. The scan is
    /// 100x50 drawn over page (10,20)-(210,120); the patch covers page
    /// (60,70)-(110,120), i.e. pixels x 25..50, rows 0..25.
    #[test]
    fn a_patch_drawn_over_the_scan_is_painted_black() {
        let bytes = pdf(
            b"q 200 0 0 100 10 20 cm /Scan Do Q q 50 0 0 50 60 70 cm /Patch Do Q",
            vec![
                ("Scan", gray_image(100, 50, |_, _| 200)),
                ("Patch", gray_image(4, 4, |_, _| 255)),
            ],
            vec![],
        );
        let p = only_page(&bytes);
        let img = p.image.expect("composited scan");
        assert_eq!((img.width, img.height), (100, 50));
        assert_eq!(
            at(&img, 30, 10),
            0,
            "under the patch: black, even though the patch is white"
        );
        assert_eq!(at(&img, 30, 40), 200, "below the patch");
        assert_eq!(at(&img, 10, 10), 200, "left of the patch");
        assert!(!p.has_text_layer);
    }

    #[test]
    fn filled_rectangles_after_the_scan_are_painted_in_their_colour_and_before_it_are_not() {
        let bytes = pdf(
            b"0 g 10 20 200 100 re f q 200 0 0 100 10 20 cm /Scan Do Q \
              0 g 60 70 50 50 re f 1 g 10 20 20 20 re f",
            vec![("Scan", gray_image(100, 50, |_, _| 128))],
            vec![],
        );
        let img = only_page(&bytes).image.unwrap();
        assert_eq!(at(&img, 30, 10), 0, "black box over the scan");
        assert_eq!(at(&img, 2, 45), 255, "white-out box over the scan");
        assert_eq!(
            at(&img, 80, 40),
            128,
            "the box drawn BEFORE the scan is under it"
        );
    }

    #[test]
    fn a_redaction_annotation_is_painted() {
        let redact = Object::Dictionary(dictionary! {
            "Type" => "Annot", "Subtype" => "Redact",
            "Rect" => vec![60.into(), 70.into(), 110.into(), 120.into()],
        });
        let bytes = pdf(
            b"q 200 0 0 100 10 20 cm /Scan Do Q",
            vec![("Scan", gray_image(100, 50, |_, _| 200))],
            vec![("Annots", Object::Array(vec![redact]))],
        );
        let img = only_page(&bytes).image.unwrap();
        assert_eq!(at(&img, 30, 10), 0);
        assert_eq!(at(&img, 30, 40), 200);
    }

    /// A page stored with `/Rotate 90` comes back the way a viewer shows it:
    /// turned a quarter clockwise, so the scan's top-left pixel is top-right.
    #[test]
    fn page_rotate_is_applied() {
        let bytes = pdf(
            b"q 200 0 0 100 10 20 cm /Scan Do Q",
            vec![(
                "Scan",
                gray_image(100, 50, |x, y| if x == 0 && y == 0 { 7 } else { 200 }),
            )],
            vec![("Rotate", Object::Integer(90))],
        );
        let img = only_page(&bytes).image.unwrap();
        assert_eq!((img.width, img.height), (50, 100));
        assert_eq!(at(&img, 49, 0), 7);
    }

    /// An image placed with a rotating matrix: its columns run DOWN the page,
    /// which on screen is down — shown upright, it is turned a quarter
    /// clockwise, so its top-left pixel lands top-right.
    #[test]
    fn a_rotated_placement_is_turned_upright() {
        let bytes = pdf(
            b"q 0 -100 200 0 10 200 cm /Scan Do Q",
            vec![(
                "Scan",
                gray_image(100, 50, |x, y| if x == 0 && y == 0 { 7 } else { 200 }),
            )],
            vec![],
        );
        let img = only_page(&bytes).image.unwrap();
        assert_eq!((img.width, img.height), (50, 100));
        assert_eq!(at(&img, 49, 0), 7);
    }

    #[test]
    fn ccitt_group4_round_trips_and_black_is_black() {
        use fax::{Color, VecWriter, encoder::Encoder};
        let (w, h) = (64usize, 8usize);
        let black = |x: usize, y: usize| (8..24).contains(&x) && y >= 2;
        let mut enc = Encoder::new(VecWriter::new());
        for y in 0..h {
            enc.encode_line(
                (0..w).map(|x| {
                    if black(x, y) {
                        Color::Black
                    } else {
                        Color::White
                    }
                }),
                w as u32,
            )
            .unwrap();
        }
        let data = enc.finish().unwrap().finish();
        let stream = Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image",
                "Width" => w as i64, "Height" => h as i64,
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 1,
                "Filter" => "CCITTFaxDecode",
                "DecodeParms" => dictionary! { "K" => -1, "Columns" => w as i64 },
            },
            data,
        );
        let bytes = pdf(
            b"q 64 0 0 8 0 0 cm /Scan Do Q",
            vec![("Scan", stream)],
            vec![],
        );
        let img = only_page(&bytes).image.expect("CCITT G4 decodes");
        for y in 0..h {
            for x in 0..w {
                assert_eq!(at(&img, x, y) == 0, black(x, y), "pixel ({x},{y})");
            }
        }
    }

    /// A black box drawn over born-digital text hides it on screen and leaves
    /// it in the content stream. The text is still extracted, so the page
    /// must say it is a redaction risk — and a plain text page must not.
    #[test]
    fn text_under_a_black_box_is_flagged_as_a_redaction_risk() {
        let font = "BT /F1 12 Tf 10 700 Td (123 Secret Lane) Tj ET";
        let plain = only_page(&pdf(font.as_bytes(), vec![], vec![]));
        assert!(plain.has_text_layer);
        assert!(!plain.redaction_risk, "nothing covers this text");

        let covered = format!("{font} 0 g 5 695 200 20 re f");
        let p = only_page(&pdf(covered.as_bytes(), vec![], vec![]));
        assert!(
            p.redaction_risk,
            "a black box drawn after the text may be hiding it"
        );

        let white_first = format!("1 g 0 0 612 792 re f {font}");
        let p = only_page(&pdf(white_first.as_bytes(), vec![], vec![]));
        assert!(
            !p.redaction_risk,
            "a background fill BEFORE the text hides nothing"
        );
    }

    #[test]
    fn a_text_layer_is_reported() {
        let bytes = pdf(b"BT /F1 12 Tf 10 10 Td (hello) Tj ET", vec![], vec![]);
        let p = only_page(&bytes);
        assert!(p.has_text_layer);
        assert!(p.image.is_none());
        assert_eq!(p.note.as_deref(), Some("no image on this page"));
    }

    #[test]
    fn unsupported_and_oversized_images_are_reported_not_guessed() {
        let mut jbig2 = gray_image(4, 4, |_, _| 0);
        jbig2.dict.set("Filter", "JBIG2Decode");
        let p = only_page(&pdf(
            b"q 10 0 0 10 0 0 cm /Scan Do Q",
            vec![("Scan", jbig2)],
            vec![],
        ));
        assert!(p.image.is_none());
        assert!(p.note.unwrap().contains("JBIG2"));

        let mut huge = gray_image(1, 1, |_, _| 0);
        huge.dict.set("Width", 20_000);
        huge.dict.set("Height", 20_000);
        let p = only_page(&pdf(
            b"q 10 0 0 10 0 0 cm /Scan Do Q",
            vec![("Scan", huge)],
            vec![],
        ));
        assert!(p.image.is_none());
        assert!(p.note.unwrap().contains("pixel limit"));
    }

    /// A stencil mask (`/ImageMask true`) paints where the sample is 0.
    #[test]
    fn a_one_bit_stencil_mask_paints_zero_bits_black() {
        let stream = Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image",
                "Width" => 8, "Height" => 1, "ImageMask" => true, "BitsPerComponent" => 1,
            },
            vec![0b1111_0000],
        );
        let img = only_page(&pdf(
            b"q 8 0 0 1 0 0 cm /Scan Do Q",
            vec![("Scan", stream)],
            vec![],
        ))
        .image
        .unwrap();
        assert_eq!(img.data, vec![255, 255, 255, 255, 0, 0, 0, 0]);
    }
}
