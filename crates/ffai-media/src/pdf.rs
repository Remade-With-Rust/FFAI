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
//! Parsing, decryption and image decoding are `rusty_pdf_core`'s: DCT, CCITT
//! (Group 3 one- and two-dimensional, Group 4), JBIG2 with its globals,
//! JPEG 2000, and raw/Flate/LZW/RunLength samples at 1 to 16 bits in gray,
//! RGB, CMYK, Indexed and ICC spaces. Every image is bounded by
//! [`MAX_PIXELS`] before anything is allocated for it, and decompression is
//! capped at what the declared dimensions need, so a hostile PDF cannot drive
//! an unbounded allocation.

// Page geometry is f64 affine arithmetic on coordinates a PDF writer chose. A
// fused multiply-add rounds differently from every other reader of the same
// file, and nothing here needs the extra precision.
#![allow(clippy::suboptimal_flops)]

use rusty_pdf_core::{
    ColorSpace, Dictionary, Document, ImageOptions, Object, ObjectId, Operation, Page,
};

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
    /// The page's text layer, when it has one (text in content-stream order,
    /// no positions). **Read `redaction_risk` before using it.**
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
    let doc = Document::load(pdf).map_err(|e| Error::Media(format!("PDF parse: {e}")))?;
    Ok((0..doc.page_count())
        .map(|index| match doc.page(index) {
            Ok(p) => page(&doc, index, &p),
            Err(e) => PdfPage {
                index,
                has_text_layer: false,
                image: None,
                note: Some(format!("page could not be read: {e}")),
                text: None,
                redaction_risk: true,
            },
        })
        .collect())
}

fn page(doc: &Document, index: usize, p: &Page<'_>) -> PdfPage {
    let mut out = PdfPage {
        index,
        has_text_layer: false,
        image: None,
        note: None,
        text: None,
        redaction_risk: false,
    };
    let walked = match walk_page(doc, p) {
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
    let annotations = annotation_overlays(doc, p);
    let covering_annotations = annotations.as_ref().map_or(true, |a| !a.is_empty());
    out.redaction_risk = walked.dark_fill_after_text || covering_annotations;
    if walked.text {
        out.text = p.extract_text(MAX_FORM_CONTENT).ok();
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
    match annotations {
        Ok(a) => overlays.extend(a),
        Err(e) => {
            out.note = Some(format!(
                "annotations could not be read, image withheld: {e}"
            ));
            return out;
        }
    }
    paint(&mut image, &scan.ctm, &overlays);
    let (turns, mirrored) = upright_turns(&scan.ctm, i64::from(p.rotate()));
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

const fn real(o: &Object) -> Option<f64> {
    match o {
        // A coordinate or colour beyond 2^52 is not a page; precision there
        // is irrelevant.
        #[allow(clippy::cast_precision_loss)]
        Object::Integer(i) => Some(*i as f64),
        Object::Real(r) => Some(*r),
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

fn walk_page(doc: &Document, p: &Page<'_>) -> std::result::Result<Walked, String> {
    let ops = p.operations().map_err(|e| e.to_string())?;
    let xobjects: Vec<Dictionary> = p
        .resources()
        .and_then(|r| doc.lookup_dict(r, b"XObject"))
        .cloned()
        .into_iter()
        .collect();
    let mut w = Walked {
        scan: None,
        overlays: Vec::new(),
        text: false,
        dark_fill_after_text: false,
    };
    walk(doc, &ops, &xobjects, IDENTITY, &mut w, 0)?;
    Ok(w)
}

fn resolve_xobject(dicts: &[Dictionary], name: &[u8]) -> Option<ObjectId> {
    dicts.iter().find_map(|d| d.get(name)?.as_reference().ok())
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
    let subtype = doc
        .lookup(&stream.dict, b"Subtype")
        .ok()
        .and_then(|t| t.as_name().ok());
    match subtype {
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
            let matrix = doc
                .lookup(&stream.dict, b"Matrix")
                .ok()
                .and_then(|m| m.as_array().ok())
                .map(|a| a.iter().filter_map(real).collect::<Vec<_>>())
                .filter(|v| v.len() == 6)
                .map_or(IDENTITY, |v| [v[0], v[1], v[2], v[3], v[4], v[5]]);
            let content = doc
                .decode_stream(stream, MAX_FORM_CONTENT)
                .map_err(|e| format!("form XObject content: {e}"))?;
            let ops = rusty_pdf_core::Content::parse_with(&content, doc.limits())
                .map_err(|e| format!("form XObject decode: {e}"))?
                .operations;
            let mut inner = xobjects.to_vec();
            if let Some(r) = doc
                .lookup_dict(&stream.dict, b"Resources")
                .and_then(|r| doc.lookup_dict(r, b"XObject"))
            {
                inner.insert(0, r.clone());
            }
            walk(doc, &ops, &inner, concat(matrix, ctm), w, depth + 1)?;
        }
        _ => {}
    }
    Ok(())
}

fn walk(
    doc: &Document,
    ops: &[Operation],
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

/// Opaque annotations drawn over the page, painted black. `Page::annotations`
/// reads inline entries as well as references and fails on an entry that is
/// not a dictionary: a skipped redaction is the one failure this module
/// exists to prevent.
fn annotation_overlays(doc: &Document, p: &Page<'_>) -> std::result::Result<Vec<Overlay>, String> {
    let mut out = Vec::new();
    for a in p.annotations().map_err(|e| e.to_string())? {
        let sub = doc
            .lookup(a, b"Subtype")
            .ok()
            .and_then(|s| s.as_name().ok());
        if !matches!(
            sub,
            Some(b"Redact" | b"Square" | b"Circle" | b"Polygon" | b"Stamp")
        ) {
            continue;
        }
        let Some(rect) = doc
            .lookup(a, b"Rect")
            .ok()
            .and_then(|r| r.as_array().ok())
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

/// The scan's image `XObject` as 8-bit gray or RGB. A stencil mask comes
/// back with 0 where it paints, which as gray is black: the module's
/// fail-safe reading of a mask drawn as a scan. An alpha channel (JPEG 2000
/// `SMaskInData`) is dropped: OCR reads the colour, and a mask cannot reveal
/// anything the colour hides.
fn decode_image_xobject(doc: &Document, id: ObjectId) -> std::result::Result<ImageBuffer, String> {
    let stream = doc
        .get_object(id)
        .and_then(Object::as_stream)
        .map_err(|e| e.to_string())?;
    let img = doc
        .decode_image(stream, &ImageOptions::with_max_pixels(MAX_PIXELS))
        .map_err(|e| e.to_string())?;
    let (format, data) = match img.color_space {
        ColorSpace::Gray => (PixelFormat::Gray8, img.samples),
        _ => (PixelFormat::Rgb8, img.to_rgb8().map_err(|e| e.to_string())?),
    };
    Ok(ImageBuffer {
        width: img.width,
        height: img.height,
        format,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{Dictionary, Document, Object, Stream, dictionary};

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
        // Not a JPEG 2000 codestream: an error note, never pixels.
        let mut jpx = gray_image(4, 4, |_, _| 0);
        jpx.dict.set("Filter", "JPXDecode");
        let p = only_page(&pdf(
            b"q 10 0 0 10 0 0 cm /Scan Do Q",
            vec![("Scan", jpx)],
            vec![],
        ));
        assert!(p.image.is_none());
        assert!(p.note.unwrap().contains("JPEG 2000"));

        let mut huge = gray_image(1, 1, |_, _| 0);
        huge.dict.set("Width", 20_000);
        huge.dict.set("Height", 20_000);
        let p = only_page(&pdf(
            b"q 10 0 0 10 0 0 cm /Scan Do Q",
            vec![("Scan", huge)],
            vec![],
        ));
        assert!(p.image.is_none());
        let note = p.note.unwrap();
        assert!(note.contains("pixel"), "{note}");
    }

    /// rusty_jbig2's documented page: 16x1, one MMR generic region, 8 white
    /// then 8 black pixels.
    #[rustfmt::skip]
    const JBIG2_PAGE: [u8; 61] = [
        0, 0, 0, 0, 48, 0, 1, 0, 0, 0, 19, // page information, page 1
        0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 1, 38, 0, 1, 0, 0, 0, 20, // immediate generic region
        0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, //
        0b0011_0011, 0b0001_0100, // horizontal mode, white 8, black 8
    ];

    fn jbig2_image(decode: Option<Vec<Object>>) -> Stream {
        let mut dict = dictionary! {
            "Type" => "XObject", "Subtype" => "Image",
            "Width" => 16, "Height" => 1,
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 1,
            "Filter" => "JBIG2Decode",
        };
        if let Some(d) = decode {
            dict.set("Decode", d);
        }
        Stream::new(dict, JBIG2_PAGE.to_vec())
    }

    #[test]
    fn jbig2_scans_decode_with_black_as_black() {
        let p = only_page(&pdf(
            b"q 16 0 0 1 0 0 cm /Scan Do Q",
            vec![("Scan", jbig2_image(None))],
            vec![],
        ));
        let img = p.image.expect("JBIG2 decodes");
        assert_eq!((img.width, img.height), (16, 1));
        assert!((0..8).all(|x| at(&img, x, 0) == 255), "left half white");
        assert!((8..16).all(|x| at(&img, x, 0) == 0), "right half black");

        // A [1 0] /Decode flips it, as for any 1-bit image.
        let p = only_page(&pdf(
            b"q 16 0 0 1 0 0 cm /Scan Do Q",
            vec![("Scan", jbig2_image(Some(vec![1.into(), 0.into()])))],
            vec![],
        ));
        let img = p.image.expect("JBIG2 decodes");
        assert_eq!((at(&img, 0, 0), at(&img, 15, 0)), (0, 255));
    }

    #[test]
    fn a_jbig2_page_of_the_wrong_size_is_refused() {
        let mut s = jbig2_image(None);
        s.dict.set("Width", 8);
        let p = only_page(&pdf(
            b"q 8 0 0 1 0 0 cm /Scan Do Q",
            vec![("Scan", s)],
            vec![],
        ));
        assert!(p.image.is_none());
        assert!(p.note.unwrap().contains("JBIG2"));
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
