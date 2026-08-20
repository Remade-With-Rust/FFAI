//! Read MANY images with one named Carmenta engine, models loaded ONCE.
//!
//! Exists for the OmniDocBench **Text OCR task** (plan item 5): region-level
//! recognition over thousands of GT crops. `ocr_text` pays the full model load
//! per process — fine for 1 page, ruinous for 10 000 crops — so this reads a
//! filelist and streams results, one line per crop:
//!
//!     <path>\t<recognized text, newlines replaced by \u{23CE}>
//!
//! Newlines inside a crop's text are replaced with a visible sentinel so the
//! output stays one-record-per-line and the consumer can restore them. A crop
//! that fails to load or recognize is emitted with empty text rather than
//! skipped — a silently shrinking population is §3 rule 10.
//!
//! Usage: `ocr_batch <engine> <filelist>` where filelist is one path per line.

#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage};
use ffai_core::engine::{OcrEngine, OcrOptions};
use std::io::Write;

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args.next().expect("usage: ocr_batch <engine> <filelist>");
    let list = args.next().expect("usage: ocr_batch <engine> <filelist>");

    let (det, rec) = match name.as_str() {
        "craft-crnn" => (DetStage::Craft, RecStage::Crnn),
        "craft-parseq" => (DetStage::Craft, RecStage::Parseq),
        "mobiledet-crnn" => (DetStage::MobileDet, RecStage::Crnn),
        "mobiledet-parseq" => (DetStage::MobileDet, RecStage::Parseq),
        "mobiledet-svtr" => (DetStage::MobileDet, RecStage::Svtr),
        other => panic!("unknown engine `{other}`"),
    };
    let engine = match det {
        DetStage::Craft if rec == RecStage::Crnn => CraftCrnn::new(),
        DetStage::Craft => CraftCrnn::new_parseq(),
        DetStage::MobileDet => CraftCrnn::new_mobiledet(rec),
        DetStage::Composed => CraftCrnn::new_composed(rec),
    };

    // FFAI_OCR_CONF=1: emit per-line GEOMETRY as well, for offline ordering
    // analysis (plan §23 step 1). One record per image either way; lines are
    // joined by U+23CE and, in conf mode, fields within a line by U+001F
    // (x␟y␟w␟h␟conf␟text). Same env var as `ocr_text` — format only, the
    // engine's behaviour is identical.
    let with_conf = std::env::var("FFAI_OCR_CONF").is_ok();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let paths = std::fs::read_to_string(&list).expect("read filelist");
    for path in paths.lines().map(str::trim).filter(|p| !p.is_empty()) {
        let text = ffai_media::load_image(std::path::Path::new(path))
            .ok()
            .and_then(|img| engine.recognize(&img, &OcrOptions::default()).ok())
            .map(|o| {
                o.blocks
                    .iter()
                    .flat_map(|b| b.lines.iter())
                    .map(|l| {
                        if with_conf {
                            let b = l.bbox.as_ref();
                            format!(
                                "{}\u{1F}{}\u{1F}{}\u{1F}{}\u{1F}{:.4}\u{1F}{}",
                                b.map_or(0.0, |b| b.x),
                                b.map_or(0.0, |b| b.y),
                                b.map_or(0.0, |b| b.width),
                                b.map_or(0.0, |b| b.height),
                                l.confidence.unwrap_or(-1.0),
                                l.text
                            )
                        } else {
                            l.text.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\u{23CE}")
            })
            .unwrap_or_default();
        writeln!(out, "{path}\t{}", text.replace('\n', "\u{23CE}")).expect("write");
    }
}
