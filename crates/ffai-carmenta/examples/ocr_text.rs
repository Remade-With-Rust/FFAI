//! Read one image with one named Carmenta engine and print its text.
//!
//! The same thing `ffai ocr` does, minus `ffai-cli`. It exists because the
//! measurement loop should not be able to be blocked by an unrelated crate:
//! the CLI links every component, so a sibling campaign mid-edit takes the
//! whole binary — and with it every sweep — down with it. This depends on
//! `ffai-carmenta` alone.
//!
//! Usage: `ocr_text <engine> <image>`

// MATCH THE SHIPPING BINARY (§8.104). `ffai-cli` sets mimalloc as its global
// allocator with a recorded 1.64x justification, and a global allocator is a
// LINK-TIME choice that examples do not inherit. Every wall-clock number this
// example produced before this line was therefore measured on a configuration
// that does not ship — the system allocator, whose 1 MiB throughput is 16x
// lower and whose 16-thread scaling is 5.09x against mimalloc's 6.07x.
//
// A measurement harness that differs from the product in its allocator is
// measuring the wrong program.
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage};
use ffai_core::engine::{OcrEngine, OcrOptions};

fn main() {
    let _profile_guard = ();
    let mut args = std::env::args().skip(1);
    let name = args.next().expect("usage: ocr_text <engine> <image>");
    let path = args.next().expect("usage: ocr_text <engine> <image>");

    let (det, rec) = match name.as_str() {
        "craft-crnn" => (DetStage::Craft, RecStage::Crnn),
        "craft-parseq" => (DetStage::Craft, RecStage::Parseq),
        "mobiledet-crnn" => (DetStage::MobileDet, RecStage::Crnn),
        "mobiledet-parseq" => (DetStage::MobileDet, RecStage::Parseq),
        "composed-crnn" => (DetStage::Composed, RecStage::Crnn),
        "composed-parseq" => (DetStage::Composed, RecStage::Parseq),
        other => panic!("unknown engine `{other}`"),
    };
    let engine = match det {
        DetStage::Craft if rec == RecStage::Crnn => CraftCrnn::new(),
        DetStage::Craft => CraftCrnn::new_parseq(),
        DetStage::MobileDet => CraftCrnn::new_mobiledet(rec),
        DetStage::Composed => CraftCrnn::new_composed(rec),
    };

    let img = ffai_media::load_image(std::path::Path::new(&path)).expect("load image");
    let out = engine.recognize(&img, &OcrOptions::default()).expect("recognize");
    // FFAI_OCR_CONF=1 prefixes each line with its recognition confidence, so a
    // probe can ask whether low-confidence output is separable from good output
    // — globally, or only relative to the page it came from.
    let with_conf = std::env::var("FFAI_OCR_CONF").is_ok();
    for block in &out.blocks {
        for line in &block.lines {
            if with_conf {
                let b = line.bbox.as_ref();
                println!(
                    "{:.4}	{}	{}	{}	{}	{}",
                    line.confidence.unwrap_or(-1.0),
                    b.map_or(0.0, |b| b.x), b.map_or(0.0, |b| b.y),
                    b.map_or(0.0, |b| b.width), b.map_or(0.0, |b| b.height),
                    line.text
                );
            } else {
                println!("{}", line.text);
            }
        }
    }
    if std::env::var_os("FFAI_PROFILE").is_some() {
        eprint!("{}", ffai_carmenta::profile::profile().report());
    }

}

// FFAI_PROFILE=1 prints the stage breakdown. The CLI already does this
// (main.rs:533) but the CLI links every component, and this example exists
// precisely so a sibling campaign's mid-edit cannot block a measurement.
