//! Read one image with one named Carmenta engine and print its text.
//!
//! The same thing `ffai ocr` does, minus `ffai-cli`. It exists because the
//! measurement loop should not be able to be blocked by an unrelated crate:
//! the CLI links every component, so a sibling campaign mid-edit takes the
//! whole binary — and with it every sweep — down with it. This depends on
//! `ffai-carmenta` alone.
//!
//! Usage: `ocr_text <engine> <image>`

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage};
use ffai_core::engine::{OcrEngine, OcrOptions};

fn main() {
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
    for block in &out.blocks {
        for line in &block.lines {
            println!("{}", line.text);
        }
    }
}
