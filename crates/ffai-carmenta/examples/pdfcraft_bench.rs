//! Read a list of pages with `mobiledet-svtr` from an OFFLINE weights root and
//! write one JSON line per page in the shape PdfCraft's `score.py` reads:
//! `{"page", "ms", "lines": [[[text, [l, t, r, b]], ...], ...]}`.
//!
//! It exists so a caller's corpus (artcraft's pdfcraft-ocr pages, rendered
//! with ground truth by construction) can be scored against THIS tree, not
//! only against the release on crates.io. A line with word geometry emits one
//! entry per word; a line without it emits the whole line as one entry with
//! the line box, which the scorer's CER is indifferent to.
//!
//! Usage: `pdfcraft_bench <weights_root> <page.png>...`
//! Env: `BENCH_REC=crnn` for mobiledet-crnn, `BENCH_ORIENT=1` for
//! `auto_orient`, `BENCH_DESKEW=1` for `deskew`, `BENCH_LANG=en,fr` for
//! `languages`.

#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage};
use ffai_core::engine::{OcrEngine, OcrOptions};

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = args.next().expect("usage: pdfcraft_bench <weights_root> <page.png>...");
    let pages: Vec<String> = args.collect();
    let rec = match std::env::var("BENCH_REC").as_deref() {
        Ok("crnn") => RecStage::Crnn,
        _ => RecStage::Svtr,
    };
    let engine = CraftCrnn::offline(rec, DetStage::MobileDet, Path::new(&root));
    let t = Instant::now();
    engine.preload().expect("preload");
    eprintln!("load {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    let opts = OcrOptions {
        auto_orient: std::env::var("BENCH_ORIENT").is_ok_and(|v| v == "1"),
        deskew: std::env::var("BENCH_DESKEW").is_ok_and(|v| v == "1"),
        languages: std::env::var("BENCH_LANG")
            .map(|v| v.split(',').map(str::to_string).collect())
            .unwrap_or_default(),
        ..OcrOptions::default()
    };
    let all = Instant::now();
    let (mut words_total, mut lines_with_words, mut lines_total) = (0usize, 0usize, 0usize);
    for p in &pages {
        let img = ffai_media::load_image(Path::new(p)).expect("load image");
        let t = Instant::now();
        let out = engine.recognize(&img, &opts).expect("recognize");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let name = Path::new(p).file_stem().unwrap().to_string_lossy().into_owned();
        let mut s = format!("{{\"page\":\"{}\",\"ms\":{ms:.1},\"lines\":[", esc(&name));
        for (li, line) in out.lines().enumerate() {
            lines_total += 1;
            if li > 0 {
                s.push(',');
            }
            s.push('[');
            let entries: Vec<(String, [f32; 4])> = if line.words.is_empty() {
                let b = line.bbox.map_or([0.0; 4], |b| [b.x, b.y, b.x + b.width, b.y + b.height]);
                vec![(line.text.clone(), b)]
            } else {
                lines_with_words += 1;
                words_total += line.words.len();
                line.words
                    .iter()
                    .map(|w| {
                        let b = w.bbox.map_or([0.0; 4], |b| [b.x, b.y, b.x + b.width, b.y + b.height]);
                        (w.text.clone(), b)
                    })
                    .collect()
            };
            for (wi, (text, b)) in entries.iter().enumerate() {
                if wi > 0 {
                    s.push(',');
                }
                let _ = write!(
                    s,
                    "[\"{}\",[{:.1},{:.1},{:.1},{:.1}]]",
                    esc(text),
                    b[0],
                    b[1],
                    b[2],
                    b[3]
                );
            }
            s.push(']');
        }
        s.push_str("]}");
        println!("{s}");
    }
    eprintln!(
        "pages {} total {:.1} ms; lines {lines_total}, with words {lines_with_words}, words {words_total}",
        pages.len(),
        all.elapsed().as_secs_f64() * 1e3
    );
    if std::env::var_os("FFAI_PROFILE").is_some() {
        eprint!("{}", ffai_carmenta::profile::profile().report());
    }
}
