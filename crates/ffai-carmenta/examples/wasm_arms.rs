//! Step 0 of `docs/plans/carmenta-wasm-plan.md`: what does wasm cost the conv?
//!
//! A browser build has no AVX2 and no threads, so it runs `conv3x3_scalar`
//! single-threaded. §8.101 measured every *scalar* form of this kernel BELOW
//! candle's im2col (0.21x-0.67x) and only the vectorised one above it (1.65x),
//! which raises a question the record does not answer: **is the arm a wasm
//! build actually takes faster or slower than candle's?** If it is slower, the
//! first wasm change is a `cfg` picking candle there, and that is a one-line
//! win available before any kernel is written.
//!
//! This measures the three arms on real pages. It does NOT rotate them
//! in-process: `std::env::set_var` is `unsafe` in edition 2024 and candle's
//! own threads are live, so the arm is read from the environment and the
//! ROTATION IS THE CALLER'S — the same shape as Diana's `FFAI_DIANA_NO_AVX2`
//! runs. `tools/carmenta_wasm_arms.sh` drives it ABBA-style.
//!
//! ```sh
//! FFAI_REC_SERIAL=1 FFAI_CONV3X3=scalar \
//!   cargo run --release -p ffai-carmenta --example wasm_arms -- 5 page1.png page2.png
//! ```
//!
//! `FFAI_REC_SERIAL=1` is not optional for this measurement: without it a
//! multi-line page fans out across cores and the number stops being the
//! browser's. Reported per image is the **min** of N, which is the estimator
//! this repo uses for a hot in-process loop — the mean would be measuring the
//! machine's other tenants.

use ffai_core::engine::OcrOptions;
use ffai_core::registry::EngineRegistry;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let reps: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .expect("usage: wasm_arms <reps> <image.png> [image.png ...]");
    // The ENGINE is an arm too, and on browser-sized input it is the
    // dominant one: a 620x200 capture spends 86% of its time in `det_fwd`,
    // not in the recognizer the conv kernel lives in. Naming it here is what
    // let that be measured instead of assumed.
    let engine_name = args.next().expect("usage: wasm_arms <reps> <engine> <image.png> ...");
    let paths: Vec<PathBuf> = args.map(PathBuf::from).collect();
    assert!(!paths.is_empty(), "usage: wasm_arms <reps> <engine> <image.png> ...");

    // Name the arm in the output. A timing whose configuration is not printed
    // beside it is how a three-arm comparison becomes three unlabelled numbers.
    let arm = match std::env::var("FFAI_CONV3X3").as_deref() {
        Ok("0") => "candle-im2col",
        Ok("scalar") => "ours-scalar",  // the arm a wasm build takes
        _ => "ours-avx2",
    };
    let serial = std::env::var("FFAI_REC_SERIAL").is_ok();
    println!("conv arm: {arm}   rec_serial: {serial}   reps: {reps}");
    if !serial {
        eprintln!("WARNING: FFAI_REC_SERIAL is unset — this is not a browser-shaped number");
    }

    let mut reg = EngineRegistry::new();
    ffai_carmenta::register(&mut reg);
    let engine = reg.ocr(Some(&engine_name)).expect("unknown engine");
    println!("engine: {}", engine.info().name);
    let opts = OcrOptions::default();

    let images: Vec<_> = paths
        .iter()
        .map(|p| {
            let img = ffai_media::load_image(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            (p.clone(), img)
        })
        .collect();

    // One untimed pass: weights load lazily on first recognize, and model load
    // inside a timed run is the defect §6.1 fixed at the benchmark level.
    let _ = engine.recognize(&images[0].1, &opts);

    let mut total_min = 0f64;
    for (path, img) in &images {
        let mut best = f64::MAX;
        let mut lines = 0usize;
        for _ in 0..reps {
            let t0 = Instant::now();
            let out = engine.recognize(img, &opts).expect("recognize failed");
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
                lines = out.blocks.iter().map(|b| b.lines.len()).sum();
            }
        }
        total_min += best;
        println!(
            "  {:<28} {:>9.1} ms   {:>3} lines   {}x{}",
            path.file_name().unwrap_or_default().to_string_lossy(),
            best,
            lines,
            img.width,
            img.height,
        );
    }
    println!("TOTAL {arm}: {total_min:.1} ms over {} images", images.len());
    // Where it went, when asked. The stages are already there (§8.100 kept
    // them); an instrument that cannot say WHICH stage is the browser problem
    // is an instrument that only says "slow".
    if std::env::var_os("FFAI_PROFILE").is_some() {
        eprint!("{}", ffai_carmenta::profile::profile().report());
    }
}
