//! Does the win survive in the configuration PRODUCTION actually runs?
//!
//! # Why `vision_arm_ab` is not enough
//!
//! `vision_arm_ab` times ONE tower with [`siglip::set_kernels_parallel`] left at
//! its default `true`. Production does neither of those things. On a 24-core box
//! `engine::tile_workers` returns **6**, and the engine then sets
//!
//! ```text
//! kernels_parallel = workers * 6 <= cores   ->   6 * 6 = 36 <= 24   ->   FALSE
//! ```
//!
//! so a real caption runs **six concurrent towers with the custom kernels
//! SERIAL**, while the tower A/B measured **one tower with them parallel**. A
//! kernel can win handsomely in the second regime and lose in the first: serial,
//! it competes with nothing; six-up, every core is already busy and the only
//! thing that still matters is how many bytes it moves.
//!
//! This is the reachability check — *does the shipping path reach the code I
//! measured, in the state I measured it?* A win that only exists at
//! `kernels_parallel = true` is not a win; it is a benchmark artefact.
//!
//! So: the real engine, the real tile count, the real worker count, one whole
//! caption per sample, ABBA-interleaved in one process.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example caption_arm_ab -- fuse_bias
//! ```
use ffai_core::engine::{VlmEngine, VlmOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};
use std::path::Path;
use std::time::Instant;

const IMG: usize = 384;
const ROUNDS: usize = 3;

/// The same deterministic pattern the oracle tests use.
fn reference_image() -> ImageBuffer {
    let mut data = vec![0u8; IMG * IMG * 3];
    let mut i = 0;
    for y in 0..IMG {
        let fy = y as f64 / IMG as f64;
        for x in 0..IMG {
            let fx = x as f64 / IMG as f64;
            let r = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fx).sin();
            let g = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fy + 1.0).sin();
            let b = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * (fx + fy) + 2.0).sin();
            data[i] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            i += 3;
        }
    }
    ImageBuffer { width: IMG as u32, height: IMG as u32, format: PixelFormat::Rgb8, data }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = std::env::args().nth(1).unwrap_or_else(|| "fuse_bias".into());
    // `kernels_parallel` and `tile_workers` are read from the environment by
    // the engine on EVERY call, so setting the variable between captions
    // toggles them in-process — which is what keeps this an interleaved A/B
    // rather than two processes measured an hour apart.
    fn set_env(key: &'static str, on: &'static str, off: &'static str) -> impl Fn(bool) -> bool {
        move |v| {
            let prev = std::env::var(key).ok();
            // SAFETY: single-threaded here — set between captions, never during
            // one. No other thread reads the environment at this point.
            #[allow(unsafe_code)]
            unsafe {
                std::env::set_var(key, if v { on } else { off });
            }
            prev.as_deref() == Some(on)
        }
    }

    let toggle: Box<dyn Fn(bool) -> bool> = match arm.as_str() {
        "fuse_bias" => Box::new(ffai_argus::siglip::set_fuse_bias),
        "late_norm" => Box::new(ffai_argus::siglip::set_late_normalize),
        "fused_ln" => Box::new(ffai_argus::siglip::set_fused_ln),
        "head_attn" => Box::new(ffai_argus::siglip::set_head_attn),
        // ON = kernels parallel; OFF = the shipped heuristic's answer (serial
        // at 6 workers on 24 cores).
        "kernels_parallel" => Box::new(set_env("FFAI_ARGUS_KERNELS_PARALLEL", "1", "0")),
        // ON = 4 tile workers; OFF = the shipped 6.
        "workers4" => Box::new(set_env("FFAI_ARGUS_TILE_WORKERS", "4", "6")),
        "workers8" => Box::new(set_env("FFAI_ARGUS_TILE_WORKERS", "8", "6")),
        other => return Err(format!("unknown arm {other}").into()),
    };

    let manifests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let engine = ffai_argus::SmolVlm::with_manifest_dir(manifests);
    let img = reference_image();
    let opts = VlmOptions { max_new_tokens: Some(32), ..VlmOptions::default() };

    let mut run = |on: bool| -> Result<(f64, String), Box<dyn std::error::Error>> {
        let prev = toggle(on);
        let t = Instant::now();
        let text = engine.describe_image(&img, &opts)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        toggle(prev);
        Ok((ms, text))
    };

    // Correctness first: the arms must produce the SAME CAPTION. A faster arm
    // that says something else is not an optimisation.
    let (_, on_text) = run(true)?;
    let (_, off_text) = run(false)?;
    println!("  arm `{arm}` ON : {on_text:?}");
    println!("  arm `{arm}` OFF: {off_text:?}");
    if on_text != off_text {
        return Err("arms produce different captions — not a like-for-like A/B".into());
    }
    println!("  captions AGREE\n");

    let (mut a, mut b) = (Vec::new(), Vec::new());
    for r in 0..ROUNDS {
        for on in if r % 2 == 0 { [true, false] } else { [false, true] } {
            let (ms, _) = run(on)?;
            if on {
                a.push(ms);
            } else {
                b.push(ms);
            }
        }
        // A second, order-reversed pair per round completes the ABBA.
        for on in if r % 2 == 0 { [false, true] } else { [true, false] } {
            let (ms, _) = run(on)?;
            if on {
                a.push(ms);
            } else {
                b.push(ms);
            }
        }
        println!("  round {}/{ROUNDS}", r + 1);
    }

    let stat = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[0], v[v.len() / 2])
    };
    let (amin, amed) = stat(&mut a);
    let (bmin, bmed) = stat(&mut b);
    println!("\n  WHOLE CAPTION — real engine, real workers — {} samples/arm\n", a.len());
    println!("  {:<18} {:>10} {:>10}", "arm", "min ms", "median ms");
    println!("  {:-<18} {:->10} {:->10}", "", "", "");
    println!("  {:<18} {bmin:>10.0} {bmed:>10.0}", format!("{arm} OFF"));
    println!("  {:<18} {amin:>10.0} {amed:>10.0}", format!("{arm} ON"));
    println!("  {:-<18} {:->10} {:->10}", "", "", "");
    println!("  {:<18} {:>9.3}x {:>9.3}x", "speedup", bmin / amin, bmed / amed);
    Ok(())
}
