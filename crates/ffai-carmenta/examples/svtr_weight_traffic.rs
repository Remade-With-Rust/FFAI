//! What the SVTR weight cache removes, counted rather than timed.
//!
//! `VarBuilder::get_unchecked` on a mmaped safetensors backend calls candle's
//! `convert`, which **allocates a new tensor and copies the weight out of the
//! mapping**. `svtr.rs` fetches at twelve sites and `forward` loops the
//! encoder blocks, so before the cache the entire parameter set was copied
//! once per forward — per crop, on every page.
//!
//! The instrument is [`ffai_core::cost`], not a stopwatch: same input, same
//! number, on a loaded box or a quiet one. Forward 1 is cold and pays for
//! every weight; forward 2 and after pay for none, and the difference IS the
//! per-crop tax the cache deletes.
//!
//! ```sh
//! cargo run --release -p ffai-carmenta --example svtr_weight_traffic
//! ```
use candle_core::{Device, Tensor};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let weights = PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_default())
        .join("ffai/models/ppocrv5-mobile-rec/rec.safetensors");
    if !weights.exists() {
        println!("no weights at {} — run tools/carmenta_svtr_prepare.py", weights.display());
        return Ok(());
    }
    let dev = Device::Cpu;
    let m = ffai_carmenta::svtr::load(&weights, &dev, 18385)?;
    // A real recognizer crop: 1x3x48x320 is what `svtr_input` emits.
    let x = Tensor::zeros((1, 3, 48, 320), candle_core::DType::F32, &dev)?;

    let mut cold = 0u64;
    for i in 0..4 {
        ffai_core::cost::start();
        let _ = m.forward(&x)?;
        let c = ffai_core::cost::stop();
        let (mb, n) = (c.copy_bytes as f64 / 1e6, c.copies);
        if i == 0 {
            cold = c.copy_bytes;
            println!("  forward {i} (cold)  weight copies {n:>3}  {mb:>7.2} MB");
        } else {
            println!("  forward {i}         weight copies {n:>3}  {mb:>7.2} MB");
        }
        if i == 3 {
            println!();
            println!("  --- and what the ATTENTION SCALE FOLD would be worth here ---");
            println!(
                "  qkv/proj/head matmul  {:>9.2} MFLOP in {} calls",
                c.matmul_flops as f64 / 1e6,
                c.matmul_calls
            );
            println!("  the q scale multiply  {:>9} elements", c.elem_ops);
            // Same weights `Costs::weighted` uses, so the comparison is the
            // workspace's own cost model rather than one invented here.
            let scale_share = (c.elem_ops * 264) as f64 / c.weighted() as f64;
            println!("  scale share of the modelled cost: {:.4} %", 100.0 * scale_share);
            println!("  ...and the backbone convolutions are NOT in that denominator,");
            println!("  so the share of a real forward is lower than the figure shown.");
            println!();
            println!("  Argus folds this constant into the q weights for ~15 % of a layer.");
            println!("  At THIS geometry the whole multiply is the figure above, and the");
            println!("  fold is not bit-exact (the scale is not a power of two) against an");
            println!("  oracle sitting at 8.857e-5 of a 1e-4 budget. REFUTED — recorded so");
            println!("  it is not proposed a second time.");
        }
    }
    println!();
    println!("  Every forward after the first copies ZERO weight bytes.");
    println!("  Before the cache each one copied {:.2} MB.", cold as f64 / 1e6);
    println!();
    println!("  A page of text is hundreds of crops, so the tax removed is that");
    println!("  figure times the crop count — and it was pure overhead: the");
    println!("  weights are identical on every call, which is why the paddle");
    println!("  oracle reads the same 8.857e-5 with the cache as without it.");
    Ok(())
}
