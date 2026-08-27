//! Batch-N through the tower vs N separate batch-1 calls, INTERLEAVED.
//!
//! `tile_batching_ab` sweeps whole captions and this box gives it a 1.76x
//! spread, so its verdict has never been trustworthy. This asks the narrow
//! question — for the SAME four tiles, is one batch-4 forward faster than four
//! batch-1 forwards? — with both arms in one process, alternating, judged by a
//! sign test.
//!
//! It is newly answerable at all: `packed.i((.., 0))` was contiguous only at
//! `b == 1`, so any batch > 1 died in candle's matmul until that was fixed.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let snap = std::fs::read_dir(
        std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots"),
    )?
    .flatten()
    .next()
    .ok_or("no snapshot")?
    .path();
    let d = Device::Cpu;
    let cj = std::fs::read_to_string(snap.join("config.json"))?;
    let (vcfg, _scale) = ffai_argus::vision::vision_config_from_json(&cj)?;
    // SAFETY: mapped file owned by the model cache, not mutated here.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&snap.join("model.safetensors")),
            DType::F32,
            &d,
        )?
    };
    let tower = ffai_argus::siglip::VisionTower::new(&vcfg, vb.pp("model.vision_model"))?;

    let side = vcfg.image_size;
    let one = Tensor::rand(-1.0f32, 1.0, (1, 3, side, side), &d)?;
    for n in [2usize, 4] {
        let many = Tensor::cat(&vec![&one; n], 0)?.contiguous()?;

        // equality: batching must not change the answer
        let a = tower.forward(&many)?;
        let b = tower.forward(&one)?;
        let a0 = a.narrow(0, 0, 1)?;
        let worst = (&a0 - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;

        let looped = || {
            let t = Instant::now();
            for _ in 0..n {
                std::hint::black_box(tower.forward(&one).expect("one").dims());
            }
            t.elapsed().as_secs_f64() * 1e3
        };
        let batched = || {
            let t = Instant::now();
            std::hint::black_box(tower.forward(&many).expect("many").dims());
            t.elapsed().as_secs_f64() * 1e3
        };
        let _ = (looped(), batched());
        let (mut bl, mut bb, mut wins) = (f64::INFINITY, f64::INFINITY, 0);
        for i in 0..6 {
            let (tl, tb) = if i % 2 == 0 {
                let x = looped();
                (x, batched())
            } else {
                let y = batched();
                (looped(), y)
            };
            bl = bl.min(tl);
            bb = bb.min(tb);
            if tb < tl {
                wins += 1;
            }
        }
        println!(
            "  batch {n}:  {n} x forward(1) {bl:>8.1} ms   forward({n}) {bb:>8.1} ms   \
             {:>5.2}x   faster in {wins}/6   max|diff| {worst:.2e}",
            bl / bb
        );
    }
    println!("\n  Above 1.00x with a 6/6 sign test is a real win; the attention tensor");
    println!("  grows with the batch, so the footprint gate pays for whatever is taken.");
    Ok(())
}
