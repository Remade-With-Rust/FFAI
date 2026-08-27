//! Do the tile workers belong in rayon's pool or outside it?
//!
//! `run_tower` spawns `workers` raw OS threads with `std::thread::scope`, each
//! of which drives candle ops that use rayon's GLOBAL pool. So up to
//! `workers + 24` threads compete for 24 cores with no shared scheduler, and
//! rayon cannot steal across the tile boundary. Running the tiles ON the pool
//! lets one work-stealing scheduler see all of it.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rayon::prelude::*;
use std::time::Instant;

#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let snap = std::fs::read_dir(std::path::Path::new(&home).join(
        ".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots",
    ))?.flatten().next().ok_or("no snapshot")?.path();
    let d = Device::Cpu;
    let cj = std::fs::read_to_string(snap.join("config.json"))?;
    let (vcfg, _s) = ffai_argus::vision::vision_config_from_json(&cj)?;
    // SAFETY: mapped file owned by the model cache.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(
        std::slice::from_ref(&snap.join("model.safetensors")), DType::F32, &d)? };
    let tower = ffai_argus::siglip::VisionTower::new(&vcfg, vb.pp("model.vision_model"))?;
    let side = vcfg.image_size;
    const TILES: usize = 17;
    let tiles: Vec<Tensor> = (0..TILES)
        .map(|_| Tensor::rand(-1.0f32, 1.0, (1, 3, side, side), &d).expect("t"))
        .collect();

    // ARM A: raw OS threads, kernels OFF — what run_tower does today.
    let os_threads = |workers: usize| -> f64 {
        ffai_argus::siglip::set_kernels_parallel(false);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let t = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= TILES { break; }
                    std::hint::black_box(tower.forward(&tiles[i]).expect("f").dims());
                });
            }
        });
        t.elapsed().as_secs_f64() * 1e3
    };
    // ARM B: rayon's own pool, kernels ON — one scheduler sees everything.
    let on_pool = || -> f64 {
        ffai_argus::siglip::set_kernels_parallel(true);
        let t = Instant::now();
        tiles.par_iter().for_each(|x| {
            std::hint::black_box(tower.forward(x).expect("f").dims());
        });
        t.elapsed().as_secs_f64() * 1e3
    };

    let _ = (os_threads(6), on_pool());
    let (mut a, mut b, mut wins) = (f64::INFINITY, f64::INFINITY, 0);
    for i in 0..5 {
        let (ta, tb) = if i % 2 == 0 { let x = os_threads(6); (x, on_pool()) }
                       else { let y = on_pool(); (os_threads(6), y) };
        a = a.min(ta); b = b.min(tb);
        if tb < ta { wins += 1; }
    }
    println!("  17 tiles, 24 cores");
    println!("  {:<40} {a:>8.1} ms", "6 OS threads, kernels off [ships]");
    println!("  {:<40} {b:>8.1} ms   {:>5.2}x   faster in {wins}/5", "rayon par_iter, kernels on", a / b);
    Ok(())
}
