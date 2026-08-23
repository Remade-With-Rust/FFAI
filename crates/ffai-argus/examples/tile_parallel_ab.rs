//! Tiles are independent. Does running them on separate THREADS beat running
//! them one after another?
//!
//! # Why this is not the batching experiment again
//!
//! `tile_batching_ab` grew the BATCH dimension — one `forward` over `n` tiles —
//! and measured 1.07x. Refuted. This is a different axis: the same batch-1
//! forwards, issued concurrently from `n` threads.
//!
//! They are different because of what `vision_ops_probe` found. candle's CPU
//! backend uses rayon for conv2d and **nothing else**: its elementwise and
//! layout kernels are single-threaded. Half of a `SigLIP` layer is exactly
//! those ops — GELU alone is 19.7 % — so during half the tower this box runs
//! one core out of twenty-four.
//!
//! Batching cannot fix that; the kernel stays single-threaded however long the
//! array is. Threading can: seventeen tiles is seventeen independent single-
//! threaded workloads, which is the one shape that fills the machine without
//! touching candle at all.
//!
//! The risk is the other half. Matmul *does* use every core, so `n` concurrent
//! towers oversubscribe during the GEMMs. Whether the elementwise win pays for
//! the GEMM contention is not something to reason about — hence this file.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example tile_parallel_ab
//! ```

use candle_core::{Device, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("workspace root")?;
    let manifests = ffai_models::load_dir(&root.join("models"))?;
    let manifest = manifests
        .iter()
        .find(|m| m.name == ffai_argus::engine::MODEL)
        .ok_or("no smolvlm manifest")?;
    let resolved = manifest.fetch()?;
    let weights = resolved.file("model.safetensors")?.to_path_buf();
    let config = std::fs::read_to_string(resolved.file("config.json")?)?;

    let device = Device::Cpu;
    let vision = Arc::new(ffai_argus::vision::load(&weights, &config, &device)?);

    let (w, h) = (512usize, 512usize);
    let mut px = vec![0u8; w * h * 3];
    for (i, v) in px.iter_mut().enumerate() {
        *v = (i % 251) as u8;
    }
    let pre = ffai_argus::preprocess::preprocess_rgb8(&px, w, h);
    let per = 3 * pre.tile * pre.tile;
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    println!("{} tiles, {cores} logical cores\n", pre.tiles);

    // Pre-build the input tensors once so the measurement is the TOWER, not
    // tensor construction.
    let tiles: Vec<Tensor> = (0..pre.tiles)
        .map(|t| {
            Tensor::from_vec(
                pre.pixel_values[t * per..(t + 1) * per].to_vec(),
                (1, 3, pre.tile, pre.tile),
                &device,
            )
            .expect("tile")
        })
        .collect();
    let tiles = Arc::new(tiles);

    let sequential = || -> f64 {
        let t = Instant::now();
        let mut out = Vec::with_capacity(tiles.len());
        for tile in tiles.iter() {
            out.push(vision.forward(tile).expect("tower"));
        }
        std::hint::black_box(out.len());
        t.elapsed().as_secs_f64() * 1e3
    };

    let threaded = |workers: usize| -> f64 {
        let t = Instant::now();
        let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let results = Arc::new(std::sync::Mutex::new(vec![None; tiles.len()]));
        std::thread::scope(|s| {
            for _ in 0..workers {
                let (v, tl, n, r) = (
                    Arc::clone(&vision),
                    Arc::clone(&tiles),
                    Arc::clone(&next),
                    Arc::clone(&results),
                );
                s.spawn(move || {
                    // Work-stealing by atomic counter rather than a fixed split:
                    // tiles cost the same in principle, but the last worker in a
                    // fixed split waits on whichever core the OS descheduled.
                    loop {
                        let i = n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= tl.len() {
                            break;
                        }
                        let out = v.forward(&tl[i]).expect("tower");
                        r.lock().expect("results")[i] = Some(out);
                    }
                });
            }
        });
        std::hint::black_box(results.lock().expect("r").len());
        t.elapsed().as_secs_f64() * 1e3
    };

    // Correctness FIRST: a faster arm that computes something else is not a
    // faster arm. Compare the threaded result against the sequential one.
    {
        let want: Vec<Tensor> = tiles.iter().map(|t| vision.forward(t).expect("seq")).collect();
        let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let got = Arc::new(std::sync::Mutex::new(vec![None; tiles.len()]));
        std::thread::scope(|s| {
            for _ in 0..4 {
                let (v, tl, n, r) = (
                    Arc::clone(&vision),
                    Arc::clone(&tiles),
                    Arc::clone(&next),
                    Arc::clone(&got),
                );
                s.spawn(move || loop {
                    let i = n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= tl.len() {
                        break;
                    }
                    r.lock().expect("g")[i] = Some(v.forward(&tl[i]).expect("par"));
                });
            }
        });
        let got = got.lock().expect("g");
        let mut worst = 0.0f32;
        for (i, w) in want.iter().enumerate() {
            let g = got[i].as_ref().expect("tile");
            worst = worst.max((w - g)?.abs()?.max_all()?.to_scalar::<f32>()?);
        }
        println!("threaded vs sequential: max_abs = {worst:.3e}");
        assert!(worst == 0.0, "threading changed the result — stop here");
        println!("  bit-identical ✓\n");
    }

    let _ = sequential(); // warm

    println!("{:>10}  {:>10}  {:>8}   runs (ms)", "workers", "min (ms)", "vs seq");
    let mut base = f64::INFINITY;
    // Interleave: sequential, then each worker count, three rounds, so a slow
    // patch of wall-clock hits every arm rather than whichever ran first.
    let mut runs: std::collections::BTreeMap<usize, Vec<f64>> = std::collections::BTreeMap::new();
    let counts: Vec<usize> = vec![1, 2, 4, 6, 8, 12, 17];
    for _ in 0..3 {
        runs.entry(0).or_default().push(sequential());
        for &n in &counts {
            runs.entry(n).or_default().push(threaded(n));
        }
    }
    for (n, v) in &runs {
        let m = v.iter().copied().fold(f64::INFINITY, f64::min);
        if *n == 0 {
            base = m;
        }
        let label = if *n == 0 {
            "sequential".to_string()
        } else {
            n.to_string()
        };
        let fmt: Vec<String> = v.iter().map(|x| format!("{x:.0}")).collect();
        println!(
            "{label:>10}  {m:>10.0}  {:>7.2}x   {}",
            base / m,
            fmt.join(" ")
        );
    }
    println!(
        "\nHalf a SigLIP layer is single-threaded in candle (elementwise + layout).\n\
         If threading pays, that is the half it is paying for."
    );
    Ok(())
}
