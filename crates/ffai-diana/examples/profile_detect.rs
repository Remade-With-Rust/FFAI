//! M-D2.0's measurement spine: a deterministic in-process benchmark plus the
//! stage profile, with the noise floor reported every run.
//!
//! Two numbers matter and they are taken separately, because the profiler
//! inflates what it measures:
//!
//!   - **throughput** with `FFAI_PROFILE` UNSET (the honest ms/image),
//!   - **the stage breakdown** with it set (the relative ranking).
//!
//! Never quote a speed from a profiled run, and never rank stages from an
//! unprofiled one.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example profile_detect -- [image] [runs]
//! FFAI_PROFILE=1 cargo run --release -p ffai-diana --example profile_detect
//! ```

use ffai_core::engine::{DetectEngine, DetectOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "corpora/clips/diana-coco/coco-032.png".into());
    let runs: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(15);

    let image = ffai_media::load_image(std::path::Path::new(&path))?;
    // `FFAI_DIANA_SQUARE=1` selects the square-padded engine — the A/B arm
    // for the rectangular default, and the geometry the parity gate pins.
    let engine = if std::env::var_os("FFAI_DIANA_SQUARE").is_some() {
        ffai_diana::engine::Yolo26::square()
    } else {
        ffai_diana::engine::Yolo26::new()
    };
    // The bench decode settings, not the human-facing defaults — mAP needs
    // the low-confidence tail (see run_detect_engine).
    let opts = DetectOptions { confidence: 0.001, max_detections: 100, ..Default::default() };

    engine.detect(&image, &opts)?; // warm: weights, caches, allocator
    ffai_diana::profile::reset(); // ...and do not count the warm pass

    let mut ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = std::time::Instant::now();
        engine.detect(&image, &opts)?;
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    ms.sort_by(|a, b| a.total_cmp(b));
    let (min, med, max) = (ms[0], ms[ms.len() / 2], ms[ms.len() - 1]);

    println!(
        "{} ({}x{}) · {runs} runs · threads {:?}",
        path,
        image.width,
        image.height,
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .unwrap_or_else(|| std::thread::available_parallelism()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "?".into()))
    );
    println!(
        "min {min:.1} ms · median {med:.1} ms · max {max:.1} ms · spread {:.2}x  \
         ({:.1} img/s at min)",
        max / min,
        1000.0 / min
    );
    println!(
        "NOISE FLOOR: any A/B delta under {:.1} ms ({:.1}%) is not a result",
        max - min,
        (max - min) / med * 100.0
    );

    if ffai_diana::profile::is_enabled() {
        print!("{}", ffai_diana::profile::profile().report());
        println!(
            "\n(profiled run — quote the RANKING from here, the THROUGHPUT from an \
             unprofiled one)"
        );
    } else {
        println!("\n(set FFAI_PROFILE=1 for the stage breakdown)");
    }
    Ok(())
}
