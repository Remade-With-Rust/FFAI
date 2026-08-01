//! Paired A/B of rayon pool sizes, inside ONE process.
//!
//! `RAYON_NUM_THREADS` sweeps showed CPU work per image falling 1.66x from
//! 24 threads to 8 — we burn a third of our CPU on parallel overhead. But
//! those were separate processes measured minutes apart on a box three other
//! benchmarks are using, and the WALL column disagreed with the CPU column:
//! more threads grab more of a contended machine even while doing more total
//! work. Wall is what the speed gate measures, so the disagreement has to be
//! settled on wall.
//!
//! Sequential measurement cannot settle it — the two arms would sample two
//! different machines. So: both pools live in one process, alternate ABBA
//! per round, and the verdict is the paired win rate. Under the null that is
//! a fair coin, `z = (wins - N/2) / (0.5*sqrt(N))`, and |z| > 2 is real
//! however far the medians drift.
//!
//! Why this is expected to matter at all: this box is 16 physical / 24
//! logical, an Intel hybrid — 8 P-cores with SMT plus 8 E-cores. Torch picks
//! exactly 8. Our kernels are `par_chunks_mut`, i.e. BARRIERS, and a
//! barrier's cost is set by its slowest participant, so handing equal chunks
//! to unequal cores makes every barrier wait on an E-core.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example pool_ab -- [tier] [rounds] [a] [b]
//! ```

use std::time::Instant;

use ffai_core::engine::{DetectEngine, DetectOptions};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let tier = args.next().unwrap_or_else(|| "n".into());
    let rounds: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(24);
    let na: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(24);
    let nb: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf();
    let mut clips: Vec<_> = std::fs::read_dir(root.join("corpora/clips/diana-coco"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    clips.sort();
    clips.truncate(4);
    let images: Vec<_> =
        clips.iter().map(|p| ffai_media::load_image(p)).collect::<Result<Vec<_>, _>>()?;

    let engine = ffai_diana::engine::Yolo26::build(
        &tier,
        ffai_diana::image::Geometry::Rect,
        root.join("models"),
    );
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };

    let pool_a = rayon::ThreadPoolBuilder::new().num_threads(na).build()?;
    let pool_b = rayon::ThreadPoolBuilder::new().num_threads(nb).build()?;

    // Warm BOTH pools: a cold pool pays thread spawn on its first task, and
    // whichever arm ran first would otherwise carry that cost forever.
    pool_a.install(|| engine.detect(&images[0], &opts)).map(|_| ())?;
    pool_b.install(|| engine.detect(&images[0], &opts)).map(|_| ())?;

    let run = |pool: &rayon::ThreadPool| -> f64 {
        let t = Instant::now();
        pool.install(|| {
            for im in &images {
                engine.detect(im, &opts).expect("detect");
            }
        });
        t.elapsed().as_secs_f64() * 1e3
    };

    println!("tier {tier} · {} images/round · {rounds} rounds · A={na} threads B={nb} threads",
             images.len());
    let mut ratios = Vec::new();
    let mut wins = 0usize; // B (fewer threads) faster
    for r in 0..rounds {
        let (a, b) = if r % 2 == 0 {
            let a = run(&pool_a);
            let b = run(&pool_b);
            (a, b)
        } else {
            let b = run(&pool_b);
            let a = run(&pool_a);
            (a, b)
        };
        if b < a {
            wins += 1;
        }
        ratios.push(a / b);
        println!("  round {:2}/{rounds}: A({na}) {a:7.1} ms   B({nb}) {b:7.1} ms   A/B {:.3}x",
                 r + 1, a / b);
    }

    let n = rounds as f64;
    let z = (wins as f64 - n / 2.0) / (0.5 * n.sqrt());
    println!();
    println!("B({nb}) faster in {wins}/{rounds}   z = {z:+.2}");
    println!("median A/B ratio: {:.4}x   (>1 means fewer threads WINS)", median(&mut ratios));
    println!(
        "{}",
        if z.abs() > 2.0 {
            "verdict: REAL at |z| > 2"
        } else {
            "verdict: inside the noise — not a result"
        }
    );
    Ok(())
}
