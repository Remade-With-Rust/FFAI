//! D6/D3 probe: is our latency gap MORE WORK, or the SAME work spread worse?
//!
//! The bench reports wall-clock per-image latency and we trail the reference
//! on it. Wall alone cannot say why, and the two causes want opposite fixes:
//!
//!   * same CPU time, worse wall  -> we parallelise badly. The work is fine;
//!     the fix is occupancy, and the ceiling is (cores x current wall).
//!   * more CPU time              -> our kernels genuinely do more work. The
//!     fix is the kernels, and parallelism buys nothing it is not already
//!     buying.
//!
//! CPU time sums across threads, so `cpu / wall` IS the mean occupancy —
//! the number of cores we actually kept busy. That single ratio routes the
//! whole campaign, and it costs one run.
//!
//! It is also the instrument that survives a loaded box: CPU time does not
//! accrue while descheduled, which matters here because this machine is
//! usually running another benchmark.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example cpu_vs_wall -- [tier] [images]
//! ```

use std::time::Instant;

use ffai_core::engine::{DetectEngine, DetectOptions};

#[cfg(windows)]
mod cpu {
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(h: isize, c: *mut u64, e: *mut u64, k: *mut u64, u: *mut u64) -> i32;
    }
    /// Process CPU time (kernel + user), in seconds, summed over threads.
    pub fn secs() -> f64 {
        let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
        if ok == 0 {
            return f64::NAN;
        }
        // FILETIME ticks are 100 ns.
        (k + u) as f64 * 1e-7
    }
}

#[cfg(not(windows))]
mod cpu {
    pub fn secs() -> f64 {
        // getrusage(RUSAGE_SELF) via libc would go here; the campaign box is
        // Windows, and a wrong number is worse than an absent one.
        f64::NAN
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let tier = args.next().unwrap_or_else(|| "n".into());
    let want: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(12);

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf();

    // Which corpus. Every tight probe so far read v2's 45 images while the
    // bench that suggested per-tier parity read v3's 450 — so corpus (and
    // therefore image-size distribution) is an axis the refutation had not
    // varied. FFAI_DIANA_CLIPS varies it.
    let clips_dir = std::env::var("FFAI_DIANA_CLIPS")
        .unwrap_or_else(|_| "corpora/clips/diana-coco".to_string());
    let mut clips: Vec<_> = std::fs::read_dir(root.join(&clips_dir))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    clips.sort();
    clips.truncate(want);
    if clips.is_empty() {
        return Err("no clips — run tools/diana_coco_corpus.py".into());
    }

    // Pre-decoded, exactly as the bench harness hands them to the engine, so
    // this probe measures the same span the ledger's latency column does.
    let images: Vec<_> =
        clips.iter().map(|p| ffai_media::load_image(p)).collect::<Result<Vec<_>, _>>()?;

    let engine = ffai_diana::engine::Yolo26::build(
        &tier,
        ffai_diana::image::Geometry::Rect,
        root.join("models"),
    );
    // MATCHED to the reference probe, not to our own defaults.
    //
    // D6 question 1 is "do both arms do identical work", and these two
    // probes did not: ours ran conf 0.25 while the reference ran conf 0.001
    // with max_det 100, and our head decodes the manifest's max_det of 300
    // regardless. Three times the top-k rows on our side, in a comparison
    // being published as a ~1.9x ratio. Overridable so the asymmetry can be
    // priced rather than argued about.
    let conf: f32 = std::env::var("FFAI_DIANA_CONF").ok().and_then(|v| v.parse().ok()).unwrap_or(0.001);
    let maxd: usize =
        std::env::var("FFAI_DIANA_MAXDET").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let opts = DetectOptions { confidence: conf, max_detections: maxd, ..Default::default() };
    // Three warm calls, matching tools/diana_d6_cpu.py. Symmetry is the
    // point: the reference needed three because torch selects kernels
    // lazily, and giving one arm more warmup than the other is a listed way
    // to manufacture a result.
    for _ in 0..3 {
        engine.detect(&images[0], &opts)?;
    }

    println!("tier {tier} · {} images from {clips_dir} · rayon threads {}", images.len(), rayon::current_num_threads());
    println!("{:>4}  {:>10} {:>10} {:>10}  {}", "img", "wall ms", "cpu ms", "occupancy", "dets");

    let (mut tw, mut tc) = (0f64, 0f64);
    let (mut walls, mut cpus) = (Vec::new(), Vec::new());
    for (i, image) in images.iter().enumerate() {
        let c0 = cpu::secs();
        let w0 = Instant::now();
        let out = engine.detect(image, &opts)?;
        let wall = w0.elapsed().as_secs_f64();
        let cpu = cpu::secs() - c0;
        tw += wall;
        tc += cpu;
        walls.push(wall);
        cpus.push(cpu);
        println!(
            "{i:>4}  {:>10.1} {:>10.1} {:>9.2}x  {}",
            wall * 1e3,
            cpu * 1e3,
            cpu / wall,
            out.detections.len()
        );
    }

    let occ = tc / tw;
    let cores = rayon::current_num_threads() as f64;
    println!();
    println!("total wall {:.1} ms · total cpu {:.1} ms", tw * 1e3, tc * 1e3);
    println!("mean per image: wall {:.1} ms · cpu {:.1} ms", tw * 1e3 / images.len() as f64, tc * 1e3 / images.len() as f64);
    // MEDIAN is what the ratio tool reads: one straggler in a short run
    // moves a mean by more than the effect being measured, and the
    // reference-side probe reports the median for the same reason.
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() { f64::NAN } else if v.len() % 2 == 1 { v[v.len() / 2] } else { (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0 }
    };
    println!("median per image: wall {:.1} ms · cpu {:.1} ms", med(&mut walls) * 1e3, med(&mut cpus) * 1e3);
    println!("OCCUPANCY {occ:.2}x of {cores:.0} threads = {:.0}% busy", 100.0 * occ / cores);
    // State the ceiling explicitly, so the next step is chosen on arithmetic
    // rather than on appetite: perfect occupancy is worth exactly cores/occ,
    // and NOTHING more. Any gap left after that is work, not scheduling.
    println!(
        "perfect occupancy would put per-image wall at {:.1} ms — a prize of {:.2}x. \
         Beyond that the fix is LESS WORK, not more threads.",
        tc * 1e3 / images.len() as f64 / cores,
        cores / occ
    );
    Ok(())
}
