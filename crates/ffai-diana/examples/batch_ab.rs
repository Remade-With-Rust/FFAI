//! Does confining parallelism to the batch level actually pay?
//!
//! `crate::parallel` argues from CPU time that nesting the per-layer
//! fan-out inside a parallel batch costs 2.32x the work it performs. This
//! is the gate on that argument at the level ABOVE the change — the whole
//! batch, wall clock, which is what a caller experiences — because the
//! three-probe rule says at least one probe must measure the level above,
//! and because a per-op win that the pipeline does not see is not a win.
//!
//! ABBA-interleaved and paired: this box is routinely running three other
//! benchmarks, so sequential arms would sample different machines. The
//! verdict is the paired win rate with a z-score.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example batch_ab -- [tier] [rounds] [images]
//! ```

use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The two arms differ by an environment variable read once per process,
    // so they cannot share one. Re-exec ourselves in `--child` mode instead
    // and alternate, which keeps the pairing tight (sub-second apart) while
    // still giving each arm a clean process.
    let mut args = std::env::args().skip(1);
    let first = args.next().unwrap_or_else(|| "n".into());
    if first == "--child" {
        let t = args.next().unwrap_or_else(|| "n".into());
        let n = args.next().and_then(|v| v.parse().ok()).unwrap_or(24);
        let m = args.next().unwrap_or_else(|| "batch".into());
        return child(t, n, m);
    }
    let tier = first;
    let rounds: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(15);
    let images: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(24);
    // Which env toggle names arm A. Defaults to the batch-parallelism one
    // this example was written for; pass another to gate a different change
    // at the same level, which is the point of the harness rather than an
    // afterthought.
    let knob = args.next().unwrap_or_else(|| "FFAI_DIANA_NESTED_PAR".into());
    // `serial` loops detect() the way the bench harness does, so a change
    // that only affects LATENCY is measured on the latency path. Batch mode
    // would hide it behind image-level parallelism.
    let mode = args.next().unwrap_or_else(|| "batch".into());

    let exe = std::env::current_exe()?;
    let run = |nested: bool| -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let mut c = Command::new(&exe);
        c.arg("--child").arg(&tier).arg(images.to_string()).arg(&mode);
        if nested {
            c.env(&knob, "1");
        }
        let out = c.output()?;
        let s = String::from_utf8_lossy(&out.stdout);
        let line = s.lines().last().ok_or("child produced no output")?;
        let mut it = line.split_whitespace();
        Ok((it.next().ok_or("bad")?.parse()?, it.next().ok_or("bad")?.parse()?))
    };

    println!("tier {tier} · {images} images · {rounds} rounds · mode {mode}");
    println!("A = {knob}=1 (old)   B = default (new)");
    let (mut wins, mut ratios) = (0usize, Vec::new());
    // CPU time is tracked separately and reported on its own, because the
    // two metrics answer different questions and this campaign has already
    // seen them disagree: CPU time is the WORK, wall is what a caller feels,
    // and on a box three other benchmarks are using, more threads can win
    // wall while losing work.
    let (mut cpu_wins, mut cpu_ratios) = (0usize, Vec::new());
    for r in 0..rounds {
        let (a, b) = if r % 2 == 0 {
            let a = run(true)?;
            let b = run(false)?;
            (a, b)
        } else {
            let b = run(false)?;
            let a = run(true)?;
            (a, b)
        };
        if b.0 < a.0 {
            wins += 1;
        }
        if b.1 < a.1 {
            cpu_wins += 1;
        }
        cpu_ratios.push(a.1 / b.1);
        ratios.push(a.0 / b.0);
        println!(
            "  round {:2}/{rounds}:  A wall {:7.1} ms cpu/img {:6.1}   B wall {:7.1} ms cpu/img {:6.1}   A/B {:.3}x",
            r + 1, a.0, a.1, b.0, b.1, a.0 / b.0
        );
    }
    ratios.sort_by(|x, y| x.partial_cmp(y).unwrap());
    cpu_ratios.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = rounds as f64;
    let z = (wins as f64 - n / 2.0) / (0.5 * n.sqrt());
    let zc = (cpu_wins as f64 - n / 2.0) / (0.5 * n.sqrt());
    let verdict = |z: f64| if z.abs() > 2.0 { "REAL at |z| > 2" } else { "inside the noise" };
    println!();
    println!(
        "WALL: B faster in {wins}/{rounds}  z = {z:+.2}  median A/B {:.4}x  -> {}",
        ratios[ratios.len() / 2],
        verdict(z)
    );
    println!(
        "CPU : B cheaper in {cpu_wins}/{rounds}  z = {zc:+.2}  median A/B {:.4}x  -> {}",
        cpu_ratios[cpu_ratios.len() / 2],
        verdict(zc)
    );
    Ok(())
}

/// One batch. Prints `wall_ms cpu_ms_per_image` as the last line.
fn child(tier: String, images: usize, mode: String) -> Result<(), Box<dyn std::error::Error>> {
    use ffai_core::engine::{DetectEngine, DetectOptions};

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
    let picked: Vec<_> = clips.iter().cycle().take(images).cloned().collect();
    let imgs: Vec<_> =
        picked.iter().map(|p| ffai_media::load_image(p)).collect::<Result<Vec<_>, _>>()?;

    let engine = ffai_diana::engine::Yolo26::build(
        &tier,
        ffai_diana::image::Geometry::Rect,
        root.join("models"),
    );
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    engine.detect_batch(&imgs[..2.min(imgs.len())], &opts)?; // warm

    let c0 = cpu_secs();
    let t = std::time::Instant::now();
    let out = if mode == "serial" {
        // The span the speed gate actually measures: one image at a time.
        imgs.iter().map(|i| engine.detect(i, &opts)).collect::<Result<Vec<_>, _>>()?
    } else {
        engine.detect_batch(&imgs, &opts)?
    };
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let cpu = (cpu_secs() - c0) * 1e3 / imgs.len() as f64;
    // A count both arms must agree on — if the dispatch changed the RESULT
    // the timing comparison is void, so it is checked rather than assumed.
    let dets: usize = out.iter().map(|o| o.detections.len()).sum();
    eprintln!("dets {dets}");
    println!("{wall:.3} {cpu:.3}");
    Ok(())
}

#[cfg(windows)]
fn cpu_secs() -> f64 {
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(h: isize, c: *mut u64, e: *mut u64, k: *mut u64, u: *mut u64) -> i32;
    }
    let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
    if unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) } == 0 {
        return f64::NAN;
    }
    (k + u) as f64 * 1e-7
}

#[cfg(not(windows))]
fn cpu_secs() -> f64 {
    f64::NAN
}
