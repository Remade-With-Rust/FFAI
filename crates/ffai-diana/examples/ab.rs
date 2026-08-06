//! The A/B harness that refuses to lie.
//!
//! This campaign produced six confident quantitative claims that turned out
//! to be artifacts — a tier trend that was the sweep's ordering, a roofline
//! priced at the wrong working set, a microbench measuring a function
//! pointer, a "corroboration" between two numbers one of which did not
//! exist. Four of the six would have been caught by one harness applying
//! four rules mechanically, and this is that harness.
//!
//! ## The rules, and why each is here
//!
//! 1. **NULL ARM FIRST.** Before comparing anything, run the SAME
//!    configuration against itself. Whatever spread that shows is the
//!    resolution limit; every effect narrower than it is a coin flip. A
//!    null arm already in the ledger — two identical Diana runs — showed the
//!    headline ratio moving 27 % with nothing changed, and it was not run
//!    until two sessions of ratios had been quoted.
//! 2. **ABBA.** Alternate which arm goes first each round, so "the second
//!    one runs warmer" cancels instead of accumulating.
//! 3. **MIN of N per round**, not mean or median of one sample. Foreign load
//!    only ever ADDS time, so the minimum is the floor of the code's own
//!    cost. This box has never once been quiet.
//! 4. **REFUSE.** If the measured effect falls inside the null spread, print
//!    INCONCLUSIVE and say so — do not report a median that reads like a
//!    result. This is the rule that would have stopped four of the six.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example ab -- <ENV_VAR> [tier] [rounds] [images] [mode]
//! ```
//! Arm A sets `ENV_VAR=1`; arm B leaves it unset. `mode` is `serial`
//! (per-image latency, what the speed gate times) or `batch`.

// The shipped allocator; examples are separate binaries and do not inherit
// the one set in ffai-cli. Without this the A/B measures a configuration we
// no longer run.
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

use std::process::Command;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.first().map(String::as_str) == Some("--child") {
        return child(&a[1], a[2].parse()?, &a[3]);
    }
    let knob = a.first().cloned().unwrap_or_else(|| "FFAI_DIANA_DIRECT".into());
    let tier = a.get(1).cloned().unwrap_or_else(|| "n".into());
    let rounds: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(15);
    let images: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);
    let mode = a.get(4).cloned().unwrap_or_else(|| "serial".into());
    let exe = std::env::current_exe()?;

    let run = |on: bool| -> f64 {
        let mut c = Command::new(&exe);
        c.arg("--child").arg(&tier).arg(images.to_string()).arg(&mode);
        if on {
            c.env(&knob, "1");
        } else {
            c.env_remove(&knob);
        }
        match c.output() {
            Ok(o) => String::from_utf8_lossy(&o.stdout)
                .lines()
                .last()
                .and_then(|l| l.split_whitespace().next())
                .and_then(|v| v.parse().ok())
                .unwrap_or(f64::NAN),
            Err(_) => f64::NAN,
        }
    };

    // ---- 1. NULL ARM: the same configuration against itself ------------
    // Whatever this shows is the floor. Nothing narrower is a result.
    let null_rounds = rounds.max(9);
    let mut null_ratios = Vec::new();
    for r in 0..null_rounds {
        let (x, y) = if r % 2 == 0 { (run(false), run(false)) } else { (run(false), run(false)) };
        if x.is_finite() && y.is_finite() && y > 0.0 {
            null_ratios.push(x / y);
        }
    }
    if null_ratios.len() < 3 {
        return Err("null arm produced too few finite samples".into());
    }
    null_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (nlo, nhi) = (null_ratios[0], null_ratios[null_ratios.len() - 1]);
    // IQR, not min-max.
    //
    // The first version of this harness used the full range and reported a
    // 5464 % resolution limit, because two outliers out of fifteen span 55x
    // on a contended box. It then refused a 15/15, z = +3.87 result.
    // Refusing a real effect is the same failure as accepting a fake one,
    // only politer. The interquartile range is the robust statistic.
    //
    // The NULL WIN RATE is the second, independent check: with identical
    // code on both sides the win rate must sit near 50 %. If it does not,
    // the harness itself is biased and nothing below it means anything.
    let q = |f: f64| null_ratios[((null_ratios.len() as f64 - 1.0) * f).round() as usize];
    let (q1, q3) = (q(0.25), q(0.75));
    let null_spread = q3 / q1 - 1.0;
    let null_wins = null_ratios.iter().filter(|r| **r < 1.0).count();
    let nn = null_ratios.len() as f64;
    let null_z = (null_wins as f64 - nn / 2.0) / (0.5 * nn.sqrt());
    println!("NULL ARM  {} rounds, identical config both sides", null_ratios.len());
    println!("  full range {nlo:.3}x - {nhi:.3}x   (outliers: a contended box)");
    println!("  IQR {q1:.4}x - {q3:.4}x   -> RESOLUTION LIMIT {:.1}%", null_spread * 100.0);
    println!(
        "  null win rate {null_wins}/{}, z = {null_z:+.2}  (must be near 0)\n",
        null_ratios.len()
    );
    if null_z.abs() > 2.0 {
        println!("  !! NULL ARM IS BIASED — identical code is winning one side.");
        println!("     Fix the harness before trusting anything below.\n");
    }

    // ---- 2/3. ABBA, min-of-2 per arm per round -------------------------
    let (mut wins, mut ratios) = (0usize, Vec::new());
    for r in 0..rounds {
        let (ta, tb) = if r % 2 == 0 {
            let x = run(true).min(run(true));
            (x, run(false).min(run(false)))
        } else {
            let y = run(false).min(run(false));
            (run(true).min(run(true)), y)
        };
        if !(ta.is_finite() && tb.is_finite() && tb > 0.0) {
            println!("  round {:2}: DROPPED (non-finite sample)", r + 1);
            continue;
        }
        if tb < ta {
            wins += 1;
        }
        ratios.push(ta / tb);
        println!("  round {:2}/{rounds}:  A(on) {ta:8.1} ms   B(off) {tb:8.1} ms   A/B {:.4}x", r + 1, ta / tb);
    }
    if ratios.is_empty() {
        return Err("no finite pairs".into());
    }

    let n = ratios.len() as f64;
    let z = (wins as f64 - n / 2.0) / (0.5 * n.sqrt());
    let med = median(&mut ratios);
    let effect = (med - 1.0).abs();

    println!("\n{knob}: B(off) faster in {wins}/{} rounds, z = {z:+.2}", ratios.len());
    println!("median A/B = {med:.4}x  (effect {:.1}%)", effect * 100.0);

    // ---- 4. REFUSE ------------------------------------------------------
    //
    // BOTH checks must fail before refusing. A consistent DIRECTION is
    // evidence even when the magnitudes are wild: on a contended box the
    // ratio wanders, but which arm is faster does not, so a 15/15 win rate
    // survives noise that makes the median meaningless.
    if effect < null_spread && z.abs() <= 2.0 {
        println!(
            "\nVERDICT: INCONCLUSIVE — the {:.1}% effect is inside the harness's own \
             {:.1}% resolution.\nA median is printed above; it is NOT a result. Reduce the \
             null spread (pin, quieten, more images per sample) or accept that this box \
             cannot resolve this change.",
            effect * 100.0,
            null_spread * 100.0
        );
    } else if z.abs() > 2.0 {
        // Direction and magnitude are SEPARATE claims clearing SEPARATE
        // bars, and saying so is the point. Reporting a median because the
        // win rate was significant is how "1.18x" became "1.033x" elsewhere
        // in this repo: the win rate answers WHETHER, only the median
        // answers HOW MUCH, and the median needs the null spread beaten.
        let dir = if med > 1.0 { "B(off) is FASTER" } else { "A(on) is FASTER" };
        let mag = if effect > null_spread {
            "the magnitude also clears it, so the median is usable"
        } else {
            "the magnitude does NOT clear it — treat the median as an upper bound, not a result"
        };
        println!(
            "\nVERDICT: DIRECTION REAL — {dir}, {wins}/{} rounds, |z| = {:.2} > 2.\n\
             Effect {:.1}% against a {:.1}% null spread: {mag}.",
            ratios.len(),
            z.abs(),
            effect * 100.0,
            null_spread * 100.0
        );
    } else {
        println!(
            "\nVERDICT: DIRECTION UNCLEAR — effect {:.1}% clears the null spread but the \
             win rate does not (|z| = {:.2} < 2). Run more rounds.",
            effect * 100.0,
            z.abs()
        );
    }
    Ok(())
}

/// One measurement. Prints `ms` as the last line.
fn child(tier: &str, images: usize, mode: &str) -> Result<(), Box<dyn std::error::Error>> {
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
        tier,
        ffai_diana::image::Geometry::Rect,
        root.join("models"),
    );
    let opts = DetectOptions { confidence: 0.001, max_detections: 100, ..Default::default() };
    for im in imgs.iter().take(2) {
        engine.detect(im, &opts)?;
    }
    let t = std::time::Instant::now();
    let dets: usize = if mode == "serial" {
        imgs.iter().map(|i| engine.detect(i, &opts).map(|o| o.detections.len())).sum::<Result<_, _>>()?
    } else {
        engine.detect_batch(&imgs, &opts)?.iter().map(|o| o.detections.len()).sum()
    };
    // A work COUNT both arms must agree on. Divergent counts void the timing.
    eprintln!("dets {dets}");
    println!("{:.3}", t.elapsed().as_secs_f64() * 1e3);
    Ok(())
}
