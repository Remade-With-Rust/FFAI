//! Is the bounds-check hoist in `resize_bilinear_u8` worth anything?
//!
//! The end-to-end OCR arm could not answer this: a page is dominated by
//! detector and recognizer inference, and a null arm (the same binary against
//! itself) measured 0.49 s of median spread against a 0.16 s before/after
//! gap. When the noise is 3x the effect, the instrument is the wrong one.
//!
//! This is the right altitude: the kernel alone, both versions compiled into
//! ONE binary so there is no build-to-build variation, ABBA-interleaved so
//! monotonic drift cancels, and gated byte-identical before any timing is
//! reported — if the two disagree the number means nothing.
//!
//! `cargo run --release -p ffai-carmenta --example resize_bce_worth`

use std::hint::black_box;
use std::time::Instant;

/// The pre-change kernel, verbatim: `f32::floor` and whole-slice indexing, so
/// every tap carries a `panic_bounds_check`.
fn resize_old(
    src: &[u8],
    bpp: usize,
    c: usize,
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Vec<f32> {
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    let at = |y: usize, x: usize| f32::from(src[(y * sw + x) * bpp + c]);
    let mut out = vec![0f32; dw * dh];
    for oy in 0..dh {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - fy.floor();
        for ox in 0..dw {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - fx.floor();
            let top = at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx;
            let bot = at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx;
            out[oy * dw + ox] = top * (1.0 - wy) + bot * wy;
        }
    }
    out
}

fn main() {
    // A page-shaped source and the detector's real output size: a 1700x2200
    // scan entering `mobiledet_input` at the 1280 short-side ceiling.
    let (sw, sh) = (1700usize, 2200usize);
    let (dw, dh) = (1280usize, 1656usize);
    let bpp = 3usize;

    // Deterministic pixels — a fixed LCG, so the run is reproducible and the
    // data is not a constant the optimiser can fold away.
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let src: Vec<u8> = (0..sw * sh * bpp)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        })
        .collect();

    // ---- CORRECTNESS GATE, before any timing is taken ----
    for c in 0..bpp {
        let a = resize_old(&src, bpp, c, sw, sh, dw, dh);
        let b = ffai_carmenta::image::resize_bilinear_u8(&src, bpp, c, sw, sh, dw, dh);
        assert_eq!(a.len(), b.len(), "length differs on channel {c}");
        let bad = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            bad,
            0,
            "channel {c}: {bad} of {} samples differ — the timing below would be meaningless",
            a.len()
        );
    }
    println!("byte-identical on all {bpp} channels\n");

    let time = |f: &dyn Fn() -> Vec<f32>| {
        let t = Instant::now();
        black_box(f());
        t.elapsed().as_secs_f64() * 1e3
    };
    let old = || resize_old(&src, bpp, 0, sw, sh, dw, dh);
    let new = || ffai_carmenta::image::resize_bilinear_u8(&src, bpp, 0, sw, sh, dw, dh);

    // Warm-up, untimed: first touch of `out` faults in 8 MB.
    black_box(old());
    black_box(new());

    const ROUNDS: usize = 15;
    let (mut vo, mut vn) = (Vec::new(), Vec::new());
    for r in 0..ROUNDS {
        // ABBA: order flips each round so any monotonic drift (thermal, or
        // another process ramping) lands on both arms equally.
        if r % 2 == 0 {
            vo.push(time(&old));
            vn.push(time(&new));
        } else {
            vn.push(time(&new));
            vo.push(time(&old));
        }
    }
    // The paired sign test FIRST — `stat` sorts in place, and sorting would
    // silently destroy the pairing this test depends on.
    let wins = vo.iter().zip(&vn).filter(|(o, n)| n < o).count();

    let stat = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (v[0], v[v.len() / 2], mean)
    };
    let (omin, omed, omean) = stat(&mut vo);
    let (nmin, nmed, nmean) = stat(&mut vn);
    println!("{sw}x{sh} -> {dw}x{dh}, one channel, {ROUNDS} ABBA rounds\n");
    println!(
        "{:<10} {:>9} {:>9} {:>9}",
        "arm", "min ms", "med ms", "mean ms"
    );
    println!("{:<10} {:>9.2} {:>9.2} {:>9.2}", "old", omin, omed, omean);
    println!("{:<10} {:>9.2} {:>9.2} {:>9.2}", "new", nmin, nmed, nmean);
    println!(
        "\nspeedup on min {:.3}x  median {:.3}x",
        omin / nmin,
        omed / nmed
    );

    // Per-round, which arm won: immune to drift in a way a ratio of
    // aggregates is not.
    println!("new faster in {wins}/{ROUNDS} paired rounds");
}
