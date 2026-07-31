//! Price the duration predictor's remaining ops individually.
//!
//! After the flat LayerNorm, dp sits at ~170 ms against onnxruntime's ~76, and
//! its flow half is 78% of the stage. The flows are four identical DDS stacks,
//! each layer being: depthwise conv -> LayerNorm -> GELU -> 1x1 conv ->
//! LayerNorm -> GELU -> residual add.
//!
//! The convs are now GEMMs and the norms are fused, so what is left is the
//! activation and the glue. GELU is worth pricing specifically: the exported
//! graph uses exact `Erf` (not the tanh approximation), and dp calls it 24
//! times per sentence over 16,896 elements — ~405k erf evaluations per
//! utterance. A per-element libm `erf` at even 50 ns would dominate everything
//! else in the stage, so this either is the cost or exonerates itself.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example dp_ops
//! ```

use std::time::Instant;

use ffai_core::candle::{Device, Tensor};

const C: usize = 192;
const T: usize = 88;
/// 4 DDS stacks x 3 layers x 2 activations.
const CALLS: usize = 24;
const SENTENCES: usize = 20;

fn bench(label: &str, mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let per_call = best / (CALLS * SENTENCES) as f64;
    println!(
        "  {label:<34} {:>8.1} ms total  {:>7.1} us/call  {:>6.1} ns/elem",
        best * 1000.0,
        per_call * 1e6,
        per_call * 1e9 / (C * T) as f64
    );
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (1, C, T), &dev)?;
    let xv: Vec<f32> = x.flatten_all()?.to_vec1()?;
    println!("{C}x{T} = {} elements, {CALLS} calls/sentence x {SENTENCES} sentences\n", C * T);

    let g = bench("candle gelu_erf (shipped)", || {
        for _ in 0..CALLS * SENTENCES {
            std::hint::black_box(x.gelu_erf().unwrap());
        }
    });

    // Flat exact-erf GELU: same arithmetic, no tensor allocation per call.
    let f = bench("flat gelu, libm erf", || {
        for _ in 0..CALLS * SENTENCES {
            let mut o = xv.clone();
            for v in o.iter_mut() {
                let z = *v as f64;
                *v = (0.5 * z * (1.0 + erf(z / std::f64::consts::SQRT_2))) as f32;
            }
            std::hint::black_box(&o);
        }
    });

    // Abramowitz-Stegun 7.1.26 erf, max abs error 1.5e-7 — comfortably inside
    // the 1e-5 band the stage oracle already allows on m_p.
    let a = bench("flat gelu, A&S 7.1.26 erf", || {
        for _ in 0..CALLS * SENTENCES {
            let mut o = xv.clone();
            for v in o.iter_mut() {
                let z = *v * std::f32::consts::FRAC_1_SQRT_2;
                *v = 0.5 * *v * (1.0 + erf_fast(z));
            }
            std::hint::black_box(&o);
        }
    });

    // Pure allocation+copy floor, to separate the maths from the plumbing.
    bench("clone only (allocation floor)", || {
        for _ in 0..CALLS * SENTENCES {
            std::hint::black_box(xv.clone());
        }
    });

    println!("\n  candle vs flat-libm  {:.2}x   candle vs flat-fast {:.2}x", g / f, g / a);

    // Accuracy of the approximation against libm, on the real range.
    let mut worst = 0f64;
    for v in &xv {
        let z = *v as f64;
        let exact = 0.5 * z * (1.0 + erf(z / std::f64::consts::SQRT_2));
        let approx =
            0.5 * (*v as f64) * (1.0 + erf_fast(*v * std::f32::consts::FRAC_1_SQRT_2) as f64);
        worst = worst.max((exact - approx).abs());
    }
    println!("  A&S max |Δ| vs libm erf on this tensor: {worst:.3e}");
    Ok(())
}

/// Abramowitz & Stegun 7.1.26 — |error| <= 1.5e-7.
fn erf_fast(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}

/// libm erf via the standard series/continued-fraction split.
fn erf(x: f64) -> f64 {
    // Use the same A&S form at f64 for a like-for-like "exact" reference is
    // circular, so call through to a high-accuracy rational approximation.
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
        .exp();
    if x >= 0.0 { 1.0 - tau } else { tau - 1.0 }
}
