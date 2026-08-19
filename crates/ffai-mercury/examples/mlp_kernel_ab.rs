//! Probe #2 for the int8-decoder-MLP question: the **op level**, at the
//! decoder MLP's exact shapes.
//!
//! Probe #1 (`pipeline_ab --test mlp_int8`, the level *above*) came back
//! INCONCLUSIVE, on a lever whose arithmetic prize was 3.1–4.7 % — comfortably
//! above the harness's 2 % floor. Two readings survive that, and they need
//! opposite responses:
//!
//! - **the kernel is not actually faster here**, in which case the lever is
//!   genuinely dead and the pipeline result is fully explained; or
//! - **the kernel is faster and something downstream absorbs it**, in which
//!   case the target is the absorber, not the kernel.
//!
//! `codec-six-whys-unknowns` says to vary the axis that could flip the answer.
//! The axis here is **level**, and this is the probe one level down. It also
//! varies **shape**: the campaign's int8 win (4.34×, z = +6.4) was measured on
//! the *vocabulary* projection — 384 → 51864, streaming 40 MB — while the MLP
//! is 384 → 1536 and 1536 → 384, two orders of magnitude less traffic per
//! call. A refutation expires when its baseline moves; a *confirmation* expires
//! when its shape does.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example mlp_kernel_ab
//! ```

use std::time::Instant;

use ffai_core::candle::{DType, Device, Tensor};
use ffai_mercury::asr::f16_gemv::F16Gemv;
use ffai_mercury::asr::vocab_int8::Int8Vocab;

struct Verdict {
    a_wins: usize,
    rounds: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

fn interleave(rounds: usize, mut a: impl FnMut() -> f64, mut b: impl FnMut() -> f64) -> Verdict {
    a();
    b();
    let (mut va, mut vb, mut wins) = (Vec::new(), Vec::new(), 0);
    for i in 0..rounds {
        let (ta, tb) = if i % 2 == 0 {
            let ta = a();
            (ta, b())
        } else {
            let tb = b();
            (a(), tb)
        };
        if ta < tb {
            wins += 1;
        }
        va.push(ta);
        vb.push(tb);
    }
    Verdict {
        a_wins: wins,
        rounds,
        a: va,
        b: vb,
    }
}

fn report(name: &str, la: &str, lb: &str, mut v: Verdict) {
    v.a.sort_by(f64::total_cmp);
    v.b.sort_by(f64::total_cmp);
    let (ma, mb) = (v.a[v.a.len() / 2], v.b[v.b.len() / 2]);
    let z = (v.a_wins as f64 - 0.5 * v.rounds as f64) / (0.5 * (v.rounds as f64).sqrt());
    println!("\n{name}");
    println!("  {la:<20} med {ma:8.1} us", ma = ma * 1e3);
    println!("  {lb:<20} med {mb:8.1} us", mb = mb * 1e3);
    println!(
        "  paired: {la} won {}/{} · ratio {:.3}x · {}",
        v.a_wins,
        v.rounds,
        mb / ma,
        if z.abs() < 2.0 {
            format!("INCONCLUSIVE (z={z:+.1})")
        } else if z > 0.0 {
            format!("{la} FASTER (z={z:+.1})")
        } else {
            format!("{lb} FASTER (z={z:+.1})")
        }
    );
}

/// One (out, in) weight matrix through both kernels, plus candle's f32 matmul
/// as the thing they both have to beat.
fn probe(
    dev: &Device,
    rounds: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let w = Tensor::randn(0f32, 0.05, (out_dim, in_dim), dev)?;
    let x = Tensor::randn(0f32, 1.0, (1, in_dim), dev)?;

    let f16 = F16Gemv::new(&w)?.ok_or("f16 kernel declined this shape")?;
    let i8k = Int8Vocab::new(&w)?.ok_or("int8 kernel declined this shape")?;

    // Same answer? The int8 path is a quantization, so this is a magnitude
    // check, not an equality — but a wildly wrong result would mean the
    // timings below are of the wrong computation.
    let (yf, yi) = (f16.forward(&x)?, i8k.forward(&x)?);
    let d = (yf.to_dtype(DType::F32)? - yi.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let scale = yf.abs()?.max_all()?.to_scalar::<f32>()?;
    println!(
        "\n=== ({out_dim} x {in_dim}), m=1 · max|f16-int8| {d:.4} ({:.2} % of peak)",
        d / scale * 100.0
    );

    let wt = w.t()?.contiguous()?;
    report(
        &format!("  int8 vs f16 GEMV ({out_dim}x{in_dim})"),
        "int8",
        "f16",
        interleave(
            rounds,
            || {
                let t = Instant::now();
                std::hint::black_box(i8k.forward(&x).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
            || {
                let t = Instant::now();
                std::hint::black_box(f16.forward(&x).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
        ),
    );
    report(
        &format!("  int8 vs candle f32 matmul ({out_dim}x{in_dim})"),
        "int8",
        "candle f32",
        interleave(
            rounds,
            || {
                let t = Instant::now();
                std::hint::black_box(i8k.forward(&x).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
            || {
                let t = Instant::now();
                std::hint::black_box(x.matmul(&wt).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
        ),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let rounds = 201; // ops this small are noisy; the paired stat needs the N

    println!("decoder MLP kernels, tiny.en shapes, m=1, {rounds} paired rounds");
    println!("expected pipeline prize if int8 wins the MLP: 9.4 % share x (1-1/speedup)");

    // fc1: (1,384) @ (384,1536)  -> weight stored (out=1536, in=384)
    probe(&dev, rounds, 1536, 384)?;
    // fc2: (1,1536) @ (1536,384) -> weight stored (out=384, in=1536)
    probe(&dev, rounds, 384, 1536)?;
    // The shape int8 was proven on, for calibration: the vocabulary projection.
    probe(&dev, rounds, 51864, 384)?;
    Ok(())
}
