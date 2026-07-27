//! Interleaved A/B — resolve a delta on a machine that will not hold still.
//!
//! This box's noise floor swings between ~4 % and ~245 % over minutes, so
//! "measure A, then measure B" is worthless: the arms sample different
//! machines. Waiting for quiet is not a plan either, because you cannot know
//! you got it.
//!
//! The fix is to **alternate A and B within one process**, many times, so
//! both arms experience the same drift, then compare paired samples. The
//! verdict is reported three ways, strongest first:
//!
//! - **paired win rate** — in how many of N head-to-head rounds did A beat B?
//!   Robust to drift, because each round is a fair coin toss under identical
//!   conditions. This is the number to trust.
//! - **median ratio** — the central estimate of the effect size.
//! - **range overlap** — do the min/max intervals overlap at all? Non-
//!   overlapping ranges are a hard verdict (codec-analyzer's bar).
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example interleaved_ab
//! ```

use std::time::Instant;

use ffai_core::candle::{DType, Device, Tensor};

struct Verdict {
    a_wins: usize,
    rounds: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

/// Alternate the two closures `rounds` times, timing each in turn.
fn interleave(rounds: usize, mut a: impl FnMut() -> f64, mut b: impl FnMut() -> f64) -> Verdict {
    // Warm both before counting, so neither pays first-touch for the other.
    a();
    b();
    let (mut va, mut vb, mut wins) = (Vec::new(), Vec::new(), 0);
    for i in 0..rounds {
        // Alternate which arm goes first, so a systematic "second is warmer"
        // effect cancels instead of accumulating into one arm.
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
    Verdict { a_wins: wins, rounds, a: va, b: vb }
}

fn report(name: &str, label_a: &str, label_b: &str, mut v: Verdict) {
    v.a.sort_by(f64::total_cmp);
    v.b.sort_by(f64::total_cmp);
    let (ma, mb) = (v.a[v.a.len() / 2], v.b[v.b.len() / 2]);
    let overlap = v.a[0] <= *v.b.last().unwrap() && v.b[0] <= *v.a.last().unwrap();
    let rate = v.a_wins as f64 / v.rounds as f64 * 100.0;

    println!("\n{name}");
    println!(
        "  {label_a:<22} med {ma:7.3} ms   [{:.3} .. {:.3}]",
        v.a[0],
        v.a.last().unwrap()
    );
    println!(
        "  {label_b:<22} med {mb:7.3} ms   [{:.3} .. {:.3}]",
        v.b[0],
        v.b.last().unwrap()
    );
    println!(
        "  paired: {label_a} won {}/{} rounds ({rate:.0} %) · median ratio {:.2}x · ranges {}",
        v.a_wins,
        v.rounds,
        mb / ma,
        if overlap { "OVERLAP" } else { "DISJOINT" }
    );
    // A fair coin over N rounds lands within ~3 sigma of 50 %; outside that,
    // the paired result is real regardless of what the medians drifted to.
    let sigma = 0.5 * (v.rounds as f64).sqrt();
    let z = (v.a_wins as f64 - 0.5 * v.rounds as f64) / sigma;
    println!(
        "  VERDICT: {}",
        if z.abs() < 2.0 {
            format!("INCONCLUSIVE (z={z:+.1}) — no reliable difference")
        } else if z > 0.0 {
            format!("{label_a} FASTER (z={z:+.1})")
        } else {
            format!("{label_b} FASTER (z={z:+.1})")
        }
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let rounds = 41;
    println!("interleaved A/B, {rounds} paired rounds per test");

    // UNRESOLVED #1 (mission plan section 6.13): is f16 really faster than f32
    // for the vocabulary projection? The adaptive selector currently ships f16
    // on the strength of a single-run 1.56x.
    {
        let x32 = Tensor::randn(0f32, 1., (1, 384), &dev)?;
        let w32 = Tensor::randn(0f32, 1., (384, 51864), &dev)?;
        let x16 = x32.to_dtype(DType::F16)?;
        let w16 = w32.to_dtype(DType::F16)?;
        let v = interleave(
            rounds,
            || {
                let t = Instant::now();
                std::hint::black_box(x16.matmul(&w16).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
            || {
                let t = Instant::now();
                std::hint::black_box(x32.matmul(&w32).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
        );
        report("vocabulary projection (1x384 @ 384x51864)", "f16", "f32", v);
    }

    // UNRESOLVED #2 (section 6.16): the f16 cross-attention K/V cache was
    // reverted on a 12 % delta measured once. Both matmuls, one round each.
    {
        let (h, kv, hd) = (6usize, 1500usize, 64usize);
        let q32 = Tensor::randn(0f32, 1., (1, h, 1, hd), &dev)?.contiguous()?;
        let k32 = Tensor::randn(0f32, 1., (1, h, hd, kv), &dev)?.contiguous()?;
        let q16 = q32.to_dtype(DType::F16)?;
        let k16 = k32.to_dtype(DType::F16)?;
        let v = interleave(
            rounds,
            || {
                let t = Instant::now();
                std::hint::black_box(q16.matmul(&k16).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
            || {
                let t = Instant::now();
                std::hint::black_box(q32.matmul(&k32).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
        );
        report("cross-attn scores q@k", "f16 K/V", "f32 K/V", v);
    }

    // The op that killed that idea: candle's softmax at each dtype.
    {
        let s32 = Tensor::randn(0f32, 1., (1, 6, 1, 1500), &dev)?;
        let s16 = s32.to_dtype(DType::F16)?;
        let v = interleave(
            rounds,
            || {
                let t = Instant::now();
                std::hint::black_box(candle_nn::ops::softmax_last_dim(&s16).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
            || {
                let t = Instant::now();
                std::hint::black_box(candle_nn::ops::softmax_last_dim(&s32).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
        );
        report("softmax over scores", "f16", "f32", v);
    }

    // THE DECIDING TEST for section 6.16: the whole cross-attention chain.
    // The pieces disagree — f16 wins both matmuls and loses the softmax
    // between them — so only the chain says which way the sum falls, with
    // the dtype conversions that a real implementation must pay included.
    {
        let (h, kv, hd) = (6usize, 1500usize, 64usize);
        let q32 = Tensor::randn(0f32, 1., (1, h, 1, hd), &dev)?.contiguous()?;
        let k32 = Tensor::randn(0f32, 1., (1, h, hd, kv), &dev)?.contiguous()?;
        let v32 = Tensor::randn(0f32, 1., (1, h, kv, hd), &dev)?.contiguous()?;
        let q16 = q32.to_dtype(DType::F16)?;
        let k16 = k32.to_dtype(DType::F16)?;
        let v16 = v32.to_dtype(DType::F16)?;
        let v = interleave(
            rounds,
            || {
                let t = Instant::now();
                // f16 cache, f32 softmax, with both conversions paid.
                let qk = q16.matmul(&k16).unwrap();
                let w = candle_nn::ops::softmax_last_dim(&qk.to_dtype(DType::F32).unwrap()).unwrap();
                let ctx = w.to_dtype(DType::F16).unwrap().matmul(&v16).unwrap();
                std::hint::black_box(ctx);
                t.elapsed().as_secs_f64() * 1e3
            },
            || {
                let t = Instant::now();
                let qk = q32.matmul(&k32).unwrap();
                let w = candle_nn::ops::softmax_last_dim(&qk).unwrap();
                std::hint::black_box(w.matmul(&v32).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
        );
        report("FULL cross-attn chain", "f16 K/V + conversions", "all f32", v);
    }

    // UNRESOLVED #3 (section 6.10): the shipped GELU against candle's, to
    // confirm the campaign's headline win survives a paired test.
    {
        let a = Tensor::randn(0f32, 1., (1500, 1536), &dev)?;
        let v = interleave(
            rounds,
            || {
                let t = Instant::now();
                std::hint::black_box(ffai_mercury::asr::text_decoder::fast_gelu(&a).unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
            || {
                let t = Instant::now();
                std::hint::black_box(a.gelu().unwrap());
                t.elapsed().as_secs_f64() * 1e3
            },
        );
        report("GELU (encoder activation)", "ours (Pade)", "candle (tanh)", v);
    }

    Ok(())
}
