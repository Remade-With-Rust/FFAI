//! Ceiling probe for the encoder attention PREP glue — priced brick by brick
//! before any of it is built.
//!
//! The anatomy bench says the three attention primitives are at or near what
//! their shapes allow (`scores q@kT` 205 GFLOP/s against a 229 GFLOP/s shape
//! ceiling = 89 %). So the encode gap is not in the matmuls. But reconciling
//! the isolated primitives against the in-context profiler shows the opposite
//! signature next door:
//!
//! | op | isolated | in context | ratio |
//! |---|---:|---:|---:|
//! | `ea prep` | 4.95 ms | 11 ms | **2.2x** |
//! | `ea merge` | 1.92 ms | 6 ms | **3.1x** |
//!
//! That is `codec-analyzer`'s "in-context ns/call >> kernel ns/call" — glue,
//! not arithmetic. The shipped prep is three separate materializations of a
//! ~2.3 MB tensor per layer:
//!
//! ```ignore
//! (split_heads(&q)? * (scale * scale))?.contiguous()?,  // multiply THEN copy
//! split_heads(&k)?.transpose(2, 3)?.contiguous()?,      // strided gather
//! split_heads(&v)?.contiguous()?,                       // copy
//! ```
//!
//! Three candidate bricks, each priced separately here:
//!
//! 1. **Fold `scale` into the q weight at load.** It depends only on
//!    `head_dim` — a signal-independent constant sitting in the per-call path,
//!    the first question `codec-eliminate-redundancy` says to ask of any hot
//!    loop.
//! 2. **Produce K already transposed.** The kernel wants `(heads, HD, seq)`;
//!    the projection writes `(seq, d)`, so every layer pays a strided gather.
//!    `k^T = w_k^T @ x^T`, and `x.t()` is a view — if candle's matmul accepts
//!    a strided operand this is free. NOTE: `OPEN.md` §2 refuted fusing ALL of
//!    q/k/v into one transposed projection, because the kernel wants q in
//!    `(seq, HD)`. That refutation does not cover K ALONE, which is the one
//!    tensor the kernel genuinely wants transposed.
//! 3. **A projection that writes the head-split layout directly**, removing
//!    the `.contiguous()` on q and v.
//!
//! The rule this probe obeys (`codec-optimize`): *a ceiling probe must remove
//! exactly the cost the brick would remove, or it prices a brick you aren't
//! building.* Each arm below is the real op sequence with one specific cost
//! deleted — not an approximation of it.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example enc_prep_ceiling
//! ```

use std::time::Instant;

use ffai_core::candle::{Device, Tensor};

fn best_of(n: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::MAX;
    for _ in 0..n {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (seq, d, heads, layers) = (1500usize, 384usize, 6usize, 4usize);
    let hd = d / heads;
    let scale = ((hd as f64).powf(-0.25)) as f64;
    let s2 = scale * scale;

    // The layer input (post layer-norm) and one projection weight, in the
    // orientation candle's `Linear` actually multiplies.
    let x = Tensor::randn(0f32, 1., (1, seq, d), &dev)?;
    let x2 = x.reshape((seq, d))?;
    let w = Tensor::randn(0f32, 1., (d, d), &dev)?;

    let split = |t: &Tensor| -> ffai_core::candle::Result<Tensor> {
        t.reshape((1, seq, heads, hd))?.transpose(1, 2)
    };

    println!("encoder attention prep, per LAYER (x{layers} per pass)");
    println!("seq={seq} d={d} heads={heads} head_dim={hd}\n");

    // ---- the projection itself: the irreducible floor ----
    let proj = best_of(20, || {
        std::hint::black_box(x2.matmul(&w).unwrap());
    });
    println!("  projection (1500,384)@(384,384)            {:6.3} ms   <- irreducible", proj * 1e3);

    let projected = x2.matmul(&w)?.reshape((1, seq, d))?;

    // ---- Q: shipped vs scale folded into the weight ----
    let q_shipped = best_of(20, || {
        let t = (split(&projected).unwrap() * s2).unwrap().contiguous().unwrap();
        std::hint::black_box(t);
    });
    let q_folded = best_of(20, || {
        // Brick 1: the scale rides in the weight, so prep is split+copy only.
        let t = split(&projected).unwrap().contiguous().unwrap();
        std::hint::black_box(t);
    });
    println!("\n  Q  shipped  (split * scale).contiguous()   {:6.3} ms", q_shipped * 1e3);
    println!("  Q  brick 1  scale folded into weight       {:6.3} ms   saves {:6.3} ms",
        q_folded * 1e3, (q_shipped - q_folded) * 1e3);

    // ---- K: shipped strided gather vs produced already-transposed ----
    let k_shipped = best_of(20, || {
        let t = split(&projected).unwrap().transpose(2, 3).unwrap().contiguous().unwrap();
        std::hint::black_box(t);
    });
    // Brick 2: k^T = w^T @ x^T. `x2.t()` is a VIEW; if candle's matmul takes a
    // strided operand, the transposed result costs one matmul and a free
    // reshape, and the strided gather disappears entirely. This arm therefore
    // includes the projection, so it is compared against projection + gather.
    let wt = w.t()?.contiguous()?;
    let k_transposed = best_of(20, || {
        let kt = wt.matmul(&x2.t().unwrap()).unwrap(); // (d, seq)
        let t = kt.reshape((1, heads, hd, seq)).unwrap();
        std::hint::black_box(t);
    });
    println!("\n  K  shipped  split.transpose(2,3).contig    {:6.3} ms   (+ {:6.3} projection)",
        k_shipped * 1e3, proj * 1e3);
    println!("  K  brick 2  w^T @ x^T, reshape only        {:6.3} ms   saves {:6.3} ms",
        k_transposed * 1e3, (k_shipped + proj - k_transposed) * 1e3);

    // ---- V: shipped copy; brick 3 would remove it entirely ----
    let v_shipped = best_of(20, || {
        let t = split(&projected).unwrap().contiguous().unwrap();
        std::hint::black_box(t);
    });
    println!("\n  V  shipped  split.contiguous()             {:6.3} ms", v_shipped * 1e3);
    println!("  V  brick 3  projection writes head layout  {:6.3} ms   saves {:6.3} ms (ceiling)",
        0.0, v_shipped * 1e3);

    // ---- the totals that decide whether any of this is worth building ----
    let per_layer_now = q_shipped + k_shipped + v_shipped;
    let b1 = q_shipped - q_folded;
    let b2 = (k_shipped + proj - k_transposed).max(0.0);
    let b3 = v_shipped + q_folded; // removing both remaining .contiguous() copies
    println!("\n  shipped prep per layer                     {:6.3} ms  x{layers} = {:6.2} ms/pass",
        per_layer_now * 1e3, per_layer_now * layers as f64 * 1e3);

    // The encoder pass this must be measured against, and the pipeline share
    // that decides whether the prize clears the harness floor.
    let enc_pass_ms = 159.0;
    let enc_share = 0.527;
    println!("\n  BRICK                          saves/layer   saves/pass   % encoder   % pipeline");
    for (name, save) in [
        ("1 fold scale into q weight", b1),
        ("2 produce K transposed", b2),
        ("3 head-layout projection", b3),
        ("1+2+3 combined", b1 + b2 + b3),
    ] {
        let per_pass = save * layers as f64 * 1e3;
        println!(
            "  {name:<30} {:8.3} ms {:10.2} ms {:9.1}% {:10.1}%",
            save * 1e3,
            per_pass,
            per_pass / enc_pass_ms * 100.0,
            per_pass / enc_pass_ms * 100.0 * enc_share
        );
    }
    println!(
        "\n  harness floor is ~2 % of pipeline (examples/pipeline_ab.rs --test null).\n  \
         A brick below that cannot be gated no matter how good it is."
    );
    Ok(())
}
