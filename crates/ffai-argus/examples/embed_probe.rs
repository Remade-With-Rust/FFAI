//! The patch embedding — the one tower stage nothing has priced.
//!
//! Two candidates visible in `Embeddings::forward`:
//!  * `conv2d` with a 16x16 kernel at stride 16 is NON-OVERLAPPING, i.e. exactly
//!    a matmul over flattened patches. candle parallelises conv2d, so it may
//!    already be fine — or the im2col may cost more than the product.
//!  * `flatten_from(2).transpose(1,2)` yields a STRIDED view, and the position
//!    `broadcast_add` then runs over it. Strided elementwise is what cost the
//!    text tower 114 ms/layer.
use candle_core::{Device, Module, Tensor};
use std::time::Instant;

fn t(f: &mut dyn FnMut() -> candle_core::Result<Tensor>) -> f64 {
    let _ = f().expect("warm");
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let s = Instant::now();
        let o = f().expect("op");
        std::hint::black_box(o.dims());
        b = b.min(s.elapsed().as_secs_f64());
    }
    b * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    const P: usize = 16;
    const SIDE: usize = 512;
    const DIM: usize = 768;
    const GRID: usize = SIDE / P; // 32
    let px = Tensor::rand(-1.0f32, 1.0, (1, 3, SIDE, SIDE), &d)?;
    let wc = Tensor::rand(-1.0f32, 1.0, (DIM, 3, P, P), &d)?;
    let bc = Tensor::rand(-1.0f32, 1.0, DIM, &d)?;
    let conv = candle_nn::Conv2d::new(
        wc.clone(),
        Some(bc.clone()),
        candle_nn::Conv2dConfig { stride: P, ..Default::default() },
    );
    let pos = Tensor::rand(-1.0f32, 1.0, (GRID * GRID, DIM), &d)?;

    println!("patch embed: (1,3,{SIDE},{SIDE}) -> (1,{},{DIM})\n", GRID * GRID);

    let c = t(&mut || conv.forward(&px));
    println!("  {:<48} {c:>8.3} ms", "conv2d stride 16 [ours]");

    let feat = conv.forward(&px)?;
    let strided = t(&mut || {
        let x = feat.flatten_from(2)?.transpose(1, 2)?;
        x.broadcast_add(&pos)
    });
    println!("  {:<48} {strided:>8.3} ms", "flatten+transpose (STRIDED) + broadcast_add [ours]");

    let contig = t(&mut || {
        let x = feat.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        x.broadcast_add(&pos)
    });
    println!("  {:<48} {contig:>8.3} ms", "  ...contiguous() first, then broadcast_add");

    // conv2d as a MATMUL. Stride == kernel, so im2col is a pure permutation:
    // no element is duplicated, unlike an overlapping conv.
    let w2 = wc.reshape((DIM, 3 * P * P))?.t()?.contiguous()?;
    let im2col = |px: &Tensor| -> candle_core::Result<Tensor> {
        px.reshape((1, 3, GRID, P, GRID, P))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((GRID * GRID, 3 * P * P))
    };
    let mm = t(&mut || im2col(&px)?.matmul(&w2)?.broadcast_add(&bc));
    println!("  {:<48} {mm:>8.3} ms   {:>6.0} GF/s",
        "conv2d AS MATMUL (im2col is a permutation)",
        2.0 * (GRID * GRID * 3 * P * P * DIM) as f64 / 1e9 / (mm / 1e3));

    // ...and it must be the SAME answer, or it is not an arm.
    let a = conv.forward(&px)?.flatten_from(2)?.transpose(1, 2)?
        .contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let bb = im2col(&px)?.matmul(&w2)?.broadcast_add(&bc)?
        .flatten_all()?.to_vec1::<f32>()?;
    let worst = a.iter().zip(&bb).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    println!("  {:<48} {worst:>8.3e}", "  max|diff| vs conv2d");

    println!("\n  embed as written : {:>8.3} ms   x17 tiles = {:>7.1} ms", c + strided, (c + strided) * 17.0);
    println!("  embed best above : {:>8.3} ms   x17 tiles = {:>7.1} ms   ({:.2}x)",
        c + contig, (c + contig) * 17.0, (c + strided) / (c + contig));
    Ok(())
}
