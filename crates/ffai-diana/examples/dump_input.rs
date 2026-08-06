//! Print statistics of the preprocessed input tensor.
//!
//! Same weights, same 384x640 letterbox, and yet the reference finds 1556
//! objects on MOT17-13 that we do not while we sit at parity on MOT17-09.
//! That pattern — marginal detections lost, obvious ones kept — is what a
//! PIXEL-level difference looks like, so this dumps the tensor the model
//! actually sees so it can be diffed against the reference's.
use ffai_diana::image::{letterbox_with, Geometry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: dump_input <image>");
    let img = ffai_media::load_image(std::path::Path::new(&path))?;
    println!("source {}x{} format {:?}", img.width, img.height, img.format);
    let (t, lb) = letterbox_with(&img, 640, Geometry::Rect, &candle_core::Device::Cpu)?;
    println!("tensor dims {:?}  scale {:.6} pad_x {} pad_y {}", t.dims(), lb.scale, lb.pad_x, lb.pad_y);
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    let (c, h, w) = (t.dims()[1], t.dims()[2], t.dims()[3]);
    for ch in 0..c {
        let plane = &v[ch * h * w..(ch + 1) * h * w];
        let mean: f32 = plane.iter().sum::<f32>() / plane.len() as f32;
        let mn = plane.iter().cloned().fold(f32::MAX, f32::min);
        let mx = plane.iter().cloned().fold(f32::MIN, f32::max);
        println!("  ch{ch}: mean {mean:.6} min {mn:.4} max {mx:.4}");
    }
    // A few interior pixels, well away from the letterbox padding.
    let px = |ch: usize, y: usize, x: usize| v[ch * h * w + y * w + x];
    println!("  sample (y=200,x=320): R {:.6} G {:.6} B {:.6}", px(0,200,320), px(1,200,320), px(2,200,320));
    println!("  sample (y=100,x=100): R {:.6} G {:.6} B {:.6}", px(0,100,100), px(1,100,100), px(2,100,100));
    Ok(())
}
