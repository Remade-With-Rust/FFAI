//! Probe: does the ENGINE's preprocessing path (grayscale + our bicubic)
//! reproduce the oracle read on the fixture crop? Splits recognition-defect
//! hypotheses: AR loop (oracle-proven) vs our crop preprocessing.
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

fn main() {
    let img = ffai_media::load_image(std::path::Path::new("corpora/refs/fixtures/trocr_line.png")).unwrap();
    let (w, h) = (img.width as usize, img.height as usize);
    let gray = ffai_carmenta::image::to_gray_f32(&img).unwrap();
    // Same crop the oracle used: left 3.2*h columns.
    let cw = (h * 32 / 10).min(w);
    let mut crop = vec![0f32; cw * h];
    for y in 0..h {
        crop[y * cw..(y + 1) * cw].copy_from_slice(&gray[y * w..y * w + cw]);
    }
    let x = ffai_carmenta::parseq::parseq_input(&crop, cw, h, &Device::Cpu).unwrap();
    let weights = ffai_models::cache_dir().join("models/parseq-tiny/parseq-tiny.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &Device::Cpu) }.unwrap();
    let p = ffai_carmenta::parseq::Parseq::new(vb).unwrap();
    println!("engine-path read: {:?}", p.recognize(&x).unwrap());
}
