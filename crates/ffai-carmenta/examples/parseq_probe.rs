//! Probe: does the ENGINE's preprocessing path (grayscale + our bicubic)
//! reproduce the oracle read on the fixture crop? Splits recognition-defect
//! hypotheses: AR loop (oracle-proven) vs our crop preprocessing.
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

fn main() {
    if std::env::var("SWAP").is_ok() {
        let input: Vec<f32> = std::fs::read("corpora/refs/fixtures/warehouse_py.bin").unwrap()
            .chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let x = candle_core::Tensor::from_vec(input, (1, 3, 32, 128), &Device::Cpu).unwrap();
        let weights = ffai_models::cache_dir().join("models/parseq-tiny/parseq-tiny.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &Device::Cpu) }.unwrap();
        let p = ffai_carmenta::parseq::Parseq::new(vb).unwrap();
        println!("py-input through OUR net: {:?}", p.recognize(&x).unwrap());
        return;
    }
    let arg = std::env::args().nth(1).unwrap_or_else(|| "corpora/refs/fixtures/trocr_line.png".into());
    let img = ffai_media::load_image(std::path::Path::new(&arg)).unwrap();
    let (w, h) = (img.width as usize, img.height as usize);
    let gray = ffai_carmenta::image::to_gray_f32(&img).unwrap();
    // Same crop the oracle used: left 3.2*h columns.
    let cw = if std::env::args().nth(1).is_some() { w } else { (h * 32 / 10).min(w) };
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
