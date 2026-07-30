//! Function-vs-function DET stage: dump our CRAFT line boxes per image as
//! JSONL, for coverage comparison against paddle's det polys (plan §8.6).
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

fn main() {
    let list = std::env::args().nth(1).expect("filelist");
    let dev = Device::Cpu;
    let p = ffai_models::cache_dir().join("models/craft-mlt/craft.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[p], DType::F32, &dev) }.unwrap();
    let craft = ffai_carmenta::craft::Craft::new(vb).unwrap();
    for path in std::fs::read_to_string(&list).unwrap().lines().filter(|l| !l.trim().is_empty()) {
        let img = ffai_media::load_image(std::path::Path::new(path.trim())).unwrap();
        let (w, h) = (img.width as usize, img.height as usize);
        let gray = ffai_carmenta::image::to_gray_f32(&img).unwrap();
        let t0 = std::time::Instant::now();
        let (input, scale) = ffai_carmenta::image::craft_input(&gray, w, h, &dev).unwrap();
        let maps = craft.forward(&input).unwrap();
        let (mh, mw, _) = maps.dims3().unwrap();
        let flat = maps.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let region: Vec<f32> = flat.iter().step_by(2).copied().collect();
        let affinity: Vec<f32> = flat.iter().skip(1).step_by(2).copied().collect();
        let boxes = ffai_carmenta::boxes::extract_boxes(&region, &affinity, mw, mh);
        let secs = t0.elapsed().as_secs_f64();
        let to_img = |v: usize| (v as f32 * 2.0 / scale) as i64;
        let rects: Vec<[i64; 4]> = boxes.iter().map(|b| [to_img(b.x0), to_img(b.y0), to_img(b.x1), to_img(b.y1)]).collect();
        println!("{}", serde_json::json!({"path": path.trim(), "rects": rects, "secs": secs}));
    }
}
