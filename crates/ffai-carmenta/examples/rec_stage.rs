//! Function-vs-function REC stage (plan §8.6): read word crops from a
//! filelist, run BOTH our recognizers on each, emit JSONL for the scorer.
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

fn main() {
    let list = std::env::args().nth(1).expect("filelist");
    let dev = Device::Cpu;
    let load = |m: &str, f: &str| {
        let p = ffai_models::cache_dir().join("models").join(m).join(f);
        unsafe { VarBuilder::from_mmaped_safetensors(&[p], DType::F32, &dev) }.unwrap()
    };
    let crnn = ffai_carmenta::crnn::Crnn::new(load("crnn-english-g2", "crnn.safetensors")).unwrap();
    let parseq = ffai_carmenta::parseq::Parseq::new(load("parseq-tiny", "parseq-tiny.safetensors")).unwrap();

    for path in std::fs::read_to_string(&list).unwrap().lines().filter(|l| !l.trim().is_empty()) {
        let img = ffai_media::load_image(std::path::Path::new(path.trim())).unwrap();
        let (w, h) = (img.width as usize, img.height as usize);
        let gray = ffai_carmenta::image::to_gray_f32(&img).unwrap();
        let t0 = std::time::Instant::now();
        let c = ffai_carmenta::image::crnn_input(&gray, w, h, 0, 0, w, h, &dev)
            .and_then(|x| {
                let l = crnn.forward(&x).map_err(ffai_carmenta::image::candle_err)?;
                ffai_carmenta::crnn::ctc_greedy(&l).map_err(ffai_carmenta::image::candle_err)
            })
            .map(|(t, _)| t)
            .unwrap_or_default();
        let crnn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let p = ffai_carmenta::parseq::parseq_input(&gray, w, h, &dev)
            .and_then(|x| parseq.recognize(&x))
            .map(|(t, _)| t)
            .unwrap_or_default();
        let parseq_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{}",
            serde_json::json!({"file": path.trim(), "crnn": c, "crnn_ms": crnn_ms, "parseq": p, "parseq_ms": parseq_ms})
        );
    }
}
