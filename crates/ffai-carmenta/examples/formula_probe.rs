//! Run PP-FormulaNet_plus-M on a formula REGION and print the LaTeX.
//!
//! Like the table model, this one is trained on CROPS, not pages. Feed it a
//! whole page and the margin-crop swallows the entire text body.
use ffai_carmenta::formula::FormulaModel;
use ffai_core::types::ImageBuffer;

fn crop(img: &ImageBuffer, x0: usize, y0: usize, x1: usize, y1: usize) -> ImageBuffer {
    let bpp = img.format.bytes_per_pixel();
    let (w, h) = (img.width as usize, img.height as usize);
    let (x1, y1) = (x1.min(w), y1.min(h));
    let (cw, ch) = (x1.saturating_sub(x0).max(1), y1.saturating_sub(y0).max(1));
    let mut data = vec![0u8; cw * ch * bpp];
    for y in 0..ch {
        let s = ((y0 + y) * w + x0) * bpp;
        data[y * cw * bpp..(y + 1) * cw * bpp].copy_from_slice(&img.data[s..s + cw * bpp]);
    }
    ImageBuffer { width: cw as u32, height: ch as u32, format: img.format, data }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let img_path = a.first().expect("usage: formula_probe <image> [x0 y0 x1 y1]");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap();
    let m = FormulaModel::new(
        &root.join("corpora/refs/fixtures/formulanet_arch.json"),
        &root.join("corpora/refs/fixtures/formulanet.safetensors"),
        &root.join("corpora/refs/fixtures/formulanet_vocab.json"),
    )
    .expect("load formula model");

    let img = ffai_media::load_image(std::path::Path::new(img_path))
        .expect("load image");
    let img = if a.len() >= 5 {
        let g = |i: usize| a[i].trim().parse::<f32>().unwrap().round() as usize;
        let c = crop(&img, g(1), g(2), g(3), g(4));
        eprintln!("cropped to {}x{}", c.width, c.height);
        c
    } else {
        img
    };

    let t = std::time::Instant::now();
    let latex = m.recognize(&img).expect("recognize");
    eprintln!("infer {:?}", t.elapsed());
    println!("{latex}");
}
