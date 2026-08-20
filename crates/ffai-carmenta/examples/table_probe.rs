//! Run SLANet_plus on a table REGION and print its structure.
//!
//! The model is trained on cropped tables, not pages: fed a full page it emits
//! 501 `<pad>` tokens, which is the model correctly saying "there is no table
//! filling this image".
use ffai_carmenta::table::TableModel;
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
    let img_path = a.first().expect("usage: table_probe <image> [x0 y0 x1 y1]");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap();
    let m = TableModel::new(
        &root.join("corpora/refs/fixtures/slanet_plus_arch.json"),
        &root.join("corpora/refs/fixtures/slanet_plus.safetensors"),
    ).expect("load table model");
    let img = ffai_media::load_image(std::path::Path::new(img_path)).expect("load image");
    let img = if a.len() >= 5 {
        let g = |i: usize| a[i].parse::<usize>().unwrap_or(0);
        let c = crop(&img, g(1), g(2), g(3), g(4));
        println!("cropped to {}x{}", c.width, c.height);
        c
    } else {
        img
    };
    let t = std::time::Instant::now();
    let s = m.recognize(&img).expect("recognize");
    let real: Vec<&str> = s.tokens.iter().map(|x| x.as_str())
        .filter(|x| *x != "<pad>").collect();
    println!("infer {:?}   {} tokens ({} non-pad), {} cells",
             t.elapsed(), s.tokens.len(), real.len(), s.cells.len());
    println!("structure: {:?}", &real[..real.len().min(40)]);
    let html = s.to_html(&[]);
    println!("html: {}", &html[..html.len().min(400)]);
}
