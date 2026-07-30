//! Print the content signal + class for given images.
fn main() {
    for p in std::env::args().skip(1) {
        let img = ffai_media::load_image(std::path::Path::new(&p)).unwrap();
        let f = ffai_carmenta::content::flatness(&img);
        let k = ffai_carmenta::content::classify(&img);
        println!("{:<46} fmt {:?} flatness {:.4} -> {:?}", p, img.format, f, k);
    }
}
