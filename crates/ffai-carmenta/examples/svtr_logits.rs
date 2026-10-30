//! Write SVTR's output probabilities for real line crops to a file, so two
//! builds — or one build under two environments — can be compared tensor for
//! tensor.
//!
//! This is the gate for changing how the recognizer COMPUTES without changing
//! what it computes. `svtr_oracle` holds the port against paddle, but its
//! input fixture is a binary that is not in every checkout; this needs only
//! the weights and an image. Run it once with `FFAI_SVTR_FUSED=0
//! FFAI_DWCONV=0` (candle's own ops, the arm the oracle pinned at 8.857e-5)
//! and once without, then compare the files: the distance between them is
//! what the fast path spends of the oracle's 1e-4 bar.
//!
//! Usage: `svtr_logits <weights_root> <image> <out.bin> <x0,y0,x1,y1>...`
//! The output is every crop's `(T, n_class)` probabilities as little-endian
//! f32, concatenated in argument order.

use std::io::Write as _;
use std::path::Path;

use candle_core::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: svtr_logits <weights_root> <image> <out.bin> <x0,y0,x1,y1>...";
    let root = args.next().ok_or(usage)?;
    let image = args.next().ok_or(usage)?;
    let out = args.next().ok_or(usage)?;
    let dev = Device::Cpu;
    let rec = Path::new(&root).join("ppocrv5-mobile-rec");
    let charset = ffai_carmenta::svtr::load_charset(&rec.join("charset.txt"))?;
    let m = ffai_carmenta::svtr::load(&rec.join("rec.safetensors"), &dev, charset.len() + 1)?;
    let img = ffai_media::load_image(Path::new(&image))?;
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
    for b in args {
        let v: Vec<usize> = b.split(',').map(str::parse).collect::<Result<_, _>>()?;
        let [x0, y0, x1, y1] = v[..] else { return Err(usage.into()) };
        let x = ffai_carmenta::svtr::svtr_input(&img, x0, y0, x1, y1, &dev)?;
        let p = m.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        for v in &p {
            f.write_all(&v.to_le_bytes())?;
        }
        eprintln!("{b}: {} values", p.len());
    }
    Ok(())
}
