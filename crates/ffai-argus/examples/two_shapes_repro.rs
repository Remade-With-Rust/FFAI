//! MINIMAL REPRODUCER: captioning two DIFFERENTLY-SHAPED images in one process.
//!
//! `ffai bench vlm` segfaults on its second clip. `ffai caption` never does,
//! the demo captions repeatedly without trouble, and
//! `a_second_caption_does_not_inherit_the_first` calls `describe_image` three
//! times and passes — because all three use the SAME image.
//!
//! The bench's clips differ in size (104x27, 164x60, 114x54, ...), so each one
//! produces a different AnyRes tile grid and therefore different tensor shapes.
//! This isolates that single variable: same engine, same prompt, two images
//! that differ only in dimensions.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example two_shapes_repro
//! ```
use ffai_core::engine::{VlmEngine, VlmOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("workspace root")?;
    let engine = ffai_argus::SmolVlm::with_manifest_dir(root.join("models"));
    let opts = VlmOptions {
        prompt: Some("what is written in the image?".into()),
        max_new_tokens: Some(16),
        ..VlmOptions::default()
    };

    let clips = ["0.png", "1.png", "2.png"];
    for (i, name) in clips.iter().enumerate() {
        let path = root.join("corpora/clips/argus-ocrbench").join(name);
        let img = ffai_media::load_image(&path)?;
        println!("  [{i}] {name}  {}x{}", img.width, img.height);
        let out = engine.describe_image(&img, &opts)?;
        println!("      -> {}", out.chars().take(60).collect::<String>());
    }
    println!("\n  survived {} differently-shaped images in one process", clips.len());
    Ok(())
}
