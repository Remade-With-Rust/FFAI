//! The README's example, compiled.
//!
//! A published README is permanent, so its code lives here where
//! `cargo build --examples` type-checks it against the real API rather than
//! against memory.
use ffai_argus::SmolVlm;
use ffai_core::engine::{VlmEngine, VlmOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("workspace root")?;
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        root.join("corpora/clips/carmenta-render/page-00.png")
            .to_string_lossy()
            .into_owned()
    });

    let engine = SmolVlm::with_manifest_dir(root.join("models"));
    let image = ffai_media::load_image(std::path::Path::new(&path))?;
    let caption = engine.describe_image(
        &image,
        &VlmOptions {
            prompt: Some("What is written in this image?".into()),
            ..VlmOptions::default()
        },
    )?;
    println!("{caption}");
    Ok(())
}
