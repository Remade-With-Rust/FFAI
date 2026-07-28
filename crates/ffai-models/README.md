# ffai-models

Weight management for [FFai](https://github.com/Remade-With-Rust/FFAI): TOML manifests, a hash-verified cache, and each model's own license surfaced before you use it.

```rust
let manifests = ffai_models::load_dir(std::path::Path::new("models"))?;
let m = manifests.iter().find(|m| m.name == "whisper-tiny-en").unwrap();
println!("{} — {}", m.name, m.license);      // licenses are not hidden
let resolved = m.fetch()?;                    // downloads once, into the cache
let weights = resolved.file("model.safetensors")?;
```

## Principle 4: weights are data, not code

Nothing is vendored. A manifest declares where weights come from and what license they carry:

```toml
name = "wav2vec2-base-960h"
task = "asr"
license = "Apache-2.0"
hf_repo = "facebook/wav2vec2-base-960h"

[[files]]
name = "model.safetensors"
```

That has a consequence FFai takes seriously: **a model you cannot fetch without clicking through a browser cannot go in a manifest.** It is why Mercury's diarization uses SpeechBrain's ECAPA-TDNN (Apache-2.0, ungated) rather than the more common pyannote weights, which are MIT-licensed *and* gated. Every model FFai fetches is fetchable without an account.

Model licenses are frequently more restrictive than FFai's own. `ffai models` lists them.

## License

MIT OR Apache-2.0 (this crate).
