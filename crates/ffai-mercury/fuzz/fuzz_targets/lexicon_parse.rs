// Pronunciation lexicons are read from the model cache as raw bytes and mapped
// Latin-1 into text before parsing (tts/lexicon.rs). The parser walks attacker-shaped
// lines, so it must be total over arbitrary bytes.
//
// Run:  cargo +nightly fuzz run lexicon_parse
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // `Lexicon::load` takes a path rather than bytes, so the harness materialises the
    // input. Writing to a temp file per iteration is slower than an in-memory API would
    // be; the right fix is a `from_bytes` entry point, tracked in the audit as the
    // follow-up to this target.
    let mut path = std::env::temp_dir();
    path.push(format!("ffai-fuzz-lexicon-{}.dict", std::process::id()));
    if let Ok(mut f) = std::fs::File::create(&path) {
        if f.write_all(data).is_ok() {
            drop(f);
            let _ = ffai_mercury::tts::lexicon::Lexicon::load(&path);
        }
    }
});
