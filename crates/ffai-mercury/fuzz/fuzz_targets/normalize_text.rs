// The text front end: the other half of the caller-controlled surface.
//
// `synthesize(text)` accepts any string a caller can build - control characters,
// astral-plane codepoints, unbounded digit runs, mixed scripts. Normalization is
// documented as expanding integers 0..=9999 and leaving everything else alone,
// so it must be TOTAL and deterministic over arbitrary input.
//
// Run:  cargo +nightly fuzz run normalize_text
#![no_main]

use libfuzzer_sys::fuzz_target;
use ffai_mercury::tts::normalize::normalize;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let a = normalize(text);
        // Byte-stable determinism is an advertised property of this crate.
        let b = normalize(text);
        assert_eq!(a, b, "normalize is not deterministic");
    }
});
