// The highest-value untrusted-input path in the crate.
//
// `ffai-mercury` reads ONNX graphs out of the model cache and parses them with no
// verification of their contents (see docs/threat-model.md, section 4). Anyone who can
// write to that directory controls these bytes. The parser must therefore never panic,
// never overflow, and never index out of bounds - for ANY input.
//
// Run:  cargo +nightly fuzz run onnx_parse
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The contract is total: every byte string is either a parse error or a Graph.
    // Anything else - a panic, an arithmetic overflow, an OOB index - is a finding.
    let _ = ffai_mercury::tts::onnx::parse(data);
});
