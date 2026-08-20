// The audio front end, which is the REAL untrusted boundary.
//
// Model files are trusted input by product decision (see docs/threat-model.md),
// but `AudioBuffer::samples` comes from the caller and is not. A caller may pass
// an empty buffer, a single sample, NaNs, or infinities, and none of it may panic:
// a panic here is a denial of service in whatever process embedded us.
//
// This boundary has already produced one real defect. `compute(&[])` panicked with
// an out-of-bounds index inside reflect_pad, because the n == 0 guard returned
// index 0 into an empty slice. Found by tests/properties.rs, fixed, and seeded
// below so it can never come back.
//
// Run:  cargo +nightly fuzz run mel_compute
#![no_main]

use libfuzzer_sys::fuzz_target;
use ffai_mercury::asr::mel::MelSpectrogram;

fuzz_target!(|data: &[u8]| {
    // Reinterpret the fuzzer's bytes as f32 samples: this reaches NaN, inf and
    // subnormals far faster than generating floats in a range would.
    let samples: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let m = MelSpectrogram::new(80);
    let chunk = m.compute(&samples);

    // Shape contract must hold for every input, not just well-formed audio.
    assert_eq!(chunk.data.len(), chunk.n_mels * chunk.n_frames);
    assert_eq!(chunk.n_frames, MelSpectrogram::n_frames(samples.len()));
});
