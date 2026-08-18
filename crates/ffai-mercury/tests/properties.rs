//! Property tests for the invariants this crate advertises (gate H-28).
//!
//! Scope is deliberately the MODEL-FREE surface, so these run in CI where no
//! weights are cached: the audio front-end and text normalization. That is also
//! where the real trust boundary now sits — model files are trusted input
//! (see docs/threat-model.md), but the audio and text a CALLER hands us are not.
//! A caller may legitimately pass an empty buffer, a single sample, NaNs, or
//! infinities, and none of that may panic.

use ffai_mercury::asr::mel::{self, MelSpectrogram};
use ffai_mercury::tts::normalize::normalize;
use proptest::prelude::*;

/// Samples including the values a real decoder can produce: NaN and +/-inf.
fn any_samples(max: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop_oneof![
            4 => (-1.0f32..1.0),
            1 => prop::num::f32::ANY,          // NaN, inf, subnormals
        ],
        0..max,
    )
}

proptest! {
    /// `pad_or_trim_to` is documented as "zero-pad or truncate to `target`".
    /// The length contract is what every downstream shape calculation assumes.
    #[test]
    fn pad_or_trim_to_always_returns_target_len(
        samples in any_samples(4096),
        target in 0usize..8192,
    ) {
        prop_assert_eq!(mel::pad_or_trim_to(&samples, target).len(), target);
    }

    /// Truncation must preserve the prefix; padding must be zeros.
    #[test]
    fn pad_or_trim_to_preserves_prefix_and_zero_fills(
        samples in prop::collection::vec(-1.0f32..1.0, 0..512),
        target in 0usize..1024,
    ) {
        let out = mel::pad_or_trim_to(&samples, target);
        let take = samples.len().min(target);
        prop_assert_eq!(&out[..take], &samples[..take]);
        prop_assert!(out[take..].iter().all(|&x| x == 0.0));
    }

    /// The front end must be TOTAL over caller audio. Short buffers are the
    /// interesting case: `compute` reflect-pads by N_FFT/2, and reflect padding
    /// of a buffer shorter than the pad width is where an out-of-bounds index
    /// would live if there were one.
    #[test]
    fn mel_compute_never_panics(samples in any_samples(2048)) {
        let m = MelSpectrogram::new(80);
        let chunk = m.compute(&samples);
        // Shape contract: data is (n_mels, n_frames) row-major.
        prop_assert_eq!(chunk.n_frames, MelSpectrogram::n_frames(samples.len()));
        prop_assert_eq!(chunk.data.len(), chunk.n_mels * chunk.n_frames);
    }

    /// Byte-stable determinism is an advertised property of this crate — the
    /// 0.7.0 version bump exists because synthesis output changed. Here it is
    /// pinned at the front end, where it can be checked without weights.
    #[test]
    fn mel_compute_is_bit_deterministic(samples in any_samples(1024)) {
        let m = MelSpectrogram::new(80);
        let a = m.compute(&samples);
        let b = m.compute(&samples);
        prop_assert_eq!(a.data.len(), b.data.len());
        // to_bits(), not ==: NaN != NaN, and we are asserting BYTE stability.
        prop_assert!(
            a.data.iter().zip(b.data.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
            "two identical calls produced different bytes"
        );
    }

    /// `resized` is a pad-or-truncate on the frame axis; same length contract.
    #[test]
    fn mel_resized_has_requested_frames(
        samples in any_samples(1024),
        frames in 0usize..64,
    ) {
        let chunk = MelSpectrogram::new(80).compute(&samples);
        let r = chunk.resized(frames);
        prop_assert_eq!(r.n_frames, frames);
        prop_assert_eq!(r.data.len(), r.n_mels * frames);
    }

    /// Text normalization is total: any string a caller can build, including
    /// control characters, lone surrogught-free unicode and huge digit runs.
    #[test]
    fn normalize_never_panics(text in ".*") {
        let _ = normalize(&text);
    }

    /// ...and deterministic, for the same byte-stability reason.
    #[test]
    fn normalize_is_deterministic(text in ".*") {
        prop_assert_eq!(normalize(&text), normalize(&text));
    }
}

// ---------------------------------------------------------------------------
// Structure-aware fuzzing of the hand-rolled protobuf parser.
//
// `tts/onnx.rs` decodes ONNX with hand-written offset arithmetic. Random bytes
// almost never survive the first length check, so they exercise the framing and
// nothing behind it. These generators emit WELL-FRAMED protobuf with arbitrary
// field numbers, wire types and payloads, which is what gets the fuzzer past the
// Reader and into the code that does arithmetic on attacker-chosen dims and
// lengths — where a hand-rolled parser actually breaks.
// ---------------------------------------------------------------------------

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// One protobuf field: tag, then a payload matching its wire type.
fn any_field() -> impl Strategy<Value = Vec<u8>> {
    (
        1u32..=8,
        0u8..=5,
        prop::collection::vec(any::<u8>(), 0..48),
        any::<u64>(),
    )
        .prop_map(|(field, wire, payload, v)| {
            let wire = match wire {
                0 | 1 | 2 | 5 => wire,
                _ => 2, // the parser rejects the rest; keep them rare but present
            };
            let mut out = Vec::new();
            varint(((field as u64) << 3) | wire as u64, &mut out);
            match wire {
                0 => varint(v, &mut out),
                1 => out.extend_from_slice(&v.to_le_bytes()),
                5 => out.extend_from_slice(&(v as u32).to_le_bytes()),
                _ => {
                    varint(payload.len() as u64, &mut out);
                    out.extend_from_slice(&payload);
                }
            }
            out
        })
}

/// A message is a concatenation of fields; nesting one inside a length-delimited
/// field is how real ONNX carries graphs, nodes and tensors.
fn any_message(depth: u32) -> impl Strategy<Value = Vec<u8>> {
    let leaf = prop::collection::vec(any_field(), 0..6).prop_map(|fields| fields.concat());
    leaf.prop_recursive(depth, 256, 4, |inner| {
        (1u32..=8, prop::collection::vec(inner, 1..3)).prop_map(|(field, kids)| {
            let mut out = Vec::new();
            for kid in kids {
                varint(((field as u64) << 3) | 2, &mut out);
                varint(kid.len() as u64, &mut out);
                out.extend_from_slice(&kid);
            }
            out
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// The contract is total: every byte string is a parse error or a Graph.
    /// A panic here is a denial of service on anything that loads a voice.
    #[test]
    fn onnx_parse_is_total_over_well_framed_protobuf(bytes in any_message(3)) {
        let _ = ffai_mercury::tts::onnx::parse(&bytes);
    }

    /// Truncation is the classic parser killer: every prefix of a valid message
    /// must fail cleanly rather than read past the end.
    #[test]
    fn onnx_parse_survives_every_truncation(
        bytes in any_message(2),
        cut in 0usize..512,
    ) {
        let n = cut.min(bytes.len());
        let _ = ffai_mercury::tts::onnx::parse(&bytes[..n]);
    }

    /// And over pure noise, which exercises the framing itself.
    #[test]
    fn onnx_parse_is_total_over_arbitrary_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..2048),
    ) {
        let _ = ffai_mercury::tts::onnx::parse(&bytes);
    }
}

// ---------------------------------------------------------------------------
// Targeted: attacker-chosen tensor dims.
//
// The framing survives fuzzing, so the arithmetic behind it is where a
// hand-rolled parser breaks. `parse_tensor` computes the expected element count
// as `dims.iter().product()` over values it casts with `as usize` — an unchecked
// multiply over an unvalidated, possibly NEGATIVE i64.
// ---------------------------------------------------------------------------

fn tag(field: u64, wire: u64, out: &mut Vec<u8>) {
    varint((field << 3) | wire, out);
}

/// A TensorProto with the given dims, float data_type, and no payload.
fn tensor_with_dims(dims: &[i64]) -> Vec<u8> {
    let mut t = Vec::new();
    for &d in dims {
        tag(1, 0, &mut t); // dims, varint
        varint(d as u64, &mut t);
    }
    tag(2, 0, &mut t); // data_type
    varint(1, &mut t); // DT_FLOAT
    tag(8, 2, &mut t); // name
    varint(4, &mut t);
    t.extend_from_slice(b"evil");
    t
}

/// Wrap a TensorProto as ModelProto.graph.initializer so `parse` reaches it.
fn model_with_tensor(tensor: &[u8]) -> Vec<u8> {
    let mut graph = Vec::new();
    tag(5, 2, &mut graph); // GraphProto.initializer
    varint(tensor.len() as u64, &mut graph);
    graph.extend_from_slice(tensor);

    let mut model = Vec::new();
    tag(7, 2, &mut model); // ModelProto.graph
    varint(graph.len() as u64, &mut model);
    model.extend_from_slice(&graph);
    model
}

#[test]
fn onnx_rejects_hostile_dims_without_panicking_or_wrapping() {
    // Each of these is a real ONNX shape an attacker (or a corrupt file) can
    // declare. None may panic, and none may be ACCEPTED: a tensor whose dims do
    // not describe its data is what turns into an out-of-bounds read downstream.
    let hostile: &[&[i64]] = &[
        &[-1], // ONNX's dynamic-dimension marker
        &[-1, -1],
        &[i64::MIN],
        &[1 << 32, 1 << 32], // product wraps usize to exactly 0
        &[1 << 40, 1 << 40],
        &[i64::MAX, 2],
        &[0], // legal but empty
        &[u32::MAX as i64, u32::MAX as i64],
    ];
    for dims in hostile {
        let bytes = model_with_tensor(&tensor_with_dims(dims));
        match ffai_mercury::tts::onnx::parse(&bytes) {
            Err(_) => {}
            Ok(g) => {
                for init in &g.initializers {
                    let expected: usize = init.dims.iter().copied().product();
                    assert_eq!(
                        init.data.len(),
                        expected,
                        "dims {dims:?} produced an initializer whose data ({}) does not \
                         match its declared shape {:?} — downstream indexing is unsound",
                        init.data.len(),
                        init.dims
                    );
                }
            }
        }
    }
}
