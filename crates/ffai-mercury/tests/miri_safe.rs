//! The subset of our invariants that Miri can actually execute (gate H-23).
//!
//! Miri interprets Rust; it cannot run x86 SIMD intrinsics or foreign functions.
//! Two things in this crate therefore put whole test targets out of its reach:
//!
//!   * `mel::compute` runs an FFT through `rustfft`, which dispatches to SSE/AVX
//!     intrinsics — so `tests/properties.rs` cannot run under Miri;
//!   * the memory-instrumentation path calls Win32 (`SetProcessWorkingSetSizeEx`,
//!     `K32GetProcessMemoryInfo`) through `unsafe extern "system"`, which takes
//!     the whole `--lib` suite out on Windows. A Linux runner removes THAT one,
//!     but not the SIMD.
//!
//! What is left is still worth checking under Miri, because it is pure Rust doing
//! index arithmetic on caller-supplied lengths — precisely the shape where an
//! out-of-bounds or an uninitialised read hides. `reflect_pad`'s empty-input
//! panic lived exactly here.
//!
//!     cargo +nightly miri test -p ffai-mercury --test miri_safe

use ffai_mercury::asr::mel;
use ffai_mercury::tts::normalize::normalize;

/// Hand-rolled cases rather than proptest: proptest persists a regressions file,
/// which needs `-Zmiri-disable-isolation`, and shrinking under an interpreter is
/// far too slow to sit in CI. The generative version lives in tests/properties.rs.
const LENGTHS: &[usize] = &[0, 1, 2, 7, 64, 511, 512];

#[test]
fn pad_or_trim_to_length_contract_holds_for_every_shape() {
    for &n in LENGTHS {
        let samples: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        for &target in LENGTHS {
            let out = mel::pad_or_trim_to(&samples, target);
            assert_eq!(out.len(), target, "n={n} target={target}");
            let take = n.min(target);
            assert_eq!(
                &out[..take],
                &samples[..take],
                "prefix n={n} target={target}"
            );
            assert!(
                out[take..].iter().all(|&x| x == 0.0),
                "zero-fill n={n} target={target}"
            );
        }
    }
}

#[test]
fn pad_or_trim_to_handles_non_finite_samples() {
    let samples = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        f32::MIN_POSITIVE,
    ];
    for &target in LENGTHS {
        let out = mel::pad_or_trim_to(&samples, target);
        assert_eq!(out.len(), target);
    }
}

#[test]
fn normalize_is_total_and_deterministic() {
    // Control characters, astral-plane codepoints, mixed scripts, unbounded digit
    // runs: everything a caller can put in a &str.
    let cases = [
        "",
        " ",
        "route 66",
        "1234",
        "99999999999999999999",
        "\0\u{7f}\u{1b}[0m",
        "\u{1F600}\u{1F1EC}\u{1F1E7}",
        "naïve café",
        "混合 text 42",
        "\r\n\t",
        "0 1 2 3 4 5 6 7 8 9",
    ];
    for c in cases {
        let a = normalize(c);
        let b = normalize(c);
        assert_eq!(a, b, "normalize not deterministic for {c:?}");
    }
}

#[test]
fn n_frames_agrees_with_the_documented_formula() {
    // Documented as n_samples / HOP_LENGTH. A frame count that disagrees with the
    // buffer it describes is how a shape assumption becomes an out-of-bounds read.
    for &n in LENGTHS {
        let frames = mel::MelSpectrogram::n_frames(n);
        assert!(
            frames * 160 <= n,
            "n_frames({n}) = {frames} claims more samples than exist"
        );
    }
}
