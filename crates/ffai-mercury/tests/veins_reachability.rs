//! Vein 11 of `docs/plans/mercury-instruction-veins.md`: a fast path that
//! nothing calls is the defect every correctness gate passes. These pin, on
//! an x86-64 box with AVX2, that each hand-written kernel SERVES the shapes
//! whisper-tiny.en and the VITS voice actually present — so a future
//! predicate change that quietly routes production onto a fallback fails a
//! test instead of a benchmark.
//!
//! On a machine without AVX2 (or a non-x86 target) every assertion inverts:
//! the kernels must decline, and the candle fallbacks carry the work.

use ffai_core::candle::{DType, Device, Tensor};
use ffai_mercury::asr::flash_attn;

fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[test]
fn cross_attention_kernel_serves_the_whisper_tiny_cache_shape() {
    let dev = Device::Cpu;
    // tiny.en: 6 heads, head_dim 64, 1500 encoder frames.
    let kt = Tensor::zeros((1, 6, 64, 1500), DType::F32, &dev).unwrap();
    let v = Tensor::zeros((1, 6, 1500, 64), DType::F32, &dev).unwrap();
    assert_eq!(flash_attn::serves(&kt, &v), avx2());
}

#[test]
fn encoder_attention_kernels_serve_the_whisper_tiny_window() {
    let dev = Device::Cpu;
    let q = Tensor::zeros((1, 1500, 384), DType::F32, &dev).unwrap();
    let v = Tensor::zeros((1, 1500, 384), DType::F32, &dev).unwrap();
    let kt = Tensor::zeros((1, 6, 64, 1500), DType::F32, &dev).unwrap();
    let unsplit = flash_attn::attend_merged_unsplit(&q, &kt, &v).unwrap();
    assert_eq!(unsplit.is_some(), avx2(), "unsplit merged kernel (vein 14)");
    let qs = Tensor::zeros((1, 6, 1500, 64), DType::F32, &dev).unwrap();
    let vs = Tensor::zeros((1, 6, 1500, 64), DType::F32, &dev).unwrap();
    let split = flash_attn::attend_merged(&qs, &kt, &vs).unwrap();
    assert_eq!(split.is_some(), avx2(), "split merged kernel (legacy arm)");
}

#[test]
fn single_row_f16_gemv_serves_every_whisper_width() {
    let dev = Device::Cpu;
    for (out, inp) in [
        (384usize, 384usize),
        (1536, 384),
        (384, 1536),
        (512, 512),
        (768, 768),
    ] {
        let w = Tensor::zeros((out, inp), DType::F32, &dev).unwrap();
        let served = ffai_mercury::asr::f16_gemv::F16Gemv::new(&w)
            .unwrap()
            .is_some();
        #[cfg(target_arch = "x86_64")]
        let want = is_x86_feature_detected!("f16c");
        #[cfg(not(target_arch = "x86_64"))]
        let want = false;
        assert_eq!(served, want, "f16 gemv for ({out}, {inp})");
    }
}

#[test]
fn int8_vocabulary_projection_serves_the_whisper_tiny_embedding_shape() {
    let dev = Device::Cpu;
    // Shape only: 1024 of the 51 865 rows is enough to exercise the predicate.
    let emb = Tensor::zeros((1024, 384), DType::F32, &dev).unwrap();
    let served = ffai_mercury::asr::vocab_int8::Int8Vocab::new(&emb)
        .unwrap()
        .is_some();
    #[cfg(target_arch = "x86_64")]
    let want = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let want = false;
    assert_eq!(served, want);
}
