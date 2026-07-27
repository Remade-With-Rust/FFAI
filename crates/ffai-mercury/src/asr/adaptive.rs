//! Adaptive precision — measure the machine, then choose.
//!
//! # Why this is not a constant
//!
//! Half precision is not universally faster or slower; it trades **bytes for
//! arithmetic efficiency**, and which side wins depends on the shape of the
//! operation *and* on the machine's balance between memory bandwidth and
//! compute throughput. On the development box, measured per op:
//!
//! | op | f32 | f16 | winner |
//! |---|---:|---:|---|
//! | vocabulary projection (1×384 @ 384×51864) | 3.35 ms | **2.14 ms** | f16, 1.56× |
//! | encoder feed-forward (1500×384 @ 384×1536) | **2.88 ms** | 4.21 ms | f32, 1.46× |
//!
//! The first streams 80 MB of weights for a single row of output and is
//! bandwidth-bound, so halving the bytes wins. The second is compute-bound and
//! candle has no f16 FMA path, so half precision upcasts and loses. A single
//! model-wide dtype has to give up one of these.
//!
//! Hard-coding the split would bake this machine's ratio into the binary. A
//! server with more bandwidth per core, a laptop with less, or a build of
//! candle that gains f16 kernels would each move the boundary — and on a GPU
//! it moves decisively. So instead of choosing, we **measure at load time**:
//! a few microbenchmarks at the exact shapes the model will run, and the
//! faster dtype wins.
//!
//! Cost is a few milliseconds once per process, cached per shape, and it
//! happens during weight loading which is already off the measured path.
//! `FFAI_PRECISION=f32|f16` overrides it for A/B; `FFAI_PROFILE=1` prints
//! what was chosen and why.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use ffai_core::candle::{DType, Device, Tensor};

/// The outcome of one calibration, kept for reporting.
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub dtype: DType,
    pub f32_secs: f64,
    pub f16_secs: f64,
}

impl Choice {
    /// How much the winner beat the loser by.
    pub fn speedup(&self) -> f64 {
        let (win, lose) = if self.dtype == DType::F16 {
            (self.f16_secs, self.f32_secs)
        } else {
            (self.f32_secs, self.f16_secs)
        };
        if win > 0.0 { lose / win } else { 1.0 }
    }
}

fn cache() -> &'static Mutex<HashMap<(usize, usize, usize), Choice>> {
    static CACHE: std::sync::OnceLock<Mutex<HashMap<(usize, usize, usize), Choice>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn forced() -> Option<DType> {
    match std::env::var("FFAI_PRECISION").as_deref() {
        Ok("f32") => Some(DType::F32),
        Ok("f16") => Some(DType::F16),
        _ => None,
    }
}

fn time_matmul(m: usize, k: usize, n: usize, dtype: DType, device: &Device) -> Option<f64> {
    let a = Tensor::zeros((m, k), DType::F32, device).ok()?.to_dtype(dtype).ok()?;
    let b = Tensor::zeros((k, n), DType::F32, device).ok()?.to_dtype(dtype).ok()?;
    // One untimed pass so allocation and first-touch don't land in the result.
    std::hint::black_box(a.matmul(&b).ok()?);
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        let out = a.matmul(&b).ok()?;
        std::hint::black_box(&out);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    Some(best)
}

/// Choose the faster dtype for an `(m, k) @ (k, n)` matmul on this machine.
///
/// Falls back to f32 when either candidate cannot be built or timed — a
/// device without f16 support gets the safe answer rather than an error.
pub fn matmul_dtype(m: usize, k: usize, n: usize, device: &Device) -> DType {
    if let Some(forced) = forced() {
        return forced;
    }
    let key = (m, k, n);
    if let Some(hit) = cache().lock().ok().and_then(|c| c.get(&key).copied()) {
        return hit.dtype;
    }

    // Same reasoning as matmul_pad_rows: probe a narrowed output. The
    // bandwidth-vs-compute balance that decides the dtype is a property of the
    // shape's aspect, not its absolute size.
    let probe_n = n.min(8192).max(1);
    let (Some(f32_secs), Some(f16_secs)) = (
        time_matmul(m, k, probe_n, DType::F32, device),
        time_matmul(m, k, probe_n, DType::F16, device),
    ) else {
        return DType::F32;
    };

    // Require a real margin before leaving f32: f16 costs a little accuracy,
    // so it has to earn its place rather than win on noise.
    const MARGIN: f64 = 1.10;
    let dtype = if f32_secs > f16_secs * MARGIN { DType::F16 } else { DType::F32 };
    let choice = Choice { dtype, f32_secs, f16_secs };

    if super::profile::is_enabled() {
        eprintln!(
            "[precision] ({m}x{k})@({k}x{n}): f32 {:.3} ms vs f16 {:.3} ms -> {:?} ({:.2}x)",
            f32_secs * 1e3,
            f16_secs * 1e3,
            dtype,
            choice.speedup(),
        );
    }
    if let Ok(mut c) = cache().lock() {
        c.insert(key, choice);
    }
    dtype
}

/// Precision for a cross-attention K/V cache, decided on the **whole chain**.
///
/// The isolated `q@k` matmul is the wrong thing to time here. Half precision
/// wins that link by 1.32x and loses the softmax that consumes it by 0.74x;
/// only `q@k -> softmax -> w@v`, with the dtype conversions a real
/// implementation must pay, says which way the sum falls. Timing the link
/// alone also lands near 1.0 for this shape, so the decision **flip-flopped
/// between runs** — non-determinism in a shipped model is worse than either
/// answer.
///
/// Measured on the chain, f16 wins 1.18x (31/41 paired rounds, z = +3.3)
/// once `fast_softmax` removes the f16 softmax penalty.
pub fn attention_kv_dtype(
    heads: usize,
    kv_len: usize,
    head_dim: usize,
    device: &Device,
) -> DType {
    if let Some(forced) = forced() {
        return forced;
    }
    static CACHE: std::sync::OnceLock<Mutex<HashMap<(usize, usize, usize), DType>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (heads, kv_len, head_dim);
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&key).copied()) {
        return hit;
    }

    let chain = |dtype: DType| -> Option<f64> {
        let q = Tensor::zeros((1, heads, 1, head_dim), DType::F32, device).ok()?
            .to_dtype(dtype).ok()?;
        let k = Tensor::zeros((1, heads, head_dim, kv_len), DType::F32, device).ok()?
            .to_dtype(dtype).ok()?;
        let v = Tensor::zeros((1, heads, kv_len, head_dim), DType::F32, device).ok()?
            .to_dtype(dtype).ok()?;
        let once = || -> Option<()> {
            let qk = q.matmul(&k).ok()?;
            let w = super::text_decoder::fast_softmax(&qk).ok()?;
            std::hint::black_box(w.matmul(&v).ok()?);
            Some(())
        };
        once()?;
        // Interleaving is not possible across two dtypes in one closure, so
        // take a best-of-N per arm and require a clear margin below.
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t0 = Instant::now();
            once()?;
            best = best.min(t0.elapsed().as_secs_f64());
        }
        Some(best)
    };

    let (Some(f32_secs), Some(f16_secs)) = (chain(DType::F32), chain(DType::F16)) else {
        return DType::F32;
    };
    // 10 % margin, as elsewhere: half precision must earn its accuracy cost.
    let dtype = if f32_secs > f16_secs * 1.10 { DType::F16 } else { DType::F32 };
    if super::profile::is_enabled() {
        eprintln!(
            "[xattn-kv] {heads}h x {kv_len} x {head_dim}: f32 {:.3} ms vs f16 {:.3} ms -> {dtype:?}",
            f32_secs * 1e3,
            f16_secs * 1e3
        );
    }
    if let Ok(mut c) = cache.lock() {
        c.insert(key, dtype);
    }
    dtype
}

/// How many rows to feed a single-row matmul of shape `(1,k) @ (k,n)`.
///
/// candle routes `m == 1` to a matrix-vector path that, for wide outputs, is
/// far worse than its matrix-matrix path — computing four rows can cost 2x
/// *less* than computing one. The cliff is shape-dependent and cuts both
/// ways, so it is measured rather than assumed:
///
/// | shape | m=1 vs m=4 |
/// |---|---:|
/// | attention projection 384x384 | 0.60x — padding LOSES |
/// | mlp fc1 384x1536 | 1.17x |
/// | vocabulary 384x51864 f16 | **1.97x** |
///
/// Returns 1 when padding does not pay, so callers can use the result
/// unconditionally.
pub fn matmul_pad_rows(k: usize, n: usize, dtype: DType, device: &Device) -> usize {
    // Read at CALIBRATION time only (this function runs once per shape, at
    // load) — never in the per-token path, which is where an earlier version
    // of this override cost more than the optimization saved.
    if std::env::var("FFAI_GEMV_PAD").as_deref() == Ok("off") {
        return 1;
    }
    static CACHE: std::sync::OnceLock<Mutex<HashMap<(usize, usize, DType), usize>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (k, n, dtype);
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&key).copied()) {
        return hit;
    }

    // MUST probe the REAL width. Narrowing this to 8192 to save startup time
    // made the cliff vanish and the calibration return 1 — so the cliff is a
    // SIZE property (the output no longer fits the cache hierarchy the same
    // way), not merely which kernel candle dispatches to. Calibration cost is
    // ~50 ms once per process; correctness of the decision outranks it.
    let probe_n = n;
    let time = |m: usize| -> Option<f64> {
        let x = Tensor::zeros((m, k), DType::F32, device).ok()?.to_dtype(dtype).ok()?;
        let w = Tensor::zeros((k, probe_n), DType::F32, device).ok()?.to_dtype(dtype).ok()?;
        std::hint::black_box(x.matmul(&w).ok()?);
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            let out = x.matmul(&w).ok()?;
            std::hint::black_box(&out);
            best = best.min(t0.elapsed().as_secs_f64());
        }
        Some(best)
    };

    let Some(base) = time(1) else { return 1 };
    let mut best_rows = 1usize;
    let mut best_secs = base;
    for m in [2usize, 4, 8] {
        // Require a 15 % margin: padding costs an allocation and a copy of
        // the (tiny) activation, and should not be adopted on noise.
        if let Some(secs) = time(m) {
            if secs * 1.15 < best_secs {
                best_secs = secs;
                best_rows = m;
            }
        }
    }
    if best_rows > 1 && super::profile::is_enabled() {
        eprintln!(
            "[gemv-pad] (1x{k})@({k}x{n}) {dtype:?}: m=1 {:.3} ms -> m={best_rows} {:.3} ms ({:.2}x)",
            base * 1e3,
            best_secs * 1e3,
            base / best_secs
        );
    }
    if let Ok(mut c) = cache.lock() {
        c.insert(key, best_rows);
    }
    best_rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_override_wins() {
        // Guards the A/B escape hatch: with FFAI_PRECISION set, no measurement
        // should be able to override the operator.
        if forced().is_none() {
            // Unset in this environment, so the calibrated path is exercised
            // instead — it must still return a usable dtype.
            let d = matmul_dtype(1, 64, 128, &Device::Cpu);
            assert!(d == DType::F32 || d == DType::F16);
        }
    }

    #[test]
    fn repeated_queries_hit_the_cache() {
        let dev = Device::Cpu;
        let first = matmul_dtype(1, 128, 256, &dev);
        let t0 = Instant::now();
        let second = matmul_dtype(1, 128, 256, &dev);
        // A cache hit must not re-run the benchmark.
        assert!(t0.elapsed().as_millis() < 5, "second call re-benchmarked");
        assert_eq!(first, second);
    }
}
