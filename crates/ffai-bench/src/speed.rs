//! Best-of-N wall-clock measurement, ported from Prometheus
//! (`prom-trial::speed`). We report the *best* (minimum) time — the run least
//! perturbed by scheduler noise — plus the median for a spread sense.

use ffai_core::error::Result;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedStats {
    /// Best (minimum) wall-clock seconds across runs.
    pub best_secs: f64,
    /// Median wall-clock seconds.
    pub median_secs: f64,
    /// Number of timed runs.
    pub runs: usize,
}

/// Time `f` `n` times (n ≥ 1). Any error aborts and propagates — a failed
/// run is not a fast run.
pub fn best_of_n(n: usize, mut f: impl FnMut() -> Result<()>) -> Result<SpeedStats> {
    let n = n.max(1);
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        f()?;
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(SpeedStats {
        best_secs: times[0],
        median_secs: times[times.len() / 2],
        runs: n,
    })
}

impl SpeedStats {
    /// `baseline.best / self.best` — > 1 means `self` is faster.
    #[must_use]
    pub fn speedup_over(&self, baseline: &Self) -> f64 {
        if self.best_secs <= 0.0 {
            f64::INFINITY
        } else {
            baseline.best_secs / self.best_secs
        }
    }
}

/// Real-time factor for media processing: media seconds processed per
/// wall-clock second (e.g. RTF 20 = a minute of audio in 3 s). Higher is
/// faster; the natural "×-realtime" number for ASR/TTS claims.
#[must_use]
pub fn real_time_factor(media_secs: f64, wall_secs: f64) -> f64 {
    if wall_secs <= 0.0 {
        f64::INFINITY
    } else {
        media_secs / wall_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_of_n_counts_and_orders() {
        let s = best_of_n(5, || Ok(())).unwrap();
        assert_eq!(s.runs, 5);
        assert!(s.best_secs <= s.median_secs);
    }

    #[test]
    fn rtf_is_media_per_wall() {
        assert!((real_time_factor(60.0, 3.0) - 20.0).abs() < 1e-12);
    }
}
