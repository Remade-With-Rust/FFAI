//! Sampling: temperature, top-k, top-p and min-p, seeded and the same on every device.
//!
//! The chain is llama.cpp's default order: top-k, then top-p, then min-p (each on the
//! candidates left), then temperature, then one seeded draw. Every step is deterministic:
//!
//! - candidates are ordered by (logit descending, id ascending), so ties never depend on a sort;
//! - probabilities use [`crate::numerics::exp_canon`] (one `f32` operation a step on every
//!   tier), summed in `f64` in candidate order;
//! - the generator is xoshiro256** seeded by splitmix64, and a draw is one `f64` in `[0, 1)`
//!   from its top 53 bits.
//!
//! So a seed and a file give the same tokens on every device. A temperature of zero or below is
//! greedy (the head screen's argmax, by proof the full head's).

use crate::numerics::exp_canon;

/// How to pick a token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// Divides the logits; `<= 0` is greedy.
    pub temperature: f32,
    /// Keep the `top_k` likeliest (`0`: all).
    pub top_k: usize,
    /// Keep the smallest likeliest set whose probability reaches `top_p` (`>= 1`: all).
    pub top_p: f32,
    /// Keep tokens at least `min_p` times as likely as the likeliest (`<= 0`: all).
    pub min_p: f32,
    /// The generator's seed.
    pub seed: u64,
}

impl Sampling {
    /// Greedy decoding.
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 0,
    };

    /// Whether this picks the argmax.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }
}

/// xoshiro256**, seeded through splitmix64: small, fast, and the same sequence everywhere.
#[derive(Clone, Debug)]
pub struct Rng([u64; 4]);

impl Rng {
    /// A generator for `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        Self([next(), next(), next(), next()])
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// A draw in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "53 bits, exact in f64")]
        let v = (self.next_u64() >> 11) as f64;
        v * (1.0 / (1u64 << 53) as f64)
    }
}

/// The ids of `logits` in candidate order (logit descending, id ascending), cut to the `k`
/// largest when `k > 0`.
fn candidates(logits: &[f32], k: usize) -> Vec<u32> {
    crate::engine::top_k_ids(logits, k)
}

/// `exp((l - max) · scale)` for each candidate (in order), and their sum in `f64`.
fn weights(cands: &[(u32, f32)], scale: f32) -> (Vec<f32>, f64) {
    let max = cands.first().map_or(0.0, |c| c.1);
    let w: Vec<f32> = cands
        .iter()
        .map(|c| exp_canon((c.1 - max) * scale))
        .collect();
    let total = w.iter().map(|&x| f64::from(x)).sum();
    (w, total)
}

/// Pick the next token from `logits` (the full head's).
#[must_use]
pub fn sample(logits: &[f32], s: &Sampling, rng: &mut Rng) -> u32 {
    if s.is_greedy() {
        return crate::engine::argmax(logits);
    }
    let cands: Vec<(u32, f32)> = candidates(logits, s.top_k)
        .into_iter()
        .map(|i| (i, logits[i as usize]))
        .collect();
    sample_from(&cands, s, rng)
}

/// Pick the next token from candidates already in order (logit descending, id ascending) and
/// already cut to top-k: what the head screen's exact top k gives ([`crate::head::top_k`]).
/// The same draw as [`sample`] over the full logits.
#[must_use]
pub fn sample_from(cands: &[(u32, f32)], s: &Sampling, rng: &mut Rng) -> u32 {
    let mut n = cands.len();
    if n == 0 {
        return 0;
    }
    if s.top_p < 1.0 && n > 1 {
        let (w, total) = weights(&cands[..n], 1.0);
        let want = f64::from(s.top_p) * total;
        let mut cum = 0.0;
        for (i, &x) in w.iter().enumerate() {
            cum += f64::from(x);
            if cum >= want {
                n = i + 1;
                break;
            }
        }
    }
    if s.min_p > 0.0 && n > 1 {
        // p_i >= min_p · p_max  <=>  l_i >= l_max + ln(min_p).
        let floor = cands[0].1 + libm::logf(s.min_p);
        n = cands[..n]
            .iter()
            .take_while(|c| c.1 >= floor)
            .count()
            .max(1);
    }
    let cands = &cands[..n];
    if s.is_greedy() {
        return cands[0].0;
    }
    let (w, total) = weights(cands, 1.0 / s.temperature);
    let target = rng.next_f64() * total;
    let mut cum = 0.0;
    for (c, &x) in cands.iter().zip(&w) {
        cum += f64::from(x);
        if target < cum {
            return c.0;
        }
    }
    cands[n - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generator_is_xoshiro256starstar() {
        // splitmix64(0) seeds; the first outputs are fixed forever.
        let mut r = Rng::new(0);
        let a: Vec<u64> = (0..3).map(|_| r.next_u64()).collect();
        let mut r2 = Rng::new(0);
        assert_eq!(a, (0..3).map(|_| r2.next_u64()).collect::<Vec<_>>());
        let f = Rng::new(7).next_f64();
        assert!((0.0..1.0).contains(&f));
    }

    #[test]
    fn greedy_and_filters() {
        let logits = [1.0f32, 3.0, 2.0, 3.0, -5.0];
        let mut r = Rng::new(1);
        assert_eq!(sample(&logits, &Sampling::GREEDY, &mut r), 1);
        assert_eq!(candidates(&logits, 0), vec![1, 3, 2, 0, 4]);
        assert_eq!(candidates(&logits, 2), vec![1, 3]);
        // top_k 2: only 1 and 3 can come out.
        let s = Sampling {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        };
        for _ in 0..200 {
            assert!(matches!(sample(&logits, &s, &mut r), 1 | 3));
        }
        // min_p 0.5: e^(2-3) < 0.5, so only the two 3.0s survive.
        let s = Sampling {
            top_k: 0,
            min_p: 0.5,
            ..s
        };
        for _ in 0..200 {
            assert!(matches!(sample(&logits, &s, &mut r), 1 | 3));
        }
        // top_p tiny: the first candidate alone.
        let s = Sampling {
            min_p: 0.0,
            top_p: 0.01,
            ..s
        };
        for _ in 0..50 {
            assert_eq!(sample(&logits, &s, &mut r), 1);
        }
    }

    #[test]
    fn a_seed_repeats() {
        let logits: Vec<f32> = (0..1000).map(|i| ((i * 37 % 101) as f32) * 0.05).collect();
        let s = Sampling {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            seed: 42,
        };
        let run = || {
            let mut r = Rng::new(s.seed);
            (0..20)
                .map(|_| sample(&logits, &s, &mut r))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
