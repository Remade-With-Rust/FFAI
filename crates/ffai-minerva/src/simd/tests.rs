//! Every twin against its scalar oracle, at every tier this CPU supports, `assert_eq!` on bits.
//! Inputs are adversarial on purpose: every tail length, saturation extremes, subnormals, signed
//! zeros, infinities; the f16 conversions exhaustively.

use super::*;
use rusty_ternary::i2s::pack;

/// A deterministic stream (PCG-ish).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn f32(&mut self, scale: f32) -> f32 {
        #[expect(clippy::cast_precision_loss, reason = "a 31-bit value")]
        let u = self.next() as f32 / 2_147_483_648.0;
        (u * 2.0 - 1.0) * scale
    }
    /// Floats with the awkward values mixed in.
    fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| match self.next() % 23 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(1 + (self.next() as u32 % 1000)),
                3 => -f32::MIN_POSITIVE * 0.5,
                4 => scale * if i % 2 == 0 { 1.0 } else { -1.0 },
                _ => self.f32(scale),
            })
            .collect()
    }
    fn halves(&mut self, n: usize) -> Vec<u16> {
        // Finite halves (no Inf/NaN), subnormals and zeros included.
        (0..n)
            .map(|_| {
                let h = (self.next() & 0xffff) as u16;
                if (h & 0x7c00) == 0x7c00 {
                    h & 0xbfff
                } else {
                    h
                }
            })
            .collect()
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn simd_tiers() -> Vec<Isa> {
    let all = Isa::all();
    assert!(all[0] == Isa::SCALAR);
    all
}

#[test]
fn this_host_has_a_simd_tier() {
    // A host without one would make every test below compare scalar with scalar.
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    assert!(simd_tiers().len() > 1, "{:?}", simd_tiers());
    eprintln!(
        "tiers: {:?}",
        simd_tiers().iter().map(|i| i.name()).collect::<Vec<_>>()
    );
}

#[test]
fn ternary_gemv_matches_the_oracle() {
    let mut rng = Rng(1);
    for (rows, cols) in [
        (1usize, 128usize),
        (3, 256),
        (17, 640),
        (9, 2560),
        (5, 6912),
    ] {
        let trits: Vec<i8> = (0..rows * cols)
            .map(|_| (rng.next() % 3) as i8 - 1)
            .collect();
        let packed = pack(&trits, rows, cols, 1.0).unwrap();
        let codes = &packed[..rows * cols / 4];
        for case in 0..4 {
            let a: Vec<i8> = match case {
                0 => (0..cols).map(|_| (rng.next() % 256) as i8).collect(),
                1 => vec![-128; cols],
                2 => vec![127; cols],
                _ => (0..cols)
                    .map(|i| if i % 2 == 0 { -128 } else { 127 })
                    .collect(),
            };
            let mut want = vec![0i32; rows];
            rusty_ternary::kernel::gemv_codes(codes, cols, &a, &mut want).unwrap();
            for isa in simd_tiers() {
                let mut got = vec![0i32; rows];
                ternary_gemv(isa, codes, &a, act_sum(&a), &mut got);
                assert_eq!(got, want, "{} {rows}x{cols} case {case}", isa.name());
            }
        }
    }
    // The extremes, every weight +1 and -1: the sums that would wrap a 16-bit accumulator.
    for t in [1i8, -1] {
        let cols = 6912;
        let packed = pack(&vec![t; cols], 1, cols, 1.0).unwrap();
        for v in [-128i8, 127] {
            let a = vec![v; cols];
            for isa in simd_tiers() {
                let mut y = [0i32];
                ternary_gemv(isa, &packed[..cols / 4], &a, act_sum(&a), &mut y);
                assert_eq!(y[0], i32::from(t) * i32::from(v) * 6912, "{}", isa.name());
            }
        }
    }
}

#[test]
fn ternary_gemm_matches_the_oracle() {
    let mut rng = Rng(13);
    for (rows, cols, m) in [
        (1usize, 128usize, 1usize),
        (3, 256, 7),
        (5, 640, 8),
        (4, 2560, 9),
        (3, 6912, 17),
        (2, 2560, 64),
        (7, 256, 13),
        (6, 640, 4),
    ] {
        let trits: Vec<i8> = (0..rows * cols)
            .map(|_| (rng.next() % 3) as i8 - 1)
            .collect();
        let packed = pack(&trits, rows, cols, 1.0).unwrap();
        let codes = &packed[..rows * cols / 4];
        for case in 0..3 {
            let a: Vec<i8> = match case {
                0 => (0..m * cols).map(|_| (rng.next() % 256) as i8).collect(),
                1 => vec![-128; m * cols],
                _ => vec![127; m * cols],
            };
            let sums: Vec<i32> = a.chunks_exact(cols).map(act_sum).collect();
            let mut by_token = vec![0i32; m * rows];
            rusty_ternary::kernel::gemm_codes(codes, rows, cols, m, &a, &mut by_token).unwrap();
            // The oracle is `[m, rows]`; the kernel's output is `[rows, m]`.
            let want: Vec<i32> = (0..rows * m)
                .map(|i| by_token[(i % m) * rows + i / m])
                .collect();
            for isa in simd_tiers() {
                let mut got = vec![0i32; rows * m];
                ternary_gemm(isa, codes, cols, m, &a, &sums, &mut got);
                assert_eq!(got, want, "{} {rows}x{cols} m {m} case {case}", isa.name());
            }
        }
    }
}

#[test]
fn i8f_matvec_matches_the_oracle() {
    let mut rng = Rng(12);
    for (rows, n) in [
        (1usize, 1usize),
        (3, 15),
        (4, 16),
        (5, 17),
        (7, 33),
        (9, 128),
        (5, 2560),
        (3, 2567),
    ] {
        let q: Vec<i8> = (0..rows * n)
            .map(|_| {
                i8::try_from(rng.next() % 255)
                    .unwrap_or(0)
                    .wrapping_sub(127)
            })
            .collect();
        let x: Vec<f32> = rng
            .floats(n, 30.0)
            .iter()
            .map(|&v| num::round_f16(v))
            .collect();
        let mut want = vec![0f32; rows];
        i8f_matvec_scalar(&q, &x, &mut want);
        for isa in simd_tiers() {
            let mut got = vec![0f32; rows];
            i8f_matvec(isa, &q, &x, &mut got);
            assert_eq!(bits(&got), bits(&want), "{} {rows}x{n}", isa.name());
        }
    }
}

#[test]
fn i8_gemv_matches_the_oracle() {
    let mut rng = Rng(11);
    for (rows, n) in [
        (1usize, 1usize),
        (3, 15),
        (4, 16),
        (5, 31),
        (7, 32),
        (9, 33),
        (17, 2560),
        (3, 2575),
    ] {
        for case in 0..3 {
            let pick = |rng: &mut Rng| -> i8 {
                match case {
                    0 => i8::try_from(rng.next() % 255)
                        .unwrap_or(0)
                        .wrapping_sub(127),
                    1 => {
                        if rng.next().is_multiple_of(2) {
                            127
                        } else {
                            -127
                        }
                    }
                    _ => 127,
                }
            };
            let q: Vec<i8> = (0..rows * n).map(|_| pick(&mut rng)).collect();
            let a: Vec<i8> = (0..n).map(|_| pick(&mut rng)).collect();
            let mut want = vec![0i32; rows];
            i8_gemv_scalar(&q, &a, &mut want);
            for isa in simd_tiers() {
                let mut got = vec![0i32; rows];
                i8_gemv(isa, &q, &a, &mut got);
                assert_eq!(got, want, "{} {rows}x{n} case {case}", isa.name());
            }
        }
    }
}

#[test]
fn f16_matvec_matches_the_oracle() {
    let mut rng = Rng(2);
    for (rows, len) in [
        (1usize, 1usize),
        (3, 15),
        (4, 16),
        (5, 17),
        (7, 33),
        (9, 128),
        (3, 2560),
        (2, 2567),
    ] {
        let m = rng.halves(rows * len);
        let mb: Vec<u8> = m.iter().flat_map(|h| h.to_le_bytes()).collect();
        let x = rng.floats(len, 3.0);
        let want: Vec<f32> = (0..rows)
            .map(|r| num::dot16_f16(&m[r * len..(r + 1) * len], &x))
            .collect();
        for isa in simd_tiers() {
            let mut got = vec![0f32; rows];
            f16_matvec_bytes(isa, &mb, &x, &mut got);
            assert_eq!(bits(&got), bits(&want), "{} bytes {rows}x{len}", isa.name());
        }
        // Strided: rows `stride` apart starting at `offset`, as in the KV cache.
        let (offset, stride) = (3usize, len + 5);
        let v = rng.halves(offset + rows * stride);
        let want: Vec<f32> = (0..rows)
            .map(|r| num::dot16_f16(&v[offset + r * stride..offset + r * stride + len], &x))
            .collect();
        for isa in simd_tiers() {
            let mut got = vec![0f32; rows];
            f16_matvec_strided(isa, &v, offset, stride, &x, &mut got);
            assert_eq!(
                bits(&got),
                bits(&want),
                "{} strided {rows}x{len}",
                isa.name()
            );
        }
    }
}

#[test]
fn weighted_sum_matches_the_oracle() {
    let mut rng = Rng(3);
    for (n, len) in [
        (1usize, 1usize),
        (2, 7),
        (5, 16),
        (9, 31),
        (33, 128),
        (300, 128),
        (4, 130),
    ] {
        let (offset, stride) = (7usize, len + 3);
        let v = rng.halves(offset + n * stride);
        let p: Vec<f32> = rng
            .floats(n, 1.0)
            .iter()
            .map(|x| num::round_f16(x.abs()))
            .collect();
        let mut want = vec![0f32; len];
        num::weighted_sum_f16(&p, &v[offset..], stride, &mut want);
        for isa in simd_tiers() {
            let mut got = vec![9f32; len];
            weighted_sum_f16(isa, &p, &v, offset, stride, &mut got);
            assert_eq!(bits(&got), bits(&want), "{} {n}x{len}", isa.name());
        }
    }
}

#[test]
fn act_quant_matches_the_oracle() {
    let mut rng = Rng(4);
    for n in [1usize, 3, 15, 16, 17, 31, 32, 33, 63, 2560, 6912] {
        for scale in [1e-7f32, 0.01, 1.0, 300.0] {
            let mut x = rng.floats(n, scale);
            if n > 4 {
                // Exact ties of the rounding: x·s = k + 0.5.
                x[1] = scale * (0.5 / 127.0);
                x[2] = -scale * (2.5 / 127.0);
            }
            let mut want = vec![0i8; n];
            let ws = rusty_ternary::quant::act_quant(&x, &mut want);
            for isa in simd_tiers() {
                let mut got = vec![0i8; n];
                let (gs, gsum) = act_quant_sum(isa, &x, &mut got);
                assert_eq!(
                    gsum,
                    act_sum(&want),
                    "{} n {n} scale {scale}: the fused sum",
                    isa.name()
                );
                assert_eq!(
                    gs.to_bits(),
                    ws.to_bits(),
                    "{} n {n} scale {scale}",
                    isa.name()
                );
                assert_eq!(got, want, "{} n {n} scale {scale}", isa.name());
            }
        }
    }
}

#[test]
fn rms_norm_matches_the_oracle() {
    let mut rng = Rng(5);
    for n in [1usize, 3, 4, 5, 7, 8, 9, 15, 128, 2560, 6912, 6915] {
        let x = rng.floats(n, 4.0);
        let w = rng.floats(n, 2.0);
        let mut want = vec![0f32; n];
        num::rms_norm(&x, &w, 1e-5, &mut want);
        for isa in simd_tiers() {
            let mut got = vec![0f32; n];
            rms_norm(isa, &x, &w, 1e-5, &mut got);
            assert_eq!(bits(&got), bits(&want), "{} n {n}", isa.name());
        }
    }
}

#[test]
fn softmax_matches_the_oracle() {
    let mut rng = Rng(6);
    for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 1024, 1073] {
        for scale in [0.1f32, 5.0, 60.0, 200.0] {
            let x = rng.floats(n, scale);
            let mut want = x.clone();
            num::softmax(&mut want, 0.088_388_35);
            for isa in simd_tiers() {
                let mut got = x.clone();
                softmax(isa, &mut got, 0.088_388_35);
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{} n {n} scale {scale}",
                    isa.name()
                );
            }
        }
    }
}

#[test]
fn exp_twins_match_on_a_dense_sweep() {
    // Softmax feeds exp with x - max <= 0; sweep that range densely plus the clamps.
    let mut x: Vec<f32> = (0..=100_000)
        .map(|i| -100.0 + 100.0 * (i as f32 / 100_000.0))
        .collect();
    x.extend([0.0, -0.0, -87.0, -87.000_01, -1e30, f32::NEG_INFINITY]);
    // Softmax of a row with a single 0 max: every element's exp is exp_canon(x).
    let want: Vec<u32> = x.iter().map(|&v| num::exp_canon(v).to_bits()).collect();
    for isa in simd_tiers() {
        // Through softmax's first pass: prepend the max (0) so x - max = x exactly.
        let mut row = vec![0f32];
        row.extend(&x);
        let mut got = row.clone();
        softmax(isa, &mut got, 1.0);
        let mut want_row = row.clone();
        num::softmax(&mut want_row, 1.0);
        assert_eq!(bits(&got), bits(&want_row), "{}", isa.name());
    }
    assert_eq!(want.len(), x.len());
}

#[test]
fn f16_to_f32_is_exact_for_every_half() {
    let all: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    let mut want = vec![0f32; 65536];
    f16_bytes_to_f32(Isa::SCALAR, &all, &mut want);
    for isa in simd_tiers() {
        let mut got = vec![0f32; 65536];
        f16_bytes_to_f32(isa, &all, &mut got);
        for (h, (g, w)) in got.iter().zip(&want).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "{} half {h:#06x}", isa.name());
            } else {
                assert_eq!(g.to_bits(), w.to_bits(), "{} half {h:#06x}", isa.name());
            }
        }
    }
}

fn check_f2h(isa: Isa, src: &[f32]) {
    let mut want = vec![0u16; src.len()];
    f32_to_f16(Isa::SCALAR, src, &mut want);
    let mut got = vec![0u16; src.len()];
    f32_to_f16(isa, src, &mut got);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        if src[i].is_nan() {
            assert!(
                (g & 0x7c00) == 0x7c00 && (g & 0x3ff) != 0,
                "{} NaN {:#010x} -> {g:#06x}",
                isa.name(),
                src[i].to_bits()
            );
        } else {
            assert_eq!(g, w, "{} f32 {:#010x}", isa.name(), src[i].to_bits());
        }
    }
    let mut r = src.to_vec();
    round_f16_slice(isa, &mut r);
    for (i, v) in r.iter().enumerate() {
        let w = num::round_f16(src[i]);
        assert!(
            if w.is_nan() {
                v.is_nan()
            } else {
                v.to_bits() == w.to_bits()
            },
            "{} round {:#010x}",
            isa.name(),
            src[i].to_bits()
        );
    }
}

#[test]
fn f32_to_f16_matches_on_every_16th_float_and_the_boundaries() {
    // Every 16th bit pattern (268M) is too slow for a unit test; take every 4099th (a prime
    // stride, so every exponent and every low-mantissa residue is hit), plus every pattern
    // around the rounding and range boundaries.
    let mut src: Vec<f32> = (0..=u32::MAX / 4099)
        .map(|i| f32::from_bits(i * 4099))
        .collect();
    for edge in [
        0x3880_0000u32,
        0x3300_0000,
        0x477f_e000,
        0x4780_0000,
        0x7f80_0000,
        0x3380_0000,
        0x387f_c000,
    ] {
        for d in 0..4096u32 {
            for s in [0u32, 0x8000_0000] {
                src.push(f32::from_bits((edge - 2048 + d) | s));
            }
        }
    }
    for isa in simd_tiers() {
        check_f2h(isa, &src);
    }
}

/// Every one of the 2^32 floats (a minute or so in release). `cargo test --release -- --ignored`.
#[test]
#[ignore = "exhaustive: every f32 bit pattern"]
fn f32_to_f16_matches_on_every_float() {
    let chunk = 1usize << 24;
    for c in 0..256u32 {
        let src: Vec<f32> = (0..chunk as u32)
            .map(|i| f32::from_bits((c << 24) | i))
            .collect();
        for isa in simd_tiers().into_iter().skip(1) {
            check_f2h(isa, &src);
        }
    }
}

#[test]
fn elementwise_twins_match_the_oracles() {
    let mut rng = Rng(7);
    for n in [1usize, 3, 4, 7, 8, 9, 15, 16, 17, 64, 2560, 6913] {
        let sum: Vec<i32> = (0..n)
            .map(|_| (rng.next() % 1_769_473) as i32 - 884_736)
            .collect();
        let mut want = vec![0f32; n];
        num::finish_rows(&sum, 37.25, 1.218_850_5, &mut want);
        let g = rng.floats(n, 8.0);
        let u = rng.floats(n, 8.0);
        let mut g_want = g.clone();
        num::relu2_mul(&mut g_want, &u);
        let mut a_want = g.clone();
        num::add_assign(&mut a_want, &u);
        for isa in simd_tiers() {
            let mut got = vec![0f32; n];
            finish_rows(isa, &sum, 37.25, 1.218_850_5, &mut got);
            assert_eq!(bits(&got), bits(&want), "{} finish {n}", isa.name());
            let mut gg = g.clone();
            relu2_mul(isa, &mut gg, &u);
            assert_eq!(bits(&gg), bits(&g_want), "{} relu2 {n}", isa.name());
            let mut aa = g.clone();
            add_assign(isa, &mut aa, &u);
            assert_eq!(bits(&aa), bits(&a_want), "{} add {n}", isa.name());
        }
    }
    for half in [1usize, 3, 4, 8, 9, 64] {
        let (mut c, mut s) = (vec![0f32; half], vec![0f32; half]);
        num::rope_table(1234, 2 * half, 500_000.0, &mut c, &mut s);
        let x = rng.floats(2 * half, 3.0);
        let mut want = x.clone();
        num::rope_neox(&mut want, &c, &s);
        for isa in simd_tiers() {
            let mut got = x.clone();
            rope_neox(isa, &mut got, &c, &s);
            assert_eq!(bits(&got), bits(&want), "{} rope {half}", isa.name());
        }
    }
}

#[test]
fn the_census_counts_the_tier_that_ran() {
    census::reset();
    let isa = Isa::best();
    let mut y = [0f32; 4];
    add_assign(isa, &mut y, &[1.0; 4]);
    let snap = census::snapshot();
    assert!(
        snap.iter()
            .any(|&(k, t, n)| k == "elementwise" && t == isa.name() && n >= 4),
        "{snap:?}"
    );
}

#[test]
fn an_unsupported_tier_cannot_be_named() {
    assert_eq!(Isa::named("scalar"), Some(Isa::SCALAR));
    assert_eq!(Isa::named("no-such-isa"), None);
    #[cfg(not(target_arch = "aarch64"))]
    assert_eq!(Isa::named("neon"), None);
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(Isa::named("avx2"), None);
}
