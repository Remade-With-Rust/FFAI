# H-10 worklist — cheapest first

Generated 2026-08-18 from `cargo vet suggest`. Regenerate any time with that command.

`cargo vet certify` records **a named person** having read the code, which is why this
file stops at the commands and does not run them. Read, then certify.

## Part 1 — version diffs (79 crates)

An audit already exists for a nearby version; only the delta needs reading. This is by far
the best value in the backlog and several are two-line version bumps.

| # | crate | from → to | files | churn | command |
|--:|---|---|--:|--:|---|
| 1 | `futures-core` | 0.3.32 → 0.3.33 | 2 | 4 | `cargo vet diff futures-core 0.3.32 0.3.33` |
| 2 | `futures-macro` | 0.3.32 → 0.3.33 | 2 | 4 | `cargo vet diff futures-macro 0.3.32 0.3.33` |
| 3 | `futures-sink` | 0.3.32 → 0.3.33 | 2 | 4 | `cargo vet diff futures-sink 0.3.32 0.3.33` |
| 4 | `quinn-udp` | 0.5.14 → 0.5.15 | 2 | 8 | `cargo vet diff quinn-udp 0.5.14 0.5.15` |
| 5 | `futures-io` | 0.3.31 → 0.3.33 | 2 | 16 | `cargo vet diff futures-io 0.3.31 0.3.33` |
| 6 | `zerofrom` | 0.1.7 → 0.1.8 | 2 | 17 | `cargo vet diff zerofrom 0.1.7 0.1.8` |
| 7 | `wasm-bindgen-macro` | 0.2.105 → 0.2.121 | 3 | 24 | `cargo vet diff wasm-bindgen-macro 0.2.105 0.2.121` |
| 8 | `crypto-common` | 0.1.6 → 0.1.7 | 3 | 33 | `cargo vet diff crypto-common 0.1.6 0.1.7` |
| 9 | `stable_deref_trait` | 1.2.0 → 1.2.1 | 3 | 63 | `cargo vet diff stable_deref_trait 1.2.0 1.2.1` |
| 10 | `rand_core` | 0.10.0 → 0.10.1 | 4 | 19 | `cargo vet diff rand_core 0.10.0 0.10.1` |
| 11 | `idna_adapter` | 1.2.1 → 1.2.2 | 4 | 37 | `cargo vet diff idna_adapter 1.2.1 1.2.2` |
| 12 | `version_check` | 0.9.4 → 0.9.5 | 4 | 59 | `cargo vet diff version_check 0.9.4 0.9.5` |
| 13 | `webpki-root-certs` | 1.0.4 → 1.0.9 | 4 | 80 | `cargo vet diff webpki-root-certs 1.0.4 1.0.9` |
| 14 | `find-msvc-tools` | 0.1.8 → 0.1.9 | 4 | 85 | `cargo vet diff find-msvc-tools 0.1.8 0.1.9` |
| 15 | `pkg-config` | 0.3.32 → 0.3.33 | 5 | 27 | `cargo vet diff pkg-config 0.3.32 0.3.33` |
| 16 | `rusty-fork` | 0.3.0 → 0.3.1 | 5 | 69 | `cargo vet diff rusty-fork 0.3.0 0.3.1` |
| 17 | `rustc-hash` | 2.1.1 → 2.1.3 | 5 | 80 | `cargo vet diff rustc-hash 2.1.1 2.1.3` |
| 18 | `rand` | 0.9.4 → 0.9.5 | 6 | 55 | `cargo vet diff rand 0.9.4 0.9.5` |
| 19 | `futures-executor` | 0.3.31 → 0.3.33 | 6 | 89 | `cargo vet diff futures-executor 0.3.31 0.3.33` |
| 20 | `openssl-probe` | 0.1.6 → 0.2.1 | 6 | 402 | `cargo vet diff openssl-probe 0.1.6 0.2.1` |
| 21 | `hermit-abi` | 0.2.6 → 0.3.3 | 6 | 1040 | `cargo vet diff hermit-abi 0.2.6 0.3.3` |
| 22 | `shlex` | 1.3.0 → 2.0.1 | 7 | 281 | `cargo vet diff shlex 1.3.0 2.0.1` |
| 23 | `wasm-bindgen-shared` | 0.2.100 → 0.2.121 | 7 | 345 | `cargo vet diff wasm-bindgen-shared 0.2.100 0.2.121` |
| 24 | `bytemuck_derive` | 1.9.3 → 1.11.0 | 7 | 570 | `cargo vet diff bytemuck_derive 1.9.3 1.11.0` |
| 25 | `webpki-roots` | 0.26.8 → 1.0.9 | 7 | 1899 | `cargo vet diff webpki-roots 0.26.8 1.0.9` |
| 26 | `webpki-roots` | 0.26.8 → 0.26.11 | 7 | 5910 | `cargo vet diff webpki-roots 0.26.8 0.26.11` |
| 27 | `arrayvec` | 0.7.6 → 0.7.8 | 8 | 137 | `cargo vet diff arrayvec 0.7.6 0.7.8` |
| 28 | `futures-channel` | 0.3.31 → 0.3.33 | 8 | 276 | `cargo vet diff futures-channel 0.3.31 0.3.33` |
| 29 | `fastrand` | 2.3.0 → 2.5.0 | 8 | 448 | `cargo vet diff fastrand 2.3.0 2.5.0` |
| 30 | `safetensors` | 0.3.3 → 0.4.5 | 8 | 721 | `cargo vet diff safetensors 0.3.3 0.4.5` |
| 31 | `oneshot` | 0.1.11 → 0.1.13 | 8 | 815 | `cargo vet diff oneshot 0.1.11 0.1.13` |
| 32 | `crypto-common` | 0.1.6 → 0.2.2 | 8 | 928 | `cargo vet diff crypto-common 0.1.6 0.2.2` |
| 33 | `raw-cpuid` | 11.5.0 → 11.6.0 | 8 | 1219 | `cargo vet diff raw-cpuid 11.5.0 11.6.0` |
| 34 | `tracing-attributes` | 0.1.30 → 0.1.31 | 9 | 220 | `cargo vet diff tracing-attributes 0.1.30 0.1.31` |
| 35 | `mimalloc` | 0.1.37 → 0.1.52 | 9 | 398 | `cargo vet diff mimalloc 0.1.37 0.1.52` |
| 36 | `tower-layer` | 0.1.0 → 0.3.2 | 9 | 690 | `cargo vet diff tower-layer 0.1.0 0.3.2` |
| 37 | `block-buffer` | 0.10.4 → 0.12.1 | 9 | 1245 | `cargo vet diff block-buffer 0.10.4 0.12.1` |
| 38 | `safetensors` | 0.3.3 → 0.8.0 | 9 | 1873 | `cargo vet diff safetensors 0.3.3 0.8.0` |
| 39 | `iana-time-zone` | 0.1.64 → 0.1.65 | 10 | 179 | `cargo vet diff iana-time-zone 0.1.64 0.1.65` |
| 40 | `wasm-bindgen-futures` | 0.4.49 → 0.4.71 | 10 | 1047 | `cargo vet diff wasm-bindgen-futures 0.4.49 0.4.71` |
| 41 | `log` | 0.4.29 → 0.4.33 | 11 | 382 | `cargo vet diff log 0.4.29 0.4.33` |
| 42 | `objc2` | 0.6.3 → 0.6.4 | 11 | 450 | `cargo vet diff objc2 0.6.3 0.6.4` |
| 43 | `crc32fast` | 1.4.2 → 1.5.0 | 12 | 315 | `cargo vet diff crc32fast 1.4.2 1.5.0` |
| 44 | `chacha20` | 0.10.0 → 0.10.1 | 12 | 318 | `cargo vet diff chacha20 0.10.0 0.10.1` |
| 45 | `bitflags` | 2.11.1 → 2.13.1 | 12 | 721 | `cargo vet diff bitflags 2.11.1 2.13.1` |
| 46 | `cc` | 1.2.53 → 1.4.0 | 12 | 1065 | `cargo vet diff cc 1.2.53 1.4.0` |
| 47 | `getset` | 0.1.3 → 0.1.7 | 12 | 1148 | `cargo vet diff getset 0.1.3 0.1.7` |
| 48 | `rand` | 0.10.1 → 0.10.2 | 13 | 276 | `cargo vet diff rand 0.10.1 0.10.2` |
| 49 | `castaway` | 0.2.2 → 0.2.4 | 13 | 315 | `cargo vet diff castaway 0.2.2 0.2.4` |
| 50 | `tracing-core` | 0.1.34 → 0.1.36 | 13 | 410 | `cargo vet diff tracing-core 0.1.34 0.1.36` |
| 51 | `zeroize` | 1.8.2 → 1.9.0 | 13 | 798 | `cargo vet diff zeroize 1.8.2 1.9.0` |
| 52 | `tinyvec` | 1.9.0 → 1.12.0 | 13 | 10412 | `cargo vet diff tinyvec 1.9.0 1.12.0` |
| 53 | `js-sys` | 0.3.77 → 0.3.98 | 13 | 19312 | `cargo vet diff js-sys 0.3.77 0.3.98` |
| 54 | `simd-adler32` | 0.3.7 → 0.3.10 | 15 | 500 | `cargo vet diff simd-adler32 0.3.7 0.3.10` |
| 55 | `half` | 2.5.0 → 2.7.1 | 15 | 590 | `cargo vet diff half 2.5.0 2.7.1` |
| 56 | `futures` | 0.3.31 → 0.3.33 | 16 | 328 | `cargo vet diff futures 0.3.31 0.3.33` |
| 57 | `libloading` | 0.9.0 → 0.8.9 | 16 | 1182 | `cargo vet diff libloading 0.9.0 0.8.9` |
| 58 | `untrusted` | 0.7.1 → 0.9.0 | 16 | 1402 | `cargo vet diff untrusted 0.7.1 0.9.0` |
| 59 | `core-foundation` | 0.10.0 → 0.10.1 | 17 | 193 | `cargo vet diff core-foundation 0.10.0 0.10.1` |
| 60 | `constant_time_eq` | 0.3.1 → 0.4.2 | 18 | 2340 | `cargo vet diff constant_time_eq 0.3.1 0.4.2` |
| 61 | `pin-project-lite` | 0.2.16 → 0.2.17 | 19 | 292 | `cargo vet diff pin-project-lite 0.2.16 0.2.17` |
| 62 | `bytemuck` | 1.22.0 → 1.25.2 | 19 | 358 | `cargo vet diff bytemuck 1.22.0 1.25.2` |
| 63 | `wasm-bindgen` | 0.2.100 → 0.2.121 | 20 | 5194 | `cargo vet diff wasm-bindgen 0.2.100 0.2.121` |
| 64 | `tokio-rustls` | 0.24.0 → 0.26.4 | 21 | 2880 | `cargo vet diff tokio-rustls 0.24.0 0.26.4` |
| 65 | `time-macros` | 0.2.27 → 0.2.32 | 22 | 2665 | `cargo vet diff time-macros 0.2.27 0.2.32` |
| 66 | `zlib-rs` | 0.6.3 → 0.6.6 | 28 | 1373 | `cargo vet diff zlib-rs 0.6.3 0.6.6` |
| 67 | `compact_str` | 0.7.1 → 0.9.1 | 31 | 3936 | `cargo vet diff compact_str 0.7.1 0.9.1` |
| 68 | `tracing-subscriber` | 0.3.20 → 0.3.23 | 33 | 897 | `cargo vet diff tracing-subscriber 0.3.20 0.3.23` |
| 69 | `ureq` | 2.9.1 → 2.12.1 | 34 | 1253 | `cargo vet diff ureq 2.9.1 2.12.1` |
| 70 | `zip` | 2.4.2 → 8.6.0 | 47 | 13402 | `cargo vet diff zip 2.4.2 8.6.0` |
| 71 | `ttf-parser` | 0.19.0 → 0.25.1 | 48 | 5478 | `cargo vet diff ttf-parser 0.19.0 0.25.1` |
| 72 | `sha2` | 0.10.9 → 0.11.0 | 51 | 2860 | `cargo vet diff sha2 0.10.9 0.11.0` |
| 73 | `tracing` | 0.1.41 → 0.1.44 | 65 | 61824 | `cargo vet diff tracing 0.1.41 0.1.44` |
| 74 | `jni` | 0.21.1 → 0.22.4 | 143 | 33032 | `cargo vet diff jni 0.21.1 0.22.4` |
| 75 | `zerocopy-derive` | 0.8.27 → 0.8.55 | 166 | 31448 | `cargo vet diff zerocopy-derive 0.8.27 0.8.55` |
| 76 | `rustls` | 0.21.6 → 0.23.42 | 180 | 55703 | `cargo vet diff rustls 0.21.6 0.23.42` |
| 77 | `rustls-webpki` | 0.101.4 → 0.103.10 | 255 | 16322 | `cargo vet diff rustls-webpki 0.101.4 0.103.10` |
| 78 | `zerocopy` | 0.8.27 → 0.8.55 | 651 | 29601 | `cargo vet diff zerocopy 0.8.27 0.8.55` |
| 79 | `web-sys` | 0.3.76 → 0.3.98 | 1557 | 59748 | `cargo vet diff web-sys 0.3.76 0.3.98` |

## Part 2 — full reads, smallest first (showing 40 of 232)

| # | crate | version | lines | command |
|--:|---|---|--:|---|
| 1 | `candle-ug` | 0.11.0 | 84 | `cargo vet inspect candle-ug 0.11.0` |
| 2 | `tokio_with_wasm_proc` | 0.8.8 | 96 | `cargo vet inspect tokio_with_wasm_proc 0.8.8` |
| 3 | `darling_macro` | 0.13.4 | 113 | `cargo vet inspect darling_macro 0.13.4` |
| 4 | `darling_macro` | 0.23.0 | 125 | `cargo vet inspect darling_macro 0.23.0` |
| 5 | `ffai-argus` | 0.6.1 | 165 | `cargo vet inspect ffai-argus 0.6.1` |
| 6 | `pulp-wasm-simd-flag` | 0.1.1 | 202 | `cargo vet inspect pulp-wasm-simd-flag 0.1.1` |
| 7 | `dyn-stack-macros` | 0.1.3 | 216 | `cargo vet inspect dyn-stack-macros 0.1.3` |
| 8 | `git-version` | 0.3.9 | 240 | `cargo vet inspect git-version 0.3.9` |
| 9 | `jni-sys-macros` | 0.4.1 | 261 | `cargo vet inspect jni-sys-macros 0.4.1` |
| 10 | `rusty_alloc-api` | 0.3.2 | 298 | `cargo vet inspect rusty_alloc-api 0.3.2` |
| 11 | `ffai-models` | 0.6.1 | 332 | `cargo vet inspect ffai-models 0.6.1` |
| 12 | `rff-codec` | 0.1.0 | 333 | `cargo vet inspect rff-codec 0.1.0` |
| 13 | `intel-mkl-src` | 0.8.1 | 339 | `cargo vet inspect intel-mkl-src 0.8.1` |
| 14 | `rff-format` | 0.1.0 | 339 | `cargo vet inspect rff-format 0.1.0` |
| 15 | `derive_builder_macro` | 0.20.2 | 376 | `cargo vet inspect derive_builder_macro 0.20.2` |
| 16 | `wasite` | 1.0.2 | 405 | `cargo vet inspect wasite 1.0.2` |
| 17 | `ffai-wasm` | 0.1.0 | 470 | `cargo vet inspect ffai-wasm 0.1.0` |
| 18 | `block` | 0.1.6 | 492 | `cargo vet inspect block 0.1.6` |
| 19 | `rusty_h264` | 0.8.0 | 496 | `cargo vet inspect rusty_h264 0.8.0` |
| 20 | `sync_wrapper` | 0.1.2 | 496 | `cargo vet inspect sync_wrapper 0.1.2` |
| 21 | `reborrow` | 0.5.5 | 504 | `cargo vet inspect reborrow 0.5.5` |
| 22 | `git-version-macro` | 0.3.9 | 518 | `cargo vet inspect git-version-macro 0.3.9` |
| 23 | `rawpointer` | 0.2.1 | 559 | `cargo vet inspect rawpointer 0.2.1` |
| 24 | `env_home` | 0.1.0 | 564 | `cargo vet inspect env_home 0.1.0` |
| 25 | `urlencoding` | 2.1.3 | 571 | `cargo vet inspect urlencoding 2.1.3` |
| 26 | `pyo3-macros` | 0.23.5 | 577 | `cargo vet inspect pyo3-macros 0.23.5` |
| 27 | `rff-format-mkv` | 0.1.0 | 584 | `cargo vet inspect rff-format-mkv 0.1.0` |
| 28 | `tower-service` | 0.3.2 | 607 | `cargo vet inspect tower-service 0.3.2` |
| 29 | `gemm-f64` | 0.19.0 | 646 | `cargo vet inspect gemm-f64 0.19.0` |
| 30 | `gemm-f64` | 0.18.2 | 651 | `cargo vet inspect gemm-f64 0.18.2` |
| 31 | `gemm-f32` | 0.19.0 | 699 | `cargo vet inspect gemm-f32 0.19.0` |
| 32 | `gemm-f32` | 0.18.2 | 704 | `cargo vet inspect gemm-f32 0.18.2` |
| 33 | `lru-slab` | 0.1.2 | 778 | `cargo vet inspect lru-slab 0.1.2` |
| 34 | `gemm-c64` | 0.19.0 | 779 | `cargo vet inspect gemm-c64 0.19.0` |
| 35 | `gemm-c64` | 0.18.2 | 784 | `cargo vet inspect gemm-c64 0.18.2` |
| 36 | `rff-core` | 0.1.1 | 787 | `cargo vet inspect rff-core 0.1.1` |
| 37 | `intel-mkl-tool` | 0.8.1 | 802 | `cargo vet inspect intel-mkl-tool 0.8.1` |
| 38 | `symlink` | 0.1.0 | 805 | `cargo vet inspect symlink 0.1.0` |
| 39 | `gemm-c32` | 0.19.0 | 816 | `cargo vet inspect gemm-c32 0.19.0` |
| 40 | `dirs-sys` | 0.5.0 | 819 | `cargo vet inspect dirs-sys 0.5.0` |

Those 40 total **19,432 lines** — the whole tail is cheap; the cost is concentrated
in a handful of giants documented in `README.md`.

## Part 3 — first-party, a policy decision rather than an audit

26 exemptions are the org's own code: 10 FFai workspace members and 16 sibling
`rff-*` / `rusty_*` crates. They appear only because `audit-as-crates-io = true`
makes `cargo vet` treat first-party code as third-party.

Keep it if the intent is to catch a published version drifting from local source —
that is a real supply-chain risk for a multi-repo org. Otherwise setting it `false`
for the workspace members removes 10 at a stroke, and the 16 siblings are certifiable
by whoever maintains them, which is this org.

