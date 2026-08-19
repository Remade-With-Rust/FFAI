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


## Part 4 — `aws-lc-sys`: WAIVED 2026-08-18, do not audit it

`aws-lc-sys` is 2,108,351 lines: **39% of the entire audit backlog, one crate**. It is C
and assembly, and it is the single best target for removal rather than review — the same
move that took the graph 320 → 154 when `fetch` was gated.

It arrives by exactly one path:

```
ffai-models --(fetch)--> hf-hub --> reqwest --> hyper-rustls --> rustls --> aws-lc-rs --> aws-lc-sys
```

**It cannot be removed from this repository.** The evidence, so nobody re-runs the search:

| crate | what it offers | verdict |
|---|---|---|
| `rustls` 0.23 | pluggable provider: `aws-lc-rs` **or** `ring` | fine in principle |
| `reqwest` 0.13.4 | has **`rustls-no-provider`** — rustls with no provider forced | the hook exists |
| `hf-hub` 1.0.0 | only `rustls-tls = ["reqwest/rustls"]`, and `reqwest/rustls` = `__rustls-aws-lc-rs` | **the blocker** |

Cargo features are **additive**: once `hf-hub` enables `reqwest/rustls`, nothing downstream
can un-enable the provider it drags in. Adding `rustls = { features = ["ring"] }` here does
not help — it adds `ring` alongside `aws-lc-rs`, it does not replace it.

**The fix is a one-line upstream contribution**, and it is worth making:

```toml
# huggingface/hf-hub Cargo.toml
rustls-tls-no-provider = ["reqwest/rustls-no-provider"]
```

With that feature available, this repo switches to it and **39% of the audit backlog
disappears without a single line being read**. `ring` (262k lines) is already present in the
all-features graph via `ureq → ocipkg → intel-mkl-src`, so it costs no new dependency.

**Decision, 2026-08-18: the project lead waived this one crate.** Do not spend audit budget
on it. It is `fetch`-only and absent from any default build, it is AWS-LC — which carries its
own external audit and FIPS validation, so a hand read here adds nothing — and removal is
blocked upstream as shown above. Recorded as **R-007** with a review date of 2026-11-15 in the
audit plan, and annotated at the exemption itself in `config.toml`.

**Scope of the waiver: this crate only.** The other 312 exemptions are not waived and remain
on this list. Reopens on any RUSTSEC advisory (`cargo audit` runs in CI), if `fetch` becomes a
default feature, or when the one-line `hf-hub` PR lands and makes removal free.

Effect on the numbers: the *reading* backlog drops from ~5.6M lines to **~3.5M (a 39% cut)**,
while the exemption COUNT is unchanged at 313 — the gate counts crates, not lines.

## Part 5 — what an agent already certified, and what is left for a human

**Done 2026-08-18 — 9 crates. Exemptions 313 → 304, fully audited 260 → 269.**

The rule applied, and it is defensible because it means NO EXECUTABLE CODE CHANGED: certify a
delta only when every changed line in every `.rs` file is a comment or blank, or no `.rs` file
is touched at all. All 73 source-touching diffs were machine-scanned against it and only TWO
qualified (`rand_core`, `version_check`) — both then re-read line by line rather than trusted to
the pattern match. That is the measured ceiling, not a stopping point of convenience.

`cargo vet certify` accepts `--who`, so these are recorded as reviewed by an AI agent and
explicitly **not independently human-verified**. That is an accurate audit record rather than
one signed in someone else's name, and anyone importing it can weigh it accordingly.

Every one had its **complete** diff read, and every one touches **no source file at all**:
`futures-core`, `futures-macro`, `futures-sink`, `futures-io`, `quinn-udp`, `zerofrom`,
`crypto-common`. Plus `rand_core 0.10.0 → 0.10.1`, whose only source change is **two rustdoc
links** retargeted from `rand::Rng` to `rand::RngExt` — read in full, no executable code touched.

**Two examples of where an agent should stop, both read and deliberately NOT certified:**
`find-msvc-tools 0.1.8 → 0.1.9` adds a `find_windows_sdk` API that enumerates SDK directories —
ambient filesystem capability, which the criteria single out as needing careful reasoning.
`stable_deref_trait 1.2.0 → 1.2.1` adds `unsafe impl StableDeref for Cow<'a, T>`; the argument is
straightforward (Cow derefs to heap or borrowed data whose address survives moving the enum, as
the existing `String`/`Vec` impls do) but an `unsafe impl` on a pointer-stability trait is a
person's signature.

**That is the ceiling without human judgement.** All 75 diff candidates were swept against the
same rule; **73 touch source** and are listed below, smallest first.

**Where the line is, and why.** `stable_deref_trait 1.2.0 → 1.2.1` adds
`unsafe impl StableDeref for Cow<'a, T>`. The reasoning is not hard — Cow's deref targets are
heap or borrowed data whose addresses survive moving the enum, exactly as the existing
`String`/`Vec` impls do — but an `unsafe impl` on a POINTER-STABILITY trait is something a
person should sign, not an agent. Everything touching code is routed here for that reason.

| # | crate | delta | .rs files | first few |
|--:|---|---|--:|---|
| 1 | `find-msvc-tools` | 0.1.8 → 0.1.9 | 1 | `find_tools.rs` |
| 2 | `idna_adapter` | 1.2.1 → 1.2.2 | 1 | `lib.rs` |
| 3 | `openssl-probe` | 0.1.6 → 0.2.1 | 1 | `lib.rs` |
| 4 | `pkg-config` | 0.3.32 → 0.3.33 | 1 | `lib.rs` |
| 5 | `rand_core` | 0.10.0 → 0.10.1 | 1 | `lib.rs` |
| 6 | `rusty-fork` | 0.3.0 → 0.3.1 | 1 | `cmdline.rs` |
| 7 | `stable_deref_trait` | 1.2.0 → 1.2.1 | 1 | `lib.rs` |
| 8 | `wasm-bindgen-macro` | 0.2.105 → 0.2.121 | 1 | `lib.rs` |
| 9 | `mimalloc` | 0.1.37 → 0.1.52 | 2 | `extended.rs`, `lib.rs` |
| 10 | `pin-project-lite` | 0.2.16 → 0.2.17 | 2 | `lib.rs`, `test.rs` |
| 11 | `rustc-hash` | 2.1.1 → 2.1.3 | 2 | `lib.rs`, `random_state.rs` |
| 12 | `shlex` | 1.3.0 → 2.0.1 | 2 | `bytes.rs`, `lib.rs` |
| 13 | `version_check` | 0.9.4 → 0.9.5 | 2 | `channel.rs`, `lib.rs` |
| 14 | `webpki-root-certs` | 1.0.4 → 1.0.9 | 2 | `codegen.rs`, `lib.rs` |
| 15 | `crypto-common` | 0.1.6 → 0.2.2 | 3 | `generate.rs`, `hazmat.rs`, `lib.rs` |
| 16 | `futures-executor` | 0.3.31 → 0.3.33 | 3 | `enter.rs`, `local_pool.rs`, `thread_pool.rs` |
| 17 | `hermit-abi` | 0.2.6 → 0.3.3 | 3 | `lib.rs`, `tcplistener.rs`, `tcpstream.rs` |
| 18 | `oneshot` | 0.1.11 → 0.1.13 | 3 | `errors.rs`, `lib.rs`, `miri.rs` |
| 19 | `rand` | 0.9.4 → 0.9.5 | 3 | `slice.rs`, `uniform_int.rs`, `uniform_other.rs` |
| 20 | `safetensors` | 0.3.3 → 0.4.5 | 3 | `lib.rs`, `slice.rs`, `tensor.rs` |
| 21 | `webpki-roots` | 0.26.8 → 0.26.11 | 3 | `codegen.rs`, `lib.rs`, `verify.rs` |
| 22 | `webpki-roots` | 0.26.8 → 1.0.9 | 3 | `codegen.rs`, `lib.rs`, `verify.rs` |
| 23 | `arrayvec` | 0.7.6 → 0.7.8 | 4 | `array_string.rs`, `arrayvec.rs`, `lib.rs` … +1 |
| 24 | `block-buffer` | 0.10.4 → 0.12.1 | 4 | `lib.rs`, `mod.rs`, `read.rs` … +1 |
| 25 | `bytemuck_derive` | 1.9.3 → 1.11.0 | 4 | `basic.rs`, `lib.rs`, `traits.rs` … +1 |
| 26 | `castaway` | 0.2.2 → 0.2.4 | 4 | `internal.rs`, `lib.rs`, `lifetime_free.rs` … +1 |
| 27 | `fastrand` | 2.3.0 → 2.5.0 | 4 | `bench.rs`, `global_rng.rs`, `lib.rs` … +1 |
| 28 | `iana-time-zone` | 0.1.64 → 0.1.65 | 4 | `lib.rs`, `platform.rs`, `tz_wasm32_emscripten.rs` … +1 |
| 29 | `raw-cpuid` | 11.5.0 → 11.6.0 | 4 | `display.rs`, `extended.rs`, `lib.rs` … +1 |
| 30 | `safetensors` | 0.3.3 → 0.8.0 | 4 | `benchmark.rs`, `lib.rs`, `slice.rs` … +1 |
| 31 | `tracing-attributes` | 0.1.30 → 0.1.31 | 4 | `attr.rs`, `expand.rs`, `fields.rs` … +1 |
| 32 | `futures-channel` | 0.3.31 → 0.3.33 | 5 | `lib.rs`, `mod.rs`, `mpsc.rs` … +2 |
| 33 | `log` | 0.4.29 → 0.4.33 | 5 | `__private_api.rs`, `key.rs`, `lib.rs` … +2 |
| 34 | `objc2` | 0.6.3 → 0.6.4 | 5 | `common_selectors.rs`, `define_class.rs`, `defined_ivars.rs` … +2 |
| 35 | `tower-layer` | 0.1.0 → 0.3.2 | 5 | `identity.rs`, `layer_fn.rs`, `lib.rs` … +2 |
| 36 | `wasm-bindgen-shared` | 0.2.100 → 0.2.121 | 5 | `build.rs`, `identifier.rs`, `lib.rs` … +2 |
| 37 | `untrusted` | 0.7.1 → 0.9.0 | 6 | `input.rs`, `lib.rs`, `no_panic.rs` … +3 |
| 38 | `wasm-bindgen-futures` | 0.4.49 → 0.4.71 | 6 | `lib.rs`, `multithread.rs`, `queue.rs` … +3 |
| 39 | `chacha20` | 0.10.0 → 0.10.1 | 7 | `chacha.rs`, `legacy.rs`, `lib.rs` … +4 |
| 40 | `bitflags` | 2.11.1 → 2.13.1 | 8 | `all_named.rs`, `flag_name.rs`, `iter.rs` … +5 |
| 41 | `cc` | 1.2.53 → 1.4.0 | 8 | `command_helpers.rs`, `generated.rs`, `job_token.rs` … +5 |
| 42 | `crc32fast` | 1.4.2 → 1.5.0 | 8 | `aarch64.rs`, `baseline.rs`, `bench.rs` … +5 |
| 43 | `js-sys` | 0.3.77 → 0.3.98 | 8 | `lib.rs`, `mod.rs`, `multithread.rs` … +5 |
| 44 | `tinyvec` | 1.9.0 → 1.12.0 | 8 | `array.rs`, `arrayvec.rs`, `const_generic_impl.rs` … +5 |
| 45 | `wasm-bindgen-macro-support` | 0.2.100 → 0.2.121 | 8 | `ast.rs`, `codegen.rs`, `encode.rs` … +5 |
| 46 | `zeroize` | 1.8.2 → 1.9.0 | 8 | `aarch64.rs`, `alloc.rs`, `barrier.rs` … +5 |
| 47 | `getset` | 0.1.3 → 0.1.7 | 9 | `clone_getters.rs`, `generate.rs`, `lib.rs` … +6 |
| 48 | `rand` | 0.10.1 → 0.10.2 | 9 | `lib.rs`, `mod.rs`, `rng.rs` … +6 |
| 49 | `simd-adler32` | 0.3.7 → 0.3.10 | 9 | `avx2.rs`, `avx512.rs`, `lib.rs` … +6 |
| 50 | `tracing-core` | 0.1.34 → 0.1.36 | 9 | `callsite.rs`, `dispatcher.rs`, `field.rs` … +6 |
| 51 | `half` | 2.5.0 → 2.7.1 | 11 | `aarch64.rs`, `arch.rs`, `bfloat.rs` … +8 |
| 52 | `tokio-rustls` | 0.24.0 → 0.26.4 | 11 | `badssl.rs`, `client.rs`, `early-data.rs` … +8 |
| 53 | `bytemuck` | 1.22.0 → 1.25.2 | 12 | `anybitpattern.rs`, `array_tests.rs`, `cast_slice_tests.rs` … +9 |
| 54 | `libloading` | 0.9.0 → 0.8.9 | 12 | `as_filename.rs`, `as_symbol_name.rs`, `changelog.rs` … +9 |
| 55 | `constant_time_eq` | 0.3.1 → 0.4.2 | 13 | `bench.rs`, `bench_classic.rs`, `bench_generic.rs` … +10 |
| 56 | `futures` | 0.3.31 → 0.3.33 | 13 | `async_await_macros.rs`, `auto_traits.rs`, `compat.rs` … +10 |
| 57 | `core-foundation` | 0.10.0 → 0.10.1 | 15 | `array.rs`, `base.rs`, `bundle.rs` … +12 |
| 58 | `wasm-bindgen` | 0.2.100 → 0.2.121 | 16 | `build.rs`, `cast.rs`, `closure.rs` … +13 |
| 59 | `sha2` | 0.10.9 → 0.11.0 | 17 | `aarch64_sha2.rs`, `aarch64_sha3.rs`, `block_api.rs` … +14 |
| 60 | `time-macros` | 0.2.27 → 0.2.32 | 18 | `ast.rs`, `component.rs`, `date.rs` … +15 |
| 61 | `ureq` | 2.9.1 → 2.12.1 | 21 | `agent.rs`, `agent_test.rs`, `decoder.rs` … +18 |
| 62 | `tracing-subscriber` | 0.3.20 → 0.3.23 | 23 | `ansi_escaping.rs`, `builder.rs`, `chrono_crate.rs` … +20 |
| 63 | `compact_str` | 0.7.1 → 0.9.1 | 24 | `arbitrary.rs`, `borsh.rs`, `bytes.rs` … +21 |
| 64 | `zlib-rs` | 0.6.3 → 0.6.6 | 26 | `acle.rs`, `adler32.rs`, `allocate.rs` … +23 |
| 65 | `rustls-webpki` | 0.101.4 → 0.103.10 | 34 | `alg_tests.rs`, `aws_lc_rs_algs.rs`, `better_tls.rs` … +31 |
| 66 | `ttf-parser` | 0.19.0 → 0.25.1 | 35 | `aat.rs`, `avar.rs`, `cff1.rs` … +32 |
| 67 | `zip` | 2.4.2 → 8.6.0 | 35 | `aes.rs`, `aes_ctr.rs`, `aex_encryption.rs` … +32 |
| 68 | `tracing` | 0.1.41 → 0.1.44 | 57 | `debug.rs`, `debug_n.rs`, `debug_np.rs` … +54 |
| 69 | `jni` | 0.21.1 → 0.22.4 | 78 | `api_calls.rs`, `auto.rs`, `auto_elements.rs` … +75 |
| 70 | `rustls` | 0.21.6 → 0.23.42 | 88 | `alert.rs`, `anchors.rs`, `api.rs` … +85 |
| 71 | `zerocopy-derive` | 0.8.27 → 0.8.55 | 93 | `absence_of_deprecated_warning.rs`, `crate_path.rs`, `deprecated.rs` … +90 |
| 72 | `zerocopy` | 0.8.27 → 0.8.55 | 160 | `as_bytes_dynamic_size.rs`, `as_bytes_static_size.rs`, `build.rs` … +157 |
| 73 | `web-sys` | 0.3.76 → 0.3.98 | 1554 | `gen_AbortController.rs`, `gen_AbortSignal.rs`, `gen_AbstractRange.rs` … +1551 |

Work them with `cargo vet diff <crate> <from> <to> --mode=local`, then
`cargo vet certify <crate> <from> <to>` once satisfied. The first dozen touch a single
source file each.

