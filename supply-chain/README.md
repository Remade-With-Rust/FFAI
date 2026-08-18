# Supply-chain audits (`cargo vet`) — gate H-10

`cargo vet check` runs on every PR. It answers one question: **can a dependency
enter this tree without anyone looking at it?** The answer is now no.

## What the numbers mean

As introduced, 2026-08-15:

| | count | meaning |
|---|---:|---|
| fully audited | 98 | covered by an imported third-party audit |
| partially audited | 3 | audited for a weaker criterion than we require |
| **exempted** | **474** | **recorded gaps — present when vetting started, nobody has audited them** |

**An exemption is not a certification.** It is a written acknowledgement that a crate
is trusted only because it was already here. Gate H-10 asks for every dependency to
be covered by an audit, so it stays **open** while that 474 stands — reporting it as
met would be exactly the "checklist that records intentions" the standard warns about.

## What it does buy, today

The tree is frozen. A new dependency fails CI unless it is covered by an imported
audit or given an explicit exemption in a reviewed commit. That is the ratchet, and
it is the part that stops the problem growing while the burn-down happens.

## Why imports are exhausted as a lever

All **nine** sources in cargo-vet's registry are imported. Adding the last two
(`actix`, `ariel-os`) changed the numbers by zero, and `cargo vet regenerate
exemptions` — which recomputes the MINIMAL exemption set given every audit we can
see — moved 474 to 473.

That is the ceiling of borrowed trust for this tree. The registry covers
general-purpose infrastructure; this workspace is ML and audio (`candle`,
`tokenizers`, `rustfft`, `hf-hub`, `xet`, `aws-lc-sys`), which no major
organisation has audited. Further progress has to be earned, not imported.

Note also that an exemption SHADOWS an import: once a crate is exempted, vet stops
asking whether an audit now covers it. `regenerate exemptions` is what re-asks.

## The largest lever is a smaller graph

**135 of the 430 exempted crates — 31% — exist only because of the optional weight
downloader.** Building `ffai-mercury` with `--no-default-features` drops the
`hf-hub → reqwest → rustls → aws-lc-rs` subtree, so a consumer who ships their own
weights carries a third fewer unaudited crates and no C crypto at all.

Removing a dependency audits it perfectly, in zero time. That ranks above auditing
one by hand, and it is the first thing to try for each remaining cluster.

## Imported audit sources

mozilla · google · bytecode-alliance · embark-studios · zcash · isrg · fermyon · actix · ariel-os

(All nine in the registry. There are no more to add.)

Adding a source is the cheapest way to shrink the 474 — it costs one `cargo vet
import <org>` and covers whatever that organisation has already audited.

## Burning down an exemption

```bash
cargo vet suggest                     # what to audit next, ranked by reachability
cargo vet diff <crate> <old> <new>    # review a version bump rather than a whole crate
cargo vet certify <crate> <version>   # only after ACTUALLY reading it
cargo vet prune                       # drop exemptions no longer needed
```

Prioritise by blast radius, not alphabetically: crates that parse untrusted input,
crates with `unsafe`, and build-time proc macros first. `cargo geiger` output
(`crates/ffai-mercury/docs/geiger-baseline.txt`) is a reasonable ranking input.

**Never `certify` a crate you have not read.** A false certification is worse than an
exemption, because it stops anyone else looking.
