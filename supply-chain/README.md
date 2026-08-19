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

---

## The remaining backlog, by shape rather than by total (2026-08-18)

`cargo vet suggest` reports an "estimated audit backlog" of ~5.6M lines, and
quoting that number alone is misleading enough that it stalled this gate twice.
The distribution is what matters:

| Slice | Crates | Lines | Share of backlog |
|---|--:|--:|--:|
| 5 largest | 5 | 8,414,399 | **77%** |
| ≤ 2000 lines each | 85 | 78,099 | 0.7% |
| ≤ 1000 lines each | 50 | 27,587 | 0.3% |
| ≤ 500 lines each | 21 | 6,033 | 0.1% |

Median crate: **4,110 lines**. The backlog is not a wall, it is five boulders
and a long tail of pebbles.

**This makes line count the wrong planning metric.** H-10 is about exemption
COVERAGE, not lines read. Auditing the 85 crates at ≤2000 lines each would clear
**27% of the 313 exemptions for 0.7% of the reading** — by far the best ratio
available, and none of it blocked on anything.

### The five boulders

| Crate | Lines | Note |
|---|--:|---|
| `aws-lc-sys` 0.43.0 | 2,108,351 | **`fetch`-feature only.** Absent from a default build entirely. The cheapest way to remove it is to not need it — the same move that took the graph 320 → 154. |
| `ring` 0.17.14 | 261,791 | C and assembly; reachable only through the same `fetch` TLS stack. |
| `cudarc` 0.17.8 / 0.19.8 | 361,425 | CUDA backend, `--all-features` only. Two versions in the graph, which is itself worth fixing. |
| `winapi` 0.3.9 | 181,323 | Superseded by `windows-sys` upstream; arrives via `fs2`, `ntapi`, `xet-runtime`. |

Four of the five are optional-feature paths that no shipped binary compiles.
Auditing them is real work spent on code the product does not run.

### First-party crates appear in this list — that is a policy question, not an audit

`ffai-argus`, `ffai-models`, `ffai-wasm`, and the sibling `rff-codec`,
`rff-format`, `rusty_h264`, `rusty_alloc-api` show up as needing audit because
`[policy.*] audit-as-crates-io = true` tells `cargo vet` to treat the published
crates.io versions as third-party.

That may well be deliberate — it catches a published version drifting from the
local source, which is a real supply-chain risk for a multi-repo org. But it is a
**decision to confirm, not a backlog to grind**: if these are meant to be trusted
as first-party, setting `audit-as-crates-io = false` removes them at a stroke.
Left as-is pending the maintainer's call, because silently flipping a deliberate
security setting to improve a number is the wrong trade.

### What is genuinely exhausted

- **Imports.** The cargo-vet registry contains exactly nine peers; all nine are
  imported. Guessed names (chromium, hyperium, rustsec, coreos) return
  "no peer named X found in the registry".
- **Publisher trust.** Iterated to convergence — the round after the last batch
  produced zero suggestions.
- **`cargo vet prune`.** Run; removed nothing. The exemption set is minimal.

What is left is human attestation. `cargo vet certify` records a named person
having read the code, so it is not something to automate or bulk-apply.
