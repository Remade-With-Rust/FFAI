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

## Imported audit sources

mozilla · google · bytecode-alliance · embark-studios · zcash · isrg · fermyon

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
