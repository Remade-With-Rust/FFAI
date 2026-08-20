# Security policy

FFai is a media and inference toolkit written in Rust. Several of its crates parse
untrusted input — audio, images, documents, model files — so we treat security reports
as first-class work.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report privately through GitHub's private vulnerability reporting:

1. Go to <https://github.com/Remade-With-Rust/FFAI/security/advisories/new>
2. Describe the issue, the affected crate and version, and how to reproduce it.
3. Include a proof-of-concept input if you have one — a crashing file is worth a
   thousand words, and we will add it to the regression corpus once fixed.

If GitHub reporting is unavailable to you, open a public issue containing **only** the
words "security report, requesting private channel" and nothing else, and a maintainer
will open a private advisory to continue in.

## What to expect

| Stage | Target |
|---|---|
| Acknowledgement of your report | 3 working days |
| Initial assessment and severity | 10 working days |
| Fix or documented mitigation | 90 days from acknowledgement |
| Public advisory + credit | on release of the fix, or at 90 days |

### If the report involves personal data

Some FFai crates process personal data — `ffai-mercury` derives **voiceprints**, which are
special-category biometric data under GDPR Art 9 (see
`crates/ffai-mercury/docs/data-inventory.md`). If your report describes a defect that could
expose personal data, say so explicitly in the first line. That starts a **72-hour clock**
under GDPR Art 33 for any deployer acting as a controller, and we will notify affected
integrators within that window rather than waiting for a fix.

We follow coordinated disclosure. We will agree a disclosure date with you, and we will
credit you in the advisory unless you ask us not to.

If a report is already being exploited in the wild, tell us — the 90-day clock is not a
shield and we will move immediately.

## Scope

**In scope** — anything that lets untrusted input cause memory unsafety, a panic that
becomes a denial of service, incorrect output presented as correct, or disclosure of
data the caller did not intend to expose. Parsers, decoders, and `unsafe` blocks are the
highest-value targets: `ffai-mercury` (audio, model files), `ffai-carmenta` (documents,
images), `ffai-diana`, `ffai-media`.

**Out of scope** — vulnerabilities in dependencies with no FFai-specific exposure (report
those upstream, and tell us so we can pin or patch), issues requiring a compromised local
machine or write access to the model cache *by design* (see each crate's threat model),
and results from automated scanners without a demonstrated impact.

## Supported versions

FFai is pre-1.0. Security fixes land on the latest published minor of each crate. There
is no long-term-support branch yet; when a crate reaches 1.0.0 under the hardening
process (`docs/plans/use-protection-please.md` per crate), this table gains a support
window.

## Hardening programme

Each crate carries a hardening audit against a 41-gate standard plus 14 compliance
controls, tracked in `crates/<name>/docs/plans/use-protection-please.md` and summarised
at the bottom of each crate's README. A crate does not ship 1.0.0 until its
v1.0.0-blocking gates pass.
