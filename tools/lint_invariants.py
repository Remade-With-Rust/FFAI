#!/usr/bin/env python3
"""Project-specific invariant lint (hardening gate H-22).

Clippy checks Rust. This checks FFai: the invariants our own audit established,
which no general linter knows about. Each rule exists because an audit found the
thing it prevents.

    python tools/lint_invariants.py            # check
    python tools/lint_invariants.py --list     # show what is enforced

Deterministic, stdlib-only, no network. Exit 1 on any violation.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

UNSAFE_START = re.compile(r"^\s*(unsafe\s+(impl|fn)\b|.*\bunsafe\s*\{)")
SAFETY = re.compile(r"//\s*SAFETY:")
ALLOW_UNSAFE = re.compile(r"#\[allow\(unsafe_code\)\]")
BANNED = {
    "mem::forget": "use ManuallyDrop + an explicit drop; forget() leaks silently",
    "Box::leak": "leaks for the process lifetime - justify or use an arena",
    "transmute": "prefer a typed conversion; transmute hides layout assumptions",
}
LINT_ALLOW = re.compile(r"//\s*LINT-ALLOW:")
# Everything from the first test marker to EOF is test code: a panic there is a
# failing test, not a denial of service.
TEST_START = re.compile(r"\s*(#\[cfg\(test\)\]|mod tests)")


def crate_dirs():
    for m in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        yield m.parent


def rule_safety_comments(crate, viols):
    """R1. Every unsafe site carries a SAFETY comment.

    Only enforced for crates that have an UNSAFE.md - that file is the crate
    opting in to the inventory discipline, so the rule adopts incrementally
    instead of failing every crate on day one.
    """
    if not (crate / "UNSAFE.md").exists():
        return
    for f in sorted((crate / "src").rglob("*.rs")):
        lines = f.read_text(encoding="utf-8", errors="replace").split("\n")
        for i, line in enumerate(lines):
            if not UNSAFE_START.match(line):
                continue
            if "unsafe_code" in line or line.lstrip().startswith("//"):
                continue
            window = lines[max(0, i - 8):i + 1]
            if not any(SAFETY.search(w) for w in window):
                viols.append(
                    f"{f.relative_to(ROOT)}:{i+1}: unsafe site with no `// SAFETY:` "
                    f"comment within 8 lines (gate H-16)")


def rule_allow_unsafe_inventoried(crate, viols):
    """R2. `#[allow(unsafe_code)]` hides a site from the lint, so it must be in UNSAFE.md.

    This is the exact failure mode the mercury audit hit: 25 sites warned, 3 were
    invisible, and an inventory built from lint output alone undercounted.
    """
    unsafe_md = crate / "UNSAFE.md"
    if not unsafe_md.exists():
        return
    inventory = unsafe_md.read_text(encoding="utf-8", errors="replace")
    for f in sorted((crate / "src").rglob("*.rs")):
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").split("\n")):
            if ALLOW_UNSAFE.search(line):
                stem = f.relative_to(crate / "src").as_posix()
                if stem not in inventory:
                    viols.append(
                        f"{f.relative_to(ROOT)}:{i+1}: `#[allow(unsafe_code)]` hides this "
                        f"site from the lint and `{stem}` is not named in "
                        f"{unsafe_md.relative_to(ROOT)} (gate H-16)")


def rule_banned_constructs(crate, viols):
    """R3. Constructs STANDARD.md section 4.2 bans outright."""
    for f in sorted((crate / "src").rglob("*.rs")):
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").split("\n")):
            if LINT_ALLOW.search(line):
                continue
            for needle, why in BANNED.items():
                if needle in line and not line.lstrip().startswith("//"):
                    viols.append(
                        f"{f.relative_to(ROOT)}:{i+1}: `{needle}` is banned - {why}. "
                        f"Add `// LINT-ALLOW:` with a justification to override")


def rule_no_stdout_in_library(crate, viols):
    """R4. A library must not write to stdout.

    stdout is the caller's channel - ours is stderr, and every content-bearing
    write there is env-gated (gate C-08). `dbg!` is debugging left behind.
    """
    # Library code only. A binary's stdout IS its output channel, so println! in
    # ffai-cli is correct rather than a violation. The rule is about a LIBRARY
    # stealing the caller's stdout.
    if not (crate / "src" / "lib.rs").exists():
        return
    for f in sorted((crate / "src").rglob("*.rs")):
        if f.name == "main.rs" or "bin" in f.relative_to(crate / "src").parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").split("\n")):
            s = line.lstrip()
            if s.startswith("//") or LINT_ALLOW.search(line):
                continue
            if re.search(r"\bprintln!\(|\bdbg!\(", line):
                viols.append(
                    f"{f.relative_to(ROOT)}:{i+1}: `println!`/`dbg!` in library code - "
                    f"stdout belongs to the caller (gate C-08)")


def rule_no_bare_unwrap(crate, viols):
    """R5. No bare `.unwrap()` in non-test library code; state the invariant.

    `.expect("why this cannot fail")` and `.unwrap()` panic identically. The
    difference is that one leaves the invariant on the page for the next reader,
    and the other leaves them to re-derive it. On an untrusted path a panic is a
    denial of service, so the invariant is the thing worth auditing.

    ffai-mercury already had ZERO bare unwraps and 18 documented expects when
    this rule was written - it locks in a discipline already being followed
    rather than imposing a new one.
    """
    if not (crate / "UNSAFE.md").exists():
        return
    for f in sorted((crate / "src").rglob("*.rs")):
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        cut = len(lines)
        for i, line in enumerate(lines):
            if TEST_START.match(line):
                cut = i
                break
        for i, line in enumerate(lines[:cut]):
            s = line.lstrip()
            if s.startswith("//") or LINT_ALLOW.search(line):
                continue
            if ".unwrap()" in line:
                viols.append(
                    f"{f.relative_to(ROOT)}:{i+1}: bare `.unwrap()` in library code - use "
                    f"`.expect(\"why this cannot fail\")` so the invariant is on the page "
                    f"(gate H-18)")


RULES = [
    ("R1 unsafe sites carry a SAFETY comment", rule_safety_comments),
    ("R2 #[allow(unsafe_code)] sites are inventoried in UNSAFE.md", rule_allow_unsafe_inventoried),
    ("R3 banned constructs (mem::forget, Box::leak, transmute)", rule_banned_constructs),
    ("R4 no println!/dbg! in library code", rule_no_stdout_in_library),
    ("R5 no bare .unwrap() in library code", rule_no_bare_unwrap),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list the rules and exit")
    args = ap.parse_args()

    if args.list:
        for name, fn in RULES:
            print(f"{name}\n    {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    viols: list[str] = []
    crates = list(crate_dirs())
    for crate in crates:
        for _, fn in RULES:
            fn(crate, viols)

    opted_in = [c.name for c in crates if (c / "UNSAFE.md").exists()]
    print(f"lint_invariants: {len(crates)} crates, "
          f"{len(opted_in)} with UNSAFE.md ({', '.join(opted_in) or 'none'})")
    if viols:
        print(f"\n{len(viols)} violation(s):\n", file=sys.stderr)
        for v in viols:
            print(f"  {v}", file=sys.stderr)
        return 1
    print("no violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
