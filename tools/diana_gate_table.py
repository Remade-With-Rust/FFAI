"""Render the per-tier gate table from the ledger — the launch copy's source.

Every number the README quotes for Diana should come out of here rather than
out of a paragraph, because the claim that got this tool written was a
BLANKET one: "speed FAILS, ~2.5x behind". That was measured at the n tier
and stated for all five. The same ledger shows the gap narrowing with model
size and the gate PASSING at m.

Reads the rows after a given ledger id (the corpus's baseline-only row), and
prints one line per engine with the matched reference beside it.

Usage:
    python tools/diana_gate_table.py [--hash <corpus_manifest_hash>] [--last N]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "bench" / "ledger.jsonl"

TIER_ORDER = {t: i for i, t in enumerate(["n", "s", "m", "l", "x"])}


def tier_of(name: str) -> tuple[int, int]:
    m = re.match(r"yolo26([nsmlx])(-square)?$", name)
    if not m:
        return (99, 0)
    return (TIER_ORDER[m.group(1)], 1 if m.group(2) else 0)


def p50(notes: list[str]) -> float | None:
    for n in notes or []:
        m = re.search(r"p50 (\d+) ms", n)
        if m:
            return float(m.group(1))
    return None


def cache_mib(notes: list[str]) -> float:
    """MiB of harness-held pre-decoded images, if the row predates the fix."""
    for n in notes or []:
        # Old rows: "peak includes X MiB". New rows already subtract it and
        # say "footprint EXCLUDES X MiB", so nothing further is owed.
        m = re.match(r"peak includes ([\d.]+) MiB", n)
        if m:
            return float(m.group(1))
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default="")
    # NOT --corpus. v2 and v3 both declare name = "diana-coco", so filtering
    # by name silently mixes a 45-image row into a 450-image table — which is
    # exactly what it did the first time this ran. The manifest HASH is the
    # corpus's identity; the name is a label.
    ap.add_argument("--hash", default="", help="corpus_manifest_hash; default = the newest row's")
    ap.add_argument("--last", type=int, default=10, help="take the last N engine rows")
    args = ap.parse_args()

    rows = [json.loads(l) for l in LEDGER.open(encoding="utf-8") if '"detect"' in l]
    rows = [r for r in rows if r.get("engine")]
    want = args.hash or (rows[-1]["corpus_manifest_hash"] if rows else "")
    rows = [r for r in rows if r.get("corpus_manifest_hash") == want]
    if not rows:
        raise SystemExit("no engine rows for that corpus hash")
    print(f"corpus {rows[0]['corpus']} @ {want[:12]}  ({len(rows)} engine rows)\n")
    if args.after:
        rows = [r for r in rows if r["id"] > args.after]
    # Keep the LAST row per engine. The ledger is append-only and losses stay
    # in it — a reference that crashed mid-sweep left a row with mAP 0.00 —
    # but a status table should show the current state, not every attempt.
    # The count of superseded rows is printed so they are not silently hidden.
    latest, superseded = {}, 0
    for r in rows:
        if r["engine"]["name"] in latest:
            superseded += 1
        latest[r["engine"]["name"]] = r
    rows = sorted(latest.values(), key=lambda r: tier_of(r["engine"]["name"]))[-args.last :]

    hdr = ("engine", "mAP50", "ref", "d pp", "MiB", "ref", "x", "p50 ms", "ref", "x", "gates")
    print("%-16s %7s %7s %6s  %6s %6s %5s  %7s %7s %5s  %s" % hdr)
    corrected = 0
    for r in rows:
        e, ref = r["engine"], (r["references"] or [{}])[0]
        adj = cache_mib(e["notes"])
        corrected += 1 if adj else 0
        ours_mib = e["steady_bytes"] / 1048576 - adj
        ref_mib = (ref.get("steady_bytes") or 0) / 1048576
        a, b = p50(e["notes"]), p50(ref.get("notes", []))
        gates = {g["kind"]: g["outcome"] for g in (r.get("gates") or {}).get("results", [])}
        flags = "".join(
            k[0].upper() if gates.get(k) == "pass" else k[0]
            for k in ("correctness", "quality", "speed", "footprint")
        )
        print(
            "%-16s %7.2f %7.2f %+6.2f  %6.0f %6.0f %4.1fx  %7s %7s %5s  %s%s"
            % (
                e["name"],
                (e["map50"] or 0) * 100,
                (ref.get("map50") or 0) * 100,
                ((e["map50"] or 0) - (ref.get("map50") or 0)) * 100,
                ours_mib,
                ref_mib,
                (ref_mib / ours_mib) if ours_mib else 0,
                f"{a:.0f}" if a else "-",
                f"{b:.0f}" if b else "-",
                f"{a / b:.2f}x" if a and b else "-",
                flags,
                "  (mem hand-corrected)" if adj else "",
            )
        )
    print()
    print("gate flags: UPPERCASE = pass, lowercase = fail, order C/Q/S/F")
    if superseded:
        print(f"({superseded} earlier attempt(s) superseded and still in the ledger)")
    if corrected:
        print(
            f"WARNING: {corrected} row(s) predate the harness footprint fix and were "
            "corrected here rather than at source — those memory figures do NOT "
            "trace to their ledger line. Re-run them."
        )


if __name__ == "__main__":
    main()
