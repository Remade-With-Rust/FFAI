#!/usr/bin/env python3
"""Audit the Diana Ir campaign ledger against one fixed standard.

    python3 tools/diana_ir_audit.py ~/ffai-ir/ledger.txt [bar]

A counted win is a ledger row that (1) is one labelled change counted alone,
(2) is compared with the row immediately before it on the same instrument,
(3) removed more than `bar` Ir (default 100,000: ten times the same-binary
spread), and (4) kept the fnv anchor of the row before it (byte-identical).
Reverts, re-counts of an unchanged tree and baseline re-runs are not changes
and are skipped. The vein is read from the label (`v7b` -> 7, `vhead1x1` ->
9, `v13c2` -> 13). Nothing is chosen by hand: the three largest qualifying
rows per vein are printed, and any vein with fewer than three is flagged.
"""
import re
import sys
from collections import defaultdict

path = sys.argv[1]
bar = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000

SKIP = {"baseline", "baseline2", "baseline3", "v4crev", "v1erev", "v2dfix", "v2d"}
# v2d: the inline-copy change itself regressed and its later count (v2dfix) is a
# revert; both are non-wins and excluded as changes that were undone.
ALIAS = {"vhead1x1": 9, "v13c2": 13}

rows = defaultdict(list)
for line in open(path):
    parts = line.split()
    if len(parts) < 3 or parts[0] not in ("detect", "track"):
        continue
    m = re.search(r"fnv=([0-9a-f]+)", line)
    rows[parts[0]].append((parts[1], int(parts[2]), m.group(1) if m else "?"))

wins = defaultdict(list)
all_rows = []
for inst, rs in rows.items():
    prev = None
    for label, ir, fnv in rs:
        if prev is not None and label not in SKIP:
            delta = ir - prev[1]
            if label in ALIAS:
                vein = ALIAS[label]
            else:
                mm = re.match(r"v(\d+)", label)
                vein = int(mm.group(1)) if mm else None
            same_bits = fnv == prev[2]
            qualifies = vein is not None and delta < -bar and same_bits
            all_rows.append((inst, label, vein, delta, same_bits, qualifies))
            if qualifies:
                wins[vein].append((delta, label, inst))
        prev = (label, ir, fnv)

print(f"standard: one change, counted alone vs the previous row, > {bar:,} Ir removed, fnv unchanged\n")
short = []
for vein in range(1, 16):
    ws = sorted(wins[vein])
    top = ws[:3]
    extra = len(ws) - 3
    cells = ", ".join(f"{l} {d:+,}" for d, l, _ in top)
    flag = "" if len(ws) >= 3 else "  <-- FEWER THAN THREE"
    more = f"  (+{extra} more above the bar)" if extra > 0 else ""
    print(f"vein {vein:>2}: {cells}{more}{flag}")
    if len(ws) < 3:
        short.append(vein)
print()
print("not counted (below bar, numerics, or regression):")
for inst, label, vein, delta, same_bits, q in all_rows:
    if not q:
        why = "regression" if delta >= 0 else ("fnv changed" if not same_bits else f"below bar ({-delta:,})")
        print(f"  {label:>10} vein {vein}  {delta:+,}  {why}")
print()
print("RESULT:", "every vein has three" if not short else f"short: {short}")
