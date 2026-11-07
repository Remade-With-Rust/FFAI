#!/usr/bin/env python3
"""The four censuses on a callgrind output file (`instruction-counting` §5).

    python3 tools/diana_ir_census.py ~/ffai-ir/detect-baseline.out [N] [filter]

Prints, for the top N symbols: (a) SELF cost, (b) CALL COUNTS with inclusive
cost per call, (c) caller->callee EDGES. The line census is callgrind_annotate's
own job (`--auto=yes`). An optional substring filter restricts every table to
symbols containing it — handy for "everything in ffai_diana".

Parses the callgrind format directly: `fn=(id) name` defines a symbol (name
compression: later references are `(id)` alone), cost lines are `<pos> <Ir>`,
and `cfn=` + `calls=N ...` + one cost line records a call edge with its
inclusive cost.
"""
import re
import sys
from collections import defaultdict

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
flt = sys.argv[3] if len(sys.argv) > 3 else ""

names = {}
self_cost = defaultdict(int)
calls_to = defaultdict(int)       # callee -> calls
incl_to = defaultdict(int)        # callee -> inclusive Ir (summed over edges)
edge_calls = defaultdict(int)     # (caller, callee) -> calls
edge_incl = defaultdict(int)      # (caller, callee) -> inclusive Ir
total = 0

def resolve(tok):
    m = re.match(r"\((\d+)\)\s*(.*)", tok)
    if not m:
        return tok
    i, nm = m.group(1), m.group(2).strip()
    if nm:
        names[i] = nm
    return names.get(i, f"({i})")

cur = None
pending_call = None
with open(path, errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("fn="):
            cur = resolve(line[3:])
            continue
        if line.startswith("cfn="):
            pending_callee = resolve(line[4:])
            continue
        if line.startswith("calls="):
            n = int(line.split()[0][6:])
            pending_call = (pending_callee, n)
            continue
        if line.startswith(("fl=", "fi=", "fe=", "cfi=", "cfl=", "ob=", "cob=")):
            continue
        if line.startswith("summary:") or line.startswith("totals:"):
            total = int(line.split()[1])
            continue
        m = re.match(r"^[+\-*]?\d+\s+(\d+)", line) or re.match(r"^\S+\s+(\d+)$", line)
        if m and cur is not None:
            cost = int(m.group(1))
            if pending_call is not None:
                callee, n = pending_call
                calls_to[callee] += n
                incl_to[callee] += cost
                edge_calls[(cur, callee)] += n
                edge_incl[(cur, callee)] += cost
                pending_call = None
            else:
                self_cost[cur] += cost

def short(s, n=90):
    s = re.sub(r"<.*?>", "<>", s) if len(s) > n else s
    return s if len(s) <= n else s[: n - 3] + "..."

def keep(s):
    return flt in s

print(f"total Ir {total:,}\n")
print(f"(a) SELF cost — top {top}")
print(f"{'Ir':>14} {'share':>7}  symbol")
for sym, c in sorted(self_cost.items(), key=lambda kv: -kv[1]):
    if not keep(sym):
        continue
    print(f"{c:>14,} {100*c/total:6.2f}%  {short(sym)}")
    top -= 1
    if top == 0:
        break

top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
print(f"\n(b) CALL COUNTS — top {top} by inclusive Ir")
print(f"{'calls':>10} {'incl Ir':>14} {'per call':>10}  callee")
for sym, inc in sorted(incl_to.items(), key=lambda kv: -kv[1]):
    if not keep(sym):
        continue
    n = calls_to[sym]
    print(f"{n:>10,} {inc:>14,} {inc/max(n,1):>10,.1f}  {short(sym)}")
    top -= 1
    if top == 0:
        break

top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
print(f"\n(c) EDGES — top {top} by inclusive Ir")
print(f"{'calls':>10} {'incl Ir':>14} {'per call':>10}  caller -> callee")
for (a, b), inc in sorted(edge_incl.items(), key=lambda kv: -kv[1]):
    if not (keep(a) or keep(b)):
        continue
    n = edge_calls[(a, b)]
    print(f"{n:>10,} {inc:>14,} {inc/max(n,1):>10,.1f}  {short(a, 50)} -> {short(b, 60)}")
    top -= 1
    if top == 0:
        break
