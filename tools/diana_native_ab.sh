#!/usr/bin/env bash
# Paired A/B of the SAME code built two ways: baseline x86-64 vs
# `-C target-cpu=native`.
#
# D6f in docs/whys/diana-latency.md found the crate has no `target-cpu`, so
# every hand-written kernel here compiles for SSE2 — no AVX2, no FMA. This
# prices that at the level that matters (a whole detect pass on the serial
# path the speed gate times) rather than on one kernel.
#
# The two arms are two BINARIES, so they cannot share a process the way
# `batch_ab` pairs an env toggle. They are alternated ABBA instead, which
# still gives both arms the same drift, and the verdict is the paired win
# rate: z = (wins - N/2) / (0.5*sqrt(N)), |z| > 2 is real.
#
# Usage:  bash tools/diana_native_ab.sh [tier] [rounds] [images]
set -u
TIER="${1:-n}"; ROUNDS="${2:-15}"; IMAGES="${3:-8}"
A=target/release/examples/batch_ab.exe            # baseline x86-64
B=target-native/release/examples/batch_ab.exe     # -C target-cpu=native

for f in "$A" "$B"; do
  [ -x "$f" ] || { echo "missing $f"; exit 1; }
done

run() { "$1" --child "$TIER" "$IMAGES" serial 2>/dev/null | tail -1 | awk '{print $1}'; }

wins=0; ratios=()
for ((r=0; r<ROUNDS; r++)); do
  if (( r % 2 == 0 )); then a=$(run "$A"); b=$(run "$B")
  else                      b=$(run "$B"); a=$(run "$A"); fi
  ratios+=("$(python -c "print($a/$b)")")
  if python -c "import sys; sys.exit(0 if $b < $a else 1)"; then wins=$((wins+1)); fi
  printf '  round %2d/%d:  baseline %8s ms   native %8s ms   A/B %s\n' \
    "$((r+1))" "$ROUNDS" "$a" "$b" "$(python -c "print(f'{$a/$b:.3f}')")"
done

python - "$wins" "$ROUNDS" "${ratios[@]}" <<'EOF'
import sys, math, statistics
wins, n = int(sys.argv[1]), int(sys.argv[2])
ratios = [float(x) for x in sys.argv[3:]]
z = (wins - n / 2) / (0.5 * math.sqrt(n))
print()
print(f"native faster in {wins}/{n}   z = {z:+.2f}   median baseline/native = {statistics.median(ratios):.4f}x")
print("verdict:", "REAL at |z| > 2" if abs(z) > 2 else "inside the noise — not a result")
EOF
