#!/usr/bin/env bash
# ABBA-interleaved arm rotation for `examples/wasm_arms.rs` — Step 0 of
# docs/plans/carmenta-wasm-plan.md.
#
# Every arm is a separate PROCESS because `std::env::set_var` is unsafe in
# edition 2024 and candle's threads are live, so the arms cannot be rotated
# in-process. That makes interleaving the caller's job, and it is not optional:
# this machine has measured 2.4x spread on IDENTICAL configurations minutes
# apart (§8.100 D6 is the same trap — a whole diagnosis was built on a sweep
# that ran while other jobs did). Round-robin across arms, min over rounds:
# the minimum is the estimator that survives a busy machine, and interleaving
# is what stops a load spike landing on one arm.
#
#   tools/carmenta_wasm_arms.sh <rounds> <engine> <image.png> [image.png ...]
#
# Reports min-of-rounds per arm. A gap that does not survive this is noise.

set -u

ROUNDS="${1:?usage: carmenta_wasm_arms.sh <rounds> <engine> <image> [image...]}"
ENGINE="${2:?usage: carmenta_wasm_arms.sh <rounds> <engine> <image> [image...]}"
shift 2
IMAGES=("$@")
[ ${#IMAGES[@]} -gt 0 ] || { echo "no images given" >&2; exit 2; }

BIN=./target/release/examples/wasm_arms.exe
[ -x "$BIN" ] || BIN=./target/release/examples/wasm_arms
[ -x "$BIN" ] || {
  echo "build it first: cargo build --release -p ffai-carmenta --example wasm_arms" >&2
  exit 2
}

# name:env-assignments. `baseline` is what ships today: our kernel in the
# recognizer, candle in the detector.
ARMS=(
  "baseline:"
  "rec-scalar:FFAI_CONV3X3=scalar"
  "rec-candle:FFAI_CONV3X3=0"
  # det-kernel was measured 2.48x SLOWER on CRAFT's shapes and removed from
  # the rotation; `FFAI_CONV3X3_DET=1` still reproduces it.
)

declare -A BEST
for arm in "${ARMS[@]}"; do BEST["${arm%%:*}"]=999999999; done

for ((r = 1; r <= ROUNDS; r++)); do
  for arm in "${ARMS[@]}"; do
    name="${arm%%:*}"
    envs="${arm#*:}"
    # FFAI_REC_SERIAL models the browser: one thread, no line fan-out.
    out=$(env FFAI_REC_SERIAL=1 ${envs:+$envs} "$BIN" 1 "$ENGINE" "${IMAGES[@]}" 2>/dev/null \
          | grep '^TOTAL' | awk '{print $3}')
    [ -n "$out" ] || { echo "round $r arm $name: no output" >&2; continue; }
    printf 'round %d  %-12s %10s ms\n' "$r" "$name" "$out"
    cur="${BEST[$name]}"
    BEST[$name]=$(awk -v a="$cur" -v b="$out" 'BEGIN{print (b<a)?b:a}')
  done
done

echo
echo "min-of-$ROUNDS, engine=$ENGINE, ${#IMAGES[@]} image(s), single-threaded"
base="${BEST[baseline]}"
for arm in "${ARMS[@]}"; do
  name="${arm%%:*}"
  awk -v n="$name" -v v="${BEST[$name]}" -v b="$base" \
      'BEGIN{printf "  %-12s %10.1f ms   %5.2fx baseline\n", n, v, v/b}'
done
