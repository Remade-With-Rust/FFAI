#!/usr/bin/env bash
# Deterministic instruction counts (callgrind Ir) for Argus — run under WSL.
#
#   wsl bash tools/argus_ir.sh build <label>                 # build + stash the binary
#   wsl bash tools/argus_ir.sh count <label> [modes...] [-c CORE]  # count a stashed binary
#   wsl bash tools/argus_ir.sh run   <label> [modes...]      # build then count
#
# Modes: tile prefill decode caption pre assemble (default: all six).
#
# Build and count are split so that brick N can be counted (on its own core)
# while brick N+1 builds: `count` only ever runs the binary stashed under its
# label, never whatever is in the target dir, so a later build cannot leak in.
#
# ONE script, invoked identically every time (`instruction-counting` §2): a
# different shell invocation moves the count by a fixed ~950-instruction
# pedestal. Same binary twice must read a spread of 0; a rebuild from identical
# source 6-10. Re-check both at the start of a session.
#
# Single-threaded BY CONSTRUCTION: `taskset` to one core (candle's gemm and
# rayon read available_parallelism), RAYON_NUM_THREADS=1, one tile worker, and
# every region runs inside a one-thread rayon pool's `install` so the toggled
# function and the work beneath it share one thread.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"

CMD="${1:?usage: argus_ir.sh build|count|run <label> [modes...]}"
LABEL="${2:?usage: argus_ir.sh build|count|run <label> [modes...]}"
shift 2
CORE="${ARGUS_IR_CORE:-2}"
MODES=()
while [ $# -gt 0 ]; do
    case "$1" in
        -c) CORE="$2"; shift 2 ;;
        *) MODES+=("$1"); shift ;;
    esac
done
[ ${#MODES[@]} -eq 0 ] && MODES=(tile prefill decode caption pre assemble)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/ffai-target}"
export CARGO_PROFILE_RELEASE_DEBUG="${CARGO_PROFILE_RELEASE_DEBUG:-line-tables-only}"
OUT="${ARGUS_IR_OUT:-$HOME/ffai-ir/argus}"
mkdir -p "$OUT/bin"

# WSL-local copy of the three SmolVLM files (the manifest checks
# $FFAI_CACHE/models/<name>/ before the HF hub, which WSL cannot see).
export FFAI_CACHE="${FFAI_CACHE:-$HOME/ffai-cache-argus}"
export RAYON_NUM_THREADS=1
export FFAI_ARGUS_TILE_WORKERS=1
unset FFAI_PROFILE FFAI_VIS_PROFILE FFAI_TEXT_PROFILE FFAI_ARGUS_PROFILE

do_build() {
    local log="$OUT/build-$LABEL.log"
    # A failed build must FAIL, not leave the previous binary to be counted.
    if ! cargo build --release -p ffai-argus --example ir_argus >"$log" 2>&1; then
        echo "BUILD FAILED — see $log" >&2
        grep -E "^error" -A8 "$log" | head -60 >&2
        exit 1
    fi
    cp "$CARGO_TARGET_DIR/release/examples/ir_argus" "$OUT/bin/$LABEL"
    echo "built $LABEL ($(sha256sum "$OUT/bin/$LABEL" | cut -c1-12))"
}

do_count() {
    local bin="$OUT/bin/$LABEL"
    [ -x "$bin" ] || { echo "no stashed binary for $LABEL" >&2; exit 1; }
    for mode in "${MODES[@]}"; do
        local out="$OUT/$mode-$LABEL.out"
        local stdout ir threads
        stdout="$(taskset -c "$CORE" valgrind --tool=callgrind --quiet \
            --collect-atstart=no --toggle-collect="*ir_region_${mode}*" \
            --callgrind-out-file="$out" "$bin" "$mode" 2>"$OUT/$mode-$LABEL.stderr")" || {
            echo "$mode $LABEL FAILED — see $OUT/$mode-$LABEL.stderr" >&2; tail -5 "$OUT/$mode-$LABEL.stderr" >&2; continue; }
        ir="$(awk '/^summary:/ {print $2}' "$out")"
        threads="$(grep -c '^thread:' "$out" || true)"
        printf '%-9s %-22s Ir %16s  threads=%s  %s\n' "$mode" "$LABEL" "$(printf "%'d" "$ir")" "${threads:-1}" "$stdout"
        echo "$mode $LABEL $ir $stdout" >> "$OUT/ledger.txt"
    done
}

case "$CMD" in
    build) do_build ;;
    count) do_count ;;
    run) do_build; do_count ;;
    *) echo "unknown command $CMD" >&2; exit 2 ;;
esac
