#!/usr/bin/env bash
# Deterministic instruction counts (callgrind Ir) for Diana — run under WSL.
#
#   wsl bash tools/diana_ir.sh <label>          # build + count both instruments
#   wsl bash tools/diana_ir.sh <label> detect    # one instrument only
#
# ONE script, invoked identically every time: a different shell invocation moves
# the count by a fixed ~950-instruction pedestal (`instruction-counting` §2), so
# every arm goes through this file. Same binary twice must read a spread of 0;
# a rebuild from identical source reads 6-10. Re-check both at the start of a
# session before comparing anything.
#
# Single-threaded BY CONSTRUCTION: `taskset` to one core (candle's gemm and
# rayon read available_parallelism), FFAI_DIANA_THREADS=1 (our pool, if any
# path installs one), RAYON_NUM_THREADS=1, and the region runs inside
# `serial_scope`. callgrind's header reports the thread count; >1 is a broken arm.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"

LABEL="${1:?usage: diana_ir.sh <label> [detect|track|all]}"
WHAT="${2:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Build on WSL-local disk: /mnt/e is slow for the thousands of small writes a
# build makes, and the target dir is not shared with the Windows build anyway.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/ffai-target}"
# Line tables only: changes no instruction, lets `callgrind_annotate --auto=yes`
# attribute to source lines.
export CARGO_PROFILE_RELEASE_DEBUG="${CARGO_PROFILE_RELEASE_DEBUG:-line-tables-only}"
OUT="${DIANA_IR_OUT:-$HOME/ffai-ir}"
mkdir -p "$OUT"

# Model cache: the converted weights live in the Windows per-user cache.
export FFAI_CACHE="${FFAI_CACHE:-/mnt/c/Users/talmo/AppData/Local/ffai}"
export FFAI_DIANA_COUNT=1
export FFAI_DIANA_THREADS=1
export RAYON_NUM_THREADS=1
unset FFAI_PROFILE FFAI_DIANA_ROOFLINE

IMG="${DIANA_IR_IMAGE:-corpora/cache/coco-val2017/000000000785.jpg}"
DETS="${DIANA_IR_DETS:-corpora/clips/mot17-02/diana_ungated.tsv}"
CORE="${DIANA_IR_CORE:-2}"

# A failed build must FAIL the gate, not run the previous binary
# (`codec-measurement`, "A FAILED BUILD makes a gate report identical").
LOG="$OUT/build-$LABEL.log"
if ! cargo build --release -p ffai-diana --example ir_detect --example ir_track >"$LOG" 2>&1; then
    echo "BUILD FAILED — see $LOG" >&2
    grep -E "^error" -A6 "$LOG" | head -60 >&2
    exit 1
fi
BIN="$CARGO_TARGET_DIR/release/examples"
df -h "$HOME" | awk 'NR==2 {print "disk free: " $4}'

run_one() {
    local name="$1" region="$2" bin="$3"; shift 3
    local out="$OUT/$name-$LABEL.out"
    local stdout
    stdout="$(taskset -c "$CORE" valgrind --tool=callgrind --quiet \
        --collect-atstart=no --toggle-collect="*${region}*" \
        --callgrind-out-file="$out" "$bin" "$@" 2>"$OUT/$name-$LABEL.stderr")"
    local ir threads
    ir="$(awk '/^summary:/ {print $2}' "$out")"
    threads="$(grep -c '^thread:' "$out" || true)"
    printf '%-8s Ir %16s  threads=%s  %s\n' "$name" "$(printf "%'d" "$ir")" "${threads:-1}" "$stdout"
    echo "$name $LABEL $ir $stdout" >> "$OUT/ledger.txt"
}

case "$WHAT" in
    detect|all) run_one detect ir_region_detect "$BIN/ir_detect" "$IMG" n ;;&
    track|all)  run_one track  ir_region_track  "$BIN/ir_track" "$DETS" ;;
esac
echo "callgrind outputs: $OUT/*-$LABEL.out  (census: callgrind_annotate --threshold=94 <file>)"
