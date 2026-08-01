#!/usr/bin/env bash
# Bench Diana on diana-coco-v3 at every tier, in both geometries, each
# against ONLY its matched reference.
#
# One invocation per (tier, geometry) because the harness benches one engine
# per run, and `--only` is what keeps the reference cost from being paid ten
# times over. The matched reference is not a nicety: the gate compares
# against whichever reference declares the same `config` string, so passing
# the wrong one grades a rectangular engine against a square baseline —
# the M-D0 defect, mechanised.
#
# Usage:  bash tools/diana_v3_sweep.sh [corpus.toml]
set -u
CORPUS="${1:-corpora/diana-coco-v3.toml}"
BIN=target/release/examples/bench_detect.exe

cargo build --release -p ffai-diana --example bench_detect -j 4 2>&1 | grep -E "^error" && exit 1

for tier in n s m l x; do
  for geom in rect square; do
    if [ "$geom" = rect ]; then
      engine="yolo26${tier}"
      ref="ultralytics-yolo26${tier}-rect"
    else
      engine="yolo26${tier}-square"
      ref="ultralytics-yolo26${tier}"
    fi
    echo "############ $engine  vs  $ref ############"
    "$BIN" "$CORPUS" --runs 1 --engine "$engine" --only "$ref" 2>&1 \
      | grep -vE "^warning|^\s+\||^\s+-|Compiling|Finished|Running"
  done
done
