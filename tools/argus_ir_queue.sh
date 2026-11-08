#!/usr/bin/env bash
# Run a list of `label mode` counts, P at a time, and keep the WSL session
# alive until they finish (children of an exiting wsl.exe are killed).
#   wsl bash tools/argus_ir_queue.sh P "v1a prefill" "v1a decode" ...
set -uo pipefail
P="$1"; shift
cd "$(dirname "$0")/.."
i=0
for job in "$@"; do
    i=$((i + 1))
    echo "$job $(( (i % 22) + 1 ))"
done | xargs -P "$P" -L1 bash -c 'bash tools/argus_ir.sh count "$0" "$1" -c "$2" > ~/ffai-ir/argus/log-$0-$1.txt 2>&1; cut -c1-260 ~/ffai-ir/argus/log-$0-$1.txt'
