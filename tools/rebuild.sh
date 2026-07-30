#!/usr/bin/env bash
# STALE-BINARY GUARD. On 2026-07-30 a cargo build failed with "Access is
# denied" (a previous ffai.exe still held open), the failure scrolled past,
# and every measurement for the next HOUR ran the old binary. That produced
# four contradictory numbers and one wrong deletion of a working feature.
#
# Rule: never measure without proving the binary is newer than the source.
set -e
cd "$(dirname "$0")/.."
powershell -NoProfile -Command "Get-Process ffai,ffai-demo -EA SilentlyContinue | Stop-Process -Force" 2>/dev/null || true
sleep 1
rm -f target/release/ffai.exe
cargo build --release -q -p ffai-cli "$@"
test -f target/release/ffai.exe || { echo "REBUILD FAILED — binary absent"; exit 1; }
newest_src=$(find crates -name '*.rs' -newer target/release/ffai.exe | head -1)
if [ -n "$newest_src" ]; then
  echo "REBUILD STALE — $newest_src is newer than the binary"; exit 1
fi
echo "binary verified fresh: $(ls --time-style=+%H:%M:%S -la target/release/ffai.exe | awk '{print $6}')"
