# Per-tier Diana-vs-Ultralytics latency at the PINNED FLOOR.
#
# The goal said "re-measure on a quiet box". This box has not been quiet once
# all session — right now it sits at 98% with another session's benchmark on
# it — and waiting for quiet is unfalsifiable. So take the measurement the
# measurement-core skill says survives a loaded machine:
#
#   * PIN each arm to the same cores at High priority. Pinning does not
#     RESERVE the core, but it removes scheduler migration, which is what
#     turns a 1.06x spread into 2.02x.
#   * Take the MINIMUM per-image wall over N repetitions, not the mean or
#     median. Foreign load only ever adds time, so the minimum is the honest
#     floor of the code's own cost. The skill's own data: unpinned min 1047 ms
#     vs pinned min 1038 ms — the MINIMA AGREE even when the spreads do not.
#   * Alternate which arm runs first per tier, so neither systematically
#     samples the busier half of a minute.
#
# This is the third probe on the load axis: the bench sweep and both
# CPU-ratio runs were all taken under load with no pinning. If the flat
# ~1.75x result is itself an artifact of measuring under contention, this is
# where it changes.
#
# Usage:  pwsh -File tools/diana_pinned_floor.ps1 [-Images 9] [-Reps 3]

param([int]$Images = 9, [int]$Reps = 3, [string]$Mask = "ffff")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ours = Join-Path $root "target\release\examples\cpu_vs_wall.exe"
$py   = Join-Path $root ".venv-diana\Scripts\python.exe"
$refs = Join-Path $root "tools\diana_d6_cpu.py"
$tmp  = Join-Path $env:TEMP "diana_floor"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

function Invoke-Pinned {
    # NOT $Args: that collides with PowerShell's automatic $args variable
    # and the parameter silently arrives empty.
    param([string]$Exe, [string[]]$ArgList, [string]$Out)
    $p = Start-Process -FilePath $Exe -ArgumentList $ArgList -PassThru -NoNewWindow `
            -RedirectStandardOutput $Out -WorkingDirectory $root
    # Touch Handle BEFORE WaitForExit or the process object's timing fields
    # read empty after exit.
    $null = $p.Handle
    try {
        $p.ProcessorAffinity = [IntPtr]([Convert]::ToInt64($Mask, 16))
        $p.PriorityClass = 'High'
    } catch { }
    $p.WaitForExit()
    return $Out
}

# Per-image wall times are the leading two columns of both probes' tables:
#   "   0       630.0      703.1      1.12x  1"
function Get-MinWall {
    param([string]$Path)
    $vals = @()
    foreach ($line in Get-Content $Path) {
        if ($line -match '^\s*\d+\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x') { $vals += [double]$matches[1] }
    }
    if ($vals.Count -eq 0) { return [double]::NaN }
    # Drop the first sample: even after warm calls it carries page faults on
    # the first touch of each image buffer, and it is the same courtesy both
    # arms get.
    if ($vals.Count -gt 1) { $vals = $vals[1..($vals.Count - 1)] }
    return ($vals | Measure-Object -Minimum).Minimum
}

"per-tier PINNED FLOOR (min per-image wall, ms) - affinity 0x$Mask, High priority"
"$Images images x $Reps reps per arm, arms alternated"
"{0,-5} {1,12} {2,12} {3,8}" -f "tier", "Diana min", "ref min", "ratio"

$rows = @()
$tiers = @("n", "s", "m", "l", "x")
for ($i = 0; $i -lt $tiers.Count; $i++) {
    $t = $tiers[$i]
    $a = [double]::PositiveInfinity
    $b = [double]::PositiveInfinity
    for ($r = 0; $r -lt $Reps; $r++) {
        # Alternate on tier index AND rep, so ordering cannot align with tier.
        if ((($i + $r) % 2) -eq 0) {
            $a = [Math]::Min($a, (Get-MinWall (Invoke-Pinned $ours @($t, "$Images") "$tmp\o_$t`_$r.txt")))
            $b = [Math]::Min($b, (Get-MinWall (Invoke-Pinned $py @($refs, "--model", "corpora/cache/yolo26$t.pt", "--images", "$Images") "$tmp\r_$t`_$r.txt")))
        } else {
            $b = [Math]::Min($b, (Get-MinWall (Invoke-Pinned $py @($refs, "--model", "corpora/cache/yolo26$t.pt", "--images", "$Images") "$tmp\r_$t`_$r.txt")))
            $a = [Math]::Min($a, (Get-MinWall (Invoke-Pinned $ours @($t, "$Images") "$tmp\o_$t`_$r.txt")))
        }
    }
    $rows += [pscustomobject]@{ tier = $t; ours = $a; ref = $b; ratio = $a / $b }
    "{0,-5} {1,12:N1} {2,12:N1} {3,7:N2}x" -f $t, $a, $b, ($a / $b)
}

""
$rr = $rows | ForEach-Object { $_.ratio }
$mean = ($rr | Measure-Object -Average).Average
$mn = ($rr | Measure-Object -Minimum).Minimum
$mx = ($rr | Measure-Object -Maximum).Maximum
"ratio mean {0:N2}x  range {1:N2}x - {2:N2}x  (spread {3:N0}%)" -f $mean, $mn, $mx, (100 * ($mx / $mn - 1))
"A tier TREND requires the range to exceed the spread of repeated"
"measurement at one tier. A flat result means the gap does not depend on tier."
