# Paired A/B harness for Diana bricks — CPU-TIME FIRST.
#
# WHY CPU TIME, NOT WALL
# ---------------------
# Pinning a process (ProcessorAffinity + High priority) RESTRICTS it to a
# core; it does not RESERVE that core. A foreign load — another session's
# build, a demo process, an IDE indexer — still lands there, and on an SMT
# machine it also occupies the sibling logical CPU where priority buys
# nothing. Elapsed wall then counts time we spent DESCHEDULED.
#
# CPU time does not accrue off-core. On the campaign that produced this
# rule, the same comparison read 0.78-1.50 by wall and 0.950-1.089 by CPU
# time: 5x tighter, and it needs no quiet machine.
#
# Order of instruments, cheapest and most robust first:
#   1. a deterministic COUNTER (immune to every timing artifact)
#   2. pinned CPU time            <- this harness
#   3. pinned wall (verified-quiet box only)
#   4. paired win-rate z          <- reported here on CPU time
#
# ***  CPU TIME SUMS ACROSS THREADS - RUN SERIAL-WORK BRICKS AT 1 THREAD  ***
#
# CPU time answers "how much total work", not "how long did it take". In a
# 24-thread program that distinction bites: a brick that removes SERIAL work
# shows fully in wall but is diluted ~24x in the summed CPU figure.
#
# Both halves of this were measured on the same two bricks:
#
#   brick                       what it removes    24t CPU        1t CPU
#   zero-copy marshalling       a serial memcpy    z = -1.28 (!)  z = +3.41
#   rectangular inference       parallel compute   z = +4.69      -
#
# The zero-copy brick is REAL - 19/22 at one thread - and reads as a
# non-result at 24 threads purely because of the summing. So:
#
#   * removes PARALLEL work (arithmetic, kernels)  -> CPU time at any N
#   * removes SERIAL work (copies, allocation,
#     single-threaded glue)                        -> RAYON_NUM_THREADS=1,
#                                                     where CPU == wall
#
# Getting this backwards prunes a good brick on a confident-looking z-score,
# which is the expensive direction of the refutation asymmetry.
#
# TRAPS THIS FILE ENCODES
# -----------------------
# * `$null = $p.Handle` MUST run BEFORE WaitForExit. .NET only caches the
#   handle once accessed; without it `TotalProcessorTime` reads EMPTY after
#   exit and silently injects 0/Inf into the median — a sample the
#   instrument failed to take is NOT a tie.
# * Arms must be of comparable DURATION. Any per-invocation overhead
#   (process launch, model load) is paid once per run, so it inflates the
#   shorter arm by a larger fraction. The harness prints the launch tax as a
#   share of each arm and warns when it is not negligible.
# * ABBA ordering, so "whichever ran first" cannot bias the result.
# * N >= 20 before judging a paired estimator: the same brick read
#   z = 1.26 at N=10 and z = 3.84 at N=22.
#
#   .\tools\diana_ab.ps1 -Knob FFAI_DIANA_NO_ZEROCOPY -Reps 22 -Runs 8
param(
    [string]$Knob,
    [string]$ExeB,
    [string[]]$ExeArgs,
    [string]$MedianPattern = 'median ([\d.]+) ms',
    [string]$CountPattern,
    # SS6: the profiler is part of the system under test. Knob mode needs it (that is
    # how profile_detect prints a median); two-binary mode usually must NOT have it.
    [bool]$ProfileEnv = $true,
    # Discarded reps. The first process pays cold OS file-cache cost for the model;
    # SS3 wants that OUT of the paired samples, not averaged into them.
    [int]$Warmup = 2,
    [int]$Reps = 22,
    [int]$Runs = 8,
    [string]$Image = "corpora/clips/diana-coco/coco-032.png",
    [string]$Exe = "target\release\examples\profile_detect.exe"
)

if (-not $Knob -and -not $ExeB) { throw "pass -Knob (env-toggle mode) or -ExeB (two-binary mode)" }
if (-not (Test-Path $Exe)) { throw "missing $Exe" }
if ($ExeB -and -not (Test-Path $ExeB)) { throw "missing $ExeB" }
$twoBin = [bool]$ExeB
# Image/Runs is the knob-mode default (profile_detect); a two-binary arm names its own.
if (-not $PSBoundParameters.ContainsKey('ExeArgs')) {
    $ExeArgs = if ($twoBin) { @() } else { @($Image, $Runs) }
}
if (-not $PSBoundParameters.ContainsKey('ProfileEnv')) { $ProfileEnv = -not $twoBin
}

function Invoke-Arm([bool]$Off) {
    # $Off selects arm B. In knob mode that means "knob set"; in two-binary mode
    # it means "run ExeB". The env knob is left untouched in two-binary mode so
    # the only difference between the arms is the executable.
    $exeToRun = $Exe
    if ($twoBin) {
        if ($Off) { $exeToRun = $ExeB }
    }
    elseif ($Off) { Set-Item -Path "Env:\$Knob" -Value "1" }
    else { Remove-Item -Path "Env:\$Knob" -ErrorAction SilentlyContinue }
    if ($ProfileEnv) { $env:FFAI_PROFILE = "1" }
    $out = Join-Path $env:TEMP "diana_ab.txt"
    $t0 = Get-Date
    # Start-Process rejects an EMPTY ArgumentList, so splat it only when non-empty.
    $sp = @{ FilePath = $exeToRun; PassThru = $true; NoNewWindow = $true
             RedirectStandardOutput = $out }
    if ($ExeArgs -and $ExeArgs.Count -gt 0) { $sp.ArgumentList = $ExeArgs }
    $p = Start-Process @sp
    $null = $p.Handle          # MUST precede WaitForExit - see header
    $p.PriorityClass = 'High'
    $p.WaitForExit()
    $wallTotal = ((Get-Date) - $t0).TotalMilliseconds
    $txt = Get-Content $out -Raw
    $med = if ($txt -match $MedianPattern) { [double]$Matches[1] } else { [double]::NaN }
    $cnt = if ($CountPattern -and $txt -match $CountPattern) { $Matches[1] } else { $null }
    $cpu = try { $p.TotalProcessorTime.TotalMilliseconds } catch { [double]::NaN }
    [pscustomobject]@{ cpu_ms = $cpu; median_ms = $med; launch_ms = $wallTotal; work = $cnt }
}

for ($w = 0; $w -lt $Warmup; $w++) { $null = Invoke-Arm $false; $null = Invoke-Arm $true }

$A = @(); $B = @()
for ($i = 0; $i -lt $Reps; $i++) {
    if ($i % 2 -eq 0) { $A += Invoke-Arm $false; $B += Invoke-Arm $true }
    else { $B += Invoke-Arm $true; $A += Invoke-Arm $false }
}

function Med($xs) {
    $s = @($xs | Where-Object { -not [double]::IsNaN($_) } | Sort-Object)
    if ($s.Count -eq 0) { return [double]::NaN }
    $s[[int]($s.Count / 2)]
}

# A sample the instrument failed to take is not a tie - drop it LOUDLY.
$dropped = @($A.cpu_ms + $B.cpu_ms | Where-Object { [double]::IsNaN($_) }).Count
if ($dropped -gt 0) { Write-Warning "$dropped CPU-time sample(s) missing - excluded from the medians" }

$aCpu = Med ($A.cpu_ms); $bCpu = Med ($B.cpu_ms)
$aWall = Med ($A.median_ms); $bWall = Med ($B.median_ms)

# Paired win rate on CPU TIME - the robust axis.
$n = [Math]::Min($A.Count, $B.Count)
$w = 0; for ($i = 0; $i -lt $n; $i++) { if ($A[$i].cpu_ms -lt $B[$i].cpu_ms) { $w++ } }
$z = if ($n -gt 0) { ($w - $n / 2) / (0.5 * [Math]::Sqrt($n)) } else { 0 }

# WORK-COUNT PARITY (SS4): divergent counts void the comparison outright.
if ($CountPattern) {
    # NOTE: never name this property `count` - it is intrinsic on arrays and
    # $A.count silently returns the number of reps, making the check vacuous.
    $ca = @($A.work | Where-Object { $_ } | Select-Object -Unique)
    $cb = @($B.work | Where-Object { $_ } | Select-Object -Unique)
    "work count  A: $($ca -join ',')   B: $($cb -join ',')"
    if ($ca.Count -ne 1 -or $cb.Count -ne 1) {
        throw "work count VARIES WITHIN an arm - the run is not deterministic, comparison void"
    }
    if ($ca[0] -ne $cb[0]) {
        throw "work count DIFFERS between arms (A=$($ca[0]) B=$($cb[0])) - the arms did different work, comparison VOID"
    }
    "  counts match -> the arms did identical work"
}

$mode = if ($twoBin) { "A=$Exe  B=$ExeB" } else { "knob=$Knob" }
"$mode  reps=$Reps  (ABBA, High priority, CPU-time primary, work-count checked)"
"{0,-26} {1,12} {2,12} {3,9}" -f "METRIC", "ours", "off", "ratio"
"{0,-26} {1,12:N0} {2,12:N0} {3,9:N3}" -f "CPU ms  (PRIMARY)", $aCpu, $bCpu, ($bCpu / $aCpu)
"{0,-26} {1,12:N1} {2,12:N1} {3,9:N3}" -f "detect median ms (wall)", $aWall, $bWall, ($bWall / $aWall)
"paired CPU-time wins: $w / $n   z = {0:N2}" -f $z

# The in-process metric excludes process launch and model load entirely, so for a
# short arm it is the cleaner axis. Reported alongside, never instead.
$w2 = 0; $n2 = 0
for ($i = 0; $i -lt $n; $i++) {
    if (-not [double]::IsNaN($A[$i].median_ms) -and -not [double]::IsNaN($B[$i].median_ms)) {
        $n2++; if ($A[$i].median_ms -lt $B[$i].median_ms) { $w2++ }
    }
}
$z2 = if ($n2 -gt 0) { ($w2 - $n2 / 2) / (0.5 * [Math]::Sqrt($n2)) } else { 0 }
"paired in-process wins: $w2 / $n2   z = {0:N2}" -f $z2
if ([Math]::Abs($z) -lt 2) { "  => |z| < 2: NOT a verdict at this N" }

# Duration guard: the launch tax must be small in BOTH arms or the shorter
# one is inflated by a larger fraction.
$armWall = Med ($A.launch_ms)
"arm wall ~{0:N0} ms; process launch + model load is carried by BOTH arms" -f $armWall
if ($armWall -lt 8000) {
    "  NOTE: short arms. The fixed launch+load cost is IDENTICAL in both arms,"
    "  so it DILUTES the ratio (conservative - understates a real win) rather"
    "  than biasing it. The paired win-rate is a sign test and is unaffected."
    "  Raise -Runs if you need the ratio's magnitude rather than its sign."
}

Remove-Item -Path "Env:\$Knob" -ErrorAction SilentlyContinue
Remove-Item Env:\FFAI_PROFILE -ErrorAction SilentlyContinue
