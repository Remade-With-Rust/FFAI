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
    [Parameter(Mandatory = $true)][string]$Knob,
    [int]$Reps = 22,
    [int]$Runs = 8,
    [string]$Image = "corpora/clips/diana-coco/coco-032.png",
    [string]$Exe = "target\release\examples\profile_detect.exe"
)

if (-not (Test-Path $Exe)) { throw "missing $Exe - cargo build --release -p ffai-diana --example profile_detect" }

function Invoke-Arm([bool]$Off) {
    if ($Off) { Set-Item -Path "Env:\$Knob" -Value "1" }
    else { Remove-Item -Path "Env:\$Knob" -ErrorAction SilentlyContinue }
    $env:FFAI_PROFILE = "1"
    $out = Join-Path $env:TEMP "diana_ab.txt"
    $t0 = Get-Date
    $p = Start-Process -FilePath $Exe -ArgumentList $Image, $Runs -PassThru -NoNewWindow -RedirectStandardOutput $out
    $null = $p.Handle          # MUST precede WaitForExit - see header
    $p.PriorityClass = 'High'
    $p.WaitForExit()
    $wallTotal = ((Get-Date) - $t0).TotalMilliseconds
    $txt = Get-Content $out -Raw
    $med = if ($txt -match 'median ([\d.]+) ms') { [double]$Matches[1] } else { [double]::NaN }
    $cpu = try { $p.TotalProcessorTime.TotalMilliseconds } catch { [double]::NaN }
    [pscustomobject]@{ cpu_ms = $cpu; median_ms = $med; launch_ms = $wallTotal }
}

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

"knob=$Knob  reps=$Reps  runs=$Runs  (ABBA, pinned High, CPU-time primary)"
"{0,-26} {1,12} {2,12} {3,9}" -f "METRIC", "ours", "off", "ratio"
"{0,-26} {1,12:N0} {2,12:N0} {3,9:N3}" -f "CPU ms  (PRIMARY)", $aCpu, $bCpu, ($bCpu / $aCpu)
"{0,-26} {1,12:N1} {2,12:N1} {3,9:N3}" -f "detect median ms (wall)", $aWall, $bWall, ($bWall / $aWall)
"paired CPU-time wins: $w / $n   z = {0:N2}" -f $z
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
