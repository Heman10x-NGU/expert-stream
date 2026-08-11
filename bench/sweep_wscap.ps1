# sweep_wscap.ps1 - decode tok/s as a function of resident cache size.
#
# The hard working-set cap (MEASURED_GROUND_TRUTH section 9) turns the OS into a bounded LRU
# cache over the expert bank, and the cap IS the cache size. Sweeping it gives us the
# cache-size-vs-throughput curve that section 7 listed as unmeasured, for free, and it gives
# N-2 a measured dumb-LRU baseline to beat instead of a theoretical one.
#
# WHY REDIRECTION IS SAFE HERE AND WAS NOT BEFORE. A killed process loses its block-buffered
# stdout; a process that exits normally flushes it. Capped runs complete. If one does get
# killed by the watchdog its log will be empty, and this script reports that as "no result"
# rather than silently dropping the point.
#
# SAFETY: reads the model read-only, deletes nothing, kills nothing except the child that
# run_first_output.ps1 itself supervises.
#
# PowerShell 5.1. Pure ASCII. No && || ternary ??

param(
    # 10000 was in this list and should not have been. The cap decides how much RAM is left for
    # the REST OF THE MACHINE: measured, everything-but-llama costs ~4,040 MB, so a 10,000 MB
    # cap leaves ~1.7 GB free - the regime that was killing the agent process driving these
    # runs. run_first_output.ps1 now refuses such caps outright; this list stays inside the safe
    # band so the sweep never trips that guard.
    [int[]]  $CapsMB     = @(4000, 5500, 7000, 8000),
    [string] $PromptFile = "bench\prompts\p1.txt",
    [int]    $NPredict   = 16,
    [int]    $CtxSize    = 512,
    [int]    $TimeoutSec = 900
)

$ErrorActionPreference = "Stop"

$repo   = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_first_output.ps1"
if (-not (Test-Path -LiteralPath $runner)) { throw "runner not found: $runner" }

$promptPath = $PromptFile
if (-not [System.IO.Path]::IsPathRooted($promptPath)) { $promptPath = Join-Path $repo $PromptFile }
if (-not (Test-Path -LiteralPath $promptPath)) { throw "prompt not found: $promptPath" }
if ($promptPath -match '\s') { throw "prompt path contains a space; the -File command line cannot carry it" }

$outDir = Join-Path $repo "bench\results"
New-Item -ItemType Directory -Force $outDir | Out-Null
$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$csv    = Join-Path $outDir ("wscap-sweep-{0}.csv" -f $stamp)

$results = @()

foreach ($cap in $CapsMB) {
    Write-Host ""
    Write-Host ("================ working-set cap {0} MB ================" -f $cap) -ForegroundColor Cyan

    # Let the standby list settle so each run starts from a comparable state. Without this the
    # first run of the sweep is measured against a cold cache and the rest against a warm one.
    Start-Sleep -Seconds 10
    $os0 = Get-CimInstance Win32_OperatingSystem
    $freeBefore = [math]::Round($os0.FreePhysicalMemory / 1KB, 0)

    $log = Join-Path $outDir ("wscap-{0}mb-{1}.log" -f $cap, $stamp)
    $sw  = [System.Diagnostics.Stopwatch]::StartNew()

    & powershell -NoProfile -ExecutionPolicy Bypass -File $runner `
        -PromptFile $promptPath -NPredict $NPredict -CtxSize $CtxSize `
        -WorkingSetCapMB $cap -TimeoutSec $TimeoutSec -Tag ("wscap{0}" -f $cap) *>&1 |
        Tee-Object -FilePath $log

    # NOTE: no `| Out-Null` here. The first attempt at this sweep piped the tee into Out-Null,
    # which made the whole run invisible - no progress, no way to tell a stall from a slow run.
    $sw.Stop()

    $promptTps = $null
    $genTps    = $null
    $killed    = $false
    if (Test-Path -LiteralPath $log) {
        foreach ($line in (Get-Content -LiteralPath $log)) {
            if ($line -match 'Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s') {
                $promptTps = [double]$Matches[1]
                $genTps    = [double]$Matches[2]
            }
            if ($line -match 'WATCHDOG') { $killed = $true }
        }
    }

    if ($null -eq $genTps) {
        Write-Host ("  NO RESULT at {0} MB (killed={1}) - point dropped, not guessed." -f $cap, $killed) -ForegroundColor Yellow
    } else {
        Write-Host ("  prompt {0} t/s   generation {1} t/s   wall {2:N0} s" -f $promptTps, $genTps, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    }

    $results += [pscustomobject]@{
        cap_mb          = $cap
        prompt_tps      = $promptTps
        generation_tps  = $genTps
        wall_s          = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        killed          = $killed
        free_ram_before = $freeBefore
        log             = (Split-Path -Leaf $log)
    }
    $results | Export-Csv -NoTypeInformation -Path $csv
}

Write-Host ""
Write-Host "==================== SWEEP RESULT ====================" -ForegroundColor Cyan
Write-Host ("{0,10} {1,14} {2,16} {3,10}" -f "cap MB", "prompt t/s", "generation t/s", "killed")
foreach ($r in $results) {
    Write-Host ("{0,10} {1,14} {2,16} {3,10}" -f $r.cap_mb, $r.prompt_tps, $r.generation_tps, $r.killed)
}
Write-Host ""
Write-Host ("CSV: {0}" -f $csv)

$good = $results | Where-Object { $null -ne $_.generation_tps }
if ($good.Count -ge 2) {
    $lo = $good[0]
    $hi = $good[$good.Count - 1]
    Write-Host ""
    Write-Host ("{0} MB -> {1} t/s, {2} MB -> {3} t/s" -f $lo.cap_mb, $lo.generation_tps, $hi.cap_mb, $hi.generation_tps)
    if ($lo.generation_tps -gt 0) {
        Write-Host ("ratio over the swept range: {0:N2}x" -f ($hi.generation_tps / $lo.generation_tps))
        Write-Host "A FLAT curve means the cache is too small to matter at any of these sizes and"
        Write-Host "the win must come from reading less or reading faster, not from caching more."
    }
}
Write-Host ""
