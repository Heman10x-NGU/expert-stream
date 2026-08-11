# batch_moe_trace.ps1 - run several SHORT routing captures and keep going if some fail.
#
# WHY SHORT PROMPTS, AND WHY MANY OF THEM.
# A single-shot prefill touches the union of every prompt token's experts at once, and that
# union saturates fast:
#     12 tokens -> ~19 GB touched -> survives
#     19 tokens -> ~28 GB touched -> guard killed it (394 MB free RAM)
#     39 tokens -> ~46 GB touched -> guard killed it (467 MB free RAM)
#    800 tokens -> ~75 GB touched -> guard killed it (535 MB free RAM)
# Neither --ubatch-size chunking nor --no-repack moved that ceiling.
#
# But union(N) for N consecutive tokens only ever needs N consecutive tokens. So many short
# captures pool perfectly for the overlap curve: a T-token capture contributes T-N+1 windows
# at window size N, and windows never span captures.
#
# Each capture runs under guard.ps1, so a bad one is killed without taking the machine down,
# and this script records the failure and moves on.
#
# PowerShell 5.1. Pure ASCII.

param(
    [string] $PromptGlob = "s??.txt",
    [int]    $CtxSize    = 128,
    [int]    $UBatch     = 8,
    [int]    $PauseSec   = 15
)

$ErrorActionPreference = "Continue"

$repo      = Split-Path -Parent $PSScriptRoot
$promptDir = Join-Path $repo "bench\prompts"
$guard     = Join-Path $repo "tools\guard.ps1"
$runner    = Join-Path $repo "bench\run_moe_trace.ps1"

$files = Get-ChildItem (Join-Path $promptDir $PromptGlob) | Sort-Object Name
if ($files.Count -eq 0) { throw "no prompt files matched $PromptGlob in $promptDir" }

Write-Host ""
Write-Host ("BATCH ROUTING CAPTURE - {0} prompts" -f $files.Count)
Write-Host ("ctx {0}, ubatch {1}, {2}s pause between runs" -f $CtxSize, $UBatch, $PauseSec)
Write-Host ""

$results = @()
$i = 0
foreach ($f in $files) {
    $i++
    $tag = $f.BaseName
    Write-Host ("[{0}/{1}] {2} ... " -f $i, $files.Count, $tag) -NoNewline

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $runArg = ('"{0}" -PromptFile "{1}" -CtxSize {2} -UBatch {3} -Tag {4}' -f $runner, $f.FullName, $CtxSize, $UBatch, $tag)
    & powershell -NoProfile -File $guard -Run $runArg *> $null
    $code = $LASTEXITCODE
    $sw.Stop()

    $csv = Get-ChildItem (Join-Path $repo "bench\results") -Filter ("moe_trace-{0}-*.csv" -f $tag) -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $rows = 0
    if ($csv) { $rows = (Get-Content $csv.FullName | Measure-Object -Line).Lines - 1 }

    if ($code -eq 0 -and $rows -gt 0) {
        Write-Host ("OK   {0,6:N1}s  {1,6} rows" -f $sw.Elapsed.TotalSeconds, $rows) -ForegroundColor Green
        $status = "ok"
    } elseif ($code -eq 2) {
        Write-Host ("KILLED BY GUARD  {0,6:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Red
        $status = "guard_killed"
    } else {
        Write-Host ("FAILED exit {0}  {1,6:N1}s" -f $code, $sw.Elapsed.TotalSeconds) -ForegroundColor Red
        $status = "failed"
    }

    $results += [pscustomobject]@{
        tag = $tag; status = $status; exit_code = $code
        seconds = [math]::Round($sw.Elapsed.TotalSeconds,1); rows = $rows
    }

    # let the page cache settle before the next capture, otherwise pressure accumulates
    # across runs and a later capture gets killed for the previous one's residue.
    if ($i -lt $files.Count) { Start-Sleep -Seconds $PauseSec }
}

Write-Host ""
Write-Host "SUMMARY"
$results | Format-Table -AutoSize
$ok = @($results | Where-Object { $_.status -eq "ok" })
$totalRows = ($results | ForEach-Object { $_.rows } | Measure-Object -Sum).Sum
Write-Host ("captures ok : {0} / {1}" -f $ok.Count, $results.Count)
Write-Host ("total rows  : {0}" -f $totalRows)
if ($ok.Count -gt 0) {
    Write-Host ("token positions captured (approx): {0}" -f [math]::Round($totalRows / (43*6), 0))
}
Write-Host ""
