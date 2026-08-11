# sweep_expert_reads.ps1 - the measurement that decides whether the expert bank
# needs a 76 GB repack, and how much queue depth is worth.
#
# WHAT IT ANSWERS
#   1. Does issuing expert reads in parallel (queue depth > 1) raise throughput?
#      llama.cpp's mmap path faults pages essentially one at a time; the QD=1
#      number here (~400 MB/s, ~4.2 s/token) matches the 0.2 tok/s the real
#      model produces, which is the reason to believe this benchmark models the
#      right thing.
#   2. Is the SPLIT layout (gate/up/down as three separate tensors, 774 reads
#      per token) actually slower than a hypothetical PACKED layout (258
#      contiguous reads per token)? Only a real gap justifies the repack.
#
# METHOD
#   Same seed at every point, so every configuration replays the IDENTICAL
#   sequence of experts. Changing the access pattern between configurations
#   would confound the thing being measured.
#
#   Repeats per point, because the earlier disk sweeps measured the same file at
#   318 MB/s and at 1,050 MB/s minutes apart (SLC cache state and thermals). A
#   single sample from this drive is not evidence.
#
# SAFETY
#   expert_read_bench.exe is read-only: GENERIC_READ, FILE_SHARE_READ, and it
#   never creates or writes a model file. Every invocation is bounded by
#   -MaxSeconds. Nothing here deletes anything.
#
# PowerShell 5.1. Pure ASCII. No && || ternary ??

param(
    [int[]]  $QueueDepths = @(1, 2, 4, 8, 16, 32),
    [string[]] $Modes     = @("split", "packed"),
    [int]    $Tokens      = 2,
    [int]    $Repeats     = 2,
    [int]    $MaxSeconds  = 90,
    [int]    $Sector      = 4096,
    [int]    $Seed        = 42
)

$ErrorActionPreference = "Stop"

$repo     = Split-Path -Parent $PSScriptRoot
$exe      = Join-Path $repo "build\expert_read_bench.exe"
$manifest = Join-Path $repo "bench\results\expert_manifest.csv"
$csv      = Join-Path $repo "bench\results\expert_read_bench.csv"

if (-not (Test-Path -LiteralPath $exe))      { throw "not built: $exe  (gcc -O2 -o build\expert_read_bench.exe src\expert_read_bench.c)" }
if (-not (Test-Path -LiteralPath $manifest)) { throw "manifest missing: $manifest  (run tools\make_expert_manifest.py)" }

# A stale CSV from an earlier code revision would silently mix incomparable
# numbers into the same table. Move it aside rather than delete it - deleting
# files is not this script's job.
if (Test-Path -LiteralPath $csv) {
    $bak = $csv -replace "\.csv$", ("-superseded-{0}.csv" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
    Rename-Item -LiteralPath $csv -NewName (Split-Path -Leaf $bak)
    Write-Host ("existing results moved aside to {0}" -f (Split-Path -Leaf $bak)) -ForegroundColor Yellow
}

$total = $Modes.Count * $QueueDepths.Count * $Repeats
$n = 0
$sw = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host ""
Write-Host ("expert read sweep: {0} points ({1} modes x {2} queue depths x {3} repeats), {4} tokens each" -f `
    $total, $Modes.Count, $QueueDepths.Count, $Repeats, $Tokens)
Write-Host ("unbuffered reads - the OS page cache is bypassed, so nothing else on the machine is evicted")
Write-Host ""

# Repeat is the OUTER loop and mode the inner one, on purpose. Running every
# split point and then every packed point would put minutes of thermal drift and
# SLC cache state change between the two arms of the comparison, and that drift
# would be indistinguishable from a layout effect. Interleaved, both modes see
# the same drive conditions.
for ($r = 1; $r -le $Repeats; $r++) {
    foreach ($qd in $QueueDepths) {
        foreach ($mode in $Modes) {
            $n++
            Write-Host ("[{0}/{1}] mode={2} qd={3} rep={4} ... " -f $n, $total, $mode, $qd, $r) -NoNewline

            # Not redirected: this exe flushes and exits normally, and a previous
            # round of this project lost three multi-minute measurements to a
            # block-buffered stdout that was discarded on kill.
            & $exe --manifest $manifest --mode $mode --qd $qd --tokens $Tokens `
                   --max-seconds $MaxSeconds --sector $Sector --seed $Seed --csv $csv | Out-Null
            $code = $LASTEXITCODE

            if ($code -eq 0) {
                $last = Import-Csv $csv | Select-Object -Last 1
                Write-Host ("{0} MB/s, {1} ms/token" -f $last.mbps, $last.ms_per_token) -ForegroundColor Green
            } elseif ($code -eq 3) {
                Write-Host "hit --max-seconds, partial result recorded" -ForegroundColor Yellow
            } else {
                # Do not invent a data point. A failed run is a hole in the table,
                # not a zero.
                Write-Host ("FAILED exit {0} - no row recorded" -f $code) -ForegroundColor Red
            }
        }
    }
}

$sw.Stop()
Write-Host ""
Write-Host ("sweep finished in {0:N1} s" -f $sw.Elapsed.TotalSeconds)

if (-not (Test-Path -LiteralPath $csv)) {
    Write-Host "no results file was produced - every run failed." -ForegroundColor Red
    return
}

$rows = Import-Csv $csv
Write-Host ""
Write-Host "MEDIAN THROUGHPUT BY MODE AND QUEUE DEPTH"
Write-Host ""

# True median. The obvious [int](($n-1)/2) indexing silently returns the LOWER
# of the two middle values on an even sample count, which is a minimum dressed
# up as a median. Spread is reported alongside it because this drive has been
# measured at 318 and at 1,050 MB/s on the same file minutes apart - a centre
# with no spread next to it would invite conclusions the data cannot support.
function Get-Median([double[]] $v) {
    $s = @($v | Sort-Object)
    $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
}

$summary = foreach ($mode in $Modes) {
    foreach ($qd in $QueueDepths) {
        $g = @($rows | Where-Object { $_.mode -eq $mode -and [int]$_.qd -eq $qd -and [int]$_.timed_out -eq 0 })
        if ($g.Count -eq 0) { continue }
        $mb = @($g | ForEach-Object { [double]$_.mbps })
        $medMb = Get-Median $mb
        # ms/token is derived from the SAME runs, not sorted independently, so
        # the two columns of a row always describe one consistent population.
        $medMs = Get-Median @($g | ForEach-Object { [double]$_.ms_per_token })
        [pscustomobject]@{
            mode          = $mode
            qd            = $qd
            n             = $g.Count
            median_MBps   = [math]::Round($medMb, 1)
            min_MBps      = [math]::Round(($mb | Measure-Object -Minimum).Minimum, 1)
            max_MBps      = [math]::Round(($mb | Measure-Object -Maximum).Maximum, 1)
            median_ms_tok = [math]::Round($medMs, 0)
            io_only_tok_s = [math]::Round(1000.0 / $medMs, 3)
        }
    }
}
$summary | Format-Table -AutoSize
Write-Host ("full table: {0}" -f $csv)
Write-Host ""
Write-Host "io_only_tok_s is an UPPER BOUND: it counts disk time only, with zero"
Write-Host "compute, zero attention, and perfect overlap. The real engine cannot beat it."
