# sweep_arena.ps1 - O-5: does a bigger expert arena actually buy hit rate?
#
# WHY THIS EXISTS
# The Vulkan offload freed 1,453 MB of system RAM (MEASURED_GROUND_TRUTH section 17) and we
# have never been able to SPEND it. Every projection that says "arena X -> hit rate Y" comes
# from the LRU simulator, and the only two arenas ever validated against the real engine are
# 2000 MB (19.6%) and 2600 MB (22.7%). This walks the arena upward until the machine refuses,
# and records where that wall actually is.
#
# WHY IT WALKS UP INSTEAD OF JUMPING TO THE BIGGEST
# A refused or killed run costs ~2 minutes and tells us nothing except "too big". Walking up
# means every rung below the wall is a real data point, and the wall is found exactly once.
#
# SAFETY. This script:
#   - launches run_first_output.ps1, which owns ALL the memory validation. One owner for that
#     rule; this script never second-guesses the cap arithmetic.
#   - DELETES NOTHING and KILLS NOTHING it did not launch.
#   - refuses to start a rung if free RAM is already below the floor.
#   - ABORTS THE WHOLE SWEEP on the first guard kill rather than hammering a machine that has
#     just told us it is out of memory. Continuing past a kill is how you wedge a laptop.
#   - pauses between rungs so the page cache from the previous run is not charged to the next.
#
# PowerShell 5.1. Pure ASCII. No && || ternary ??

param(
    [int[]]  $ArenaMB          = @(2400, 2900, 3400, 3900),
    [int]    $WorkingSetCapMB  = 6500,
    [int]    $GpuLayers        = 24,
    [string] $Device           = "Vulkan0",
    [int]    $CtxSize          = 512,
    [int]    $NPredict         = 8,
    [int]    $Seed             = 42,
    [int]    $PauseSec         = 20,
    [int]    $MinFreeToStartMB = 6000,
    # WITHOUT THIS THE SWEEP MEASURES THE WRONG THING AND SAYS NOTHING ABOUT IT.
    # run_first_output.ps1 clears V4F_EXPERT_MANIFEST when -ExpertManifest is empty, which
    # disables expert-stream entirely. The run still completes, still prints a wall time, and
    # still looks like a valid data point - it is just the unaccelerated path. An arena sweep
    # with the arena disabled would produce a flat line and a wrong conclusion.
    [string] $ExpertManifest   = "",
    # run_first_output.ps1 defaults to the CPU-only build, which has no Vulkan backend at all.
    # Pointing -GpuLayers at it produces an llama-cli usage error, a 1-second run, and - before
    # the too-fast guard below existed - a cheerful "ok" in the summary table. Select the binary
    # explicitly whenever the GPU is involved.
    [string] $LlamaCli         = "",
    [string] $Tag              = "o5"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo   = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_first_output.ps1"
if (-not (Test-Path $runner)) { throw "runner not found: $runner" }

if ($ExpertManifest -eq "") {
    $ExpertManifest = Join-Path $repo "bench\results\expert_manifest.csv"
}
if (-not (Test-Path -LiteralPath $ExpertManifest)) {
    throw ("expert manifest not found: {0}. Without it expert-stream is DISABLED and this sweep would measure the unaccelerated path while looking valid. Generate it with tools\make_expert_manifest.py." -f $ExpertManifest)
}

if ($LlamaCli -eq "") {
    if ($GpuLayers -gt 0) {
        $LlamaCli = "D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\build-vulkan\bin\llama-cli.exe"
    } else {
        $LlamaCli = "D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\build-cpu\bin\llama-cli.exe"
    }
}
if (-not (Test-Path -LiteralPath $LlamaCli)) { throw "llama-cli not found: $LlamaCli" }
# Start-Process inherits this process's environment, so this reaches run_first_output.ps1.
$env:EXPERT_STREAM_LLAMA_CLI = $LlamaCli

# PREFLIGHT: prove the named device actually exists in THIS binary before burning rungs on it.
# The failure this prevents is not hypothetical - it happened, and the sweep reported "ok".
if ($GpuLayers -gt 0) {
    if ($Device -eq "") { throw "GpuLayers $GpuLayers requires -Device. The integrated GPU's memory is system RAM; offloading there frees nothing." }
    $devs = & $LlamaCli --list-devices 2>&1 | Out-String
    if ($devs -notmatch [regex]::Escape($Device)) {
        Write-Host $devs
        throw ("device '{0}' not present in {1}. The binary has no such backend, so -ngl would be silently ignored or rejected." -f $Device, $LlamaCli)
    }
    Write-Host ("preflight OK: {0} exposes {1}" -f (Split-Path -Leaf $LlamaCli), $Device) -ForegroundColor Green
}

$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = Join-Path $repo "bench\results"
New-Item -ItemType Directory -Force $outDir | Out-Null
$csv    = Join-Path $outDir ("arena_sweep-{0}-{1}.csv" -f $Tag, $stamp)

function Get-FreeMB {
    return [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB, 0)
}

Write-Host ""
Write-Host "===================================================================="
Write-Host "O-5 ARENA SWEEP"
Write-Host "===================================================================="
Write-Host ("  arenas       : {0}" -f ($ArenaMB -join ", "))
Write-Host ("  ws cap       : {0} MB" -f $WorkingSetCapMB)
Write-Host ("  gpu          : -ngl {0} on {1}" -f $GpuLayers, $Device)
Write-Host ("  ctx {0}, {1} tokens, seed {2}" -f $CtxSize, $NPredict, $Seed)
Write-Host ("  free RAM now : {0} MB" -f (Get-FreeMB))
Write-Host ("  csv          : {0}" -f $csv)
Write-Host "===================================================================="
Write-Host ""

$rows = @()
$aborted = $false

foreach ($arena in $ArenaMB) {
    $freeNow = Get-FreeMB
    if ($freeNow -lt $MinFreeToStartMB) {
        Write-Host ("SKIPPING arena {0}: only {1} MB free, need {2} MB to start a rung safely." -f `
            $arena, $freeNow, $MinFreeToStartMB) -ForegroundColor Yellow
        $rows += [pscustomobject]@{
            arena_mb = $arena; ws_cap_mb = $WorkingSetCapMB; ngl = $GpuLayers
            status = "skipped_low_ram"; free_before_mb = $freeNow
            hit_rate_pct = ""; fallbacks = ""; failed_reads = ""; wall_sec = ""; output = ""
        }
        continue
    }

    Write-Host ("--- arena {0} MB (free {1} MB) ---" -f $arena, $freeNow) -ForegroundColor Cyan

    $logTag  = "{0}-a{1}" -f $Tag, $arena
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $argv = @(
        "-NoProfile", "-File", $runner,
        "-ExpertArenaMB", "$arena",
        "-ExpertManifest", $ExpertManifest,
        "-WorkingSetCapMB", "$WorkingSetCapMB",
        "-GpuLayers", "$GpuLayers",
        "-Device", $Device,
        "-NoKvOffload",
        "-OverrideTensor", "exps=CPU",
        "-UBatch", "64",
        "-CtxSize", "$CtxSize",
        "-NPredict", "$NPredict",
        "-Seed", "$Seed",
        "-Tag", $logTag
    )
    # WHY THIS RUN IS REDIRECTED WHEN run_first_output.ps1 DELIBERATELY IS NOT.
    # That script keeps llama-cli on the inherited console because a force-kill discards a
    # redirected C stream's block buffer, and it needs its diagnostics to survive a kill.
    # The sweep has the opposite problem: the one number it exists to collect - the
    # expert-stream cache summary - is printed to that console and is therefore invisible to
    # the caller. Scraping the log files does not work; they do not contain it, and the first
    # attempt at this sweep reported a perfectly good 19.4%-hit run as having measured nothing.
    # TRADE-OFF, stated rather than hidden: a rung killed by the watchdog may lose the tail of
    # its output. That is acceptable because a killed rung's verdict comes from its EXIT CODE,
    # which redirection cannot lose, and the surviving rungs are the ones carrying numbers.
    $rungOut = Join-Path $outDir ("arena_rung-{0}-a{1}.out.log" -f $Tag, $arena)
    $rungErr = Join-Path $outDir ("arena_rung-{0}-a{1}.err.log" -f $Tag, $arena)
    $proc = Start-Process -FilePath "powershell" -ArgumentList $argv -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $rungOut -RedirectStandardError $rungErr
    $code = $proc.ExitCode
    $sw.Stop()

    $hit = ""; $fb = ""; $failed = ""; $outText = ""
    foreach ($f in @($rungOut, $rungErr)) {
        if (-not (Test-Path -LiteralPath $f)) { continue }
        $txt = Get-Content -LiteralPath $f -Raw -ErrorAction SilentlyContinue
        if ($null -eq $txt) { continue }
        # Real format, straight from expert-stream.c:
        #   expert-stream: 10044 reads, 21.34 GiB read | cache 2423/12504 = 19.4% hits,
        #                  5.18 GiB served from RAM | 37 fallbacks, 0 failed
        $m = [regex]::Match($txt, 'cache\s+\d+/\d+\s*=\s*([0-9.]+)%\s*hits')
        if ($m.Success -and $hit -eq "") { $hit = $m.Groups[1].Value }
        $m = [regex]::Match($txt, '(\d+)\s+fallbacks')
        if ($m.Success -and $fb -eq "") { $fb = $m.Groups[1].Value }
        $m = [regex]::Match($txt, '(\d+)\s+failed')
        if ($m.Success -and $failed -eq "") { $failed = $m.Groups[1].Value }
    }
    # Show the operator what happened, since the console no longer streams it live.
    if (Test-Path -LiteralPath $rungOut) {
        Get-Content -LiteralPath $rungOut -ErrorAction SilentlyContinue |
            Where-Object { $_ -match 'expert-stream:|min free RAM|working-set cap|wall time' } |
            ForEach-Object { Write-Host ("      " + $_) }
    }

    $status = "ok"
    if ($code -eq 2) { $status = "guard_killed" }
    elseif ($code -ne 0) { $status = ("failed_exit_{0}" -f $code) }

    # TOO-FAST GUARD. A real rung loads 82.5 GB of mapped weights and generates 8 tokens at
    # roughly 2 s each; nothing legitimate finishes in under 30 seconds. Without this check a
    # binary that rejects its own arguments exits 0 in 1.0 s and lands in the summary as "ok",
    # which is precisely how sixteen captures once reported success while producing nothing.
    # An arena sweep whose rungs all failed identically would still draw a perfectly flat line.
    if ($status -eq "ok" -and $sw.Elapsed.TotalSeconds -lt 30) {
        $status = "failed_too_fast"
    }
    # A rung that produced no hit rate measured nothing, whatever its exit code says.
    if ($status -eq "ok" -and $hit -eq "") {
        $status = "failed_no_hitrate"
    }
    # ...but distinguish the one case that is a RESULT rather than a fault: the reader's own
    # budget guard declining the arena. That is exactly what the sweep is walking up to find.
    # Note the run does NOT stop there - llama-cli carries on with expert-stream disabled, on
    # the slow 4 KB page-fault path, so this rung burns minutes and measures nothing. Detect it
    # from the reader's own message rather than inferring it from silence.
    foreach ($f in @($rungOut, $rungErr)) {
        if (-not (Test-Path -LiteralPath $f)) { continue }
        $txt = Get-Content -LiteralPath $f -Raw -ErrorAction SilentlyContinue
        if ($null -ne $txt -and $txt -match 'arena \d+ MB too large for (\d+) MB available') {
            $status = "arena_refused_budget"
            $outText = ("reader had {0} MB available" -f $Matches[1])
        }
    }

    Write-Host ("    {0}  {1:N1}s  hit={2}%  fallbacks={3}  failed={4}" -f `
        $status, $sw.Elapsed.TotalSeconds, $hit, $fb, $failed)

    $rows += [pscustomobject]@{
        arena_mb = $arena; ws_cap_mb = $WorkingSetCapMB; ngl = $GpuLayers
        status = $status; free_before_mb = $freeNow
        hit_rate_pct = $hit; fallbacks = $fb; failed_reads = $failed
        wall_sec = [math]::Round($sw.Elapsed.TotalSeconds,1); output = $outText
    }

    if ($status -eq "guard_killed") {
        Write-Host ""
        Write-Host ("ABORTING SWEEP: arena {0} MB was killed by the watchdog. That is the wall." -f $arena) -ForegroundColor Red
        Write-Host "Not attempting anything larger - the machine has just said it is out of memory."
        $aborted = $true
        break
    }
    if ($status -eq "arena_refused_budget") {
        Write-Host ""
        Write-Host ("WALL FOUND: the reader declined arena {0} MB ({1}). Every rung below it is valid." -f $arena, $outText) -ForegroundColor Yellow
        Write-Host "Stopping here. Raising -WorkingSetCapMB is the only way past this, and that trades"
        Write-Host "against the watchdog floor - check min free RAM on the last good rung first."
        $aborted = $true
        break
    }
    if ($status -like "failed*") {
        Write-Host ""
        Write-Host ("ABORTING SWEEP: rung {0} MB reported '{1}'. Something is wrong with the SETUP, not the arena size," -f $arena, $status) -ForegroundColor Red
        Write-Host "so every remaining rung would fail the same way and the table would look like a flat result."
        $aborted = $true
        break
    }

    if ($arena -ne $ArenaMB[-1]) {
        Write-Host ("    settling {0}s..." -f $PauseSec)
        Start-Sleep -Seconds $PauseSec
    }
}

$rows | Export-Csv -Path $csv -NoTypeInformation
Write-Host ""
Write-Host "SUMMARY"
$rows | Format-Table -AutoSize
Write-Host ("csv: {0}" -f $csv)
if ($aborted) { Write-Host "sweep aborted at the wall (this is a result, not a failure)." -ForegroundColor Yellow }
Write-Host ""
