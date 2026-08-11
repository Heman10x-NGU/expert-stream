<#
.SYNOPSIS
    Safety watchdog for running an 82.5 GB LLM on a 16 GB RAM Windows laptop.

.DESCRIPTION
    This script launches (or, in -DryRun mode, merely observes) a supervised
    process and polls system resources at a fixed interval. If any resource
    crosses a dangerous threshold and STAYS there, the supervised process
    (and all of its children) is force killed before the machine becomes
    unresponsive.

    THIS IS A SAFETY-CRITICAL SCRIPT. The whole point of it is that a runaway
    inference process can NEVER be allowed to make the machine unusable.
    When in doubt, this script should err on the side of killing the
    supervised process rather than letting it keep running.

.PARAMETER Run
    Path to a script (or command) to launch and supervise. Required unless
    -DryRun is specified. Launched as: powershell -NoProfile -File <Run>

.PARAMETER MinFreeRamMB
    If free physical RAM drops below this value for 3 consecutive samples,
    the supervised process is killed. Default 700 MB.

.PARAMETER MaxPagefileMB
    If pagefile (D:\pagefile.sys) CurrentUsage rises above this value for
    3 consecutive samples, the supervised process is killed. Default 6000 MB.

    WHY THIS MATTERS: on this machine, D: and E: are the SAME PHYSICAL DISK,
    and the 82.5 GB model lives on E:. If Windows starts aggressively paging
    (because RAM is nearly exhausted), those pagefile reads/writes to D:
    physically contend with the disk I/O needed to stream model weights from
    E:. Heavy pagefile use is not just a RAM symptom, it directly slows down
    (or stalls) model loading/inference too, and can spiral into a death
    spiral of thrashing. Catching this early is as important as catching low
    RAM directly.

.PARAMETER MaxGpuTempC
    If GPU temperature (GTX 1660 Ti) rises above this value for 3 consecutive
    samples, the supervised process is killed. Default 90 C.

.PARAMETER PollSeconds
    Seconds between samples. Default 2.

.PARAMETER LogPath
    CSV file to append samples to. Default:
    <repo root>\bench\results\guard-<timestamp>.csv

.PARAMETER DryRun
    Monitor only. Do not launch anything via -Run. Useful for testing the
    guard's sampling/logging loop, or for watching resources while you run
    something manually in another window.

.NOTES
    Target shell: Windows PowerShell 5.1 (powershell.exe), NOT PowerShell 7.
    ASCII only. No em-dashes, no smart quotes, no unicode arrows.
#>

param(
    [string]$Run,
    [int]$MinFreeRamMB = 700,
    [int]$MaxPagefileMB = 6000,
    [int]$MaxGpuTempC = 90,
    [int]$PollSeconds = 2,
    [string]$LogPath,
    [switch]$DryRun
)

# ---------------------------------------------------------------------------
# Setup: repo root, STOP-file path, default log path
# ---------------------------------------------------------------------------

# tools\guard.ps1 -> repo root is the parent of $PSScriptRoot
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $RepoRoot) {
    $RepoRoot = (Get-Location).Path
}

# Manual kill switch: drop a file named STOP in the repo root at any time
# and the guard will kill the supervised process on its next poll.
$StopFilePath = Join-Path $RepoRoot "STOP"

if (-not $LogPath -or $LogPath.Trim() -eq "") {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $LogPath = Join-Path $RepoRoot "bench\results\guard-$timestamp.csv"
}

$LogDir = Split-Path -Parent $LogPath
if ($LogDir -and -not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

if (-not $DryRun) {
    if (-not $Run -or $Run.Trim() -eq "") {
        Write-Host "ERROR: -Run <script path> is required unless -DryRun is specified." -ForegroundColor Red
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

function Get-FreeRamMB {
    # Win32_OperatingSystem.FreePhysicalMemory is reported in KB.
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round($os.FreePhysicalMemory / 1KB, 1)
}

function Get-PagefileUsageMB {
    # Win32_PageFileUsage.CurrentUsage is already reported in MB.
    # NOTE: Measure-Object -Property does not work on hashtables, so we
    # extract the property values with ForEach-Object first, then sum them.
    $pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue
    if (-not $pf) {
        return 0
    }
    $sum = ($pf | ForEach-Object { $_.CurrentUsage } | Measure-Object -Sum).Sum
    if (-not $sum) {
        $sum = 0
    }
    return $sum
}

function Get-GpuStats {
    # Returns a PSCustomObject with UsedMB / FreeMB / TempC, or $null if
    # nvidia-smi is unavailable or the query fails for any reason. GPU
    # monitoring is best-effort: a missing GPU must never crash the guard.
    try {
        $raw = & nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits 2>$null
        if (-not $raw) {
            return $null
        }
        $line = ($raw | Select-Object -First 1).ToString().Trim()
        if ($line -eq "") {
            return $null
        }
        $parts = $line -split '\s*,\s*'
        if ($parts.Count -lt 3) {
            return $null
        }
        return [PSCustomObject]@{
            UsedMB = [int]($parts[0].Trim())
            FreeMB = [int]($parts[1].Trim())
            TempC  = [int]($parts[2].Trim())
        }
    } catch {
        return $null
    }
}

function Stop-ProcessTree {
    # Recursively kills a process and all of its descendants. We must kill
    # children first (a killed parent can leave orphaned children behind),
    # so we recurse down to the leaves before killing the process itself.
    param(
        [Parameter(Mandatory = $true)]
        [int]$TargetProcessId
    )
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$TargetProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -TargetProcessId $child.ProcessId
    }
    try {
        Stop-Process -Id $TargetProcessId -Force -ErrorAction Stop
    } catch {
        # Process may have already exited on its own; that is fine.
    }
}

function Write-GuardRow {
    param(
        [string]$Path,
        [string]$Timestamp,
        [double]$ElapsedSec,
        $FreeRamMB,
        $PagefileMB,
        $GpuUsedMB,
        $GpuFreeMB,
        $GpuTempC,
        $ProcAlive,
        $ProcWsMB,
        [string]$KilledReason
    )
    $row = [PSCustomObject]@{
        timestamp     = $Timestamp
        elapsed_sec   = $ElapsedSec
        free_ram_mb   = $FreeRamMB
        pagefile_mb   = $PagefileMB
        gpu_used_mb   = $GpuUsedMB
        gpu_free_mb   = $GpuFreeMB
        gpu_temp_c    = $GpuTempC
        proc_alive    = $ProcAlive
        proc_ws_mb    = $ProcWsMB
        killed_reason = $KilledReason
    }
    $row | Export-Csv -Path $Path -Append -NoTypeInformation
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

$preflightFreeRam = Get-FreeRamMB
$preflightPagefile = Get-PagefileUsageMB
$preflightGpu = Get-GpuStats

Write-Host "===================================================================="
Write-Host "GUARD PREFLIGHT"
Write-Host "===================================================================="
Write-Host ("  Free RAM now      : {0} MB" -f $preflightFreeRam)
if ($preflightGpu) {
    Write-Host ("  GPU free VRAM now : {0} MB (used {1} MB, temp {2} C)" -f $preflightGpu.FreeMB, $preflightGpu.UsedMB, $preflightGpu.TempC)
} else {
    Write-Host "  GPU free VRAM now : unavailable (nvidia-smi not found or query failed)"
}
Write-Host ("  Pagefile usage now: {0} MB" -f $preflightPagefile)
Write-Host "  ----"
Write-Host ("  Threshold MinFreeRamMB  : {0} MB (kill after 3 consecutive samples below)" -f $MinFreeRamMB)
Write-Host ("  Threshold MaxPagefileMB : {0} MB (kill after 3 consecutive samples above)" -f $MaxPagefileMB)
Write-Host ("  Threshold MaxGpuTempC   : {0} C (kill after 3 consecutive samples above)" -f $MaxGpuTempC)
Write-Host ("  PollSeconds             : {0} s" -f $PollSeconds)
Write-Host ("  STOP file               : {0}" -f $StopFilePath)
Write-Host ("  Log path                : {0}" -f $LogPath)
if ($DryRun) {
    Write-Host "  Mode                    : DRY RUN (monitor only, nothing will be launched)"
} else {
    Write-Host ("  Will launch             : {0}" -f $Run)
}
Write-Host "===================================================================="

if (-not $DryRun) {
    if ($preflightFreeRam -lt $MinFreeRamMB) {
        Write-Host ""
        Write-Host ("REFUSING TO LAUNCH: free RAM ({0} MB) is already below MinFreeRamMB ({1} MB)." -f $preflightFreeRam, $MinFreeRamMB) -ForegroundColor Red
        Write-Host "Close other applications and free up memory before starting the run." -ForegroundColor Red
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Launch supervised process (unless -DryRun)
# ---------------------------------------------------------------------------

$proc = $null

if (-not $DryRun) {
    try {
        $proc = Start-Process -FilePath "powershell" -ArgumentList @('-NoProfile', '-File', $Run) -PassThru
    } catch {
        Write-Host ("ERROR: failed to launch -Run target '{0}': {1}" -f $Run, $_.Exception.Message) -ForegroundColor Red
        exit 1
    }

    try {
        $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    } catch {
        Write-Host ("WARNING: could not set process priority to BelowNormal: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
    }

    Write-Host ("Launched PID {0}: {1}" -f $proc.Id, $Run)
}

# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

$freeRamLowCount = 0
$pagefileHighCount = 0
$gpuTempHighCount = 0

$minFreeRamSeen = [double]::MaxValue
$maxPagefileSeen = 0
$maxGpuTempSeen = 0

$killedReason = $null
$exitCode = 0
$startTime = Get-Date

try {
    while ($true) {
        Start-Sleep -Seconds $PollSeconds

        $now = Get-Date
        $elapsed = [math]::Round(($now - $startTime).TotalSeconds, 1)

        $freeRam = Get-FreeRamMB
        $pagefileMB = Get-PagefileUsageMB
        $gpu = Get-GpuStats

        $gpuUsed = $null
        $gpuFree = $null
        $gpuTemp = $null
        if ($gpu) {
            $gpuUsed = $gpu.UsedMB
            $gpuFree = $gpu.FreeMB
            $gpuTemp = $gpu.TempC
        }

        $procAlive = $false
        $procWsMB = 0
        if ($proc) {
            try {
                $proc.Refresh()
                if (-not $proc.HasExited) {
                    $procAlive = $true
                    $procWsMB = [math]::Round($proc.WorkingSet64 / 1MB, 1)
                }
            } catch {
                $procAlive = $false
            }
        }

        if ($freeRam -lt $minFreeRamSeen) { $minFreeRamSeen = $freeRam }
        if ($pagefileMB -gt $maxPagefileSeen) { $maxPagefileSeen = $pagefileMB }
        if ($gpuTemp -ne $null -and $gpuTemp -gt $maxGpuTempSeen) { $maxGpuTempSeen = $gpuTemp }

        Write-GuardRow -Path $LogPath -Timestamp (Get-Date -Format "yyyy-MM-dd HH:mm:ss") `
            -ElapsedSec $elapsed -FreeRamMB $freeRam -PagefileMB $pagefileMB `
            -GpuUsedMB $gpuUsed -GpuFreeMB $gpuFree -GpuTempC $gpuTemp `
            -ProcAlive $procAlive -ProcWsMB $procWsMB -KilledReason ""

        $gpuTempDisplay = "n/a"
        if ($gpuTemp -ne $null) { $gpuTempDisplay = "$gpuTemp C" }
        $gpuMemDisplay = "n/a"
        if ($gpuUsed -ne $null) { $gpuMemDisplay = "$gpuUsed/$gpuFree MB" }

        $statusLine = "[{0,7}s] freeRAM={1,7}MB pagefile={2,6}MB gpu={3,14} gputemp={4,6} proc_alive={5,5} proc_ws={6,8}MB" -f `
            $elapsed, $freeRam, $pagefileMB, $gpuMemDisplay, $gpuTempDisplay, $procAlive, $procWsMB
        Write-Host $statusLine

        # -------------------------------------------------------------
        # Consecutive-sample counters.
        #
        # WHY 3 CONSECUTIVE SAMPLES INSTEAD OF 1: a single low-RAM,
        # high-pagefile, or hot-GPU reading is often just noise -- a
        # transient allocation burst, a moment where the OS is shuffling
        # model layers between GPU and CPU, a brief GC pause, etc. If we
        # killed on the first bad sample, perfectly healthy runs would get
        # aborted constantly and this tool would be useless. Requiring the
        # SAME condition to hold for 3 samples in a row (default 2s apart,
        # so ~6s of sustained badness) filters out spikes while still
        # reacting fast enough to a genuine, sustained resource crisis
        # before it wedges the machine. A single good sample resets the
        # counter to zero -- the condition must be persistent, not just
        # "seen 3 times ever".
        # -------------------------------------------------------------

        if ($freeRam -lt $MinFreeRamMB) {
            $freeRamLowCount++
        } else {
            $freeRamLowCount = 0
        }

        if ($pagefileMB -gt $MaxPagefileMB) {
            $pagefileHighCount++
        } else {
            $pagefileHighCount = 0
        }

        if ($gpuTemp -ne $null -and $gpuTemp -gt $MaxGpuTempC) {
            $gpuTempHighCount++
        } else {
            $gpuTempHighCount = 0
        }

        # Manual kill switch: check every loop.
        if (Test-Path -LiteralPath $StopFilePath) {
            $killedReason = "STOP file detected at $StopFilePath"
            Remove-Item -LiteralPath $StopFilePath -Force -ErrorAction SilentlyContinue
            break
        }

        if ($freeRamLowCount -ge 3) {
            $killedReason = "Free RAM below $MinFreeRamMB MB for 3 consecutive samples (last sample: $freeRam MB)"
            break
        }

        if ($pagefileHighCount -ge 3) {
            # See header comment for why D:/E: being the same physical disk
            # makes pagefile pressure especially dangerous on this machine.
            $killedReason = "Pagefile usage above $MaxPagefileMB MB for 3 consecutive samples (last sample: $pagefileMB MB). D: and E: are the same physical disk as the model directory -- pagefile thrashing steals disk I/O from model reads."
            break
        }

        if ($gpuTempHighCount -ge 3) {
            $killedReason = "GPU temperature above $MaxGpuTempC C for 3 consecutive samples (last sample: $gpuTemp C)"
            break
        }

        if ($proc -and -not $procAlive) {
            # "Exited on its own" is NOT the same as "succeeded". Capture the child's exit
            # code and propagate it, otherwise a run that failed in 2 seconds is reported
            # as a clean completion and the guard exits 0 - which silently masks the
            # failure of whatever we were actually trying to measure.
            try { $script:childExitCode = $proc.ExitCode } catch { $script:childExitCode = $null }
            if ($null -ne $script:childExitCode -and $script:childExitCode -ne 0) {
                Write-Host ("Supervised process exited on its own with EXIT CODE {0} - it FAILED." -f $script:childExitCode) -ForegroundColor Red
            } else {
                Write-Host "Supervised process has exited on its own. Clean completion."
            }
            break
        }

        # In -DryRun mode with nothing launched, keep monitoring
        # indefinitely until Ctrl+C or a STOP file appears.
    }
}
finally {
    # -----------------------------------------------------------------
    # This finally block is the core safety guarantee of this script: it
    # runs whether the loop above ended via a kill condition, the
    # supervised process exiting on its own, an unhandled error, OR the
    # user hitting Ctrl+C. Without this, Ctrl+C-ing the guard would leave
    # an orphaned 82 GB inference process running in the background with
    # nothing left to supervise it -- exactly the disaster this script
    # exists to prevent. So: if the supervised process is still alive at
    # this point for ANY reason, it gets killed, no exceptions.
    # -----------------------------------------------------------------

    $endTime = Get-Date
    $durationSec = [math]::Round(($endTime - $startTime).TotalSeconds, 1)

    if ($proc) {
        try {
            $proc.Refresh()
        } catch {
        }
    }

    if ($proc -and -not $proc.HasExited) {
        Write-Host ""
        Write-Host "===================================================================="
        Write-Host "GUARD SHUTTING DOWN SUPERVISED PROCESS" -ForegroundColor Red
        if ($killedReason) {
            Write-Host ("REASON: {0}" -f $killedReason) -ForegroundColor Red
        } else {
            $killedReason = "Guard is exiting (Ctrl+C or unexpected termination) -- killing supervised process so it is never orphaned."
            Write-Host ("REASON: {0}" -f $killedReason) -ForegroundColor Red
        }
        Write-Host "===================================================================="

        # Try graceful shutdown first.
        try {
            $proc.CloseMainWindow() | Out-Null
        } catch {
        }

        $graceDeadline = (Get-Date).AddSeconds(5)
        while ((Get-Date) -lt $graceDeadline) {
            try {
                $proc.Refresh()
            } catch {
            }
            if ($proc.HasExited) {
                break
            }
            Start-Sleep -Milliseconds 250
        }

        try {
            $proc.Refresh()
        } catch {
        }

        if (-not $proc.HasExited) {
            Write-Host ("Process did not exit gracefully within 5s -- force killing process tree (PID {0})." -f $proc.Id) -ForegroundColor Red
            Stop-ProcessTree -TargetProcessId $proc.Id
        } else {
            Write-Host "Process exited gracefully after CloseMainWindow()."
        }

        $exitCode = 2
    }

    if ($killedReason) {
        $exitCode = 2
        try {
            $finalFreeRam = Get-FreeRamMB
        } catch {
            $finalFreeRam = ""
        }
        try {
            $finalPagefile = Get-PagefileUsageMB
        } catch {
            $finalPagefile = ""
        }
        Write-GuardRow -Path $LogPath -Timestamp (Get-Date -Format "yyyy-MM-dd HH:mm:ss") `
            -ElapsedSec $durationSec -FreeRamMB $finalFreeRam -PagefileMB $finalPagefile `
            -GpuUsedMB "" -GpuFreeMB "" -GpuTempC "" -ProcAlive $false -ProcWsMB "" `
            -KilledReason $killedReason
    }

    $minFreeRamDisplay = $minFreeRamSeen
    if ($minFreeRamSeen -eq [double]::MaxValue) {
        $minFreeRamDisplay = "n/a (no samples taken)"
    }

    Write-Host ""
    Write-Host "===================================================================="
    Write-Host "GUARD SUMMARY"
    Write-Host "===================================================================="
    Write-Host ("  Duration       : {0} sec" -f $durationSec)
    Write-Host ("  Min free RAM   : {0} MB" -f $minFreeRamDisplay)
    Write-Host ("  Max pagefile   : {0} MB" -f $maxPagefileSeen)
    Write-Host ("  Max GPU temp   : {0} C" -f $maxGpuTempSeen)
    if ($killedReason) {
        Write-Host ("  Killed         : YES -- {0}" -f $killedReason) -ForegroundColor Red
    } elseif ($null -ne $script:childExitCode -and $script:childExitCode -ne 0) {
        Write-Host ("  Killed         : no, but the run FAILED (child exit code {0})" -f $script:childExitCode) -ForegroundColor Red
    } else {
        Write-Host "  Killed         : no (clean completion)"
    }
    Write-Host ("  Log file       : {0}" -f $LogPath)
    Write-Host "===================================================================="

    # A guard kill (2) outranks everything. Otherwise surface the child's own exit code, so
    # a failed measurement can never be mistaken for a successful one by a caller or a human
    # skimming the summary.
    if ($exitCode -eq 0 -and $null -ne $script:childExitCode -and $script:childExitCode -ne 0) {
        $exitCode = $script:childExitCode
    }

    exit $exitCode
}
