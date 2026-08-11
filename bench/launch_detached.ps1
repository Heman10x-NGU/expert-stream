# launch_detached.ps1 - start a measurement run that OUTLIVES the shell that started it.
#
# WHY THIS EXISTS
# The agent driving these runs keeps exiting a few minutes into a long background command, and
# when it goes, its child processes go with it. Three multi-minute measurements were lost that
# way, one of them after the run had already produced its answer. Any measurement that takes
# longer than the driver's lifetime has to be decoupled from the driver.
#
# Win32_Process.Create spawns the process from the WMI provider, so the new process is NOT a
# child of this shell and is not in its job object. Killing or losing this shell leaves the run
# alone. Results land in bench\results\runs.csv, which run_first_output.ps1 appends to as each
# run finishes, so progress survives even if nothing is watching.
#
# THIS IS SAFE TO FIRE AND FORGET because the thing it launches limits itself:
#   - run_first_output.ps1 has a hard -TimeoutSec and kills its child when it expires
#   - it has a free-RAM watchdog that kills on 3 consecutive samples below the floor
#   - it applies a hard working-set cap with an enforced upper bound
# It is still printed with its PID and a stop command, because "fire and forget" must never
# mean "no way to stop it".
#
# PowerShell 5.1. Pure ASCII. No && || ternary ??

param(
    [Parameter(Mandatory=$true)][string] $RunArgs,
    # Whitelisted by NAME, not by path. This script hands a command line to
    # Win32_Process.Create, which will happily start anything; restricting it to
    # the two measurement runners in this folder means a typo cannot turn it into
    # a general-purpose detached process launcher.
    [ValidateSet("run_first_output.ps1", "run_moe_trace.ps1")]
    [string] $Runner = "run_first_output.ps1",
    [switch] $WhatIfOnly
)

$ErrorActionPreference = "Stop"

$repo   = Split-Path -Parent $PSScriptRoot
# NOT named $runner: PowerShell variable names are case-insensitive, so assigning
# to $runner would reassign the [ValidateSet] parameter $Runner and re-run its
# validation against a full path, which is not in the set.
$runnerPath = Join-Path $PSScriptRoot $Runner
if (-not (Test-Path -LiteralPath $runnerPath)) { throw "runner not found: $runnerPath" }

# The command line goes through Win32_Process.Create, which uses the Win32 tokenizer: only
# DOUBLE quotes are honoured. Paths with spaces must be double-quoted here, and -RunArgs must
# not contain single-quoted values expecting shell semantics.
if ($RunArgs -match "'") {
    throw "RunArgs contains a single quote. The Win32 tokenizer ignores single quotes - use a path with no spaces instead."
}

$log = Join-Path $repo ("bench\results\detached-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$cmd = ('powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" {1}' -f $runnerPath, $RunArgs)

Write-Host ""
Write-Host "detached launch"
Write-Host ("  command : {0}" -f $cmd)
Write-Host ("  cwd     : {0}" -f $repo)
Write-Host ("  console log is NOT captured (the point is that nothing has to stay attached);")
Write-Host ("  per-run results are appended to bench\results\runs.csv as each run finishes.")
Write-Host ""

if ($WhatIfOnly) {
    Write-Host "-WhatIfOnly: nothing launched."
    return
}

$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $cmd; CurrentDirectory = $repo }

if ($res.ReturnValue -ne 0) {
    # 0 success, 2 access denied, 3 insufficient privilege, 8 unknown failure, 9 path not found,
    # 21 invalid parameter. Anything non-zero means nothing was started.
    throw ("Win32_Process.Create failed with ReturnValue {0} - nothing was launched." -f $res.ReturnValue)
}

$pidNew = [int]$res.ProcessId
Start-Sleep -Milliseconds 700
$alive = Get-Process -Id $pidNew -ErrorAction SilentlyContinue
if (-not $alive) {
    Write-Host ("WARNING: PID {0} was created but is already gone. It probably failed instantly;" -f $pidNew) -ForegroundColor Yellow
    Write-Host "         re-run the same arguments through run_first_output.ps1 directly to see the error." -ForegroundColor Yellow
    return
}

Write-Host ("LAUNCHED detached: PID {0} ({1})" -f $pidNew, $alive.ProcessName) -ForegroundColor Green
Write-Host ""
Write-Host "to watch:"
Write-Host ("  Import-Csv '{0}\bench\results\runs.csv' | Format-Table -AutoSize" -f $repo)
Write-Host "to stop it:"
Write-Host ("  Stop-Process -Id {0}" -f $pidNew)
Write-Host ("  (or create the STOP file: New-Item -ItemType File -Path '{0}\STOP' -Force)" -f $repo)
Write-Host ""
