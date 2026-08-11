# unblock-admin.ps1
#
# SAFETY MODEL, read this first:
#   - REPORT-ONLY BY DEFAULT. Running it with no arguments changes NOTHING. It prints exactly
#     what it would do and stops. You must pass -Apply for any change to happen.
#   - IT NEVER DELETES A FILE. Nothing in this repo deletes anything on your machine.
#   - IT NEVER KILLS A PROCESS. An earlier version did; that was wrong. See section 3.
#   - Every change it can make is reversible, and the undo command is printed next to it.
#   - It touches exactly two things: Defender EXCLUSION PATHS (it does not disable Defender)
#     and one registry value (TdrDelay).
#
# USAGE
#   Report only (safe, default):
#     powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\unblock-admin.ps1
#   Actually apply, in an ADMINISTRATOR shell:
#     powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\unblock-admin.ps1 -Apply
#
# PowerShell 5.1. Pure ASCII.

param(
    [switch] $Apply,
    [switch] $SkipTdr    # skip the registry change, do Defender exclusions only
)

$ErrorActionPreference = "Continue"

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

$isAdmin = Test-Admin

Write-Host ""
Write-Host "===================================================================="
if ($Apply) {
    Write-Host " UNBLOCK - APPLY MODE (changes will be made)" -ForegroundColor Yellow
} else {
    Write-Host " UNBLOCK - REPORT ONLY (nothing will be changed)" -ForegroundColor Green
}
Write-Host "===================================================================="
Write-Host ("  Administrator : {0}" -f $(if ($isAdmin) { "yes" } else { "NO" }))

if ($Apply -and -not $isAdmin) {
    Write-Host ""
    Write-Host "  -Apply requires an ADMINISTRATOR shell. Nothing was changed." -ForegroundColor Red
    Write-Host "  Right-click Start -> Terminal (Admin), then re-run with -Apply."
    exit 1
}

# --------------------------------------------------------------------------
# 1. Defender exclusion paths
#
# This does NOT disable Defender or turn off real-time protection. It tells
# Defender to skip scanning specific directories that we read tens of GB from.
# Everything outside these paths stays protected exactly as before.
#
# EDGE CASE: an exclusion means a malicious file placed in one of these
# directories would not be scanned. These are directories holding model
# weights and our own build output, so the risk is that you later download
# something untrusted INTO one of them. Keep them for model data only.
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "[1] Defender exclusion paths (protection stays ON everywhere else)"

$paths = @(
    "E:\models",
    "E:\tools\llamacpp",
    "D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\build-cpu",
    "D:\2025_Cursor_Dev\tools\w64devkit"
)

$existing = @()
try {
    $pref = Get-MpPreference -ErrorAction Stop
    if ($pref.ExclusionPath) { $existing = @($pref.ExclusionPath) }
} catch {
    Write-Host "      (could not read current exclusions - needs admin to query)" -ForegroundColor DarkGray
}

foreach ($p in $paths) {
    if (-not (Test-Path $p)) {
        Write-Host ("      SKIP    {0}  (path does not exist)" -f $p) -ForegroundColor DarkGray
        continue
    }
    if ($existing -contains $p) {
        Write-Host ("      ALREADY {0}" -f $p) -ForegroundColor DarkGray
        continue
    }
    if ($Apply) {
        try {
            Add-MpPreference -ExclusionPath $p -ErrorAction Stop
            Write-Host ("      ADDED   {0}" -f $p) -ForegroundColor Green
        } catch {
            Write-Host ("      FAILED  {0}  ({1})" -f $p, $_.Exception.Message) -ForegroundColor Red
        }
    } else {
        Write-Host ("      WOULD ADD  {0}" -f $p) -ForegroundColor Cyan
    }
}

# --------------------------------------------------------------------------
# 2. WDDM TdrDelay
#
# Windows resets the GPU driver if a single kernel runs longer than TdrDelay
# seconds. The default is 2. A large prefill kernel can exceed that, and the
# result is a screen blank, "unspecified launch failure" from CUDA, and a
# dead run.
#
# EDGE CASE YOU SHOULD WEIGH, because it is a real trade:
#   Raising this to 60 means that if the GPU ever genuinely hangs, your
#   display will freeze for a FULL MINUTE before Windows recovers it,
#   instead of recovering in about 2 seconds. You are trading fast recovery
#   from a real hang for tolerance of long legitimate kernels.
#   30 is a reasonable compromise and is used below.
#   This only matters once we run CUDA. Our current build is CPU-only, so
#   you can skip this entirely for now with -SkipTdr.
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "[2] WDDM TdrDelay"

if ($SkipTdr) {
    Write-Host "      SKIPPED by -SkipTdr" -ForegroundColor DarkGray
} else {
    $k = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
    $cur = $null
    try { $cur = (Get-ItemProperty -Path $k -Name TdrDelay -ErrorAction SilentlyContinue).TdrDelay } catch { }
    if ($null -eq $cur) {
        Write-Host "      current : not set (Windows default is 2 seconds)"
    } else {
        Write-Host ("      current : {0}" -f $cur)
    }
    if ($Apply) {
        try {
            Set-ItemProperty -Path $k -Name TdrDelay -Value 30 -Type DWord -ErrorAction Stop
            Write-Host "      SET to 30. A REBOOT is required before it takes effect." -ForegroundColor Yellow
        } catch {
            Write-Host ("      FAILED: {0}" -f $_.Exception.Message) -ForegroundColor Red
        }
    } else {
        Write-Host "      WOULD SET to 30 (reboot required)" -ForegroundColor Cyan
    }
}

# --------------------------------------------------------------------------
# 3. VRAM holders - REPORTED ONLY, NEVER KILLED
#
# An earlier version of this script killed these. That was wrong for two
# reasons, and one of them is now measured:
#   - NVIDIA Overlay.exe RESPAWNS immediately. Killing it gained 47 MiB and
#     the process came straight back with a new PID. It does not work.
#   - Killing msedgewebview2 can take down the UI of whatever app is hosting
#     it, with possible data loss in that app. Not our call to make.
# The GUI route (assign apps to the integrated Radeon GPU) is both safer and
# permanent, so that is what we recommend instead.
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "[3] Processes currently holding the discrete GPU (REPORT ONLY - nothing is killed)"

$gpuProcs = @()
try { $gpuProcs = @(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>$null) } catch { }
if ($gpuProcs.Count -eq 0) {
    Write-Host "      (nvidia-smi returned nothing)" -ForegroundColor DarkGray
} else {
    foreach ($line in $gpuProcs) { Write-Host ("      {0}" -f $line) }
}
$vram = ""
try { $vram = (nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>$null) } catch { }
Write-Host ""
Write-Host ("      VRAM used/free : {0}" -f $vram)
Write-Host "      Attention needs 4864 MiB. Target at least 5400 MiB free."

Write-Host ""
Write-Host "      THE PERMANENT FIX (do this by hand, it beats killing anything):"
Write-Host "        Settings -> System -> Display -> Graphics"
Write-Host "        For WezTerm, Explorer, any browser, NVIDIA app, Cloudflare WARP:"
Write-Host "          Options -> Power saving (integrated Radeon) -> Save"
Write-Host "        Then sign out and back in."
Write-Host "      Why: this laptop has a Radeon iGPU. Anything rendering on the 1660 Ti is"
Write-Host "      taking VRAM we need. Getting attention into VRAM doubles the expert cache"
Write-Host "      from about 4.5 GB to about 9 GB, and cache size is what decides tok/s."

# --------------------------------------------------------------------------
Write-Host ""
Write-Host "===================================================================="
Write-Host " UNDO"
Write-Host "===================================================================="
Write-Host '  Remove-MpPreference -ExclusionPath "E:\models"     (repeat per path)'
Write-Host '  Remove-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -Name TdrDelay'
Write-Host ""
if (-not $Apply) {
    Write-Host "  NOTHING WAS CHANGED. Re-run with -Apply in an admin shell to apply." -ForegroundColor Green
    Write-Host ""
}
