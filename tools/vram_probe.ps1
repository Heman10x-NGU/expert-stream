# vram_probe.ps1 - can the attention set fit on the graphics card right now?
#
# The single highest-leverage open item in this project is a few hundred megabytes
# of video memory. The attention tensors are 5,102,369,536 bytes (4,866 MiB) and
# they are needed for every token, so putting them on the card takes ~5 GB out of
# system RAM and roughly doubles what the expert arena can be.
#
# Free VRAM is NOT a constant. It moves with whatever else is drawing on the card,
# so "4.46-4.84 GB free" recorded once in a log is a snapshot, not a property of
# the machine. This script re-measures it, names what is holding the memory, and
# says whether the offload would fit - so the decision is made against the state
# the run will actually see.
#
# READ-ONLY BY CONSTRUCTION. It queries and prints. It does not kill a process,
# change a display setting, touch the registry, or write anything outside
# bench\results\. Nothing here can leave the machine in a different state.
#
# Usage:
#   .\tools\vram_probe.ps1
#   .\tools\vram_probe.ps1 -CudaOverheadMB 600   # stricter headroom assumption
#   .\tools\vram_probe.ps1 -Record               # append a row to bench\results\

[CmdletBinding()]
param(
    # Bytes CUDA needs on top of the weights: context, kernels, compute buffers.
    # THIS IS AN ESTIMATE, NOT A MEASUREMENT. It is a parameter precisely so the
    # verdict below can never be mistaken for a measured fact.
    [ValidateRange(0, 4096)]
    [int] $CudaOverheadMB = 400,

    [switch] $Record
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Attention tensor bytes, from the GGUF metadata. See MEASURED_GROUND_TRUTH 1.2.
$ATTN_BYTES = 5102369536L
$ATTN_MIB   = [math]::Round($ATTN_BYTES / 1MB, 0)

function Find-NvidiaSmi {
    $cmd = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'),
        (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe')
    )
    foreach ($c in $candidates) { if (Test-Path -LiteralPath $c) { return $c } }
    return $null
}

Write-Host ""
Write-Host "vram_probe - " -NoNewline
Write-Host (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') -ForegroundColor DarkGray

# ---------------------------------------------------------------- adapters

Write-Host ""
Write-Host "display adapters" -ForegroundColor Cyan
$adapters = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue)
if ($adapters.Count -eq 0) {
    Write-Host "  none reported by WMI." -ForegroundColor Yellow
} else {
    foreach ($a in $adapters) {
        Write-Host ("  {0}" -f $a.Name)
    }
}

$hasIgpu = $false
foreach ($a in $adapters) {
    if ($a.Name -match 'Radeon|Intel|UHD|Iris|Vega') { $hasIgpu = $true }
}

# ---------------------------------------------------------------- nvidia-smi

$smi = Find-NvidiaSmi
if (-not $smi) {
    Write-Host ""
    Write-Host "nvidia-smi not found. Cannot measure free VRAM." -ForegroundColor Yellow
    Write-Host "Nothing else in this script can run without it. Exiting cleanly."
    exit 0
}

$q = & $smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>$null
if ($LASTEXITCODE -ne 0 -or -not $q) {
    Write-Host ""
    Write-Host "nvidia-smi ran but returned nothing usable. No NVIDIA GPU visible?" -ForegroundColor Yellow
    exit 0
}

# One line per GPU. This machine has one, but do not assume it.
$gpus = @()
foreach ($line in @($q)) {
    $f = $line -split '\s*,\s*'
    if ($f.Count -lt 4) { continue }
    $gpus += [pscustomobject]@{
        Name     = $f[0]
        TotalMiB = [int] $f[1]
        UsedMiB  = [int] $f[2]
        FreeMiB  = [int] $f[3]
    }
}

if ($gpus.Count -eq 0) {
    Write-Host "could not parse nvidia-smi output." -ForegroundColor Yellow
    exit 0
}

$g = $gpus[0]

Write-Host ""
Write-Host "video memory" -ForegroundColor Cyan
Write-Host ("  {0}" -f $g.Name)
Write-Host ("  total {0,6:N0} MiB" -f $g.TotalMiB)
Write-Host ("  used  {0,6:N0} MiB" -f $g.UsedMiB)
Write-Host ("  free  {0,6:N0} MiB" -f $g.FreeMiB) -ForegroundColor Green

# ---------------------------------------------------------------- who holds it

Write-Host ""
Write-Host "processes with a context on the card" -ForegroundColor Cyan

$apps = & $smi --query-compute-apps=pid --format=csv,noheader,nounits 2>$null
$pidList = @()
if ($LASTEXITCODE -eq 0 -and $apps) {
    foreach ($line in @($apps)) {
        $t = ($line -split ',')[0].Trim()
        if ($t -match '^\d+$') { $pidList += [int] $t }
    }
}

if ($pidList.Count -eq 0) {
    Write-Host "  none reported."
} else {
    # Under WDDM, nvidia-smi cannot attribute VRAM per process - it returns [N/A].
    # Names are still useful: they tell you what to close, which is the whole point.
    $desktop = @()
    foreach ($procId in $pidList) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $p) { continue }
        Write-Host ("  {0,-8} {1}" -f $p.Id, $p.ProcessName)
        if ($p.ProcessName -in @('dwm','explorer','ShellExperienceHost','StartMenuExperienceHost',
                                 'SearchHost','TextInputHost','ShellHost','ApplicationFrameHost')) {
            $desktop += $p.ProcessName
        }
    }
    Write-Host ""
    Write-Host "  note: per-process VRAM is unavailable under Windows WDDM, so the" -ForegroundColor DarkGray
    Write-Host "  numbers above are absent by design rather than missing." -ForegroundColor DarkGray

    if ($desktop.Count -gt 0) {
        Write-Host ""
        Write-Host ("  dwm/shell are rendering on this card ({0})." -f ($desktop -join ', ')) -ForegroundColor Yellow
        if ($hasIgpu) {
            Write-Host "  An integrated GPU is present, so the desktop can be moved off." -ForegroundColor Yellow
        }
    }
}

# ---------------------------------------------------------------- verdict

$needMiB = $ATTN_MIB + $CudaOverheadMB
$slackMiB = $g.FreeMiB - $needMiB

Write-Host ""
Write-Host "would the attention set fit?" -ForegroundColor Cyan
Write-Host ("  attention weights      {0,6:N0} MiB   (5,102,369,536 bytes, from GGUF metadata)" -f $ATTN_MIB)
Write-Host ("  assumed CUDA overhead  {0,6:N0} MiB   ESTIMATE, -CudaOverheadMB to change" -f $CudaOverheadMB)
Write-Host ("  required               {0,6:N0} MiB" -f $needMiB)
Write-Host ("  free right now         {0,6:N0} MiB" -f $g.FreeMiB)

if ($slackMiB -ge 256) {
    Write-Host ("  VERDICT: FITS with {0:N0} MiB to spare. Worth trying a real run." -f $slackMiB) -ForegroundColor Green
} elseif ($slackMiB -ge 0) {
    Write-Host ("  VERDICT: fits by only {0:N0} MiB. Too thin to rely on - one browser tab" -f $slackMiB) -ForegroundColor Yellow
    Write-Host "           takes it away mid-run. Free more before trusting it." -ForegroundColor Yellow
} else {
    Write-Host ("  VERDICT: SHORT BY {0:N0} MiB." -f [math]::Abs($slackMiB)) -ForegroundColor Red
}

if ($slackMiB -lt 256) {
    Write-Host ""
    Write-Host "  ways to close the gap, cheapest first:" -ForegroundColor Cyan
    Write-Host "    1. Close browsers and Electron apps. They hold VRAM for compositing."
    if ($hasIgpu) {
        Write-Host "    2. Move the desktop to the integrated GPU. Settings > System > Display >"
        Write-Host "       Graphics > Default graphics settings, or the laptop's BIOS/MUX setting."
        Write-Host "       dwm alone is usually worth several hundred MiB."
    }
    Write-Host "    3. Turn off the NVIDIA overlay (GeForce Experience in-game overlay)."
    Write-Host "    4. Accept RAM-resident attention and a smaller expert arena."
}

Write-Host ""
Write-Host "  NOTE: this is a snapshot. Re-run it immediately before a real run," -ForegroundColor DarkGray
Write-Host "  because free VRAM moves with whatever else is on screen." -ForegroundColor DarkGray

# ---------------------------------------------------------------- record

if ($Record) {
    $results = Join-Path $PSScriptRoot '..\bench\results'
    $results = [System.IO.Path]::GetFullPath($results)
    if (-not (Test-Path -LiteralPath $results)) {
        New-Item -ItemType Directory -Path $results -Force | Out-Null
    }
    $csv = Join-Path $results 'vram_probe.csv'
    if (-not (Test-Path -LiteralPath $csv)) {
        # Append-only. Nothing in this project overwrites a results file.
        'timestamp,gpu,total_mib,used_mib,free_mib,attn_mib,overhead_mib,slack_mib' |
            Out-File -FilePath $csv -Encoding ascii
    }
    $row = '{0},{1},{2},{3},{4},{5},{6},{7}' -f `
        (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), ($g.Name -replace ',', ' '),
        $g.TotalMiB, $g.UsedMiB, $g.FreeMiB, $ATTN_MIB, $CudaOverheadMB, $slackMiB
    Add-Content -Path $csv -Value $row -Encoding ascii
    Write-Host ""
    Write-Host ("  appended to {0}" -f $csv) -ForegroundColor DarkGray
}

Write-Host ""
