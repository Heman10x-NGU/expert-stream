#requires -Version 5.1
<#
.SYNOPSIS
    FIRST BASELINE MEASUREMENT: run a 284-billion-parameter LLM (82.5 GB on disk,
    IQ1_S quant, split across 3 GGUF shards) on a laptop that has only 16 GB of RAM
    and a 6 GB GPU (GTX 1660 Ti, ~4.9 GB free).

.DESCRIPTION
    WHY THIS WILL BE SLOW, ON PURPOSE:
    The model is ~82.5 GB. RAM is ~16 GB, of which only ~7.9 GB is free. VRAM is
    6 GB, of which only ~4.9 GB is free. There is no way this model "fits" in
    fast memory. With --cpu-moe, attention runs on the GPU and the (much larger)
    expert tensors run on the CPU; with mmap left ON (never --mlock, never
    --no-mmap) the OS treats the model file as file-backed, evictable pages and
    will thrash it in and out from disk (E:) as needed instead of crashing.

    The expected result is 0.05 - 0.3 tokens/sec. That terrible number is not a
    bug in this script -- it IS the deliverable. It is the baseline that every
    later optimization (quantization changes, offload tuning, speculative
    decoding, etc.) must be measured against and must beat. Do not "fix" the
    speed here. This script's only job is to run the model SAFELY and report
    HONEST numbers.

    SAFETY RULES ENFORCED BY THIS SCRIPT (see inline comments):
      1. --mlock is never passed, and is actively blocked if supplied via -ExtraArgs.
      2. --no-mmap is never passed, and is actively blocked if supplied via -ExtraArgs.
      3. Small context (--ctx-size, default 4096) to keep the KV cache small.
      4. --cpu-moe always used: attention on GPU, experts on CPU.
      5. The llama-cli.exe process is launched at BelowNormal priority so the
         desktop / mouse / keyboard stay responsive while this runs for a long time.
      6. You can abort a run in progress by creating a file named STOP in the
         repository root. The script checks for it before every run.

.PARAMETER ModelPath
    Path to shard 1 of 3. llama.cpp auto-discovers shards 2 and 3 in the same folder.

.PARAMETER BinDir
    Folder containing llama-cli.exe, ggml-cuda.dll, cublas64_12.dll, cudart64_12.dll.

.PARAMETER Ctx
    Context size. Kept small on purpose -- default 4096.

.PARAMETER NGpuLayers
    Number of transformer layers offloaded to GPU. Default 0 because we do not
    yet know how much of this model fits in ~4.9 GB of free VRAM. Raise this
    yourself once you know a safe value; this script will not guess for you.

.PARAMETER Prompt
    Fixed prompt so runs are comparable across time and across machines.

.PARAMETER NPredict
    Tokens to generate per run. Default 32 (kept small: at 0.05-0.3 tok/s this
    can already take minutes).

.PARAMETER Runs
    Number of repetitions. Default 3. Median +/- min/max is reported, never
    just the best run.

.PARAMETER Preflight
    Run every safety/environment check, print exactly what command line WOULD
    be executed for each run, and exit without touching the model or running
    any inference. Use this first, always.

.PARAMETER ExtraArgs
    Extra arguments passed through to llama-cli.exe AFTER validation. If this
    array contains --mlock or --no-mmap (in any casing/spacing) the script
    refuses to run at all, in preflight or for real.

.EXAMPLE
    .\run_baseline.ps1 -Preflight

.EXAMPLE
    .\run_baseline.ps1
#>

[CmdletBinding()]
param(
    [string]$ModelPath = 'E:\models\ds4f-iq1s\UD-IQ1_S\DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf',
    [string]$BinDir = 'E:\tools\llamacpp',
    [int]$Ctx = 4096,
    [int]$NGpuLayers = 0,
    [string]$Prompt = 'Write a Python function that reverses a linked list. Explain your approach in two sentences.',
    [int]$NPredict = 32,
    [int]$Runs = 3,
    [switch]$Preflight,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
$OutDir   = Join-Path $PSScriptRoot 'results'
$StopFile = Join-Path $RepoRoot 'STOP'
$CsvPath  = Join-Path $OutDir 'baseline-summary.csv'

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

function Format-Arg {
    param([string]$Value)
    if ($Value -match '\s') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Get-FreeRamMB {
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round($os.FreePhysicalMemory / 1KB, 1)
}

function Get-TotalRamMB {
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round($os.TotalVisibleMemorySize / 1KB, 1)
}

function Get-FreeVramMB {
    try {
        $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) {
            $line = ($raw | Select-Object -First 1).ToString().Trim()
            return [double]$line
        }
    } catch {
        # fall through
    }
    return $null
}

function Get-FreeDiskGB {
    param([string]$DriveLetter = 'E')
    try {
        $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$DriveLetter`:'"
        if ($disk) {
            return [math]::Round($disk.FreeSpace / 1GB, 2)
        }
    } catch {
        # fall through
    }
    return $null
}

function Test-DefenderExclusion {
    param([string]$PathToCheck = 'E:\models')
    $result = [ordered]@{
        Checked  = $false
        Excluded = $false
        Detail   = ''
    }
    try {
        $pref = Get-MpPreference
        $result.Checked = $true
        $excl = $pref.ExclusionPath
        if ($excl) {
            foreach ($p in $excl) {
                $pNorm = $p.TrimEnd('\')
                if ($PathToCheck.TrimEnd('\') -ieq $pNorm) {
                    $result.Excluded = $true
                    $result.Detail = "Exact match: $p"
                    break
                }
                if ($PathToCheck -like ($pNorm + '\*') -or $PathToCheck -like ($pNorm + '*')) {
                    $result.Excluded = $true
                    $result.Detail = "Covered by: $p"
                    break
                }
            }
        }
        if (-not $result.Excluded) {
            $result.Detail = 'Not found in ExclusionPath list'
        }
    } catch {
        $result.Checked = $false
        $result.Detail = "Could not query Get-MpPreference: $($_.Exception.Message)"
    }
    return $result
}

function Test-ExtraArgsSafe {
    param([string[]]$ArgsToCheck)
    $bad = @()
    foreach ($a in $ArgsToCheck) {
        if ($a -imatch '^--?mlock$') { $bad += $a }
        if ($a -imatch '^--?no-mmap$') { $bad += $a }
    }
    return $bad
}

function Get-MedianMinMax {
    param([double[]]$Values)
    $clean = $Values | Where-Object { $_ -ne $null -and -not [double]::IsNaN($_) }
    if (-not $clean -or $clean.Count -eq 0) {
        return [ordered]@{ Median = $null; Min = $null; Max = $null; Count = 0 }
    }
    $sorted = $clean | Sort-Object
    $n = $sorted.Count
    if ($n % 2 -eq 1) {
        $median = $sorted[[int](($n - 1) / 2)]
    } else {
        $median = ($sorted[$n / 2 - 1] + $sorted[$n / 2]) / 2.0
    }
    return [ordered]@{
        Median = [math]::Round($median, 4)
        Min    = [math]::Round(($sorted | Select-Object -First 1), 4)
        Max    = [math]::Round(($sorted | Select-Object -Last 1), 4)
        Count  = $n
    }
}

function Parse-LlamaLog {
    param([string[]]$LogLines)

    $r = [ordered]@{
        PrefillTokens = $null
        PrefillMs     = $null
        PrefillTps    = $null
        DecodeTokens  = $null
        DecodeMs      = $null
        DecodeTps     = $null
        TtftS         = $null
    }

    foreach ($line in $LogLines) {
        if ($line -match 'prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens.*?([0-9.]+)\s*tokens per second') {
            $r.PrefillMs     = [double]$Matches[1]
            $r.PrefillTokens = [int]$Matches[2]
            $r.PrefillTps    = [double]$Matches[3]
            $r.TtftS         = [math]::Round($r.PrefillMs / 1000.0, 4)
            continue
        }
        if ($line -match '^\s*llama_perf_context_print:\s*eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*runs.*?([0-9.]+)\s*tokens per second') {
            $r.DecodeMs     = [double]$Matches[1]
            $r.DecodeTokens = [int]$Matches[2]
            $r.DecodeTps    = [double]$Matches[3]
            continue
        }
    }
    return $r
}

function Write-CsvRow {
    param(
        [string]$Path,
        [string]$Timestamp,
        [int]$Run,
        [int]$Ctx,
        [int]$Ngl,
        [object]$PrefillTps,
        [object]$DecodeTps,
        [object]$TtftS,
        [object]$WallS,
        [object]$FreeRamBeforeMb,
        [object]$FreeRamAfterMb,
        [bool]$DefenderExcluded,
        [string]$Notes
    )
    $needsHeader = -not (Test-Path $Path)
    $row = [pscustomobject]@{
        timestamp            = $Timestamp
        run                  = $Run
        ctx                  = $Ctx
        ngl                  = $Ngl
        prefill_tps          = $PrefillTps
        decode_tps           = $DecodeTps
        ttft_s               = $TtftS
        wall_s               = $WallS
        free_ram_before_mb   = $FreeRamBeforeMb
        free_ram_after_mb    = $FreeRamAfterMb
        defender_excluded    = $DefenderExcluded
        notes                = $Notes
    }
    if ($needsHeader) {
        $row | Export-Csv -Path $Path -NoTypeInformation -Encoding ASCII
    } else {
        $row | Export-Csv -Path $Path -NoTypeInformation -Encoding ASCII -Append
    }
}

# ----------------------------------------------------------------------------
# 0. Validate ExtraArgs FIRST -- this gate applies in preflight and real runs
# ----------------------------------------------------------------------------
$badArgs = Test-ExtraArgsSafe -ArgsToCheck $ExtraArgs
if ($badArgs.Count -gt 0) {
    Write-Host ''
    Write-Host '=====================================================================' -ForegroundColor Red
    Write-Host 'REFUSING TO RUN: forbidden flag(s) detected in -ExtraArgs' -ForegroundColor Red
    Write-Host ('  ' + ($badArgs -join ', ')) -ForegroundColor Red
    Write-Host ''
    Write-Host '--mlock would try to lock the entire ~82.5 GB model into real RAM' -ForegroundColor Red
    Write-Host 'on a machine with ~16 GB total. That will not work and can hang' -ForegroundColor Red
    Write-Host 'or crash the system.' -ForegroundColor Red
    Write-Host ''
    Write-Host '--no-mmap turns off file-backed paging for the model. mmap must' -ForegroundColor Red
    Write-Host 'stay ON so the OS can evict cold pages back to disk instead of' -ForegroundColor Red
    Write-Host 'the process dying or the machine swapping itself into the ground.' -ForegroundColor Red
    Write-Host '=====================================================================' -ForegroundColor Red
    Write-Host ''
    exit 1
}

# ----------------------------------------------------------------------------
# 1. Preflight checks
# ----------------------------------------------------------------------------
Write-Host ''
Write-Host '=== PREFLIGHT: DeepSeek-V4-Flash 284B IQ1_S baseline ===' -ForegroundColor Cyan
Write-Host ''

$preflightOk = $true

# --- 1a. Model shards ---
$modelDir  = Split-Path -Parent $ModelPath
$leaf1     = Split-Path -Leaf $ModelPath
$leaf2     = $leaf1 -replace '00001-of-00003', '00002-of-00003'
$leaf3     = $leaf1 -replace '00001-of-00003', '00003-of-00003'
$shard2    = Join-Path $modelDir $leaf2
$shard3    = Join-Path $modelDir $leaf3

$expectedSizes = [ordered]@{
    $ModelPath = 5257664
    $shard2    = 49093726624
    $shard3    = 33440253504
}

Write-Host '[1] Model shards' -ForegroundColor Yellow
foreach ($shardPath in $expectedSizes.Keys) {
    $expected = $expectedSizes[$shardPath]
    if (Test-Path -LiteralPath $shardPath) {
        $actual = (Get-Item -LiteralPath $shardPath).Length
        if ($actual -eq $expected) {
            Write-Host ("    OK    {0}  ({1} bytes)" -f $shardPath, $actual) -ForegroundColor Green
        } else {
            Write-Host ("    MISMATCH  {0}  expected {1} bytes, found {2} bytes" -f $shardPath, $expected, $actual) -ForegroundColor Red
            $preflightOk = $false
        }
    } else {
        Write-Host ("    MISSING  {0}" -f $shardPath) -ForegroundColor Red
        $preflightOk = $false
    }
}

# --- 1b. Binaries ---
Write-Host '[2] llama.cpp binaries' -ForegroundColor Yellow
$cliExe  = Join-Path $BinDir 'llama-cli.exe'
$cudaDll = Join-Path $BinDir 'ggml-cuda.dll'
foreach ($binPath in @($cliExe, $cudaDll)) {
    if (Test-Path -LiteralPath $binPath) {
        Write-Host ("    OK    {0}" -f $binPath) -ForegroundColor Green
    } else {
        Write-Host ("    MISSING  {0}" -f $binPath) -ForegroundColor Red
        $preflightOk = $false
    }
}

# --- 1c. Resources ---
Write-Host '[3] System resources' -ForegroundColor Yellow
$freeRamMb  = Get-FreeRamMB
$totalRamMb = Get-TotalRamMB
$freeVramMb = Get-FreeVramMB
$freeDiskGb = Get-FreeDiskGB -DriveLetter 'E'

Write-Host ("    RAM total : {0} MB" -f $totalRamMb)
Write-Host ("    RAM free  : {0} MB" -f $freeRamMb)
if ($freeVramMb -ne $null) {
    Write-Host ("    VRAM free : {0} MB (via nvidia-smi)" -f $freeVramMb)
} else {
    Write-Host "    VRAM free : could not query nvidia-smi (is it on PATH?)" -ForegroundColor DarkYellow
}
if ($freeDiskGb -ne $null) {
    Write-Host ("    Disk E: free : {0} GB" -f $freeDiskGb)
} else {
    Write-Host "    Disk E: free : could not query" -ForegroundColor DarkYellow
}

# --- 1d. Windows Defender exclusion ---
Write-Host '[4] Windows Defender exclusion for E:\models' -ForegroundColor Yellow
$defender = Test-DefenderExclusion -PathToCheck 'E:\models'
if (-not $defender.Checked) {
    Write-Host ("    UNKNOWN  {0}" -f $defender.Detail) -ForegroundColor DarkYellow
} elseif ($defender.Excluded) {
    Write-Host ("    OK  E:\models is excluded ({0})" -f $defender.Detail) -ForegroundColor Green
} else {
    Write-Host '    WARNING: E:\models is NOT excluded from Windows Defender.' -ForegroundColor Red
    Write-Host '    Every page faulted in from the 82.5 GB model file may be scanned' -ForegroundColor Red
    Write-Host '    on read. You may end up benchmarking the antivirus scanner, not' -ForegroundColor Red
    Write-Host '    the model. This does NOT block the run, but the numbers should' -ForegroundColor Red
    Write-Host '    be treated as suspect until fixed. Fix (needs an admin shell):' -ForegroundColor Red
    Write-Host "        Add-MpPreference -ExclusionPath 'E:\models'" -ForegroundColor Red
}

# --- 1e. Extra args already validated above; report again for the record ---
Write-Host '[5] -ExtraArgs safety scan' -ForegroundColor Yellow
if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    Write-Host ("    Extra args: {0}" -f ($ExtraArgs -join ' '))
} else {
    Write-Host '    (none)'
}
Write-Host '    OK -- no --mlock / --no-mmap present' -ForegroundColor Green

Write-Host ''
if (-not $preflightOk) {
    Write-Host 'PREFLIGHT FAILED -- refusing to run. Fix the items above and re-run.' -ForegroundColor Red
    Write-Host ''
    exit 1
}
Write-Host 'PREFLIGHT PASSED (Defender warning, if any, is advisory only).' -ForegroundColor Green
Write-Host ''

# ----------------------------------------------------------------------------
# 2. Build the command line that WOULD be / IS executed
# ----------------------------------------------------------------------------
$baseArgList = @(
    '-m', (Format-Arg $ModelPath),
    '-c', $Ctx,
    '-ngl', $NGpuLayers,
    '--cpu-moe',
    '-p', (Format-Arg $Prompt),
    '-n', $NPredict,
    '--seed', '42',
    '--no-cnv'
)
if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $baseArgList += $ExtraArgs
}
$argString = $baseArgList -join ' '

Write-Host '=== Command that will be run for each of the requested runs ===' -ForegroundColor Cyan
Write-Host ("    {0} {1}" -f $cliExe, $argString)
Write-Host ("    Runs: {0}   Working dir: {1}   Priority: BelowNormal" -f $Runs, $BinDir)
Write-Host ("    STOP file checked before each run: {0}" -f $StopFile)
Write-Host ''

if ($Preflight) {
    Write-Host '-Preflight was passed: exiting without running any inference.' -ForegroundColor Cyan
    Write-Host ''
    exit 0
}

# ----------------------------------------------------------------------------
# 3. Loud warning before touching anything
# ----------------------------------------------------------------------------
Write-Host '=====================================================================' -ForegroundColor Magenta
Write-Host 'ABOUT TO RUN A 284B PARAMETER MODEL WITH 82.5 GB ON DISK, ON A' -ForegroundColor Magenta
Write-Host '16 GB RAM / 6 GB VRAM LAPTOP. THIS IS EXPECTED TO BE EXTREMELY SLOW.' -ForegroundColor Magenta
Write-Host '' -ForegroundColor Magenta
Write-Host 'Expect roughly 0.05 - 0.3 tokens/sec: a word every 3 to 20+ seconds.' -ForegroundColor Magenta
Write-Host 'Expect the machine to feel sluggish (disk thrashing, high CPU).' -ForegroundColor Magenta
Write-Host 'The llama-cli.exe process is launched at BelowNormal priority so' -ForegroundColor Magenta
Write-Host 'the desktop should stay usable, but be patient.' -ForegroundColor Magenta
Write-Host '' -ForegroundColor Magenta
Write-Host ("To ABORT at any point, create an empty file named STOP here:") -ForegroundColor Magenta
Write-Host ("    {0}" -f $StopFile) -ForegroundColor Magenta
Write-Host 'It is checked before every run (it will not kill a run already' -ForegroundColor Magenta
Write-Host 'in progress -- close the window or Ctrl+C for that; the finally' -ForegroundColor Magenta
Write-Host 'block below will make sure the process is killed, not orphaned.)' -ForegroundColor Magenta
Write-Host '=====================================================================' -ForegroundColor Magenta
Write-Host ''
Start-Sleep -Seconds 3

# ----------------------------------------------------------------------------
# 4. Run loop
# ----------------------------------------------------------------------------
$allPrefill = @()
$allDecode  = @()
$allTtft    = @()
$allWall    = @()

for ($i = 1; $i -le $Runs; $i++) {

    if (Test-Path -LiteralPath $StopFile) {
        Write-Host ("STOP file found ({0}) -- aborting before run {1}." -f $StopFile, $i) -ForegroundColor Red
        break
    }

    $ts = Get-Date -Format 'yyyyMMdd-HHmmss'
    $logPath    = Join-Path $OutDir ("baseline-run{0}-{1}.log" -f $i, $ts)
    $stdoutPath = Join-Path $OutDir ("baseline-run{0}-{1}.stdout.tmp" -f $i, $ts)
    $stderrPath = Join-Path $OutDir ("baseline-run{0}-{1}.stderr.tmp" -f $i, $ts)

    Write-Host ("--- Run {0} of {1} (starting {2}) ---" -f $i, $Runs, $ts) -ForegroundColor Cyan
    $freeRamBefore = Get-FreeRamMB
    Write-Host ("    Free RAM before: {0} MB" -f $freeRamBefore)

    $proc = $null
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $notes = ''
    $exitCode = $null

    try {
        $startParams = @{
            FilePath               = $cliExe
            ArgumentList            = $argString
            WorkingDirectory        = $BinDir
            RedirectStandardOutput  = $stdoutPath
            RedirectStandardError   = $stderrPath
            RedirectStandardInput   = 'NUL'
            NoNewWindow             = $true
            PassThru                = $true
        }
        $proc = Start-Process @startParams

        try {
            $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
        } catch {
            Write-Host ("    (could not set BelowNormal priority: {0})" -f $_.Exception.Message) -ForegroundColor DarkYellow
        }

        Write-Host "    Running... (this can take a long time, see warning above)"
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode
    }
    finally {
        $sw.Stop()
        if ($proc -and -not $proc.HasExited) {
            Write-Host '    Killing still-running llama-cli.exe process (interrupted).' -ForegroundColor Red
            try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
        }
    }

    $wallS = [math]::Round($sw.Elapsed.TotalSeconds, 3)
    $freeRamAfter = Get-FreeRamMB
    Write-Host ("    Free RAM after : {0} MB" -f $freeRamAfter)
    Write-Host ("    Wall time      : {0} s" -f $wallS)

    if ($exitCode -ne 0) {
        $notes = "nonzero exit code $exitCode"
        Write-Host ("    WARNING: llama-cli.exe exited with code {0}" -f $exitCode) -ForegroundColor Red
    }

    # Merge stdout+stderr into the final human-readable log, then clean up temps.
    $combined = @()
    $combined += "=== STDOUT ($cliExe $argString) ==="
    if (Test-Path -LiteralPath $stdoutPath) { $combined += Get-Content -LiteralPath $stdoutPath }
    $combined += ''
    $combined += '=== STDERR (llama.cpp perf timings usually appear here) ==='
    if (Test-Path -LiteralPath $stderrPath) { $combined += Get-Content -LiteralPath $stderrPath }
    $combined | Set-Content -LiteralPath $logPath -Encoding ASCII

    # SAFETY: nothing in this repo deletes files. The raw stdout/stderr captures used to be
    # removed here after being merged into $logPath. They are now kept - they are small, and
    # having the unmerged originals has already proved useful when a log looked wrong.

    $metrics = Parse-LlamaLog -LogLines $combined
    if ($metrics.PrefillTps -eq $null -and $metrics.DecodeTps -eq $null) {
        if ($notes -eq '') { $notes = 'could not parse timing lines from log' } else { $notes += '; could not parse timing lines' }
    }

    Write-Host ("    Prefill (prompt eval) : {0} tok/s" -f $metrics.PrefillTps)
    Write-Host ("    Decode  (generation)  : {0} tok/s" -f $metrics.DecodeTps)
    Write-Host ("    TTFT (approx, = prefill time) : {0} s" -f $metrics.TtftS)
    Write-Host ("    Log: {0}" -f $logPath)
    Write-Host ''

    if ($metrics.PrefillTps -ne $null) { $allPrefill += [double]$metrics.PrefillTps }
    if ($metrics.DecodeTps  -ne $null) { $allDecode  += [double]$metrics.DecodeTps }
    if ($metrics.TtftS      -ne $null) { $allTtft    += [double]$metrics.TtftS }
    $allWall += [double]$wallS

    Write-CsvRow -Path $CsvPath -Timestamp $ts -Run $i -Ctx $Ctx -Ngl $NGpuLayers `
        -PrefillTps $metrics.PrefillTps -DecodeTps $metrics.DecodeTps -TtftS $metrics.TtftS `
        -WallS $wallS -FreeRamBeforeMb $freeRamBefore -FreeRamAfterMb $freeRamAfter `
        -DefenderExcluded ([bool]$defender.Excluded) -Notes $notes
}

# ----------------------------------------------------------------------------
# 5. Final summary
# ----------------------------------------------------------------------------
$prefillStats = Get-MedianMinMax -Values $allPrefill
$decodeStats  = Get-MedianMinMax -Values $allDecode
$ttftStats    = Get-MedianMinMax -Values $allTtft
$wallStats    = Get-MedianMinMax -Values $allWall

Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host 'BASELINE SUMMARY (this terrible number is the deliverable)' -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ("Runs completed          : {0} of {1}" -f $prefillStats.Count, $Runs)
Write-Host ''
Write-Host 'PREFILL (prompt eval) tokens/sec:' -ForegroundColor Yellow
Write-Host ("    median = {0}   min = {1}   max = {2}" -f $prefillStats.Median, $prefillStats.Min, $prefillStats.Max)
Write-Host ''
Write-Host 'DECODE (generation) tokens/sec:' -ForegroundColor Yellow
Write-Host ("    median = {0}   min = {1}   max = {2}" -f $decodeStats.Median, $decodeStats.Min, $decodeStats.Max)
Write-Host ''
Write-Host 'These are DIFFERENT numbers. Prefill and decode throughput are not' -ForegroundColor DarkGray
Write-Host 'interchangeable -- reporting only one, or averaging them, misrepresents' -ForegroundColor DarkGray
Write-Host 'what this machine can actually do.' -ForegroundColor DarkGray
Write-Host ''
Write-Host ("TTFT approx (s)         : median = {0}   min = {1}   max = {2}" -f $ttftStats.Median, $ttftStats.Min, $ttftStats.Max)
Write-Host ("Wall time per run (s)   : median = {0}   min = {1}   max = {2}" -f $wallStats.Median, $wallStats.Min, $wallStats.Max)
Write-Host ''
Write-Host ("CSV      : {0}" -f $CsvPath)
Write-Host ("Log dir  : {0}" -f $OutDir)
Write-Host '=====================================================================' -ForegroundColor Cyan
