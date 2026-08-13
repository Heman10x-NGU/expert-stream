# run_first_output.ps1 - THE FIRST GENERATED TOKENS FROM DeepSeek-V4-Flash ON THIS LAPTOP.
#
# Launch THROUGH the guard, never directly:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\guard.ps1 -Run "bench\run_first_output.ps1"
#
# WHAT THIS IS AND IS NOT
# This is NOT the engine. It is stock llama.cpp, CPU-only, mmap-backed. We already know from
# MEASURED_GROUND_TRUTH.md section 4.2 that this configuration cannot survive a long
# generation, because Windows retains every faulted-in mmap page: peak memory tracks TOTAL
# DISTINCT BYTES READ, not the working set. Measured drain rate ~400 MB/s of free RAM.
#
# So this script is deliberately scoped to the largest run that has a chance of completing:
#   prompt  : a handful of tokens  -> prefill touches 43 layers * (T*6 capped) experts
#   -n      : a handful of tokens  -> each decode token adds up to 43*6*6.86 MB ~= 1.77 GB
#                                     of NEW distinct expert bytes, less whatever overlaps
#                                     with experts already faulted in.
#
# EITHER OUTCOME IS A RESULT:
#   - It emits N tokens and finishes    -> we have a real baseline tok/s, honestly measured.
#   - The guard kills it at token K     -> K is the exact measured ceiling of the mmap
#                                          approach, which is the quantitative justification
#                                          for building the bounded-cache reader.
# Do not "fix" a low number here. The number is the deliverable.
#
# SAFETY
#   - never --mlock, never --no-mmap (hard-refused below): either commits 82.5 GB of real
#     memory on a 15.4 GB machine.
#   - -fit off is REQUIRED: llama.cpp's auto-fit correctly refuses to load 82.5 GB into
#     15.4 GB and aborts. We override deliberately; mmap keeps the pages file-backed and
#     evictable, so the OS streams from SSD instead of committing.
#   - process runs at BelowNormal so the desktop stays usable.
#   - THIS SCRIPT DELETES NOTHING. The generated prompt file is kept and its path printed.
#   - Start-Process with real file redirection, NOT `2>&1 | Tee-Object`: PowerShell 5.1 wraps
#     a native exe's stderr in NativeCommandError, and llama.cpp logs everything to stderr,
#     which destroys the log.
#
# PowerShell 5.1. Pure ASCII. No && || ternary ??

# -PromptFile exists because this script is launched as `powershell -File <one long string>`
# by guard.ps1. The Win32 command-line tokenizer only honours DOUBLE quotes, so a -Prompt with
# spaces and single quotes splits into positional args and the script throws. A path with no
# spaces sidesteps quoting completely.
param(
    [string] $PromptFile = "",
    [string] $Prompt   = "Q: What is 2+2?`nA:",
    [int]    $NPredict = 8,
    [int]    $CtxSize  = 512,
    [int]    $Threads  = 8,
    [int]    $Seed     = 42,
    [switch] $Repack,
    [switch] $NoPrefetch,
    [int]    $WorkingSetCapMB = 0,
    # 1500, not 700. 700 was chosen when we had NO control over memory and every run had to be
    # allowed to go to the edge. With the working-set cap we choose the memory ceiling directly,
    # so there is no longer any reason to run the machine into the ground - and running it into
    # the ground has a measured cost beyond this process: at ~350 MB free, Windows trims every
    # working set and pages them to D:\pagefile.sys, which is the SAME PHYSICAL DISK we stream
    # the model from. The editor/agent driving these runs was repeatedly killed that way.
    [int]    $WatchdogMinFreeRamMB = 1500,
    [int]    $TimeoutSec = 900,
    # Path to bench\results\expert_manifest.csv. When set, our local ggml-cpu fork streams
    # routed experts off disk with real queue depth instead of taking 4 KB page faults.
    # MUST be a parameter rather than an inherited environment variable: detached runs are
    # started through Win32_Process.Create, which does NOT inherit this shell's environment,
    # so an env var set by the caller would silently vanish and the run would quietly measure
    # the unaccelerated path while claiming otherwise.
    [string] $ExpertManifest = "",
    [int]    $ExpertArenaMB  = 0,
    # --- GPU offload, added 2026-08-13 for the Vulkan build -------------------
    # 0 means CPU-only, which is the behaviour every measurement before today
    # used. Anything above 0 requires a binary built with a GPU backend; on a
    # CPU-only build -ngl is SILENTLY IGNORED, which is exactly how the VRAM
    # item sat at priority 2 for three days while being unexecutable.
    [int]    $GpuLayers = 0,
    # WHICH device, and this matters more than it looks. Vulkan enumerates the
    # AMD integrated GPU too, and its "free memory" is SYSTEM RAM - offloading
    # there would move attention from one part of our 16 GB to another and free
    # precisely nothing. Always name the discrete card explicitly.
    [string] $Device = "",
    # Keep the KV cache in system RAM rather than VRAM. VRAM is the scarce
    # resource here and KV is only ~43 MB at ctx 512, so it is the wrong thing
    # to spend video memory on.
    [switch] $NoKvOffload,
    # Tensor-name override, e.g. "exps=CPU" to keep routed experts on the CPU
    # where our reader handles them. Without this the GPU backend takes the
    # expert matmuls and expert-stream is bypassed entirely.
    [string] $OverrideTensor = "",
    # Micro-batch size. This is the single biggest consumer of VRAM after the
    # weights themselves, and it is NOT small: measured 2026-08-13, the default
    # of 512 asks for a 2,089,862,528 byte compute buffer - 2.09 GB - which on a
    # 6 GB card fails outright with ErrorOutOfDeviceMemory before a single token
    # is generated. PLAN_REMAINING estimated this overhead at 400 MiB. It is
    # five times that. 0 leaves llama.cpp's default alone.
    [int]    $UBatch = 0,
    [string] $Tag      = "first"
)

$ErrorActionPreference = "Stop"

$repo  = Split-Path -Parent $PSScriptRoot
# EXPERT_STREAM_LLAMA_CLI selects the binary, so the CPU-only and Vulkan builds
# can be A/B'd without editing this file. build-cpu stays the default and the
# reference: every measurement before 2026-08-13 was made with it.
$exe   = [Environment]::GetEnvironmentVariable("EXPERT_STREAM_LLAMA_CLI")
if ([string]::IsNullOrEmpty($exe)) { $exe = "D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\build-cpu\bin\llama-cli.exe" }
$model = [Environment]::GetEnvironmentVariable("EXPERT_STREAM_MODEL")
if ([string]::IsNullOrEmpty($model)) { $model = "E:\models\ds4f-iq1s\UD-IQ1_S\DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf" }

# --- edge cases checked BEFORE anything is launched -------------------------------------
if (-not (Test-Path -LiteralPath $exe))   { throw "llama-cli not built at: $exe" }
if (-not (Test-Path -LiteralPath $model)) { throw "model shard 1 not found: $model" }

# llama.cpp auto-discovers shards 2 and 3 from shard 1's name. If they are missing it fails
# deep inside the loader with a confusing error, so check here where the message is clear.
$modelDir = Split-Path -Parent $model
$leaf1    = Split-Path -Leaf $model
foreach ($n in @("00002-of-00003", "00003-of-00003")) {
    $shard = Join-Path $modelDir ($leaf1 -replace "00001-of-00003", $n)
    if (-not (Test-Path -LiteralPath $shard)) { throw "missing shard: $shard" }
}

if ($NPredict -lt 1)  { throw "-NPredict must be >= 1" }
if ($CtxSize  -lt 64) { throw "-CtxSize must be >= 64" }

$cores = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
if ($Threads -gt $cores) {
    Write-Host ("NOTE: -Threads {0} exceeds {1} logical cores; clamping to {1}." -f $Threads, $cores) -ForegroundColor Yellow
    $Threads = $cores
}

$os = Get-CimInstance Win32_OperatingSystem
$freeRamMb = [math]::Round($os.FreePhysicalMemory / 1KB, 0)
if ($freeRamMb -lt 3000) {
    throw ("Only {0} MB free RAM. Refusing to start - close something first." -f $freeRamMb)
}

$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir  = Join-Path $repo "bench\results"
New-Item -ItemType Directory -Force $outDir | Out-Null
$logFile      = Join-Path $outDir ("first_output-{0}-{1}.log" -f $Tag, $stamp)
$errFile      = Join-Path $outDir ("first_output-{0}-{1}.err.log" -f $Tag, $stamp)
$llamaLogFile = Join-Path $outDir ("first_output-{0}-{1}.llama.log" -f $Tag, $stamp)

# Prompt goes through a file so quoting/newlines survive the Win32 command line intact.
if ($PromptFile -ne "") {
    if (-not (Test-Path -LiteralPath $PromptFile)) { throw "PromptFile not found: $PromptFile" }
    $promptFile = (Resolve-Path -LiteralPath $PromptFile).Path
    $Prompt = [System.IO.File]::ReadAllText($promptFile)
} else {
    $promptFile = Join-Path $env:TEMP ("first_output_prompt_{0}.txt" -f $stamp)
    [System.IO.File]::WriteAllText($promptFile, $Prompt, (New-Object System.Text.UTF8Encoding($false)))
}

$argList = @(
    "-m", $model,
    "-f", $promptFile,
    "-c", "$CtxSize",
    "-n", "$NPredict",
    "-t", "$Threads",
    "--seed", "$Seed",
    "--temp", "0",          # greedy: same prompt must give the same tokens across runs
    "-no-cnv",              # one-shot completion, not an interactive chat loop.
                            # NOTE: the flag is "-no-cnv" with a SINGLE dash in this build
                            # ("--no-cnv" is not accepted); the long form is "--no-conversation".
    "-fit", "off",
    # Warmup runs a full forward pass to populate caches. On a machine where the model fits
    # that is free. Here it is a whole extra pass through 43 MoE layers - several GB of expert
    # reads that mmap then RETAINS - spent before the real prompt is even seen. We cannot
    # afford it. (Valid for llama-cli; it is NOT valid for the moe-trace tool's parser.)
    "--no-warmup",
    # -st (single turn) is MANDATORY, not cosmetic. llama-cli enters conversation mode and
    # blocks on stdin after generating. Under a shell that gives it a pipe, stdin hits EOF and
    # it exits; a DETACHED process has no stdin at all and blocks forever. Measured: a finished
    # 8-token run sat at the prompt with zero CPU and zero page faults, and recorded a wall
    # time of 296 s that was almost entirely the hang. -no-cnv did not prevent this.
    "-st"
)
# NOTE: do NOT pass --log-file. It diverts llama.cpp's logging into the file and the console
# stays empty, and the file is block-buffered so a watchdog kill loses it anyway. Measured:
# three runs, three zero-byte logs. The console is the only channel that survives a kill.

# --no-repack is ON BY DEFAULT because it is REQUIRED on this machine, not an optimization.
# Measured A/B, same prompt, same ctx, same binary, only this flag changed:
#     repack on (default)  private 3,664 MB  -> killed at 20 s, no output
#     --no-repack          private 1,400 MB  -> COMPLETED, produced a token
# 2.26 GB, and it is the whole margin between running and not running. The CPU backend repacks
# quantized weights into a SIMD-friendly layout at load, allocating a private buffer and
# copying into it - turning evictable file-backed pages into dirty private memory.
# (An earlier note in this repo claimed --no-repack "does not help". That was measured inside
# runs that were failing for an unrelated reason. It was wrong. See MEASURED_GROUND_TRUTH 8.2.)
# --- GPU offload flags, appended before the forbidden-flag check on purpose --
# so anything added here is still screened for --mlock / --no-mmap.
if ($GpuLayers -gt 0) {
    if ($GpuLayers -gt 43) { throw "GpuLayers $GpuLayers exceeds the model's 43 layers." }
    $argList += @("-ngl", "$GpuLayers")
    # Refuse to guess the device. Picking the integrated GPU by accident would
    # produce a run that looks offloaded and frees no system RAM at all, which
    # is a worse outcome than not running.
    if ($Device -eq "") {
        throw "GpuLayers requires -Device (e.g. Vulkan0). Run 'llama-cli --list-devices' first; the integrated GPU's memory is system RAM and offloading to it frees nothing."
    }
    $argList += @("--device", $Device)
    if ($NoKvOffload)          { $argList += "-nkvo" }
    if ($OverrideTensor -ne "") { $argList += @("-ot", $OverrideTensor) }
    if ($UBatch -gt 0)          { $argList += @("-ub", "$UBatch") }
} else {
    if ($UBatch -gt 0)          { $argList += @("-ub", "$UBatch") }
    if ($Device -ne "" -or $NoKvOffload -or $OverrideTensor -ne "") {
        Write-Host "NOTE: -Device/-NoKvOffload/-OverrideTensor ignored because -GpuLayers is 0." -ForegroundColor Yellow
    }
}

if (-not $Repack) {
    $argList += "--no-repack"
} else {
    Write-Host "WARNING: repacking ENABLED. Measured to cost 2.26 GB of private memory and to" -ForegroundColor Yellow
    Write-Host "         kill the run before any output on this machine." -ForegroundColor Yellow
}

# Hard refusal. Checked against the FINAL list, so it also catches anything added above.
foreach ($a in $argList) {
    if ($a -imatch '^--?(mlock|no-mmap)$') { throw "REFUSING TO RUN: forbidden flag $a" }
}

Write-Host ""
Write-Host "=== FIRST OUTPUT: DeepSeek-V4-Flash 284B, IQ1_S, 82.5 GB, on 15.4 GB of RAM ===" -ForegroundColor Cyan
Write-Host ("  exe       : {0}" -f $exe)
Write-Host ("  model     : {0}" -f $model)
Write-Host ("  prompt    : {0}" -f ($Prompt -replace "`n", "\n"))
Write-Host ("  n_predict : {0}   ctx {1}   threads {2}   temp 0 (greedy)" -f $NPredict, $CtxSize, $Threads)
Write-Host ("  free RAM  : {0} MB" -f $freeRamMb)
Write-Host ("  log       : {0}" -f $logFile)
Write-Host ""
Write-Host "Every generated token must stream ~1.8 GB of expert weights off the SSD at" -ForegroundColor DarkGray
Write-Host "~1.1 GB/s. Seconds per token is the EXPECTED result, not a bug." -ForegroundColor DarkGray
Write-Host ""

# LLAMA_NO_PREFETCH is read by our LOCAL PATCH at llama.cpp src/llama-model.cpp:1532.
# Stock llama.cpp calls init_mappings(true), which hands the ENTIRE mapping to
# PrefetchVirtualMemory on Windows. For an 82.5 GB model on a 15.4 GB machine that faults the
# whole file in at load, evicts everything else, and kills the run before prefill even starts.
# Start-Process inherits this process's environment, so setting it here reaches llama-cli.
if ($NoPrefetch) {
    $env:LLAMA_NO_PREFETCH = "1"
    Write-Host "LLAMA_NO_PREFETCH=1 (whole-mapping PrefetchVirtualMemory disabled)" -ForegroundColor Yellow
} else {
    Remove-Item Env:\LLAMA_NO_PREFETCH -ErrorAction SilentlyContinue
}

# V4F_EXPERT_MANIFEST is read by our LOCAL ggml-cpu fork (expert-stream.c). When set, the
# routed experts for a layer are read with one batch of overlapped unbuffered ReadFile calls
# instead of arriving as 4 KB page faults one at a time. Measured, that is the difference
# between ~469 MB/s and ~800 MB/s.
# ALWAYS set or clear it explicitly. Leaving it inherited from whatever the parent shell
# happened to have would make a run's configuration depend on invisible state, and the whole
# point of runs.csv is that a row describes the run that produced it.
if ($ExpertManifest -ne "") {
    if (-not (Test-Path -LiteralPath $ExpertManifest)) { throw "ExpertManifest not found: $ExpertManifest" }
    $env:V4F_EXPERT_MANIFEST = $ExpertManifest
    Write-Host ("expert streaming ENABLED via {0}" -f $ExpertManifest) -ForegroundColor Green
} else {
    Remove-Item Env:\V4F_EXPERT_MANIFEST -ErrorAction SilentlyContinue
}
if ($ExpertArenaMB -gt 0) {
    # Must match the bounds in expert-stream.c. This is private committed memory on a machine
    # where the model is 5x RAM, so it is bounded on both sides.
    # The LOWER bound matters more than it looks: one token's expert working set is 1.649 GiB,
    # walked cyclically, and a cache smaller than one lap evicts every entry just before its
    # turn comes round again - measured hit rate 0.0%. An arena under ~1800 MB is memory spent
    # for nothing, so anything below that is only useful for testing the plumbing.
    if ($ExpertArenaMB -lt 32 -or $ExpertArenaMB -gt 6144) { throw "ExpertArenaMB must be 32..6144" }
    if ($ExpertArenaMB -gt 32 -and $ExpertArenaMB -lt 1800) {
        Write-Host ("NOTE: arena {0} MB is below one token's 1.649 GiB expert working set - expect ~0% hits." -f $ExpertArenaMB) -ForegroundColor Yellow
    }
    $env:V4F_EXPERT_ARENA_MB = "$ExpertArenaMB"
} else {
    Remove-Item Env:\V4F_EXPERT_ARENA_MB -ErrorAction SilentlyContinue
}

# NO STREAM REDIRECTION, AND AN INLINE WATCHDOG. THIS IS DELIBERATE.
# Three runs in a row were force-killed by the external guard and produced THREE ZERO-BYTE
# logs - stdout, stderr, and llama.cpp's own --log-file. A force-kill discards whatever the C
# runtime still held in its block buffer, and a redirected stream is block-buffered because it
# is not a console. We lost the diagnostics at exactly the moment we needed them.
# Writing to the inherited console instead keeps the output line-buffered and already flushed,
# so everything printed before the kill survives.
# The cost is that the external guard can no longer be the thing that kills us (its window
# would swallow the console), so the RAM watchdog moves in here. Same rule as guard.ps1:
# kill only after MinFreeRamMB is breached on 3 CONSECUTIVE samples, because a single dip is
# noise and killing on it would abort healthy runs.
# WHY A HARD WORKING-SET CAP.
# Measured: the process working set grew to 11,845 MB while system Available fell to 370 MB.
# That means the faulted-in mmap pages were sitting in llama-cli's WORKING SET, not on the
# reclaimable standby list - Windows never trimmed them because pressure arrived faster than
# reclaim. QUOTA_LIMITS_HARDWS_MAX_ENABLE forces the trim: clean, file-backed pages above the
# cap get pushed to standby, where they are reclaimable and cost only a re-read to get back.
# That is exactly the streaming behaviour we want, with the OS acting as a (dumb) cache.
#
# SAFETY: this touches ONLY the PID we just launched, and re-verifies that PID is llama-cli
# before acting. It never enumerates or modifies any other process. If the call fails it warns
# and continues - a failed cap makes the run behave exactly as it did before.
if (-not ("WsCap" -as [type])) {
    Add-Type -Language CSharp -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class WsCap {
  [DllImport("kernel32.dll", SetLastError=true)]
  static extern bool SetProcessWorkingSetSizeEx(IntPtr h, IntPtr min, IntPtr max, uint flags);
  const uint HARDWS_MIN_DISABLE = 0x00000002;
  const uint HARDWS_MAX_ENABLE  = 0x00000004;
  public static void Apply(IntPtr h, long minBytes, long maxBytes) {
    if (!SetProcessWorkingSetSizeEx(h, (IntPtr)minBytes, (IntPtr)maxBytes,
                                    HARDWS_MIN_DISABLE | HARDWS_MAX_ENABLE)) {
      throw new Exception("SetProcessWorkingSetSizeEx failed, error " + Marshal.GetLastWin32Error());
    }
  }
}
"@
}

$sw   = [System.Diagnostics.Stopwatch]::StartNew()
$proc = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -PassThru
try { $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch { }

if ($WorkingSetCapMB -gt 0) {
    if ($proc.ProcessName -ne "llama-cli") {
        throw ("REFUSING to cap PID {0}: expected llama-cli, found '{1}'." -f $proc.Id, $proc.ProcessName)
    }
    # LOWER BOUND: a cap below the non-expert resident set (~2 GB) plus compute buffers
    # (~1.4 GB) would thrash on weights that are needed every single token.
    if ($WorkingSetCapMB -lt 3500) { throw "WorkingSetCapMB below 3500 would thrash the always-needed weights." }

    # UPPER BOUND: the cap decides how much RAM is left for everything else, so it is a system
    # safety parameter, not just a tuning knob. Measured on this machine: cap 8000 settled at
    # 3,750 MB free, so everything-but-llama costs about 15,789 - 8,000 - 3,750 = 4,040 MB.
    # A cap of 10,000 would therefore leave ~1.7 GB free, which is the regime that was killing
    # the agent process. Refuse it rather than discover it again.
    $totalMb    = [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1KB, 0)
    $overheadMb = 4040
    $predictedFreeMb = $totalMb - $WorkingSetCapMB - $overheadMb
    if ($predictedFreeMb -lt ($WatchdogMinFreeRamMB + 1000)) {
        throw ("WorkingSetCapMB {0} would leave about {1} MB free on a {2} MB machine, at or below the {3} MB watchdog floor. Use {4} or less." -f `
            $WorkingSetCapMB, $predictedFreeMb, $totalMb, $WatchdogMinFreeRamMB, ($totalMb - $overheadMb - $WatchdogMinFreeRamMB - 1000))
    }
    Write-Host ("  predicted free RAM during run: ~{0} MB" -f $predictedFreeMb) -ForegroundColor DarkGray
    try {
        [WsCap]::Apply($proc.Handle, 200MB, ([long]$WorkingSetCapMB * 1MB))
        Write-Host ("HARD working-set cap {0} MB applied to PID {1}." -f $WorkingSetCapMB, $proc.Id) -ForegroundColor Green
    } catch {
        Write-Host ("WARNING: could not apply working-set cap: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
        Write-Host "         Run continues UNCAPPED - expect the pre-cap behaviour." -ForegroundColor Yellow
    }
}

$strikes   = 0
$killed    = $false
$minFreeMb = [double]::MaxValue
while (-not $proc.HasExited) {
    Start-Sleep -Milliseconds 1500
    $os   = Get-CimInstance Win32_OperatingSystem
    $free = [math]::Round($os.FreePhysicalMemory / 1KB, 1)
    if ($free -lt $minFreeMb) { $minFreeMb = $free }

    if ($free -lt $WatchdogMinFreeRamMB) { $strikes++ } else { $strikes = 0 }

    if ($strikes -ge 3) {
        Write-Host ""
        Write-Host ("WATCHDOG: free RAM {0} MB below {1} MB for 3 consecutive samples - killing." -f $free, $WatchdogMinFreeRamMB) -ForegroundColor Red
        try { $proc.Kill() } catch { }
        $killed = $true
        break
    }
    if ($sw.Elapsed.TotalSeconds -gt $TimeoutSec) {
        Write-Host ""
        Write-Host ("WATCHDOG: exceeded {0} s timeout - killing." -f $TimeoutSec) -ForegroundColor Red
        try { $proc.Kill() } catch { }
        $killed = $true
        break
    }
}
$proc.WaitForExit()
$code = $proc.ExitCode
$sw.Stop()

Write-Host ""
Write-Host "=== RUN OVER ===" -ForegroundColor Green
Write-Host ("  killed by watchdog : {0}" -f $killed)
Write-Host ("  min free RAM seen  : {0} MB" -f $minFreeMb)

Write-Host ""
Write-Host ("exit code : {0}" -f $code)
Write-Host ("wall time : {0:N1} s" -f $sw.Elapsed.TotalSeconds)

# APPEND ONE ROW PER RUN, IMMEDIATELY.
# The agent driving these runs has been dying mid-sweep, and a sweep that only writes its CSV
# at the end loses every point when that happens. Wall time is recorded because llama-cli's
# own "[ Prompt: X t/s | Generation: Y t/s ]" line is rounded to ONE DECIMAL: 0.24 and 0.16
# both print as "0.2", so it cannot resolve anything smaller than about a 25% change. Wall
# time at a fixed prompt and a large -n has far better resolution, and differencing two -n
# values at the same cap cancels the model load time entirely.
$resultCsv = Join-Path $outDir "runs.csv"
$row = [pscustomobject]@{
    timestamp     = $stamp
    tag           = $Tag
    prompt_file   = (Split-Path -Leaf $promptFile)
    n_predict     = $NPredict
    ctx           = $CtxSize
    threads       = $Threads
    ws_cap_mb     = $WorkingSetCapMB
    repack        = [bool]$Repack
    wall_s        = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    killed        = $killed
    # A run that exits before the first poll never updates this, and writing Double.MaxValue
    # into a results file is how a nonsense number ends up in a chart later.
    min_free_mb   = $(if ($minFreeMb -eq [double]::MaxValue) { $null } else { $minFreeMb })
    exit_code     = $code
}
if (Test-Path -LiteralPath $resultCsv) { $row | Export-Csv -LiteralPath $resultCsv -NoTypeInformation -Append -Encoding ASCII }
else                                   { $row | Export-Csv -LiteralPath $resultCsv -NoTypeInformation -Encoding ASCII }
Write-Host ("appended to : {0}" -f $resultCsv)
Write-Host ("llama log : {0}" -f $llamaLogFile)
# SAFETY: nothing here deletes files. The prompt file is a few hundred bytes in %TEMP%.
Write-Host ("prompt file kept at: {0}" -f $promptFile)
exit $code
