# run_moe_trace.ps1 - capture MoE expert routing from a single prefill pass.
#
# Intended to be launched THROUGH tools\guard.ps1, never directly:
#   .\tools\guard.ps1 -Run ".\bench\run_moe_trace.ps1"
#
# WHY THE PROMPT IS SHORT. Prefill reads the UNION of the experts every prompt token
# routes to. With T tokens that is at most T*6 distinct experts per layer, so bytes read
# grow roughly linearly with prompt length:
#     T tokens  ->  43 layers * min(256, T*6) experts * ~6.86 MB
# At 8 tokens that is about 14 GB; at 512 tokens it is the entire 75 GB expert bank.
# For VALIDATION we only need to prove the callback fires and the values are sane, so we
# use the shortest prompt that exercises the path.
#
# PowerShell 5.1. Pure ASCII.

param(
    [string] $Prompt     = "The capital of France is Paris and the capital of Japan is",
    [string] $PromptFile = "",
    [int]    $CtxSize    = 512,
    [int]    $Threads    = 8,
    [int]    $UBatch     = 0,
    [switch] $NoRepack,
    [int]    $WorkingSetCapMB      = 0,
    [int]    $WatchdogMinFreeRamMB = 1500,
    [int]    $TimeoutSec           = 3600,
    [string] $Tag        = "validate"
)

$ErrorActionPreference = "Stop"

$repo  = Split-Path -Parent $PSScriptRoot
# EXPERT_STREAM_MOE_TRACE / EXPERT_STREAM_MODEL override; author's layout is the
# last-resort default. See QUICKSTART.md. moe-trace is a separate binary from
# llama-cli, so it gets its own variable rather than reusing the cli one.
$exe   = [Environment]::GetEnvironmentVariable("EXPERT_STREAM_MOE_TRACE")
if ([string]::IsNullOrEmpty($exe))   { $exe   = "D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\build-cpu\bin\llama-moe-trace.exe" }
$model = [Environment]::GetEnvironmentVariable("EXPERT_STREAM_MODEL")
if ([string]::IsNullOrEmpty($model)) { $model = "E:\models\ds4f-iq1s\UD-IQ1_S\DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf" }

if (-not (Test-Path $exe))   { throw "moe-trace not built: $exe" }
if (-not (Test-Path $model)) { throw "model shard 1 not found: $model" }

$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir  = Join-Path $repo "bench\results"
New-Item -ItemType Directory -Force $outDir | Out-Null
$csv     = Join-Path $outDir ("moe_trace-{0}-{1}.csv" -f $Tag, $stamp)
$logFile = Join-Path $outDir ("moe_trace-{0}-{1}.log" -f $Tag, $stamp)
if ($PromptFile -ne "") {
    if (-not (Test-Path $PromptFile)) { throw "PromptFile not found: $PromptFile" }
    $promptFile = $PromptFile
    $ownsPromptFile = $false
    $Prompt = "(from file: {0})" -f $PromptFile
} else {
    $promptFile = Join-Path $env:TEMP ("moe_trace_prompt_{0}.txt" -f $stamp)
    [System.IO.File]::WriteAllText($promptFile, $Prompt, (New-Object System.Text.UTF8Encoding($false)))
    $ownsPromptFile = $true
}

# SAFETY: never --mlock, never --no-mmap, never --load-mode mlock. Any of those commits
# 82.5 GB of real memory on a 15.4 GB machine. mmap must stay on so the model's pages are
# file-backed and evictable - the system then degrades instead of dying.
$argList = @(
    "-m", $model,
    "-f", $promptFile,
    "-o", $csv,
    "-c", "$CtxSize",
    "-t", "$Threads",
    "-fit", "off"
)
# Both of the flags below are OPT-IN and default to OFF, because the only configuration
# that has ever completed successfully on this machine used neither of them:
#   ctx 512, no -ub, no --no-repack  -> 12 tokens, 83.1 s, 3096 rows, validated
# After adding them, ten consecutive 12-16 token captures were all killed by the guard even
# though each started with 7-10 GB free. Do not enable them without re-measuring.
if ($UBatch -gt 0) {
    # -b (logical batch) must be >= the whole prompt: we submit it in ONE llama_decode call
    # and llama.cpp asserts GGML_ASSERT(n_tokens_all <= cparams.n_batch).
    # -ub is what actually gets chunked, one graph eval per micro-batch, which is why
    # moe-trace must APPEND per layer across evals rather than overwrite.
    $argList += @("-ub", "$UBatch", "-b", "$CtxSize")
}
if ($NoRepack) {
    # The CPU backend repacks quantized weights into a SIMD-friendly layout, copying
    # file-backed pages into private dirty memory. Plausible memory culprit; measured to
    # NOT help, and suspected to hurt. Kept available only for further measurement.
    $argList += @("--no-repack")
}
# --no-repack is a MEMORY requirement here, not a speed choice.
# The CPU backend repacks quantized weights into a SIMD-friendly layout, which allocates a
# private buffer and copies into it. That converts read-only, file-backed, instantly
# evictable mmap pages into dirty private memory the OS must page out. Symptom: process RSS
# stayed at 40 MB while system available memory fell to ~400 MB and the pagefile grew to
# 3.7 GB. Repacking is a good trade when the model fits in RAM. Ours does not.
# -b (logical batch) must be >= the whole prompt, because we submit it in ONE llama_decode
# call; llama.cpp asserts GGML_ASSERT(n_tokens_all <= cparams.n_batch) otherwise.
# -ub (micro-batch) is what actually gets chunked: llama.cpp splits the batch into ubatches
# and runs one graph eval per ubatch. That is where the memory saving comes from, and it is
# why moe-trace must APPEND per layer across evals rather than overwrite.
# CHUNKED PREFILL IS MANDATORY ON THIS MACHINE, not an optimization.
# A single-shot prefill touches the union of every prompt token's experts at once. Measured:
#   12 tokens -> ~19 GB touched -> survives
#   19 tokens -> ~28 GB touched -> guard killed it at 394 MB free RAM
#   39 tokens -> ~46 GB touched -> guard killed it at 493 MB free RAM
# Windows caches faulted-in mmap pages faster than it retires them, so peak memory tracks
# the union rather than the working set (the process RSS stayed at 40 MB throughout).
# Small micro-batches spread the same reads over time and let the OS evict in between.
# -fit off is REQUIRED here. llama.cpp's auto-fit computes whether the model fits in device
# memory and aborts with:
#   "failed to fit params to free device memory: was unable to fit model into system
#    memory by reducing context, abort"
# because 82.5 GB obviously does not fit in 15.4 GB. That refusal is correct for a normal
# user and is exactly the "clean refusal" our runbook predicted as Phase 1's starting line.
# We override it deliberately: with mmap on, the model's pages are file-backed and
# evictable, so the OS streams them from the SSD instead of committing them. That is the
# entire premise of this project. It is slow, not fatal - and guard.ps1 is watching RAM.
# NOTE: do NOT pass --no-warmup. It is not a valid option for this tool's argument parser
# (LLAMA_EXAMPLE_COMMON), and it is unnecessary anyway - moe-trace.cpp already sets
# params.warmup = false before creating the context.

$forbidden = @("--mlock", "--no-mmap")
foreach ($a in $argList) {
    if ($forbidden -contains $a) { throw "REFUSING TO RUN: forbidden flag $a" }
}

Write-Host ""
Write-Host "moe-trace validation run"
Write-Host ("  exe    : {0}" -f $exe)
Write-Host ("  model  : {0}" -f $model)
Write-Host ("  prompt : {0}" -f $Prompt)
Write-Host ("  ctx    : {0}, threads {1}" -f $CtxSize, $Threads)
Write-Host ("  csv    : {0}" -f $csv)
Write-Host ("  log    : {0}" -f $logFile)
Write-Host ""
Write-Host "NOTE: a cold prefill must stream expert weights off the SSD. Expect this to take"
Write-Host "      a while and to page heavily. guard.ps1 will kill it if RAM gets dangerous."
Write-Host ""

# DO NOT use `& $exe ... 2>&1 | Tee-Object` here. In Windows PowerShell 5.1, redirecting a
# NATIVE executable's stderr inside PowerShell wraps every stderr line in a NativeCommandError
# ErrorRecord. llama.cpp logs everything to stderr, so that turns a normal run into a wall of
# fake "errors" and loses the actual message. Start-Process with real file redirection keeps
# the two streams intact and readable.
$errFile = $logFile -replace '\.log$', '.err.log'

# WHY THIS SCRIPT NOW APPLIES A WORKING-SET CAP
# Ten consecutive capture attempts at 19-39 tokens were killed for running the
# machine out of memory, and that was diagnosed at the time as "the prefill union
# is too big". It was not. The pages were sitting in the process WORKING SET,
# where Windows cannot reclaim them, instead of on the standby list where it can.
# A hard working-set cap forces continuous trimming and turns the same run from
# "dies at 19 tokens" into "runs to completion". See docs/MEASURED_GROUND_TRUTH.md
# section 9. Without this, long traces are simply not capturable on this machine.
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

$sw = [System.Diagnostics.Stopwatch]::StartNew()
# NOT -Wait: the handle is needed while the process is alive in order to cap it.
$proc = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -PassThru `
    -RedirectStandardOutput $logFile -RedirectStandardError $errFile
try { $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch { }

if ($WorkingSetCapMB -gt 0) {
    # Cap exactly one PID - the one just launched - and re-verify what it is first.
    # This never enumerates or touches any other process on the machine.
    if ($proc.ProcessName -ne "llama-moe-trace") {
        throw ("REFUSING to cap PID {0}: expected llama-moe-trace, found '{1}'." -f $proc.Id, $proc.ProcessName)
    }
    if ($WorkingSetCapMB -lt 3500) {
        throw "WorkingSetCapMB below 3500 would thrash the always-needed non-expert weights."
    }
    $totalMb    = [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1KB, 0)
    $overheadMb = 4040
    $predictedFreeMb = $totalMb - $WorkingSetCapMB - $overheadMb
    if ($predictedFreeMb -lt ($WatchdogMinFreeRamMB + 1000)) {
        throw ("WorkingSetCapMB {0} would leave about {1} MB free on a {2} MB machine, at or below the {3} MB watchdog floor. Use {4} or less." -f `
            $WorkingSetCapMB, $predictedFreeMb, $totalMb, $WatchdogMinFreeRamMB, ($totalMb - $overheadMb - $WatchdogMinFreeRamMB - 1000))
    }
    try {
        [WsCap]::Apply($proc.Handle, 200MB, ([long]$WorkingSetCapMB * 1MB))
        Write-Host ("HARD working-set cap {0} MB applied to PID {1}." -f $WorkingSetCapMB, $proc.Id) -ForegroundColor Green
    } catch {
        Write-Host ("WARNING: could not apply working-set cap ({0}). Continuing UNCAPPED." -f $_.Exception.Message) -ForegroundColor Yellow
    }
}

# Watchdog. Kills ONLY the process this script started, never anything else.
# A kill means no CSV at all: moe-trace writes its output in one ofstream pass
# after the prefill completes. So the timeout is deliberately generous - losing
# 40 minutes of capture to an impatient limit is worse than waiting.
$strikes = 0; $killed = $false; $minFreeMb = [double]::MaxValue
while (-not $proc.HasExited) {
    Start-Sleep -Milliseconds 1500
    $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB, 1)
    if ($free -lt $minFreeMb) { $minFreeMb = $free }
    if ($free -lt $WatchdogMinFreeRamMB) { $strikes++ } else { $strikes = 0 }
    if ($strikes -ge 3) {
        Write-Host ("WATCHDOG: free RAM {0} MB below floor {1} MB three times - killing PID {2}." -f `
            $free, $WatchdogMinFreeRamMB, $proc.Id) -ForegroundColor Red
        try { $proc.Kill() } catch { }
        $killed = $true; break
    }
    if ($sw.Elapsed.TotalSeconds -gt $TimeoutSec) {
        Write-Host ("TIMEOUT after {0:N0} s - killing PID {1}." -f $sw.Elapsed.TotalSeconds, $proc.Id) -ForegroundColor Red
        try { $proc.Kill() } catch { }
        $killed = $true; break
    }
}
try { $proc.WaitForExit() } catch { }
$code = $proc.ExitCode
$sw.Stop()
if ($minFreeMb -ne [double]::MaxValue) {
    Write-Host ("min free RAM during run: {0} MB" -f $minFreeMb)
}

# llama.cpp writes its normal logging to stderr, so show that, not stdout.
if ((Test-Path $errFile) -and ((Get-Item $errFile).Length -gt 0)) {
    Write-Host "--- last 25 lines of stderr ---"
    Get-Content $errFile -Tail 25 | ForEach-Object { Write-Host ("  " + $_) }
}

Write-Host ""
Write-Host ("exit code    : {0}" -f $code)
Write-Host ("elapsed      : {0:N1} s" -f $sw.Elapsed.TotalSeconds)

if (Test-Path $csv) {
    $lines = (Get-Content $csv | Measure-Object -Line).Lines
    Write-Host ("csv rows     : {0} (including header)" -f $lines)
    Write-Host ""
    Write-Host "first 8 data rows:"
    Get-Content $csv -TotalCount 9 | Select-Object -Skip 1 | ForEach-Object { Write-Host ("  " + $_) }
} else {
    Write-Host "NO CSV PRODUCED - the callback did not fire or the run failed."
}

# SAFETY: nothing in this repo deletes files. The generated prompt file is left in place
# and its path printed; it is a few hundred bytes in %TEMP% and Windows cleans that up.
if ($ownsPromptFile) { Write-Host ("generated prompt file kept at: {0}" -f $promptFile) }
exit $code
