# chat.ps1 - actually talk to the 284B model, interactively, on this laptop.
#
# RUN IT FROM YOUR OWN TERMINAL:
#     .\chat.ps1
# then type. /exit or Ctrl+C to quit.
#
# WHAT TO EXPECT, HONESTLY
# Generation is about 0.3 tok/s, i.e. roughly 3 seconds PER TOKEN. This is a
# reasoning model - it emits a "[Start thinking]" chain before its answer - so a
# single reply of a few hundred tokens takes tens of minutes. It is a real
# conversation with a 284-billion-parameter model on a 16 GB laptop; it is not a
# chatbot you can hold a back-and-forth with at speed.
#
#     20 tokens   ~1 minute
#    100 tokens   ~5 minutes
#    500 tokens   ~28 minutes
#
# -MaxTokensPerReply caps each reply so a runaway thinking chain cannot eat an
# hour. Raise it when you want a full answer and are willing to wait.
#
# IT WILL NOT OVERLOAD THE MACHINE. The same three protections every measurement
# run in this repo uses are applied here:
#   - a hard working-set cap, so the model can never eat all of RAM
#   - a free-RAM watchdog that stops the run before the machine starts thrashing
#   - mmap stays on and --mlock is refused, so weights are never committed
#
# ONE NICE PROPERTY OF CHATTING RATHER THAN ONE-SHOTTING: the process stays
# alive between turns, so the expert cache stays warm. Later turns in a
# conversation hit the cache more often than the first one and run faster.
#
# PowerShell 5.1. Pure ASCII. No && || ternary ??

param(
    [int]    $CtxSize             = 4096,
    [int]    $MaxTokensPerReply   = 300,
    [int]    $Threads             = 8,
    [int]    $WorkingSetCapMB     = 9000,
    [int]    $ExpertArenaMB       = 2600,
    [int]    $WatchdogMinFreeRamMB = 1500,
    [switch] $NoStreaming,
    [string] $SystemPrompt        = "",
    [string] $LlamaCli            = "",
    [string] $ModelPath           = ""
)

$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot

# Paths resolve most-specific-first: the -LlamaCli / -ModelPath parameters, then
# the environment variables QUICKSTART.md tells you to set, then the author's own
# layout as a last resort. Nothing is searched for or guessed. If none of the
# three exists you get a message naming both ways to fix it, not a silent
# fallback to stock behaviour that then looks like a performance result.
$exe = $LlamaCli
if ([string]::IsNullOrEmpty($exe)) { $exe = [Environment]::GetEnvironmentVariable("EXPERT_STREAM_LLAMA_CLI") }
if ([string]::IsNullOrEmpty($exe)) { $exe = "D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\build-cpu\bin\llama-cli.exe" }

$model = $ModelPath
if ([string]::IsNullOrEmpty($model)) { $model = [Environment]::GetEnvironmentVariable("EXPERT_STREAM_MODEL") }
if ([string]::IsNullOrEmpty($model)) { $model = "E:\models\ds4f-iq1s\UD-IQ1_S\DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf" }

$manifest = Join-Path $repo "bench\results\expert_manifest.csv"

if (-not (Test-Path -LiteralPath $exe)) {
    throw ("llama-cli not found at '{0}'. Build it (QUICKSTART.md step 2), then set the EXPERT_STREAM_LLAMA_CLI environment variable or pass -LlamaCli." -f $exe)
}
if (-not (Test-Path -LiteralPath $model)) {
    throw ("model shard 1 not found at '{0}'. Download it (QUICKSTART.md step 1), then set the EXPERT_STREAM_MODEL environment variable or pass -ModelPath." -f $model)
}

# KV cache is small on this architecture and is NOT the thing that limits context
# here. head_count_kv = 1 with key/value length 512, so roughly:
#   43 layers * (512 + 512) * 2 bytes = ~86 KB per token
# 4096 tokens is therefore ~350 MB. Time is the constraint, not memory.
$kvMb = [math]::Round(43 * 1024 * 2 * $CtxSize / 1MB, 0)

if ($ExpertArenaMB -gt 0 -and $ExpertArenaMB -lt 1800) {
    # One token's expert working set is 1.649 GiB, walked cyclically. A cache
    # smaller than one lap evicts every entry just before its turn comes round
    # again and measures 0.0% hits - the memory is spent for nothing.
    Write-Host ("NOTE: arena {0} MB is under one token's 1.649 GiB working set - expect ~0% cache hits." -f $ExpertArenaMB) -ForegroundColor Yellow
}

# Refuse a configuration that would leave the machine with no headroom, rather
# than discovering it by thrashing. Same arithmetic the measurement runner uses.
$totalMb = [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1KB, 0)
$overheadMb = 4040
$predictedFreeMb = $totalMb - $WorkingSetCapMB - $overheadMb
if ($predictedFreeMb -lt ($WatchdogMinFreeRamMB + 1000)) {
    throw ("WorkingSetCapMB {0} would leave about {1} MB free on a {2} MB machine, at or below the {3} MB watchdog floor. Use {4} or less." -f `
        $WorkingSetCapMB, $predictedFreeMb, $totalMb, $WatchdogMinFreeRamMB, ($totalMb - $overheadMb - $WatchdogMinFreeRamMB - 1000))
}

# Expert streaming. Set explicitly either way so the run's behaviour never
# depends on what happened to be in the environment already.
if ($NoStreaming) {
    Remove-Item Env:\V4F_EXPERT_MANIFEST -ErrorAction SilentlyContinue
    Remove-Item Env:\V4F_EXPERT_ARENA_MB -ErrorAction SilentlyContinue
} else {
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw "expert manifest missing: $manifest  (run: python tools\make_expert_manifest.py)"
    }
    $env:V4F_EXPERT_MANIFEST = $manifest
    $env:V4F_EXPERT_ARENA_MB = "$ExpertArenaMB"
}

$argList = @(
    "-m", $model,
    "-c", "$CtxSize",
    "-n", "$MaxTokensPerReply",
    "-t", "$Threads",
    "-fit", "off",          # 82.5 GB obviously does not "fit"; we stream it deliberately
    "--no-warmup",
    "--no-repack"           # repack turns evictable file-backed pages into dirty private memory
)
if ($SystemPrompt -ne "") { $argList += @("-sys", $SystemPrompt) }
# NOTE: deliberately NO -st and NO -no-cnv here. Those force single-turn mode.
# This script is the one place in the repo that WANTS conversation mode.

# Hard-refuse the two flags that would commit 82.5 GB of real memory.
foreach ($a in $argList) {
    if ($a -eq "--mlock" -or $a -eq "--no-mmap") { throw "REFUSING TO RUN: forbidden flag $a" }
}

Write-Host ""
Write-Host "=== chat with DeepSeek V4-Flash 284B, IQ1_S, 82.5 GB, on 15.4 GB of RAM ===" -ForegroundColor Cyan
Write-Host ("  context        : {0} tokens (KV cache about {1} MB)" -f $CtxSize, $kvMb)
Write-Host ("  max per reply  : {0} tokens" -f $MaxTokensPerReply)
Write-Host ("  working-set cap: {0} MB   (predicted free during run: ~{1} MB)" -f $WorkingSetCapMB, $predictedFreeMb)
if ($NoStreaming) {
    Write-Host  "  expert reader  : stock mmap (streaming disabled)"
} else {
    Write-Host ("  expert reader  : streaming + {0} MB cache" -f $ExpertArenaMB) -ForegroundColor Green
}
Write-Host ""
Write-Host "  SPEED: about 3 seconds per token. A 300-token reply takes ~15 minutes." -ForegroundColor Yellow
Write-Host "  The first reply is the slowest - the expert cache is cold. Later turns are faster."
Write-Host ""
Write-Host "  Loading the model takes a minute or so before the prompt appears."
Write-Host "  Type /exit to quit. Ctrl+C also works."
Write-Host ""

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

# -NoNewWindow and NO redirection: the child must inherit this console's stdin,
# or you cannot type at it. That is the whole point of this script, and it is why
# the measurement runners (which redirect) cannot be reused here.
$proc = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -PassThru

# Cap exactly the PID we just started, after verifying what it is. Never
# enumerates or touches any other process.
if ($WorkingSetCapMB -gt 0) {
    Start-Sleep -Milliseconds 300
    if ($proc.HasExited) { throw "llama-cli exited immediately - check the error above." }
    if ($proc.ProcessName -ne "llama-cli") {
        throw ("REFUSING to cap PID {0}: expected llama-cli, found '{1}'." -f $proc.Id, $proc.ProcessName)
    }
    try {
        [WsCap]::Apply($proc.Handle, 200MB, ([long]$WorkingSetCapMB * 1MB))
    } catch {
        Write-Host ("WARNING: could not apply working-set cap ({0}). Continuing UNCAPPED." -f $_.Exception.Message) -ForegroundColor Yellow
    }
}

# Free-RAM watchdog. Kills ONLY the process this script started. Three
# consecutive samples below the floor, so a momentary dip does not end a chat you
# have been waiting twenty minutes for.
$strikes = 0
while (-not $proc.HasExited) {
    Start-Sleep -Seconds 2
    $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB, 0)
    if ($free -lt $WatchdogMinFreeRamMB) { $strikes++ } else { $strikes = 0 }
    if ($strikes -ge 3) {
        Write-Host ""
        Write-Host ("WATCHDOG: free RAM {0} MB below the {1} MB floor three times - stopping the model to keep the machine usable." -f $free, $WatchdogMinFreeRamMB) -ForegroundColor Red
        try { $proc.Kill() } catch { }
        break
    }
}

Write-Host ""
Write-Host "chat ended." -ForegroundColor Cyan
