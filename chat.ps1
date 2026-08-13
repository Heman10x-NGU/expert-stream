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
    # 2400, not 2600. At the 4096 default context the KV cache is 344 MB, and
    # 2600 + 344 + compute leaves attention 158 MB short of staying resident -
    # so it gets re-faulted off the same disk the experts stream from, every
    # token. 2400 is still 1.42 laps, comfortably above the 1.05 cliff.
    # The measurement runs pass their own value and are unaffected.
    [int]    $ExpertArenaMB       = 2400,
    [int]    $WatchdogMinFreeRamMB = 1500,
    [switch] $NoStreaming,
    [string] $SystemPrompt        = "",
    # Run every check and print the plan, then stop without launching anything.
    # The memory arithmetic below is the part most likely to be wrong, and it is
    # not testable if the only way to exercise it is to load 82.5 GB.
    [switch] $DryRun,
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

# KV cache size, from the GGUF headers rather than an estimate: block_count 43,
# head_count_kv 1, key_length 512, value_length 512, F16.
#   43 * 1 * (512 + 512) * 2 bytes = 88,064 B = 86.0 KB per token
# Verified against bench/results/gguf_meta.txt.
#
# NOTE: the sparse-attention indexer (head_count 64, key_length 128) is NOT in
# this figure and has never been measured. If real memory use exceeds the
# prediction below, that is the first place to look.
$kvMb = [math]::Round(43 * 1024 * 2 * $CtxSize / 1MB, 0)

if ($ExpertArenaMB -gt 0 -and $ExpertArenaMB -lt 1800) {
    # One token's expert working set is 1.649 GiB, walked cyclically. A cache
    # smaller than one lap evicts every entry just before its turn comes round
    # again and measures 0.0% hits - the memory is spent for nothing.
    Write-Host ("NOTE: arena {0} MB is under one token's 1.649 GiB working set - expect ~0% cache hits." -f $ExpertArenaMB) -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# CONTEXT-vs-CACHE BUDGET. Added 2026-08-13 after measuring that this machine
# has a hard context ceiling nobody had written down.
#
# The KV cache and the expert arena come out of the SAME working-set cap. KV
# costs 86 KB/token, which looks trivial, and it is - right up to the point
# where it squeezes the arena below one lap (1,649 MiB), at which the cache
# stops returning ANY hits at all. That is not a gentle degradation: measured,
# it is 56.6% -> 0.0%, i.e. roughly a 40% speed loss for the last few thousand
# tokens of context.
#
# The cap does NOT have to hold everything - that is the whole point of it. The
# 75.55 GB expert bank is file-backed and meant to be trimmed continuously. What
# matters is the split:
#
#   PRIVATE memory  - the arena, the KV cache, compute buffers. Cannot be
#                     trimmed. Comes straight off the cap.
#   FILE-BACKED     - the attention weights. Trimmable, but they are needed on
#                     EVERY token, so trimming them means re-faulting 4.75 GiB
#                     per token off the same disk the experts stream from.
#
# So the real constraint is: after private memory, is there still room under the
# cap for attention to stay resident?
#
#     cap - (arena + KV + compute)  >=  attention
#
# Calibration, both measured and both recorded in MEASURED_GROUND_TRUTH:
#   attention weights          4.75 GiB = 4,864 MB   (section 2, resident set)
#   compute buffers            ~1,350 MB             (1,400 MB private measured
#                                                     with --no-repack at ctx
#                                                     512, minus that 43 MB KV)
$ATTN_MB    = 4864
$COMPUTE_MB = 1350
$ONE_LAP_MB = 1689          # 1.649 GiB, the size below which hits are 0.0%

$privateMb  = $ExpertArenaMB + $kvMb + $COMPUTE_MB
$headroomMb = $WorkingSetCapMB - $privateMb - $ATTN_MB

if ($ExpertArenaMB -gt 0) {
    # Hard refusal: private memory alone does not fit under the cap. Nothing
    # can rescue this one, so stop before allocating.
    if ($privateMb -ge $WorkingSetCapMB) {
        throw ("arena {0} MB + KV {1} MB + compute {2} MB = {3} MB, which does not fit under the {4} MB cap at all. Lower -CtxSize or -ExpertArenaMB." -f `
            $ExpertArenaMB, $kvMb, $COMPUTE_MB, $privateMb, $WorkingSetCapMB)
    }
    # Soft warning: it fits, but attention gets squeezed out and will be
    # re-faulted every token. Warn rather than refuse - it still runs, and the
    # calibration constants above are measurements of one machine, not laws.
    if ($headroomMb -lt 0) {
        $maxCtx = [math]::Floor(($WorkingSetCapMB - $ATTN_MB - $COMPUTE_MB - $ExpertArenaMB) * 1MB / (43 * 1024 * 2))
        Write-Host ("WARNING: attention is {0} MB short of staying resident (cap {1} - private {2} - attention {3} = {4}). Attention is needed EVERY token, so it will be re-faulted off the same disk the experts stream from." -f `
            [math]::Abs($headroomMb), $WorkingSetCapMB, $privateMb, $ATTN_MB, $headroomMb) -ForegroundColor Yellow
        Write-Host ("         Fixes, cheapest first: -CtxSize {0} (from {1}), or -ExpertArenaMB {2} (from {3}, still {4:N2} laps)." -f `
            [math]::Max(512, $maxCtx), $CtxSize, ($ExpertArenaMB + $headroomMb), $ExpertArenaMB, (($ExpertArenaMB + $headroomMb) / $ONE_LAP_MB)) -ForegroundColor Yellow
    }
    # And the older trap, unchanged: an arena below one lap returns 0.0%.
    if ($ExpertArenaMB -lt ($ONE_LAP_MB * 1.05)) {
        Write-Host ("WARNING: arena {0} MB is {1:N2} laps. Below 1.05 the cache measures ~0% hits." -f `
            $ExpertArenaMB, ($ExpertArenaMB / $ONE_LAP_MB)) -ForegroundColor Yellow
    }
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
    Write-Host ("  expert reader  : streaming + {0} MB cache ({1:N2} laps)" -f $ExpertArenaMB, ($ExpertArenaMB / $ONE_LAP_MB)) -ForegroundColor Green
    Write-Host ("  memory budget  : cap {0} = private {1} (arena {2} + KV {3} + compute {4}) + attention {5}, headroom {6} MB" -f `
        $WorkingSetCapMB, $privateMb, $ExpertArenaMB, $kvMb, $COMPUTE_MB, $ATTN_MB, $headroomMb)
}
Write-Host ""

if ($DryRun) {
    Write-Host "DRY RUN - every check passed, nothing was launched." -ForegroundColor Cyan
    Write-Host ""
    return
}
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
