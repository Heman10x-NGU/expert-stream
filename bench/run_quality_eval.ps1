# run_quality_eval.ps1 - the measurement this project has been avoiding.
#
# Everything verified so far proves the READER is correct: our streamed output is
# byte-identical to stock llama.cpp. That says nothing about whether the MODEL is
# any good at 1.5625 bits per weight, and quality is known to collapse somewhere
# below about 3 bits. If the answers are nonsense then every speed number in this
# repo is optimising something nobody should use.
#
# This runs bench\prompts\quality\*.txt one question at a time and writes each
# answer verbatim. Grading is by hand against ANSWER_KEY.md, which fixes the
# scoring bands BEFORE the results exist so they cannot be rationalised after.
#
# THIS TAKES HOURS. Roughly 3 s per generated word, plus about a minute of model
# load per question because each question is its own process. Budget a night for
# the full set. That is why it is resumable - see below.
#
# SAFETY, same rules as every other script here:
#   - never --mlock, never --no-mmap (hard-refused): either commits 82.5 GB of
#     real memory on a 15.4 GB machine.
#   - hard working-set cap, plus the free-RAM watchdog with the 3-consecutive-
#     samples rule so one dip does not kill an hour of work.
#   - kills ONLY the process it started, after re-checking the PID is llama-cli.
#   - DELETES NOTHING. Existing answers are skipped, never overwritten. -Force
#     writes a NEW timestamped file beside the old one rather than replacing it.
#
# PowerShell 5.1. Pure ASCII. No && || ternary ??
#
# Usage:
#   .\bench\run_quality_eval.ps1                      # full set, resumable
#   .\bench\run_quality_eval.ps1 -Only q01,q07        # just those
#   .\bench\run_quality_eval.ps1 -Reasoning on -Only q01,q02,q06,q07
#   .\bench\run_quality_eval.ps1 -DryRun              # print the plan, run nothing

[CmdletBinding()]
param(
    [int]    $NPredict = 200,
    [int]    $CtxSize  = 1024,
    [int]    $Threads  = 8,
    [int]    $Seed     = 42,

    # Reasoning OFF by default and this is a deliberate trade, not a default that
    # nobody thought about. This model thinks out loud before answering, which at
    # ~3 s/word turns 15 questions into most of a day. Off, the set finishes in a
    # night. The cost is that these scores are a LOWER BOUND - the model is being
    # asked to answer without the working-out it was trained to do. ANSWER_KEY.md
    # says to re-run the arithmetic and reasoning questions with 'on' before
    # concluding anything about those two rows.
    [ValidateSet("off", "on", "auto")]
    [string] $Reasoning = "off",

    [int]    $WorkingSetCapMB       = 9000,
    [int]    $WatchdogMinFreeRamMB  = 1500,
    # Per QUESTION, not for the whole set. 200 words at ~3 s is ~10 min, plus a
    # minute of load; 1800 s leaves room for a slow question without letting one
    # bad question eat the night.
    [int]    $TimeoutSec            = 1800,

    [string] $ExpertManifest = "",
    [int]    $ExpertArenaMB  = 2600,

    [string[]] $Only    = @(),
    [switch]   $Force,
    [switch]   $DryRun,
    [string]   $Tag     = "qual"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo  = Split-Path -Parent $PSScriptRoot
# EXPERT_STREAM_LLAMA_CLI / EXPERT_STREAM_MODEL override; author's layout is the
# last-resort default. See QUICKSTART.md. GetEnvironmentVariable rather than
# $env: because this file runs under Set-StrictMode.
$exe   = [Environment]::GetEnvironmentVariable("EXPERT_STREAM_LLAMA_CLI")
if ([string]::IsNullOrEmpty($exe))   { $exe   = "D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\build-cpu\bin\llama-cli.exe" }
$model = [Environment]::GetEnvironmentVariable("EXPERT_STREAM_MODEL")
if ([string]::IsNullOrEmpty($model)) { $model = "E:\models\ds4f-iq1s\UD-IQ1_S\DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf" }
$qDir  = Join-Path $repo "bench\prompts\quality"
$outDir = Join-Path $repo "bench\results\quality"

# --- every precondition checked BEFORE the first hour is spent ---------------------------

if (-not (Test-Path -LiteralPath $exe))   { throw "llama-cli not built at: $exe" }
if (-not (Test-Path -LiteralPath $model)) { throw "model shard 1 not found: $model" }
if (-not (Test-Path -LiteralPath $qDir))  { throw "question directory not found: $qDir" }

# Shards 2 and 3 are auto-discovered from shard 1's name. Missing ones fail deep in
# the loader with a confusing message, so check here where it is clear.
$modelDir = Split-Path -Parent $model
$leaf1    = Split-Path -Leaf $model
foreach ($n in @("00002-of-00003", "00003-of-00003")) {
    $shard = Join-Path $modelDir ($leaf1 -replace "00001-of-00003", $n)
    if (-not (Test-Path -LiteralPath $shard)) { throw "missing shard: $shard" }
}

if ($NPredict -lt 16) { throw "-NPredict must be >= 16; anything less cannot hold an answer." }
if ($CtxSize  -lt 256) { throw "-CtxSize must be >= 256." }
if ($ExpertArenaMB -ne 0 -and ($ExpertArenaMB -lt 32 -or $ExpertArenaMB -gt 6144)) {
    throw "ExpertArenaMB must be 0 or 32..6144 (must match the bounds in expert-stream.c)."
}
# One token's expert working set is 1.649 GiB walked cyclically. An arena under one
# lap evicts every entry just before its turn - measured 0.0% hits. Not fatal here,
# just wasted memory, so warn rather than refuse.
if ($ExpertArenaMB -gt 32 -and $ExpertArenaMB -lt 1800) {
    Write-Host ("NOTE: arena {0} MB is under one token's 1.649 GiB lap - expect ~0% hits." -f $ExpertArenaMB) -ForegroundColor Yellow
}

$cores = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
if ($Threads -gt $cores) {
    Write-Host ("NOTE: -Threads {0} exceeds {1} logical cores; clamping." -f $Threads, $cores) -ForegroundColor Yellow
    $Threads = $cores
}

$totalMb = [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1KB, 0)
if ($WorkingSetCapMB -lt 3500) {
    throw "WorkingSetCapMB below 3500 would thrash weights that every token needs."
}
# Measured on this machine: everything-but-llama costs about 4,040 MB. A cap that
# leaves less than the watchdog floor plus a margin will kill itself on the first
# question, after paying the model-load time. Refuse it now, not in an hour.
$overheadMb = 4040
$predictedFreeMb = $totalMb - $WorkingSetCapMB - $overheadMb
if ($predictedFreeMb -lt ($WatchdogMinFreeRamMB + 1000)) {
    throw ("WorkingSetCapMB {0} would leave about {1} MB free on a {2} MB machine, at or below the {3} MB watchdog floor. Use {4} or less." -f `
        $WorkingSetCapMB, $predictedFreeMb, $totalMb, $WatchdogMinFreeRamMB, ($totalMb - $overheadMb - $WatchdogMinFreeRamMB - 1000))
}

$freeRamMb = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB, 0)
if ($freeRamMb -lt 3000) {
    throw ("Only {0} MB free RAM. Close something before starting a run this long." -f $freeRamMb)
}

if ($ExpertManifest -ne "") {
    if (-not (Test-Path -LiteralPath $ExpertManifest)) { throw "ExpertManifest not found: $ExpertManifest" }
}

# --- select questions --------------------------------------------------------------------

$questions = @(Get-ChildItem -LiteralPath $qDir -Filter "q*.txt" | Sort-Object Name)
if ($questions.Count -eq 0) { throw "no q*.txt files in $qDir" }

if ($Only.Count -gt 0) {
    $sel = @()
    foreach ($pat in $Only) {
        $hit = @($questions | Where-Object { $_.BaseName -like ($pat + "*") })
        if ($hit.Count -eq 0) {
            throw ("-Only '{0}' matched nothing. Available: {1}" -f $pat, (($questions | ForEach-Object { $_.BaseName }) -join ', '))
        }
        $sel += $hit
    }
    $questions = @($sel | Sort-Object Name -Unique)
}

New-Item -ItemType Directory -Force $outDir | Out-Null

# RESUMABILITY. At hours per full set this run WILL be interrupted - a watchdog
# kill, a reboot, a closed lid. Answers already on disk are skipped so the set can
# be restarted as many times as it takes without losing or repeating work.
$todo = @()
foreach ($q in $questions) {
    $ans = Join-Path $outDir ($q.BaseName + ".answer.txt")
    $done = (Test-Path -LiteralPath $ans) -and ((Get-Item -LiteralPath $ans).Length -gt 0)
    if ($done -and -not $Force) {
        Write-Host ("skip  {0}  (already answered)" -f $q.BaseName) -ForegroundColor DarkGray
        continue
    }
    $todo += $q
}

if ($todo.Count -eq 0) {
    Write-Host ""
    Write-Host "Every selected question already has an answer. Nothing to do." -ForegroundColor Green
    Write-Host ("Grade them against {0}\ANSWER_KEY.md" -f $qDir)
    Write-Host "Use -Force to re-ask (it writes a new timestamped file, it does not overwrite)."
    exit 0
}

$estMin = [math]::Round($todo.Count * (($NPredict * 3.0) + 60) / 60.0, 0)

Write-Host ""
Write-Host "=== IQ1_S QUALITY EVALUATION ===" -ForegroundColor Cyan
Write-Host ("  questions   : {0} to ask, {1} selected" -f $todo.Count, $questions.Count)
Write-Host ("  n_predict   : {0}   ctx {1}   threads {2}   temp 0 (greedy)" -f $NPredict, $CtxSize, $Threads)
Write-Host ("  reasoning   : {0}" -f $Reasoning)
Write-Host ("  ws cap      : {0} MB   predicted free ~{1} MB" -f $WorkingSetCapMB, $predictedFreeMb)
Write-Host ("  arena       : {0} MB" -f $ExpertArenaMB)
Write-Host ("  answers to  : {0}" -f $outDir)
Write-Host ""
Write-Host ("  WORST-CASE {0} MINUTES. It is resumable - rerun the same command after" -f $estMin) -ForegroundColor Yellow
Write-Host "  an interruption and it picks up where it stopped." -ForegroundColor Yellow
Write-Host ""

if ($DryRun) {
    Write-Host "DRY RUN. Would ask:" -ForegroundColor Cyan
    foreach ($q in $todo) {
        Write-Host ("  {0,-28} {1}" -f $q.BaseName, ([System.IO.File]::ReadAllText($q.FullName) -replace "`r?`n", " / "))
    }
    exit 0
}

# --- environment for our fork ------------------------------------------------------------
# Always set or clear explicitly. Inheriting whatever the parent shell happened to
# have would make a run's configuration depend on invisible state, and the point of
# a results file is that a row describes the run that produced it.
if ($ExpertManifest -ne "") {
    $env:V4F_EXPERT_MANIFEST = $ExpertManifest
    Write-Host ("expert streaming ENABLED via {0}" -f $ExpertManifest) -ForegroundColor Green
} else {
    Remove-Item Env:\V4F_EXPERT_MANIFEST -ErrorAction SilentlyContinue
}
if ($ExpertArenaMB -gt 0) {
    $env:V4F_EXPERT_ARENA_MB = "$ExpertArenaMB"
} else {
    Remove-Item Env:\V4F_EXPERT_ARENA_MB -ErrorAction SilentlyContinue
}

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

$summaryCsv = Join-Path $outDir "quality_runs.csv"
$stopFile   = Join-Path $repo "STOP"

$asked = 0
foreach ($q in $todo) {

    # Same stop-file convention as the other runners: create STOP in the repo root
    # to end a long run cleanly between questions, without hunting for a PID.
    if (Test-Path -LiteralPath $stopFile) {
        Write-Host ""
        Write-Host ("STOP file present at {0} - stopping between questions." -f $stopFile) -ForegroundColor Yellow
        Write-Host "Delete it yourself when you want to resume. This script does not remove it."
        break
    }

    $stamp    = Get-Date -Format "yyyyMMdd-HHmmss"
    $qText    = [System.IO.File]::ReadAllText($q.FullName)
    $ansFile  = Join-Path $outDir ($q.BaseName + ".answer.txt")
    if ((Test-Path -LiteralPath $ansFile) -and $Force) {
        # NEVER overwrite. A previous answer is data; a re-ask is a second data point.
        $ansFile = Join-Path $outDir ("{0}.answer.{1}.txt" -f $q.BaseName, $stamp)
    }
    $rawFile  = Join-Path $outDir ("{0}.raw.{1}.log" -f $q.BaseName, $stamp)

    Write-Host ""
    Write-Host ("--- {0} ---" -f $q.BaseName) -ForegroundColor Cyan
    Write-Host ("  Q: {0}" -f ($qText -replace "`r?`n", " / "))

    # -f rather than -p ON PURPOSE. Start-Process on PowerShell 5.1 joins ArgumentList
    # with spaces and does not quote; a question containing spaces, quotes or newlines
    # would be split into positional arguments and silently ask something else. A file
    # path with no spaces sidesteps the Win32 tokenizer completely.
    #
    # Conversation mode is left ON (no -no-cnv) so the GGUF's chat template is applied
    # and the model answers as an assistant instead of continuing the text. -st makes
    # it one turn and exit; without it llama-cli blocks on stdin forever after
    # answering. Both were measured the hard way.
    $argList = @(
        "-m", $model,
        "-f", $q.FullName,
        "-c", "$CtxSize",
        "-n", "$NPredict",
        "-t", "$Threads",
        "-s", "$Seed",
        "--temp", "0",
        "-st",
        "-rea", $Reasoning,
        "-fit", "off",
        "--no-warmup",
        # REQUIRED, not an optimisation. Measured A/B: repack on -> 3,664 MB private,
        # killed at 20 s with no output; --no-repack -> 1,400 MB, completed.
        "--no-repack"
    )

    # Hard refusal, checked against the FINAL list so it catches anything added above.
    foreach ($a in $argList) {
        if ($a -imatch '^--?(mlock|no-mmap)$') { throw "REFUSING TO RUN: forbidden flag $a" }
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -PassThru `
                          -RedirectStandardOutput $rawFile
    try { $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch { }

    if ($proc.ProcessName -ne "llama-cli") {
        try { $proc.Kill() } catch { }
        throw ("REFUSING to cap PID {0}: expected llama-cli, found '{1}'." -f $proc.Id, $proc.ProcessName)
    }
    try {
        [WsCap]::Apply($proc.Handle, 200MB, ([long]$WorkingSetCapMB * 1MB))
        Write-Host ("  cap {0} MB on PID {1}" -f $WorkingSetCapMB, $proc.Id) -ForegroundColor DarkGray
    } catch {
        Write-Host ("  WARNING: working-set cap failed: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
        Write-Host "           continuing UNCAPPED - this question may not finish." -ForegroundColor Yellow
    }

    $strikes   = 0
    $killed    = $false
    $killWhy   = ""
    $minFreeMb = [double]::MaxValue
    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds 1500
        $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB, 1)
        if ($free -lt $minFreeMb) { $minFreeMb = $free }

        # Three consecutive samples, not one. A single dip is noise, and killing on it
        # would throw away a question that has already cost ten minutes.
        if ($free -lt $WatchdogMinFreeRamMB) { $strikes++ } else { $strikes = 0 }
        if ($strikes -ge 3) {
            Write-Host ("  WATCHDOG: free RAM {0} MB below {1} for 3 samples - killing." -f $free, $WatchdogMinFreeRamMB) -ForegroundColor Red
            try { $proc.Kill() } catch { }
            $killed = $true; $killWhy = "ram"
            break
        }
        if ($sw.Elapsed.TotalSeconds -gt $TimeoutSec) {
            Write-Host ("  WATCHDOG: past {0} s - killing." -f $TimeoutSec) -ForegroundColor Red
            try { $proc.Kill() } catch { }
            $killed = $true; $killWhy = "timeout"
            break
        }
    }
    $proc.WaitForExit()
    $code = $proc.ExitCode
    $sw.Stop()
    $asked++

    # Save the answer even on a kill. A truncated answer still tells you whether the
    # model was producing sense, which is the entire question being asked here.
    $answer = ""
    if (Test-Path -LiteralPath $rawFile) {
        $answer = [System.IO.File]::ReadAllText($rawFile)
    }
    $header = @(
        "# question : " + $q.BaseName,
        "# asked    : " + $stamp,
        "# prompt   : " + ($qText -replace "`r?`n", " / "),
        "# settings : n=$NPredict ctx=$CtxSize reasoning=$Reasoning temp=0 seed=$Seed arena=${ExpertArenaMB}MB",
        "# wall_s   : " + [math]::Round($sw.Elapsed.TotalSeconds, 1),
        "# killed   : " + $killed + $(if ($killWhy -ne "") { " ($killWhy)" } else { "" }),
        "# exit     : " + $code,
        "# ---------------- verbatim output below, nothing edited ----------------",
        ""
    ) -join "`r`n"
    [System.IO.File]::WriteAllText($ansFile, $header + $answer)

    Write-Host ("  {0:N1} s   killed={1}   -> {2}" -f $sw.Elapsed.TotalSeconds, $killed, (Split-Path -Leaf $ansFile))

    # One row per question, written immediately. A set that only writes its summary
    # at the end loses everything when it is interrupted, and this one will be.
    $row = [pscustomobject]@{
        timestamp   = $stamp
        tag         = $Tag
        question    = $q.BaseName
        n_predict   = $NPredict
        ctx         = $CtxSize
        reasoning   = $Reasoning
        arena_mb    = $ExpertArenaMB
        ws_cap_mb   = $WorkingSetCapMB
        wall_s      = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        killed      = $killed
        kill_reason = $killWhy
        min_free_mb = $(if ($minFreeMb -eq [double]::MaxValue) { $null } else { $minFreeMb })
        exit_code   = $code
        answer_file = (Split-Path -Leaf $ansFile)
    }
    if (Test-Path -LiteralPath $summaryCsv) { $row | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation -Append -Encoding ASCII }
    else                                    { $row | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation -Encoding ASCII }
}

Write-Host ""
Write-Host "=== SET OVER ===" -ForegroundColor Green
Write-Host ("  asked this session : {0}" -f $asked)
Write-Host ("  answers            : {0}" -f $outDir)
Write-Host ("  summary            : {0}" -f $summaryCsv)
Write-Host ""
Write-Host "NEXT: grade by hand against bench\prompts\quality\ANSWER_KEY.md." -ForegroundColor Cyan
Write-Host "The scoring bands are fixed in that file ON PURPOSE, so the result cannot be"
Write-Host "talked into being fine after the fact. Record the total in docs\."
Write-Host ""
