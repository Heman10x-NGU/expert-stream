# disk-recovery-curve.ps1
#
# WHY THIS EXISTS
# We measured random 8 MB reads from the real 45.7 GB model shard twice, with identical
# read-only code, on the same idle machine:
#
#   run A (drive idle, nothing written recently) : 1,050 MB/s median, 6% spread
#   run B (immediately after writing and deleting
#          ~63 GB of synthetic benchmark files)  :   318 MB/s median, 95-1012 range
#
# Same file. Same code. 3.3x apart. The difference is that run B happened while the SSD
# controller was doing garbage collection and SLC-cache flush caused by our own writes.
#
# This script measures READ bandwidth as a function of TIME SINCE THE LAST BIG WRITE, by
# sampling the real model shard repeatedly and watching it recover.
#
# This is not academic. Our engine will write things too: a disk KV cache, logs, and the
# Windows pagefile. If background writes cost 3x of read bandwidth, then "keep writes off
# the model's physical disk" becomes a hard design rule rather than a nicety.
#
# SAFETY: opens the target read-only with OPEN_EXISTING. Never creates, writes or deletes it.
#
# PowerShell 5.1. Pure ASCII.

param(
    [string] $RealFile = "E:\models\ds4f-iq1s\UD-IQ1_S\DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf",
    [int]    $Minutes = 20,
    [int]    $IntervalSec = 60,
    [int]    $BlockKB = 8192,
    [int]    $ReadsPerSample = 96
)

$ErrorActionPreference = "Stop"
$LINK_MBPS = 3940.0

$src = @"
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public class DRC {
  const uint NOBUF=0x20000000, GR=0x80000000, OPEN_EXISTING=3, SHARE_READ=1;
  [DllImport("kernel32.dll",SetLastError=true,CharSet=CharSet.Unicode)]
  static extern SafeFileHandle CreateFileW(string n,uint a,uint s,IntPtr sec,uint d,uint f,IntPtr t);
  [DllImport("kernel32.dll",SetLastError=true)]
  static extern bool ReadFile(SafeFileHandle h,IntPtr b,uint n,out uint r,IntPtr o);
  [DllImport("kernel32.dll",SetLastError=true)]
  static extern bool SetFilePointerEx(SafeFileHandle h,long d,IntPtr np,uint m);
  public static double Sample(string path,long limit,int block,int reads,int seed){
    SafeFileHandle h=CreateFileW(path,GR,SHARE_READ,IntPtr.Zero,OPEN_EXISTING,NOBUF,IntPtr.Zero);
    if(h.IsInvalid) throw new Exception("open failed "+Marshal.GetLastWin32Error());
    IntPtr raw=Marshal.AllocHGlobal(block+4096);
    long a=raw.ToInt64(); IntPtr buf=new IntPtr(((a+4095)/4096)*4096);
    try{
      var rnd=new Random(seed); long maxUnit=(limit-block)/4096; uint got; long done=0;
      var sw=Stopwatch.StartNew();
      for(int i=0;i<reads;i++){
        long off=((long)(rnd.NextDouble()*maxUnit))*4096;
        SetFilePointerEx(h,off,IntPtr.Zero,0);
        if(!ReadFile(h,buf,(uint)block,out got,IntPtr.Zero)) throw new Exception("read failed "+Marshal.GetLastWin32Error());
        done+=got;
      }
      sw.Stop();
      return (done/1048576.0)/sw.Elapsed.TotalSeconds;
    } finally { Marshal.FreeHGlobal(raw); h.Close(); }
  }
}
"@
Add-Type -TypeDefinition $src -Language CSharp | Out-Null

if (-not (Test-Path $RealFile)) { throw "RealFile not found: $RealFile" }
$len = (Get-Item $RealFile).Length
$block = $BlockKB * 1024

Write-Host ""
Write-Host "Read bandwidth recovery curve"
Write-Host ("Target : {0}" -f $RealFile)
Write-Host ("Size   : {0:N2} GB, read-only, unbuffered, random {1} KB blocks" -f ($len/1GB), $BlockKB)
Write-Host ("Plan   : one sample every {0}s for {1} minutes" -f $IntervalSec, $Minutes)
Write-Host ""
Write-Host ("{0,10} {1,12} {2,12}" -f "elapsed_s", "MBps", "pct_of_link")
Write-Host ("-" * 38)

$rows = @()
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$deadline = $Minutes * 60
$i = 0
while ($sw.Elapsed.TotalSeconds -lt $deadline) {
    $i++
    $t = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    $mbps = [DRC]::Sample($RealFile, $len, $block, $ReadsPerSample, (500 + $i * 17))
    $pct = 100.0 * $mbps / $LINK_MBPS
    Write-Host ("{0,10:N1} {1,12:N0} {2,11:N1}%" -f $t, $mbps, $pct)
    $rows += [pscustomobject]@{
        elapsed_sec = $t
        MBps        = [math]::Round($mbps, 1)
        pct_of_link = [math]::Round($pct, 1)
    }
    $remain = $IntervalSec - ($sw.Elapsed.TotalSeconds - $t)
    if ($remain -gt 0 -and $sw.Elapsed.TotalSeconds -lt $deadline) { Start-Sleep -Seconds ([int]$remain) }
}

$outDir = Join-Path (Split-Path -Parent $PSScriptRoot) "bench\results"
New-Item -ItemType Directory -Force $outDir | Out-Null
$csv = Join-Path $outDir ("disk-recovery-{0}.csv" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$rows | Export-Csv -NoTypeInformation -Path $csv

$vals   = $rows | ForEach-Object { $_.MBps }
$sorted = $vals | Sort-Object
$median = $sorted[[int]([math]::Floor($sorted.Count / 2))]
$first3 = ($vals | Select-Object -First 3 | Measure-Object -Average).Average
$last3  = ($vals | Select-Object -Last 3  | Measure-Object -Average).Average
$stats  = $vals | Measure-Object -Average -Minimum -Maximum
$sd     = 0.0
if ($vals.Count -gt 1) {
    $sumsq = 0.0
    foreach ($v in $vals) { $sumsq += [math]::Pow($v - $stats.Average, 2) }
    $sd = [math]::Sqrt($sumsq / ($vals.Count - 1))
}
$cov = 0.0
if ($stats.Average -gt 0) { $cov = 100.0 * $sd / $stats.Average }
# Outliers: samples below 70% of the median. On a recovered drive these are almost always
# caused by a CONCURRENT WRITER on the same physical disk, which is the whole point of this
# test - so count them rather than averaging them away.
$outliers = @($vals | Where-Object { $_ -lt ($median * 0.7) })

Write-Host ""
Write-Host "VERDICT"
Write-Host ("  samples              : {0}" -f $rows.Count)
Write-Host ("  median               : {0:N0} MB/s" -f $median)
Write-Host ("  min / max            : {0:N0} / {1:N0} MB/s" -f $stats.Minimum, $stats.Maximum)
Write-Host ("  coefficient of var   : {0:N1}%" -f $cov)
Write-Host ("  first 3 avg          : {0:N0} MB/s" -f $first3)
Write-Host ("  last 3 avg           : {0:N0} MB/s" -f $last3)
Write-Host ("  samples below 70% of median : {0}" -f $outliers.Count)

# Two distinct questions, and the earlier version of this script conflated them:
#   (a) did the drive START degraded and climb back?  -> a recovery curve
#   (b) is it steady now, apart from transient dips?  -> a steady-state measurement
# If the run begins already recovered, (a) is unanswerable and reporting "no recovery"
# would be actively misleading.
$startedLow = $first3 -lt ($median * 0.8)
if ($startedLow) {
    $rec = $last3 / $first3
    Write-Host ("  recovery factor      : {0:N2}x" -f $rec)
    if ($rec -gt 1.4) {
        Write-Host "  CONFIRMED: read bandwidth recovers once the controller finishes"
        Write-Host "  post-write housekeeping. DESIGN RULE: no bulk writes on the model disk."
    } else {
        Write-Host "  Started low and did NOT recover. The low number may be the steady"
        Write-Host "  state rather than a write-induced transient. Investigate elsewhere."
    }
} else {
    Write-Host "  ALREADY RECOVERED AT START - this run measures steady state, not recovery."
    Write-Host "  (A recovery factor would be meaningless here and is not reported.)"
    if ($outliers.Count -gt 0) {
        Write-Host ""
        Write-Host ("  NOTE: {0} sample(s) fell below 70% of the median while the rest held" -f $outliers.Count)
        Write-Host "  steady. On an otherwise quiet drive that pattern points at a CONCURRENT"
        Write-Host "  WRITER on the same physical disk. Check what else was running."
    }
}
Write-Host ("  CSV: {0}" -f $csv)
Write-Host ""
