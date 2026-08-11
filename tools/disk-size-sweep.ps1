# disk-size-sweep.ps1
#
# THE POINT OF THIS SCRIPT
# Our first disk benchmark reported 2.3 GB/s. Our second reported 1.07 GB/s. The difference
# was not thermal and not the queue depth: it was the SIZE OF THE FILE BEING READ. A small
# file sits in the SSD's SLC cache and the host memory buffer, so you measure the cache
# instead of the flash.
#
# This script measures sustained random-read bandwidth as a FUNCTION of working-set size.
# The number we actually need for the design is the value at ~75 GB, because that is how big
# the expert bank is. Anything measured on a small file is a lie for our purposes.
#
# It also measures the REAL model shard read-only, which is the honest answer with no
# extrapolation at all.
#
# SAFETY: the real-file path is opened read-only with OPEN_EXISTING and is NEVER created,
# never written, never deleted. Synthetic files live in a scratch subdirectory only.
#
# PowerShell 5.1. Pure ASCII. No && || ternary ??

param(
    [int[]]  $SizesMB   = @(1024, 2048, 4096, 8192, 16384, 32768),
    [string] $ScratchDir = "E:\_bench",
    [string] $RealFile  = "",
    [int]    $BlockKB   = 8192,
    [int]    $ReadsPerRun = 192,
    [int]    $Runs      = 3,
    [switch] $IncludeSynthetic
)

# SAFETY: this script NEVER deletes a file.
# The synthetic size sweep is now OPT-IN (-IncludeSynthetic) because it writes up to 63 GB
# of test files and previously deleted them itself. It no longer does. If you opt in, it
# prints the exact list of files it created at the end so YOU can delete them.
#
# It is also opt-in for a second, measured reason: the synthetic sweep CONFOUNDS ITS OWN
# RESULT. Writing and deleting tens of GB puts the SSD controller into garbage collection,
# which cuts read bandwidth by ~3x for minutes afterwards. The real-model read-only
# measurement is the trustworthy one and is what runs by default.

$ErrorActionPreference = "Stop"
$LINK_MBPS = 3940.0   # PCIe 3.0 x4 ceiling

$src = @"
using System;
using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public class DSS {
  const uint NOBUF=0x20000000, GR=0x80000000, OPEN_EXISTING=3, SHARE_READ=1;
  [DllImport("kernel32.dll",SetLastError=true,CharSet=CharSet.Unicode)]
  static extern SafeFileHandle CreateFileW(string n,uint a,uint s,IntPtr sec,uint d,uint f,IntPtr t);
  [DllImport("kernel32.dll",SetLastError=true)]
  static extern bool ReadFile(SafeFileHandle h,IntPtr b,uint n,out uint r,IntPtr o);
  [DllImport("kernel32.dll",SetLastError=true)]
  static extern bool SetFilePointerEx(SafeFileHandle h,long d,IntPtr np,uint m);
  [DllImport("kernel32.dll",SetLastError=true,CharSet=CharSet.Unicode)]
  static extern bool GetDiskFreeSpaceW(string root,out uint spc,out uint bps,out uint fc,out uint tc);

  public static uint SectorSize(string root){
    uint spc,bps,fc,tc;
    if(GetDiskFreeSpaceW(root,out spc,out bps,out fc,out tc)) { return bps; }
    return 4096;
  }

  // Random aligned reads across [0, limitBytes). Read-only, never creates.
  public static double RandomRead(string path,long limitBytes,int block,int reads,int seed,uint sector){
    SafeFileHandle h=CreateFileW(path,GR,SHARE_READ,IntPtr.Zero,OPEN_EXISTING,NOBUF,IntPtr.Zero);
    if(h.IsInvalid) throw new Exception("CreateFileW failed ("+Marshal.GetLastWin32Error()+") on "+path);
    IntPtr raw=Marshal.AllocHGlobal(block+(int)sector);
    long a=raw.ToInt64(); IntPtr buf=new IntPtr(((a+sector-1)/sector)*sector);
    try{
      var rnd=new Random(seed);
      long maxUnit=(limitBytes-block)/sector;
      if(maxUnit<1) throw new Exception("file smaller than one block");
      uint got; long done=0;
      var sw=Stopwatch.StartNew();
      for(int i=0;i<reads;i++){
        long off=((long)(rnd.NextDouble()*maxUnit))*sector;
        SetFilePointerEx(h,off,IntPtr.Zero,0);
        if(!ReadFile(h,buf,(uint)block,out got,IntPtr.Zero))
          throw new Exception("ReadFile failed ("+Marshal.GetLastWin32Error()+")");
        done+=got;
      }
      sw.Stop();
      return (done/1048576.0)/sw.Elapsed.TotalSeconds;
    } finally { Marshal.FreeHGlobal(raw); h.Close(); }
  }

  public static void Make(string p,long size){
    var b=new byte[1<<20];
    new Random(7).NextBytes(b);
    using(var fs=new FileStream(p,FileMode.Create,FileAccess.Write,FileShare.None,1<<20)){
      for(long w=0;w<size;w+=b.Length){ fs.Write(b,0,b.Length); }
      fs.Flush(true);
    }
  }
}
"@
Add-Type -TypeDefinition $src -Language CSharp | Out-Null

function Measure-Target {
    param([string]$Path,[long]$LimitBytes,[string]$Label,[uint32]$Sector)
    $block = $BlockKB * 1024
    $rates = @()
    for ($r = 1; $r -le $Runs; $r++) {
        $mbps = [DSS]::RandomRead($Path, $LimitBytes, $block, $ReadsPerRun, (1000 + $r * 37), $Sector)
        $rates += $mbps
        if ($mbps -gt $LINK_MBPS) {
            Write-Host "  !! $([math]::Round($mbps,0)) MB/s EXCEEDS THE 3940 MB/s PCIe LINK." -ForegroundColor Red
            Write-Host "  !! THE BENCHMARK IS BROKEN, NOT THE DRIVE. Buffering is happening." -ForegroundColor Red
        }
    }
    $sorted = $rates | Sort-Object
    $med = $sorted[[int]([math]::Floor($sorted.Count / 2))]
    $mn  = $sorted[0]
    $mx  = $sorted[$sorted.Count - 1]
    $pct = 100.0 * $med / $LINK_MBPS
    Write-Host ("{0,-22} {1,10:N0} {2,10:N0} {3,10:N0} {4,9:N1}%" -f $Label, $med, $mn, $mx, $pct)
    return [pscustomobject]@{
        target      = $Label
        working_set_GB = [math]::Round($LimitBytes / 1GB, 2)
        median_MBps = [math]::Round($med, 1)
        min_MBps    = [math]::Round($mn, 1)
        max_MBps    = [math]::Round($mx, 1)
        pct_of_link = [math]::Round($pct, 1)
    }
}

New-Item -ItemType Directory -Force $ScratchDir | Out-Null
$sector = [DSS]::SectorSize(([System.IO.Path]::GetPathRoot($ScratchDir)))
Write-Host ""
Write-Host "Sustained random-read bandwidth vs WORKING SET SIZE"
Write-Host ("Block {0} KB, {1} reads/run, {2} runs, median reported. Real sector size {3} bytes." -f $BlockKB, $ReadsPerRun, $Runs, $sector)
Write-Host ""
Write-Host ("{0,-22} {1,10} {2,10} {3,10} {4,10}" -f "WORKING SET", "median", "min", "max", "% of link")
Write-Host ("-" * 68)

$results = @()
$created = @()

if ($IncludeSynthetic) {
    foreach ($mb in $SizesMB) {
        $f = Join-Path $ScratchDir ("sweep_{0}mb.bin" -f $mb)
        $bytes = [long]$mb * 1MB
        $free = (Get-PSDrive ([System.IO.Path]::GetPathRoot($ScratchDir).TrimEnd('\',':') )).Free
        # Every synthetic file is KEPT, so we need room for all of them at once, not one.
        if ($free -lt ($bytes + 5GB)) {
            Write-Host ("SKIP {0} MB - not enough free space (files are kept, not deleted)" -f $mb) -ForegroundColor Yellow
            continue
        }
        [DSS]::Make($f, $bytes)
        $created += $f
        # let the SLC cache settle after the write, otherwise we measure the write cache
        Start-Sleep -Seconds 20
        $results += Measure-Target -Path $f -LimitBytes $bytes -Label ("synthetic {0} GB" -f [math]::Round($mb/1024,0)) -Sector $sector
    }
}

if ($RealFile -ne "") {
    if (-not (Test-Path $RealFile)) {
        Write-Host "RealFile not found: $RealFile" -ForegroundColor Red
    } else {
        $len = (Get-Item $RealFile).Length
        # READ ONLY. This file is the model. It is never created, written or deleted here.
        $results += Measure-Target -Path $RealFile -LimitBytes $len -Label ("REAL model shard") -Sector $sector
    }
}

Write-Host ""
if ($created.Count -gt 0) {
    Write-Host "FILES THIS SCRIPT CREATED - it does not delete them, please remove them yourself:" -ForegroundColor Yellow
    $totalGB = 0
    foreach ($f in $created) {
        if (Test-Path $f) {
            $gb = (Get-Item $f).Length / 1GB
            $totalGB += $gb
            Write-Host ("  {0}   ({1:N1} GB)" -f $f, $gb)
        }
    }
    Write-Host ("  total: {0:N1} GB" -f $totalGB)
    Write-Host ""
}
$outDir = Join-Path (Split-Path -Parent $PSScriptRoot) "bench\results"
New-Item -ItemType Directory -Force $outDir | Out-Null
$csv = Join-Path $outDir ("disk-size-sweep-{0}.csv" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$results | Export-Csv -NoTypeInformation -Path $csv
Write-Host "CSV: $csv"
Write-Host ""

if ($results.Count -ge 2) {
    $first = $results[0]
    $last  = $results[$results.Count - 1]
    Write-Host "VERDICT"
    Write-Host ("  smallest working set ({0} GB): {1} MB/s" -f $first.working_set_GB, $first.median_MBps)
    Write-Host ("  largest  working set ({0} GB): {1} MB/s" -f $last.working_set_GB, $last.median_MBps)
    if ($last.median_MBps -gt 0) {
        $ratio = $first.median_MBps / $last.median_MBps
        Write-Host ("  overstatement factor of the small-file benchmark: {0:N2}x" -f $ratio)
        if ($ratio -gt 1.5) {
            Write-Host "  CONFIRMED: bandwidth is a function of working-set size."
            Write-Host "  Small-file benchmarks measure the SLC cache, not the flash."
            Write-Host "  USE THE LARGEST WORKING SET NUMBER FOR ALL DESIGN ESTIMATES."
        } else {
            Write-Host "  Working-set size does not explain the earlier discrepancy. Look elsewhere."
        }
    }
}
Write-Host ""
