#requires -Version 5.1
<#
disk-soak.ps1  (Task 0.17)

Purpose
-------
We have an SPCC DRAM-less NVMe SSD on E:. Short benchmarks (diskbench-qd.ps1,
a few seconds to tens of seconds per run) measured ~2.1 GB/s, but longer runs
measured only ~0.7-1.2 GB/s. In one earlier test, SEQUENTIAL 8 MB reads
measured SLOWER than RANDOM 8 MB reads, which is not physically possible on a
stable device -- it is a symptom of something changing state mid-benchmark.

Hypothesis: the SSD controller thermally throttles under sustained load,
and/or its SLC write cache (DRAM-less SPCC controllers commonly use an SLC
cache in front of slower QLC/TLC storage) is still flushing in the
background and contends with reads.

This script exists to settle the question by reading continuously for
several minutes and watching whether throughput decays over time. Our
inference engine will hammer this drive continuously for hours, so every
downstream design number (expert load latency, prefetch depth, etc.) must be
based on the SUSTAINED steady-state number this script reports, never the
burst number.

Method
------
1. Create (or reuse) a large file of RANDOM bytes on E: -- not zeros, because
   some controllers compress/dedupe all-zero data and would report a
   fictitious speed.
2. If the file was freshly created, sleep 60 seconds before measuring, to let
   any SLC write cache finish flushing to the slower backing store. This
   would otherwise contaminate the read numbers with write-flush contention
   that is unrelated to the read path we actually care about.
3. Read continuously for -Minutes using FILE_FLAG_NO_BUFFERING (unbuffered
   I/O), at random offsets aligned to the volume's real sector size (queried
   via GetDiskFreeSpaceW, NOT assumed to be 4096 -- on this machine it is
   512). This bypasses the Windows page cache so we are actually measuring
   the device, not RAM.
4. Every -WindowSec seconds, print and record one row: elapsed time, this
   window's MB/s, the cumulative MB/s so far, and drive temperature if it
   can be obtained.
5. At the end, compare the median of the first 60 seconds ("burst") against
   the median of the last 120 seconds ("sustained") and print a verdict.

Correctness requirements for unbuffered I/O (get these wrong and reads
silently corrupt or fail):
  - Read buffers must be aligned to the volume sector size. We over-allocate
    unmanaged memory by one sector and round the pointer up.
  - Read offsets must be multiples of the sector size.
  - The read length (block size) must be a multiple of the sector size.

Units
-----
File/block SIZES (-FileSizeMB, -BlockKB) use binary units (1 MB = 1048576
bytes, matching Windows convention for "how big a file do I need").
THROUGHPUT (MB/s, the PCIe link ceiling) uses DECIMAL MB (1 MB = 1,000,000
bytes), matching how NVMe vendors and tools like CrystalDiskMark report
GB/s, and matching this project's other benchmarking scripts.
#>

[CmdletBinding()]
param(
    [string]$File = 'E:\_soaktest.bin',
    [long]$FileSizeMB = 8192,
    [int]$Minutes = 10,
    [int]$BlockKB = 8192,
    [int]$WindowSec = 10
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$BytesPerMiB     = 1048576        # binary MB, used for sizing
$BytesPerMB_dec  = 1000000        # decimal MB, used for throughput
$BytesPerGB_dec  = 1000000000     # decimal GB
$LinkCeilingMBps = 3940           # PCIe 3.0 x4 ceiling, decimal MB/s

# ---------------------------------------------------------------------------
# C# helper: raw unbuffered I/O via CreateFileW / ReadFile / SetFilePointerEx
# ---------------------------------------------------------------------------
$csharpSource = @'
using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Microsoft.Win32.SafeHandles;

public static class DiskSoakIo
{
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern SafeFileHandle CreateFileW(
        string lpFileName, uint dwDesiredAccess, uint dwShareMode,
        IntPtr lpSecurityAttributes, uint dwCreationDisposition,
        uint dwFlagsAndAttributes, IntPtr hTemplateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool ReadFile(SafeFileHandle hFile, IntPtr lpBuffer,
        uint nNumberOfBytesToRead, out uint lpNumberOfBytesRead, IntPtr lpOverlapped);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetFilePointerEx(SafeFileHandle hFile, long liDistanceToMove,
        out long lpNewFilePointer, uint dwMoveMethod);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    public static extern bool GetDiskFreeSpaceW(string lpRootPathName,
        out uint lpSectorsPerCluster, out uint lpBytesPerSector,
        out uint lpNumberOfFreeClusters, out uint lpTotalNumberOfClusters);

    private const uint GENERIC_READ = 0x80000000;
    private const uint FILE_SHARE_READ = 0x1;
    private const uint FILE_SHARE_WRITE = 0x2;
    private const uint OPEN_EXISTING = 3;
    private const uint FILE_FLAG_NO_BUFFERING = 0x20000000;
    private const uint FILE_ATTRIBUTE_NORMAL = 0x80;

    public class WindowResult
    {
        public long BytesRead;
        public int ReadsCompleted;
        public int Errors;
        public double ElapsedMs;
    }

    public static SafeFileHandle OpenUnbuffered(string path, out int win32Error)
    {
        SafeFileHandle h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            IntPtr.Zero, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
        win32Error = Marshal.GetLastWin32Error();
        return h;
    }

    // Reads at random sector-aligned offsets, in a tight loop, until
    // durationMs of wall-clock time has elapsed. Returns totals for that
    // window. The handle and buffer are supplied by the caller and reused
    // across windows so the run is genuinely continuous (no reopen/close
    // between windows).
    public static WindowResult ReadForDuration(SafeFileHandle handle, IntPtr bufPtr,
        int blockSize, int sectorSize, long fileSizeBytes, int durationMs, int seed)
    {
        Random rnd = new Random(seed);
        long maxOffsetSectors = (fileSizeBytes - blockSize) / sectorSize;
        if (maxOffsetSectors < 1) maxOffsetSectors = 1;

        long bytesRead = 0;
        int reads = 0;
        int errors = 0;

        Stopwatch sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < durationMs)
        {
            long offsetSectors = (long)(rnd.NextDouble() * maxOffsetSectors);
            long offset = offsetSectors * sectorSize;
            long newPos;
            if (!SetFilePointerEx(handle, offset, out newPos, 0))
            {
                errors++;
                continue;
            }
            uint br;
            bool ok = ReadFile(handle, bufPtr, (uint)blockSize, out br, IntPtr.Zero);
            if (!ok || br != (uint)blockSize)
            {
                errors++;
                continue;
            }
            bytesRead += br;
            reads++;
        }
        sw.Stop();

        WindowResult r = new WindowResult();
        r.BytesRead = bytesRead;
        r.ReadsCompleted = reads;
        r.Errors = errors;
        r.ElapsedMs = sw.Elapsed.TotalMilliseconds;
        return r;
    }
}
'@

Add-Type -TypeDefinition $csharpSource -Language CSharp

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
function Get-DriveLetterFromPath {
    param([string]$p)
    $full = [System.IO.Path]::GetFullPath($p)
    if ($full.Length -ge 2 -and $full[1] -eq ':') {
        return $full.Substring(0, 1)
    }
    return $null
}

function Get-Median {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $n = $sorted.Count
    if ($n % 2 -eq 1) {
        return $sorted[[int](($n - 1) / 2)]
    } else {
        $a = $sorted[[int]($n / 2) - 1]
        $b = $sorted[[int]($n / 2)]
        return ($a + $b) / 2.0
    }
}

function Get-ArrayStats {
    # Deliberately takes a plain [double[]] and never runs
    # Measure-Object -Property on an array of objects/hashtables (that
    # pattern fails in Windows PowerShell 5.1). Callers must extract the
    # values with ForEach-Object first.
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -eq 0) {
        return [PSCustomObject]@{ Mean = 0; Min = 0; Max = 0; StdDev = 0; CoV = 0; Count = 0 }
    }
    $m = $Values | Measure-Object -Average -Minimum -Maximum
    $mean = $m.Average
    $sumSq = 0.0
    foreach ($v in $Values) { $sumSq += [Math]::Pow(($v - $mean), 2) }
    $stdDev = [Math]::Sqrt($sumSq / $Values.Count)
    $cov = 0
    if ($mean -ne 0) { $cov = ($stdDev / $mean) * 100.0 }
    return [PSCustomObject]@{
        Mean   = $mean
        Min    = $m.Minimum
        Max    = $m.Maximum
        StdDev = $stdDev
        CoV    = $cov
        Count  = $Values.Count
    }
}

function Write-BrokenBenchmarkWarning {
    param([double]$WindowMBps)
    Write-Host ''
    Write-Host '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' -ForegroundColor Red
    Write-Host ("WARNING: window throughput {0:N0} MB/s exceeds the PCIe 3.0 x4 link" -f $WindowMBps) -ForegroundColor Red
    Write-Host ("ceiling of {0} MB/s. BENCHMARK IS BROKEN, NOT THE DRIVE." -f $LinkCeilingMBps) -ForegroundColor Red
    Write-Host 'This means buffering is happening somewhere (page cache, controller' -ForegroundColor Red
    Write-Host 'cache, or FILE_FLAG_NO_BUFFERING silently not applied) and the reads' -ForegroundColor Red
    Write-Host 'are not actually reaching the physical device.' -ForegroundColor Red
    Write-Host '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' -ForegroundColor Red
    Write-Host ''
}

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
$filePath = [System.IO.Path]::GetFullPath($File)
$driveLetter = Get-DriveLetterFromPath $filePath
if (-not $driveLetter) {
    throw "Could not determine a drive letter from -File '$File'."
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$resultsDir = Join-Path $PSScriptRoot '..\bench\results'
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
}
$csvPath = Join-Path $resultsDir ("disk-soak-{0}.csv" -f $timestamp)

Write-Host '============================================================'
Write-Host 'disk-soak.ps1 - sustained-read soak test (Task 0.17)'
Write-Host '============================================================'
Write-Host ("Target file       : {0}" -f $filePath)
Write-Host ("File size target  : {0} MB" -f $FileSizeMB)
Write-Host ("Duration          : {0} minutes" -f $Minutes)
Write-Host ("Block size        : {0} KB" -f $BlockKB)
Write-Host ("Window size       : {0} sec" -f $WindowSec)
Write-Host ''

# ---------------------------------------------------------------------------
# Sector size -- must come from the OS, never hardcoded (this machine
# reports 512, not the commonly-assumed 4096).
# ---------------------------------------------------------------------------
$rootForSectorQuery = "$driveLetter`:\"
[uint32]$sectorsPerCluster = 0
[uint32]$bytesPerSector = 0
[uint32]$freeClusters = 0
[uint32]$totalClusters = 0
$sectorOk = [DiskSoakIo]::GetDiskFreeSpaceW($rootForSectorQuery, [ref]$sectorsPerCluster, [ref]$bytesPerSector, [ref]$freeClusters, [ref]$totalClusters)
if (-not $sectorOk -or $bytesPerSector -eq 0) {
    throw "GetDiskFreeSpaceW failed for '$rootForSectorQuery' -- cannot determine the real sector size, refusing to guess."
}
$sectorSize = [int]$bytesPerSector
Write-Host ("Sector size (real): {0} bytes" -f $sectorSize)

# ---------------------------------------------------------------------------
# Physical disk resolution, for temperature lookups later. Best-effort.
# ---------------------------------------------------------------------------
$physicalDisk = $null
try {
    $partition = Get-Partition -DriveLetter $driveLetter -ErrorAction Stop
    $disk = Get-Disk -Number $partition.DiskNumber -ErrorAction Stop
    $physicalDisk = Get-PhysicalDisk -ErrorAction Stop | Where-Object { $_.DeviceId -eq [string]$disk.Number } | Select-Object -First 1
    if ($physicalDisk) {
        Write-Host ("Physical disk     : {0}" -f $physicalDisk.FriendlyName)
    }
} catch {
    $physicalDisk = $null
}

$script:TempWarningShown = $false
function Get-DriveTempC {
    if (-not $physicalDisk) {
        if (-not $script:TempWarningShown) {
            Write-Host 'Temperature: could not resolve a physical disk object for this drive letter. Column will be blank.'
            $script:TempWarningShown = $true
        }
        return ''
    }
    try {
        $counter = Get-StorageReliabilityCounter -PhysicalDisk $physicalDisk -ErrorAction Stop
        if ($counter -and $counter.Temperature) {
            return $counter.Temperature
        }
        return ''
    } catch {
        if (-not $script:TempWarningShown) {
            Write-Host 'Temperature: Get-StorageReliabilityCounter failed. This usually needs an admin PowerShell. Column will be blank for this run.'
            $script:TempWarningShown = $true
        }
        return ''
    }
}

Write-Host ''

# ---------------------------------------------------------------------------
# Create or reuse the test file
# ---------------------------------------------------------------------------
$fileSizeBytesTarget = $FileSizeMB * $BytesPerMiB
$fileWasCreated = $false

$reuse = $false
if (Test-Path $filePath) {
    $existing = Get-Item $filePath
    if ($existing.Length -eq $fileSizeBytesTarget) {
        $reuse = $true
    }
}

if ($reuse) {
    Write-Host ("Reusing existing test file (size already matches): {0}" -f $filePath)
} else {
    Write-Host ("Creating {0} MB test file of RANDOM bytes (not zeros) at: {1}" -f $FileSizeMB, $filePath)
    Write-Host "Using random data because some controllers compress or dedupe all-zero data, which would give a fake speed reading."

    $chunkSize = 1 * $BytesPerMiB   # 1 MB buffer, as specified
    $rnd = New-Object System.Random
    $buffer = New-Object byte[] ($chunkSize)

    $fs = [System.IO.File]::Open($filePath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $written = [long]0
        while ($written -lt $fileSizeBytesTarget) {
            $rnd.NextBytes($buffer)
            $remaining = $fileSizeBytesTarget - $written
            $toWrite = $chunkSize
            if ($remaining -lt $chunkSize) { $toWrite = [int]$remaining }
            $fs.Write($buffer, 0, $toWrite)
            $written += $toWrite
        }
        $fs.Flush($true)   # flush(true) = also flush OS/device buffers, not just the .NET stream buffer
    } finally {
        $fs.Close()
    }
    $fileWasCreated = $true
    Write-Host "Test file created and flushed to disk."
}

if ($fileWasCreated) {
    Write-Host ''
    Write-Host "Sleeping 60 seconds before measuring: letting the SLC cache flush after the write, so write-flush contention does not contaminate the read numbers."
    Start-Sleep -Seconds 60
} else {
    Write-Host "File was reused (not freshly written), skipping the 60-second SLC-flush settle sleep."
}

$fileInfo = Get-Item $filePath
$fileSizeBytes = $fileInfo.Length
Write-Host ("Actual file size  : {0:N2} GB ({1} bytes)" -f ($fileSizeBytes / $BytesPerGB_dec), $fileSizeBytes)
Write-Host ''

# ---------------------------------------------------------------------------
# Prepare block size and check it is sector-aligned
# ---------------------------------------------------------------------------
$blockBytes = $BlockKB * 1024
if ($blockBytes % $sectorSize -ne 0) {
    throw "BlockKB * 1024 ($blockBytes bytes) is not a multiple of the sector size ($sectorSize). Choose a different -BlockKB."
}
if ($fileSizeBytes -le $blockBytes) {
    throw "File is not larger than one block; cannot do random reads. Increase -FileSizeMB or -BlockKB is too large."
}

# ---------------------------------------------------------------------------
# Main soak loop -- try/finally so the unmanaged buffer and file handle are
# always released, even on Ctrl-C.
# ---------------------------------------------------------------------------
$allRows = New-Object System.Collections.Generic.List[Object]
$handle = $null
$bufRaw = [IntPtr]::Zero

try {
    Write-Host "Opening file with FILE_FLAG_NO_BUFFERING (unbuffered I/O, bypasses the Windows page cache)..."
    $win32Error = 0
    $handle = [DiskSoakIo]::OpenUnbuffered($filePath, [ref]$win32Error)
    if ($null -eq $handle -or $handle.IsInvalid) {
        throw "CreateFileW failed, Win32 error $win32Error"
    }

    $bufRaw = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($blockBytes + $sectorSize)
    $rawAddr = $bufRaw.ToInt64()
    $mask = [int64]($sectorSize - 1)
    $alignedAddr = ($rawAddr + $mask) -band (-bnot $mask)
    $bufPtr = New-Object System.IntPtr($alignedAddr)

    $totalDurationSec = $Minutes * 60
    $cumulativeBytes = [long]0
    $totalReads = 0
    $totalErrors = 0
    $seedCounter = [Environment]::TickCount

    Write-Host ''
    Write-Host ("Reading continuously for {0} minutes, random {1} KB blocks, sector-aligned offsets..." -f $Minutes, $BlockKB)
    Write-Host ("{0,10} {1,12} {2,12} {3,10}" -f 'elapsed_s', 'window_MBps', 'cum_MBps', 'temp_C')

    $swTotal = [System.Diagnostics.Stopwatch]::StartNew()

    while ($true) {
        $remainingMs = ($totalDurationSec * 1000) - $swTotal.Elapsed.TotalMilliseconds
        if ($remainingMs -le 0) { break }
        $thisWindowMs = [Math]::Min($WindowSec * 1000, [int]$remainingMs)
        if ($thisWindowMs -le 0) { break }

        $seedCounter++
        $result = [DiskSoakIo]::ReadForDuration($handle, $bufPtr, $blockBytes, $sectorSize, $fileSizeBytes, [int]$thisWindowMs, $seedCounter)

        $windowSeconds = $result.ElapsedMs / 1000.0
        if ($windowSeconds -le 0) { $windowSeconds = 0.000001 }
        $windowMBps = ($result.BytesRead / $BytesPerMB_dec) / $windowSeconds

        $cumulativeBytes += $result.BytesRead
        $totalReads += $result.ReadsCompleted
        $totalErrors += $result.Errors
        $elapsedSec = $swTotal.Elapsed.TotalSeconds
        $cumulativeMBps = ($cumulativeBytes / $BytesPerMB_dec) / $elapsedSec

        $tempC = Get-DriveTempC

        $row = [PSCustomObject]@{
            TimestampUtc    = (Get-Date).ToUniversalTime().ToString('o')
            ElapsedSec      = [Math]::Round($elapsedSec, 1)
            WindowMBps      = [Math]::Round($windowMBps, 2)
            CumulativeMBps  = [Math]::Round($cumulativeMBps, 2)
            TempC           = $tempC
            WindowBytesRead = $result.BytesRead
            WindowReads     = $result.ReadsCompleted
            WindowErrors    = $result.Errors
        }
        $allRows.Add($row) | Out-Null

        Write-Host ("{0,10:N1} {1,12:N1} {2,12:N1} {3,10}" -f $row.ElapsedSec, $row.WindowMBps, $row.CumulativeMBps, $row.TempC)

        if ($result.Errors -gt 0) {
            Write-Host ("  WARNING: {0} read error(s) in this window" -f $result.Errors) -ForegroundColor Yellow
        }
        if ($windowMBps -gt $LinkCeilingMBps) {
            Write-BrokenBenchmarkWarning -WindowMBps $windowMBps
        }
    }

    $swTotal.Stop()

} finally {
    if ($bufRaw -ne [IntPtr]::Zero) {
        [System.Runtime.InteropServices.Marshal]::FreeHGlobal($bufRaw)
    }
    if ($handle -and -not $handle.IsInvalid) {
        $handle.Close()
    }
}

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '============================================================'
Write-Host 'VERDICT'
Write-Host '============================================================'

if ($allRows.Count -eq 0) {
    Write-Host "No windows completed -- cannot compute a verdict."
} else {
    $allMBpsValues = @($allRows | ForEach-Object { $_.WindowMBps })
    $totalDurationSecActual = ($allRows | Select-Object -Last 1).ElapsedSec

    $burstValues = @($allRows | Where-Object { $_.ElapsedSec -le 60 } | ForEach-Object { $_.WindowMBps })
    $sustainedCutoff = $totalDurationSecActual - 120
    $sustainedValues = @($allRows | Where-Object { $_.ElapsedSec -ge $sustainedCutoff } | ForEach-Object { $_.WindowMBps })

    $medianBurst = Get-Median -Values $burstValues
    $medianSustained = Get-Median -Values $sustainedValues

    Write-Host ("Windows recorded          : {0}" -f $allRows.Count)
    Write-Host ("Total reads / errors      : {0} / {1}" -f $totalReads, $totalErrors)

    if ($null -eq $medianBurst -or $burstValues.Count -eq 0) {
        Write-Host "Burst (first 60s) median  : not enough data"
    } else {
        Write-Host ("Burst (first 60s) median  : {0:N1} MB/s  ({1} windows)" -f $medianBurst, $burstValues.Count)
    }
    if ($null -eq $medianSustained -or $sustainedValues.Count -eq 0) {
        Write-Host "Sustained (last 120s) med.: not enough data"
    } else {
        Write-Host ("Sustained (last 120s) med.: {0:N1} MB/s  ({1} windows)" -f $medianSustained, $sustainedValues.Count)
    }

    if ($medianBurst -and $medianBurst -gt 0 -and $medianSustained -ne $null) {
        $ratio = $medianSustained / $medianBurst
        Write-Host ("Ratio sustained/burst     : {0:N3}" -f $ratio)
        Write-Host ''
        if ($ratio -lt 0.75) {
            Write-Host "THROTTLING CONFIRMED - design must use the sustained number" -ForegroundColor Red
        } elseif ($ratio -gt 0.95) {
            Write-Host "NO THROTTLING - the drive is simply slow, ~30% of its own PCIe link" -ForegroundColor Green
        } else {
            Write-Host "INCONCLUSIVE - rerun longer" -ForegroundColor Yellow
        }
    } else {
        Write-Host ''
        Write-Host "INCONCLUSIVE - rerun longer (not enough burst/sustained data to compute a ratio)" -ForegroundColor Yellow
    }

    $stats = Get-ArrayStats -Values $allMBpsValues
    Write-Host ''
    Write-Host ("Min window MB/s           : {0:N1}" -f $stats.Min)
    Write-Host ("Max window MB/s           : {0:N1}" -f $stats.Max)
    Write-Host ("Coefficient of variation  : {0:N1}%  (across all {1} windows)" -f $stats.CoV, $stats.Count)
}

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
$allRows | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host ''
Write-Host ("Per-window results written to: {0}" -f $csvPath)

# ---------------------------------------------------------------------------
# Final notes -- test file is intentionally kept for reuse
# ---------------------------------------------------------------------------
$finalFileInfo = Get-Item $filePath
Write-Host ''
Write-Host '============================================================'
Write-Host ("Test file kept for reuse : {0}" -f $finalFileInfo.FullName)
Write-Host ("Test file size           : {0:N2} GB ({1} bytes)" -f ($finalFileInfo.Length / $BytesPerGB_dec), $finalFileInfo.Length)
Write-Host ("To delete it manually    : Remove-Item -Path '{0}' -Force" -f $finalFileInfo.FullName)
Write-Host '============================================================'
