<#
diskbench-qd.ps1

Purpose
-------
We stream 6.9 MB "expert" blobs off an NVMe SSD during LLM inference. A prior
benchmark measured 2.3 GB/s at queue depth 1 (one read at a time, waiting for
each). The drive's PCIe 3.0 x4 link ceiling is 3.94 GB/s (decimal GB, i.e.
4 lanes * ~0.985 GB/s per lane after 8b/10b-class encoding overhead). This
script measures random 8 MB reads at increasing queue depths to find out
whether the missing headroom (2.3 -> 3.94 GB/s) is recoverable by issuing
more reads concurrently, or whether it is a hard ceiling elsewhere.

Implementation approach
------------------------
This script uses the THREAD-FALLBACK approach explicitly allowed by the spec:
N background threads, each with its OWN file handle opened with
FILE_FLAG_NO_BUFFERING, each issuing SYNCHRONOUS ReadFile calls at random
aligned offsets. Queue depth N is achieved because N threads have N reads
blocked in the kernel/device simultaneously. This was chosen over manual
OVERLAPPED + WaitForMultipleObjects because it is far less fiddly to get
right, while still producing genuine concurrent I/O at the device.

Correctness requirements for unbuffered I/O (get these wrong and reads
silently corrupt or fail):
  - Read buffers must be aligned to the volume sector size. We over-allocate
    unmanaged memory and round the pointer up to a sector boundary.
  - Read offsets AND lengths must both be multiples of the sector size.
  - One buffer per outstanding request - never shared across concurrent reads.

Units
-----
File/block SIZES (-FileSizeMB, -BlockMB) use binary MB (1 MB = 1048576 bytes)
because that is the normal Windows convention for "how big a file do I need".
THROUGHPUT reporting (MB/s, GB/s, link ceiling) uses DECIMAL MB/GB
(1 MB = 1,000,000 bytes), because that is where "3.94 GB/s" for a PCIe 3.0 x4
link comes from (4 lanes * ~0.985 GB/s decimal), and it matches how NVMe
vendors and most benchmarking tools (CrystalDiskMark, etc.) report GB/s.
#>

[CmdletBinding()]
param(
    [string]$Path = 'E:\',
    [long]$FileSizeMB = 4096,
    [int]$BlockMB = 8,
    [int[]]$QueueDepths = @(1,2,4,8,16,32),
    [int]$ReadsPerDepth = 256,
    [int]$Runs = 3,
    [string]$ExistingFile = '',
    [string]$OutCsv = ''
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$BytesPerMiB      = 1048576          # binary MB, used for sizing
$BytesPerMB_dec   = 1000000          # decimal MB, used for throughput
$BytesPerGB_dec   = 1000000000       # decimal GB
$LinkCeilingGBps  = 3.94
$LinkCeilingMBps  = $LinkCeilingGBps * 1000    # decimal MB/s
$SafetyMarginBytes = 5 * $BytesPerGB_dec + 5 * $BytesPerMiB * 200  # ~5 GB margin (decimal-ish, generous)
$SafetyMarginBytes = 5368709120  # exactly 5 GiB, simple and generous

# ---------------------------------------------------------------------------
# C# helper: raw unbuffered I/O via CreateFileW / ReadFile / SetFilePointerEx
# ---------------------------------------------------------------------------
$csharpSource = @'
using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Diagnostics;
using Microsoft.Win32.SafeHandles;

public static class DiskBenchQD
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

    public class RoundResult
    {
        public double ElapsedMs;
        public long BytesRead;
        public int ReadsCompleted;
        public int Errors;
    }

    private class WorkerState
    {
        public string Path;
        public long FileSizeBytes;
        public int BlockSize;
        public int SectorSize;
        public int ReadCount;
        public int Seed;
        public ManualResetEvent StartEvent;
        public long BytesRead;
        public int Errors;
        public Exception Error;
    }

    private static void WorkerProc(object stateObj)
    {
        WorkerState st = (WorkerState)stateObj;
        IntPtr raw = IntPtr.Zero;
        SafeFileHandle handle = null;
        try
        {
            raw = Marshal.AllocHGlobal(st.BlockSize + st.SectorSize);
            long rawAddr = raw.ToInt64();
            long aligned = (rawAddr + st.SectorSize - 1) & ~((long)st.SectorSize - 1);
            IntPtr bufPtr = new IntPtr(aligned);

            handle = CreateFileW(st.Path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                IntPtr.Zero, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
            if (handle == null || handle.IsInvalid)
            {
                st.Error = new Exception("CreateFileW failed, Win32 error " + Marshal.GetLastWin32Error());
                return;
            }

            Random rnd = new Random(st.Seed);
            long maxOffsetBlocks = (st.FileSizeBytes - st.BlockSize) / st.SectorSize;
            if (maxOffsetBlocks < 1) maxOffsetBlocks = 1;

            st.StartEvent.WaitOne();

            for (int i = 0; i < st.ReadCount; i++)
            {
                long offsetBlocks = (long)(rnd.NextDouble() * maxOffsetBlocks);
                long offset = offsetBlocks * st.SectorSize;
                long newPos;
                if (!SetFilePointerEx(handle, offset, out newPos, 0))
                {
                    Interlocked.Increment(ref st.Errors);
                    continue;
                }
                uint bytesRead;
                bool ok = ReadFile(handle, bufPtr, (uint)st.BlockSize, out bytesRead, IntPtr.Zero);
                if (!ok || bytesRead != (uint)st.BlockSize)
                {
                    Interlocked.Increment(ref st.Errors);
                    continue;
                }
                Interlocked.Add(ref st.BytesRead, bytesRead);
            }
        }
        catch (Exception ex)
        {
            st.Error = ex;
        }
        finally
        {
            if (handle != null) handle.Close();
            if (raw != IntPtr.Zero) Marshal.FreeHGlobal(raw);
        }
    }

    public static RoundResult RunRound(string path, long fileSizeBytes, int blockSize, int sectorSize,
        int queueDepth, int totalReads, int seedBase)
    {
        ManualResetEvent startEvent = new ManualResetEvent(false);
        Thread[] threads = new Thread[queueDepth];
        WorkerState[] states = new WorkerState[queueDepth];

        int baseCount = totalReads / queueDepth;
        int remainder = totalReads % queueDepth;

        for (int i = 0; i < queueDepth; i++)
        {
            int cnt = baseCount;
            if (i < remainder) cnt = cnt + 1;
            WorkerState st = new WorkerState();
            st.Path = path;
            st.FileSizeBytes = fileSizeBytes;
            st.BlockSize = blockSize;
            st.SectorSize = sectorSize;
            st.ReadCount = cnt;
            st.Seed = seedBase + i * 7919 + 1;
            st.StartEvent = startEvent;
            states[i] = st;
            threads[i] = new Thread(WorkerProc);
            threads[i].IsBackground = true;
        }

        for (int i = 0; i < queueDepth; i++)
        {
            threads[i].Start(states[i]);
        }

        // Best-effort: give threads a moment to reach WaitOne() before we
        // start the clock and release them, so thread-spinup jitter does not
        // get counted as I/O time. Not perfectly synchronized, but the read
        // counts are large enough that this is noise.
        Thread.Sleep(30);

        Stopwatch sw = Stopwatch.StartNew();
        startEvent.Set();

        for (int i = 0; i < queueDepth; i++)
        {
            threads[i].Join();
        }
        sw.Stop();

        for (int i = 0; i < queueDepth; i++)
        {
            if (states[i].Error != null)
            {
                throw states[i].Error;
            }
        }

        long totalBytes = 0;
        int totalErrors = 0;
        for (int i = 0; i < queueDepth; i++)
        {
            totalBytes += states[i].BytesRead;
            totalErrors += states[i].Errors;
        }

        RoundResult result = new RoundResult();
        result.ElapsedMs = sw.Elapsed.TotalMilliseconds;
        result.BytesRead = totalBytes;
        result.Errors = totalErrors;
        result.ReadsCompleted = (int)(totalBytes / blockSize);
        return result;
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
        return $full.Substring(0,1)
    }
    return $null
}

function Get-Median {
    param([double[]]$Values)
    $sorted = $Values | Sort-Object
    $n = $sorted.Count
    if ($n -eq 0) { return 0 }
    if ($n % 2 -eq 1) {
        return $sorted[[int](($n-1)/2)]
    } else {
        $a = $sorted[[int]($n/2)-1]
        $b = $sorted[[int]($n/2)]
        return ($a + $b) / 2.0
    }
}

function Write-LinkWarningIfBroken {
    param([double]$PctOfLink, [string]$Label)
    if ($PctOfLink -gt 1.0) {
        Write-Host ''
        Write-Host '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' -ForegroundColor Red
        Write-Host ("WARNING: {0} measured throughput exceeds the PCIe 3.0 x4 link" -f $Label) -ForegroundColor Red
        Write-Host 'ceiling of 3.94 GB/s. THE BENCHMARK IS BROKEN, NOT THE DRIVE.' -ForegroundColor Red
        Write-Host 'Check: FILE_FLAG_NO_BUFFERING actually set, sector alignment of' -ForegroundColor Red
        Write-Host 'offsets/lengths/buffers, and that reads are not being served from' -ForegroundColor Red
        Write-Host 'a cache (a previous benchmark once reported 65 GB/s on a 3.94 GB/s' -ForegroundColor Red
        Write-Host 'drive because of a missing flag).' -ForegroundColor Red
        Write-Host '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' -ForegroundColor Red
        Write-Host ''
    }
}

# ---------------------------------------------------------------------------
# Resolve OutCsv default
# ---------------------------------------------------------------------------
if ([string]::IsNullOrEmpty($OutCsv)) {
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $resultsDir = Join-Path $PSScriptRoot '..\bench\results'
    $OutCsv = Join-Path $resultsDir ("diskbench-qd-{0}.csv" -f $timestamp)
}
$outDir = Split-Path -Parent $OutCsv
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

Write-Host '============================================================'
Write-Host 'diskbench-qd.ps1 - random-read queue-depth scaling benchmark'
Write-Host '============================================================'
Write-Host ("Approach: N background threads, each own handle, synchronous")
Write-Host ("unbuffered ReadFile calls at random aligned offsets.")
Write-Host ''

# ---------------------------------------------------------------------------
# Determine target file path and size
# ---------------------------------------------------------------------------
$usingExisting = -not [string]::IsNullOrEmpty($ExistingFile)
$deleteAfter = $false

if ($usingExisting) {
    if (-not (Test-Path $ExistingFile)) {
        throw "ExistingFile not found: $ExistingFile"
    }
    $filePath = (Resolve-Path $ExistingFile).Path
    $driveLetterForInfo = Get-DriveLetterFromPath $filePath
} else {
    $filePath = Join-Path $Path ("diskbench_qd_testfile_{0}.bin" -f $PID)
    $driveLetterForInfo = Get-DriveLetterFromPath $Path
}

# ---------------------------------------------------------------------------
# Preflight: physical disk info
# ---------------------------------------------------------------------------
Write-Host "Target path      : $filePath"
Write-Host "Volume            : $driveLetterForInfo`:"

if ($driveLetterForInfo) {
    try {
        $partition = Get-Partition -DriveLetter $driveLetterForInfo -ErrorAction Stop
        $disk = Get-Disk -Number $partition.DiskNumber -ErrorAction Stop
        $phys = Get-PhysicalDisk -ErrorAction Stop | Where-Object { $_.DeviceId -eq [string]$disk.Number }
        if ($phys) {
            Write-Host ("Physical disk     : {0}" -f $phys.FriendlyName)
            Write-Host ("Media type        : {0}" -f $phys.MediaType)
            Write-Host ("Bus type          : {0}" -f $phys.BusType)
        } else {
            Write-Host "Physical disk     : (could not resolve PhysicalDisk from partition/disk mapping)"
        }
    } catch {
        Write-Host "Physical disk     : (unavailable - $($_.Exception.Message))"
    }
} else {
    Write-Host "Physical disk     : (could not determine drive letter)"
}

# ---------------------------------------------------------------------------
# Preflight: sector size
# ---------------------------------------------------------------------------
$rootForSectorQuery = "$driveLetterForInfo`:\"
[uint32]$sectorsPerCluster = 0
[uint32]$bytesPerSector = 0
[uint32]$freeClusters = 0
[uint32]$totalClusters = 0
$sectorOk = [DiskBenchQD]::GetDiskFreeSpaceW($rootForSectorQuery, [ref]$sectorsPerCluster, [ref]$bytesPerSector, [ref]$freeClusters, [ref]$totalClusters)
if ($sectorOk -and $bytesPerSector -gt 0) {
    $sectorSize = [int]$bytesPerSector
} else {
    $sectorSize = 4096
    Write-Host "Sector size query failed, defaulting to 4096 bytes."
}
Write-Host ("Sector size       : {0} bytes" -f $sectorSize)

# ---------------------------------------------------------------------------
# Preflight: free space (only when creating a new test file)
# ---------------------------------------------------------------------------
$fileSizeBytesRequested = $FileSizeMB * $BytesPerMiB

if (-not $usingExisting) {
    $vol = $null
    try { $vol = Get-Volume -DriveLetter $driveLetterForInfo -ErrorAction Stop } catch { $vol = $null }
    if ($vol) {
        $freeBytes = $vol.SizeRemaining
    } else {
        $psdrive = Get-PSDrive -Name $driveLetterForInfo -ErrorAction SilentlyContinue
        if ($psdrive) { $freeBytes = $psdrive.Free } else { $freeBytes = -1 }
    }
    Write-Host ("Free space on {0}: : {1:N2} GB" -f $driveLetterForInfo, ($freeBytes / $BytesPerGB_dec))
    $needed = $fileSizeBytesRequested + $SafetyMarginBytes
    if ($freeBytes -lt 0) {
        Write-Host "WARNING: could not determine free space; proceeding anyway."
    } elseif ($freeBytes -lt $needed) {
        throw ("Not enough free space on {0}: . Need {1:N2} GB (test file + 5 GB margin), have {2:N2} GB." -f $driveLetterForInfo, ($needed / $BytesPerGB_dec), ($freeBytes / $BytesPerGB_dec))
    }
}

Write-Host ''

# ---------------------------------------------------------------------------
# Main body wrapped in try/finally so the test file is always cleaned up
# ---------------------------------------------------------------------------
$allRows = New-Object System.Collections.Generic.List[Object]

try {
    if (-not $usingExisting) {
        Write-Host ("Creating {0} MB test file with random bytes at: {1}" -f $FileSizeMB, $filePath)
        $deleteAfter = $true

        $chunkSize = 128 * $BytesPerMiB
        $rnd = New-Object System.Random
        $buffer = New-Object byte[] ($chunkSize)
        $fs = [System.IO.File]::Open($filePath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        try {
            $written = [long]0
            while ($written -lt $fileSizeBytesRequested) {
                $rnd.NextBytes($buffer)
                $remaining = $fileSizeBytesRequested - $written
                $toWrite = $chunkSize
                if ($remaining -lt $chunkSize) { $toWrite = [int]$remaining }
                $fs.Write($buffer, 0, $toWrite)
                $written += $toWrite
            }
        } finally {
            $fs.Close()
        }
        Write-Host "Test file created."
    } else {
        Write-Host "Using existing file, will NOT be deleted."
    }

    $fileInfo = Get-Item $filePath
    $fileSizeBytes = $fileInfo.Length
    Write-Host ("Benchmark file size: {0:N2} GB ({1} bytes)" -f ($fileSizeBytes / $BytesPerGB_dec), $fileSizeBytes)

    $blockSizeBytes = $BlockMB * $BytesPerMiB
    if ($blockSizeBytes % $sectorSize -ne 0) {
        throw "BlockMB * 1MiB is not a multiple of the sector size ($sectorSize). Choose a different -BlockMB."
    }
    if ($fileSizeBytes -le $blockSizeBytes) {
        throw "File is not larger than one block; cannot do random reads. Increase -FileSizeMB or point -ExistingFile at a bigger file."
    }

    Write-Host ("Block size         : {0} MB ({1} bytes)" -f $BlockMB, $blockSizeBytes)
    Write-Host ("Queue depths       : {0}" -f ($QueueDepths -join ', '))
    Write-Host ("Reads per depth    : {0}" -f $ReadsPerDepth)
    Write-Host ("Runs per depth     : {0}" -f $Runs)
    Write-Host ("Link ceiling       : {0} GB/s ({1} MB/s decimal)" -f $LinkCeilingGBps, $LinkCeilingMBps)
    Write-Host ''

    $summaries = New-Object System.Collections.Generic.List[Object]
    $seedCounter = 12345

    foreach ($qd in $QueueDepths) {
        Write-Host ("--- Queue depth {0} ---" -f $qd)
        $mbpsRuns = New-Object System.Collections.Generic.List[double]
        $iopsRuns = New-Object System.Collections.Generic.List[double]

        for ($run = 1; $run -le $Runs; $run++) {
            $seedCounter += 101
            $result = [DiskBenchQD]::RunRound($filePath, $fileSizeBytes, $blockSizeBytes, $sectorSize, $qd, $ReadsPerDepth, $seedCounter)

            $elapsedSec = $result.ElapsedMs / 1000.0
            if ($elapsedSec -le 0) { $elapsedSec = 0.000001 }
            $mbps = ($result.BytesRead / $BytesPerMB_dec) / $elapsedSec
            $iops = $result.ReadsCompleted / $elapsedSec
            $pctOfLink = $mbps / $LinkCeilingMBps

            $mbpsRuns.Add($mbps)
            $iopsRuns.Add($iops)

            $rowLabel = ("QD{0} run {1}/{2}" -f $qd, $run, $Runs)
            Write-Host ("  {0}: {1:N1} MB/s, {2:N0} IOPS, {3:P1} of link ceiling, errors={4}" -f $rowLabel, $mbps, $iops, $pctOfLink, $result.Errors)
            Write-LinkWarningIfBroken -PctOfLink $pctOfLink -Label $rowLabel

            if ($result.Errors -gt 0) {
                Write-Host ("  WARNING: {0} read error(s) at QD{1} run {2}" -f $result.Errors, $qd, $run) -ForegroundColor Yellow
            }

            $row = [PSCustomObject]@{
                Timestamp    = (Get-Date -Format 'o')
                Path         = $filePath
                FileSizeMB   = [Math]::Round($fileSizeBytes / $BytesPerMiB, 2)
                BlockMB      = $BlockMB
                SectorSize   = $sectorSize
                QueueDepth   = $qd
                Run          = $run
                ElapsedMs    = [Math]::Round($result.ElapsedMs, 3)
                BytesRead    = $result.BytesRead
                ReadsDone    = $result.ReadsCompleted
                Errors       = $result.Errors
                MBps         = [Math]::Round($mbps, 2)
                IOPS         = [Math]::Round($iops, 1)
                PctOfLink    = [Math]::Round($pctOfLink, 4)
            }
            $allRows.Add($row) | Out-Null
        }

        $medianMbps = Get-Median -Values $mbpsRuns.ToArray()
        $minMbps = ($mbpsRuns | Measure-Object -Minimum).Minimum
        $maxMbps = ($mbpsRuns | Measure-Object -Maximum).Maximum
        $medianIops = Get-Median -Values $iopsRuns.ToArray()
        $medianPct = $medianMbps / $LinkCeilingMBps

        $summaries.Add([PSCustomObject]@{
            QueueDepth  = $qd
            MedianMBps  = [Math]::Round($medianMbps, 1)
            MinMBps     = [Math]::Round($minMbps, 1)
            MaxMBps     = [Math]::Round($maxMbps, 1)
            MedianIOPS  = [Math]::Round($medianIops, 0)
            PctOfLink   = [Math]::Round($medianPct * 100, 1)
        }) | Out-Null

        Write-Host ''
    }

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    $allRows | Export-Csv -Path $OutCsv -NoTypeInformation
    Write-Host ("Raw per-run results written to: {0}" -f $OutCsv)
    Write-Host ''

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    Write-Host '=== Summary (median / min / max across runs) ==='
    $summaries | Format-Table -AutoSize -Property `
        @{Label='QD'; Expression={$_.QueueDepth}}, `
        @{Label='Median MB/s'; Expression={$_.MedianMBps}}, `
        @{Label='Min MB/s'; Expression={$_.MinMBps}}, `
        @{Label='Max MB/s'; Expression={$_.MaxMBps}}, `
        @{Label='IOPS'; Expression={$_.MedianIOPS}}, `
        @{Label='% of link'; Expression={$_.PctOfLink}}

    foreach ($s in $summaries) {
        Write-LinkWarningIfBroken -PctOfLink ($s.PctOfLink / 100.0) -Label ("QD{0} median" -f $s.QueueDepth)
    }

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    $qd1Summary = $summaries | Where-Object { $_.QueueDepth -eq 1 } | Select-Object -First 1
    if (-not $qd1Summary) {
        $qd1Summary = $summaries | Sort-Object QueueDepth | Select-Object -First 1
    }
    $maxQdSummary = $summaries | Sort-Object QueueDepth -Descending | Select-Object -First 1

    Write-Host ''
    if ($qd1Summary -and $maxQdSummary -and $qd1Summary.MedianMBps -gt 0) {
        $ratio = $maxQdSummary.MedianMBps / $qd1Summary.MedianMBps
        $verdict = "QD{0} gives {1:N2}x over QD{2} ({3:N0} -> {4:N0} MB/s, {5:N1}% of link)" -f `
            $maxQdSummary.QueueDepth, $ratio, $qd1Summary.QueueDepth, $qd1Summary.MedianMBps, $maxQdSummary.MedianMBps, $maxQdSummary.PctOfLink

        Write-Host "=== VERDICT ==="
        Write-Host $verdict

        if ($ratio -ge 1.15) {
            Write-Host "Deep queues DO meaningfully help. Serialization at QD1 is costing real throughput."
        } elseif ($ratio -ge 1.05) {
            Write-Host "Deep queues help a little, but the gain is modest."
        } else {
            Write-Host "Deep queues do NOT meaningfully help here. The QD1 bottleneck is likely not queue-depth serialization."
        }
    } else {
        Write-Host "=== VERDICT === Could not compute QD1 vs max-QD comparison (missing data)."
    }

} finally {
    # SAFETY: nothing in this repo deletes files. This script used to remove its own test
    # file here. It no longer does - it reports the path so you can delete it yourself.
    if (Test-Path $filePath) {
        $sz = (Get-Item $filePath).Length / 1GB
        Write-Host ''
        Write-Host ("TEST FILE KEPT (this script does not delete anything):") -ForegroundColor Yellow
        Write-Host ("  {0}   ({1:N1} GB)" -f $filePath, $sz)
        Write-Host ("  delete it yourself with:  Remove-Item -LiteralPath '{0}' -Force" -f $filePath)
    }
}
