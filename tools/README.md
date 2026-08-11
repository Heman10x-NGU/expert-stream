# tools/

| Tool | What it does | Task |
|---|---|---|
| **`guard.ps1`** | **Safety watchdog. Never run inference without it.** | — |
| `diskbench-qd.ps1` | Random reads at queue depth 1..32 | 0.5 |
| `disk-size-sweep.ps1` | **Bandwidth vs working-set size.** Proves small-file benchmarks measure the SLC cache | 0.5 |
| `disk-recovery-curve.ps1` | **Read bandwidth vs time since the last bulk write.** Read-only on the real model shard | 0.17 |
| `disk-soak.ps1` | Sustained read soak with per-window logging | 0.17 |
| `gguf_meta.py` | Model ground truth: real dims, per-layer bytes/expert, resident budget, full tensor index | 0.19 |
| `expert_spectrum.py` | Singular-value spectra vs a random-matrix control. **Killed N-6** | 0.18 |
| `hotlist_extract.py` | Parses ds4's shipped hotlist. **Forced a retraction** | 0.13 |
| `hash_routing.py` | Layers 0-2 routing from token IDs alone, zero forward passes | 0.15 (partial) |
| `compress_test.py` | Compression kill-or-confirm on raw quantized bytes | 0.14 |

## 🔴 Standing safety rules for everything in this repo

1. **No script deletes a file. Ever.** If a tool creates scratch files, it **prints their
   paths and sizes at the end** and you delete them. *One exception, documented:* `guard.ps1`
   removes the zero-byte `STOP` file after acting on it, because that file is a deliberate
   one-shot signal and leaving it would abort every subsequent run.
2. **No script kills a process you did not ask it to.** `guard.ps1` kills only the child it
   launched — that is its entire job. `unblock-admin.ps1` used to kill VRAM holders; it no
   longer does, because (a) `NVIDIA Overlay.exe` **respawns immediately** — measured, it
   gained 47 MiB and came straight back with a new PID — and (b) killing `msedgewebview2`
   can take down another app's UI and lose its state.
3. **Anything that changes the system is REPORT-ONLY by default** and requires an explicit
   `-Apply`, prints exactly what it would change first, and prints the undo command.
4. **Read-only on `E:\models`.** Every handle onto the model opens with `OPEN_EXISTING` and
   read access only.
5. **Never `--mlock` or `--no-mmap`.** Either commits 82.5 GB of real memory on a 15.4 GB
   machine. `run_baseline.ps1` hard-refuses both before it does anything else.

**Every measurement tool follows the same three rules:** report the **median of >=3 runs with
the spread**, never best-of; **divide by the 3,940 MB/s PCIe ceiling** and shout if the result
exceeds it (the benchmark is broken, not the drive); and **include a control** — a random
matrix, a random byte block, an independent-routing baseline — because a number without a
reference point cannot be interpreted.

Anything that opens the model uses `OPEN_EXISTING`, read-only. **No tool in this directory
ever writes to `E:\models`.**

---

# tools/guard.ps1 - safety watchdog

This machine has 15.42 GB of RAM and is being asked to run an 82.5 GB LLM.
That only works because most of the model stays memory-mapped / paged /
offloaded rather than fully resident, which means the system is running
close to the edge on purpose. `guard.ps1` exists so that if something goes
wrong (a bad allocation, a runaway context length, a leak), the machine
gets its resources back automatically instead of locking up hard enough to
need a power-cycle.

Run it in **Windows PowerShell 5.1** (`powershell.exe`), not PowerShell 7.

## Quick start

Supervise a run:

```powershell
powershell -NoProfile -File tools\guard.ps1 -Run "src\run_inference.ps1"
```

Just watch resources without launching anything (e.g. while you run
something manually in another window):

```powershell
powershell -NoProfile -File tools\guard.ps1 -DryRun
```

Use custom thresholds / poll interval:

```powershell
powershell -NoProfile -File tools\guard.ps1 -Run "src\run_inference.ps1" `
    -MinFreeRamMB 900 -MaxPagefileMB 5000 -MaxGpuTempC 85 -PollSeconds 2
```

## What it does

1. Prints a preflight block: current free RAM, free VRAM, pagefile usage,
   and the active thresholds. If free RAM is *already* below
   `-MinFreeRamMB` before anything is launched, it refuses to start the run
   at all (exit code 1) rather than launching straight into a bad
   situation.
2. Launches `-Run` as `powershell -NoProfile -File <Run>` and drops its
   process priority to `BelowNormal`, so the desktop / mouse / window
   manager stay responsive even if the workload pegs the CPU.
3. Every `-PollSeconds` (default 2), samples:
   - free physical RAM
   - pagefile (`D:\pagefile.sys`) current usage
   - GPU memory used/free and temperature (`nvidia-smi`)
   - the supervised process's working set and whether it's still alive
4. Appends every sample as a row to a CSV log (default
   `bench\results\guard-<timestamp>.csv`) and prints a one-line status to
   the console.
5. If a kill condition trips, it kills the supervised process (and all of
   its child processes) and exits with code 2. If the supervised process
   simply finishes on its own, the guard exits with code 0.

## Thresholds - what they mean and why

| Parameter | Default | Meaning |
|---|---|---|
| `-MinFreeRamMB` | 700 | Kill if free physical RAM stays below this. Windows itself, background services, and the desktop need some headroom; below ~700 MB free the OS starts fighting the workload for memory and things get sluggish or start failing to allocate. |
| `-MaxPagefileMB` | 6000 | Kill if pagefile usage stays above this. **This matters more than it looks like**: the pagefile (`D:\pagefile.sys`) and the model directory (`E:\models\ds4f-iq1s`) are on the **same physical disk**. Heavy paging doesn't just mean "low RAM" - it means Windows is hammering the exact disk the model is being streamed from, so paging and model I/O compete for the same spindle/controller bandwidth. Sustained high pagefile usage is often the first sign of a death spiral (page more -> model reads stall -> inference backs up -> more memory pressure -> page more). |
| `-MaxGpuTempC` | 90 | Kill if the GTX 1660 Ti's reported temperature stays above this. Protects the hardware during long unattended runs. |
| `-PollSeconds` | 2 | How often to sample. Also sets the effective reaction time: a sustained bad condition is caught after roughly `3 * PollSeconds` seconds (see below). |

### Why 3 consecutive samples, not 1

Every kill condition (RAM, pagefile, GPU temp) requires the **same**
condition to hold for **3 consecutive samples in a row** before the guard
acts. A single bad sample is treated as noise, not an emergency - a
momentary allocation burst, a GC pause, or the model shifting layers
between GPU and CPU can all cause a one-sample spike that resolves itself
immediately. If the guard killed on the first bad reading, it would abort
healthy runs constantly and become useless. Requiring 3 in a row (any good
sample resets the counter to zero) filters out that noise while still
reacting within about 6 seconds (at the default 2s poll interval) to a
real, sustained problem - fast enough to matter, slow enough to not be
trigger-happy.

## The STOP file - manual kill switch

At any time, create an empty file named `STOP` in the repo root:

```
D:\2025_Cursor_Dev\V4-local-serving\v4flash-16gb\STOP
```

For example:

```powershell
New-Item -ItemType File -Path "D:\2025_Cursor_Dev\V4-local-serving\v4flash-16gb\STOP" -Force
```

The guard checks for this file on every poll cycle. As soon as it sees it,
it deletes the file, kills the supervised process (and its children), logs
the kill, prints the summary, and exits with code 2. This is the manual
"panic button" - useful when you can see the machine struggling but the
automatic thresholds haven't tripped yet, or when you just want to stop a
run cleanly from another window without touching the guard's own console.

## Shutdown behavior

On any kill trigger (threshold, STOP file, or the guard itself being
interrupted with Ctrl+C), the guard:

1. Prints the reason loudly.
2. Tries a graceful `CloseMainWindow()` on the supervised process and waits
   up to 5 seconds.
3. If it's still alive, force-kills it and all of its child processes
   (found recursively via `Win32_Process.ParentProcessId`).
4. Writes a final CSV row with the `killed_reason` column filled in.
5. Prints a summary: duration, minimum free RAM seen, maximum pagefile
   usage seen, maximum GPU temperature seen, and whether it was killed.

The guard also wraps its whole poll loop in `try/finally`, so if you hit
Ctrl+C on the guard itself, the supervised process is still killed before
the guard exits. The one thing this script must never do is leave an 82 GB
inference process orphaned and unsupervised in the background.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Supervised process finished on its own (clean completion), or a `-DryRun` monitoring session was stopped without any threshold tripping. |
| 1 | Refused to launch (e.g. free RAM already below `-MinFreeRamMB` at startup, or `-Run` failed to start, or `-Run` was omitted without `-DryRun`). |
| 2 | The guard killed the supervised process (threshold breach, STOP file, or the guard itself was interrupted). |

## Log format

Each row in the CSV log has these columns:

```
timestamp, elapsed_sec, free_ram_mb, pagefile_mb, gpu_used_mb, gpu_free_mb,
gpu_temp_c, proc_alive, proc_ws_mb, killed_reason
```

`killed_reason` is empty on every row except the final row written when the
guard kills the process, which explains exactly why.
