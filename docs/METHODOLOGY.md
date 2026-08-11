# Methodology

How the measurements in [RESULTS.md](RESULTS.md) were taken, and the times the method caught the
project being wrong.

The corrections are the useful part. Anyone can publish the measurements that worked.

---

## The rules

1. **Read the artifact, don't reason about it.** Open the file, read its actual header.
2. **Benchmark at the real scale.** A test set smaller than the cache measures the cache.
3. **Design an experiment that can kill the idea**, not one that can confirm it.
4. **Always include a control** — something whose answer you already know.
5. **Check your instrument's resolution before believing a null result.**
6. **Interleave A/B arms.** Hardware drifts.
7. **CPU time is the honest liveness signal.** Elapsed time and memory usage both lie.
8. **Test the absurd end of the range.** Thresholds hide there.
9. **When a measurement disagrees with the plan, the plan is wrong.**

---

## The eight corrections

### 1. The disk benchmark was measuring the disk's own cache

Published **2.3 GB/s**. Re-measured against a working set large enough to reach steady state:
**1,050 MB/s**.

**A benchmark whose test data is smaller than the data you intend to read is measuring the wrong
device.** Every disk estimate in the project doubled and the headline projection came **down**
from ~5 to ~2 tok/s.

### 2. "Thermal throttling" was actually the benchmark's own writes

The erratic read spread was attributed to heat. It was garbage collection triggered by the
benchmark's own bulk writes:

| condition | median read | spread |
|---|---|---|
| drive idle | **1,050 MB/s** | 6% |
| right after writing and deleting ~63 GB | **318 MB/s** | 95–1,012 |

3.3x, persisting for minutes. This produced a design rule that existed in no earlier version of
the plan: **no bulk writes to the model's physical disk while inference runs** — including the KV
cache and the pagefile.

### 3. `n_embd` was inferred from file size

And it was wrong. Fixed by reading the GGUF header. **Rule 1 exists because of this.**

### 4. `--no-repack` "does not help"

It is the difference between output and no output: 3,664 MB of private memory and death at 20 s,
versus 1,400 MB and a completed run.

### 5. One decimal place nearly killed the project's best idea

Two cache sizes both printed **"0.2 t/s"**. A flat line — meaning cache size does not matter,
which would have killed N-2, the one genuinely novel idea here.

Then: **llama.cpp prints speed to one decimal place.** Both 0.24 and 0.16 print as "0.2". The
instrument **cannot resolve anything under ~25%**, and effects in that range were exactly what was
being looked for.

Fixed by measuring wall-clock differences instead. **Rule 5.**

### 6. The A/B benchmark was ordered wrongly

The first version ran all of layout A, then all of layout B — ninety seconds apart, on a drive
**already measured** to change behaviour over minutes. Any drift would have landed entirely on the
comparison and looked like a layout effect.

Fixed by interleaving. The published result uses 48 alternating samples. **Rule 6.**

### 7. A "median" that was a minimum

The summary code took the lower of the two middle values on an even-sized sample, and sorted two
columns independently — so a row could pair the throughput from one run with the timing from
another.

Caught during review, before a conclusion was drawn from it.

### 8. The design doc claimed an oracle that cannot exist

Three documents and a draft post said N-2's oracle comes from "running all 43 routers before
fetching anything… a real view of the future, not a guess."

**Layer L's router consumes layer L−1's output.** The routers are cheap but they are not
independent of the layers feeding them. Only layers 0–2, which use frozen hash routing, have an
exact forward oracle — **3 of 43**.

The error entered when the design was rewritten in plain English and the word "predicted" was
dropped. The original spec never made the mistake. **Simplifying prose can delete a load-bearing
qualifier.**

Two prediction-free versions of N-2 were then built in the simulator, and **both lost to the
policy they were meant to beat.** That cost one evening; in C it would have cost a week.

---

## Things that looked like bugs and were method problems

### The status line that said "WORKING" while the process was frozen

Our own monitor printed that the process was advancing, with a CPU-time delta of `+0.0s` **on the
same line**. The label and the data contradicted each other.

That same signal later caught a real deadlock. **CPU time is the honest liveness signal** — heavy
streaming and a hang look identical from the outside, since both show a large memory footprint and
no visible progress.

```powershell
$p = Get-Process llama-cli
$c1 = $p.CPU; Start-Sleep -Seconds 10; $p.Refresh()
"cpu delta: +{0:N1}s in 10s" -f ($p.CPU - $c1)
```

### Three empty log files in a row

Killed a run, got a zero-byte log. Twice more, once with a different logging method.

**Cause:** llama.cpp writes to stdout; the C runtime block-buffers that when it is a pipe rather
than a console; force-killing throws the buffer away. Adding a log-file option made it **worse**,
because it diverted output away from the console *and* still buffered it.

**If you kill a program, you did not get its last words.** Fixed by letting runs exit cleanly.

### 296 seconds of "work" that was a hang

A detached run reported 296 s. Almost all of it was the program sitting at an interactive chat
prompt waiting for keyboard input that could never arrive — a detached process has no keyboard.

Caught by zero CPU and zero page faults. Fixed with single-turn mode (`-st`).

### A component that was perfectly healthy and completely useless

The cache reported **0.0% hits** while every error counter read zero. Nothing was failing; it was
just below one lap.

**Instrument for effectiveness, not just for errors.**

---

## Test the absurd end of the range

The most surprising measurement in the project:

| cache | LRU hit rate |
|---|---|
| 1 GB | **0.0%** |
| 2 GB | 35.6% |

The "sensible" sizes are 4, 6 and 8 GB. Testing only those would have shown a smooth, boring curve
and the conclusion "cache size matters gradually."

Testing an unreasonably small size revealed a **cliff**, and that cliff is the single most
important constraint in the cache design:

> One lap through all 43 layers touches 1.649 GiB. **A cache smaller than one lap returns exactly
> zero**, because it evicts every entry immediately before it is needed.

We then walked straight into that cliff in our own code, and it cost a run to notice.

---

## Always include a control

**N-6** proposed keeping a resident low-rank sketch of every expert. Rank-64 retains **0.0948** of
the spectral energy — which sounds like something.

Then the control: **a random matrix of identical shape retains 0.0837.** 90% energy needs 64% of
the ranks; random needs 66%. **The experts are barely less random than noise.**

**Killed in about an hour, before a line of code was written.** Without the random-matrix
baseline, 0.0948 looks like a result.

The same discipline killed the cyclic eviction policy in an evening and the 76 GB repack on its
own benchmark.

---

## How the harnesses are built

The measurement scripts carry the safety posture, because a wrong measurement on this machine
costs an hour and a hung one can cost the desktop.

| Property | Why |
|---|---|
| **Resumable** | Answers already on disk are skipped. The quality run was killed twice and lost nothing both times |
| **Never overwrite** | `-Force` writes a **new timestamped file**. Superseded results are renamed, never deleted |
| **Save on kill** | Partial output is written before the process goes away |
| **Per-question timeout** | One runaway thinking chain cannot eat the whole run |
| **RAM watchdog** | Three **consecutive** samples below the floor, not one, so a brief dip does not end a 20-minute run |
| **Stop file** | Stops a detached run without hunting for a PID |
| **Forbidden-flag refusal** | `--mlock` and `--no-mmap` throw rather than run |
| **PID re-check before capping** | The working-set cap re-verifies `ProcessName` is `llama-cli` before touching the handle. It never enumerates processes |
| **Deletes nothing** | No script deletes a file or kills a process it did not launch |

Two PowerShell 5.1 details that caused real bugs:

- **`-f file` rather than `-p "text"`.** `Start-Process -ArgumentList` joins with spaces and does
  **not** quote, so a question containing spaces, quotes or newlines gets split into positional
  arguments and silently asks something else.
- **Conversation mode left on, plus `-st`.** Without conversation mode the GGUF's chat template is
  not applied; without `-st` llama-cli blocks on stdin forever.

---

## The rule this project keeps rediscovering

> **Idle CPU only buys anything when it changes *what you read*, and only at a granularity the
> SSD can actually skip.**

Spending idle cycles to make the arithmetic cheaper is worth **exactly nothing** here, because the
arithmetic already hides inside the read wait — compute is 4.5% of a token. Spending them to avoid
a fetch is worth everything, but the unit avoided has to be big enough that the drive notices.

**The smallest read this project has ever measured is ~1.6 MB.** Any claim about smaller reads, in
either direction, is currently unfounded — which is why the gate-first neuron-skipping idea
(worth up to 2.49x on paper) is recorded as *unmeasured* rather than as either a plan or a
rejection.
