# Design

How the engine works, and why each piece is the shape it is.

---

## The constraint that determines everything

DeepSeek V4-Flash has **43 layers**, each with **256 routed experts plus 1 shared expert**,
routing **top-6** per token. At `UD-IQ1_S` one expert's three tensors are:

| tensor | ggml shape (`ne0` contiguous) | type | bytes |
|---|---|---|---|
| `ffn_gate_exps` | `[4096, 2048, 256]` | IQ1_S | 1,638,400 |
| `ffn_up_exps` | `[4096, 2048, 256]` | IQ1_S | 1,638,400 |
| `ffn_down_exps` | `[2048, 4096, 256]` | IQ3_XXS | 3,211,264 |
| | | **per expert** | **6,488,064** |

Six of those per layer, 43 layers, plus shared experts and attention:

```
one token  =  1.649 GiB of expert weights read from disk
           →  ~700 KB of activations produced

           =  2,500 bytes read per useful byte
```

The drive sustains **~850 MB/s** on random reads at these transfer sizes. So:

```
1.649 GiB / 850 MB/s  =  1.99 s/token  =  0.50 tok/s
```

**That is a hard ceiling, and it assumes perfect I/O** — zero compute, perfect overlap, infinite
queue depth. It cannot be beaten by reading faster. It can only be beaten by not reading:

```
tok/s  ≈  0.5 / (1 − hit_rate)
```

Every design decision below follows from that single equation.

---

## Why mmap loses, specifically

llama.cpp reaches expert weights through the mmap of the GGUF shard. `mul_mat_id` addresses
expert `e` as `src0->data + e * nb02`, and touching that memory faults it in.

**A page fault is blocking and one-at-a-time by construction.** So every byte of a 6.5 MB expert
arrives as ~1,600 separate 4 KB faults, serialised. During a run that moved gigabytes, the
process's `ReadTransferCount` showed **0.01 GB cumulative at 0 MB/s** — the transfers were not
even attributed to the process, because they are fault traffic, not reads.

Replaying the real 774-reads-per-token pattern against the real shards, 48 samples, layouts
interleaved so drive drift lands on both arms equally:

| layout | QD 1 | QD 2 | QD 8 |
|---|---|---|---|
| real (split across 3 shards) | **469 MB/s** | 717 | 801 |
| repacked (one contiguous file) | 704 MB/s | 757 | 762 |

Two conclusions, and the second one killed a planned week of work:

1. **Queue depth is worth 469 → ~800 MB/s, for free.** mmap's fault path *is* a QD-1 reader, and
   QD 1 measures 469 MB/s. That is exactly where the observed 0.2 tok/s comes from.
2. **Layout is worth ~5% once you have queue depth** — which does not justify writing a 76 GB
   repacked file onto a drive with 111 GB free, on hardware already measured to collapse from
   1,050 to 318 MB/s during bulk writes.

Alignment turned out to be a non-issue too: all 129 expert tensors have strides divisible by both
512 and 4096, so an unbuffered reader can read the enclosing aligned window and skip the slop for
**0.022% waste**.

---

## The reader

`engine/expert-stream.c`. Six functions, and the whole interface is in `expert-stream.h`.

```c
bool         ggml_expert_stream_active(void);
bool         ggml_expert_stream_is_expert(const struct ggml_tensor * t);
void         ggml_expert_stream_begin(const struct ggml_tensor * t);
bool         ggml_expert_stream_request(int64_t expert_id);
void         ggml_expert_stream_submit(void);
const void * ggml_expert_stream_ptr(int64_t expert_id);
```

The hook in `ggml-cpu.c` sits in `mul_mat_id`: before the expert loop, `begin` the tensor,
`request` each expert the router picked, `submit` once, then take `ptr(e)` instead of
`src0->data + e*nb02`.

**`submit` is where the entire speedup lives.** Every read for the layer is issued with
`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and then waited on together — N outstanding reads
instead of N serialised faults.

### Addressing an expert without a repack

The manifest (`bench/results/expert_manifest.csv`, built by `tools/make_expert_manifest.py` from
the GGUF headers) records for each of the 129 expert tensors: **shard index, base file offset,
per-expert stride, expert count**. Then

```
expert e lives at   base_offset + e * stride   for stride bytes
```

which is guaranteed by the ggml layout — the expert index is the slowest-moving dimension, which
is precisely why `mul_mat_id` can address it as `src0->data + cur_a * nb02` in the first place.

Unbuffered reads must be sector-aligned, so the reader reads from
`floor(off/sector) * sector` and skips `off % sector` bytes in the buffer.

**No repack. The original GGUF files are never modified, and are opened read-only.**

### Failure posture

**Every failure path falls back to the ordinary mmap pointer, which is always valid.**

`ptr()` returns `NULL` for arena-full, read-failed or short-read, and `NULL` means "use
`src0->data` as before". The module can be disabled, misconfigured, or fail at runtime, and the
worst outcome is the original speed. It never returns a pointer it is not certain about.

It is **off unless `V4F_EXPERT_MANIFEST` names a readable manifest**. Unset, every function is a
cheap no-op. That is what makes the A/B honest: same binary, one environment variable.

---

## The cache

Added in patch `0003`. It is where the remaining value is, and it has one constraint that
dominates its whole design.

### A cache smaller than one lap returns exactly zero

Simulated against real captured routing, in bytes:

| cache | LRU | Belady (optimal) |
|---|---|---|
| 1 GB | **0.0%** | 43.5% |
| 2 GB | 35.6% | 56.7% |
| 4 GB | 48.1% | 68.8% |
| 6 GB | 56.6% | **75.0%** |

Not "few" hits — **zero**, reproduced on traces of 12, 54 and 220 tokens.

Expert access is a **fixed cyclic walk** over 43 layers touching 1.649 GiB per lap. When the lap
is longer than the cache, the least-recently-used entry is *precisely* the one whose turn comes
round next. LRU evicts every entry immediately before it is needed. It is not merely suboptimal —
it is the worst possible policy for this access pattern.

So the reader reports `laps cached` at startup and **warns below 1.05**. Any arena under
~1.65 GiB is memory spent for nothing. The default 2,600 MB is 1.54 laps.

**This is also what the OS page cache is doing to us right now**, and it is why the free RAM this
machine does have is currently worth so little.

### Exact-sized pools

Experts come in a small number of distinct sizes. A single free-list would fragment, so the arena
is split into pools of exactly-sized slots — 1,188 slots across 4 pools at 2,600 MB.

**One pool is still undersized**: two layers use a rare expert size and got only 18 slots, causing
37 fallbacks to mmap. Recorded, not yet fixed.

### Measured

**28.3% of expert accesses served from RAM**, 4,948 of 17,478, with **0 failed reads** and
**byte-identical greedy output** against stock llama.cpp.

---

## The working-set cap — not engine code, and the highest-leverage change in the project

Stock llama.cpp on this machine produced **one prompt token and one generated token**, then died.

The obvious diagnosis — "Windows caches faulted pages faster than it retires them" — was close but
not actionable. The measurement that made it actionable:

```
process working set   11,845 MB   <- the pages are HERE
system Available         370 MB   <- so they are NOT on the reclaimable standby list
```

**Pages on the standby list are free for the taking. Pages in a working set are not.** Windows
*can* trim, but the fault rate outran the trimmer. One call fixes it:

```c
SetProcessWorkingSetSizeEx(h, min, max,
    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE);
```

`HARDWS_MAX_ENABLE` makes the maximum a **hard** limit, forcing continuous trimming:

| cap | working set | free RAM | result |
|---|---|---|---|
| none | grows to 11,845 MB | falls to 370 MB | dies at 30–56 s, 2 tokens |
| 8,000 MB | **pinned at 8,000 MB** | **flat at ~3,750 MB** | **runs to completion** |

**2 tokens → unbounded, with zero lines of engine code.**

The scripts apply it to **exactly the PID they started**, after re-checking `ProcessName` is
`llama-cli`, and never enumerate or touch any other process.

---

## `--no-repack`, which is the difference between output and none

Same prompt, same context, same binary, one flag:

| | private memory | outcome |
|---|---|---|
| repack on (llama.cpp default) | 3,664 MB | killed at 20 s, no output |
| `--no-repack` | **1,400 MB** | **completed** |

The CPU backend repacks quantized weights into a SIMD-friendly layout at load, which converts
**evictable file-backed pages into dirty private memory**. That is a good trade when the model
fits in RAM and a fatal one when it does not.

> An earlier version of this project claimed `--no-repack` "does not help". That was wrong and it
> is retracted here. See [METHODOLOGY.md](METHODOLOGY.md).

---

## What is deliberately not built, and why

### No custom forward pass, no GPU kernels

**Compute is ~4.5% of a token.** Waiting on the SSD is 85–95%. Making the arithmetic twice as
fast improves the whole thing by about 2%, and would take weeks to produce a slightly worse
llama.cpp. Revisit only if disk time per token drops under ~130 ms.

This is the single most important number in the project for deciding what *not* to do.

### No neuron-level sparsity (PowerInfer-style)

PowerInfer skips loading weights for neurons whose activation will be zero. It depends on **ReLU
producing exact zeros**; V4 uses SiLU, which produces very small values instead of zeros.

There is a version that survives the threshold question — reading `gate` alone (25% of an expert)
exactly determines which intermediate neurons matter, because `SiLU(gate@x)` is a multiplicative
gate. It is worth up to **2.49x** in principle.

**It dies on memory layout.** `ffn_down_exps` is stored `[2048, 4096, 256]`, so one intermediate
neuron is a single element inside each of 4,096 separate rows, 784 bytes apart. Skipping neurons
there is a gather of 0.38-byte fragments. With only `ffn_up_exps` skippable it is worth **1.25x**,
and it needs two measurements first that have not been done.

### No wide draft trees

In MoE, the union of experts grows with tree width, so a wider speculative tree can cost **more**
disk than it saves. The usual speculative-decoding intuition inverts here.

### No blanket top-6 → top-4

That changes the model's actual maths rather than how it is fetched.

### No 76 GB repack

Killed by its own benchmark — see the QD table above.

---

## Roadmap, with what each item is worth

Every figure is an **I/O-only upper bound** from real routing traces. They ignore compute, so
delivered rates will be lower.

| | Change | Status | Basis |
|---|---|---|---|
| — | working-set cap | **done** | 2 tokens → unbounded |
| **A** | unbuffered `ReadFile` at queue depth, in place, no repack | **done — 124.0 s → 95.1 s** | 0 failed reads |
| **B** | bounded expert cache, LRU, exact-sized pools | **done — 28.3% hits** | 2.6 GB arena, 1.54 laps |
| **C** | more cache: move attention to the GPU to free ~5 GB | blocked by **211 MiB** of VRAM | 4 GB → 6 GB is +21 points |
| **D** | N-1: predict expert identity before fetching | not started | prerequisite for the below |
| **E** | N-2: prediction-driven eviction approaching Belady | **needs D first** | 56.6% → 75.0% simulated |
| **F** | N-3: skip the lowest-confidence expert block | not started | HOBBIT measured 0.99 correlation |

**Growing the cache beats every eviction policy we can currently implement, by about 30x.**
2 GB → 6 GB is worth **21 points** of hit rate with the policy already running. Every
prediction-free policy change tested moves the number by **0.6 points or less** at fixed size.

That is why C is ahead of E, and it is a measured result rather than a preference — two
prediction-free versions of N-2 were built in the simulator and **both lost to plain LRU**:

| policy | 1 GB | 2 GB | 4 GB | 6 GB |
|---|---|---|---|---|
| LRU (running) | 0.0% | 35.6% | 48.1% | 56.6% |
| LFU | 21.9% | 31.2% | 43.6% | 52.6% |
| furthest-layer-away | 12.8% | 23.3% | 37.9% | 49.1% |
| LRU + protect imminent | 0.0% | 35.6% | 48.2% | 56.7% |
| Belady (the ceiling) | 43.5% | 56.7% | 68.8% | 75.0% |

The third row is the instructive failure. Layer *order* is exactly known, so "evict whatever
belongs to the layer furthest away" is an oracle needing no prediction at all. It still loses,
because **only ~36% of a layer's experts repeat from one token to the next**. Knowing *when* a
layer recurs says almost nothing about *whether that expert* recurs — and knowing which experts
recur is Belady's entire advantage.

**So N-2 needs an expert-level predictor, not a comparator change.** An earlier version of this
design claimed the oracle came free from "running all 43 routers ahead of time". It does not:
layer L's router consumes layer L−1's output. Only layers 0–2, which use frozen hash routing, have
an exact forward oracle — 3 of 43.

---

## Portability

The reader is **Windows-only today**. Two dependencies:

| | Windows | Linux equivalent |
|---|---|---|
| unbuffered queued reads | `CreateFile(FILE_FLAG_NO_BUFFERING \| FILE_FLAG_OVERLAPPED)` | `O_DIRECT` + `io_uring` |
| forced working-set trimming | `SetProcessWorkingSetSizeEx(HARDWS_MAX_ENABLE)` | `cgroup v2 memory.high` |

Both have clean counterparts, so a Linux port is mechanical rather than a redesign. It has not
been done, so nothing here is claimed about it.
