# expert-stream

**A 284-billion-parameter model generating text on a 16 GB laptop.**

DeepSeek V4-Flash-0731, 82.5 GB on disk, running on a 2020 gaming laptop with 16 GB of DDR4 and
a 6 GB GTX 1660 Ti — by keeping the weights on an NVMe SSD and streaming only the experts each
token actually routes to.

**The published floor for this model is 96 GB of RAM.** The lowest run reported anywhere else is
48 GB, on a Mac, with unified memory. This machine has **15.4 GB usable, ~11.8 GB free.**

```
> The capital of France is

[Start thinking]

1.  The user is asking for the capital of France. This is a
```

| | measured |
|---|---|
| Model on disk | **82.5 GB** (`UD-IQ1_S`, 1.5625 bits/weight) |
| RAM available to it | **~11.8 GB** — 7.0x less than the model |
| Generation length | **unbounded** (stock llama.cpp managed 2 tokens, then died) |
| Bytes streamed per token | **1.649 GiB** |
| Expert accesses that are repeats, over 220 tokens | **88.9%** |
| **Answer quality at 1.5625 bpw** | **12/15 instantly, 15/15 when allowed to think** |

Same prompt, same working-set cap, same binary — **only the expert reader differs**:

| | wall | generation | expert bytes off disk |
|---|---|---|---|
| stock llama.cpp mmap | 124.0 s | 0.2 t/s | all of them |
| **+ streaming reader** | **95.1 s** | **0.3 t/s** | all of them |
| **+ 2.6 GB expert cache** | (n=16) **114.1 s** vs 126.6 s | 0.3 t/s | **28.5% fewer** |

**23% faster from queue depth alone, then 28.3% of expert accesses served from RAM** — with zero
failed reads and **byte-identical greedy output**. That last part is the one that matters: the
offsets, the sector arithmetic and the alignment are all correct, so everything else can be built
on top of them.

**It is still slow, and that is not the point yet.** The point is that every number here has a
machine and a method behind it, [including the ones that killed our own ideas](docs/METHODOLOGY.md).

- **[QUICKSTART.md](QUICKSTART.md)** — build it, run it, and reproduce the numbers above
- **[docs/DESIGN.md](docs/DESIGN.md)** — how the engine works
- **[docs/RESULTS.md](docs/RESULTS.md)** — every measurement, with its method
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — how the measurements were taken, and the
  four times they proved us wrong

---

## The problem, stated properly

A token routes to **1.649 GiB** of expert weights to produce **~700 KB of activations** —
**2,500 bytes read per useful byte**. The drive sustains **~1,200 MB/s** on reads at this
model's transfer sizes. Those two measured numbers multiply out to a wall:

```
1.649 GiB / 1200 MB/s  =  1.41 s per token  =  0.71 tok/s
```

with zero compute, perfect overlap and infinite queue depth. **No amount of I/O engineering beats
that, because it already assumes the I/O is perfect.**

So the ceiling is not a function of how fast you read. It is a function of how often you **avoid**
reading:

```
tok/s  ≈  0.5 / (1 − cache_hit_rate)
```

Which turns the whole project into one measurable quantity. On 220 tokens of real captured
routing, **88.9% of expert accesses are repeats** — so the bytes genuinely can be avoided, and a
6 GB cache with a good eviction policy is worth **2.02 tok/s** of headroom.

---

## What is actually new here

Most of this project is careful measurement rather than invention. Three things are not.

### 1. mmap is a queue-depth-1 reader, and that is most of the loss

llama.cpp reaches expert weights through the mmap of the GGUF shard, so every byte arrives as a
**4 KB page fault** — blocking, one at a time, by construction. During a run that moved gigabytes,
the process's `ReadTransferCount` read **0.01 GB cumulative at 0 MB/s**.

Measured against the same drive, replaying the real 774-reads-per-token access pattern:

| | QD 1 | QD 2 | QD 8 |
|---|---|---|---|
| real split layout | **469 MB/s** | 717 | 801 |
| repacked contiguous layout | 704 MB/s | 757 | 762 |

**Queue depth alone recovers 469 → ~800 MB/s for free.** The engine issues every expert read for
a layer at once with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and hands ggml pointers into
its own arena. That is the entire 124.0 s → 95.1 s.

The same table killed the planned 76 GB repack: worth ~5% once you have queue depth, and it costs
a 76 GB write on a drive with 111 GB free that we had already measured collapsing under bulk
writes.

### 2. LRU returns **exactly zero** cache hits, and that is structural

Simulating a bounded expert cache against real captured routing, in bytes:

| cache | LRU | Belady (optimal) | LRU tok/s | Belady tok/s |
|---|---|---|---|---|
| 1 GB | **0.0%** | 43.5% | 0.50 | 0.89 |
| 2 GB | 35.6% | 56.7% | 0.78 | 1.16 |
| 4 GB | 48.1% | 68.8% | 0.97 | 1.62 |
| 6 GB | 56.6% | **75.0%** | 1.16 | **2.02** |

Not "few" hits — **zero**, reproduced on traces of 12, 54 and 220 tokens.

Expert access is a fixed cyclic walk over 43 layers touching **1.649 GiB per lap**. When the lap
is longer than the cache, the least-recently-used entry is *precisely* the one whose turn comes
round next, so LRU evicts every entry immediately before it is needed. **This is what the OS page
cache is doing right now**, which is why the free memory this machine does have is currently worth
nothing.

The rule that falls out, and that the cache is built around:

> **A cache smaller than one lap returns exactly zero.** Below ~1.65 GiB it is memory spent for
> nothing.

### 3. The memory ceiling was a working-set problem, not an mmap problem

Stock llama.cpp produced one prompt token and one generated token, then died. The obvious reading
— "Windows caches faulted pages faster than it retires them" — was close but not actionable. The
sharper measurement:

```
process working set   11,845 MB   <- the pages are HERE
system Available         370 MB   <- so they are NOT on the reclaimable standby list
```

Pages on the standby list are free for the taking. Pages in a working set are not. One call —
`SetProcessWorkingSetSizeEx(..., QUOTA_LIMITS_HARDWS_MAX_ENABLE)` — forces continuous trimming:

| cap | working set | free RAM | result |
|---|---|---|---|
| none | grows to 11,845 MB | falls to 370 MB | dies at 30–56 s, 2 tokens |
| 8,000 MB | **pinned at 8,000 MB** | **flat at ~3,750 MB** | **runs to completion** |

**2 tokens → unbounded, with zero lines of engine code.** This is the single highest-leverage
finding in the repo and it is a Windows API call.

---

## Is a 1.5-bit model worth reading from?

Genuinely in doubt until it was measured. Fifteen questions with the pass marks committed
**before** the model was asked anything:

**12/15 answering immediately. 15/15 when allowed to think first.**

It reasons, writes and reads code, follows fussy formatting instructions, translates, and knows
that 1 divided by 0 is undefined.

Answering instantly it said Buzz Aldrin walked on Mars, and that Sydney is the capital of
Australia. Allowed to think, it gets both right — **so the knowledge is intact and the fast path
is what fails.** It grabs the nearest strong association. You buy correctness back with words, at
about **3x the wall time.**

That result has a consequence for the roadmap: two of the remaining speed ideas work by generating
fewer tokens, and the deliberation is exactly what buys the right answers. Both must be re-scored
before they ship.

Full harness, questions, answer key and raw output are in [`bench/`](bench/). It is resumable, so
an interruption costs nothing — which was tested the hard way, twice.

---

## What is deliberately NOT built

| Not doing | Why |
|---|---|
| A from-scratch forward pass | Compute is **4.5%** of the step. Rewriting attention and dequant kernels optimizes the 4.5% and yields a worse llama.cpp |
| GPU kernels | Same reason. Revisit only if disk time drops under ~130 ms |
| Wide draft trees | In MoE the expert union grows with tree width, so a wider tree can cost more disk than it saves |
| Blanket top-6 → top-4 | Changes the model's actual maths |
| PowerInfer-style neuron sparsity | Depends on ReLU producing real zeros. V4 uses SiLU |
| Buying 64 GB of RAM (~₹11,000) | It would beat all of this. The constraint is the point |

That last row is here because it would be dishonest to present clever engineering as the only
option when a cheap purchase beats it.

---

## Repository layout

```
engine/          expert-stream.c/.h  — the streaming reader and bounded cache
                 moe-trace.cpp       — the tool that captures real routing traces
patches/         the three-commit series against llama.cpp, ready to apply
src/             expert_read_bench.c — standalone replay of the real access pattern
bench/           the measurement harnesses, prompts, and every result CSV
tools/           trace analysis, cache simulation, disk characterisation, VRAM probe
docs/            design, results, methodology
chat.ps1         talk to it
```

## Hardware this was measured on

| | |
|---|---|
| CPU | Ryzen 7 4800H, 8C/16T Zen 2 |
| RAM | 16 GB DDR4-3200 — 15,789 MB usable, ~11.8 GB available |
| GPU | GTX 1660 Ti 6 GB, SM 7.5, no tensor cores |
| SSD | 1 TB DRAM-less NVMe, PCIe 3.0 x4 — **1,050 MB/s sustained**, 27% of link |
| OS | Windows 11 |
| Model | `unsloth/DeepSeek-V4-Flash-0731-GGUF` → `UD-IQ1_S`, 82.5 GB, 43 layers, 256 routed + 1 shared expert, top-6 |

## Safety — this workload can make a laptop unusable

The model is 82.5 GB and the machine has 15.4 GB.

- **Never `--mlock`, never `--no-mmap`.** Either commits 82.5 GB of real memory. Every script here
  refuses to pass them.
- **Every run is supervised** by a watchdog with a free-RAM floor and a manual stop file.
- **No script in this repo deletes a file or kills a process it did not launch.** Scratch files
  are listed for you to remove yourself.
- **No bulk writes to the model's physical disk while inference runs**, including the KV cache and
  the pagefile. Measured: reads collapse from 1,050 MB/s to 318 MB/s and stay there for minutes.
- **Do not run the machine to its last few hundred MB.** At ~350 MB free, Windows trims every
  working set and pages them to the same disk the model is streaming from.

## Credit

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and against
[antirez/ds4](https://github.com/antirez/ds4) (MIT), which implements the V4 architecture. Ideas
taken from **HOBBIT**, **PowerInfer**, **Fiddler** and **CompactifAI** — including the ones that
did not survive contact with a measurement.

This is a research fork. **No upstream PR is intended.**

## License

MIT. See [LICENSE](LICENSE).
