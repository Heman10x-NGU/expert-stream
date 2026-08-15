# expert-stream

**A 284B-parameter model generating text on a Windows laptop with 16 GB of RAM.**

DeepSeek V4-Flash-0731 is 82.5 GB on disk. This project runs the **UD-IQ1_S** quantization
on a 2020 gaming laptop by keeping expert weights on an NVMe SSD and streaming the experts
selected for each token.

The published memory floor for this model is 96 GB. This machine has 15,789 MB of usable RAM and
about 11.8 GB free.

**The greedy output is byte-identical to stock llama.cpp.**

It is still slow, well under 1 token per second. The point of this repo is to measure what makes a
284B Mixture-of-Experts model fit, and to show exactly where the design stops working.

## At a glance

| | measured |
|---|---|
| Model | DeepSeek V4-Flash-0731, **UD-IQ1_S** |
| Model on disk | 82.5 GB across 3 GGUF shards |
| Quantization | 1.5625 bits per weight |
| Machine RAM | 15,789 MB usable, about 11.8 GB free |
| Expert bytes per token | 1.649 GiB at 0% cache hits |
| Repeated expert accesses over 220 tokens | 88.9% |
| Read-speed term | about 1,200 MB/s at the reader's transfer sizes |
| I/O-only ceiling | 0.71 tok/s at 0% cache hits |
| Windows working-set result | 2 tokens without the cap, unbounded generation with the cap |
| Streaming reader A/B | 124.0 s to 95.1 s, with byte-identical output |
| Latest 32-token cache comparison | 36.0% hits, 10.8% fewer expert bytes, 2.4% lower wall time |

The 124.0 s to 95.1 s comparison uses the same prompt, cap and binary. Only the expert reader
changes. The older end-to-end cache comparison used a 2.6 GB cache and n=16 runs:

| | wall | generation | cache result |
|---|---|---|---|
| stock llama.cpp mmap | 124.0 s | 0.2 tok/s | all expert reads from disk |
| **streaming reader** | **95.1 s** | **0.3 tok/s** | all expert reads from disk |
| **streaming reader plus 2.6 GB cache** | 114.1 s versus 126.6 s | 0.3 tok/s | 28.3% cache hits, 28.5% fewer expert bytes |

The latest cache result is a separate engine sweep. More arena produced more hits, but only a small
wall-time change. That distinction matters.

The useful results are the measurements that explain the limit:

- [QUICKSTART.md](QUICKSTART.md), build it and reproduce the smoke test
- [docs/DESIGN.md](docs/DESIGN.md), how the engine works
- [docs/RESULTS.md](docs/RESULTS.md), every measurement and its method
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md), the corrections and the tests that killed ideas

## The one number that governs the design

A token selects the top 6 of 256 experts in each of 43 layers. Each expert is 6,488,064 bytes.
That means the token asks the disk for 1.649 GiB of expert weights to produce about 700 KB of
activations, about 2,500 bytes read per useful byte.

~~~text
1.649 GiB per token / about 1,200 MB/s = 1.41 s per token = 0.71 tok/s
~~~

That is an I/O-only ceiling at 0% cache hits, with zero compute, perfect overlap and infinite
queue depth. It is not the speed the engine achieves.

The simple I/O model is:

~~~text
tok/s ~= 0.71 / (1 - cache_hit_rate)
~~~

It is a model, not a benchmark. It ignores compute, synchronization and the rest of the engine.
The latest runs measured about 3.7 s of marginal cost per token while expert streaming accounts for
about 1.41 s. Roughly 70% of a token is now outside the measured expert-read path. That is the
open problem.

The disk numbers were corrected on 2026-08-14. Older notes used 850 MB/s and 0.503 tok/s. Those
figures came from different queue depths and are not the current ceiling. The correction is kept
visible in [docs/FINAL_CONCLUSION.md](docs/FINAL_CONCLUSION.md).

## What changed

### The working-set cap made generation fit

Stock llama.cpp produced two tokens and then died. The faulted pages were in the process working
set instead of the reclaimable standby list:

| | working set | free RAM | result |
|---|---|---|---|
| no cap | grows to 11,845 MB | falls to 370 MB | dies after 30 to 56 s, at 2 tokens |
| 8,000 MB cap | pinned at 8,000 MB | stays near 3,750 MB | runs to completion |

The cap uses Windows SetProcessWorkingSetSizeEx with hard working-set limits.

**2 tokens to unbounded generation, with zero lines of engine code.**

This is a Windows memory-management result, not a new inference kernel. The reader is Windows-only
today.

### The overlapped reader removed the queue-depth-1 limit

The mmap path reaches expert weights through page faults. The streaming path uses
FILE_FLAG_NO_BUFFERING and FILE_FLAG_OVERLAPPED, issues a layer's expert reads together, then
hands ggml pointers into its own arena.

The same real 774-read-per-token pattern was replayed against the model's split shards and a
repacked layout:

| layout | QD 1 | QD 2 | QD 8 |
|---|---:|---:|---:|
| real split layout | 469 MB/s | 717 MB/s | 801 MB/s |
| repacked contiguous layout | 704 MB/s | 757 MB/s | 762 MB/s |

Queue depth recovers 469 MB/s to about 800 MB/s without changing the model layout. Repacking the
76 GB of expert data was worth about 5% after that, so it was rejected. It also caused a large
write on a drive whose read speed had already been measured collapsing after bulk writes.

The end-to-end reader comparison is 124.0 s versus 95.1 s. Under greedy decoding, the output is
byte-identical to stock llama.cpp.

### The cache has a cliff

Real routing traces touch 1.649 GiB of experts per pass through the 43 layers. The cache simulation
used traces of 12, 54 and 220 tokens:

| cache size | LRU hit rate | Belady hit rate |
|---|---:|---:|
| 1 GB | 0.0% | 43.5% |
| 2 GB | 35.6% | 56.7% |
| 4 GB | 48.1% | 68.8% |
| 6 GB | 56.6% | 75.0% |

LRU returns exactly zero hits at 1 GB because the cache is smaller than one full pass. The next
expert needed is the one just evicted. Belady is an optimal offline policy, so its values are an
upper bound, not an engine result.

The 2.02 tok/s figure retained in docs/RESULTS.md is an I/O-only simulator output from the older
0.5 tok/s conversion. It is not a measured engine speed and is not the current 0.71 tok/s
zero-hit ceiling.

The engine sweep shows the current limit:

| arena | cache hits over 32 tokens | expert GiB read | wall |
|---|---:|---:|---:|
| 2,400 MB | 28.2% | 47.42 | 206.7 s |
| 3,400 MB | **36.0%** | **42.30** | **201.7 s** |

The larger arena removed 10.8% of expert bytes and reduced wall time by 2.4%. The result is based
on two runs, one per configuration, with drive drift still possible. The earlier +62% speed
projection is not confirmed.

The best measured engine result had 21 fallbacks and 0 failed reads. Its output was byte-identical
to the CPU-only build.

## What this does not claim

- It does not claim to be fast. It is well under 1 tok/s.
- It does not claim that the +62% arena projection holds.
- It does not claim that 12/15 quality generalizes. Fifteen questions is a smoke test.
- It does not claim support for Linux or macOS. The reader is Windows-only.
- It does not claim results for other models or quantizations. Every number here is for
  UD-IQ1_S of this model on this machine.
- It does not claim that the observed fast-path failures were caused by quantization. There is no
  less-compressed control.
- It does not claim that fan-out, n-gram drafting or prediction-free N-2 policies generalize.
  The tests were enough to reject designs, not to characterize model behavior.
- Residency-aware drafting is an idea, not a result. It is unimplemented and unmeasured.
- It does not claim that expert streaming is still the main bottleneck. The latest measurements
  leave roughly 70% of token time outside that path.

## Reproduce it

This is a Windows 10/11 project. The quickstart calls for 16 GB of RAM with at least about 11 GB
free, about 90 GB free on an NVMe SSD, CMake, a C compiler and Python 3.10 or newer. A GPU is not
required for the CPU path.

1. Follow [QUICKSTART.md](QUICKSTART.md) to download the three GGUF shards and build the patched
   llama.cpp base.
2. Generate the expert manifest with the read-only tools in tools/.
3. Run the streaming smoke test, then run chat.ps1 -NoStreaming for the stock A/B.
4. Use the scripts under bench/ to capture routing, replay reads and reproduce the cache and
   quality measurements. Results are written under bench/results/.

The exact command lines are kept in the quickstart because the model path and compiler setup are
machine-specific.

## Hardware used for every number above

| | |
|---|---|
| CPU | Ryzen 7 4800H, 8C/16T Zen 2 |
| RAM | 16 GB DDR4-3200, 15,789 MB usable, about 11.8 GB free |
| GPU | GTX 1660 Ti, 6 GB, SM 7.5, no tensor cores |
| SSD | 1 TB DRAM-less NVMe, PCIe 3.0 x4, 1,050 MB/s sustained, 27% of link |
| OS | Windows 11 |
| Model | unsloth/DeepSeek-V4-Flash-0731-GGUF, UD-IQ1_S, 82.5 GB, 43 layers, 256 routed plus 1 shared expert, top 6 |

## Quality at 1.5625 bits per weight

The quality run had 15 questions. The pass marks were committed before the model was asked
anything.

| | result |
|---|---|
| Answer immediately | 12/15 |
| Think first | 15/15 |
| Cost of thinking | about 3x wall time |

The three fast-path failures were:

| question | immediate answer | with thinking |
|---|---|---|
| capital of Australia | Sydney | Canberra |
| first person on Mars | Buzz Aldrin | correctly rejects the premise |
| word problem | returned the cost | returned the change |

The result is consistent with a retrieval-path failure, but it does not prove the cause. The 15/15
rests on three re-asked questions, not a full rerun with reasoning enabled. The q14 and q15 retests
were truncated by 450 s and 500 s test-runner timeouts, although the visible output met the grading
criteria.

The full prompts, answer key, raw output and grading files are in bench/.

## What is deliberately not built

| not built | reason |
|---|---|
| from-scratch forward pass | Compute is about 4.5% of a token. Rewriting it would target a small part of the current cost |
| GPU kernels | Same reason. Revisit if disk time drops below about 130 ms |
| wide draft trees | The expert union grows with tree width and can cost more disk than it saves |
| blanket top 6 to top 4 | It changes the model's maths |
| PowerInfer-style neuron sparsity | It depends on ReLU zeros. V4 uses SiLU |
| buying more RAM | It would beat this approach. The memory constraint is the point |

## Corrections and retractions

This project keeps its failed measurements and withdrawn claims in the open.

| earlier claim or method | correction |
|---|---|
| 2.3 GB/s disk speed | The test data fit the drive's own cache. The sustained characterization settled at 1,050 MB/s, while the expert reader reaches the deeper queue-depth region |
| 850 MB/s and 0.503 tok/s as the ceiling | Those figures were from another queue-depth regime. The current reader-region ceiling uses about 1,200 MB/s and 0.71 tok/s |
| thermal throttling caused the slow disk runs | Bulk writes, including the benchmark's own writes, poisoned reads. Read speed fell from 1,050 MB/s to 318 MB/s for minutes |
| --no-repack does not help | Retracted. The repacked run used 3,664 MB and died; --no-repack used 1,400 MB and completed |
| the simulator gap was a wide margin | Retracted. At 32 tokens the gap was about 10 points at 2.4 GB and 8 at 3.4 GB |
| the 1.5-bit quantization destroyed factual knowledge | Retracted. Thinking restored the tested answers. There is no less-compressed control, so the cause remains unproven |
| an inferred model dimension from file size | Retracted. The GGUF header is the source of truth |
| a benchmark comparison with all samples from one layout | Retracted. The A/B was redone with 48 alternating samples |
| a reported median | Retracted. The value was the minimum, and columns had been sorted independently |
| an oracle for all 43 routers | Retracted. Exact forward routing was available only for layers 0 to 2, with 3 of 43 layers identified in the trace analysis. Prediction-free N-2 policies lost to LRU |

The detailed methods and source files are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and
[docs/RESULTS.md](docs/RESULTS.md).

## The graveyard

Every row below was killed by a measurement or by a control. This is part of the result.

| idea | killed by |
|---|---|
| **N-6**, resident low-rank sketch of every expert | Rank-64 retains 0.0948 of the energy. A random matrix of the same shape retains 0.0837. The experts are barely less random than noise |
| **N-2 / N-1**, evict by predicted next use | A perfect 1-token-lookahead oracle captures 0% of the LRU-to-Belady gap at every cache size |
| **N-8**, gate-first neuron skipping | 2.49x in principle, 0.16-0.55x measured. 1 KB reads run at 36 MB/s against 1,187 MB/s at 1.6 MB |
| **N-3**, drop the lowest-weight expert | Router weights are flat. The smallest share is 9.45%, and a safe threshold saves 0.9% of bytes |
| **76 GB expert repack** | Queue depth recovers 469 MB/s to about 800 MB/s. Layout is worth about 5% after that |
| **PowerInfer neuron sparsity** | It needs ReLU's exact zeros. V4 uses SiLU |
| **Saguaro-style fan-out** | The union across F candidates equals the union across F real tokens, but fan-out returns 1 token while batching returns N |
| **n-gram drafting** | Best case reached 85.6% accuracy with 55% coverage, below break-even |
| **exact prefetch of layers 0 to 2** | It saves no bytes. The queue is already past its measured peak depth, and there is no hot subset, with Gini 0.23 across all 768 slots |
| **Q6_K attention retype** | The GPU already moved attention for 24 of 43 layers out of system RAM. The change costs an 81 GB write and degrades read speed 3.3x for minutes |
| four biological analogies | Each reduced to something already in the design once stated without the metaphor |

N-6 was killed by the random-matrix control, not by the raw rank-64 score. Fan-out was killed by
using batching as the baseline instead of random draws. Without those controls, both would have
looked like wins.

## Safety

This workload can make a laptop unusable. The model is 82.5 GB and the machine has 15,789 MB of
usable RAM.

- Never use --mlock or --no-mmap. Either commits 82.5 GB of real memory. The scripts refuse to
  pass them.
- Every run is supervised by a watchdog with a free-RAM floor and a manual stop file.
- No script in this repo deletes a file or kills a process it did not launch. Scratch files are
  listed for manual removal.
- Do not write bulk data to the model's physical disk while inference runs, including the KV cache
  and pagefile. Reads were measured falling from 1,050 MB/s to 318 MB/s and staying there for
  minutes.
- Do not run the machine to its last few hundred MB. At about 350 MB free, Windows trims working
  sets and pages them to the same disk the model is streaming from.

## Repository layout

~~~
engine/          expert-stream.c/.h, the streaming reader and bounded cache
                 moe-trace.cpp, the tool that captures real routing traces
patches/         the three-commit series against llama.cpp, ready to apply
src/             expert_read_bench.c, standalone replay of the real access pattern
bench/           the measurement scripts, prompts, and every result CSV
tools/           trace analysis, cache simulation, disk characterisation, VRAM probe
docs/            design, results, methodology
chat.ps1         talk to it
~~~

## Credit

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and against
[antirez/ds4](https://github.com/antirez/ds4) (MIT), which implements the V4 architecture. Ideas
taken from **HOBBIT**, **PowerInfer**, **Fiddler** and **CompactifAI**, including the ones that
did not survive contact with a measurement.

This is a research fork. **No upstream PR is intended.**

## License

MIT. See [LICENSE](LICENSE).
