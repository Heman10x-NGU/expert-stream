# Results

Every number this project claims, with the method that produced it and the file it came from.
Nothing here is estimated unless it says so.

**Machine:** Ryzen 7 4800H (8C/16T Zen 2), 16 GB DDR4-3200 (15,789 MB usable, ~11.8 GB free),
GTX 1660 Ti 6 GB, 1 TB DRAM-less NVMe on PCIe 3.0 x4, Windows 11.
**Model:** `unsloth/DeepSeek-V4-Flash-0731-GGUF` → `UD-IQ1_S`, 82.5 GB, 43 layers, 256 routed +
1 shared expert, top-6.

---

## 1. Model geometry — read from the GGUF headers, not inferred

| | |
|---|---|
| Parameters | 284 B |
| File size | **82.5 GB** across 3 shards |
| Average bits/weight | **1.5625** |
| Layers | 43 |
| Routed experts / layer | 256, **top-6** |
| Shared experts / layer | 1 |
| Frozen hash-routed layers | **3** (layers 0–2) |
| `n_embd` | 4096 |
| `d_ff` (intermediate) | 2048 |
| `head_count_kv` | **1** |
| Expert tensors total | **129** |

Per-expert bytes at `UD-IQ1_S`:

| tensor | ggml shape | type | bytes | share |
|---|---|---|---|---|
| `ffn_gate_exps` | `[4096, 2048, 256]` | IQ1_S | 1,638,400 | 25.2% |
| `ffn_up_exps` | `[4096, 2048, 256]` | IQ1_S | 1,638,400 | 25.2% |
| `ffn_down_exps` | `[2048, 4096, 256]` | IQ3_XXS | 3,211,264 | 49.5% |
| | | **total** | **6,488,064** | |

**Method:** `tools/gguf_meta.py` reads the GGUF headers directly via llama.cpp's `gguf-py`.
Output: `bench/results/gguf_meta.txt`, `bench/results/tensor_index.json`.

> **This was got wrong once.** `n_embd` was originally inferred from file size rather than read
> from the header, and was wrong. **Read the artifact.**

### Resident set — the parts needed for every single token

| part | size |
|---|---|
| Attention weights | 4.75 GiB (**5.10 GB**) |
| Shared experts | 0.74 GiB |
| Everything else | ~1.02 GiB |
| **Total resident** | **6.51 GiB (6.99 GB)** |
| Expert bank (streamed) | **75.55 GB** |

**96.3% of attention's bytes are Q8_0 — 8.5 bits/weight**, while the experts run at 1.5625.
Retyping them to Q6_K would save **1,121 MiB**. Not done: `llama-quantize`'s `only_copy` guard
returns before per-tensor `--tensor-type` overrides are read, so it needs a patch.

---

## 2. Disk — the only bottleneck that matters

| condition | throughput |
|---|---|
| Sustained sequential read on the real model file | **1,050 MB/s** (27% of the PCIe 3.0 x4 link) |
| Random reads at this model's transfer sizes | **~850 MB/s** |
| Queue depth 1 | **469 MB/s** |
| Queue depth 2 | 717 MB/s |
| Queue depth 8 | **801 MB/s** |
| Right after writing and deleting ~63 GB | **318 MB/s**, spread 95–1,012 |

**Method:** `tools/diskbench-qd.ps1`, `tools/disk-size-sweep.ps1`, `tools/disk-soak.ps1`,
`tools/disk-recovery-curve.ps1`. Results in `bench/results/diskbench-qd-*.csv`,
`disk-size-sweep-*.csv`, `disk-soak-*.csv`, `disk-recovery-20260809-071506.csv`.

> **Two published numbers were wrong and are retracted here.**
>
> **2.3 GB/s → 1,050 MB/s.** The benchmark's test data was smaller than the drive's own cache, so
> it was measuring the cache. Every disk estimate in the project doubled and the headline
> projection came **down** from ~5 to ~2 tok/s.
>
> **"Thermal throttling" → bulk writes poison reads.** The erratic spread was blamed on heat. It
> was the benchmark's own writes putting the controller into garbage collection: 1,050 → 318 MB/s,
> **3.3x**, persisting for minutes. That produced a design rule present in no earlier plan: **no
> bulk writes to the model's physical disk during inference**, including the KV cache and the
> pagefile.

### The ceiling this implies

```
1.649 GiB per token / 850 MB/s = 1.99 s/token = 0.503 tok/s
```

with zero compute, perfect overlap and infinite queue depth. Therefore
**tok/s ≈ 0.5 / (1 − hit_rate)**.

### Layout: measured, then rejected

Replaying the real 774-reads-per-token pattern against the real shards. 48 samples, layouts
**interleaved** so drive drift lands on both arms equally.

| layout | QD 1 | QD 2 | QD 8 |
|---|---|---|---|
| real (split across 3 shards) | **469 MB/s** | 717 | 801 |
| repacked (one contiguous file) | 704 MB/s | 757 | 762 |

**Queue depth is worth 469 → ~800 MB/s for free; layout is worth ~5% after that.** The planned
76 GB repack was killed on this table.

**Alignment was never real:** all 129 expert tensors have strides divisible by both 512 and 4096,
so reading the enclosing aligned window wastes **0.022%**.

**Method:** `src/expert_read_bench.c` via `bench/sweep_expert_reads.ps1`.
Results: `bench/results/expert_read_bench.csv`, `align_test.txt`.

> **The first version of this benchmark was ordered wrongly** — all of layout A, then all of
> layout B, ninety seconds apart, on a drive already measured to drift over minutes. Any drift
> would have landed entirely on the comparison and looked like a layout effect. Fixed by
> interleaving.

---

## 3. Where the time goes

| part | share of a token |
|---|---|
| **Waiting for the SSD** | **85–95%** |
| Arithmetic | **~4.5%** |
| Everything else | the rest |

**This one table decided what was not built.** Doubling compute speed improves the whole thing by
about 2%.

---

## 4. Memory — the working-set cap

| cap | working set | free RAM | result |
|---|---|---|---|
| none | grows to **11,845 MB** | falls to **370 MB** | **dies at 30–56 s, 2 tokens** |
| 8,000 MB | **pinned at 8,000 MB** | **flat at ~3,750 MB** | **runs to completion** |

`SetProcessWorkingSetSizeEx` with `QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE`.

**2 tokens → unbounded generation, with zero lines of engine code.**

**Method:** `bench/sweep_wscap.ps1`. Results: `bench/results/wscap-4000mb-*.log`.

### `--no-repack`

| | private memory | outcome |
|---|---|---|
| repack on (default) | 3,664 MB | killed at 20 s, no output |
| `--no-repack` | **1,400 MB** | **completed** |

> **This project previously claimed `--no-repack` "does not help". Retracted.**

### KV cache is not the constraint

`head_count_kv = 1`, key/value length 512 → `43 × (512+512) × 2 B` ≈ **86 KB/token**. A
4,096-token conversation costs **~350 MB**. **Time is what limits context here, not memory.**

---

## 5. Routing and reuse — from real captured traces

**Method:** `llama-moe-trace` (patch 0001) records the experts each token actually routes to.
`bench/run_moe_trace.ps1`, analysed by `tools/analyze_routing.py` and `tools/analyze_reuse.py`.
Traces of 12, 54 and 220 tokens. Results: `bench/results/moe_trace-*.csv`, `routing_analysis.txt`,
`reuse_analysis.txt`, `overlap_curve.csv`.

| | measured |
|---|---|
| Bytes of expert weights per token | **1.649 GiB** |
| Activations produced per token | ~700 KB |
| **Read amplification** | **2,500 bytes per useful byte** |
| Expert reads per token | 774 |
| Expert accesses that are repeats, over 220 tokens | **88.9%** |
| Consecutive-token expert overlap | **21–39%, and flat** |
| Hash-routed layers with an exact forward oracle | **3 of 43** |

**Nearly all the value is in long-range recurrence, not adjacent-token agreement.** That argues
against a short lookahead window and for a real cache with a good eviction policy.

---

## 6. Cache simulation — every policy, against the optimum

Bytes-accurate simulation over the 220-token trace. **Belady is provably optimal and therefore a
ceiling on every policy that could ever be implemented.**

| policy | 1 GB | 2 GB | 4 GB | 6 GB |
|---|---|---|---|---|
| **LRU** (what runs) | **0.0%** | 35.6% | 48.1% | 56.6% |
| LFU | 21.9% | 31.2% | 43.6% | 52.6% |
| furthest-layer-away (cyclic) | 12.8% | 23.3% | 37.9% | 49.1% |
| LRU + protect imminent layers | 0.0% | 35.6% | 48.2% | 56.7% |
| **Belady (the ceiling)** | **43.5%** | **56.7%** | **68.8%** | **75.0%** |

In tok/s via `0.5 / (1 − hit)`:

| cache | LRU | Belady |
|---|---|---|
| 1 GB | 0.50 | 0.89 |
| 2 GB | 0.78 | 1.16 |
| 4 GB | 0.97 | 1.62 |
| 6 GB | 1.16 | **2.02** |

**Method:** `tools/analyze_reuse.py`.

### Three findings from this table

**LRU's 0.0% is structural, not noise.** Reproduced at 12, 54 and 220 tokens. One lap through
43 layers touches 1.649 GiB; below one lap, the least-recently-used entry is precisely the one
whose turn comes next. **A cache under ~1.65 GiB is memory spent for nothing.**

**Both prediction-free replacements for LRU lost.** "Furthest layer away" is an exact oracle
needing no prediction — layer order really is known — and it loses at every size above 1 GB,
because only ~36% of a layer's experts repeat token to token. "Protect imminent layers" is a
**no-op**: below one lap everything is imminent so it protects the whole cache; above one lap
there is no pathology left. **The bug and the fix do not overlap at any cache size.**

**Size beats cleverness by ~30x.** 2 GB → 6 GB is **+21 points**. No prediction-free policy change
moves the number by more than **0.6 points** at fixed size.

---

## 7. End-to-end — the headline

Same prompt, same working-set cap, same binary. **Only the expert reader differs.**

| | wall | generation | expert bytes off disk |
|---|---|---|---|
| stock llama.cpp mmap | 124.0 s | 0.2 t/s | all |
| **+ streaming reader** | **95.1 s** | **0.3 t/s** | all |
| **+ 2.6 GB cache** (n=16) | **114.1 s** vs 126.6 s | 0.3 t/s | **28.5% fewer** |

**Reader detail at 2,600 MB arena:** 129 tensors, 3 shards, **1,188 slots in 4 pools**, 1.54 laps
cached. 12,522 reads, 26.65 GiB read, **cache 4,948/17,478 = 28.3% hits**, **0 failed reads**.

**Output is byte-identical to stock llama.cpp under greedy decoding.** That is the claim that
matters most — the offsets, sector arithmetic and alignment are all correct.

**Method:** `bench/run_first_output.ps1`. Results: `bench/results/first_output-*.log`,
`bench/results/runs.csv`.

**Known defect:** one pool is undersized — two layers use a rare expert size and got 18 slots,
causing **37 fallbacks** to mmap. Recorded, not fixed.

> **A rounding artifact nearly killed the project's best idea.** Two cache sizes both printed
> "0.2 t/s" and looked like a flat line, which would have meant cache size does not matter. But
> llama.cpp prints speed to **one decimal place** — 0.24 and 0.16 both print as "0.2", so the
> instrument **cannot resolve anything under ~25%** and that is exactly the range being looked
> for. Fixed by measuring wall-clock differences instead. **Check your instrument's resolution
> before believing a null result.**

---

## 8. Output quality at 1.5625 bits

Fifteen questions across arithmetic, factual recall, multi-step reasoning, instruction-following,
code writing and reading, translation, coherence, self-knowledge and hallucination resistance.
**Pass marks and scoring bands were committed to `bench/prompts/quality/ANSWER_KEY.md` before the
model was asked anything.**

| | |
|---|---|
| **Answering immediately** | **12 / 15** |
| **Allowed to think first** | **15 / 15** |
| Cost of thinking | **~3x wall time** |

The three fast-path failures, and what they actually were:

| q | fast answer | thinking answer |
|---|---|---|
| q03 capital of Australia | **Sydney** | **Canberra** |
| q15 first person on Mars | **Buzz Aldrin** | correctly rejects the premise |
| q02 word problem | returned the cost | returned the change |

**These are not damaged knowledge.** In each case the fast path grabs the nearest strong
association — Aldrin for first-person-on-a-world, Sydney for Australian city, the cost instead of
the change. Allowed to deliberate, the model returns the right answer. **The knowledge is intact;
the retrieval path is what fails.**

Wall-time cost, measured: q02 **121 s → 392 s**, q03 **90.5 s → 235.5 s**.

**Method:** `bench/run_quality_eval.ps1`, ~165 minutes, resumable. Questions in
`bench/prompts/quality/`, raw output and graded answers in `bench/results/quality/`.

**Limits of this result, stated plainly:**

- The 15/15 rests on **three re-asked questions**, not a full re-run with reasoning on.
- q14 and the q15 retest were **truncated by the harness's own timeouts** (450 s / 500 s). The
  graded criteria were met in the visible output, but the runs did not finish.
- There is **no comparison against a less-compressed quant**, so "the quantization caused the
  fast-path failures" is a good inference, not a measurement.

> **The first analysis of this result was wrong and is retracted.** It concluded that 1.5 bits had
> destroyed fine factual distinctions, and predicted reasoning would **not** rescue q03 "because a
> missing fact cannot be recovered by reasoning about it". The retest returned **Canberra**
> immediately. The failure is a retrieval-path failure, not a storage failure.

**This has a consequence for the roadmap.** Two remaining speed ideas — skipping low-confidence
experts (N-3) and deadline decoding (N-7) — work by generating fewer tokens, and the deliberation
is exactly what buys the right answers. **A deadline that fires during thinking produces confident
wrong answers.** Both must be re-scored with this harness before they ship.

---

## 9. VRAM

| | |
|---|---|
| Resident set needing every token | **6.51 GiB** |
| Attention weights alone | **5.10 GB** |
| Free VRAM measured 2026-08-10 | **5,055 MiB** |
| **Shortfall** | **211 MiB** |

Free VRAM moves with whatever else is on screen, so this is **a reading, not a property of the
machine**. An earlier recorded figure of "4.46–4.84 GB free, misses by 300–600 MB" was a stale
snapshot.

**Two things make this much easier than it looks:**

1. **It was never all-or-nothing.** Attention is 43 layers of ~113 MiB and `-ngl` takes a layer
   count. `-ngl 40` closes the gap with room to spare, leaving ~340 MB in main memory instead of
   5 GB. **One flag, no firmware change.**
2. **Retyping attention Q8_0 → Q6_K saves 1,121 MiB** — 5.3x the gap — and it helps *even without
   the GPU*, because it shrinks the always-resident set and hands that memory straight to the
   cache. Blocked on a `llama-quantize` patch.

**Do not change the BIOS MUX setting for this.** 211 MiB is not worth the risk and there is a free
fix.

**Method:** `tools/vram_probe.ps1`, read-only. Results: `bench/results/vram_probe.csv`. Its CUDA
overhead figure is a **parameter**, `-CudaOverheadMB`, specifically so its verdict can never be
mistaken for a measured fact.

---

## 10. Ideas that were killed, with the measurement that killed each

| Idea | Killed by |
|---|---|
| **N-6**, a resident low-rank sketch of every expert | Rank-64 retains **0.0948** of the energy; a **random matrix of identical shape** retains **0.0837**. 90% energy needs 64% of the ranks; random needs 66%. The experts are barely less random than noise. **One hour, before any code.** |
| **76 GB expert repack** | Queue depth recovers 469 → ~800 MB/s free; layout is worth ~5% after that |
| **N-2 without a predictor** | Both prediction-free policies lose to plain LRU at every size above 1 GB |
| **PowerInfer neuron sparsity** | Needs ReLU's exact zeros; V4 uses SiLU |
| **N-8 gate-first neuron skipping** | 2.49x in principle, **1.25x as stored** — `ffn_down_exps` is `[2048, 4096, 256]`, so neuron-skipping is a gather of 0.38-byte fragments |
| **Saguaro-style fan-out** (hedge the read across F candidate next tokens) | The union across F candidates is the same size as the union across F *real consecutive* tokens (9.41 vs 9.29 experts/layer at 2), but hedging returns **one** token and batching returns **N**. Batching wins **2.03x at N=2, 7.70x at N=8** — the margin is exactly F |
| Four biological analogies | Each reduced to something already in the design once stated without the metaphor |

**The N-6 control is the point.** Without a random-matrix baseline, 0.0948 looks like a result.

**The fan-out kill has the same shape, one level up.** The measurement came back *positive* —
hedging 8 ways costs 20.68 experts per layer against 44.24 for random draws and 48.00 for
disjoint sets, so candidate futures genuinely do share most of their routing. It died because
the baseline was wrong: the comparison that matters is not "better than random", it is "better
than the boring option you already have".

**Method:** `tools/expert_spectrum.py`, `tools/compress_test.py`, `tools/analyze_fanout.py`.
Results: `bench/results/expert_spectrum.csv`, `compress_test.csv`, `fanout_analysis.txt`, `fanout_union.csv`.

---

## 11. What is NOT claimed

- **No claim that this is fast.** It is ~0.3 tok/s against a realistic ceiling of ~1.4.
- **No claim about Linux or macOS.** The reader is Windows-only and has not been ported.
- **No claim that 12/15 generalises.** Fifteen questions is a smoke test, not a benchmark.
- **No claim that quantization caused the fast-path failures.** No less-compressed quant was run.
- **No claim about other models or quantizations.** Every number here is `UD-IQ1_S` of this model
  on this machine.
- **No claim that N-2 works.** It is simulated, and the prediction-free versions of it failed.
- **No claim that the fan-out numbers generalise.** Two prefixes, eight substitutions each,
  prefill positions only. Enough to kill an idea, not enough to characterise the model.
