# expert-stream — final conclusion

**Model:** DeepSeek-V4-Flash-0731, 284B parameters, `UD-IQ1_S` (1.5625 bits per weight), 82.5 GB
on disk across 3 GGUF shards.
**Machine:** 2020 gaming laptop. 15.8 GB RAM (8–10 GB typically free), 6 GB GTX 1660 Ti, one
physical SSD holding both the model and the pagefile.
**Published floor for this model:** 96 GB.
**Date closed:** 2026-08-14.

---

## 1. The one number that governs everything

A token needs the top-6 of 256 experts in each of 43 layers. Every expert is 6,488,064 bytes.

```
1.649 GiB per token  /  ~1200 MB/s  =  1.41 s per token  =  0.71 tok/s   (at 0% hits)
```

Everything else in this repo is a consequence of that line. **The only two terms that move it
are bytes per token and bytes per second.** Any idea that improves something else — compute,
latency, elegance — improves nothing, and most of the graveyard in §4 is ideas that looked like
they attacked the ceiling and did not.

The corollary that shaped the whole design:

```
tok/s  ~=  0.71 / (1 - cache_hit_rate)
```

**The 1200 MB/s figure was settled on the last day and it corrects the whole repo.** For months
this was written as 850 MB/s and 0.503 tok/s. Both the 850 and a competing 1187 figure were
real; they are **different queue depths** — 863 MB/s at QD1, 1202 at QD4, 1260 at QD32 — and the
reader operates in the deep region. Older documents in this repo still say 0.503; they are
wrong by that factor and §3.2 of this file is the correction.

**And the ceiling is not what the engine achieves.** Measured marginal cost is ~3.7 s per token
against a 1.41 s expert-streaming cost. See §3.2 — that gap is the project's real open problem.

---

## 2. What exists and works

A fork of `llama.cpp` with **expert-stream**: a bounded, pool-allocated cache in front of the
routed experts, which reads them from the GGUF with batched overlapped unbuffered `ReadFile`
instead of taking 4 KB page faults one at a time. Around it: a working-set cap that forces
Windows to keep mmap pages reclaimable, a RAM watchdog, and a budget guard that refuses
configurations that cannot fit before loading 82.5 GB.

**Best measured configuration** (`-ngl 24` on `Vulkan0`, `-ot exps=CPU`, `-nkvo`, `-ub 64`,
working-set cap 6500 MB, arena 3400 MB):

| | measured |
|---|---|
| expert cache hit rate | **36.0%** over 32 tokens (25.2% over 8 — short runs understate, see 3.1) |
| fallbacks / failed reads | 21 / **0** |
| output vs CPU-only build | **byte-identical** |
| quality, 15-question smoke test | **12/15** default, **15/15** with thinking enabled |
| marginal cost per token | **~3.7 s** at arena 2400 |

The GPU is not used for arithmetic. **Compute is 4.5% of a token.** It is used as 6 GB of memory
that is not system RAM, which is the only scarce thing here. That is why Vulkan was chosen over
CUDA: `nvcc` on Windows needs MSVC, which this machine does not have, and the GPU's arithmetic
was never the point.

---

## 3. What the last day of measurement settled

### 3.1 The arena curve is real; the wall-clock payoff is not demonstrated

| arena | hit rate | fallbacks | GiB read | wall (8 tok) |
|---|---|---|---|---|
| 2400 MB | 19.4% | 37 | 21.34 | 117.0 s |
| 2900 MB | 23.3% | 29 | 20.34 | 112.1 s |
| 3400 MB | **25.2%** | 21 | 19.85 | 118.1 s |
| 3900 MB | refused by the reader's own budget guard | | | |

Those are cold 8-token runs. **At 32 tokens the same configurations give 28.2% and 36.0%** —
the cache is still warming at 32 tokens, so any hit rate from a short run is a floor, not a
measurement.

| arena | hit rate (32 tok) | GiB read | wall |
|---|---|---|---|
| 2400 MB | 28.2% | 47.42 | 206.7 s |
| 3400 MB | **36.0%** | 42.30 | **201.7 s** |

More arena buys more hits, monotonically, at both lengths. **It barely buys time**: bytes fell
10.8% and wall fell 2.4%. The +62% projection is **not confirmed**.

### 3.2 Bytes stopped being the binding constraint, and that is the real finding

The founding assumption of this project is that a token costs its expert bytes divided by disk
bandwidth. Two numbers from the 32-token runs say that is no longer true at the operating point
we reached:

- **Bytes removed from the critical path are worth ~1 GB/s.** 5.12 GiB saved bought 5.0 s. That
  is consistent with the 1,187 MB/s benchmark figure, not 850.
- **A token costs far more than its expert bytes.** Marginal cost from 8 to 32 tokens is
  **3.74 s per token**, carrying 1.087 GiB of expert reads — about 1.06 s at 1 GB/s.
  **Roughly 70% of a token is not expert streaming.**

**We optimised the term we could measure until it stopped being the biggest one, and did not
notice because the hit rate kept improving.**

**This is two runs, and it must be treated as a hypothesis.** The drive has measured anywhere
between 318 and 1,050 MB/s and a 2.4% difference is not separable from drift at n=1 per
configuration. **The correct next step is repeated runs with a per-phase timer — not another
optimization.** That is now the top item in §6.

### 3.2.1 A claim I had to withdraw the same day

I wrote that the engine "underperforms its own simulator by a wide margin" on the strength of
the cold 8-token runs. With the 32-token numbers the gap is about 10 points at 2.4 GB and 8 at
3.4 GB, still shrinking with length, against a simulator predicting ~38% and ~44%. **The gap is
real and unexplained; "wide margin" was wrong.** Left visible here rather than edited away.

### 3.3 Batching is the strategy that wins, and the tokens for it are not free

Union of experts across N consecutive tokens (learned layers 3-42): 6.00 → 9.29 → 14.22 → 21.49
at N=1/2/4/8, i.e. **2.69 experts per token at N=8 instead of 6** — a 55% cut in the only term
that matters. But autoregression does not hand you 8 tokens, and:

- **Hedging across candidate tokens loses to batching by a factor of N** (2.03x at N=2, 7.70x
  at N=8). Same read pattern, one token back instead of N.
- **A free n-gram drafter cannot supply them.** Every genre loses; the closest, a CSV, misses
  break-even by 4.8%.

So batched execution is the right answer and its input problem is unsolved.

---

## 4. The graveyard

Every one of these was killed by a measurement, and most were killed before any engine code was
written. **This is the actual output of the project.**

| Idea | Killed by |
|---|---|
| **N-6**, resident low-rank sketch of every expert | Rank-64 keeps 0.0948 of the energy; a **random matrix of the same shape** keeps 0.0837. The experts are barely less random than noise |
| **N-2 / N-1**, evict by predicted next use | A **perfect** 1-token-lookahead oracle captures **0%** of the LRU-to-Belady gap at every cache size. Anything visible one token ahead is also recent, and LRU keeps recent things |
| **N-8**, gate-first neuron skipping | 2.49x in principle; **0.16–0.55x measured** — 1 KB reads run at 36 MB/s against 1187 MB/s at 1.6 MB |
| **N-3**, drop the lowest-weight expert | Router weights are normalised and flat: smallest share is 9.45%, and a safe threshold saves **0.9%** of bytes |
| **76 GB expert repack** | Queue depth recovers 469 → ~800 MB/s for free; layout is worth ~5% after that |
| **PowerInfer neuron sparsity** | Needs ReLU's exact zeros. V4 uses SiLU |
| **Saguaro-style fan-out** (arXiv 2603.03251) | Union across F candidates equals union across F real tokens, but returns 1 token instead of N. **Batching wins by exactly F** |
| **n-gram drafting** | Every genre below break-even. 85.6% accuracy on the best case, but only 55% coverage |
| **Exact prefetch of layers 0-2** | Premise perfect — routing is a shipped token-id lookup table — but it saves **no bytes**, the queue is already past its measured peak depth, and the layers have no hot subset (Gini 0.23, all 768 slots used) |
| **Q6_K attention retype** | Superseded: the GPU already moved attention for 24 of 43 layers out of system RAM. Costs an 81 GB write that degrades read speed 3.3x for minutes |
| Four biological analogies | Each reduced to something already in the design once stated without the metaphor |

**Two of these were killed by a control, not by a result.** N-6 died because a random matrix of
the same shape scored almost as well. Fan-out died because the right baseline was batching, not
random. Without those controls both would have looked like wins.

---

## 5. What is not claimed

- **No claim that this is fast.** It is well under 1 tok/s.
- **No claim that the +62% arena projection holds.** It is a simulator output, the engine
  underperforms the simulator, and the wall-clock payoff did not appear at 8 tokens.
- **No claim that 12/15 generalises.** Fifteen questions is a smoke test.
- **No claim about Linux or macOS.** The reader is Windows-only.
- **No claim about other models or quantizations.** Every number is `UD-IQ1_S` of this model on
  this machine.
- **No claim that the fan-out or n-gram numbers characterise the model.** Two prefixes and four
  genres respectively — enough to kill ideas, not to describe behaviour.
- **Residency-aware drafting is an idea, not a result.** Unimplemented and unmeasured.

---

## 6. If this is resumed

In order:

1. **Per-phase timing, with repeats.** Where do the ~2.7 s per token that are not expert reads
   actually go — load, prefill, attention, KV, GPU sync, or serialization in the reader? Until
   that is known, every further optimization is aimed at a term worth 30% of a token. This is
   cheap and it outranks everything below it.
2. **Explain the simulator gap** (§3.2.1). ~10 points at 2.4 GB, unexplained.
3. **Expert-major batched execution.** Loop over experts on the outside and tokens on the
   inside, so one fetched expert serves every token in the batch that wanted it. This is the
   Flash Attention lesson — reorder the loop so every slow-tier load is fully consumed — and it
   is the only remaining idea that attacks bytes per token. Multi-day engine work.
4. **Residency-aware drafting.** Verification restores exactness, so bias the draft toward
   candidates whose experts are already resident. Two candidates differ on 3.41 of 6 experts per
   layer — 844 MiB — so there is something real to tie-break on.
5. **A clean-boot arena sweep.** Everything here ran at 65 hours of uptime with 8–10 GB free.

---

## 7. The method, which is the transferable part

- **Every idea gets the cheapest possible test first.** Simulation before C. Four ideas died in
  an evening with no engine code.
- **Every result gets a control.** A random matrix, a shuffled trace, a frozen-hash layer that
  should look random. Twice the control was the whole finding.
- **The baseline must be the boring option you already have**, not the naive one.
- **Free memory is a reading, not a property.** It changed the outcome of measurements three
  separate times, including one where sixteen consecutive runs failed while reporting success.
- **A harness that cannot distinguish a result from a broken setup will eventually hand you a
  flat line and let you believe it.**
