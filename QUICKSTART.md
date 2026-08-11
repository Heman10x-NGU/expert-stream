# Quickstart — build it, run it, and check that it worked

Everything here has been run on the machine in the README. Follow it top to bottom.

**Total time: about 90 minutes**, almost all of it downloading 82.5 GB.

| Step | Time |
|---|---|
| [0. What you need](#0-what-you-need) | — |
| [1. Get the model](#1-get-the-model) | ~60–90 min on a fast line |
| [2. Build the patched llama.cpp](#2-build-the-patched-llamacpp) | ~10 min |
| [3. Build the expert manifest](#3-build-the-expert-manifest) | ~10 s |
| [4. Smoke test](#4-smoke-test-2-minutes) | ~2 min |
| [5. Chat with it](#5-chat-with-it) | — |
| [6. Reproduce the benchmarks](#6-reproduce-the-benchmarks) | 10 min – 3 h |

---

## 0. What you need

| | |
|---|---|
| **OS** | Windows 10/11. The reader uses `CreateFile` with `FILE_FLAG_NO_BUFFERING \| FILE_FLAG_OVERLAPPED` and the memory cap uses `SetProcessWorkingSetSizeEx`. Both are Win32-only today |
| **RAM** | 16 GB, with **at least ~11 GB actually free**. Close Chrome. It genuinely matters |
| **Disk** | ~90 GB free on an **NVMe SSD**. On a spinning disk this does not work at any speed |
| **Compiler** | [w64devkit](https://github.com/skeeto/w64devkit/releases) or MSVC, plus CMake |
| **Python** | 3.10+, standard library only. No pip install needed |
| **GPU** | Not required. Everything below is CPU-only |

**A spinning disk is not "slower", it is unusable.** The design assumes ~850 MB/s of random
reads at ~1.6 MB transfer sizes.

---

## 1. Get the model

`unsloth/DeepSeek-V4-Flash-0731-GGUF`, the **`UD-IQ1_S`** quantization — three GGUF shards,
82.5 GB total.

```powershell
pip install -U "huggingface_hub[cli]"

hf download unsloth/DeepSeek-V4-Flash-0731-GGUF `
  --include "UD-IQ1_S/*" `
  --local-dir E:\models\ds4f-iq1s
```

You should end up with:

```
E:\models\ds4f-iq1s\UD-IQ1_S\
    DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf
    DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf
    DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf
```

> **Put the model on a different physical disk from your pagefile if you can.**
> Measured on this hardware: bulk writes to the model's disk drop reads from **1,050 MB/s to
> 318 MB/s**, and it stays there for minutes afterwards. If Windows starts paging to the same
> drive the model streams from, everything collapses at once.

Set this once — every script below reads it:

```powershell
$env:EXPERT_STREAM_MODEL = "E:\models\ds4f-iq1s\UD-IQ1_S\DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf"
```

---

## 2. Build the patched llama.cpp

The engine is three commits on top of llama.cpp. `patches/` holds them as a `git am` series.

```powershell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 687e77892          # the base these patches were cut against

git am ..\expert-stream\patches\*.patch
```

If `git am` fails because upstream has moved, apply them loosely instead:

```powershell
git am --abort
git apply --3way ..\expert-stream\patches\*.patch
```

Then build. **CPU only** — see [docs/DESIGN.md](docs/DESIGN.md) for why there are no GPU kernels:

```powershell
$env:PATH = "C:\path\to\w64devkit\bin;" + $env:PATH

cmake -B build-cpu -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build-cpu --target llama-cli --config Release -j 8
```

`npm error` lines during the build are harmless — that is the optional web UI falling back to a
prebuilt copy.

```powershell
$env:EXPERT_STREAM_LLAMA_CLI = "$PWD\build-cpu\bin\llama-cli.exe"
```

> **If you rebuild later, make sure no `llama-cli.exe` is running first**, or the link step fails
> on a locked file.

### Which patch does what

| Patch | What it adds |
|---|---|
| `0001` | `moe-trace`, the tool that records which experts each token actually routes to, plus an env gate for mmap prefetch |
| `0002` | `expert-stream.c/.h` — the unbuffered, queue-depth reader. **This is the 124.0 s → 95.1 s** |
| `0003` | The bounded expert cache with exact-sized pools. **This is the 0.0% → 28.3% hit rate** |

Each one is independently useful and each is off unless its environment variable is set. With
`V4F_EXPERT_MANIFEST` unset, every function in the reader is a no-op and you get stock llama.cpp.

---

## 3. Build the expert manifest

The reader needs to know where every expert lives inside the shards — shard index, base offset,
per-expert stride, expert count. That table is derived once from the GGUF headers.

```powershell
cd expert-stream

python tools\gguf_meta.py            # reads the GGUF headers -> bench/results/tensor_index.json
python tools\make_expert_manifest.py # -> bench/results/expert_manifest.csv
```

Both are **read-only** — they open the model's headers and write two small files. Neither
touches the weights.

You should see 129 expert tensors across 3 shards. A prebuilt `expert_manifest.csv` for this
exact quantization is already committed in `bench/results/`; regenerate it if your model path
differs.

---

## 4. Smoke test (2 minutes)

```powershell
.\bench\run_first_output.ps1 `
  -PromptFile ".\bench\prompts\p1.txt" `
  -NPredict 8 `
  -CtxSize 512 `
  -WorkingSetCapMB 9000 `
  -ExpertManifest ".\bench\results\expert_manifest.csv" `
  -ExpertArenaMB 2600 `
  -Tag smoketest
```

### What you should see

```
HARD working-set cap 9000 MB applied to PID 1234.
expert streaming ENABLED via ...expert_manifest.csv
expert-stream: ACTIVE - 129 tensors, 3 shards, 2600 MB arena, 1188 slots in 4 pools,
               one token = 1691 MB (1.54 laps cached)

> The capital of France is
[Start thinking]
1.  The user is asking for the capital of France. This is a

expert-stream: 12522 reads, 26.65 GiB read | cache 4948/17478 = 28.3% hits, ... 0 failed
[ Prompt: 0.3 t/s | Generation: 0.3 t/s ]

wall time : 114.1 s
```

### How to tell it actually worked

Check these five lines. **They are the whole test.**

| Line | Must say | If it doesn't |
|---|---|---|
| `expert-stream: ACTIVE` | present | The manifest was not found — the reader silently fell back to stock mmap |
| `1.54 laps cached` | **above 1.0** | Below 1.0 the cache returns **exactly zero** hits. Raise `-ExpertArenaMB` |
| `0 failed` | exactly zero | A read error. Something is wrong with offsets or alignment |
| `= 28.3% hits` | non-zero | `0.0%` means the arena is under one lap |
| `wall time` | ~110–130 s | — |

**Two minutes is normal.** Most of it is loading.

> ### Why "laps cached" is the number to watch
> One token's expert working set is **1.649 GiB**, walked cyclically across 43 layers. A cache
> smaller than one lap evicts every entry *precisely* before its turn comes round again, and
> measures **0.0%** — reproduced on traces of 12, 54 and 220 tokens. Below ~1.65 GiB the memory
> is spent for nothing. This is the single sharpest constraint in the design.

### Prove the reader is doing something

Run the same thing without `-ExpertManifest`. That is stock llama.cpp, and it should take about
**124 s** instead of 95 s. Same binary, same prompt, same cap.

---

## 5. Chat with it

```powershell
.\chat.ps1
```

Wait about a minute for loading, then type. `/exit` to quit.

### Be realistic about the speed

**About 3 seconds per token.**

| Reply length | Wait |
|---|---|
| 20 tokens | ~1 minute |
| 100 tokens | ~5 minutes |
| 300 tokens (the default cap) | ~15 minutes |

This is a **reasoning** model — it emits a `[Start thinking]` chain before answering, and every
one of those tokens costs the same 3 seconds. That is why `-MaxTokensPerReply` exists.

> **Do not turn the thinking off to save time.** We measured what that costs. Answering instantly
> the model scores **12/15** on a fixed question set and claims Buzz Aldrin walked on Mars.
> Thinking first it scores **15/15** and correctly rejects the premise. The thinking is buying
> the right answers, at about 3x the wall time.

### Useful options

```powershell
.\chat.ps1 -MaxTokensPerReply 600     # longer answers, longer waits
.\chat.ps1 -CtxSize 8192              # longer memory. Cheap here: ~86 KB/token
.\chat.ps1 -ExpertArenaMB 3000 -WorkingSetCapMB 9500   # bigger cache, if you have the RAM
.\chat.ps1 -NoStreaming               # stock behaviour, for A/B
```

**The process stays alive between turns, so the expert cache stays warm.** Later turns in a
conversation are measurably faster than the first.

---

## 6. Reproduce the benchmarks

### Disk characterisation (~10 min)

```powershell
.\tools\diskbench-qd.ps1              # queue-depth sweep
.\bench\sweep_expert_reads.ps1 -QueueDepths 1,2,4,8 -Repeats 6
```

**Benchmark against a working set larger than your drive's cache**, or you are measuring the
cache. We published 2.3 GB/s that way, then re-measured properly and got 1,050 MB/s. Every
estimate in the project doubled.

### Capture real routing and simulate cache policies (~30 min)

```powershell
.\bench\launch_detached.ps1 -Runner run_moe_trace.ps1 `
  -RunArgs "-PromptFile bench\prompts\p_long.txt -CtxSize 512 -WorkingSetCapMB 6000 -Tag mytrace"

python tools\analyze_reuse.py
```

Prints hit rates for LRU, LFU, cyclic, guard and **Belady** across cache sizes. Belady is
optimal and therefore a ceiling on every policy that could ever be implemented — which is what
makes it useful as a target rather than a plan.

### Quality evaluation (~165 min)

```powershell
.\bench\run_quality_eval.ps1 -ExpertManifest ".\bench\results\expert_manifest.csv"
```

Fifteen questions, pass marks in `bench/prompts/quality/ANSWER_KEY.md` committed before the model
was asked anything.

**It is resumable.** Answers already on disk are skipped, so an interruption costs nothing —
which was tested the hard way, twice. `-Force` writes a **new timestamped file** and never
overwrites an existing answer.

### VRAM check (read-only)

```powershell
.\tools\vram_probe.ps1
```

Reports free VRAM against the 6.51 GiB resident set and prints the cheapest fixes first. It reads
and reports; it changes nothing.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `expert-stream: ACTIVE` never appears | Manifest not found | `python tools\make_expert_manifest.py` |
| `0.0% hits` | Cache is under one lap | Raise `-ExpertArenaMB` above 1800 |
| Warning about `laps cached` below 1.05 | Same | Same |
| Dies after 1–2 tokens | No working-set cap | Pass `-WorkingSetCapMB 9000` |
| `failed to fit params to free device memory` | Missing `-fit off` | Use the scripts — they pass it |
| `WATCHDOG: free RAM ... below floor` | Machine genuinely short | Close programs, or lower the cap |
| `ExpertArenaMB must be 32..6144` | Out of range | Use 2000–3000 |
| Build fails on a locked file | `llama-cli.exe` still running | Close it and rebuild |

### Telling "slow" from "stuck"

This matters, because heavy streaming and a hang **look identical from the outside** — both show
a large memory footprint and no visible progress.

```powershell
$p = Get-Process llama-cli
$c1 = $p.CPU; Start-Sleep -Seconds 10; $p.Refresh()
"cpu delta: +{0:N1}s in 10s" -f ($p.CPU - $c1)
```

- **More than a few seconds** → working normally, just slow
- **`+0.0s`** → genuinely stuck

**CPU time is the honest liveness signal.** Elapsed time and memory usage will both happily lie
to you. That exact check is what caught a real deadlock in this code.

### If you kill a process, you did not get its last words

We got three consecutive zero-byte logs before working this out. llama.cpp writes to stdout, the
C runtime block-buffers that when it is a pipe rather than a console, and a force-kill throws the
buffer away. Adding a log-file option made it **worse**. Let runs exit cleanly instead.

---

## The safety guards, and what each is for

On by default. You do not need to enable anything.

1. **The working-set cap** — the one that makes it work at all. Without it the run dies after two
   tokens. It refuses a cap that would leave the machine short, refuses anything below 3,500 MB,
   and warns-and-continues rather than aborting if the API call fails.
2. **The free-RAM watchdog** — checks every 2 s, stops the model after **three consecutive**
   readings below the floor. Three, not one, so a brief dip does not kill a run you have waited
   twenty minutes for. **It only ever stops the process it started itself.**
3. **`--mlock` and `--no-mmap` are hard-refused.** Either commits 82.5 GB of real memory on a
   15.4 GB machine. The scripts throw rather than run.
4. **A timeout on every measurement run.**
5. **A stop file** — `New-Item -ItemType File -Path .\STOP -Force` stops a detached run without
   hunting for a PID.
6. **Nothing deletes anything.** Superseded results are renamed with a timestamp, never removed.
   Anything that ought to go is printed for you to decide about.

---

## Turning it all off

Do not pass `-ExpertManifest`, or unset `V4F_EXPERT_MANIFEST`. The reader is inactive unless it
is given a manifest and costs nothing when off. That is the A/B baseline.
