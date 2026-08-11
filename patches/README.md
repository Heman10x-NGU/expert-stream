# patches/

Three commits against llama.cpp, as a `git am` series. **927 insertions, 1 deletion.**

Base commit: **`687e77892`**.

```powershell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 687e77892
git am ..\expert-stream\patches\*.patch
```

If upstream has moved and `git am` refuses:

```powershell
git am --abort
git apply --3way ..\expert-stream\patches\*.patch
```

Then build per [`../QUICKSTART.md`](../QUICKSTART.md) step 2.

## What each one does

| Patch | Adds | Worth |
|---|---|---|
| **0001** | `tools/moe-trace/` — records which experts each token actually routes to. Plus an env gate for mmap prefetch | Every routing and reuse number in `docs/RESULTS.md` comes from this |
| **0002** | `ggml/src/ggml-cpu/expert-stream.{c,h}` — the unbuffered, queue-depth reader, and the hook in `mul_mat_id` | **124.0 s → 95.1 s** |
| **0003** | The bounded expert cache with exact-sized pools | **0.0% → 28.3%** hit rate |

## They are independent, and all three are off by default

Each is gated on its own environment variable. With `V4F_EXPERT_MANIFEST` unset, every function in
the reader is a cheap no-op and the binary behaves exactly like stock llama.cpp.

**That is deliberate, and it is what makes the A/B honest** — the comparison in the README is the
same binary with one environment variable changed, not two different builds.

You can apply 0001 alone if you only want routing traces.

## Files touched

```
ggml/src/ggml-cpu/expert-stream.c   +531   new
ggml/src/ggml-cpu/expert-stream.h    +74   new
ggml/src/ggml-cpu/ggml-cpu.c         +32   hook in mul_mat_id
ggml/src/ggml-cpu/CMakeLists.txt      +2
src/llama-model.cpp                +5 -1
tools/moe-trace/moe-trace.cpp       +278   new
tools/moe-trace/CMakeLists.txt        +5   new
tools/CMakeLists.txt                  +1
```

**Nothing in upstream's hot path is modified** except the `mul_mat_id` hook, which is one branch
on `ggml_expert_stream_is_expert()` — false for every non-expert tensor and for every tensor at
all when the module is off.

## Upstream

**This is a research fork. No PR is intended.** The Windows-only I/O path alone would make it
unsuitable, and the design is still moving.
