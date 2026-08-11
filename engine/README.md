# engine/

The source that makes this work, kept here as plain files so it can be read without cloning
llama.cpp. **To build, apply [`../patches/`](../patches/)** — these copies are for reading.

| File | Lines | What it is |
|---|---|---|
| `expert-stream.h` | 74 | The whole interface: six functions |
| `expert-stream.c` | 531 | The unbuffered queued reader and the bounded cache |
| `moe-trace.cpp` | 278 | Records which experts each token actually routes to |

Plus a 32-line hook in `ggml/src/ggml-cpu/ggml-cpu.c` inside `mul_mat_id`, and a 5-line change in
`src/llama-model.cpp`. **927 insertions total** against upstream.

## The interface

```c
bool         ggml_expert_stream_active(void);
bool         ggml_expert_stream_is_expert(const struct ggml_tensor * t);
void         ggml_expert_stream_begin(const struct ggml_tensor * t);
bool         ggml_expert_stream_request(int64_t expert_id);
void         ggml_expert_stream_submit(void);
const void * ggml_expert_stream_ptr(int64_t expert_id);
```

The call site is `mul_mat_id`: `begin` the tensor, `request` each expert the router picked,
`submit` once, then use `ptr(e)` in place of `src0->data + e*nb02`.

**`submit` is the whole speedup.** Every read for the layer is issued at once with
`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and waited on together — N outstanding reads
instead of N serialised page faults. mmap's fault path is a queue-depth-1 reader, and QD 1
measures 469 MB/s against QD 8's 801.

## Two invariants worth knowing before reading the code

**Every failure returns `NULL`, and `NULL` means "use the mmap pointer".** Arena full, read
failed, short read — all of them fall back to a pointer that is always valid. This module can be
disabled, misconfigured or fail at runtime and the worst outcome is the original speed. **It never
returns a pointer it is not certain about.**

**It is off unless `V4F_EXPERT_MANIFEST` names a readable manifest.** Unset, every function is a
cheap no-op and you have stock llama.cpp. That is what makes the A/B honest — same binary, one
environment variable.

## Environment variables

| Variable | Effect |
|---|---|
| `V4F_EXPERT_MANIFEST` | Path to the manifest CSV. **Unset = entirely disabled** |
| `V4F_EXPERT_ARENA_MB` | Cache size. **Below ~1800 the hit rate is 0.0%** — see below |

## The constraint the cache is built around

One token's expert working set is **1.649 GiB**, walked cyclically across 43 layers. A cache
smaller than one lap evicts every entry *precisely* before its turn comes round again and measures
**exactly 0.0%** hits — reproduced on traces of 12, 54 and 220 tokens.

The reader prints `laps cached` at startup and warns below 1.05. **Watch that number.** It is the
difference between a working cache and memory spent for nothing.

## Addressing an expert without repacking anything

The manifest records, per expert tensor: **shard index, base offset, per-expert stride, expert
count**. Then

```
expert e  =  base_offset + e * stride,  for stride bytes
```

guaranteed by the ggml layout, since the expert index is the slowest-moving dimension — which is
exactly why `mul_mat_id` can address it as `src0->data + cur_a * nb02`.

Unbuffered reads must be sector-aligned, so the reader reads from `floor(off/sector) * sector` and
skips `off % sector` bytes in the buffer. All 129 expert tensors have strides divisible by 512
**and** 4096, so the slop is **0.022%**.

**The GGUF files are opened read-only and never modified.**

## Platform

Windows-only today: `CreateFile` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`. The Linux
equivalent is `O_DIRECT` + `io_uring` — mechanical, but not done, so nothing is claimed about it.

## Status

**This is a research fork. No upstream PR is intended.**

Known defect: one cache pool is undersized — two layers use a rare expert size and got 18 slots,
causing 37 fallbacks to mmap. Recorded, not fixed.
