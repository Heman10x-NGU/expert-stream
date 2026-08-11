// expert-stream.h - stream MoE routed-expert weights from disk with real queue depth.
//
// LOCAL RESEARCH FORK ONLY. Not upstream, no PR intended.
//
// WHY THIS EXISTS
// When a MoE model is far larger than RAM, llama.cpp reaches its expert weights
// through the mmap of the GGUF shard. Every byte therefore arrives as a 4 KB page
// fault, and a page fault is by construction blocking and one-at-a-time. Measured
// on this machine, that behaves exactly like a queue-depth-1 reader:
//
//     mmap fault path                   ~330-469 MB/s
//     explicit unbuffered reads, QD 1    469 MB/s
//     explicit unbuffered reads, QD 2+   717-801 MB/s
//
// The gap is not alignment and not layout - both were measured and neither
// mattered. It is purely that nothing ever has more than one read outstanding.
//
// This module issues ALL of a layer's expert reads at once with
// FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, then hands ggml pointers into
// its own arena instead of into the mmap.
//
// WHAT IT DOES NOT DO (yet)
// It does not cache. Every expert is re-read every time it is used. That is
// deliberate: it isolates the queue-depth effect so the measurement has one
// cause. Caching is the next stage and is worth much more (88.9% of expert
// accesses over 220 tokens are repeats), but it introduces gigabytes of dirty
// private memory, which is a genuinely dangerous thing to add on a machine this
// size and deserves its own change.
//
// SAFETY POSTURE
// Every failure path falls back to the ordinary mmap pointer, which is always
// valid. This module can be disabled, mis-configured, or fail at runtime and the
// worst outcome is the original speed. It never returns a pointer it is not
// certain about.
//
// ENABLED ONLY when the environment variable V4F_EXPERT_MANIFEST names a readable
// manifest CSV. Unset, every function here is a cheap no-op.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

// True if streaming is configured and usable. Cheap after the first call.
bool ggml_expert_stream_active(void);

// True if this tensor is a routed-expert tensor the manifest knows how to address.
// Cheap; safe to call on every mul_mat_id.
bool ggml_expert_stream_is_expert(const struct ggml_tensor * t);

// Begin a fetch batch for one expert tensor. Must be called by a single thread.
void ggml_expert_stream_begin(const struct ggml_tensor * t);

// Queue expert `expert_id` for reading. Returns false if the arena is full, in
// which case the caller must fall back to the mmap pointer for that expert.
bool ggml_expert_stream_request(int64_t expert_id);

// Issue every queued read and wait for all of them. This is where queue depth
// comes from: N reads outstanding at once rather than N sequential faults.
void ggml_expert_stream_submit(void);

// Pointer to expert `expert_id` inside the arena, or NULL if it was not fetched
// (arena full, read failed, short read). NULL means "use src0->data as before".
const void * ggml_expert_stream_ptr(int64_t expert_id);

#ifdef __cplusplus
}
#endif
