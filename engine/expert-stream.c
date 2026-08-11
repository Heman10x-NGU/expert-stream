// expert-stream.c - see expert-stream.h for what this is and why.
//
// LOCAL RESEARCH FORK ONLY. Not upstream, no PR intended.

#include "expert-stream.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)

// Everything here is Win32 overlapped unbuffered I/O. On other platforms the
// module compiles to no-ops so the call sites need no #ifdef.
bool ggml_expert_stream_active(void) { return false; }
bool ggml_expert_stream_is_expert(const struct ggml_tensor * t) { (void) t; return false; }
void ggml_expert_stream_begin(const struct ggml_tensor * t) { (void) t; }
bool ggml_expert_stream_request(int64_t e) { (void) e; return false; }
void ggml_expert_stream_submit(void) { }
const void * ggml_expert_stream_ptr(int64_t e) { (void) e; return NULL; }

#else

#include <windows.h>

#define ES_MAX_SHARDS   8
#define ES_MAX_ROWS   512
#define ES_MAX_EXPERTS 1024   // slot_of[] is indexed by expert id
#define ES_SECTOR     4096    // E: is logical 512 / physical 4096; 4096 satisfies both

// ---------------------------------------------------------------------------
// manifest
// ---------------------------------------------------------------------------

struct es_row {
    char    name[80];      // reconstructed tensor name, e.g. blk.7.ffn_down_exps.weight
    int     shard_idx;
    int64_t base_offset;
    int64_t stride;
    int     n_expert;
};

struct es_shard {
    char   path[MAX_PATH];
    HANDLE h;
    int64_t size;
};

struct es_slot {
    OVERLAPPED     ov;
    unsigned char *buf;
    size_t         cap;      // bytes this slot can hold
    HANDLE         h;        // handle the read was issued on; completions are unordered
    int64_t        slop;     // offset of the wanted data inside buf
    int64_t        want;     // useful bytes
    DWORD          len;      // aligned bytes submitted
    int            row_idx;  // which manifest row (tensor) this slot holds, -1 if none
    int64_t        expert;   // which expert of that tensor
    int            valid;    // holds complete, usable data
    int            pending;  // a read is in flight into this slot
    uint64_t       tick;     // LRU recency
    uint64_t       gen;      // last fetch batch that referenced it (eviction guard)
};

static struct es_shard g_shard[ES_MAX_SHARDS];
static int             g_nshard;
static struct es_row   g_row[ES_MAX_ROWS];
static int             g_nrow;

static int             g_init;      // 0 not tried, 1 active, -1 disabled
static struct es_slot *g_slot;
static int             g_nslot;     // total slots
// ONE POOL PER DISTINCT STRIDE. THIS IS NOT OVER-ENGINEERING - IT IS THE
// DIFFERENCE BETWEEN A CACHE THAT WORKS AND ONE THAT RETURNS EXACTLY ZERO.
//
// One token walks 43 layers x 6 experts x {gate, up, down} = 774 slices totalling
// 1.649 GiB, in a fixed cyclic order. A cache smaller than one lap evicts every
// entry just before its turn comes round again, so it returns 0.0% hits - this
// is measured, not theoretical, at both 1 GB and (as below) at 2 GB.
//
// This model has four distinct strides: 1,638,400 / 2,162,688 (gate, up) and
// 3,211,264 / 4,456,448 (down). Sizing every slot for the largest wastes ~50%.
// A first attempt used two pools, small and large, which cut the waste to 26-28%
// - and still failed, because one lap then costs 2,316 MB of slots and a 2,048 MB
// arena was therefore still under one lap. It measured 0/17478 = 0.0% hits.
//
// Exact-sized pools drop the padding to the sector rounding alone (~0.25%), so a
// lap costs ~1,689 MB and an arena only has to beat that.
//
// Pools are also sized IN PROPORTION to how many bytes per token each stride
// accounts for, so the pools all run out at the same time. A pool that is
// relatively too small throttles the whole cache to its own capacity.
#define ES_MAX_POOLS 8
struct es_pool {
    int64_t stride;      // exact per-expert stride this pool serves
    size_t  slot_bytes;  // stride + sector slop, rounded to 4096
    int     first;       // index of its first slot in g_slot
    int     count;
};
static struct es_pool g_pool[ES_MAX_POOLS];
static int            g_npool;
static unsigned char *g_arena[ES_MAX_POOLS];

// Per-batch state. Only ever touched by the single thread that calls begin/
// request/submit, and only read by other threads after a barrier.
static const struct es_row *g_cur_row;
static int                  g_cur_row_idx = -1;
static int32_t              g_slot_of[ES_MAX_EXPERTS];
static int                  g_issued[ES_MAX_EXPERTS];
static int                  g_pending;
static uint64_t             g_tick, g_gen;

// stats
static int64_t g_stat_reads, g_stat_bytes, g_stat_fallback, g_stat_failed;
static int64_t g_stat_hits, g_stat_misses, g_stat_hit_bytes;

static void es_report(void) {
    if (g_init != 1) {
        return;
    }
    const int64_t tot = g_stat_hits + g_stat_misses;
    fprintf(stderr,
            "expert-stream: %lld reads, %.2f GiB read | cache %lld/%lld = %.1f%% hits, "
            "%.2f GiB served from RAM | %lld fallbacks, %lld failed\n",
            (long long) g_stat_reads, g_stat_bytes / 1073741824.0,
            (long long) g_stat_hits, (long long) tot,
            tot ? 100.0 * (double) g_stat_hits / (double) tot : 0.0,
            g_stat_hit_bytes / 1073741824.0,
            (long long) g_stat_fallback, (long long) g_stat_failed);
}

static void es_disable(const char * why) {
    // Disabling is always safe: every caller falls back to src0->data, which is
    // the mmap pointer that worked before this module existed.
    fprintf(stderr, "expert-stream: DISABLED (%s) - falling back to mmap\n", why);
    g_init = -1;
}

static int64_t es_round_dn(int64_t v, int64_t a) { return v - (v % a); }
static int64_t es_round_up(int64_t v, int64_t a) { return ((v + a - 1) / a) * a; }

// THREAD SAFETY - THIS CAUSED A REAL DEADLOCK, DO NOT SIMPLIFY.
//
// ggml_compute_forward_mul_mat_id runs on every thread in the pool, and the call
// site uses ggml_expert_stream_is_expert() to decide whether to execute an extra
// ggml_barrier(). Control flow around a barrier MUST be identical on every
// thread: if one thread takes the barrier and another skips it, the first waits
// forever for a peer that has already moved on.
//
// The first version initialised lazily with a plain `if (g_init != 0) return;`
// and set g_init = -1 pessimistically on entry. Threads that called during that
// window saw -1, answered "not an expert", and skipped the barrier while the
// initialising thread took it. Symptom: the whole process at 0.0 s of CPU with
// a 6 GB working set, looking exactly like slow I/O rather than a hang.
//
// InitOnceExecuteOnce makes every thread block until initialisation is complete,
// so all of them observe the same final g_init and therefore the same branch.
static INIT_ONCE g_once = INIT_ONCE_STATIC_INIT;

static void es_init_locked(void) {
    g_init = -1;   // pessimistic until everything below succeeds

    const char * manifest = getenv("V4F_EXPERT_MANIFEST");
    if (!manifest || !*manifest) {
        return;    // not configured: silent, this is the normal case upstream
    }

    FILE * f = fopen(manifest, "r");
    if (!f) {
        fprintf(stderr, "expert-stream: cannot open V4F_EXPERT_MANIFEST '%s'\n", manifest);
        return;
    }

    char line[1024];
    int seen_header = 0;
    g_nshard = 0;
    g_nrow   = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            int idx; char p[MAX_PATH];
            // Field width is mandatory: an unbounded %[^\n] would overrun p.
            if (sscanf(line, "#S %d %259[^\n\r]", &idx, p) == 2) {
                if (idx < 0 || idx >= ES_MAX_SHARDS) { fclose(f); es_disable("shard index out of range"); return; }
                size_t plen = strlen(p);
                if (plen >= MAX_PATH)               { fclose(f); es_disable("shard path too long");     return; }
                memcpy(g_shard[idx].path, p, plen + 1);
                if (idx + 1 > g_nshard) g_nshard = idx + 1;
            }
            continue;
        }
        if (!seen_header) { seen_header = 1; continue; }
        if (line[0] == '\n' || line[0] == '\r' || line[0] == 0) continue;

        int layer, shard_idx, n_expert;
        char kind[32];
        long long base, stride;
        if (sscanf(line, "%d,%31[^,],%d,%lld,%lld,%d",
                   &layer, kind, &shard_idx, &base, &stride, &n_expert) != 6) {
            fclose(f); es_disable("malformed manifest row"); return;
        }
        if (g_nrow >= ES_MAX_ROWS)                       { fclose(f); es_disable("too many manifest rows");   return; }
        if (shard_idx < 0 || shard_idx >= ES_MAX_SHARDS) { fclose(f); es_disable("bad shard_idx");            return; }
        if (stride <= 0 || n_expert <= 0)                { fclose(f); es_disable("bad stride/n_expert");      return; }
        if (n_expert > ES_MAX_EXPERTS)                   { fclose(f); es_disable("n_expert exceeds slot_of"); return; }
        // The stride MUST be a whole number of sectors, otherwise neighbouring
        // experts share a sector and cannot be addressed independently. This is
        // measured to hold for this model; assert it rather than assume it.
        if (stride % ES_SECTOR != 0)                     { fclose(f); es_disable("stride not sector-aligned"); return; }

        struct es_row * r = &g_row[g_nrow];
        snprintf(r->name, sizeof r->name, "blk.%d.ffn_%s_exps.weight", layer, kind);
        r->shard_idx   = shard_idx;
        r->base_offset = (int64_t) base;
        r->stride      = (int64_t) stride;
        r->n_expert    = n_expert;
        g_nrow++;
    }
    fclose(f);

    if (g_nshard == 0 || g_nrow == 0) { es_disable("manifest declared no shards or no rows"); return; }

    // ---- open shards, read-only, unbuffered ----
    for (int i = 0; i < g_nshard; i++) {
        if (!g_shard[i].path[0]) { es_disable("manifest skipped a shard index"); return; }
        g_shard[i].h = CreateFileA(g_shard[i].path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                   OPEN_EXISTING,
                                   FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                                   NULL);
        if (g_shard[i].h == INVALID_HANDLE_VALUE) { es_disable("CreateFile failed on a shard"); return; }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(g_shard[i].h, &sz))    { es_disable("GetFileSizeEx failed");         return; }
        g_shard[i].size = sz.QuadPart;
    }

    // ---- one pool per distinct stride ----
    g_npool = 0;
    for (int i = 0; i < g_nrow; i++) {
        int found = 0;
        for (int p = 0; p < g_npool; p++) if (g_pool[p].stride == g_row[i].stride) { found = 1; break; }
        if (found) continue;
        if (g_npool >= ES_MAX_POOLS) { es_disable("more distinct strides than pools"); return; }
        g_pool[g_npool].stride     = g_row[i].stride;
        // Rounded to the sector, NOT to 64K: 64K rounding reintroduces the very
        // padding this split exists to remove.
        g_pool[g_npool].slot_bytes = (size_t) es_round_up(g_row[i].stride + ES_SECTOR, ES_SECTOR);
        g_npool++;
    }

    // Bytes each stride contributes to one token, used to size the pools in
    // proportion so they all fill up together.
    int64_t lap_bytes = 0;
    int64_t pool_lap[ES_MAX_POOLS];
    for (int p = 0; p < g_npool; p++) pool_lap[p] = 0;
    for (int i = 0; i < g_nrow; i++) {
        for (int p = 0; p < g_npool; p++) {
            if (g_pool[p].stride == g_row[i].stride) {
                pool_lap[p] += 6 * g_pool[p].slot_bytes;   // 6 experts per tensor per token
                lap_bytes   += 6 * g_pool[p].slot_bytes;
                break;
            }
        }
    }

    int arena_mb = 192;
    const char * am = getenv("V4F_EXPERT_ARENA_MB");
    if (am && *am) {
        arena_mb = atoi(am);
        // An arena big enough to matter is also big enough to hurt: this is
        // private committed memory on a machine where the model is 5x RAM, and
        // unlike the file-backed mmap pages it replaces, every page of it is
        // dirty and anonymous - trimming it means writing to the pagefile on the
        // same physical disk the model streams from.
        if (arena_mb < 32 || arena_mb > 6144) { es_disable("V4F_EXPERT_ARENA_MB outside 32..6144"); return; }
    }

    // Refuse to commit an arena that does not leave the machine room to breathe.
    // Falling back to mmap is slow; driving the box into the pagefile is worse,
    // and was measured to cost 3.3x on reads for minutes afterwards.
    MEMORYSTATUSEX ms; ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) {
        const int64_t avail_mb = (int64_t) (ms.ullAvailPhys / (1024 * 1024));
        if ((int64_t) arena_mb > avail_mb - 1500) {
            fprintf(stderr, "expert-stream: arena %d MB too large for %lld MB available\n",
                    arena_mb, (long long) avail_mb);
            es_disable("arena would leave under 1500 MB free");
            return;
        }
    }

    const size_t arena_bytes = (size_t) arena_mb * 1024 * 1024;

    // Warn loudly rather than silently returning nothing. Below one lap the hit
    // rate is not "low", it is zero, and the memory is pure loss.
    const double laps = (double) arena_bytes / (double) lap_bytes;
    if (laps < 1.05) {
        fprintf(stderr,
                "expert-stream: WARNING arena %d MB holds only %.2f of one token's %lld MB "
                "expert working set - expect ~0%% hits\n",
                arena_mb, laps, (long long) (lap_bytes / (1024 * 1024)));
    }

    g_nslot = 0;
    for (int p = 0; p < g_npool; p++) {
        const double share = (double) pool_lap[p] / (double) lap_bytes;
        int n = (int) ((double) arena_bytes * share / (double) g_pool[p].slot_bytes);
        if (n < 8) n = 8;                       // a pool this small throttles everything
        g_pool[p].first = g_nslot;
        g_pool[p].count = n;
        g_nslot += n;
    }

    g_slot = (struct es_slot *) calloc((size_t) g_nslot, sizeof(struct es_slot));
    if (!g_slot) { es_disable("out of memory for slot table"); return; }

    // One VirtualAlloc per pool, subdivided. slot_bytes is a multiple of the
    // sector, so every sub-buffer stays sector-aligned, which unbuffered reads
    // require of the destination address as well as the offset and length.
    for (int p = 0; p < g_npool; p++) {
        const size_t bytes = (size_t) g_pool[p].count * g_pool[p].slot_bytes;
        g_arena[p] = (unsigned char *) VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!g_arena[p]) { es_disable("arena allocation failed"); return; }
        for (int k = 0; k < g_pool[p].count; k++) {
            struct es_slot * s = &g_slot[g_pool[p].first + k];
            s->buf      = g_arena[p] + (size_t) k * g_pool[p].slot_bytes;
            s->cap      = g_pool[p].slot_bytes;
            s->row_idx  = -1;
            s->ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);   // manual reset
            if (!s->ov.hEvent) { es_disable("event creation failed"); return; }
        }
    }

    g_init = 1;
    atexit(es_report);
    fprintf(stderr, "expert-stream: ACTIVE - %d tensors, %d shards, %d MB arena, %d slots in %d pools, "
                    "one token = %lld MB (%.2f laps cached)\n",
            g_nrow, g_nshard, arena_mb, g_nslot, g_npool,
            (long long) (lap_bytes / (1024 * 1024)), laps);
    for (int p = 0; p < g_npool; p++) {
        fprintf(stderr, "expert-stream:   pool %d: stride %lld -> %d slots x %zu bytes\n",
                p, (long long) g_pool[p].stride, g_pool[p].count, g_pool[p].slot_bytes);
    }
}

static BOOL CALLBACK es_init_once_cb(PINIT_ONCE once, PVOID param, PVOID * ctx) {
    (void) once; (void) param; (void) ctx;
    es_init_locked();
    return TRUE;
}

static void es_init(void) {
    // Blocks every caller until initialisation finishes, so all threads observe
    // the same g_init and take the same branch at the call site. See the note
    // above es_init_locked.
    InitOnceExecuteOnce(&g_once, es_init_once_cb, NULL, NULL);
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

bool ggml_expert_stream_active(void) {
    es_init();
    return g_init == 1;
}

static const struct es_row * es_find(const char * name) {
    // Linear scan over ~129 rows, 3 times per layer. Against 1.6 GiB of I/O per
    // token this is not measurable, and a hash map would be more code to get
    // wrong for no gain.
    for (int i = 0; i < g_nrow; i++) {
        if (strcmp(g_row[i].name, name) == 0) return &g_row[i];
    }
    return NULL;
}

bool ggml_expert_stream_is_expert(const struct ggml_tensor * t) {
    if (!ggml_expert_stream_active() || !t) return false;
    return es_find(t->name) != NULL;
}

void ggml_expert_stream_begin(const struct ggml_tensor * t) {
    g_cur_row     = NULL;
    g_cur_row_idx = -1;
    g_pending     = 0;
    if (g_init != 1 || !t) return;
    for (int i = 0; i < g_nrow; i++) {
        if (strcmp(g_row[i].name, t->name) == 0) { g_cur_row = &g_row[i]; g_cur_row_idx = i; break; }
    }
    if (!g_cur_row) return;
    // Slot CONTENTS deliberately survive this: the cache spans calls, layers and
    // tokens. Only the per-call expert -> slot mapping is reset.
    for (int i = 0; i < ES_MAX_EXPERTS; i++) g_slot_of[i] = -1;
    g_gen++;
}

bool ggml_expert_stream_request(int64_t expert_id) {
    if (g_init != 1 || !g_cur_row) return false;
    if (expert_id < 0 || expert_id >= g_cur_row->n_expert) return false;
    if (expert_id >= ES_MAX_EXPERTS) return false;

    const struct es_row * r = g_cur_row;

    // Which pool this slice belongs to. Pools are exact-sized per stride, so a
    // slice only ever lands in a buffer that fits it precisely.
    int lo = -1, hi = -1;
    for (int p = 0; p < g_npool; p++) {
        if (g_pool[p].stride == r->stride) { lo = g_pool[p].first; hi = lo + g_pool[p].count; break; }
    }
    if (lo < 0) { g_stat_fallback++; return false; }   // stride with no pool: should be impossible

    // ---- already cached? ----
    for (int i = lo; i < hi; i++) {
        struct es_slot * s = &g_slot[i];
        if (s->valid && s->row_idx == g_cur_row_idx && s->expert == expert_id) {
            s->tick = ++g_tick;
            s->gen  = g_gen;
            g_slot_of[expert_id] = i;
            g_stat_hits++;
            g_stat_hit_bytes += r->stride;
            return true;   // no I/O at all
        }
    }
    g_stat_misses++;

    // ---- pick a victim: LRU, but never one this batch is already using ----
    int victim = -1;
    uint64_t best = ~(uint64_t) 0;
    for (int i = lo; i < hi; i++) {
        struct es_slot * s = &g_slot[i];
        if (s->gen == g_gen || s->pending) continue;   // in use by this fetch batch
        if (!s->valid) { victim = i; break; }          // free slot beats evicting anything
        if (s->tick < best) { best = s->tick; victim = i; }
    }
    if (victim < 0) {
        // Every slot in this pool is already spoken for by the current batch.
        // Caller reads through the mmap instead: correct, just slower. Happens
        // in prefill, where one layer can route to far more experts than decode.
        g_stat_fallback++;
        return false;
    }

    const int64_t off   = r->base_offset + expert_id * r->stride;
    const int64_t aoff  = es_round_dn(off, ES_SECTOR);
    const int64_t slop  = off - aoff;
    int64_t       alen  = es_round_up(slop + r->stride, ES_SECTOR);

    // An unbuffered read may not run past the sector-rounded end of file.
    const int64_t eofcap = es_round_up(g_shard[r->shard_idx].size, ES_SECTOR) - aoff;
    if (alen > eofcap) alen = es_round_dn(eofcap, ES_SECTOR);

    struct es_slot * s = &g_slot[victim];
    if (alen <= 0 || (size_t) alen > s->cap) { g_stat_fallback++; return false; }

    // Invalidate BEFORE the read starts. If it fails or is short, the slot must
    // not still claim to hold its previous contents.
    s->valid = 0;
    s->ov.Offset     = (DWORD) (aoff & 0xFFFFFFFF);
    s->ov.OffsetHigh = (DWORD) (aoff >> 32);
    ResetEvent(s->ov.hEvent);
    s->h       = g_shard[r->shard_idx].h;
    s->slop    = slop;
    s->want    = r->stride;
    s->len     = (DWORD) alen;
    s->row_idx = g_cur_row_idx;
    s->expert  = expert_id;
    s->tick    = ++g_tick;
    s->gen     = g_gen;

    BOOL rc = ReadFile(s->h, s->buf, (DWORD) alen, NULL, &s->ov);
    if (!rc && GetLastError() != ERROR_IO_PENDING) {
        s->row_idx = -1;
        g_stat_failed++;
        return false;   // not queued; caller falls back
    }

    s->pending = 1;
    g_slot_of[expert_id] = victim;
    g_issued[g_pending++] = victim;
    return true;
}

void ggml_expert_stream_submit(void) {
    if (g_init != 1 || g_pending == 0) return;

    // The reads are ALREADY outstanding - they were issued as they were
    // requested, which is the entire point. This only collects them, so all of
    // them were in flight together.
    for (int k = 0; k < g_pending; k++) {
        struct es_slot * s = &g_slot[g_issued[k]];
        DWORD got = 0;
        s->pending = 0;
        if (!GetOverlappedResult(s->h, &s->ov, &got, TRUE)) {
            DWORD e = GetLastError();
            if (e != ERROR_HANDLE_EOF) {
                g_stat_failed++;
                s->row_idx = -1;
                g_slot_of[s->expert] = -1;   // force fallback for this expert
                continue;
            }
        }
        if ((int64_t) got < s->slop + s->want) {
            // Short read. Not enough bytes for the whole expert, so this slot
            // cannot be used. Falling back is correct; using it would be silent
            // corruption, which at IQ1_S reads as quantisation noise.
            g_stat_failed++;
            s->row_idx = -1;
            g_slot_of[s->expert] = -1;
            continue;
        }
        s->valid = 1;
        g_stat_reads++;
        g_stat_bytes += got;
    }
    g_pending = 0;
}

const void * ggml_expert_stream_ptr(int64_t expert_id) {
    if (g_init != 1 || !g_cur_row) return NULL;
    if (expert_id < 0 || expert_id >= ES_MAX_EXPERTS) return NULL;
    const int32_t si = g_slot_of[expert_id];
    if (si < 0 || si >= g_nslot) return NULL;
    const struct es_slot * s = &g_slot[si];
    // Belt and braces: the slot must still hold what this call asked for.
    if (!s->valid || s->row_idx != g_cur_row_idx || s->expert != expert_id) return NULL;
    // buf is page-aligned and slop is a multiple of 32 (GGUF aligns tensors to
    // 32), so this pointer is 32-byte aligned and ggml's SIMD loads are fine.
    return s->buf + s->slop;
}

#endif // _WIN32
