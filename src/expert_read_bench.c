/*
 * expert_read_bench.c
 *
 * WHAT THIS DECIDES
 * -----------------
 * Whether the expert bank has to be repacked into an aligned, expert-major
 * experts.bin (a 76 GB bulk write we do not have the disk or the time for), or
 * whether an unbuffered reader can address experts in place inside the original
 * GGUF shards.
 *
 * Alignment is already settled and is NOT the question: every per-expert stride
 * in this model is divisible by 512 and by 4096, and each tensor's base offset
 * is misaligned by a constant, known slop (416 bytes at mod-512). An unbuffered
 * reader absorbs that by reading the enclosing sector-aligned window and
 * handing back buf + slop -- 512 wasted bytes on a ~6.8 MB expert, and since
 * GGUF aligns tensors to 32 bytes the returned pointer stays 32-byte aligned,
 * so ggml's SIMD loads are unaffected.
 *
 * The one real question left is CONTIGUITY. gate, up and down are three
 * separate tensors living far apart in the file, so one expert costs three
 * seeks, not one:
 *
 *     43 MoE layers x 6 routed experts x 3 tensors = 774 reads per token
 *
 * A repack would turn that into 258 larger reads. That is only worth 76 GB of
 * writes if the small reads actually run slower. Measure it.
 *
 * WHAT IT MEASURES
 * ----------------
 *   mode=split    the real layout: 774 reads/token at the true strides
 *                 (1,638,400 / 2,162,688 / 3,211,264 / 4,456,448 bytes)
 *   mode=packed   the hypothetical repacked layout: 258 reads/token, each one
 *                 contiguous and the size of a whole expert (gate+up+down).
 *                 Same total bytes, one third the seeks. This is the ceiling a
 *                 repack could possibly buy.
 *
 * Both use FILE_FLAG_NO_BUFFERING, so the OS page cache is bypassed entirely.
 * That is the point: these numbers are the disk, not the standby list, and
 * running this does not evict anything else on the machine.
 *
 * SAFETY
 * ------
 * Read-only. Opens the shards with GENERIC_READ and FILE_SHARE_READ only, never
 * creates, writes, or deletes a file, and bounds itself with both a token limit
 * and a wall-clock limit so it cannot run away unattended.
 *
 * BUILD (w64devkit, 64-bit)
 *   gcc -O2 -o build/expert_read_bench.exe src/expert_read_bench.c
 *
 * USAGE
 *   expert_read_bench.exe --manifest <csv> [--mode split|packed] [--qd N]
 *                         [--tokens N] [--max-seconds N] [--seed N]
 *                         [--sector N] [--csv <path>]
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>

#define MAX_SHARDS   8
#define MAX_ROWS   512
#define MAX_QD      32

/* ------------------------------------------------------------------ */
/* manifest                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    int      layer;
    int      kind;          /* 0 gate, 1 up, 2 down */
    int      shard_idx;
    int64_t  base_offset;
    int64_t  stride;
    int      n_expert;
} row_t;

typedef struct {
    char    path[MAX_PATH];
    HANDLE  h;
    int64_t size;
} shard_t;

static shard_t g_shard[MAX_SHARDS];
static int     g_nshard = 0;
static row_t   g_row[MAX_ROWS];
static int     g_nrow = 0;

/* layer -> the three row indices, so a token replay can walk layers in order */
static int  g_layer_id[MAX_ROWS];
static int  g_nlayer = 0;
static int  g_layer_row[MAX_ROWS][3];

static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "FATAL: "); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n");
    va_end(ap);
    exit(2);
}

static int kind_of(const char *s) {
    if (!strcmp(s, "gate")) return 0;
    if (!strcmp(s, "up"))   return 1;
    if (!strcmp(s, "down")) return 2;
    return -1;
}

static void load_manifest(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) die("cannot open manifest %s", path);

    char line[1024];
    int seen_header = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            int idx; char p[MAX_PATH];
            /* "#S <idx> <path>" - path may contain spaces, so take the rest of
               the line rather than a %s token. The field width is MANDATORY:
               an unbounded %[^\n] here would let a long path in the manifest
               overrun this stack buffer. */
            if (sscanf(line, "#S %d %259[^\n\r]", &idx, p) == 2) {
                if (idx < 0 || idx >= MAX_SHARDS) die("shard index %d out of range", idx);
                size_t plen = strlen(p);
                if (plen >= MAX_PATH) die("shard %d path too long (%zu chars)", idx, plen);
                memcpy(g_shard[idx].path, p, plen + 1);
                if (idx + 1 > g_nshard) g_nshard = idx + 1;
            }
            continue;
        }
        if (!seen_header) { seen_header = 1; continue; }   /* column header */
        if (line[0] == '\n' || line[0] == '\r' || line[0] == 0) continue;

        char kind[32];
        row_t r;
        long long base, stride;
        if (sscanf(line, "%d,%31[^,],%d,%lld,%lld,%d",
                   &r.layer, kind, &r.shard_idx, &base, &stride, &r.n_expert) != 6) {
            die("malformed manifest line: %s", line);
        }
        r.kind = kind_of(kind);
        if (r.kind < 0) die("unknown kind '%s'", kind);
        r.base_offset = (int64_t)base;
        r.stride      = (int64_t)stride;
        if (r.shard_idx < 0 || r.shard_idx >= MAX_SHARDS) die("bad shard_idx %d", r.shard_idx);
        if (r.stride <= 0 || r.n_expert <= 0) die("bad stride/n_expert on layer %d", r.layer);
        if (g_nrow >= MAX_ROWS) die("too many manifest rows (max %d)", MAX_ROWS);
        g_row[g_nrow++] = r;
    }
    fclose(f);

    if (g_nshard == 0) die("manifest declared no shards (#S lines missing)");
    if (g_nrow == 0)   die("manifest had no data rows");

    /* group rows by layer; every layer must have all three kinds or a replay
       would read less than a real token does and report a flattering number */
    for (int i = 0; i < MAX_ROWS; i++) for (int k = 0; k < 3; k++) g_layer_row[i][k] = -1;
    for (int i = 0; i < g_nrow; i++) {
        int L = -1;
        for (int j = 0; j < g_nlayer; j++) if (g_layer_id[j] == g_row[i].layer) { L = j; break; }
        if (L < 0) { L = g_nlayer++; g_layer_id[L] = g_row[i].layer; }
        if (g_layer_row[L][g_row[i].kind] != -1)
            die("duplicate tensor for layer %d kind %d", g_row[i].layer, g_row[i].kind);
        g_layer_row[L][g_row[i].kind] = i;
    }
    for (int L = 0; L < g_nlayer; L++)
        for (int k = 0; k < 3; k++)
            if (g_layer_row[L][k] < 0)
                die("layer %d is missing kind %d - manifest incomplete", g_layer_id[L], k);
}

/* ------------------------------------------------------------------ */
/* rng: xorshift64*, fixed seed, so a run is reproducible              */
/* ------------------------------------------------------------------ */

static uint64_t g_rng;
static uint64_t rnd(void) {
    g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27;
    return g_rng * 2685821657736338717ULL;
}

/* pick k DISTINCT values in [0,n) - real routing picks distinct experts, and
   allowing repeats would fake cache hits that the real thing does not get */
static void pick_distinct(int *out, int k, int n) {
    for (int i = 0; i < k; i++) {
        int v, dup;
        do {
            v = (int)(rnd() % (uint64_t)n);
            dup = 0;
            for (int j = 0; j < i; j++) if (out[j] == v) { dup = 1; break; }
        } while (dup);
        out[i] = v;
    }
}

/* ------------------------------------------------------------------ */
/* work list                                                           */
/* ------------------------------------------------------------------ */

typedef struct {
    int      shard_idx;
    int64_t  off;        /* true (unaligned) byte offset of the wanted data */
    int64_t  len;        /* true length of the wanted data                  */
} work_t;

typedef struct {
    OVERLAPPED     ov;
    unsigned char *buf;
    DWORD          len;      /* aligned length actually submitted */
    int64_t        want;     /* useful bytes this read is for      */
    /* The handle this read was issued on. Completions arrive out of order, so
       it cannot be recovered from a completion counter - GetOverlappedResult
       must be given the same handle ReadFile was given. */
    HANDLE         h;
    int            busy;
} slot_t;

static int64_t round_dn(int64_t v, int64_t a) { return v - (v % a); }
static int64_t round_up(int64_t v, int64_t a) { return ((v + a - 1) / a) * a; }

int main(int argc, char **argv) {
    const char *manifest = NULL;
    const char *mode     = "split";
    const char *csvout   = NULL;
    int qd = 4, tokens = 4, max_seconds = 120, sector = 4096;
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--manifest")    && i+1 < argc) manifest    = argv[++i];
        else if (!strcmp(argv[i], "--mode")        && i+1 < argc) mode        = argv[++i];
        else if (!strcmp(argv[i], "--csv")         && i+1 < argc) csvout      = argv[++i];
        else if (!strcmp(argv[i], "--qd")          && i+1 < argc) qd          = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens")      && i+1 < argc) tokens      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-seconds") && i+1 < argc) max_seconds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sector")      && i+1 < argc) sector      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")        && i+1 < argc) seed        = strtoull(argv[++i], NULL, 10);
        else die("unknown or incomplete argument: %s", argv[i]);
    }
    if (!manifest) die("--manifest is required");
    if (qd < 1 || qd > MAX_QD) die("--qd must be 1..%d", MAX_QD);
    if (tokens < 1) die("--tokens must be >= 1");
    if (max_seconds < 1) die("--max-seconds must be >= 1");
    if (sector <= 0 || (sector & (sector - 1)) != 0) die("--sector must be a power of two");
    int packed = !strcmp(mode, "packed");
    if (!packed && strcmp(mode, "split")) die("--mode must be split or packed");

    g_rng = seed ? seed : 1;
    load_manifest(manifest);

    /* ---------------- open shards, read-only, unbuffered ---------------- */
    for (int i = 0; i < g_nshard; i++) {
        if (!g_shard[i].path[0]) die("manifest never declared shard %d", i);
        g_shard[i].h = CreateFileA(
            g_shard[i].path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, NULL);
        if (g_shard[i].h == INVALID_HANDLE_VALUE)
            die("CreateFile(%s) failed, error %lu", g_shard[i].path, GetLastError());
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(g_shard[i].h, &sz))
            die("GetFileSizeEx(%s) failed, error %lu", g_shard[i].path, GetLastError());
        g_shard[i].size = sz.QuadPart;
    }

    /* ---------------- build the whole work list up front ---------------- */
    /* Doing this before timing keeps RNG and bookkeeping out of the measured
       window, so the number reported is disk time and nothing else. */
    int per_token = packed ? g_nlayer * 6 : g_nlayer * 6 * 3;
    long long nwork = (long long)per_token * tokens;
    work_t *work = (work_t *)malloc((size_t)nwork * sizeof(work_t));
    if (!work) die("out of memory building work list (%lld items)", nwork);

    long long w = 0;
    int64_t want_total = 0, biggest = 0;
    for (int t = 0; t < tokens; t++) {
        for (int L = 0; L < g_nlayer; L++) {
            int e[6];
            int n_expert = g_row[g_layer_row[L][0]].n_expert;
            pick_distinct(e, 6, n_expert);
            for (int j = 0; j < 6; j++) {
                if (packed) {
                    /* Hypothetical repack: gate+up+down of one expert as a
                       single contiguous run. There is no such run in the real
                       file, so read an equally sized contiguous run starting
                       where gate's expert starts. Same bytes, one third the
                       seeks -- the best a repack could do. */
                    int64_t total = 0;
                    for (int k = 0; k < 3; k++) total += g_row[g_layer_row[L][k]].stride;
                    row_t *r = &g_row[g_layer_row[L][0]];
                    int64_t off = r->base_offset + (int64_t)e[j] * r->stride;
                    /* keep it inside the shard */
                    if (off + total > g_shard[r->shard_idx].size)
                        off = g_shard[r->shard_idx].size - total;
                    if (off < 0) die("shard %d smaller than one packed expert", r->shard_idx);
                    work[w].shard_idx = r->shard_idx;
                    work[w].off = off;
                    work[w].len = total;
                    if (total > biggest) biggest = total;
                    want_total += total; w++;
                } else {
                    for (int k = 0; k < 3; k++) {
                        row_t *r = &g_row[g_layer_row[L][k]];
                        work[w].shard_idx = r->shard_idx;
                        work[w].off = r->base_offset + (int64_t)e[j] * r->stride;
                        work[w].len = r->stride;
                        if (r->stride > biggest) biggest = r->stride;
                        want_total += r->stride; w++;
                    }
                }
            }
        }
    }
    if (w != nwork) die("work list build mismatch: %lld vs %lld", w, nwork);

    /* ---------------- buffers ---------------- */
    /* One slot per outstanding read. Slot size covers the biggest read plus a
       full sector of slop, since the read window starts below the wanted data. */
    size_t slot_bytes = (size_t)round_up(biggest + sector, 65536);
    double buf_mb = (double)slot_bytes * qd / (1024.0 * 1024.0);
    if (buf_mb > 512.0)
        die("qd %d would need %.0f MB of buffers on a 15.4 GB machine - refusing", qd, buf_mb);

    slot_t slot[MAX_QD];
    memset(slot, 0, sizeof slot);
    for (int i = 0; i < qd; i++) {
        slot[i].buf = (unsigned char *)VirtualAlloc(NULL, slot_bytes,
                                                    MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!slot[i].buf) die("VirtualAlloc(%zu) failed, error %lu", slot_bytes, GetLastError());
        slot[i].ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);   /* manual reset */
        if (!slot[i].ov.hEvent) die("CreateEvent failed, error %lu", GetLastError());
    }

    printf("expert_read_bench\n");
    printf("  manifest   : %s\n", manifest);
    printf("  shards     : %d\n", g_nshard);
    printf("  MoE layers : %d\n", g_nlayer);
    printf("  mode       : %s (%d reads/token)\n", mode, per_token);
    printf("  qd         : %d\n", qd);
    printf("  tokens     : %d   (%lld reads total)\n", tokens, nwork);
    printf("  sector     : %d\n", sector);
    printf("  buffers    : %d x %zu bytes = %.1f MB\n", qd, slot_bytes, buf_mb);
    printf("  bytes want : %lld (%.3f GiB, %.3f GiB/token)\n",
           (long long)want_total, want_total / 1073741824.0,
           want_total / 1073741824.0 / tokens);
    printf("  UNBUFFERED: the OS page cache is bypassed, so this is the disk.\n");
    fflush(stdout);

    /* ---------------- replay ---------------- */
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    long long next = 0, done = 0;
    int64_t read_total = 0;      /* aligned bytes actually transferred */
    int inflight = 0;
    int timed_out = 0;

    while (next < nwork || inflight > 0) {
        while (inflight < qd && next < nwork) {
            int s = -1;
            for (int i = 0; i < qd; i++) if (!slot[i].busy) { s = i; break; }
            if (s < 0) break;

            work_t *k = &work[next];
            int64_t aoff = round_dn(k->off, sector);
            int64_t slop = k->off - aoff;
            int64_t alen = round_up(slop + k->len, sector);
            /* An unbuffered read may not run past the sector-rounded end of the
               file. Clamp rather than let ReadFile fail at the last expert. */
            int64_t cap = round_up(g_shard[k->shard_idx].size, sector) - aoff;
            if (alen > cap) alen = round_dn(cap, sector);
            if (alen <= 0) die("computed empty read at offset %lld", (long long)aoff);

            slot[s].ov.Offset     = (DWORD)(aoff & 0xFFFFFFFF);
            slot[s].ov.OffsetHigh = (DWORD)(aoff >> 32);
            ResetEvent(slot[s].ov.hEvent);
            slot[s].len  = (DWORD)alen;
            slot[s].want = k->len;
            slot[s].h    = g_shard[k->shard_idx].h;
            slot[s].busy = 1;

            BOOL ok = ReadFile(g_shard[k->shard_idx].h, slot[s].buf, (DWORD)alen, NULL, &slot[s].ov);
            if (!ok && GetLastError() != ERROR_IO_PENDING)
                die("ReadFile failed at offset %lld len %lld, error %lu",
                    (long long)aoff, (long long)alen, GetLastError());
            inflight++; next++;
        }

        HANDLE hw[MAX_QD]; int map[MAX_QD]; int n = 0;
        for (int i = 0; i < qd; i++) if (slot[i].busy) { hw[n] = slot[i].ov.hEvent; map[n] = i; n++; }
        if (n == 0) break;

        DWORD wr = WaitForMultipleObjects((DWORD)n, hw, FALSE, 30000);
        if (wr == WAIT_TIMEOUT) die("a read did not complete within 30 s - disk stalled");
        if (wr >= WAIT_OBJECT_0 + (DWORD)n) die("WaitForMultipleObjects failed, error %lu", GetLastError());
        int s = map[wr - WAIT_OBJECT_0];

        DWORD got = 0;
        if (!GetOverlappedResult(slot[s].h, &slot[s].ov, &got, FALSE)) {
            DWORD e = GetLastError();
            if (e != ERROR_HANDLE_EOF) die("GetOverlappedResult failed, error %lu", e);
        }
        read_total += got;
        slot[s].busy = 0;
        inflight--; done++;

        QueryPerformanceCounter(&t1);
        double el = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;
        if (el > max_seconds) { timed_out = 1; break; }
    }

    QueryPerformanceCounter(&t1);
    double elapsed = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;

    /* Drain anything still outstanding before freeing buffers the kernel may
       still be writing into. Skipping this is a use-after-free on timeout. */
    for (int i = 0; i < qd; i++) {
        if (slot[i].busy) {
            DWORD got = 0;
            GetOverlappedResult(slot[i].h, &slot[i].ov, &got, TRUE);   /* TRUE = block until done */
            slot[i].busy = 0;
        }
    }

    double mbps      = (read_total / (1024.0 * 1024.0)) / (elapsed > 0 ? elapsed : 1e-9);
    double tokens_done = (double)done / per_token;
    double ms_per_tok  = tokens_done > 0 ? (elapsed * 1000.0) / tokens_done : 0.0;

    printf("\nRESULT%s\n", timed_out ? "  (STOPPED BY --max-seconds, partial)" : "");
    printf("  reads done      : %lld of %lld\n", done, nwork);
    printf("  tokens replayed : %.2f\n", tokens_done);
    printf("  bytes read      : %lld (%.3f GiB)\n", (long long)read_total, read_total / 1073741824.0);
    printf("  elapsed         : %.3f s\n", elapsed);
    printf("  throughput      : %.1f MB/s\n", mbps);
    printf("  ms per token    : %.1f   -> %.3f tok/s if I/O were the only cost\n",
           ms_per_tok, ms_per_tok > 0 ? 1000.0 / ms_per_tok : 0.0);

    if (csvout) {
        /* Append, with a header only when creating the file, so repeated sweeps
           accumulate into one table instead of overwriting each other. */
        int fresh = 0;
        FILE *chk = fopen(csvout, "r");
        if (chk) fclose(chk); else fresh = 1;
        FILE *cf = fopen(csvout, "a");
        if (!cf) {
            fprintf(stderr, "WARNING: could not append to %s - result printed above only\n", csvout);
        } else {
            if (fresh) fprintf(cf, "mode,qd,sector,tokens_req,reads_done,reads_req,"
                                   "bytes_read,elapsed_s,mbps,ms_per_token,timed_out,seed\n");
            fprintf(cf, "%s,%d,%d,%d,%lld,%lld,%lld,%.3f,%.1f,%.1f,%d,%llu\n",
                    mode, qd, sector, tokens, done, nwork, (long long)read_total,
                    elapsed, mbps, ms_per_tok, timed_out, (unsigned long long)seed);
            fclose(cf);
            printf("  appended to     : %s\n", csvout);
        }
    }

    for (int i = 0; i < qd; i++) {
        if (slot[i].ov.hEvent) CloseHandle(slot[i].ov.hEvent);
        if (slot[i].buf) VirtualFree(slot[i].buf, 0, MEM_RELEASE);
    }
    for (int i = 0; i < g_nshard; i++)
        if (g_shard[i].h && g_shard[i].h != INVALID_HANDLE_VALUE) CloseHandle(g_shard[i].h);
    free(work);
    return timed_out ? 3 : 0;
}
