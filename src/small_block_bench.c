// small_block_bench.c - how fast is this SSD BELOW the sizes we have measured?
//
// WHY THIS EXISTS
// N-8 (gate-first intra-expert skipping, docs/IDEA_REVIEW_intra_expert_sparsity.md)
// has a ceiling of 2.49x on bytes per token, the largest number on the roadmap.
// It needs to fetch individual neuron rows of ffn_up_exps, which at IQ1_S are
// 800 BYTES each.
//
// The smallest read this project has ever measured is about 1.6 MB. The
// queue-depth sweep ran at 8 MB. WE HAVE NO DATA BELOW 1.6 MB IN EITHER
// DIRECTION, so every claim about 800-byte reads - mine included - is currently
// unfounded. This measures it.
//
// The fallback if tiny reads collapse is to fetch at 256-neuron granularity,
// which is 200 KB contiguous. So 200 KB is the other number that matters, and a
// verdict is printed for both.
//
// TWO PATTERNS, because uniform random is not what N-8 would actually do:
//   rand       offsets uniform across the whole shard. The pessimistic case.
//   clustered  offsets scattered INSIDE one 6.5 MB expert, which is exactly
//              N-8's access pattern: you have already decided to touch this
//              expert and are picking rows out of it.
//
// SAFETY
// Opens ONE existing model shard READ-ONLY with FILE_SHARE_READ. Writes nothing,
// creates nothing, deletes nothing. It cannot modify the model. Worst case it
// makes the disk busy for a couple of minutes.
//
// FILE_FLAG_NO_BUFFERING constraints, which are themselves part of the result:
// offset and length must both be multiples of the sector size (512 here), and
// the buffer must be sector-aligned. THERE IS NO SUCH THING AS AN 800-BYTE READ.
// The smallest possible read is one 512-byte sector, and an 800-byte row that
// straddles a sector boundary costs TWO sectors - 1024 bytes for 800 wanted,
// 28% waste before any throughput number is considered.
//
// Build:
//   gcc -O2 -o small_block_bench.exe small_block_bench.c

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define SECTOR      512
#define MAX_QD      16
#define TARGET_BYTES (256ull * 1024 * 1024)
#define MIN_READS    200
#define MAX_READS    20000

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t xrand(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return g_rng;
}

static double now_s(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}

// One expert at IQ1_S is 6,488,064 bytes. Round to a sector multiple for the
// clustered pattern's window.
#define EXPERT_WINDOW ((6488064ull / SECTOR) * SECTOR)

typedef struct { const char *name; uint64_t size; } bsize_t;

static int run_one(HANDLE h, uint64_t file_size, uint64_t bsz, int qd,
                   int clustered, double *out_mbps, double *out_iops)
{
    uint64_t n = TARGET_BYTES / bsz;
    if (n < MIN_READS) n = MIN_READS;
    if (n > MAX_READS) n = MAX_READS;

    // Aligned buffers, one per outstanding request.
    char *bufs[MAX_QD];
    OVERLAPPED ov[MAX_QD];
    HANDLE ev[MAX_QD];
    for (int i = 0; i < qd; i++) {
        bufs[i] = (char *)VirtualAlloc(NULL, (SIZE_T)bsz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!bufs[i]) { fprintf(stderr, "VirtualAlloc failed\n"); return 0; }
        ev[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
    }

    // Pick the cluster window once per batch when clustered, so the reads really
    // are inside one expert rather than merely near each other.
    uint64_t win_base = 0;
    uint64_t span = (file_size > bsz) ? (file_size - bsz) : 0;

    double t0 = now_s();
    uint64_t done = 0;
    while (done < n) {
        int batch = qd;
        if ((uint64_t)batch > n - done) batch = (int)(n - done);

        if (clustered) {
            uint64_t w = (span > EXPERT_WINDOW) ? (span - EXPERT_WINDOW) : 0;
            win_base = w ? ((xrand() % w) / SECTOR) * SECTOR : 0;
        }

        for (int i = 0; i < batch; i++) {
            uint64_t off;
            if (clustered) {
                uint64_t inner = (EXPERT_WINDOW > bsz) ? (EXPERT_WINDOW - bsz) : 0;
                off = win_base + (inner ? ((xrand() % inner) / SECTOR) * SECTOR : 0);
            } else {
                off = span ? ((xrand() % span) / SECTOR) * SECTOR : 0;
            }
            ZeroMemory(&ov[i], sizeof(OVERLAPPED));
            ResetEvent(ev[i]);
            ov[i].hEvent = ev[i];
            ov[i].Offset     = (DWORD)(off & 0xFFFFFFFFull);
            ov[i].OffsetHigh = (DWORD)(off >> 32);
            DWORD got = 0;
            if (!ReadFile(h, bufs[i], (DWORD)bsz, &got, &ov[i])) {
                if (GetLastError() != ERROR_IO_PENDING) {
                    fprintf(stderr, "ReadFile failed at %llu size %llu err %lu\n",
                            (unsigned long long)off, (unsigned long long)bsz, GetLastError());
                    for (int j = 0; j < qd; j++) { VirtualFree(bufs[j], 0, MEM_RELEASE); CloseHandle(ev[j]); }
                    return 0;
                }
            }
        }
        for (int i = 0; i < batch; i++) {
            DWORD got = 0;
            if (!GetOverlappedResult(h, &ov[i], &got, TRUE) || got != bsz) {
                fprintf(stderr, "short/failed read: got %lu want %llu err %lu\n",
                        got, (unsigned long long)bsz, GetLastError());
                for (int j = 0; j < qd; j++) { VirtualFree(bufs[j], 0, MEM_RELEASE); CloseHandle(ev[j]); }
                return 0;
            }
        }
        done += batch;
    }
    double dt = now_s() - t0;

    for (int i = 0; i < qd; i++) { VirtualFree(bufs[i], 0, MEM_RELEASE); CloseHandle(ev[i]); }

    if (dt <= 0.0) return 0;
    *out_mbps = (double)(done * bsz) / dt / (1024.0 * 1024.0);
    *out_iops = (double)done / dt;
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: small_block_bench <path-to-gguf-shard> [csv-out]\n");
        return 2;
    }
    const char *path = argv[1];
    const char *csv  = (argc > 2) ? argv[2] : NULL;

    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "cannot open %s (err %lu)\n", path, GetLastError());
        return 2;
    }
    LARGE_INTEGER fs;
    GetFileSizeEx(h, &fs);
    uint64_t file_size = (uint64_t)fs.QuadPart;

    printf("small_block_bench - unbuffered reads below the sizes we have measured\n");
    printf("file    %s\n", path);
    printf("size    %.1f GB   sector %d   READ-ONLY, writes nothing\n\n",
           (double)file_size / (1024.0*1024.0*1024.0), SECTOR);

    bsize_t sizes[] = {
        { "512 B",   512ull },
        { "1 KB",    1024ull },        // an 800-byte row straddling a sector
        { "4 KB",    4096ull },        // what mmap faults in
        { "64 KB",   65536ull },
        { "200 KB",  204800ull },      // 256-neuron group, N-8 clustered variant
        { "1.6 MB",  1638400ull },     // smallest size measured before today
        { "8 MB",    8388608ull },     // the queue-depth sweep size
    };
    int nsz = (int)(sizeof(sizes)/sizeof(sizes[0]));
    int qds[] = { 1, 8 };

    FILE *fo = NULL;
    if (csv) {
        fo = fopen(csv, "w");
        if (fo) fprintf(fo, "pattern,block,block_bytes,qd,mbps,iops\n");
    }

    for (int p = 0; p < 2; p++) {
        const char *pname = p ? "clustered" : "rand";
        printf("%s  %s\n", pname,
               p ? "(inside one 6.5 MB expert - N-8's real pattern)"
                 : "(uniform across the whole shard)");
        printf("  %-8s %12s %12s %12s %12s\n", "block", "QD1 MB/s", "QD1 kIOPS", "QD8 MB/s", "QD8 kIOPS");
        for (int s = 0; s < nsz; s++) {
            double m[2] = {0,0}, io[2] = {0,0};
            for (int q = 0; q < 2; q++) {
                g_rng = 0x9E3779B97F4A7C15ull ^ (uint64_t)(s * 31 + q * 7 + p * 101);
                if (!run_one(h, file_size, sizes[s].size, qds[q], p, &m[q], &io[q])) {
                    CloseHandle(h);
                    if (fo) fclose(fo);
                    return 1;
                }
                if (fo) fprintf(fo, "%s,%s,%llu,%d,%.1f,%.1f\n", pname, sizes[s].name,
                                (unsigned long long)sizes[s].size, qds[q], m[q], io[q]);
            }
            printf("  %-8s %12.1f %12.1f %12.1f %12.1f\n",
                   sizes[s].name, m[0], io[0]/1000.0, m[1], io[1]/1000.0);
        }
        printf("\n");
    }

    CloseHandle(h);
    if (fo) fclose(fo);
    return 0;
}
