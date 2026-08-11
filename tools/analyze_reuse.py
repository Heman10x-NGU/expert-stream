#!/usr/bin/env python3
"""
analyze_reuse.py

THE MEASUREMENT THE WHOLE REMAINING PROJECT DEPENDS ON.

expert_read_bench established the disk ceiling: ~850 MB/s sustained on random
reads at this model's stride sizes. A token routes to 1.649 GiB of expert
weights. If every one of those bytes has to come off the SSD, the hard ceiling
is ~2.0 s/token, i.e. ~0.5 tok/s, with zero compute and perfect overlap.

So any target above ~0.5 tok/s is not reachable by reading faster. It is only
reachable by NOT READING - by serving experts out of a RAM cache. That makes
cache hit rate the single number that determines what this project can achieve:

    achievable tok/s  ~=  0.5 / (1 - hit_rate_by_bytes)

    hit rate   0%  ->  0.50 tok/s
              50%  ->  1.00
              75%  ->  2.00
              90%  ->  5.00

This script measures the hit rate a bounded cache actually gets on a REAL
routing trace captured from the model, under two policies:

  LRU     what the OS page cache already gives us for free, and therefore the
          baseline any hand-written cache has to beat to justify existing.

  BELADY  evict whichever cached expert is needed furthest in the future. Not
          implementable (it requires knowing the future) but it is the proven
          optimal, so it is the CEILING on every possible eviction policy,
          including the N-2 lookahead oracle. If Belady is not much better than
          LRU here, N-2 is not worth building and should be cut.

Accounting is in BYTES, not expert counts. Per-expert sizes vary from 6.49 MB
to 8.78 MB across layers (Unsloth dynamic quant), and it is bytes that cost
time, so a hit-rate measured in expert counts would be the wrong number.

Reads bench/results/moe_trace-*.csv and bench/results/expert_manifest.csv.
Read-only. No model file is opened.

Usage:
    python tools\\analyze_reuse.py [--trace <csv>] [--sizes-gb 1,2,4,6,8]
"""

from __future__ import annotations

import argparse
import csv as csvmod
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "bench", "results")
MANIFEST = os.path.join(RESULTS, "expert_manifest.csv")
OUT_TXT = os.path.join(RESULTS, "reuse_analysis.txt")

_LOG: list[str] = []


def out(msg: str = "") -> None:
    msg = str(msg).encode("ascii", errors="backslashreplace").decode("ascii")
    print(msg)
    _LOG.append(msg)


def load_expert_bytes() -> dict[int, int]:
    """layer -> bytes for ONE expert (gate+up+down strides summed)."""
    if not os.path.isfile(MANIFEST):
        raise SystemExit(f"FATAL: {MANIFEST} not found. Run tools\\make_expert_manifest.py first.")
    per_layer: dict[int, int] = {}
    with open(MANIFEST, "r", encoding="ascii") as f:
        for line in f:
            if line.startswith("#") or line.startswith("layer,"):
                continue
            parts = line.strip().split(",")
            if len(parts) != 6:
                continue
            layer = int(parts[0]); stride = int(parts[4])
            per_layer[layer] = per_layer.get(layer, 0) + stride
    if not per_layer:
        raise SystemExit("FATAL: manifest had no usable rows.")
    return per_layer


def load_trace(path: str) -> list[tuple[int, int, int]]:
    """Return the access stream as [(token_pos, layer, expert_id), ...] in
    execution order: tokens ascending, and within a token layers ascending.

    The CSV is grouped by layer (the writer walks a map keyed by layer), NOT in
    execution order, so it MUST be re-sorted here. Simulating a cache against
    the file's natural order would measure a schedule the model never runs.
    """
    rows = []
    with open(path, "r", encoding="ascii", newline="") as f:
        rd = csvmod.DictReader(f)
        need = {"token_pos", "layer", "expert_id"}
        if not need.issubset(set(rd.fieldnames or [])):
            raise SystemExit(f"FATAL: {path} missing columns {need - set(rd.fieldnames or [])}")
        for r in rd:
            rows.append((int(r["token_pos"]), int(r["layer"]), int(r["expert_id"])))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def simulate(stream: list[tuple[int, int]], cost: dict[int, int],
             cap_bytes: int, policy: str, nlayers: int = 43,
             guard: int = 4) -> tuple[int, int]:
    """stream is [(layer, expert)] in execution order.

    Returns (hit_bytes, miss_bytes). A repeat access WITHIN the same cache state
    counts as a hit, which is correct: the second of two consecutive uses of the
    same expert genuinely costs no I/O.

    POLICIES
    --------
    lru     what the OS page cache gives free. The baseline to beat.
    lfu     frequency. Tested to see if anything cheap gets near Belady.
    cyclic  NEW. Evict by distance to next use measured in LAYERS, using only
            the fact that the model always walks layers 0..N-1 in that order.
    belady  the unimplementable optimum. The ceiling.

    WHY 'cyclic' EXISTS, because this is the part that was got wrong once.

    The N-2 design in HLD.md derives distance-to-next-use from N-1's predicted
    routing, which is ~92% accurate and needs the whole lookahead machinery.
    docs/explained/09_whats_left.md then described that oracle as "run all 43
    routers before fetching anything ... a real view of the future, not a guess."
    THAT IS WRONG. Layer L's router consumes layer L-1's output, so the routers
    cannot be run ahead of the layers that feed them. Only layers 0-2, which
    route on a frozen hash of the token id, have an exact oracle.

    But an exact oracle exists anyway, and it needs no prediction at all: the
    ORDER of layers is fixed and known. An entry cached for layer L' has its
    next possible use at layer L', which from the current layer L is

        distance = ((L' - L - 1) mod nlayers) + 1     -> 1..nlayers

    Entries for the layer we are standing on score nlayers, i.e. furthest away,
    which is correct: they were used this lap and cannot be used again until the
    next one. This is Belady restricted to what is knowable without predicting
    anything, and it inverts LRU exactly where LRU is provably worst: on a cyclic
    walk, the least-recently-used entry is the one whose turn comes SOONEST.

    Ties (all entries of one layer share a distance) break on frequency, evicting
    the least-used, because within a layer that is the only signal left.

    guard   NEW. Plain LRU, except an entry is not evictable while its layer is
            within the next `guard` layers of the walk.

    'cyclic' was measured and LOSES to LRU at every cache size above 1 GB (see
    MEASURED_GROUND_TRUTH 12.1). Layer distance turns out to be the wrong signal:
    only ~36% of a layer's experts repeat between tokens, so "this layer comes
    round in 43" barely predicts "this expert is wanted again", and sorting purely
    by it discards the recency signal that does. 'guard' is the narrower claim
    that survives that result - keep LRU, which is measurably fine at size, and
    only override it in the one place LRU is provably pathological: evicting an
    entry whose turn is imminent.
    """
    # LFU is here because Belady needs the future and LRU is measured to be the
    # worst possible policy on a cyclic pattern. The design question stage B has
    # to answer is whether anything IMPLEMENTABLE gets near Belady, because if it
    # does, the whole N-2 lookahead machinery can be skipped.
    #
    # LFU should suit this workload in principle: routing is skewed, so a small
    # set of experts is used far more than the rest, and frequency is exactly
    # what "keep the hot ones" means. Unlike Belady it needs nothing but a
    # counter, and unlike LRU it is immune to the cyclic-eviction pathology,
    # because being used once per lap does not make an entry look cold.
    freq: dict[tuple[int, int], int] = {}

    if policy == "belady":
        # next_use[i] = the next index at which the same (layer,expert) is
        # accessed, or a sentinel past the end meaning "never again".
        nxt = [len(stream)] * len(stream)
        last: dict[tuple[int, int], int] = {}
        for i in range(len(stream) - 1, -1, -1):
            k = stream[i]
            nxt[i] = last.get(k, len(stream))
            last[k] = i

    cached: dict[tuple[int, int], int] = {}   # key -> LRU tick or next-use
    used = 0
    tick = 0
    hit_b = 0
    miss_b = 0

    def store(key, i):
        if policy in ("lru", "guard"):
            cached[key] = tick
        elif policy == "lfu":
            cached[key] = freq[key]
        elif policy == "cyclic":
            cached[key] = 0          # value unused; the key function reads layer
        else:
            cached[key] = nxt[i]

    for i, key in enumerate(stream):
        layer = key[0]
        sz = cost[layer]
        tick += 1
        freq[key] = freq.get(key, 0) + 1
        if key in cached:
            hit_b += sz
            store(key, i)
            continue

        miss_b += sz

        # An expert larger than the whole cache can never be retained. Charge the
        # miss and move on rather than evicting everything for something that
        # will not fit.
        if sz > cap_bytes:
            continue

        while used + sz > cap_bytes and cached:
            if policy == "guard":
                # LRU, but an entry whose layer is imminent is off limits. This is
                # the single place LRU is provably wrong on a cyclic walk: the
                # least-recently-used entry is the one whose turn comes soonest.
                # Everything else about LRU is left alone deliberately.
                def imminent(k):
                    d = ((k[0] - layer - 1) % nlayers) + 1
                    return d <= guard
                pool = [k for k in cached if not imminent(k)]
                if not pool:
                    pool = list(cached)      # all guarded: fall back to plain LRU
                victim = min(pool, key=lambda k: cached[k])
            elif policy in ("lru", "lfu"):
                # smallest tick (LRU) or smallest use count (LFU)
                victim = min(cached, key=lambda k: cached[k])
            elif policy == "cyclic":
                # Furthest away in the fixed layer walk; least-used breaks the tie.
                # The entries for the CURRENT layer are excluded from eviction: in
                # the real engine a layer's whole expert set has to be resident at
                # once, so evicting one to make room for the next of the same batch
                # is not a choice the reader is allowed to make. Without this the
                # simulation would model a policy the engine cannot implement.
                evictable = [k for k in cached if k[0] != layer]
                if not evictable:
                    break
                victim = max(evictable,
                             key=lambda k: (((k[0] - layer - 1) % nlayers) + 1,
                                            -freq.get(k, 0)))
            else:
                # evict the entry needed furthest in the future (or never)
                victim = max(cached, key=lambda k: cached[k])
            used -= cost[victim[0]]
            del cached[victim]

        # The loop above can exit with the entry still not fitting, if the only
        # remaining entries belong to the current layer. Charge the miss and do
        # not insert, rather than blowing the cap.
        if used + sz > cap_bytes:
            continue

        store(key, i)
        used += sz

    return hit_b, miss_b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="")
    ap.add_argument("--sizes-gb", default="1,2,3,4,6,8,12,16,24,32")
    ap.add_argument("--disk-mbps", type=float, default=850.0,
                    help="sustained random read MB/s measured by expert_read_bench")
    ap.add_argument("--guard", type=int, default=4,
                    help="GUARD policy: layers ahead that are protected from eviction")
    ap.add_argument("--guard-sweep", default="",
                    help="comma-separated guard widths to sweep, e.g. 1,2,4,8,16")
    args = ap.parse_args()

    cost = load_expert_bytes()

    if args.trace:
        traces = [args.trace]
    else:
        traces = sorted(glob.glob(os.path.join(RESULTS, "moe_trace-*.csv")))
    if not traces:
        raise SystemExit("FATAL: no moe_trace-*.csv found.")

    out("analyze_reuse.py - does a bounded expert cache actually help?")
    out(f"per-expert bytes: min={min(cost.values()):,} mean={sum(cost.values())//len(cost):,} "
        f"max={max(cost.values()):,}  over {len(cost)} MoE layers")
    out(f"disk assumption : {args.disk_mbps:.0f} MB/s sustained random read (measured)")
    out("")

    sizes = [float(s) for s in args.sizes_gb.split(",") if s.strip()]

    for tp in traces:
        rows = load_trace(tp)
        if not rows:
            out(f"{os.path.basename(tp)}: EMPTY, skipped")
            continue
        ntok = len(set(r[0] for r in rows))
        nlay = len(set(r[1] for r in rows))
        stream = [(r[1], r[2]) for r in rows]
        bytes_needed = sum(cost[l] for l, _ in stream)

        out("=" * 92)
        out(f"TRACE {os.path.basename(tp)}")
        out(f"  tokens {ntok}   layers {nlay}   accesses {len(stream):,}"
            f"   bytes if nothing were cached {bytes_needed/1073741824:.3f} GiB"
            f"   ({bytes_needed/ntok/1073741824:.3f} GiB/token)")

        # ---- how much reuse is even present, before any policy ----
        distinct = len(set(stream))
        out(f"  distinct (layer,expert) pairs touched = {distinct:,} of {len(stream):,} accesses"
            f"  -> {100.0*(1-distinct/len(stream)):.1f}% of accesses are repeats")

        # consecutive-token overlap per layer: the quantity N-1/N-2 lookahead
        # would exploit. Reported as a fraction of the 6 slots.
        per_tok: dict[tuple[int, int], set] = {}
        for t, l, e in rows:
            per_tok.setdefault((t, l), set()).add(e)
        toks = sorted(set(t for t, _, _ in rows))
        ov = []
        for i in range(1, len(toks)):
            a, b = toks[i - 1], toks[i]
            for l in range(nlay):
                sa, sb = per_tok.get((a, l)), per_tok.get((b, l))
                if sa and sb:
                    ov.append(len(sa & sb) / float(len(sb)))
        if ov:
            out(f"  consecutive-token expert overlap = {100.0*sum(ov)/len(ov):.1f}% "
                f"(mean over {len(ov):,} layer-pairs)")

        if ntok < 16:
            out("  NOTE: this trace is short. Hit rates below are a LOWER BOUND - a cache")
            out("        cannot show reuse that a 12-token window does not contain yet.")

        # ---- cache simulation ----
        out("")
        out(f"  {'cache':>8} {'experts':>9} {'LRU':>8} {'LFU':>8} {'CYCLIC':>8} "
            f"{'GUARD':>8} {'Belady':>8}   {'LRU t/s':>8} {'GRD t/s':>8} {'Belady t/s':>11}")
        mean_sz = sum(cost.values()) / len(cost)
        rows_for_verdict = []
        for gb in sizes:
            cap = int(gb * (1024 ** 3))
            res = {}
            for pol in ("lru", "lfu", "cyclic", "guard", "belady"):
                h, m = simulate(stream, cost, cap, pol, nlayers=nlay, guard=args.guard)
                rate = 100.0 * h / (h + m) if (h + m) else 0.0
                # seconds/token = miss bytes per token / disk rate
                sec = (m / ntok) / (args.disk_mbps * 1024 * 1024)
                res[pol] = (rate, 1.0 / sec if sec > 0 else 0.0)
            out(f"  {gb:>6.0f}GB {cap/mean_sz:>9,.0f} "
                f"{res['lru'][0]:>7.1f}% {res['lfu'][0]:>7.1f}% {res['cyclic'][0]:>7.1f}% "
                f"{res['guard'][0]:>7.1f}% {res['belady'][0]:>7.1f}%   "
                f"{res['lru'][1]:>8.2f} {res['guard'][1]:>8.2f} {res['belady'][1]:>11.2f}")
            rows_for_verdict.append((gb, res))

        out("")
        out(f"  GUARD = LRU that will not evict an entry whose layer is within {args.guard}")
        out("  layers of the current one. tok/s columns are I/O-ONLY UPPER BOUNDS: miss bytes")
        out("  divided by measured disk throughput, zero compute, perfect overlap.")
        out("")

        # How much of the LRU->Belady gap does each implementable policy close?
        # Computed rather than eyeballed off the table, because the sign of the
        # answer is what decides whether N-1's lookahead machinery is needed.
        out("  Share of the LRU-to-Belady gap closed (negative = worse than LRU):")
        for gb, res in rows_for_verdict:
            lo, hi = res['lru'][0], res['belady'][0]
            span = hi - lo
            def closed(v):
                return (100.0 * (v - lo) / span) if span > 1e-9 else float('nan')
            out(f"    {gb:>5.0f}GB  LRU {lo:5.1f}%  Belady {hi:5.1f}%   "
                f"CYCLIC {closed(res['cyclic'][0]):>7.1f}%   "
                f"LFU {closed(res['lfu'][0]):>7.1f}%   "
                f"GUARD {closed(res['guard'][0]):>7.1f}%")
        out("")

        if args.guard_sweep:
            out("  GUARD width sweep (hit rate %):")
            widths = [int(w) for w in args.guard_sweep.split(",") if w.strip()]
            hdr = "    " + f"{'cache':>7}" + "".join(f"{('g=' + str(w)):>8}" for w in widths)
            out(hdr + f"{'LRU':>8}")
            for gb in sizes:
                cap = int(gb * (1024 ** 3))
                cells = []
                for w in widths:
                    h, m = simulate(stream, cost, cap, "guard", nlayers=nlay, guard=w)
                    cells.append(100.0 * h / (h + m) if (h + m) else 0.0)
                hl, ml = simulate(stream, cost, cap, "lru", nlayers=nlay)
                base = 100.0 * hl / (hl + ml) if (hl + ml) else 0.0
                out("    " + f"{gb:>5.0f}GB" + "".join(f"{c:>7.1f}%" for c in cells)
                    + f"{base:>7.1f}%")
            out("")

    out("=" * 92)
    out("HOW TO READ THIS")
    out("  If Belady is far above LRU, a lookahead-driven eviction policy (N-2) is worth")
    out("  building. If they are close, the OS page cache is already near-optimal for this")
    out("  access pattern and N-2 should be cut - the win would have to come from reading")
    out("  fewer bytes (N-3 expert skipping, smaller quant), not from smarter eviction.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT_TXT, "w", encoding="ascii", errors="replace", newline="\n") as f:
        f.write("\n".join(_LOG) + "\n")
    out("")
    out(f"written to: {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
