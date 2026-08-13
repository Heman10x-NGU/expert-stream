#!/usr/bin/env python3
"""
analyze_predictors.py

THE QUESTION THIS ANSWERS
-------------------------
N-2 wants to evict by distance-to-next-use, which needs the future. Belady has
the future and beats LRU by 18.4 points at 6 GB (75.0% vs 56.6%). Two
prediction-free substitutes were built and both LOST to LRU (see
MEASURED_GROUND_TRUTH 12), so N-2 now needs a real expert-identity predictor.

Before anyone builds one, two things are worth knowing, and neither has been
measured:

  1. HOW FAR AHEAD would a predictor have to see to be worth it?
     If one token of lookahead already captures most of Belady's advantage, a
     shallow predictor is enough and N-2 is alive. If it takes 32 tokens, the
     predictor has to be near-perfect over a long horizon and N-2 is dead.

     Policy `beladyK` answers this: Belady, but the oracle goes blind past K
     tokens. It is an UPPER BOUND on any predictor with a K-token horizon,
     because it is a perfect predictor with a K-token horizon.

  2. Does a purely statistical, zero-ML predictor already work?
     Policy `recur` answers this. Each expert gets an exponentially-weighted
     estimate of its own inter-arrival gap, built only from history. Evict the
     one whose predicted next use is furthest away. Implementable today: two
     counters per cache entry, no model, no training, no lookahead.

WHY THIS IS DIFFERENT FROM THE `cyclic` POLICY THAT ALREADY FAILED
`cyclic` predicted WHEN A LAYER comes round, which is exactly known and turned
out to be the wrong signal - only ~36% of a layer's experts repeat between
tokens. `recur` predicts when THAT PARTICULAR EXPERT comes round, which is the
signal Belady actually exploits. It is the smallest possible version of N-1.

THE CONTROL
-----------
This file re-implements LRU and Belady from scratch rather than importing the
simulator. That is deliberate: `lru` and `belady` here MUST reproduce the
numbers analyze_reuse.py already published. If they do not, this tool is wrong
and its new policies mean nothing. The check is printed, not assumed, and a
mismatch is a loud failure rather than a footnote.

Read-only. Touches no model file and launches nothing.
"""

from __future__ import annotations

import argparse
import glob
import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_reuse import load_expert_bytes, load_trace  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "bench", "results")

INF = float("inf")

# Published by analyze_reuse.py on the 220-token trace. The control.
KNOWN = {
    ("lru", 2.0): 35.6, ("lru", 4.0): 48.1, ("lru", 6.0): 56.6,
    ("belady", 2.0): 56.7, ("belady", 4.0): 68.8, ("belady", 6.0): 75.0,
}


def simulate(stream, tok_of, cost, cap_bytes, policy, k_tokens=0, alpha=0.3):
    """stream is [(layer, expert)] in execution order; tok_of[i] is its token.

    Returns (hit_bytes, miss_bytes).

    Eviction always removes the entry with the LARGEST priority, where priority
    means "how far in the future I think this is next needed". LRU is expressed
    in the same frame by using negative recency, so one heap serves every
    policy and the policies differ only in how priority is computed.

    A lazy-deletion heap is used because priorities change on every hit and the
    cache holds ~950 entries at 6 GB against 56,761 accesses - rescanning the
    whole cache per eviction is ~1e9 operations and turns a 20-second
    measurement into an afternoon.
    """
    n = len(stream)

    # ---- oracles, precomputed ------------------------------------------------
    nxt = None
    if policy in ("belady", "beladyk"):
        nxt = [n] * n
        last = {}
        for i in range(n - 1, -1, -1):
            key = stream[i]
            nxt[i] = last.get(key, n)
            last[key] = i

    # ---- recurrence state ----------------------------------------------------
    # gap_est[key] = EWMA of the gap, in ACCESS INDICES, between uses of this
    # exact (layer, expert). last_i[key] = where it was last seen.
    gap_est: dict = {}
    last_i: dict = {}
    # An expert seen only once has no gap estimate. Rather than inventing one,
    # use the running mean of every gap observed so far, which is the least
    # informative honest guess. Before any gap exists at all it is one lap.
    gap_sum = 0.0
    gap_cnt = 0
    default_gap = float(len(set(s[0] for s in stream)) * 6) if stream else 1.0

    def priority(i, key):
        """Larger = evict sooner. Always 'how far away is its next use'."""
        if policy == "lru":
            # Furthest in the future == least recently used, so negate recency.
            return -last_i.get(key, -1)
        if policy == "belady":
            return float(nxt[i]) if nxt[i] < n else INF
        if policy == "beladyk":
            j = nxt[i]
            if j >= n:
                return INF
            # The oracle goes blind past k_tokens. Beyond the horizon it knows
            # nothing, which is INF ("might never come back"), not the true
            # distance. Anything else would smuggle in unbounded lookahead.
            if tok_of[j] - tok_of[i] > k_tokens:
                return INF
            return float(j)
        if policy == "recur":
            g = gap_est.get(key)
            if g is None:
                g = (gap_sum / gap_cnt) if gap_cnt else default_gap
            return float(last_i.get(key, i)) + g
        raise ValueError(policy)

    cached: dict = {}          # key -> current priority
    version: dict = {}         # key -> bump count, for lazy heap deletion
    heap: list = []            # (-priority, seq, key, version)
    used = 0
    hit_b = 0
    miss_b = 0
    seq = 0

    def store(i, key):
        nonlocal seq
        p = priority(i, key)
        cached[key] = p
        version[key] = version.get(key, 0) + 1
        seq += 1
        # -p so heapq's min-heap pops the LARGEST priority. -INF sorts first,
        # which is what we want: "never needed again" is evicted first.
        heapq.heappush(heap, (-p, seq, key, version[key]))

    for i, key in enumerate(stream):
        layer = key[0]
        sz = cost[layer]

        # Update recurrence stats BEFORE the hit/miss decision, so the estimate
        # reflects everything visible at this instant and nothing later.
        if key in last_i:
            g = i - last_i[key]
            gap_sum += g
            gap_cnt += 1
            gap_est[key] = g if key not in gap_est else (
                alpha * g + (1.0 - alpha) * gap_est[key])
        prev_i = last_i.get(key)
        last_i[key] = i

        if key in cached:
            hit_b += sz
            store(i, key)
            continue

        miss_b += sz
        if sz > cap_bytes:
            continue

        while used + sz > cap_bytes and cached:
            # Pop until the top of the heap is a live, current entry.
            victim = None
            while heap:
                negp, _, k, v = heapq.heappop(heap)
                if k in cached and version.get(k) == v:
                    victim = k
                    break
            if victim is None:
                break
            used -= cost[victim[0]]
            del cached[victim]

        if used + sz > cap_bytes:
            continue

        store(i, key)
        used += sz

    return hit_b, miss_b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="")
    ap.add_argument("--sizes-gb", default="2,4,6")
    ap.add_argument("--k-tokens", default="1,2,4,8,16,32",
                    help="lookahead horizons to sweep, in tokens")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="EWMA weight for the recurrence predictor")
    ap.add_argument("--disk-mbps", type=float, default=850.0)
    args = ap.parse_args()

    cost = load_expert_bytes()

    if args.trace:
        tp = args.trace
    else:
        cands = sorted(glob.glob(os.path.join(RESULTS, "moe_trace-*.csv")),
                       key=lambda p: os.path.getsize(p), reverse=True)
        if not cands:
            raise SystemExit("FATAL: no moe_trace-*.csv found.")
        tp = cands[0]

    rows = load_trace(tp)
    if not rows:
        raise SystemExit(f"FATAL: {tp} is empty.")

    stream = [(r[1], r[2]) for r in rows]
    tok_of = [r[0] for r in rows]
    ntok = len(set(tok_of))
    sizes = [float(s) for s in args.sizes_gb.split(",") if s.strip()]
    ks = [int(s) for s in args.k_tokens.split(",") if s.strip()]

    print("analyze_predictors.py - how far ahead would N-2 have to see?")
    print(f"trace   {os.path.basename(tp)}   {ntok} tokens, {len(stream):,} accesses")
    print(f"alpha   {args.alpha} (EWMA weight for `recur`)")
    print()

    # ---------------- the control, before any new number is reported ---------
    print("CONTROL - must reproduce analyze_reuse.py's published numbers")
    print(f"  {'cache':>6} {'policy':>8} {'this tool':>10} {'published':>10} {'delta':>8}")
    ok = True
    for pol in ("lru", "belady"):
        for gb in (2.0, 4.0, 6.0):
            h, m = simulate(stream, tok_of, cost, int(gb * 1024 ** 3), pol)
            got = 100.0 * h / (h + m) if (h + m) else 0.0
            want = KNOWN[(pol, gb)]
            d = got - want
            flag = "" if abs(d) <= 0.15 else "   <== MISMATCH"
            if abs(d) > 0.15:
                ok = False
            print(f"  {gb:>5.0f}G {pol:>8} {got:>9.1f}% {want:>9.1f}% {d:>+7.2f}{flag}")
    print()
    if not ok:
        print("FATAL: the control failed. This tool disagrees with the validated")
        print("simulator, so its new policies mean nothing. Fix before reading on.")
        return 2
    print("control passed - LRU and Belady both reproduce. New policies below.")
    print()

    # ---------------- the actual measurement ---------------------------------
    hdr = f"  {'cache':>6} {'LRU':>7} {'recur':>7}"
    for k in ks:
        hdr += f" {('k=' + str(k)):>7}"
    hdr += f" {'Belady':>7}"
    print("HIT RATE - Belady with a K-token horizon is an UPPER BOUND on any")
    print("predictor that sees K tokens ahead, because it is a perfect one.")
    print(hdr)

    table = []
    for gb in sizes:
        cap = int(gb * 1024 ** 3)
        row = {}
        h, m = simulate(stream, tok_of, cost, cap, "lru")
        row["lru"] = 100.0 * h / (h + m)
        h, m = simulate(stream, tok_of, cost, cap, "recur", alpha=args.alpha)
        row["recur"] = 100.0 * h / (h + m)
        for k in ks:
            h, m = simulate(stream, tok_of, cost, cap, "beladyk", k_tokens=k)
            row[f"k{k}"] = 100.0 * h / (h + m)
        h, m = simulate(stream, tok_of, cost, cap, "belady")
        row["belady"] = 100.0 * h / (h + m)
        row["gb"] = gb
        table.append(row)

        line = f"  {gb:>5.0f}G {row['lru']:>6.1f}% {row['recur']:>6.1f}%"
        for k in ks:
            line += f" {row['k' + str(k)]:>6.1f}%"
        line += f" {row['belady']:>6.1f}%"
        print(line)

    # ---------------- what fraction of the prize each horizon buys -----------
    print()
    print("FRACTION OF THE LRU->BELADY GAP CAPTURED")
    print("  This is the number that decides N-2. A horizon that captures little")
    print("  of the gap cannot be worth building a predictor for, because a")
    print("  PERFECT predictor at that horizon would already not be worth it.")
    hdr = f"  {'cache':>6} {'gap':>7} {'recur':>7}"
    for k in ks:
        hdr += f" {('k=' + str(k)):>7}"
    print(hdr)
    for row in table:
        gap = row["belady"] - row["lru"]
        line = f"  {row['gb']:>5.0f}G {gap:>6.1f}pt"
        if gap <= 0.01:
            line += "   n/a"
        else:
            line += f" {100.0 * (row['recur'] - row['lru']) / gap:>6.0f}%"
            for k in ks:
                line += f" {100.0 * (row[f'k{k}'] - row['lru']) / gap:>6.0f}%"
        print(line)

    # ---------------- speed, since hit rate is not the deliverable -----------
    print()
    print(f"IMPLIED tok/s at {args.disk_mbps:.0f} MB/s  (0.5 / (1 - hit))")
    print(f"  {'cache':>6} {'LRU':>7} {'recur':>7} {'Belady':>7}")
    for row in table:
        f = lambda p: 0.503 / max(1e-9, (1.0 - row[p] / 100.0))
        print(f"  {row['gb']:>5.0f}G {f('lru'):>7.2f} {f('recur'):>7.2f} {f('belady'):>7.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
