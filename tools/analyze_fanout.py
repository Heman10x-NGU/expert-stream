#!/usr/bin/env python3
"""
analyze_fanout.py -- O-6: what does it COST to prefetch experts for F possible next tokens?

WHY THIS EXISTS
  Saguaro (arXiv 2603.03251, "Speculative Speculative Decoding") removes the sequential
  dependence between drafting and verification by pre-speculating for a SET of possible
  verification outcomes, then looking the real outcome up in a cache. Its budget is idle
  draft-GPU compute, which is free to them because the draft sits on separate hardware.

  Our budget is not compute. It is DISK BYTES, which is the one resource we are short of.
  So the same idea only survives the move if covering F possible futures costs much less
  than F times the bytes of covering one. That is the number this script measures.

  Concretely: a token needs top-6 experts in each of 43 layers. If we hedge across F
  candidate next tokens, we must read the UNION of their expert sets. If routing is
  dominated by the prefix rather than by the identity of the last token, that union is
  barely larger than 6 and hedging is nearly free. If routing is dominated by the last
  token, the union approaches 6F and hedging is hopeless.

METHOD
  Each capture is a prefill over a prompt of the form  <shared prefix> <one varying word>.
  Within a group (fanA*, fanB*) every prompt is byte-identical up to the final word.
  We compare routing at the FINAL token position only.

THE BUILT-IN CONTROL (this is the important part)
  Because the prefix bytes are identical across a group, routing at every position EXCEPT
  the last MUST be identical across that group's captures. Attention is causal, so position
  i cannot see token i+1. If that check fails, the capture machinery is nondeterministic and
  every number below is noise. The script says so and refuses to print a verdict.

BASELINES PRINTED ALONGSIDE EVERY RESULT
  random    : union of F independent uniform top-6 draws from 256 experts.
              E[union] = 256 * (1 - (250/256)^F). This is the "no structure at all" case.
  sequential: union over the last F CONSECUTIVE REAL positions of a single capture. This is
              the already-measured batching number, and it is the thing fan-out must beat to
              be worth anything, because batching gets it without guessing.

LAYER GROUPS ARE NOT POOLED
  Layers 0,1,2 use frozen hash routing keyed on the TOKEN ID with no hidden-state input.
  A different final token therefore changes their routing essentially completely, by
  construction, and no amount of prefix sharing can help. Reporting them mixed in with the
  learned layers would understate the learned layers by exactly the amount that is
  interesting. They are reported separately and never averaged together.

HONESTY CAVEAT
  Two prefixes and eight substitutions each is a first data point, not a property of the
  model. Prompt tokens are also not generated tokens. Treat the numbers as directional.

Usage:
  python analyze_fanout.py                 # auto-discovers bench/results/moe_trace-fan*.csv
  python analyze_fanout.py <csv> [<csv>..] # explicit
"""

import glob
import itertools
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(REPO, "bench", "results")

N_LAYERS = 43
TOP_K = 6
N_EXPERTS = 256
HASH_LAYERS = list(range(0, 3))
LEARNED_LAYERS = list(range(3, N_LAYERS))

# Bytes per expert, all three tensors, measured from the GGUF headers:
#   ffn_gate_exps  [4096,2048,256] IQ1_S    1,638,400 B
#   ffn_up_exps    [4096,2048,256] IQ1_S    1,638,400 B
#   ffn_down_exps  [2048,4096,256] IQ3_XXS  3,211,264 B
BYTES_PER_EXPERT = 6_488_064

OUT_TXT = os.path.join(RESULTS_DIR, "fanout_analysis.txt")
OUT_CSV = os.path.join(RESULTS_DIR, "fanout_union.csv")

_log_fh = None


def say(msg=""):
    print(msg)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")


def load_capture(path):
    """Return dict with per-(pos,layer) frozenset of expert ids."""
    df = pd.read_csv(path)
    need = {"token_pos", "layer", "slot", "expert_id"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit("ERROR: %s missing columns %s" % (path, sorted(missing)))

    sets = defaultdict(set)
    for pos, layer, eid in zip(df["token_pos"].values, df["layer"].values, df["expert_id"].values):
        sets[(int(pos), int(layer))].add(int(eid))
    sets = {k: frozenset(v) for k, v in sets.items()}

    positions = sorted({p for (p, _) in sets})
    return {
        "path": path,
        "tag": os.path.basename(path).split("-")[1] if "-" in os.path.basename(path) else os.path.basename(path),
        "sets": sets,
        "positions": positions,
        "T": len(positions),
    }


def group_of(tag):
    # fanA3 -> fanA ; fanB7 -> fanB
    return tag[:4] if len(tag) >= 4 else tag


def prefix_control(caps):
    """Identical prefixes must give identical routing at every position but the last.

    Returns (ok, T_common, n_compared, n_mismatch, first_mismatch)."""
    T_common = min(c["T"] for c in caps)
    if T_common < 2:
        return (False, T_common, 0, 0, "captures have fewer than 2 token positions")

    ref = caps[0]
    n_cmp = 0
    n_bad = 0
    first_bad = None
    # positions 0 .. T_common-2 are prefix positions for every capture in the group
    for pos in range(0, T_common - 1):
        for layer in range(N_LAYERS):
            a = ref["sets"].get((pos, layer))
            if a is None:
                continue
            for other in caps[1:]:
                b = other["sets"].get((pos, layer))
                if b is None:
                    continue
                n_cmp += 1
                if a != b:
                    n_bad += 1
                    if first_bad is None:
                        first_bad = "pos %d layer %d: %s has %s, %s has %s" % (
                            pos, layer, ref["tag"], sorted(a), other["tag"], sorted(b))
    return (n_bad == 0, T_common, n_cmp, n_bad, first_bad)


def mean_union_over_subsets(setlist, F):
    """Exact mean |union| over ALL size-F subsets of setlist. len(setlist) <= 8 so C(8,F)<=70."""
    n = len(setlist)
    if F > n:
        return None
    tot = 0
    cnt = 0
    for combo in itertools.combinations(range(n), F):
        u = set()
        for i in combo:
            u |= setlist[i]
        tot += len(u)
        cnt += 1
    return tot / cnt


def random_union_expectation(F):
    return N_EXPERTS * (1.0 - ((N_EXPERTS - TOP_K) / float(N_EXPERTS)) ** F)


def analyze_group(gname, caps, rows_out):
    say("")
    say("=" * 78)
    say("GROUP %s -- %d captures" % (gname, len(caps)))
    say("=" * 78)
    for c in sorted(caps, key=lambda x: x["tag"]):
        say("  %-8s T=%3d  %s" % (c["tag"], c["T"], os.path.basename(c["path"])))

    caps = sorted(caps, key=lambda x: x["tag"])

    # ---- token-count agreement -------------------------------------------------
    Ts = sorted({c["T"] for c in caps})
    if len(Ts) > 1:
        say("")
        say("  NOTE: captures do not all have the same token count: %s" % Ts)
        say("        A varying word that tokenizes into 2 subwords shifts the final position.")
        say("        The comparison below still uses each capture's OWN final position, which")
        say("        remains 'the routing after a complete candidate continuation', but the")
        say("        candidates are then not all the same length. Flagged, not hidden.")

    # ---- the control -----------------------------------------------------------
    ok, T_common, n_cmp, n_bad, first_bad = prefix_control(caps)
    say("")
    say("  PREFIX CONTROL (identical prefix bytes must give identical routing)")
    say("    positions compared : 0..%d  (%d set comparisons)" % (max(T_common - 2, 0), n_cmp))
    if ok:
        say("    result             : PASS - %d/%d identical" % (n_cmp, n_cmp))
    else:
        say("    result             : FAIL - %d/%d differ" % (n_bad, n_cmp))
        say("    first mismatch     : %s" % first_bad)
        say("")
        say("  REFUSING to report fan-out numbers for this group. If routing is not")
        say("  reproducible across identical prefixes, the union sizes below would be")
        say("  measuring capture noise, not routing structure.")
        return False

    # ---- fan-out union at the FINAL position ------------------------------------
    say("")
    say("  MEAN DISTINCT EXPERTS PER LAYER when hedging across F candidate next tokens")
    say("  (exact mean over all C(%d,F) subsets; 6 = the cost of not hedging)" % len(caps))
    say("")
    say("    %-3s | %-18s | %-18s | %-10s | %s" %
        ("F", "learned L3-42", "hash L0-2", "random", "seq. real tokens"))
    say("    %s" % ("-" * 74))

    # sequential baseline: union over last F consecutive REAL positions, averaged over caps
    def sequential_union(F):
        vals = []
        for c in caps:
            if c["T"] < F:
                continue
            last = c["positions"][-1]
            per_layer = []
            for layer in LEARNED_LAYERS:
                u = set()
                good = True
                for back in range(F):
                    s = c["sets"].get((last - back, layer))
                    if s is None:
                        good = False
                        break
                    u |= s
                if good:
                    per_layer.append(len(u))
            if per_layer:
                vals.append(float(np.mean(per_layer)))
        return float(np.mean(vals)) if vals else None

    for F in range(1, len(caps) + 1):
        per_layer_learned = []
        per_layer_hash = []
        for layer in range(N_LAYERS):
            setlist = []
            for c in caps:
                s = c["sets"].get((c["positions"][-1], layer))
                if s is not None:
                    setlist.append(s)
            if len(setlist) < F:
                continue
            m = mean_union_over_subsets(setlist, F)
            if layer in HASH_LAYERS:
                per_layer_hash.append(m)
            else:
                per_layer_learned.append(m)

        learned = float(np.mean(per_layer_learned)) if per_layer_learned else float("nan")
        hashed = float(np.mean(per_layer_hash)) if per_layer_hash else float("nan")
        rnd = random_union_expectation(F)
        seq = sequential_union(F)
        seq_s = ("%.2f" % seq) if seq is not None else "n/a"

        say("    %-3d | %6.2f  (%5.2fx)    | %6.2f  (%5.2fx)    | %6.2f     | %s" %
            (F, learned, learned / TOP_K, hashed, hashed / TOP_K, rnd, seq_s))

        rows_out.append({
            "group": gname, "F": F,
            "learned_union": round(learned, 4),
            "learned_x": round(learned / TOP_K, 4),
            "hash_union": round(hashed, 4),
            "random_union": round(rnd, 4),
            "sequential_union": round(seq, 4) if seq is not None else "",
        })

    # ---- byte translation --------------------------------------------------------
    say("")
    say("  WHAT THAT COSTS IN BYTES PER TOKEN (43 layers x experts x %.2f MB)" %
        (BYTES_PER_EXPERT / 1024.0 / 1024.0))
    say("")
    say("    %-3s | %-14s | %-12s | %s" % ("F", "MiB/token", "vs F=1", "verdict"))
    say("    %s" % ("-" * 62))
    base = None
    for r in [r for r in rows_out if r["group"] == gname]:
        # weight the two layer groups by their true layer counts
        experts_per_tok = (r["learned_union"] * len(LEARNED_LAYERS)
                           + r["hash_union"] * len(HASH_LAYERS))
        mib = experts_per_tok * BYTES_PER_EXPERT / 1024.0 / 1024.0
        if base is None:
            base = mib
        ratio = mib / base
        if r["F"] == 1:
            verdict = "baseline"
        elif ratio < 1.5:
            verdict = "CHEAP - hedging %d ways costs %.0f%% more bytes" % (r["F"], (ratio - 1) * 100)
        elif ratio < r["F"] * 0.75:
            verdict = "sublinear"
        else:
            verdict = "near-linear - hedging buys nothing"
        say("    %-3d | %10.1f    | %8.2fx   | %s" % (r["F"], mib, ratio, verdict))

    # ---- the comparison that actually decides it ---------------------------------
    #
    # Sublinear growth is NOT the bar. Both of these strategies read a union of expert
    # sets, and both need a predictor to work at all. The difference is what they hand
    # back for the bytes:
    #
    #   hedging F ways   -> read union(F candidate tokens), deliver ONE token
    #   batching N deep  -> read union(N consecutive tokens), deliver N tokens
    #
    # The unions are nearly the same size. The output is not. Bytes per DELIVERED
    # token is the only quantity the 0.503 tok/s ceiling responds to, so that is what
    # gets printed.
    say("")
    say("  BYTES PER *DELIVERED* TOKEN -- hedging vs batching (learned layers 3-42)")
    say("")
    say("    %-3s | %-22s | %-22s | %s" % ("F/N", "hedge F: 1 token", "batch N: N tokens", "batching wins by"))
    say("    %s" % ("-" * 82))
    per_expert_mib = BYTES_PER_EXPERT / 1024.0 / 1024.0 * len(LEARNED_LAYERS)
    for r in [x for x in rows_out if x["group"] == gname]:
        F = r["F"]
        hedge = r["learned_union"] * per_expert_mib          # union(F) bytes, 1 token out
        seq = r["sequential_union"]
        if seq == "":
            continue
        batch = seq * per_expert_mib / float(F)              # union(N) bytes, N tokens out
        say("    %-3d | %10.1f MiB/tok      | %10.1f MiB/tok      | %s" %
            (F, hedge, batch,
             "-" if F == 1 else "%.2fx" % (hedge / batch)))

    return True


def main():
    global _log_fh

    args = sys.argv[1:]
    if args:
        paths = args
    else:
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "moe_trace-fan*.csv")))
    if not paths:
        raise SystemExit("ERROR: no moe_trace-fan*.csv found in %s. Run:\n"
                         "  powershell -NoProfile -File bench\\batch_moe_trace.ps1 "
                         "-PromptGlob \"fan*.txt\"" % RESULTS_DIR)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _log_fh = open(OUT_TXT, "w", encoding="ascii", errors="replace", newline="\n")

    say("O-6 FAN-OUT COST ANALYSIS")
    say("what it costs in expert reads to hedge across F candidate next tokens")
    say("")
    say("CAVEAT, up front: two prefixes, eight substitutions each, prefill positions only.")
    say("This is a first data point, not a property of the model.")

    # keep only the newest capture per tag (re-runs are common)
    by_tag = {}
    for p in paths:
        c = load_capture(p)
        prev = by_tag.get(c["tag"])
        if prev is None or os.path.getmtime(p) > os.path.getmtime(prev["path"]):
            by_tag[c["tag"]] = c

    groups = defaultdict(list)
    for c in by_tag.values():
        groups[group_of(c["tag"])].append(c)

    rows = []
    any_ok = False
    for gname in sorted(groups):
        caps = groups[gname]
        if len(caps) < 2:
            say("")
            say("skipping group %s: only %d capture(s), need at least 2." % (gname, len(caps)))
            continue
        if analyze_group(gname, caps, rows):
            any_ok = True

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        say("")
        say("wrote %s" % OUT_CSV)

    say("")
    say("=" * 78)
    say("HOW TO READ THIS")
    say("=" * 78)
    say("  The 'learned L3-42' column is the one that matters. If hedging 2 ways costs")
    say("  well under 12.00 experts/layer, routing is prefix-dominated and Saguaro-style")
    say("  fan-out transfers to an SSD-bound engine. If it sits near the 'random' column,")
    say("  it does not, and the idea dies here rather than after a week of C.")
    say("")
    say("  Compare against 'seq. real tokens'. Batching over N REAL consecutive tokens")
    say("  gets its union for free, with no guessing and no wasted reads. Fan-out has to")
    say("  beat that to be worth any complexity at all.")
    say("")
    say("  Layers 0-2 are frozen hash routing on the token id. Their column is expected to")
    say("  track 'random' closely. That is the design of the model, not a result.")

    say("")
    say("wrote %s" % OUT_TXT)
    _log_fh.close()

    if not any_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
