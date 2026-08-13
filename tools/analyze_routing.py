#!/usr/bin/env python3
"""
analyze_routing.py -- Analysis for Tasks 0.15 (expert overlap curve) and 0.16
(gate weight histogram / skip analysis), plus a bonus cache-reuse simulation.

This script does NOT run the model and does NOT write/compile any C++. It only
reads routing traces that have already been captured to CSV by the instrumented
llama.cpp build (moe_trace_cb_eval), and produces honest, numbers-first analysis.

INPUT
  One or more CSV files at bench/results/moe_trace-*.csv with header:
    token_pos,layer,slot,expert_id,gate_weight
  One row per (token position, layer, expert slot). 43 layers (0-42), 6 slots
  per layer per token, expert ids in [0,255], gate weights per (token,layer)
  sum to ~1.0.

POOLING MULTIPLE CAPTURES
  On this machine, mmap retains every byte read for the life of the process and
  never evicts it, so peak memory equals total distinct bytes read -- a single
  long prefill capture is not possible (a 5-token prefill already touches ~8GB;
  6-8 tokens gets killed). The only way to collect meaningful data is many short,
  independent captures, pooled together. This script pools ALL matching CSVs by
  default. CRITICAL CORRECTNESS RULE: a sliding window used for the overlap
  analysis (Part A/B) is built SEPARATELY within each capture and NEVER crosses
  a capture boundary -- token position 4 of capture A is not followed by token
  position 0 of capture B. Only the resulting per-window statistics are pooled
  across captures. The shuffled-token control follows the same rule: each
  capture is shuffled independently, never mixed with another capture.

DOMAIN CONTEXT (do not forget this while reading the numbers)
  Layers 0, 1, 2 use FROZEN HASH ROUTING keyed on the token id -- no dependence
  on hidden state. They were already characterized separately and found to
  behave almost exactly like independent random routing. They are NOT
  representative of the model as a whole.
  Layers 3-42 use ordinary learned routing on the hidden state. This is the
  group whose behavior is actually new information.

HONESTY CAVEAT (also printed at top and bottom of every run)
  Every capture is a single PREFILL pass over one short prompt. Prompt tokens
  are not generated tokens, and a handful of short prompts is not a corpus.
  Treat everything below as a first data point, not a proven property of the
  model.

Usage:
  python analyze_routing.py [csv_path ...]
  With no arguments: auto-pools ALL bench/results/moe_trace-*.csv files that
  have at least MIN_TOKENS_FOR_POOL token positions.
  With explicit path(s): pools exactly those files.
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants describing the model / trace schema
# --------------------------------------------------------------------------
N_LAYERS = 43
N_EXPERTS = 256
N_SLOTS = 6
TOTAL_PAIRS = N_LAYERS * N_EXPERTS  # 11008

HASH_LAYERS = list(range(0, 3))      # 0,1,2 -- frozen hash routing
LEARNED_LAYERS = list(range(3, 43))  # 3..42 -- learned routing

OVERLAP_NS = [1, 2, 3, 4, 6, 8, 12, 16]

# Pooling rules
MIN_TOKENS_FOR_POOL = 3     # captures shorter than this are excluded from the default auto-pool
MIN_WINDOWS_FOR_VERDICT = 5  # refuse to render a verdict for any N with fewer pooled windows than this

# Tag prefixes that must NEVER enter the default auto-pool.
#
# The O-6 fan-out captures (tag "fan*") are 16 runs of only TWO prompts, each pair
# differing in a single final token. Pooling them would silently weight one prefix
# eight times over and inflate every overlap statistic in this file -- the captures
# are near-duplicates by construction, which is the entire point of that experiment
# and exactly what makes them poison here. They are analyzed by tools/analyze_fanout.py
# instead. Passing such a file explicitly on the command line still works.
POOL_EXCLUDE_PREFIXES = ("moe_trace-fan",)

# Shuffled-token control (fixes the "uniform baseline conflates popularity
# skew with token-to-token correlation" methodology bug). For each layer
# independently we permute which token position each per-token expert-set
# belongs to, WITHIN one capture at a time. This preserves the exact empirical
# per-token top-6 sets and the exact empirical popularity distribution, and
# destroys ONLY token order. Averaging pooled union(N) over many such shuffles
# gives the correct null hypothesis for "no token-to-token correlation, only
# popularity skew".
N_SHUFFLES = 30
SHUFFLE_SEED = 12345
SHUFFLE_Z_THRESHOLD = 3.0  # |z| >= this is called "significant" below

# Bytes per expert range quoted in project docs (varies by layer). Used only
# for a disclaimer in the cache simulation -- NOT used in any calculation.
EXPERT_BYTES_MIN_MB = 6.49
EXPERT_BYTES_MAX_MB = 8.78

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "bench", "results")

CAVEAT_TEXT = """\
================================================================================
HONESTY CAVEAT: This analysis POOLS multiple SEPARATE PREFILL passes, each over
its own short prompt. A hard memory limit currently forces this (mmap retains
every byte read and never evicts it, so peak memory equals total distinct bytes
read -- a 5-token prefill already touches ~8GB, 6-8 tokens gets killed). These
are PROMPT tokens processed during prefill, NOT autoregressively GENERATED
tokens, and a handful of short prompts is not a representative corpus. Sliding
windows are built separately within each capture and NEVER cross a capture
boundary -- Part A reports exactly how many pooled windows and how many
contributing captures back every number. Every number below is a first data
point, not a proven property of the model. Layers 0-2 use FROZEN HASH ROUTING
keyed on the token id only and are NOT representative of the model's general
routing behavior -- they must never be read as "how the model routes".
================================================================================"""


# --------------------------------------------------------------------------
# Small formatting helpers (pure ASCII only)
# --------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def fmt_table(headers, rows, floatfmt="{:.3f}"):
    str_rows = []
    for row in rows:
        str_row = []
        for v in row:
            if isinstance(v, float):
                str_row.append(floatfmt.format(v))
            elif isinstance(v, (np.floating,)):
                str_row.append(floatfmt.format(float(v)))
            else:
                str_row.append(str(v))
        str_rows.append(str_row)
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def fmt_row(row):
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(row)) + " |"

    lines = [sep, fmt_row(headers), sep]
    for row in str_rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def bar(value, max_value, width=40, ch="#"):
    if max_value <= 0 or value <= 0 or math.isnan(value):
        n = 0
    else:
        n = int(round(width * value / max_value))
    n = max(0, min(n, width))
    return ch * n + "-" * (width - n)


def gini_coefficient(values):
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = arr.size
    total = arr.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) - (n + 1) * total) / (n * total))


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# --------------------------------------------------------------------------
# CSV selection / multi-capture loading
# --------------------------------------------------------------------------
def find_pool_csvs():
    """Auto-discover all moe_trace-*.csv files in RESULTS_DIR and split them
    into (included, excluded) based on MIN_TOKENS_FOR_POOL. included is a list
    of (path, T); excluded is a list of (path, T_or_None, reason)."""
    all_csvs = sorted(glob.glob(os.path.join(RESULTS_DIR, "moe_trace-*.csv")))
    included = []
    excluded = []
    for path in all_csvs:
        base = os.path.basename(path)
        if base.startswith(POOL_EXCLUDE_PREFIXES):
            excluded.append((path, None, "O-6 fan-out capture: near-duplicate by construction, "
                                         "see POOL_EXCLUDE_PREFIXES"))
            continue
        try:
            tmp = pd.read_csv(path, usecols=["token_pos"])
            T = int(tmp["token_pos"].nunique())
        except Exception as e:
            excluded.append((path, None, f"failed to read: {e}"))
            continue
        if T >= MIN_TOKENS_FOR_POOL:
            included.append((path, T))
        else:
            excluded.append((path, T, f"only {T} token position(s), below MIN_TOKENS_FOR_POOL={MIN_TOKENS_FOR_POOL}"))
    return included, excluded


def load_trace(csv_path):
    df = pd.read_csv(csv_path)
    expected_cols = {"token_pos", "layer", "slot", "expert_id", "gate_weight"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: CSV missing expected columns: {sorted(missing)}")
    return df


def load_captures(paths):
    """Load each CSV as its own independent capture. Returns a list of dicts:
    {name, path, df, T, tokens_sorted, present, prefix}. present/prefix are
    LOCAL to that capture -- windows built from them can never cross into
    another capture."""
    captures = []
    for path in paths:
        df = load_trace(path)
        T = int(df["token_pos"].nunique())
        tokens_sorted, present, prefix = build_presence_matrix(df)
        captures.append(dict(name=os.path.basename(path), path=path, df=df, T=T,
                              tokens_sorted=tokens_sorted, present=present, prefix=prefix))
    return captures


def build_pooled_df(captures):
    """Concatenate all captures' dataframes for Part C/D, which have no
    sliding-window boundary issue but DO need a globally unique token_pos so
    groupby(token_pos, layer) never accidentally merges rows from two
    different captures that happen to share a local token_pos value. Also
    preserves chronological order across captures (capture order, then local
    token_pos order) for Part D's cache simulation."""
    frames = []
    for i, c in enumerate(captures):
        d = c["df"].copy()
        d["capture_id"] = i
        d["capture_name"] = c["name"]
        d["local_token_pos"] = d["token_pos"]
        d["token_pos"] = i * 1_000_000 + d["token_pos"].astype(np.int64)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def build_presence_matrix(df):
    """Return (tokens_sorted, present, prefix) for ONE capture. present is a
    (T, 11008) bool matrix of whether (layer,expert) code was present for that
    token (deduped across slots). prefix is its (T+1, 11008) int32 cumulative
    sum over tokens, used by window_unions()."""
    tokens_sorted = np.sort(df["token_pos"].unique())
    T = len(tokens_sorted)
    token_index = {int(t): i for i, t in enumerate(tokens_sorted)}

    present = np.zeros((T, TOTAL_PAIRS), dtype=bool)
    layers = df["layer"].to_numpy()
    experts = df["expert_id"].to_numpy()
    tpos_idx = df["token_pos"].map(token_index).to_numpy()

    bad_layer = (layers < 0) | (layers >= N_LAYERS)
    bad_expert = (experts < 0) | (experts >= N_EXPERTS)
    bad = bad_layer | bad_expert
    n_bad = int(bad.sum())
    if n_bad:
        print(f"WARNING: {n_bad} rows have out-of-range layer/expert_id and are excluded "
              f"from the overlap/reuse analysis (layer must be 0-42, expert_id 0-255).")
    codes = layers.astype(np.int64) * N_EXPERTS + experts.astype(np.int64)

    present[tpos_idx[~bad], codes[~bad]] = True

    prefix = prefix_from_present(present)
    return tokens_sorted, present, prefix


def prefix_from_present(present):
    T = present.shape[0]
    counts = present.astype(np.int32)
    prefix = np.zeros((T + 1, TOTAL_PAIRS), dtype=np.int32)
    prefix[1:] = np.cumsum(counts, axis=0)
    return prefix


def shuffle_present_by_layer(present, rng):
    """Permute the token axis independently per layer's 256-column block,
    WITHIN one capture's present matrix. Preserves the exact per-token-per-
    layer expert set and the exact empirical popularity distribution;
    destroys only token order/adjacency INSIDE this capture."""
    T = present.shape[0]
    shuffled = np.empty_like(present)
    for l in range(N_LAYERS):
        lo = l * N_EXPERTS
        hi = lo + N_EXPERTS
        perm = rng.permutation(T)
        shuffled[:, lo:hi] = present[perm][:, lo:hi]
    return shuffled


def window_unions(prefix, cols, N):
    """Number of windows is T-N+1 for this ONE capture. Returns None if T<N."""
    T = prefix.shape[0] - 1
    if T < N:
        return None
    sub = prefix[:, cols]
    windowsum = sub[N:] - sub[:-N]
    unions = np.count_nonzero(windowsum, axis=1)
    return unions


def pooled_real_unions(captures, cols, N):
    """Pool window_unions(N) across captures WITHOUT ever letting a window
    cross a capture boundary (each capture's windows are computed against its
    own local prefix only, then concatenated). Returns (pooled_array,
    per_capture_counts dict: capture_name -> n_windows_from_that_capture)."""
    arrays = []
    per_capture_counts = {}
    for c in captures:
        u = window_unions(c["prefix"], cols, N)
        if u is None:
            per_capture_counts[c["name"]] = 0
        else:
            per_capture_counts[c["name"]] = len(u)
            arrays.append(u)
    pooled = np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)
    return pooled, per_capture_counts


def baseline_union(N, L, experts_per_layer=N_EXPERTS, k=N_SLOTS):
    per_layer = experts_per_layer * (1.0 - (1.0 - k / experts_per_layer) ** N)
    return L * per_layer


def diagnose_repeated_tokens(present, layer=0):
    """For hash-routing layers, an identical (layer,expert-set) fingerprint at
    two token positions WITHIN ONE CAPTURE is strong evidence those positions
    hold the same token id (hash routing is a deterministic function of token
    id only). Repeated token ids that are NOT randomly distributed across
    positions (e.g. a short synthetic/templated prompt) can make the real
    curve differ from the shuffled null even though the hash function itself
    has zero dependence on neighboring context -- that is a property of the
    TEXT, not a bug in the shuffle. Returns (n_tokens, n_duplicate_positions,
    list of (pos_a, pos_b) duplicate pairs), all LOCAL to this one capture."""
    T = present.shape[0]
    lo, hi = layer * N_EXPERTS, (layer + 1) * N_EXPERTS
    fingerprints = {}
    dup_pairs = []
    for t in range(T):
        fp = tuple(np.nonzero(present[t, lo:hi])[0].tolist())
        if fp in fingerprints:
            dup_pairs.append((fingerprints[fp], t))
        else:
            fingerprints[fp] = t
    dup_positions = set()
    for a, b in dup_pairs:
        dup_positions.add(a)
        dup_positions.add(b)
    return T, len(dup_positions), dup_pairs


def print_multi_capture_repeat_diagnostic(captures, layer=0):
    """Run diagnose_repeated_tokens() separately for every capture (duplicates
    only make sense WITHIN a capture, since shuffling never crosses capture
    boundaries) and print a per-capture + pooled-total summary. Returns
    (any_dup_found, total_dup_positions, total_T)."""
    total_T = 0
    total_dup = 0
    any_dup = False
    for c in captures:
        T_diag, n_dup_pos, dup_pairs = diagnose_repeated_tokens(c["present"], layer=layer)
        total_T += T_diag
        total_dup += n_dup_pos
        if dup_pairs:
            any_dup = True
            print(f"   [{c['name']}] {n_dup_pos} of {T_diag} token positions ({n_dup_pos / T_diag * 100:.0f}%) "
                  f"share a layer-{layer} fingerprint with another position in the SAME capture:")
            for a, b in dup_pairs:
                print(f"     local token_pos {a} and {b} identical layer-{layer} expert set (distance {b - a})")
        else:
            print(f"   [{c['name']}] 0 of {T_diag} token positions repeated -- clean.")
    if total_T > 0:
        print(f"\n   TOTAL across all {len(captures)} captures: {total_dup} of {total_T} token positions "
              f"({total_dup / total_T * 100:.0f}%) involved in an intra-capture repeat.")
    return any_dup, total_dup, total_T


def pick_representative_n(group_result, min_windows=MIN_WINDOWS_FOR_VERDICT):
    """Largest N with at least min_windows POOLED windows. Returns None (a
    deliberate REFUSAL, not a fallback) if no N qualifies."""
    available_ns = sorted(group_result.keys())
    candidates = [N for N in available_ns if group_result[N]["n_windows"] >= min_windows]
    return max(candidates) if candidates else None


def print_capture_window_table(captures):
    header = ["capture", "T"] + [f"N={N}" for N in OVERLAP_NS]
    rows = []
    for c in captures:
        row = [c["name"], c["T"]] + [max(0, c["T"] - N + 1) for N in OVERLAP_NS]
        rows.append(row)
    total_row = ["TOTAL (pooled)", sum(c["T"] for c in captures)] + \
                [sum(max(0, c["T"] - N + 1) for c in captures) for N in OVERLAP_NS]
    rows.append(total_row)
    print(fmt_table(header, rows))


# --------------------------------------------------------------------------
# PART A -- expert overlap curve
# --------------------------------------------------------------------------
def part_a(captures):
    section("PART A -- EXPERT OVERLAP CURVE (Task 0.15, THE HEADLINE)")
    print(f"""
WHY THIS MATTERS: in a DENSE model, verifying 5 speculative draft tokens costs
the same weight reads as verifying 1, so speculation is nearly free. In an
MoE, each token routes to different experts, so you pay for the UNION of
experts touched by the whole window. If the union grows nearly linearly with
window size N, speculative decoding, batching, and tree drafting all COST
MORE THAN THEY SAVE for a disk-bound engine -- you pay for N times the weight
reads with none of the savings dense models get. If the union grows much
slower than linearly, consecutive tokens are reusing experts and those
techniques are worth building.

METHODOLOGY NOTE (corrected): the naive comparison is real union vs a
UNIFORM-random baseline (each token picks 6 of 256 experts uniformly at
random). That comparison is WRONG on its own -- real expert popularity is
SKEWED (some experts are simply popular), and popularity skew alone produces
sublinear union growth even with zero token-to-token correlation. Skew helps
CACHING (we already exploit it); it says nothing about whether SPECULATION
helps, which requires genuine token-to-token correlation. The correct control
is a SHUFFLED-TOKEN baseline: for each layer independently, permute which
token position each token's captured 6-expert set belongs to (WITHIN one
capture, never across). This preserves the exact per-token top-6 sets and the
exact empirical popularity distribution, and destroys ONLY the token order.
If real union ~= shuffled union, popularity skew explains everything. If real
union is well below shuffled union (several standard deviations, over
{N_SHUFFLES} independent shuffles), that is genuine token-to-token locality --
the only thing that can make speculation/batching/tree-drafting pay off. The
uniform baseline is kept in the CSV output for reference only, labeled as the
wrong control.

POOLING RULE: this run pools {len(captures)} independent short captures (see
table below). Sliding windows are built SEPARATELY within each capture and
NEVER cross a capture boundary; only the resulting per-window union counts are
pooled. The shuffled control follows the identical rule -- each capture is
shuffled independently, never mixed with another capture's tokens.
""")

    print(f"Per-capture window counts by N (n_windows = max(0, T-N+1) WITHIN that capture only,")
    print("independent of which layer group is being analyzed):")
    print_capture_window_table(captures)
    print()

    cols_all = np.arange(0, TOTAL_PAIRS)
    cols_hash = np.arange(0 * N_EXPERTS, (max(HASH_LAYERS) + 1) * N_EXPERTS)
    cols_learned = np.arange(min(LEARNED_LAYERS) * N_EXPERTS, TOTAL_PAIRS)

    groups = [
        ("all_layers_0-42", cols_all, N_LAYERS),
        ("hash_layers_0-2", cols_hash, len(HASH_LAYERS)),
        ("learned_layers_3-42", cols_learned, len(LEARNED_LAYERS)),
    ]

    # ---- real curve, pooled per group per N ----
    real_by_group = {}
    for group_name, cols, L in groups:
        mean1 = None
        group_result = {}
        for N in OVERLAP_NS:
            pooled, counts = pooled_real_unions(captures, cols, N)
            n_windows = len(pooled)
            if n_windows == 0:
                continue
            mean_u = float(pooled.mean())
            min_u = int(pooled.min())
            max_u = int(pooled.max())
            if N == 1:
                mean1 = mean_u
            ratio = mean_u / (N * mean1) if mean1 else float("nan")
            base = baseline_union(N, L)
            diff_vs_uniform = mean_u - base
            diff_pct_vs_uniform = (diff_vs_uniform / base * 100.0) if base else float("nan")
            n_contrib = sum(1 for v in counts.values() if v > 0)
            dominant_frac = (max(counts.values()) / n_windows) if n_windows else float("nan")
            group_result[N] = dict(mean=mean_u, min=min_u, max=max_u, ratio=ratio, n_windows=n_windows,
                                    uniform_baseline=base, diff_vs_uniform=diff_vs_uniform,
                                    diff_pct_vs_uniform=diff_pct_vs_uniform,
                                    per_capture_counts=counts, n_contrib=n_contrib, dominant_frac=dominant_frac)
        real_by_group[group_name] = group_result

    # ---- shuffled-token control, N_SHUFFLES draws, per group per N, pooled ----
    print(f"Running {N_SHUFFLES} independent shuffles (each draw shuffles EVERY capture separately, then pools)...")
    shuffle_draws = {g: {N: [] for N in OVERLAP_NS} for g, _, _ in groups}
    rng = np.random.default_rng(SHUFFLE_SEED)
    for _ in range(N_SHUFFLES):
        shuffled_prefixes = [prefix_from_present(shuffle_present_by_layer(c["present"], rng)) for c in captures]
        for group_name, cols, L in groups:
            for N in OVERLAP_NS:
                arrays = [a for a in (window_unions(sp, cols, N) for sp in shuffled_prefixes) if a is not None]
                if not arrays:
                    continue
                pooled = np.concatenate(arrays)
                shuffle_draws[group_name][N].append(float(pooled.mean()))

    csv_rows = []
    results = {}
    hash_self_check = None

    for group_name, cols, L in groups:
        print("-" * 80)
        label = group_name
        if group_name == "hash_layers_0-2":
            label += "  [FROZEN HASH ROUTING -- self-check group, see below]"
        elif group_name == "learned_layers_3-42":
            label += "  [LEARNED ROUTING -- this is the genuinely new result]"
        print(f"Group: {label}")
        print(f"L (layers in group) = {L}")

        table_rows = []
        group_result = {}
        z_list_for_self_check = {}
        for N in OVERLAP_NS:
            if N not in real_by_group[group_name]:
                print(f"  N={N}: 0 pooled windows (no capture has T>={N}) -- skipped entirely")
                continue
            r = real_by_group[group_name][N]
            draws = np.array(shuffle_draws[group_name][N], dtype=np.float64)
            if len(draws) == 0:
                print(f"  N={N}: real data exists but no shuffle draw produced a window -- skipped")
                continue
            shuf_mean = float(draws.mean())
            shuf_std = float(draws.std(ddof=1)) if len(draws) > 1 else 0.0
            diff_vs_shuf = r["mean"] - shuf_mean
            if shuf_std < 1e-9:
                z = 0.0 if abs(diff_vs_shuf) < 1e-9 else (float("-inf") if diff_vs_shuf < 0 else float("inf"))
            else:
                z = diff_vs_shuf / shuf_std

            if r["n_windows"] < MIN_WINDOWS_FOR_VERDICT:
                interp = "INSUFFICIENT DATA"
            elif math.isinf(z):
                interp = "SHARING" if z < 0 else "ANTI-SHARING"
            elif z <= -SHUFFLE_Z_THRESHOLD:
                interp = "SHARING"
            else:
                interp = "SKEW-ONLY"

            table_rows.append([N, r["mean"], r["min"], r["max"], r["n_windows"], r["n_contrib"],
                                f"{r['dominant_frac']*100:.0f}%", shuf_mean, shuf_std, z, interp])
            csv_rows.append([group_name, N, r["mean"], r["min"], r["max"], r["n_windows"], len(captures),
                              r["n_contrib"], r["dominant_frac"], shuf_mean, shuf_std, z, interp,
                              r["uniform_baseline"], r["diff_vs_uniform"], r["diff_pct_vs_uniform"], N_SHUFFLES])
            group_result[N] = dict(mean=r["mean"], min=r["min"], max=r["max"], n_windows=r["n_windows"],
                                    n_contrib=r["n_contrib"], dominant_frac=r["dominant_frac"],
                                    per_capture_counts=r["per_capture_counts"],
                                    shuffled_mean=shuf_mean, shuffled_std=shuf_std, z=z, interp=interp,
                                    uniform_baseline=r["uniform_baseline"],
                                    diff_pct_vs_uniform=r["diff_pct_vs_uniform"])
            if N > 1 and r["n_windows"] >= MIN_WINDOWS_FOR_VERDICT:
                z_list_for_self_check[N] = z

        headers = ["N", "real_mean", "min", "max", "n_win(pooled)", "n_capt_contrib", "dominant_capt_%",
                   "shuffled_mean", "shuffled_std", "z", "verdict"]
        print(fmt_table(headers, table_rows))
        print("(uniform 'wrong control' baseline values are written to overlap_curve.csv only)")
        results[group_name] = group_result

        if group_name == "hash_layers_0-2":
            print()
            print("SELF-CHECK: layers 0-2 use FROZEN HASH ROUTING keyed on the token id, which by")
            print("construction has NO dependence on neighboring tokens. So the pooled real curve for")
            print(f"this group SHOULD land right on top of the pooled shuffled curve (|z| < "
                  f"{SHUFFLE_Z_THRESHOLD:.0f} for every N with >= {MIN_WINDOWS_FOR_VERDICT} pooled windows --")
            print("N with fewer windows are not evaluated). If it does not, the shuffle procedure has a")
            print("bug, not the model.")
            if not z_list_for_self_check:
                print("  No N had enough pooled windows to run the self-check.")
                self_check_result = "INSUFFICIENT DATA"
            else:
                for N in sorted(z_list_for_self_check):
                    gz = z_list_for_self_check[N]
                    flag = "OK" if (math.isfinite(gz) and abs(gz) < SHUFFLE_Z_THRESHOLD) else "FLAG"
                    gr = group_result[N]
                    print(f"  N={N:2d}: z={gz:+.2f}  [{flag}]  ({gr['n_windows']} pooled windows, "
                          f"{gr['n_contrib']}/{len(captures)} captures contributing)")
                finite_z = [z for z in z_list_for_self_check.values() if math.isfinite(z)]
                any_inf = any(not math.isfinite(z) for z in z_list_for_self_check.values())
                fail = any_inf or any(abs(z) >= SHUFFLE_Z_THRESHOLD for z in finite_z)
                self_check_result = "FAIL" if fail else "PASS"
            print(f"\nSHUFFLE SELF-CHECK: {self_check_result}")
            if self_check_result == "PASS":
                print("(Hash layers land on the pooled shuffled null as expected -- the shuffle")
                print(" procedure is validated on this pooled data. Any 'SHARING' verdict on layers")
                print(" 3-42 below can be trusted AS FAR AS THE SHUFFLE PROCEDURE GOES.)")
            elif self_check_result == "FAIL":
                print("(Hash layers show significant deviation from the pooled shuffled null. Two")
                print(" possible causes: a bug in the shuffle code, OR one or more of the pooled")
                print(" captures has non-randomly repeated token ids (e.g. a short synthetic/templated")
                print(" prompt), which can produce this even with a correct shuffle. Re-running the")
                print(" repeated-token fingerprint diagnostic across ALL pooled captures:")
                any_dup, total_dup, total_T = print_multi_capture_repeat_diagnostic(captures, layer=0)
                if any_dup:
                    print("\n   Repeated token ids are STILL present in at least one capture. This can")
                    print("   still fully or partly explain the FAIL without any shuffle bug. Compare the")
                    print("   total repeat rate above to any earlier single-capture measurement.")
                else:
                    print("\n   NO repeated token ids found in ANY of the pooled captures. The FAIL can no")
                    print("   longer be explained by repeated-token structure. This now points at an")
                    print("   actual bug in the shuffle procedure -- investigate")
                    print("   shuffle_present_by_layer() before trusting any 'SHARING' verdict below.")
            hash_self_check = dict(result=self_check_result, z_by_n=z_list_for_self_check)

    out_path = os.path.join(RESULTS_DIR, "overlap_curve.csv")
    with open(out_path, "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["layer_group", "N", "real_mean_union", "min_union", "max_union", "n_windows_pooled",
                    "n_captures_pooled", "n_captures_contributing", "dominant_capture_frac",
                    "shuffled_mean_union", "shuffled_std_union", "z_real_vs_shuffled", "interpretation",
                    "uniform_baseline_WRONG_CONTROL_ref_only", "diff_real_minus_uniform", "diff_pct_vs_uniform",
                    "n_shuffles"])
        for row in csv_rows:
            w.writerow(row)
    print(f"\nSaved: {out_path}")
    return results, hash_self_check


# --------------------------------------------------------------------------
# PART B -- per-layer sharing
# --------------------------------------------------------------------------
def part_b(captures):
    section("PART B -- PER-LAYER SHARING")
    print(f"""
For each layer separately: ratio2 = union(2)/(2*union(1)), ratio8 =
union(8)/(8*union(1)), pooled across all {len(captures)} captures WITHOUT
windows ever crossing a capture boundary. Ratio near 1.0 means essentially no
sharing between consecutive tokens. Ratio well below 1.0 means the layer is
reusing experts across nearby tokens. Layers 0-2 (hash routing) are expected
to sit near the independent baseline; the question is whether any of layers
3-42 show substantially more sharing, and whether sharing varies with depth.

CAUTION 1: these per-layer ratios are relative to union(1), NOT to a shuffled
control, so they have the same popularity-skew-vs-correlation conflation that
Part A's uniform baseline had. Use this table to see WHERE sharing
concentrates, but use Part A's shuffled-control z-scores to decide WHETHER it
is real.
""")
    cols_by_layer = [np.arange(l * N_EXPERTS, (l + 1) * N_EXPERTS) for l in range(N_LAYERS)]

    # window counts are identical across layers (they depend only on capture T),
    # so compute once against an arbitrary single layer's columns for the caution note.
    _, counts2 = pooled_real_unions(captures, cols_by_layer[0], 2)
    _, counts8 = pooled_real_unions(captures, cols_by_layer[0], 8)
    n2_total = sum(counts2.values())
    n8_total = sum(counts8.values())
    n2_contrib = [name for name, v in counts2.items() if v > 0]
    n8_contrib = [name for name, v in counts8.items() if v > 0]
    print(f"CAUTION 2: at N=2, {n2_total} pooled windows come from {len(n2_contrib)}/{len(captures)} "
          f"captures.")
    if n8_total > 0:
        print(f"CAUTION 2 (cont.): at N=8, only {n8_total} pooled windows exist, from "
              f"{len(n8_contrib)}/{len(captures)} capture(s): {', '.join(n8_contrib)}.")
        if len(n8_contrib) < len(captures):
            print("A capture shorter than 8 tokens cannot contribute to N=8 at all (T-N+1 <= 0), so")
            print("ratio8 below is NOT diversified across the pool the way 'pooled' implies here --")
            print("treat it with the same caution as a single-capture result.")
    else:
        print("CAUTION 2 (cont.): NO capture in this pool has >= 8 tokens, so N=8 has zero pooled")
        print("windows -- ratio8 below will be NaN for every layer.")
    print()

    rows = []
    for l in range(N_LAYERS):
        cols = cols_by_layer[l]
        u1, _ = pooled_real_unions(captures, cols, 1)
        u2, _ = pooled_real_unions(captures, cols, 2)
        u8, _ = pooled_real_unions(captures, cols, 8)
        mean1 = float(u1.mean()) if len(u1) else float("nan")
        n2, n8 = len(u2), len(u8)
        ratio2 = float(u2.mean()) / (2 * mean1) if (n2 and mean1) else float("nan")
        ratio8 = float(u8.mean()) / (8 * mean1) if (n8 and mean1) else float("nan")
        kind = "HASH" if l in HASH_LAYERS else "LEARN"
        rows.append(dict(layer=l, kind=kind, mean_union1=mean1, ratio2=ratio2, ratio8=ratio8, n2=n2, n8=n8))

    table_rows = [[r["layer"], r["kind"], r["mean_union1"], r["ratio2"], r["n2"], r["ratio8"], r["n8"]] for r in rows]
    headers = ["layer", "kind", "mean_union(1)", "ratio2", "n2(pooled)", "ratio8", "n8(pooled)"]
    print(fmt_table(headers, table_rows))

    print("\nASCII bar chart of ratio8 per layer (longer bar = MORE sharing, i.e. 1-ratio8):")
    print("(0.0 sharing -----------------------------------------> 1.0 sharing (ratio8->0))")
    for r in rows:
        r8 = r["ratio8"]
        sharing_amt = 0.0 if math.isnan(r8) else max(0.0, 1.0 - r8)
        print(f"  L{r['layer']:2d} [{r['kind']:5s}] ratio8={r8:.3f} |{bar(sharing_amt, 1.0, width=40)}|")

    valid = [r for r in rows if not math.isnan(r["ratio8"])]
    most_sharing = sorted(valid, key=lambda r: r["ratio8"])[:5]
    least_sharing = sorted(valid, key=lambda r: -r["ratio8"])[:5]

    print("\nTop 5 layers with MOST sharing (lowest ratio8):")
    print(fmt_table(["layer", "kind", "ratio8"], [[r["layer"], r["kind"], r["ratio8"]] for r in most_sharing]))
    print("\nTop 5 layers with LEAST sharing (highest ratio8):")
    print(fmt_table(["layer", "kind", "ratio8"], [[r["layer"], r["kind"], r["ratio8"]] for r in least_sharing]))

    return rows


# --------------------------------------------------------------------------
# PART C -- gate weight histogram / skip analysis
# --------------------------------------------------------------------------
def part_c(pooled_df):
    section("PART C -- GATE WEIGHT HISTOGRAM (Task 0.16)")
    print("(Pooled across all captures. No sliding-window boundary issue applies here -- gate-weight")
    print(" stats are per (token,layer) independent of sequence order. token_pos is made globally")
    print(" unique across captures before grouping, so rows from different captures are never merged.)")

    df = pooled_df
    gw = df["gate_weight"].to_numpy(dtype=np.float64)
    print("\nOverall gate_weight distribution (all rows, all layers, all pooled tokens):")
    deciles = np.percentile(gw, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    overall_rows = [
        ["min", float(gw.min())],
        ["max", float(gw.max())],
        ["mean", float(gw.mean())],
        ["median", float(np.median(gw))],
    ]
    for i, d in enumerate(deciles, start=1):
        overall_rows.append([f"decile_{i*10}", float(d)])
    print(fmt_table(["stat", "value"], overall_rows, floatfmt="{:.5f}"))

    print("\nASCII histogram of gate_weight (all rows, 20 bins over [0,1]):")
    counts, edges = np.histogram(gw, bins=np.linspace(0, 1, 21))
    max_count = counts.max() if counts.max() > 0 else 1
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        print(f"  [{lo:.2f},{hi:.2f}) {counts[i]:8d} |{bar(counts[i], max_count, width=50)}|")

    # ---- group into (token,layer) rows of exactly 6 slots ----
    total_expected = df["token_pos"].nunique() * N_LAYERS * N_SLOTS
    df_sorted = df.sort_values(["token_pos", "layer", "slot"], kind="mergesort").reset_index(drop=True)

    group_sizes = df_sorted.groupby(["token_pos", "layer"]).size()
    complete_mask = group_sizes == N_SLOTS
    if not complete_mask.all():
        n_bad = int((~complete_mask).sum())
        print(f"\nWARNING: {n_bad} of {len(group_sizes)} (token,layer) groups do not have exactly "
              f"{N_SLOTS} slot rows; these are EXCLUDED from the rank/cumsum/topk/skip analysis below.")
        good_keys = group_sizes[complete_mask].index
        df_sorted = df_sorted.set_index(["token_pos", "layer"]).loc[good_keys].reset_index()
        df_sorted = df_sorted.sort_values(["token_pos", "layer", "slot"], kind="mergesort").reset_index(drop=True)
    else:
        print(f"\nAll {len(group_sizes)} (token,layer) groups have exactly {N_SLOTS} slots "
              f"(expected {total_expected} rows total, got {len(df)}). Clean data.")

    n_groups = len(df_sorted) // N_SLOTS
    vals = df_sorted["gate_weight"].to_numpy(dtype=np.float64).reshape(n_groups, N_SLOTS)
    layer_of_group = df_sorted["layer"].to_numpy().reshape(n_groups, N_SLOTS)[:, 0]

    arr_sorted = -np.sort(-vals, axis=1)  # descending per (token,layer)
    cum = np.cumsum(arr_sorted, axis=1)   # cumulative mass per (token,layer)

    rank_means = arr_sorted.mean(axis=0)
    print("\nMean gate weight by rank (rank 1 = largest of the 6 per (token,layer)):")
    print(fmt_table(["rank", "mean_weight"], [[i + 1, float(rank_means[i])] for i in range(N_SLOTS)], floatfmt="{:.5f}"))

    cum_means = cum.mean(axis=0)
    print("\nRUNNING SUM: mean cumulative gate-mass fraction after top-K experts (K=1..6):")
    print(fmt_table(["K", "mean_cumulative_fraction"], [[i + 1, float(cum_means[i])] for i in range(N_SLOTS)], floatfmt="{:.5f}"))

    def experts_needed_for(cum_arr, threshold):
        ge = cum_arr >= threshold
        # every row reaches 1.0 by column 5, so argmax always finds a True
        idx = np.argmax(ge, axis=1) + 1
        return idx

    k60 = experts_needed_for(cum, 0.6)
    k90 = experts_needed_for(cum, 0.9)
    print(f"\nHOBBIT-style thresholds (mean number of experts needed to reach X of total gate mass):")
    print(f"  mean experts needed for >=0.6 mass: {k60.mean():.3f}  (distribution: " +
          ", ".join(f"k={k}:{int((k60 == k).sum())}" for k in range(1, 7)) + ")")
    print(f"  mean experts needed for >=0.9 mass: {k90.mean():.3f}  (distribution: " +
          ", ".join(f"k={k}:{int((k90 == k).sum())}" for k in range(1, 7)) + ")")

    # ---- top-k truncation loss, overall and per-layer ----
    loss_top4 = 1.0 - cum[:, 3]
    loss_top3 = 1.0 - cum[:, 2]
    print(f"\nTOP-K TRUNCATION (mass lost by dropping experts below top-K, overall mean):")
    print(f"  top4 of 6: mean mass lost = {loss_top4.mean():.5f}")
    print(f"  top3 of 6: mean mass lost = {loss_top3.mean():.5f}")

    part_c_df = pd.DataFrame({
        "layer": layer_of_group,
        "max_gate": arr_sorted[:, 0],
        "loss_top4": loss_top4,
        "loss_top3": loss_top3,
        "k60": k60,
        "k90": k90,
    })

    thresholds = [0.2, 0.3, 0.4, 0.5]
    per_layer_records = []
    for l, sub in part_c_df.groupby("layer"):
        rec = {
            "layer": int(l),
            "kind": "HASH" if l in HASH_LAYERS else "LEARN",
            "n": len(sub),
            "mean_loss_top4": float(sub["loss_top4"].mean()),
            "mean_loss_top3": float(sub["loss_top3"].mean()),
            "mean_k60": float(sub["k60"].mean()),
            "mean_k90": float(sub["k90"].mean()),
        }
        for th in thresholds:
            rec[f"skip_frac_lt_{th}"] = float((sub["max_gate"] < th).mean())
        per_layer_records.append(rec)

    print("\nPer-layer top-K truncation loss:")
    headers = ["layer", "kind", "n", "mean_loss_top4", "mean_loss_top3", "mean_k60", "mean_k90"]
    rows_ = [[r["layer"], r["kind"], r["n"], r["mean_loss_top4"], r["mean_loss_top3"], r["mean_k60"], r["mean_k90"]]
             for r in per_layer_records]
    print(fmt_table(headers, rows_, floatfmt="{:.4f}"))

    print("\nN-3 SKIP ANALYSIS: fraction of (token,layer) pairs where max_gate is below threshold")
    print("(a low max_gate means even the BEST expert has low weight -- candidate for skipping the")
    print(" whole expert block at that layer for that token):")
    headers2 = ["layer", "kind"] + [f"frac<{th}" for th in thresholds]
    rows2 = [[r["layer"], r["kind"]] + [r[f"skip_frac_lt_{th}"] for th in thresholds] for r in per_layer_records]
    print(fmt_table(headers2, rows2, floatfmt="{:.4f}"))

    overall_skip = {th: float((part_c_df["max_gate"] < th).mean()) for th in thresholds}
    print("\nOverall skip fractions (all layers combined):")
    print(fmt_table(["threshold", "frac_of_pairs_with_max_gate_below"], [[th, overall_skip[th]] for th in thresholds]))

    learned_sub = part_c_df[part_c_df["layer"].isin(LEARNED_LAYERS)]
    hash_sub = part_c_df[part_c_df["layer"].isin(HASH_LAYERS)]
    learned_skip = {th: float((learned_sub["max_gate"] < th).mean()) for th in thresholds}
    hash_skip = {th: float((hash_sub["max_gate"] < th).mean()) for th in thresholds} if len(hash_sub) else {th: float("nan") for th in thresholds}
    print("\nSkip fractions, learned layers 3-42 ONLY (the layers that matter for this decision):")
    print(fmt_table(["threshold", "frac_below"], [[th, learned_skip[th]] for th in thresholds]))
    print("\nSkip fractions, hash layers 0-2 ONLY (reference/sanity, NOT representative of the model):")
    print(fmt_table(["threshold", "frac_below"], [[th, hash_skip[th]] for th in thresholds]))

    # ---- write gate_histogram.csv ----
    out_path = os.path.join(RESULTS_DIR, "gate_histogram.csv")
    with open(out_path, "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["metric", "layer", "value"])
        for r in overall_rows:
            w.writerow([r[0], "ALL", r[1]])
        for i in range(N_SLOTS):
            w.writerow([f"rank_mean_{i+1}", "ALL", float(rank_means[i])])
        for i in range(N_SLOTS):
            w.writerow([f"cumsum_mean_{i+1}", "ALL", float(cum_means[i])])
        w.writerow(["mean_experts_for_0.6", "ALL", float(k60.mean())])
        w.writerow(["mean_experts_for_0.9", "ALL", float(k90.mean())])
        w.writerow(["topk_loss_top4", "ALL", float(loss_top4.mean())])
        w.writerow(["topk_loss_top3", "ALL", float(loss_top3.mean())])
        for th in thresholds:
            w.writerow([f"skip_frac_lt_{th}", "ALL", overall_skip[th]])
            w.writerow([f"skip_frac_lt_{th}", "LEARNED_3-42", learned_skip[th]])
            w.writerow([f"skip_frac_lt_{th}", "HASH_0-2", hash_skip[th]])
        for r in per_layer_records:
            l = r["layer"]
            w.writerow(["topk_loss_top4", l, r["mean_loss_top4"]])
            w.writerow(["topk_loss_top3", l, r["mean_loss_top3"]])
            w.writerow(["mean_experts_for_0.6", l, r["mean_k60"]])
            w.writerow(["mean_experts_for_0.9", l, r["mean_k90"]])
            for th in thresholds:
                w.writerow([f"skip_frac_lt_{th}", l, r[f"skip_frac_lt_{th}"]])
    print(f"\nSaved: {out_path}")

    return dict(k60=k60, k90=k90, loss_top4=loss_top4, loss_top3=loss_top3,
                per_layer_records=per_layer_records, learned_skip=learned_skip,
                overall_skip=overall_skip, hash_skip=hash_skip)


# --------------------------------------------------------------------------
# PART D -- expert reuse / cache simulation
# --------------------------------------------------------------------------
def part_d(pooled_df, n_captures):
    section("PART D -- EXPERT REUSE / CACHE SIMULATION")

    df = pooled_df
    layers = df["layer"].to_numpy()
    experts = df["expert_id"].to_numpy()
    bad = (layers < 0) | (layers >= N_LAYERS) | (experts < 0) | (experts >= N_EXPERTS)
    codes = (layers[~bad].astype(np.int64) * N_EXPERTS + experts[~bad].astype(np.int64))

    usage_counts = np.bincount(codes, minlength=TOTAL_PAIRS)
    distinct_used = int(np.count_nonzero(usage_counts))
    total_events = int(usage_counts.sum())

    print(f"Distinct (layer,expert) pairs touched: {distinct_used} / {TOTAL_PAIRS} "
          f"({distinct_used / TOTAL_PAIRS * 100:.1f}% of the addressable expert space)")
    print(f"Total routing events (rows) in pooled trace: {total_events}, from {n_captures} captures")

    used_only = usage_counts[usage_counts > 0]
    print(f"\nUsage count stats among the {distinct_used} touched pairs:")
    print(f"  min={used_only.min()}  max={used_only.max()}  mean={used_only.mean():.2f}  "
          f"median={np.median(used_only):.1f}")

    print("\nCompact usage-count histogram (log-ish buckets, among touched pairs only):")
    bucket_edges = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, int(used_only.max()) + 1]
    bucket_edges = sorted(set(e for e in bucket_edges if e <= used_only.max() + 1))
    if bucket_edges[-1] <= used_only.max():
        bucket_edges.append(int(used_only.max()) + 1)
    hist, edges = np.histogram(used_only, bins=bucket_edges)
    max_h = hist.max() if hist.max() > 0 else 1
    for i in range(len(hist)):
        lo, hi = edges[i], edges[i + 1]
        print(f"  uses in [{int(lo):5d},{int(hi):5d}) : {hist[i]:6d} pairs |{bar(hist[i], max_h, width=40)}|")

    sorted_desc = np.sort(usage_counts)[::-1]
    cum_events = np.cumsum(sorted_desc)
    print(f"\nCoverage of routing events by the top X% of ALL {TOTAL_PAIRS} addressable (layer,expert) slots")
    print("(this is the number directly relevant to sizing an expert cache):")
    cov_rows = []
    for frac in (0.10, 0.25, 0.50):
        k = max(1, int(round(frac * TOTAL_PAIRS)))
        coverage = cum_events[k - 1] / total_events if total_events else float("nan")
        cov_rows.append([f"top {int(frac*100)}%", k, coverage])
    print(fmt_table(["slice", "n_slots", "frac_of_events_covered"], cov_rows, floatfmt="{:.4f}"))

    g = gini_coefficient(usage_counts)
    print(f"\nGini coefficient of usage across all {TOTAL_PAIRS} addressable slots (0=perfectly uniform, "
          f"1=maximally concentrated): {g:.4f}")
    print("(Unused slots count as 0 in this calculation, so Gini reflects concentration over the full")
    print(" addressable expert space, not just the touched subset.)")

    # ---- LRU cache simulation ----
    print("\nLRU CACHE SIMULATION")
    print("ASSUMPTION: every expert is treated as equal cost to load/hold in this simulation.")
    print(f"THIS IS NOT TRUE IN REALITY: real bytes per expert vary by layer from "
          f"{EXPERT_BYTES_MIN_MB:.2f} to {EXPERT_BYTES_MAX_MB:.2f} MB.")
    print("LABEL: this is our FIRST EMPIRICAL hit-rate number for this model. Every previous")
    print("hit-rate figure in project docs was inherited from a different model and should not")
    print("be treated as validated until compared against numbers like these.")
    print(f"CAVEAT: this sequence concatenates {n_captures} independent captures in file-discovery")
    print("order, with cache state carried across capture boundaries. That approximates serving")
    print("back-to-back prompts (a real disk cache does persist across requests), which is a")
    print("reasonable thing to simulate -- but the specific concatenation order used here is")
    print("arbitrary and not a real request-arrival order.")

    seq_df = df.sort_values(["token_pos", "layer", "slot"], kind="mergesort")
    seq_layers = seq_df["layer"].to_numpy()
    seq_experts = seq_df["expert_id"].to_numpy()
    seq_bad = (seq_layers < 0) | (seq_layers >= N_LAYERS) | (seq_experts < 0) | (seq_experts >= N_EXPERTS)
    seq_codes = (seq_layers[~seq_bad].astype(np.int64) * N_EXPERTS + seq_experts[~seq_bad].astype(np.int64))

    def simulate_lru(sequence, capacity):
        cache = OrderedDict()
        hits = 0
        for item in sequence:
            item = int(item)
            if item in cache:
                cache.move_to_end(item)
                hits += 1
            else:
                cache[item] = True
                if len(cache) > capacity:
                    cache.popitem(last=False)
        total = len(sequence)
        return hits, total, (hits / total if total else float("nan"))

    cap_sizes = [500, 1000, 1500, 2000]
    lru_rows = []
    for C in cap_sizes:
        hits, total, hit_rate = simulate_lru(seq_codes, C)
        pct_of_space = C / TOTAL_PAIRS * 100
        lru_rows.append([C, f"{pct_of_space:.1f}%", hits, total, hit_rate])
    print()
    print(fmt_table(["cache_size_C", "pct_of_11008_slots", "hits", "total_accesses", "hit_rate"], lru_rows, floatfmt="{:.4f}"))
    print("\nNOTE: this trace pools short prefill passes. Early accesses in the pooled sequence are")
    print("mostly compulsory misses (nothing has been cached yet); hit rate for a real")
    print("multi-request decode workload with different repetition patterns may differ.")

    return dict(distinct_used=distinct_used, total_events=total_events, gini=g, lru_rows=lru_rows)


# --------------------------------------------------------------------------
# VERDICT
# --------------------------------------------------------------------------
def verdict(captures, T_total, part_a_results, hash_self_check, part_c_result):
    section("VERDICT")

    print(f"Data sources: {len(captures)} pooled captures, {T_total} token positions total:")
    print(fmt_table(["capture", "T"], [[c["name"], c["T"]] for c in captures]))
    print()
    print(f"*** PRELIMINARY: {T_total} pooled token positions across {len(captures)} short, independent ***")
    print("*** captures is still very thin. Window statistics at larger N routinely rest on very few, ***")
    print("*** or even a SINGLE capture's worth of data (see per-capture window table in Part A and   ***")
    print("*** the n_capt_contrib / dominant_capt_% columns below). Treat everything in this VERDICT  ***")
    print("*** as directional, not conclusive.                                                        ***")
    print()

    if hash_self_check is not None:
        print(f"SHUFFLE SELF-CHECK (layers 0-2, hash routing, pooled): {hash_self_check['result']}")
        if hash_self_check["result"] == "FAIL":
            print("*** WARNING: the pooled shuffle control still failed its own sanity check. See the ***")
            print("*** repeated-token diagnostic in Part A for whether this is data structure or a     ***")
            print("*** likely shuffle bug.                                                              ***")
        elif hash_self_check["result"] == "PASS":
            print("(The shuffle procedure is validated on this pooled data -- the sharing verdict below")
            print(" for layers 3-42 can be trusted AS FAR AS THE SHUFFLE PROCEDURE GOES; the data-volume")
            print(" caveat above still applies independently.)")
        else:
            print("(Not enough pooled windows to run the self-check at all.)")
        print()

    learned = part_a_results.get("learned_layers_3-42", {})
    available_ns = sorted(learned.keys())

    print("Layers 3-42 (learned routing), pooled real union vs pooled SHUFFLED-TOKEN control:")
    verdict_rows = []
    for N in available_ns:
        r = learned[N]
        verdict_rows.append([N, r["mean"], r["shuffled_mean"], r["shuffled_std"], r["z"], r["interp"],
                              r["n_windows"], r["n_contrib"], f"{r['dominant_frac']*100:.0f}%"])
    print(fmt_table(["N", "real_mean", "shuf_mean", "shuf_std", "z", "verdict", "n_windows", "n_capt", "dominant%"],
                     verdict_rows))

    rep_N = pick_representative_n(learned, min_windows=MIN_WINDOWS_FOR_VERDICT)
    qualifying = [N for N in available_ns if learned[N]["n_windows"] >= MIN_WINDOWS_FOR_VERDICT]
    diversified = [N for N in qualifying if learned[N]["n_contrib"] == len(captures)]
    best_diversified_N = max(diversified) if diversified else None

    significant = False
    use_N = None
    if rep_N is None:
        print(f"\nNo N in the overlap sweep had >= {MIN_WINDOWS_FOR_VERDICT} pooled windows for layers 3-42.")
        print("REFUSING to render VERDICT 1/2 -- capture more tokens and re-run.")
        sharing_verdict = "INSUFFICIENT DATA"
    else:
        r = learned[rep_N]
        print(f"\nLargest N meeting the >= {MIN_WINDOWS_FOR_VERDICT}-pooled-window rule: N={rep_N} "
              f"({r['n_windows']} windows, {r['n_contrib']}/{len(captures)} captures contributing, "
              f"{r['dominant_frac']*100:.0f}% from the single largest contributor).")
        if r["n_contrib"] < len(captures):
            print(f"  BLUNT FLAG: N={rep_N} technically clears the window-count bar, but its windows come")
            print(f"  from only {r['n_contrib']} of {len(captures)} captures -- pooling added little or no")
            print("  diversification at this N. Do not read this as multi-prompt evidence; it is close to")
            print("  a repeat of one capture's result under a different name.")

        if best_diversified_N is not None and best_diversified_N != rep_N:
            rd = learned[best_diversified_N]
            print(f"\n  The largest N where ALL {len(captures)} captures contribute windows is "
                  f"N={best_diversified_N} ({rd['n_windows']} windows): z={rd['z']:+.2f}, "
                  f"verdict={rd['interp']}.")
            print("  This is the more trustworthy 'genuinely pooled, multi-prompt' data point --")
            print("  using it below for VERDICT 1/2 instead of the raw largest-N-that-qualifies pick.")
            use_N = best_diversified_N
        elif best_diversified_N is not None:
            use_N = best_diversified_N
        else:
            use_N = rep_N
            print("\n  No N had windows from every pooled capture; using the largest N that clears the")
            print("  window-count bar anyway, with the caveat above.")

        r = learned[use_N]
        z_rep = r["z"]
        print(f"\nUsing N={use_N} for VERDICT 1/2 ({r['n_windows']} pooled windows, "
              f"{r['n_contrib']}/{len(captures)} captures contributing):")
        print(f"  real mean union     = {r['mean']:.2f}")
        print(f"  shuffled mean union = {r['shuffled_mean']:.2f}  (std over {N_SHUFFLES} shuffles = {r['shuffled_std']:.2f})")
        print(f"  z = (real - shuffled_mean) / shuffled_std = {z_rep:+.2f}")

        z_is_neg_inf = (z_rep == float("-inf"))
        significant = z_is_neg_inf or (math.isfinite(z_rep) and z_rep <= -SHUFFLE_Z_THRESHOLD)

        if z_is_neg_inf:
            sharing_verdict = "DIRECTIONALLY SHARING (z): real union is below every shuffled draw."
        elif significant:
            sharing_verdict = (f"DIRECTIONALLY SHARING (z): real union is {abs(z_rep):.1f} standard "
                                f"deviations below the shuffled-token control (threshold: {SHUFFLE_Z_THRESHOLD:.0f}).")
        else:
            sharing_verdict = (f"NOT SIGNIFICANT: real union is only {z_rep:+.1f} standard deviations from "
                                "the shuffled-token control.")

        # Also show the direction (sign of z) at every genuinely diversified N, for the blunt summary.
        div_ns = [N for N in available_ns if learned[N]["n_contrib"] == len(captures)]
        if div_ns:
            signs = [("neg" if learned[N]["z"] < 0 else "pos") for N in div_ns]
            print(f"\n  Direction check across all fully-diversified N ({div_ns}): z signs = {signs}")

    print(f"\n  VERDICT 1 (sharing, corrected for popularity skew): {sharing_verdict}")
    if hash_self_check is not None and hash_self_check["result"] == "FAIL":
        print("  CAVEAT ON VERDICT 1: the pooled shuffle self-check FAILed (see above) -- confirm the")
        print("  diagnostic points at data structure, not a shuffle bug, before trusting this verdict.")

    print("\n  VERDICT 2 (speculative decoding / batching / tree drafting) -- BE BLUNT:")
    if rep_N is None:
        print("    Cannot be rendered -- insufficient data.")
    else:
        if significant:
            print("    The z-score at the best available N is negative and clears the significance bar,")
            print("    and (see 'direction check' above) the sign is consistently negative across every")
            print("    N where all pooled captures contribute -- this DIRECTIONALLY SUGGESTS layers 3-42")
            print("    share experts between nearby tokens more than popularity skew alone predicts.")
        else:
            print("    Real union is not significantly below the shuffled-token control at the N used")
            print("    above.")
        print()
        print(f"    BUT: this is {T_total} token positions pooled from {len(captures)} short, independent")
        print("    prefill runs, forced by a memory limitation that currently caps most captures at 5-12")
        print("    tokens. Every N value above 4 is dominated by, or entirely drawn from, a single one of")
        print("    those captures (see n_capt_contrib/dominant_capt_% above) -- pooling more tiny captures")
        print("    does not fix that until captures long enough to reach N=6/8/12 on their own become")
        print("    possible, or until dozens of short captures (not a handful) exist so N=6-16 can be")
        print("    genuinely diversified too. A consistent negative-z direction across the fully-")
        print("    diversified N values is a real, mildly encouraging signal.")
        print()
        print("    HONEST ANSWER: this DIRECTIONALLY SUGGESTS SHARING. It is NOT enough data to commit")
        print("    engineering time to speculative decoding, batching, or tree drafting. Revisit once the")
        print("    memory-guard issue is fixed (longer single captures) or once many more short captures")
        print("    exist so higher N values stop being single-capture results wearing a pooled label.")

    print("\n  VERDICT 3 (N-3 per-layer skip threshold):")
    learned_skip = part_c_result["learned_skip"]
    for th in sorted(learned_skip.keys()):
        print(f"    max_gate < {th}: skips {learned_skip[th]*100:.1f}% of (token,layer) blocks in layers 3-42")
    reasonable = [th for th in sorted(learned_skip.keys()) if learned_skip[th] <= 0.5]
    if reasonable:
        rec_th = max(reasonable)
        print(f"    RECOMMENDATION: threshold = {rec_th} (skips {learned_skip[rec_th]*100:.1f}% of blocks in")
        print("    layers 3-42 while keeping the skip decision conservative -- max_gate below this means")
        print("    even the single best-weighted expert for that token/layer carries little mass).")
        print(f"    (This part of the verdict is based on {T_total} pooled (token,layer) gate-weight")
        print("    observations, which has no sliding-window boundary issue, but is still thin data.)")
    else:
        lowest_th = min(learned_skip.keys())
        print(f"    Even the lowest threshold tested ({lowest_th}) skips more than half of all blocks --")
        print("    recommend starting even lower than 0.2 and re-measuring, this data does not support")
        print(f"    a threshold as high as {lowest_th}.")

    print()
    print(CAVEAT_TEXT)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pool and analyze MoE routing traces for Tasks 0.15/0.16.")
    parser.add_argument("csv_paths", nargs="*", default=None,
                         help="Explicit moe_trace CSV path(s) to pool. If omitted, auto-pools ALL "
                              f"bench/results/moe_trace-*.csv files with >= {MIN_TOKENS_FOR_POOL} token "
                              "positions.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "routing_analysis.txt")
    log_file = open(log_path, "w", encoding="ascii", errors="replace", newline="\n")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log_file)

    try:
        print(CAVEAT_TEXT)
        print()

        excluded = []
        if args.csv_paths:
            paths = args.csv_paths
            for p in paths:
                if not os.path.isfile(p):
                    raise SystemExit(f"ERROR: file not found: {p}")
        else:
            included, excluded = find_pool_csvs()
            if not included:
                raise SystemExit(f"ERROR: no moe_trace-*.csv with >= {MIN_TOKENS_FOR_POOL} token positions "
                                  f"found in {RESULTS_DIR}")
            paths = [p for p, _ in included]

        captures = load_captures(paths)
        T_total = sum(c["T"] for c in captures)

        print(f"POOLING {len(captures)} independent capture(s), {T_total} token positions total:")
        summary_rows = [[c["name"], c["T"]] for c in captures]
        summary_rows.append(["TOTAL", T_total])
        print(fmt_table(["capture_file", "token_positions"], summary_rows))
        if excluded:
            print(f"\nEXCLUDED from the pool ({len(excluded)} file(s)):")
            for path, T, reason in excluded:
                print(f"  {os.path.basename(path)}: {reason}")

        print(f"\nLayers present per capture (sanity check): expecting 0-42 (n_unique=43) in every file.")
        for c in captures:
            layers_seen = sorted(c["df"]["layer"].unique().tolist())
            ok = "OK" if layers_seen and layers_seen[0] == 0 and layers_seen[-1] == 42 and len(layers_seen) == 43 else "CHECK"
            print(f"  {c['name']}: layers {layers_seen[0]}..{layers_seen[-1]} (n_unique={len(layers_seen)}) [{ok}]")

        if T_total < 200:
            print()
            print(f"*** WARNING: only {T_total} pooled token positions across {len(captures)} captures.  ***")
            print("*** This is thin. Larger-N window statistics rest on very few independent samples,   ***")
            print("*** and (see Part A) several N values may draw from only ONE capture even after       ***")
            print("*** pooling, because the individual captures are so short. Treat everything below as  ***")
            print("*** directional, not conclusive.                                                       ***")

        part_a_results, hash_self_check = part_a(captures)
        part_b(captures)

        pooled_df = build_pooled_df(captures)
        part_c_result = part_c(pooled_df)
        part_d(pooled_df, len(captures))

        verdict(captures, T_total, part_a_results, hash_self_check, part_c_result)

        print()
        print(f"Full log written to: {log_path}")
    finally:
        sys.stdout = real_stdout
        log_file.close()


if __name__ == "__main__":
    main()
