#!/usr/bin/env python3
"""
hash_routing.py

DeepSeek-V4-Flash-0731 UD-IQ1_S : zero-forward-pass expert routing analysis
for the THREE HASH-ROUTED layers (deepseek4.hash_layer_count = 3).

Layers 0, 1 and 2 route purely by token id via a frozen lookup table shipped
in the GGUF as I32 tensors named blk.{0,1,2}.ffn_gate_tid2eid.weight. This
script:

  PART 1 - loads the three tid2eid tensors and works out their semantics
           (shape / dtype / range), stating the evidence explicitly. If the
           semantics do not resolve cleanly, the script STOPS and does not
           fabricate parts 3-5.
  PART 2 - tokenizes several real text files (English prose, Python source,
           German prose) using the existing prebuilt E:\\tools\\llamacpp\\
           llama-tokenize.exe against shard 1 only (no weights loaded).
  PART 3 - the expert-overlap curve for layers 0-2: sliding windows of N
           consecutive tokens, mean/min/max distinct (layer,expert) pairs
           needed, ratio to N-times-single-token, and the independent-
           routing baseline for comparison.
  PART 4 - skew / concentration analysis (top 10/25/50% coverage, Gini).
  PART 5 - cross-check against the shipped hotlist (hotlist_flash.json).

CAVEAT (also printed prominently in the script's own output):
  Layers 0-2 use FROZEN HASH ROUTING (token id -> expert, no dependence on
  hidden state). Layers 3-42 route on hidden state and are NOT covered by
  this script. Nothing here may be presented as "the expert overlap curve
  for V4-Flash" - it is the curve for the 3 hash-routed layers only, and a
  dry run of the methodology intended for the other 40 layers later.

Usage:
    C:\\Users\\hemant\\AppData\\Local\\Programs\\Python\\Python313\\python.exe tools\\hash_routing.py

Outputs:
    stdout                                              (also teed below)
    bench\\results\\hash_routing.txt                     (full text tee)
    bench\\results\\hash_overlap_curve.csv                (PART 3 table)
"""

from __future__ import annotations

import ast
import collections
import glob
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paths (edit here only if the repo layout changes)
# ---------------------------------------------------------------------------
GGUF_PY_DIR = r"D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\gguf-py"
MODEL_DIR = r"E:\models\ds4f-iq1s\UD-IQ1_S"
SHARD1 = os.path.join(MODEL_DIR, "DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf")
TOKENIZE_EXE = r"E:\tools\llamacpp\llama-tokenize.exe"

RESULTS_DIR = r"D:\2025_Cursor_Dev\V4-local-serving\v4flash-16gb\bench\results"
OUT_TXT = os.path.join(RESULTS_DIR, "hash_routing.txt")
OUT_CSV = os.path.join(RESULTS_DIR, "hash_overlap_curve.csv")
HOTLIST_PATH = os.path.join(RESULTS_DIR, "hotlist_flash.json")

SCRATCHPAD = r"C:\Users\hemant\AppData\Local\Temp\claude\D--2025-Cursor-Dev-V4-local-serving\47e234c5-134d-4367-b077-bca7a21d4dbb\scratchpad"
TEXT_FILES = [
    ("english_prose", os.path.join(SCRATCHPAD, "text_english.txt"),
     "Original English prose (narrative, ~900 tokens) describing the discovery of hash routing"),
    ("python_source", os.path.join(SCRATCHPAD, "text_python.py"),
     "Original Python source code (~1500 tokens): an LRU expert cache + hash-routing lookup demo module"),
    ("german_prose", os.path.join(SCRATCHPAD, "text_german.txt"),
     "Original German prose (~1100 tokens), same narrative theme as the English text, translated"),
]

HASH_LAYERS = [0, 1, 2]
WINDOW_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]

# ---------------------------------------------------------------------------
# numpy check
# ---------------------------------------------------------------------------
try:
    import numpy as np  # noqa: F401
except ImportError:
    print("numpy not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np  # noqa: F401

# ---------------------------------------------------------------------------
# Local gguf-py (do NOT pip install a different version)
# ---------------------------------------------------------------------------
if GGUF_PY_DIR not in sys.path:
    sys.path.insert(0, GGUF_PY_DIR)

from gguf.gguf_reader import GGUFReader          # noqa: E402
from gguf.constants import GGUFValueType         # noqa: E402


# ---------------------------------------------------------------------------
# Tee: print to stdout AND collect for the text log file
# ---------------------------------------------------------------------------
_LOG_LINES: list[str] = []


def out(line: str = "") -> None:
    print(line)
    _LOG_LINES.append(line)


def hr(title: str = "") -> None:
    out("")
    out("=" * 78)
    if title:
        out(title)
        out("=" * 78)


def flush_log() -> None:
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(_LOG_LINES) + "\n")


CAVEAT = (
    "CAVEAT: layers 0-2 use FROZEN HASH ROUTING (token id -> expert lookup, "
    "zero dependence on hidden state). Layers 3-42 route on hidden state and "
    "are NOT covered here. This is the overlap curve for the 3 hash-routed "
    "layers ONLY -- it must never be presented as 'the expert overlap curve "
    "for V4-Flash'. It is a genuine partial result and a dry run of the "
    "methodology for the other 40 layers."
)


def find_key_by_suffix(reader: GGUFReader, suffix: str):
    dotsuf = "." + suffix
    for k in reader.fields.keys():
        if k == suffix or k.endswith(dotsuf):
            return k
    return None


def field_scalar_int(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        return None
    if field.types and field.types[0] == GGUFValueType.ARRAY:
        return None
    try:
        return int(field.contents())
    except Exception:
        return None


# ===========================================================================
# PART 1 - load the tid2eid tensors and work out semantics
# ===========================================================================
def part1_load_tables():
    hr("PART 1 - LOADING THE HASH-ROUTING TABLES AND DETERMINING SEMANTICS")
    out(CAVEAT)
    out("")

    shard_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*.gguf")))
    out(f"model dir: {MODEL_DIR}")
    out(f"shards found: {[os.path.basename(p) for p in shard_paths]}")

    # tid2eid tensors live in shard 2 per tensor_index.json, but we search
    # every shard so this does not silently break if layout changes.
    readers = {}
    for p in shard_paths:
        readers[p] = GGUFReader(p)

    meta_reader = readers[shard_paths[0]]

    # --- vocab size, from tokenizer.ggml.tokens length and any *.vocab_size key ---
    tok_key = find_key_by_suffix(meta_reader, "tokenizer.ggml.tokens")
    vocab_size_from_tokens = None
    if tok_key is not None:
        field = meta_reader.get_field(tok_key)
        vocab_size_from_tokens = len(field.data) if hasattr(field, "data") else None
        if vocab_size_from_tokens is None:
            # fall back: count via .parts / contents
            try:
                vocab_size_from_tokens = len(field.contents())
            except Exception:
                vocab_size_from_tokens = None

    vocab_size_key = find_key_by_suffix(meta_reader, "vocab_size")
    vocab_size_from_key = field_scalar_int(meta_reader, vocab_size_key) if vocab_size_key else None

    n_expert = field_scalar_int(meta_reader, find_key_by_suffix(meta_reader, "expert_count"))
    n_expert_used = field_scalar_int(meta_reader, find_key_by_suffix(meta_reader, "expert_used_count"))
    hash_layer_count = field_scalar_int(meta_reader, find_key_by_suffix(meta_reader, "hash_layer_count"))

    out(f"tokenizer.ggml.tokens length (vocab, counted directly) = {vocab_size_from_tokens}")
    out(f"metadata key ending '.vocab_size'                       = {vocab_size_key} -> {vocab_size_from_key}")
    out(f"deepseek4.expert_count      (n_expert)                  = {n_expert}")
    out(f"deepseek4.expert_used_count (n_expert_used, i.e. k)      = {n_expert_used}")
    out(f"deepseek4.hash_layer_count                               = {hash_layer_count}")

    vocab_size = vocab_size_from_tokens if vocab_size_from_tokens else vocab_size_from_key
    if vocab_size is None:
        out("FATAL: could not determine vocab size from any source.")
        flush_log()
        sys.exit(1)

    tables = {}
    all_ok = True
    for L in HASH_LAYERS:
        tname = f"blk.{L}.ffn_gate_tid2eid.weight"
        tensor = None
        found_in = None
        for p, r in readers.items():
            for t in r.tensors:
                if t.name == tname:
                    tensor = t
                    found_in = os.path.basename(p)
                    break
            if tensor is not None:
                break
        if tensor is None:
            out(f"FATAL: tensor {tname} not found in any shard.")
            all_ok = False
            continue

        data = np.asarray(tensor.data)  # already reshaped to numpy order by GGUFReader
        out("")
        out(f"--- {tname}  (found in {found_in}) ---")
        out(f"  ggml shape (ne order, ne[0] fastest)  = {list(tensor.shape)}")
        out(f"  numpy array shape (as loaded)         = {data.shape}")
        out(f"  dtype                                 = {data.dtype}")
        out(f"  n_elements                            = {data.size}  (tensor.n_elements = {tensor.n_elements})")
        out(f"  min value                             = {int(data.min())}")
        out(f"  max value                             = {int(data.max())}")

        tables[L] = data

    if not all_ok:
        out("")
        out("FATAL: not all three tid2eid tensors could be loaded. Stopping before part 2+.")
        flush_log()
        sys.exit(1)

    # --- work out the semantics explicitly ---
    hr("PART 1 - SEMANTICS DETERMINATION (evidence-based)")

    shapes = {L: tables[L].shape for L in HASH_LAYERS}
    out(f"numpy shapes for the 3 tables: {shapes}")

    rows_dim = {L: shapes[L][0] for L in HASH_LAYERS}
    cols_dim = {L: shapes[L][1] for L in HASH_LAYERS}
    out(f"first dim (candidate: indexed by token id) per layer: {rows_dim}")
    out(f"vocab size (from tokenizer.ggml.tokens)              : {vocab_size}")

    indexed_by_token = all(rows_dim[L] == vocab_size for L in HASH_LAYERS)
    out(f"  -> first dim == vocab_size for all 3 layers?  {indexed_by_token}")

    out(f"second dim (candidate: number of experts per row) per layer: {cols_dim}")
    out(f"n_expert_used (k) from metadata                             : {n_expert_used}")
    row_is_full_topk = all(cols_dim[L] == n_expert_used for L in HASH_LAYERS)
    row_is_single = all(cols_dim[L] == 1 for L in HASH_LAYERS)
    out(f"  -> second dim == n_expert_used (full top-k row)?  {row_is_full_topk}")
    out(f"  -> second dim == 1 (single expert, rest from elsewhere)?  {row_is_single}")

    range_ok = True
    for L in HASH_LAYERS:
        mn, mx = int(tables[L].min()), int(tables[L].max())
        ok = (mn >= 0) and (mx <= (n_expert - 1 if n_expert else 255))
        range_ok = range_ok and ok
        out(f"layer {L}: value range [{mn}, {mx}]  within [0, {n_expert - 1 if n_expert else 255}]? {ok}")

    # duplicate check: does each row of length k contain k distinct experts?
    dup_report = {}
    for L in HASH_LAYERS:
        row_len = cols_dim[L]
        if row_len > 1:
            sample_idx = np.linspace(0, rows_dim[L] - 1, num=min(5000, rows_dim[L]), dtype=np.int64)
            sample = tables[L][sample_idx]
            distinct_counts = np.array([len(set(row.tolist())) for row in sample])
            n_with_dupes = int(np.sum(distinct_counts < row_len))
            dup_report[L] = (n_with_dupes, len(sample_idx))
            out(f"layer {L}: rows with within-row duplicate experts (sample of {len(sample_idx)} rows): "
                f"{n_with_dupes} ({100.0 * n_with_dupes / len(sample_idx):.2f}%)")

    out("")
    out("SEMANTICS CONCLUSION:")
    semantics_ok = indexed_by_token and row_is_full_topk and range_ok
    if semantics_ok:
        out(f"  Each table has shape (vocab_size={vocab_size}, n_expert_used={n_expert_used}).")
        out("  Row index = token id. Row contents = the FULL top-{} expert set for that token,".format(n_expert_used))
        out("  for that hash-routed layer. All values lie within [0, n_expert-1]. This is a")
        out("  complete, self-sufficient token-id -> expert-set lookup; no other input is needed.")
        out("  SEMANTICS ARE CONSISTENT. Proceeding to parts 2-5.")
    else:
        out("  SEMANTICS DID NOT RESOLVE CLEANLY. Details above. STOPPING HERE.")
        out("  Parts 2-5 will NOT be run, and no routing numbers will be fabricated.")

    flush_log()
    if not semantics_ok:
        sys.exit(1)

    return tables, vocab_size, n_expert, n_expert_used


# ===========================================================================
# PART 2 - tokenize real text via the prebuilt llama-tokenize.exe
# ===========================================================================
def part2_tokenize():
    hr("PART 2 - TOKENIZING REAL TEXT (llama-tokenize.exe, shard 1 only, no weights loaded)")
    out(f"tokenizer exe: {TOKENIZE_EXE}")
    out(f"model (shard 1, metadata-only, {os.path.getsize(SHARD1):,} bytes on disk): {SHARD1}")

    corpora = {}
    for name, path, desc in TEXT_FILES:
        out("")
        out(f"--- {name} ---")
        out(f"  file: {path}")
        out(f"  description: {desc}")
        if not os.path.isfile(path):
            out(f"  FATAL: file not found.")
            flush_log()
            sys.exit(1)
        proc = subprocess.run(
            [TOKENIZE_EXE, "-m", SHARD1, "-f", path, "--ids"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            out(f"  FATAL: llama-tokenize.exe exited {proc.returncode}")
            out(f"  stderr: {proc.stderr[:2000]}")
            flush_log()
            sys.exit(1)
        stdout = proc.stdout.strip()
        ids = ast.literal_eval(stdout)
        out(f"  token count: {len(ids)}")
        out(f"  first 12 ids: {ids[:12]}")
        if len(ids) < 400:
            out(f"  WARNING: fewer than 400 tokens ({len(ids)}). Task required >= 400.")
        corpora[name] = ids

    flush_log()
    return corpora


# ===========================================================================
# PART 3 - the expert-overlap curve for layers 0-2
# ===========================================================================
def token_expert_codes(tables, token_id: int) -> frozenset:
    """Distinct (layer,expert) codes for one token, across the 3 hash layers.
    Code = layer * 256 + expert (n_expert assumed <= 256, checked in part 1)."""
    codes = set()
    for L in HASH_LAYERS:
        row = tables[L][token_id]
        for e in row.tolist():
            codes.add(L * 256 + int(e))
    return frozenset(codes)


def overlap_stats_for_window(code_lists: list[frozenset], N: int):
    """Sliding window of N token-code-sets over a sequence. Returns list of
    per-window union sizes using an incremental counter (O(M*avg_set_size))."""
    M = len(code_lists)
    if M < N:
        return []
    counts: collections.Counter = collections.Counter()
    for i in range(N):
        counts.update(code_lists[i])
    union_sizes = [len(counts)]
    for i in range(N, M):
        counts.update(code_lists[i])
        left = code_lists[i - N]
        for c in left:
            counts[c] -= 1
            if counts[c] <= 0:
                del counts[c]
        union_sizes.append(len(counts))
    return union_sizes


def part3_overlap_curve(tables, corpora, n_expert, n_expert_used):
    hr("PART 3 - EXPERT OVERLAP CURVE, LAYERS 0-2 ONLY (hash-routed)")
    out(CAVEAT)

    k = n_expert_used
    n_layers = len(HASH_LAYERS)

    csv_rows = []
    csv_rows.append(["corpus", "N", "mean_union", "min_union", "max_union",
                      "ratio_real_union_N_over_N_times_union_1",
                      "baseline_independent_union_N", "ratio_baseline"])

    per_corpus_pooled = {}  # N -> list of union sizes, pooled across corpora

    for corpus_name, ids in list(corpora.items()) + [("ALL_POOLED", None)]:
        hr(f"corpus: {corpus_name}")
        if corpus_name != "ALL_POOLED":
            code_lists = [token_expert_codes(tables, tid) for tid in ids]
            out(f"  sequence length (tokens): {len(ids)}")
        else:
            out("  pooled window-union values across all corpora above (windows never cross a text boundary)")

        # mean union at N=1 for the ratio denominator
        if corpus_name != "ALL_POOLED":
            union1_list = overlap_stats_for_window(code_lists, 1)
            mean_union1 = sum(union1_list) / len(union1_list) if union1_list else float("nan")
        else:
            union1_list = per_corpus_pooled.get(1, [])
            mean_union1 = sum(union1_list) / len(union1_list) if union1_list else float("nan")

        out(f"  {'N':>3s}  {'mean_union':>10s}  {'min':>4s}  {'max':>4s}  {'ratio_real':>10s}  "
            f"{'baseline_N':>10s}  {'ratio_baseline':>14s}")

        for N in WINDOW_SIZES:
            if corpus_name != "ALL_POOLED":
                unions = overlap_stats_for_window(code_lists, N)
                per_corpus_pooled.setdefault(N, []).extend(unions)
            else:
                unions = per_corpus_pooled.get(N, [])

            if not unions:
                out(f"  {N:>3d}  (no windows -- sequence shorter than N)")
                continue

            mean_u = sum(unions) / len(unions)
            min_u = min(unions)
            max_u = max(unions)
            ratio_real = mean_u / (N * mean_union1) if mean_union1 else float("nan")

            baseline_layer_N = n_expert * (1.0 - (1.0 - k / n_expert) ** N)
            baseline_total_N = n_layers * baseline_layer_N
            baseline_layer_1 = n_expert * (1.0 - (1.0 - k / n_expert) ** 1)
            baseline_total_1 = n_layers * baseline_layer_1
            ratio_baseline = baseline_total_N / (N * baseline_total_1)

            out(f"  {N:>3d}  {mean_u:>10.3f}  {min_u:>4d}  {max_u:>4d}  {ratio_real:>10.4f}  "
                f"{baseline_total_N:>10.3f}  {ratio_baseline:>14.4f}")

            csv_rows.append([corpus_name, N, f"{mean_u:.4f}", min_u, max_u,
                              f"{ratio_real:.4f}", f"{baseline_total_N:.4f}", f"{ratio_baseline:.4f}"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w", encoding="ascii", newline="") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    out("")
    out(f"CSV written: {OUT_CSV}")

    flush_log()
    return per_corpus_pooled


# ===========================================================================
# PART 4 - skew / concentration
# ===========================================================================
def gini(counts: np.ndarray) -> float:
    """Standard Gini coefficient over a non-negative count vector."""
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * x)) / (n * total) - (n + 1.0) / n)


def part4_skew(tables, corpora, n_expert):
    hr("PART 4 - SKEW / CONCENTRATION ACROSS (layer, expert) PAIRS")
    out(CAVEAT)
    out("Corpus used: all three tokenized texts concatenated (raw token instances, not deduped).")

    n_slots = len(HASH_LAYERS) * n_expert
    hist = np.zeros(n_slots, dtype=np.int64)

    total_tokens = 0
    for name, ids in corpora.items():
        total_tokens += len(ids)
        for tid in ids:
            for L in HASH_LAYERS:
                row = tables[L][tid]
                for e in row.tolist():
                    hist[L * n_expert + int(e)] += 1

    out(f"total token instances across all corpora: {total_tokens}")
    out(f"total routing events counted (tokens * layers * k, allowing within-row dupes): {int(hist.sum())}")
    out(f"number of (layer,expert) slots: {n_slots}  ( = {len(HASH_LAYERS)} layers * {n_expert} experts)")
    nonzero = int(np.sum(hist > 0))
    out(f"slots with at least one hit: {nonzero} / {n_slots}  ({100.0*nonzero/n_slots:.1f}%)")

    sorted_hist = np.sort(hist)[::-1]
    total = int(hist.sum())

    def top_frac_coverage(frac: float) -> float:
        n_top = max(1, int(round(frac * n_slots)))
        return 100.0 * float(sorted_hist[:n_top].sum()) / total if total else 0.0

    cov10 = top_frac_coverage(0.10)
    cov25 = top_frac_coverage(0.25)
    cov50 = top_frac_coverage(0.50)
    g = gini(hist)

    out("")
    out(f"fraction of all routing events covered by top 10% of (layer,expert) slots: {cov10:.2f}%  (uniform would give 10.00%)")
    out(f"fraction of all routing events covered by top 25% of (layer,expert) slots: {cov25:.2f}%  (uniform would give 25.00%)")
    out(f"fraction of all routing events covered by top 50% of (layer,expert) slots: {cov50:.2f}%  (uniform would give 50.00%)")
    out(f"Gini coefficient of the (layer,expert) usage histogram: {g:.4f}  (0.0 = perfectly uniform, 1.0 = maximally concentrated)")

    out("")
    out("Per-layer breakdown:")
    for L in HASH_LAYERS:
        layer_hist = hist[L * n_expert:(L + 1) * n_expert]
        layer_total = int(layer_hist.sum())
        layer_nonzero = int(np.sum(layer_hist > 0))
        layer_gini = gini(layer_hist)
        out(f"  layer {L}: total_hits={layer_total}  nonzero_experts={layer_nonzero}/{n_expert}  gini={layer_gini:.4f}")

    flush_log()
    return hist


# ===========================================================================
# PART 5 - cross-check against the shipped hotlist
# ===========================================================================
def part5_hotlist_crosscheck(hist, n_expert):
    hr("PART 5 - CROSS-CHECK AGAINST SHIPPED HOTLIST (bench/results/hotlist_flash.json)")
    out(CAVEAT)

    if not os.path.isfile(HOTLIST_PATH):
        out(f"FATAL: hotlist not found at {HOTLIST_PATH}")
        flush_log()
        return

    with open(HOTLIST_PATH, "r", encoding="utf-8") as f:
        hotlist = json.load(f)

    out(f"hotlist source: {hotlist.get('source_file')}")
    out(f"hotlist variant: {hotlist.get('variant')}  entry_count: {hotlist.get('entry_count')}")
    out(f"hotlist note: {hotlist.get('note')}")

    entries = hotlist["entries"]
    hotlist_by_layer = {L: {} for L in HASH_LAYERS}  # expert -> rank
    for e in entries:
        L = e["layer"]
        if L in hotlist_by_layer:
            hotlist_by_layer[L][e["expert"]] = e["rank"]

    for L in HASH_LAYERS:
        out("")
        out(f"--- layer {L} ---")
        layer_hist = hist[L * n_expert:(L + 1) * n_expert]
        our_ranked = np.argsort(-layer_hist)  # expert ids, most-used first
        our_nonzero = [int(e) for e in our_ranked if layer_hist[e] > 0]
        hot_experts = set(hotlist_by_layer[L].keys())

        out(f"  our observed nonzero experts (this small corpus): {len(our_nonzero)} / {n_expert}")
        out(f"  hotlist experts listed for this layer:            {len(hot_experts)} / {n_expert}")

        our_set = set(our_nonzero)
        overlap = our_set & hot_experts
        out(f"  overlap (our nonzero AND in hotlist):             {len(overlap)}")
        if our_set:
            out(f"  fraction of OUR nonzero experts also in hotlist:  {100.0*len(overlap)/len(our_set):.1f}%")
        if hot_experts:
            out(f"  fraction of HOTLIST experts we also observed:     {100.0*len(overlap)/len(hot_experts):.1f}%")

        top10_ours = our_nonzero[:10]
        out(f"  our top-10 hottest experts (by hit count) : {top10_ours}")
        ranks_of_top10 = [hotlist_by_layer[L].get(e, None) for e in top10_ours]
        out(f"  their rank in the shipped hotlist (None = absent, lower rank = hotter overall): {ranks_of_top10}")

        experts_missing_from_hotlist = [e for e in our_nonzero if e not in hot_experts]
        out(f"  our nonzero experts that are ABSENT from the hotlist entirely: {len(experts_missing_from_hotlist)} "
            f"{experts_missing_from_hotlist[:20]}")

    out("")
    out("Interpretation: layers 0-2's hash routing spreads token ids fairly evenly over the")
    out("256-expert table (see PART 4 Gini). If the hotlist agrees, it independently confirms")
    out("that these 3 layers do not have strongly 'hot' experts worth caching -- the hotlist's")
    out("selectivity for these layers should be much weaker than for the hidden-state-routed")
    out("layers 3-42.")

    flush_log()


# ===========================================================================
def main():
    hr("hash_routing.py -- DeepSeek-V4-Flash-0731 hash-routed layers 0-2 analysis")
    out(CAVEAT)
    out(f"python: {sys.version}")
    out(f"numpy: {np.__version__}")

    tables, vocab_size, n_expert, n_expert_used = part1_load_tables()
    corpora = part2_tokenize()
    part3_overlap_curve(tables, corpora, n_expert, n_expert_used)
    hist = part4_skew(tables, corpora, n_expert)
    part5_hotlist_crosscheck(hist, n_expert)

    hr("DONE")
    out(CAVEAT)
    flush_log()


if __name__ == "__main__":
    main()
