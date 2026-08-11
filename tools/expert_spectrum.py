#!/usr/bin/env python3
"""
expert_spectrum.py -- KILL-OR-CONFIRM experiment for a low-rank-sketch MoE cache idea.

IDEA UNDER TEST
----------------
Cache misses on routed-expert weights currently stall on SSD reads of ~7 MB per
expert. Proposed fix: keep a small low-rank SVD sketch (rank r) of every expert
resident in RAM, and use the sketch to approximate the expert's output on a
cache miss instead of stalling.

This only works if expert weight matrices are actually low rank. This script
measures the real singular-value spectrum of sampled expert matrices from the
DeepSeek-V4-Flash UD-IQ1_S GGUF, compares it against a random matrix of the
same shape (the null hypothesis / Marchenko-Pastur baseline), and reports a
plain VERDICT: LOW RANK CONFIRMED, NOT LOW RANK, or MIXED.

Read-only with respect to the model. Never materializes a full expert tensor
stack -- only ever holds one expert's one 2-D matrix (a few MB dense) at a time.

Pure ASCII output.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = r"E:\models\ds4f-iq1s\UD-IQ1_S"
DEFAULT_GGUF_PY = r"D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\gguf-py"
DEFAULT_CSV_OUT = r"D:\2025_Cursor_Dev\V4-local-serving\v4flash-16gb\bench\results\expert_spectrum.csv"
DEFAULT_TXT_OUT = r"D:\2025_Cursor_Dev\V4-local-serving\v4flash-16gb\bench\results\expert_spectrum.txt"

EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\.(ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight$")
KIND_LABEL = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}
KIND_INDEX = {"gate": 0, "up": 1, "down": 2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="Directory containing the *.gguf shards (default: %(default)s)")
    p.add_argument("--gguf-py", default=DEFAULT_GGUF_PY,
                    help="Path to the local gguf-py package root (default: %(default)s)")
    p.add_argument("--layers", type=int, default=3,
                    help="Number of layers to sample, spread early/mid/late across depth (default: 3)")
    p.add_argument("--experts-per-layer", type=int, default=2,
                    help="Number of experts to sample per layer (default: 2)")
    p.add_argument("--rank-list", default="8,16,32,64,128,256",
                    help="Comma-separated ranks to report energy-retained at (default: %(default)s)")
    p.add_argument("--matrices", default="gate,up,down",
                    help="Comma-separated subset of {gate,up,down} to test (default: all three)")
    p.add_argument("--seed", type=int, default=42,
                    help="RNG seed for expert selection and random-matrix baseline (default: 42)")
    p.add_argument("--max-dim-for-exact-svd", type=int, default=8192,
                    help="If max(rows,cols) exceeds this, fall back to randomized SVD (default: 8192)")
    p.add_argument("--randomized-components", type=int, default=400,
                    help="Number of random projections for randomized SVD fallback (default: 400)")
    p.add_argument("--csv-out", default=DEFAULT_CSV_OUT, help="CSV output path")
    p.add_argument("--txt-out", default=DEFAULT_TXT_OUT, help="Console-tee text output path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging (tee to console + text file, flushed as we go so a crash mid-run
# still leaves useful output on disk)
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "w", encoding="ascii", errors="replace")

    def log(self, msg=""):
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tee = Tee(args.txt_out)
    log = tee.log

    log("=" * 100)
    log("expert_spectrum.py -- low-rank-sketch kill-or-confirm experiment")
    log("=" * 100)
    log(f"model_dir            = {args.model_dir}")
    log(f"gguf_py              = {args.gguf_py}")
    log(f"layers               = {args.layers}")
    log(f"experts_per_layer    = {args.experts_per_layer}")
    log(f"rank_list            = {args.rank_list}")
    log(f"matrices             = {args.matrices}")
    log(f"seed                 = {args.seed}")
    log("")

    sys.path.insert(0, args.gguf_py)
    try:
        import numpy as np
    except ImportError:
        log("numpy not found. Install it with: python -m pip install numpy")
        tee.close()
        sys.exit(1)

    try:
        import gguf
        import gguf.quants as gq
        from gguf.constants import GGMLQuantizationType
    except Exception as e:
        log(f"FAILED to import local gguf-py from {args.gguf_py}: {e}")
        log(traceback.format_exc())
        tee.close()
        sys.exit(1)

    rank_list = sorted(set(int(x.strip()) for x in args.rank_list.split(",") if x.strip()))
    wanted_kinds = [k.strip() for k in args.matrices.split(",") if k.strip()]
    for k in wanted_kinds:
        if k not in KIND_INDEX:
            log(f"Unknown matrix kind '{k}'. Must be one of gate,up,down.")
            tee.close()
            sys.exit(1)

    # -----------------------------------------------------------------
    # Locate shards, build tensor index for expert tensors only.
    # -----------------------------------------------------------------
    shard_paths = sorted(glob.glob(os.path.join(args.model_dir, "*.gguf")))
    if not shard_paths:
        log(f"No .gguf files found in {args.model_dir}")
        tee.close()
        sys.exit(1)
    log(f"Found {len(shard_paths)} shard(s):")
    for sp in shard_paths:
        log(f"  {sp}  ({os.path.getsize(sp) / 1e9:.2f} GB)")
    log("")

    log("Opening shards (metadata only, memmap -- this does not load tensor data into RAM)...")
    tensor_index = {}   # name -> ReaderTensor
    meta_reader = None
    for sp in shard_paths:
        t0 = time.time()
        r = gguf.GGUFReader(sp)
        log(f"  opened {os.path.basename(sp)} in {time.time() - t0:.2f}s, {len(r.tensors)} tensors")
        if meta_reader is None or r.get_field("general.architecture") is not None:
            if r.get_field("deepseek4.block_count") is not None or r.get_field("deepseek2.block_count") is not None:
                meta_reader = r
        for t in r.tensors:
            m = EXPERT_TENSOR_RE.match(t.name)
            if m:
                tensor_index[t.name] = t
    log(f"Indexed {len(tensor_index)} routed-expert tensors across shards.")
    log("")

    # -----------------------------------------------------------------
    # Discover MoE layer list and expert count from tensor names/shapes
    # (robust to metadata key name changing between arch versions).
    # -----------------------------------------------------------------
    layers_present = sorted(set(int(EXPERT_TENSOR_RE.match(n).group(1)) for n in tensor_index))
    if not layers_present:
        log("No routed-expert tensors (blk.N.ffn_{gate,up,down}_exps.weight) found. Aborting.")
        tee.close()
        sys.exit(1)

    # infer n_expert from the first tensor's leading (memmap) axis
    sample_t = next(iter(tensor_index.values()))
    n_expert = sample_t.data.shape[0]
    log(f"MoE layers present: {len(layers_present)} (indices {layers_present[0]}..{layers_present[-1]})")
    log(f"Experts per layer (from tensor shape): {n_expert}")
    log("")

    # -----------------------------------------------------------------
    # CRITICAL PRE-FLIGHT CHECK: verify dequantization support for every
    # quant type we are actually going to touch, BEFORE touching any data.
    # -----------------------------------------------------------------
    def dequant_supported(qtype):
        if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            return True
        return qtype in gq._type_traits

    needed_types = {}
    for name, t in tensor_index.items():
        m = EXPERT_TENSOR_RE.match(name)
        kind = KIND_LABEL[m.group(2)]
        if kind in wanted_kinds:
            needed_types.setdefault(t.tensor_type, []).append(name)

    unsupported = {qt: names for qt, names in needed_types.items() if not dequant_supported(qt)}
    if unsupported:
        log("!" * 100)
        log("STOP: dequantization is NOT implemented in this local gguf-py for the following quant type(s)")
        log("actually used by the routed-expert tensors we would need to read:")
        for qt, names in unsupported.items():
            log(f"  {qt.name} (used by {len(names)} tensor(s), e.g. {names[0]})")
        supported_names = sorted(qt.name for qt in gq._type_traits.keys()) + ["F16", "F32"]
        log("")
        log("Quant types gguf-py DOES support dequantization for:")
        log("  " + ", ".join(sorted(supported_names)))
        log("")
        log("Per instructions: NOT writing a custom decoder for the unsupported type(s) -- a subtly")
        log("wrong hand-rolled decoder would produce false spectrum numbers, which is worse than no")
        log("answer. This is the finding. Halting before reading any tensor data.")
        log("!" * 100)
        tee.close()
        sys.exit(3)

    log("Dequantization support check PASSED for all quant types we will touch:")
    for qt in needed_types:
        log(f"  {qt.name}: supported")
    log("")

    # -----------------------------------------------------------------
    # Pick layers (spread across depth) and experts (reproducible RNG).
    # -----------------------------------------------------------------
    n_layers_req = max(1, args.layers)
    if n_layers_req >= len(layers_present):
        chosen_layers = list(layers_present)
    else:
        idxs = np.round(np.linspace(0, len(layers_present) - 1, n_layers_req)).astype(int)
        seen = []
        for i in idxs:
            li = layers_present[int(i)]
            if li not in seen:
                seen.append(li)
        chosen_layers = seen

    py_rng = random.Random(args.seed)
    plan = []  # list of (layer, expert_idx)
    for layer in chosen_layers:
        k = min(args.experts_per_layer, n_expert)
        experts = sorted(py_rng.sample(range(n_expert), k))
        for e in experts:
            plan.append((layer, e))

    log(f"Sampling plan: {len(chosen_layers)} layer(s) x up to {args.experts_per_layer} expert(s) = "
        f"{len(plan)} expert(s) total")
    log(f"  layers  : {chosen_layers}")
    for layer in chosen_layers:
        es = [e for (l, e) in plan if l == layer]
        depth_tag = ("early" if layer == min(layers_present) else
                     "late" if layer == max(layers_present) else "mid")
        log(f"    layer {layer:3d} ({depth_tag:5s}): experts {es}")
    log("")

    # -----------------------------------------------------------------
    # Randomized SVD fallback (Halko et al.), only used if a matrix
    # exceeds --max-dim-for-exact-svd. Loudly flagged when triggered.
    # -----------------------------------------------------------------
    def randomized_singular_values(M, k, n_oversample, n_iter, rng):
        m, n = M.shape
        l = min(k + n_oversample, min(m, n))
        Omega = rng.standard_normal((n, l)).astype(np.float32)
        Y = M @ Omega
        for _ in range(n_iter):
            Y, _ = np.linalg.qr(Y)
            Z = M.T @ Y
            Z, _ = np.linalg.qr(Z)
            Y = M @ Z
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ M
        _, s, _ = np.linalg.svd(B, full_matrices=False)
        return s[:k]

    # -----------------------------------------------------------------
    # CSV setup
    # -----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    fieldnames = [
        "layer", "expert", "matrix", "quant_type", "rows", "cols", "full_rank",
        "svd_method", "dequant_time_s", "svd_time_s",
        "frob_sq_real", "frob_sq_random", "std_real",
        "original_quant_bytes",
    ]
    for r in rank_list:
        fieldnames += [f"energy_real_r{r}", f"energy_random_r{r}",
                       f"sketch_bytes_r{r}", f"compression_ratio_r{r}"]
    fieldnames += [
        "rank90_real", "rank95_real", "rank99_real",
        "rank90_real_frac", "rank95_real_frac", "rank99_real_frac",
        "rank90_random", "rank95_random", "rank99_random",
        "rank90_random_frac", "rank95_random_frac", "rank99_random_frac",
    ]
    csv_fh = open(args.csv_out, "w", newline="", encoding="ascii")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    csv_writer.writeheader()

    def rank_for_threshold(cum_energy, threshold, truncated_at=None):
        idx = np.searchsorted(cum_energy, threshold)
        if idx >= len(cum_energy):
            if truncated_at is not None:
                return None  # never reached within the (possibly truncated) spectrum we computed
            return len(cum_energy)
        return int(idx) + 1

    all_rows = []
    total_matrices = len(plan) * len(wanted_kinds)
    done = 0
    t_start = time.time()

    for layer, expert in plan:
        for kind in wanted_kinds:
            tensor_name = f"blk.{layer}.ffn_{'gate' if kind=='gate' else ('up' if kind=='up' else 'down')}_exps.weight"
            t = tensor_index.get(tensor_name)
            if t is None:
                log(f"WARNING: tensor {tensor_name} not found, skipping")
                continue

            done += 1
            qtype = t.tensor_type
            log(f"[{done}/{total_matrices}] layer={layer:3d} expert={expert:3d} matrix={kind:4s} "
                f"qtype={qtype.name:10s} -- dequantizing...")

            # ---- slice exactly one expert's 2-D matrix, copy off the memmap, dequantize ----
            t0 = time.time()
            byte_slice = np.array(t.data[expert], copy=True)
            original_bytes = int(byte_slice.nbytes)
            M = gq.dequantize(byte_slice, qtype)
            M = np.ascontiguousarray(M, dtype=np.float32)
            dequant_time = time.time() - t0
            del byte_slice
            rows, cols = M.shape
            full_rank = min(rows, cols)

            approx_note = ""
            if max(rows, cols) > args.max_dim_for_exact_svd:
                svd_method = "randomized"
                approx_note = (f"  *** APPROXIMATE: matrix exceeds --max-dim-for-exact-svd "
                                f"({args.max_dim_for_exact_svd}); using randomized SVD with "
                                f"{args.randomized_components} projections. Spectrum TAIL beyond "
                                f"rank {args.randomized_components} is NOT measured. ***")
                log(approx_note)
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, layer, expert, KIND_INDEX[kind], 1]))
                t0 = time.time()
                s = randomized_singular_values(M, args.randomized_components, 20, 4, rng)
            else:
                svd_method = "exact"
                t0 = time.time()
                s = np.linalg.svd(M, compute_uv=False)
            svd_time = time.time() - t0

            frob_sq_real = float(np.sum(M.astype(np.float64) ** 2))
            std_real = float(M.std())
            del M

            # ---- random baseline: same shape, normal(0, std_real) ----
            rng_r = np.random.default_rng(np.random.SeedSequence([args.seed, layer, expert, KIND_INDEX[kind], 2]))
            R = rng_r.standard_normal((rows, cols)).astype(np.float32) * np.float32(std_real)
            if svd_method == "randomized":
                rng_r2 = np.random.default_rng(np.random.SeedSequence([args.seed, layer, expert, KIND_INDEX[kind], 3]))
                s_r = randomized_singular_values(R, args.randomized_components, 20, 4, rng_r2)
            else:
                s_r = np.linalg.svd(R, compute_uv=False)
            frob_sq_random = float(np.sum(R.astype(np.float64) ** 2))
            del R

            cum_real = np.cumsum(s.astype(np.float64) ** 2) / frob_sq_real
            cum_rand = np.cumsum(s_r.astype(np.float64) ** 2) / frob_sq_random
            truncated = (svd_method == "randomized")

            row = {
                "layer": layer, "expert": expert, "matrix": kind, "quant_type": qtype.name,
                "rows": rows, "cols": cols, "full_rank": full_rank, "svd_method": svd_method,
                "dequant_time_s": round(dequant_time, 4), "svd_time_s": round(svd_time, 4),
                "frob_sq_real": frob_sq_real, "frob_sq_random": frob_sq_random, "std_real": std_real,
                "original_quant_bytes": original_bytes,
            }

            log(f"    shape=({rows},{cols}) full_rank={full_rank} orig_quant_bytes={original_bytes} "
                f"({original_bytes/1e6:.3f} MB)  dequant={dequant_time:.2f}s  svd[{svd_method}]={svd_time:.2f}s")
            log(f"    {'rank':>6s}  {'energy_real':>12s}  {'energy_random':>14s}  {'gap':>8s}  "
                f"{'sketch_int8_B':>13s}  {'compression_x':>13s}")
            for r in rank_list:
                rr = min(r, len(cum_real))
                er = float(cum_real[rr - 1]) if rr <= len(cum_real) else float(cum_real[-1])
                rr2 = min(r, len(cum_rand))
                ea = float(cum_rand[rr2 - 1]) if rr2 <= len(cum_rand) else float(cum_rand[-1])
                sketch_bytes = r * (rows + cols)
                ratio = original_bytes / sketch_bytes if sketch_bytes > 0 else float("inf")
                gap = er - ea
                log(f"    {r:6d}  {er:12.4f}  {ea:14.4f}  {gap:8.4f}  {sketch_bytes:13d}  {ratio:13.3f}")
                row[f"energy_real_r{r}"] = round(er, 6)
                row[f"energy_random_r{r}"] = round(ea, 6)
                row[f"sketch_bytes_r{r}"] = sketch_bytes
                row[f"compression_ratio_r{r}"] = round(ratio, 4)

            for label, thr in (("90", 0.90), ("95", 0.95), ("99", 0.99)):
                rk_real = rank_for_threshold(cum_real, thr, truncated_at=truncated)
                rk_rand = rank_for_threshold(cum_rand, thr, truncated_at=truncated)
                row[f"rank{label}_real"] = rk_real if rk_real is not None else f">{len(cum_real)}(truncated)"
                row[f"rank{label}_random"] = rk_rand if rk_rand is not None else f">{len(cum_rand)}(truncated)"
                row[f"rank{label}_real_frac"] = round(rk_real / full_rank, 4) if isinstance(rk_real, int) else None
                row[f"rank{label}_random_frac"] = round(rk_rand / full_rank, 4) if isinstance(rk_rand, int) else None

            log(f"    rank-for-90%energy: real={row['rank90_real']} ({row['rank90_real_frac']} of full rank)  "
                f"random={row['rank90_random']} ({row['rank90_random_frac']} of full rank)")
            log(f"    rank-for-95%energy: real={row['rank95_real']} ({row['rank95_real_frac']} of full rank)  "
                f"random={row['rank95_random']} ({row['rank95_random_frac']} of full rank)")
            log(f"    rank-for-99%energy: real={row['rank99_real']} ({row['rank99_real_frac']} of full rank)  "
                f"random={row['rank99_random']} ({row['rank99_random_frac']} of full rank)")
            log("")

            csv_writer.writerow(row)
            csv_fh.flush()
            all_rows.append(row)

    csv_fh.close()
    elapsed = time.time() - t_start
    log(f"Done. Processed {done} matrices in {elapsed:.1f}s. CSV written to {args.csv_out}")
    log("")

    # -----------------------------------------------------------------
    # Aggregate + VERDICT
    # -----------------------------------------------------------------
    if not all_rows:
        log("No matrices were successfully processed; cannot render a verdict.")
        tee.close()
        sys.exit(1)

    mid_rank = rank_list[len(rank_list) // 2]  # representative rank for the headline gap number

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    overall_gap_mid = mean([row[f"energy_real_r{mid_rank}"] - row[f"energy_random_r{mid_rank}"] for row in all_rows])
    overall_real_mid = mean([row[f"energy_real_r{mid_rank}"] for row in all_rows])
    overall_rand_mid = mean([row[f"energy_random_r{mid_rank}"] for row in all_rows])
    overall_rank90_frac_real = mean([row["rank90_real_frac"] for row in all_rows])
    overall_rank90_frac_rand = mean([row["rank90_random_frac"] for row in all_rows])
    overall_ratio_mid = mean([row[f"compression_ratio_r{mid_rank}"] for row in all_rows])

    log("=" * 100)
    log("AGGREGATE SUMMARY (mean across all sampled matrices)")
    log("=" * 100)
    log(f"representative rank for headline numbers: r={mid_rank}")
    log(f"  mean energy@r={mid_rank}   real={overall_real_mid:.4f}   random={overall_rand_mid:.4f}   "
        f"gap={overall_gap_mid:+.4f}")
    log(f"  mean rank-for-90%-energy as fraction of full rank: real={overall_rank90_frac_real:.4f}   "
        f"random={overall_rank90_frac_rand:.4f}")
    log(f"  mean compression ratio at r={mid_rank} (int8 sketch bytes vs original quantized bytes): "
        f"{overall_ratio_mid:.3f}x")
    log("")

    # breakdown by matrix kind
    log("Breakdown by matrix kind:")
    by_kind = {}
    for row in all_rows:
        by_kind.setdefault(row["matrix"], []).append(row)
    for kind, rows in by_kind.items():
        g = mean([row[f"energy_real_r{mid_rank}"] - row[f"energy_random_r{mid_rank}"] for row in rows])
        r90f = mean([row["rank90_real_frac"] for row in rows])
        log(f"  {kind:5s}: n={len(rows):2d}  mean_gap@r{mid_rank}={g:+.4f}  mean_rank90_frac={r90f:.4f}")
    log("")

    # breakdown by layer depth (early/mid/late tag)
    log("Breakdown by layer:")
    by_layer = {}
    for row in all_rows:
        by_layer.setdefault(row["layer"], []).append(row)
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        g = mean([row[f"energy_real_r{mid_rank}"] - row[f"energy_random_r{mid_rank}"] for row in rows])
        r90f = mean([row["rank90_real_frac"] for row in rows])
        log(f"  layer {layer:3d}: n={len(rows):2d}  mean_gap@r{mid_rank}={g:+.4f}  mean_rank90_frac={r90f:.4f}")
    log("")

    # ---- decide verdict ----
    GAP_STRONG = 0.15     # 15 percentage points of energy beats random at mid rank -> meaningfully low rank
    GAP_WEAK = 0.05        # below this, indistinguishable from random for practical purposes
    kind_gaps = {k: mean([row[f"energy_real_r{mid_rank}"] - row[f"energy_random_r{mid_rank}"] for row in v])
                 for k, v in by_kind.items()}
    strong_kinds = [k for k, g in kind_gaps.items() if g >= GAP_STRONG]
    weak_kinds = [k for k, g in kind_gaps.items() if g < GAP_WEAK]
    mixed_kinds = [k for k, g in kind_gaps.items() if GAP_WEAK <= g < GAP_STRONG]

    log("=" * 100)
    log("VERDICT")
    log("=" * 100)

    if overall_gap_mid >= GAP_STRONG and not weak_kinds:
        log(f"LOW RANK CONFIRMED: at rank {mid_rank}, real expert matrices retain {overall_real_mid*100:.1f}% "
            f"of Frobenius energy on average, versus {overall_rand_mid*100:.1f}% for a random matrix of the "
            f"same shape and matched standard deviation (gap = {overall_gap_mid*100:.1f} points). "
            f"Reaching 90% energy needs only {overall_rank90_frac_real*100:.1f}% of full rank on average "
            f"(vs {overall_rank90_frac_rand*100:.1f}% for random). At r={mid_rank}, the int8 rank-r sketch "
            f"is {overall_ratio_mid:.2f}x the size of the original quantized tensor "
            f"({'smaller' if overall_ratio_mid > 1 else 'NOT smaller'} in practice).")
    elif overall_gap_mid < GAP_WEAK and not strong_kinds:
        log(f"NOT LOW RANK: at rank {mid_rank}, real expert matrices retain {overall_real_mid*100:.1f}% of "
            f"Frobenius energy on average, versus {overall_rand_mid*100:.1f}% for a random matrix of the same "
            f"shape (gap = only {overall_gap_mid*100:.1f} points, i.e. real spectrum is within "
            f"{max(0.0, GAP_WEAK)*100:.0f} points of random). Reaching 90% energy needs "
            f"{overall_rank90_frac_real*100:.1f}% of full rank on average -- essentially the whole matrix. "
            f"Kill the sketch tier.")
    else:
        log("MIXED result:")
        if strong_kinds:
            log(f"  Matrices that DO look low-rank (gap >= {GAP_STRONG*100:.0f} pts at r={mid_rank}): "
                f"{strong_kinds} -- mean gaps: " +
                ", ".join(f"{k}={kind_gaps[k]*100:.1f}pts" for k in strong_kinds))
        if mixed_kinds:
            log(f"  Matrices that are borderline (gap {GAP_WEAK*100:.0f}-{GAP_STRONG*100:.0f} pts): "
                f"{mixed_kinds} -- mean gaps: " +
                ", ".join(f"{k}={kind_gaps[k]*100:.1f}pts" for k in mixed_kinds))
        if weak_kinds:
            log(f"  Matrices that look essentially random (gap < {GAP_WEAK*100:.0f} pts): {weak_kinds} -- "
                f"mean gaps: " + ", ".join(f"{k}={kind_gaps[k]*100:.1f}pts" for k in weak_kinds))
        # depth check
        layer_gaps = {l: mean([row[f"energy_real_r{mid_rank}"] - row[f"energy_random_r{mid_rank}"] for row in v])
                      for l, v in by_layer.items()}
        spread = max(layer_gaps.values()) - min(layer_gaps.values())
        if spread >= GAP_WEAK:
            log(f"  Depth also matters: per-layer gap@r{mid_rank} ranges from "
                f"{min(layer_gaps.values())*100:.1f} to {max(layer_gaps.values())*100:.1f} points "
                f"across sampled layers {sorted(layer_gaps.keys())}.")
        log(f"  Overall mean gap at r={mid_rank} = {overall_gap_mid*100:.1f} points; mean sketch/original "
            f"size ratio at r={mid_rank} = {overall_ratio_mid:.2f}x.")
        log("  Recommendation: do not greenlight a blanket sketch tier; if pursued, restrict it to the "
            "matrix kind(s)/layers shown above as genuinely low rank, and re-validate compression ratio "
            "there specifically.")

    log("=" * 100)
    tee.close()


if __name__ == "__main__":
    main()
