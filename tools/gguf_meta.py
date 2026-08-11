#!/usr/bin/env python3
"""
gguf_meta.py

Ground-truth extraction tool for the DeepSeek-V4-Flash-0731 UD-IQ1_S GGUF
multi-shard model. Reads header metadata and the full tensor index directly
from the GGUF files (no estimates, no guessing) using the local llama.cpp
gguf-py reader. Never touches tensor .data for large tensors - shapes,
types, element counts and byte offsets only.

Usage:
    C:\\Users\\hemant\\AppData\\Local\\Programs\\Python\\Python313\\python.exe tools\\gguf_meta.py

Outputs:
    stdout                                         (also teed below)
    bench\\results\\gguf_meta.txt                    (full text tee of stdout)
    bench\\results\\tensor_index.json                (full tensor index, all shards)
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Paths. Outputs are repo-relative so they follow the checkout. The two inputs
# that live outside the repo come from the environment first:
#     EXPERT_STREAM_MODEL_DIR   folder holding the GGUF shards
#     EXPERT_STREAM_MODEL       shard 1 itself; its parent is used if the above
#                               is unset, so one variable serves both scripts
#     EXPERT_STREAM_GGUF_PY     llama.cpp's gguf-py, for the GGUF reader
# The author's own layout remains as the last-resort default.
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_env_dir = os.environ.get("EXPERT_STREAM_MODEL_DIR") or ""
if not _env_dir:
    _env_model = os.environ.get("EXPERT_STREAM_MODEL") or ""
    _env_dir = os.path.dirname(_env_model) if _env_model else ""

GGUF_PY_DIR = os.environ.get("EXPERT_STREAM_GGUF_PY") or r"D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\gguf-py"
MODEL_DIR = _env_dir or r"E:\models\ds4f-iq1s\UD-IQ1_S"
OUT_TXT = os.path.join(REPO, "bench", "results", "gguf_meta.txt")
OUT_JSON = os.path.join(REPO, "bench", "results", "tensor_index.json")

# ---------------------------------------------------------------------------
# numpy check (per instructions: check, install if missing)
# ---------------------------------------------------------------------------
try:
    import numpy as np  # noqa: F401
except ImportError:
    import subprocess
    print("numpy not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np  # noqa: F401

# ---------------------------------------------------------------------------
# Local gguf-py (do NOT pip install a different version)
# ---------------------------------------------------------------------------
if GGUF_PY_DIR not in sys.path:
    sys.path.insert(0, GGUF_PY_DIR)

from gguf.gguf_reader import GGUFReader          # noqa: E402
from gguf.constants import GGUFValueType, GGMLQuantizationType  # noqa: E402


# ---------------------------------------------------------------------------
# Tee: print to stdout AND collect for the text log file
# ---------------------------------------------------------------------------
_LOG_LINES: list[str] = []


def out(msg: str = "") -> None:
    msg = str(msg)
    # Pure ASCII output only (per spec) - also avoids Windows cp1252 console
    # crashes on stray unicode bytes/chars found inside GGUF string metadata
    # (e.g. tokenizer special-token strings).
    msg = msg.encode("ascii", errors="backslashreplace").decode("ascii")
    print(msg)
    _LOG_LINES.append(msg)


def flush_log() -> None:
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="ascii", errors="replace", newline="\n") as f:
        f.write("\n".join(_LOG_LINES) + "\n")


def hr(title: str = "") -> None:
    out("")
    out("=" * 100)
    if title:
        out(title)
        out("=" * 100)


# ---------------------------------------------------------------------------
# Field formatting helpers (safe on huge arrays - only decodes what it shows)
# ---------------------------------------------------------------------------
def format_field_value(field) -> str:
    if not field.types:
        return "None"
    if field.types[0] == GGUFValueType.ARRAY:
        total = len(field.data)
        n_show = min(8, total)
        try:
            vals = field.contents(slice(0, n_show))
        except Exception as e:  # defensive - never let a weird field kill the run
            return f"<unreadable array: {e}>"
        suffix = "" if total <= 8 else f" ... ({total} total)"
        return f"{vals}{suffix}"
    try:
        return f"{field.contents()}"
    except Exception as e:
        return f"<unreadable: {e}>"


def field_scalar_int(reader: GGUFReader, key: str):
    """Return (found_key, int_value) for an exact-match scalar KV field, or (None, None)."""
    field = reader.get_field(key)
    if field is None:
        return None, None
    if field.types and field.types[0] == GGUFValueType.ARRAY:
        return key, None  # array, not scalar - caller must handle
    try:
        return key, int(field.contents())
    except Exception:
        return key, None


def find_key_by_suffix(reader: GGUFReader, suffix: str):
    """Find the first KV key in this reader ending with the given suffix (arch-prefix agnostic)."""
    dotsuf = "." + suffix
    for k in reader.fields.keys():
        if k == suffix or k.endswith(dotsuf):
            return k
    return None


# ---------------------------------------------------------------------------
# Locate model shards
# ---------------------------------------------------------------------------
def locate_shards() -> list[str]:
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.gguf")))
    if not files:
        raise SystemExit(f"FATAL: no *.gguf files found under {MODEL_DIR}")
    return files


# ---------------------------------------------------------------------------
# Tensor categorization for the resident-budget breakdown
# ---------------------------------------------------------------------------
def categorize_tensor(name: str) -> str:
    if "_exps." in name or name.endswith("_exps"):
        return "routed_experts"
    if "_shexp." in name or name.endswith("_shexp"):
        return "shared_experts"
    if "ffn_gate_inp" in name or "exp_probs_b" in name or "ffn_gate_tid2eid" in name:
        return "routing"
    if "indexer" in name:
        return "indexer"          # DeepSeek-V4 sparse-attention indexer (indexer.proj, indexer_compressor_*)
    if re.search(r"(^|\.)hc_", name) or "_hc_" in name:
        return "hyper_connection"  # DeepSeek-V4 hyper-connection tensors (hc_attn_*, hc_ffn_*, output_hc_*)
    if ".attn_" in name or name.endswith(".attn"):
        return "attention"
    if "ffn_norm" in name or "post_ffw_norm" in name or "pre_ffw_norm" in name or "ffn_sub_norm" in name:
        return "norms"
    if "ffn_gate" in name or "ffn_up" in name or "ffn_down" in name:
        return "dense_ffn"
    if name in ("token_embd.weight", "token_embd_norm.weight"):
        return "embeddings"
    if name == "output.weight":
        return "lm_head"
    if name == "output_norm.weight":
        return "output_norm"
    return "other"


EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def main() -> None:
    out("gguf_meta.py - DeepSeek-V4-Flash-0731 UD-IQ1_S ground-truth extraction")
    out(f"gguf-py source: {GGUF_PY_DIR}")
    out(f"model dir: {MODEL_DIR}")
    out(f"python: {sys.version}")
    out(f"numpy: {np.__version__}")

    shard_paths = locate_shards()
    out(f"found {len(shard_paths)} shard(s):")
    shard_sizes = {}
    for p in shard_paths:
        sz = os.path.getsize(p)
        shard_sizes[p] = sz
        out(f"  {os.path.basename(p)}  {sz:,} bytes")
    total_model_bytes = sum(shard_sizes.values())
    out(f"total on-disk model size (sum of all shards) = {total_model_bytes:,} bytes")

    # -----------------------------------------------------------------
    # Open every shard with GGUFReader independently. Report failures
    # explicitly rather than silently skipping.
    # -----------------------------------------------------------------
    readers: dict[str, GGUFReader | None] = {}
    open_errors: dict[str, str] = {}
    for p in shard_paths:
        try:
            r = GGUFReader(p)
            readers[p] = r
        except Exception as e:
            readers[p] = None
            open_errors[p] = f"{type(e).__name__}: {e}"

    hr("SHARD OPEN STATUS")
    for p in shard_paths:
        r = readers[p]
        if r is not None:
            out(f"OK   {os.path.basename(p)}  tensors={len(r.tensors)}  kv_fields={len(r.fields)}")
        else:
            out(f"FAIL {os.path.basename(p)}  ERROR: {open_errors[p]}")

    if readers[shard_paths[0]] is None:
        out("")
        out("FATAL: shard 1 (the metadata shard) failed to parse. Cannot continue.")
        flush_log()
        sys.exit(1)

    meta_reader: GGUFReader = readers[shard_paths[0]]

    # ===================================================================
    # 1. ALL KV METADATA FROM SHARD 1
    # ===================================================================
    hr("1. ALL KEY-VALUE METADATA (shard 1)")
    for key, field in meta_reader.fields.items():
        out(f"{key} = {format_field_value(field)}")

    interest_substrings = [
        "block_count", "embedding_length", "feed_forward_length",
        "expert_count", "expert_used_count", "expert_shared_count",
        "expert_feed_forward_length", "attention.head_count", "rope",
        "vocab_size", "split.count", "split.tensors.count",
    ]
    hr("1b. KEYS OF SPECIAL INTEREST (filtered view of section 1)")
    any_hit = False
    for key, field in meta_reader.fields.items():
        if any(s in key for s in interest_substrings):
            any_hit = True
            out(f"{key} = {format_field_value(field)}")
    if not any_hit:
        out("(none of the special-interest substrings matched any key)")

    # ===================================================================
    # 2. DERIVED SUMMARY
    # ===================================================================
    hr("2. DERIVED SUMMARY (each value traced back to its source KV key)")

    arch_field = meta_reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field is not None else None
    out(f"general.architecture = {arch}")

    def report_scalar(label: str, suffix: str):
        k = find_key_by_suffix(meta_reader, suffix)
        if k is None:
            out(f"{label:24s} = UNKNOWN   (no KV key ending in '.{suffix}' found)")
            return None
        field = meta_reader.get_field(k)
        if field.types and field.types[0] == GGUFValueType.ARRAY:
            out(f"{label:24s} = UNKNOWN   (source key '{k}' is an ARRAY, not scalar: {format_field_value(field)})")
            return None
        val = field.contents()
        out(f"{label:24s} = {val}    (source key: {k})")
        return val

    n_layer = report_scalar("n_layer", "block_count")
    n_embd = report_scalar("n_embd", "embedding_length")
    n_expert = report_scalar("n_expert", "expert_count")
    n_expert_used = report_scalar("n_expert_used", "expert_used_count")
    n_expert_shared = report_scalar("n_expert_shared", "expert_shared_count")
    n_ff_expert = report_scalar("n_ff_expert", "expert_feed_forward_length")
    n_ff_dense = report_scalar("n_ff_dense", "feed_forward_length")
    leading_dense = report_scalar("leading_dense_block_count", "leading_dense_block_count")

    # ===================================================================
    # 3. TENSOR INDEX OVER ALL SHARDS
    # ===================================================================
    hr("3. TENSOR INDEX (all shards)")
    tensor_index = []
    per_shard_counts = {}
    for p in shard_paths:
        r = readers[p]
        if r is None:
            out(f"SKIPPED (failed to open): {os.path.basename(p)}")
            continue
        per_shard_counts[os.path.basename(p)] = len(r.tensors)
        for t in r.tensors:
            tensor_index.append({
                "name": t.name,
                "shape_ggml_order": [int(x) for x in t.shape.tolist()],
                "n_dims": int(len(t.shape)),
                "ggml_type": t.tensor_type.name,
                "ggml_type_id": int(t.tensor_type),
                "n_elements": int(t.n_elements),
                "n_bytes": int(t.n_bytes),
                "shard_file": os.path.basename(p),
                "data_offset": int(t.data_offset),
            })

    out(f"total tensors indexed across all shards = {len(tensor_index)}")
    for fname, cnt in per_shard_counts.items():
        out(f"  {fname}: {cnt} tensors")

    dup_check = {}
    for te in tensor_index:
        dup_check.setdefault(te["name"], []).append(te["shard_file"])
    dups = {k: v for k, v in dup_check.items() if len(v) > 1}
    if dups:
        out(f"WARNING: {len(dups)} tensor name(s) appear in more than one shard:")
        for k, v in list(dups.items())[:20]:
            out(f"  {k}: {v}")
    else:
        out("no duplicate tensor names across shards (good - each tensor lives in exactly one shard)")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="ascii") as f:
        json.dump({
            "model_dir": MODEL_DIR,
            "shards": [os.path.basename(p) for p in shard_paths],
            "shard_sizes_bytes": {os.path.basename(p): sz for p, sz in shard_sizes.items()},
            "total_model_bytes": total_model_bytes,
            "tensor_count": len(tensor_index),
            "tensors": tensor_index,
        }, f, indent=1)
    out(f"full tensor index written to: {OUT_JSON}")

    # ===================================================================
    # 4. EXPERT SIZE REPORT
    # ===================================================================
    hr("4. EXPERT SIZE REPORT")

    # discover routed-expert tensors by regex on the real names we found
    exps_by_layer: dict[int, dict[str, dict]] = {}
    for te in tensor_index:
        m = EXPS_RE.match(te["name"])
        if m:
            bid = int(m.group(1))
            kind = m.group(2)  # gate/up/down
            exps_by_layer.setdefault(bid, {})[kind] = te

    if not exps_by_layer:
        out("UNKNOWN: no tensors matching 'blk.N.ffn_(gate|up|down)_exps.weight' were found.")
        out("Cannot compute per-expert byte sizes. Dumping first 40 tensor names containing 'exp' for inspection:")
        shown = 0
        for te in tensor_index:
            if "exp" in te["name"].lower():
                out(f"  {te['name']}  type={te['ggml_type']} shape={te['shape_ggml_order']}")
                shown += 1
                if shown >= 40:
                    break
    else:
        moe_layer_ids = sorted(exps_by_layer.keys())
        out(f"routed-expert tensor names found (pattern blk.N.ffn_{{gate,up,down}}_exps.weight):")
        out(f"  example (layer {moe_layer_ids[0]}):")
        for kind in ("gate", "up", "down"):
            te = exps_by_layer[moe_layer_ids[0]].get(kind)
            if te:
                out(f"    blk.{moe_layer_ids[0]}.ffn_{kind}_exps.weight  type={te['ggml_type']}  "
                    f"shape={te['shape_ggml_order']}  n_bytes={te['n_bytes']:,}")

        out("")
        out(f"MoE layers detected (have ffn_*_exps tensors) = {len(moe_layer_ids)} : {moe_layer_ids}")
        if n_layer is not None:
            dense_layer_ids = sorted(set(range(n_layer)) - set(moe_layer_ids))
            out(f"n_layer (metadata block_count) = {n_layer}")
            out(f"dense (non-MoE) layers detected by absence of _exps tensors = {len(dense_layer_ids)} : {dense_layer_ids}")
            if leading_dense is not None:
                out(f"metadata leading_dense_block_count = {leading_dense} "
                    f"({'matches' if leading_dense == len(dense_layer_ids) else 'DOES NOT MATCH'} detected dense layer count)")
            naive_total = n_layer * n_expert if n_expert is not None else None
            actual_total = len(moe_layer_ids) * n_expert if n_expert is not None else None
            out(f"NAIVE total routed experts (n_layer * n_expert) = {naive_total}  <- WRONG if any layer is dense")
            out(f"ACTUAL total routed experts in model (moe_layer_count * n_expert) = {actual_total}")
        else:
            out("n_layer UNKNOWN from metadata - cannot cross-check dense-layer count")

        # representative layer = first (lowest-index) MoE layer
        rep_bid = moe_layer_ids[0]
        rep = exps_by_layer[rep_bid]
        out("")
        out(f"REPRESENTATIVE LAYER = blk.{rep_bid} (first MoE layer found)")
        gate_te, up_te, down_te = rep.get("gate"), rep.get("up"), rep.get("down")
        if gate_te and up_te and down_te and n_expert:
            out(f"  ffn_gate_exps.weight: type={gate_te['ggml_type']} n_bytes={gate_te['n_bytes']:,}")
            out(f"  ffn_up_exps.weight:   type={up_te['ggml_type']} n_bytes={up_te['n_bytes']:,}")
            out(f"  ffn_down_exps.weight: type={down_te['ggml_type']} n_bytes={down_te['n_bytes']:,}")
            layer_total = gate_te['n_bytes'] + up_te['n_bytes'] + down_te['n_bytes']
            bytes_per_expert_rep = layer_total / n_expert
            out(f"  n_expert (metadata expert_count) = {n_expert}")
            out(f"  BYTES PER SINGLE EXPERT (layer {rep_bid}) = (gate+up+down)/n_expert"
                f" = {layer_total:,} / {n_expert} = {bytes_per_expert_rep:,.1f} bytes")
            # sanity check shape last dim vs n_expert
            for kind, te in (("gate", gate_te), ("up", up_te), ("down", down_te)):
                last_dim = te["shape_ggml_order"][-1] if te["shape_ggml_order"] else None
                if last_dim != n_expert:
                    out(f"  WARNING: {kind} tensor last dim ({last_dim}) != metadata n_expert ({n_expert})")
        else:
            out("  UNKNOWN: missing one of gate/up/down for the representative layer, or n_expert unknown.")
            bytes_per_expert_rep = None

        # per-layer table (constant vs varies - Unsloth UD dynamic quant)
        out("")
        out("PER-LAYER BYTES-PER-EXPERT TABLE (all detected MoE layers):")
        out(f"{'layer':>6} {'gate_type':>10} {'up_type':>10} {'down_type':>10} {'gate_bytes':>14} "
            f"{'up_bytes':>14} {'down_bytes':>14} {'total_bytes':>16} {'bytes/expert':>16}")
        per_layer_bpe = {}
        for bid in moe_layer_ids:
            layer = exps_by_layer[bid]
            g, u, d = layer.get("gate"), layer.get("up"), layer.get("down")
            if not (g and u and d):
                out(f"{bid:>6} INCOMPLETE (missing one of gate/up/down)")
                continue
            gb, ub, db = g["n_bytes"], u["n_bytes"], d["n_bytes"]
            tb = gb + ub + db
            ne = n_expert if n_expert else (g["shape_ggml_order"][-1] if g["shape_ggml_order"] else None)
            bpe = tb / ne if ne else None
            per_layer_bpe[bid] = bpe
            out(f"{bid:>6} {g['ggml_type']:>10} {u['ggml_type']:>10} {d['ggml_type']:>10} "
                f"{gb:>14,} {ub:>14,} {db:>14,} {tb:>16,} "
                f"{(f'{bpe:,.1f}' if bpe is not None else 'UNKNOWN'):>16}")

        vals = [v for v in per_layer_bpe.values() if v is not None]
        out("")
        if vals:
            vmin, vmax, vmean = min(vals), max(vals), sum(vals) / len(vals)
            if len(set(round(v, 3) for v in vals)) == 1:
                out(f"bytes-per-expert is CONSTANT across all {len(vals)} MoE layers = {vmin:,.1f} bytes")
            else:
                out(f"bytes-per-expert VARIES across layers (Unsloth UD dynamic quant - different bit widths per layer)")
                out(f"  min  = {vmin:,.1f} bytes")
                out(f"  max  = {vmax:,.1f} bytes")
                out(f"  mean = {vmean:,.1f} bytes")
        else:
            out("UNKNOWN: could not compute bytes-per-expert for any layer")

        # total routed-expert bytes and percentage of model
        total_exps_bytes = sum(
            te["n_bytes"] for te in tensor_index if EXPS_RE.match(te["name"])
        )
        out("")
        out(f"TOTAL bytes of all routed-expert tensors (all layers, gate+up+down) = {total_exps_bytes:,} bytes"
            f"  ({total_exps_bytes / (1024**3):,.2f} GiB)")
        pct = 100.0 * total_exps_bytes / total_model_bytes if total_model_bytes else float("nan")
        out(f"as percentage of total on-disk model size ({total_model_bytes:,} bytes) = {pct:.2f} %")

    # ===================================================================
    # 5. RESIDENT BUDGET REPORT (everything that is NOT a routed expert)
    # ===================================================================
    hr("5. RESIDENT BUDGET REPORT (everything NOT a routed expert)")
    cat_bytes: dict[str, int] = {}
    cat_count: dict[str, int] = {}
    other_names = []
    for te in tensor_index:
        cat = categorize_tensor(te["name"])
        cat_bytes[cat] = cat_bytes.get(cat, 0) + te["n_bytes"]
        cat_count[cat] = cat_count.get(cat, 0) + 1
        if cat == "other":
            other_names.append(te["name"])

    resident_total = sum(v for k, v in cat_bytes.items() if k != "routed_experts")
    routed_total = cat_bytes.get("routed_experts", 0)
    grand_total = sum(cat_bytes.values())

    out(f"{'category':>16} {'tensors':>10} {'bytes':>18} {'GiB':>10} {'% of model':>12}")
    for cat in sorted(cat_bytes.keys(), key=lambda c: -cat_bytes[c]):
        b = cat_bytes[cat]
        flag = "  <-- EXCLUDED FROM RESIDENT BUDGET" if cat == "routed_experts" else ""
        out(f"{cat:>16} {cat_count[cat]:>10} {b:>18,} {b/(1024**3):>10,.3f} "
            f"{100.0*b/grand_total if grand_total else float('nan'):>11.2f}%{flag}")

    out("")
    out(f"RESIDENT TOTAL (everything except routed experts) = {resident_total:,} bytes "
        f"({resident_total/(1024**3):,.2f} GiB)")
    out(f"ROUTED-EXPERT TOTAL (excluded above)               = {routed_total:,} bytes "
        f"({routed_total/(1024**3):,.2f} GiB)")
    out(f"GRAND TOTAL (all tensors, sanity check)             = {grand_total:,} bytes "
        f"({grand_total/(1024**3):,.2f} GiB)")
    out(f"grand total vs on-disk model size check: tensors={grand_total:,} bytes  "
        f"vs  files={total_model_bytes:,} bytes  "
        f"(difference = {total_model_bytes - grand_total:,} bytes; expected to be >0, covers headers/padding/alignment)")

    if other_names:
        out("")
        out(f"WARNING: {len(other_names)} tensor(s) fell into 'other' (unclassified) category - listing up to 40:")
        for n in other_names[:40]:
            out(f"  {n}")

    budget_16gb_ram = 16 * (1024**3)
    budget_6gb_vram = 6 * (1024**3)
    out("")
    out(f"vs stated resident budget of 16 GiB RAM ({budget_16gb_ram:,} bytes) + 6 GiB VRAM ({budget_6gb_vram:,} bytes) "
        f"= {budget_16gb_ram + budget_6gb_vram:,} bytes total:")
    if resident_total <= (budget_16gb_ram + budget_6gb_vram):
        out(f"  FITS: resident total ({resident_total/(1024**3):,.2f} GiB) <= budget "
            f"({(budget_16gb_ram+budget_6gb_vram)/(1024**3):,.2f} GiB), "
            f"headroom = {(budget_16gb_ram+budget_6gb_vram-resident_total)/(1024**3):,.2f} GiB")
    else:
        out(f"  DOES NOT FIT: resident total ({resident_total/(1024**3):,.2f} GiB) > budget "
            f"({(budget_16gb_ram+budget_6gb_vram)/(1024**3):,.2f} GiB), "
            f"overage = {(resident_total-(budget_16gb_ram+budget_6gb_vram))/(1024**3):,.2f} GiB")

    # ===================================================================
    # 6. QUANT TYPE INVENTORY (per distinct ggml type, overall)
    # ===================================================================
    hr("6. GGML QUANT TYPE INVENTORY (all tensors, all shards)")
    type_bytes: dict[str, int] = {}
    type_count: dict[str, int] = {}
    for te in tensor_index:
        t = te["ggml_type"]
        type_bytes[t] = type_bytes.get(t, 0) + te["n_bytes"]
        type_count[t] = type_count.get(t, 0) + 1
    out(f"{'ggml_type':>12} {'tensors':>10} {'bytes':>18} {'GiB':>10} {'% of model':>12}")
    for t in sorted(type_bytes.keys(), key=lambda k: -type_bytes[k]):
        b = type_bytes[t]
        out(f"{t:>12} {type_count[t]:>10} {b:>18,} {b/(1024**3):>10,.3f} "
            f"{100.0*b/grand_total if grand_total else float('nan'):>11.2f}%")

    out("")
    out("QUANT TYPE PER CATEGORY (category -> {type: (tensor_count, bytes)}):")
    cat_type_map: dict[str, dict[str, list]] = {}
    for te in tensor_index:
        cat = categorize_tensor(te["name"])
        t = te["ggml_type"]
        d = cat_type_map.setdefault(cat, {})
        if t not in d:
            d[t] = [0, 0]
        d[t][0] += 1
        d[t][1] += te["n_bytes"]
    for cat in sorted(cat_type_map.keys()):
        out(f"  {cat}:")
        for t, (cnt, b) in sorted(cat_type_map[cat].items(), key=lambda kv: -kv[1][1]):
            out(f"    {t:>10}  tensors={cnt:>6}  bytes={b:>18,}  ({b/(1024**3):,.3f} GiB)")

    hr("DONE")
    out(f"text log written to: {OUT_TXT}")
    out(f"tensor index JSON written to: {OUT_JSON}")

    flush_log()


if __name__ == "__main__":
    main()
