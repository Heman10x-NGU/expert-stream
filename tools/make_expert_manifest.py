#!/usr/bin/env python3
"""
make_expert_manifest.py

Emits the expert slice table that src/expert_read_bench.c (and later the real
reader) needs to address any single expert directly inside the original GGUF
shards, with no repack.

For each routed-expert tensor blk.N.ffn_{gate,up,down}_exps.weight we record:
    shard index, base file offset, per-expert stride, expert count

Expert e of that tensor then lives at exactly:
    base_offset + e * stride     ... for stride bytes

which is guaranteed by the ggml layout for a 3D tensor: the expert index is the
slowest-moving dimension, which is precisely why mul_mat_id can address it as
    src0->data + cur_a * nb02

The consumer is responsible for the unbuffered-read slop: read from
floor(off/sector)*sector and skip (off % sector) bytes in the buffer.

Reads bench/results/tensor_index.json. Opens no model file. Read-only.

Usage:
    python tools\\make_expert_manifest.py
"""

from __future__ import annotations

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_JSON = os.path.join(REPO, "bench", "results", "tensor_index.json")
# The directory holding the three GGUF shards. EXPERT_STREAM_MODEL_DIR overrides
# it; if that names the shard file rather than the folder we take its parent, so
# the same variable works whichever way the reader set it.
_env_dir = os.environ.get("EXPERT_STREAM_MODEL_DIR") or ""
if not _env_dir:
    _env_model = os.environ.get("EXPERT_STREAM_MODEL") or ""
    _env_dir = os.path.dirname(_env_model) if _env_model else ""
MODEL_DIR = _env_dir or r"E:\models\ds4f-iq1s\UD-IQ1_S"
OUT_CSV = os.path.join(REPO, "bench", "results", "expert_manifest.csv")

EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def main() -> int:
    if not os.path.isfile(INDEX_JSON):
        print(f"FATAL: {INDEX_JSON} not found. Run tools\\gguf_meta.py first.")
        return 2

    with open(INDEX_JSON, "r", encoding="ascii") as f:
        idx = json.load(f)

    # Shard order must match the order the reader will open them in, so the
    # shard_idx column means the same thing on both sides. Sorted basename is
    # the same order gguf_meta.py used (glob + sorted).
    shards = sorted(idx["shards"])
    shard_pos = {name: i for i, name in enumerate(shards)}

    rows = []
    for t in idx["tensors"]:
        m = EXPS_RE.match(t["name"])
        if not m:
            continue
        layer = int(m.group(1))
        kind = m.group(2)
        n_expert = t["shape_ggml_order"][-1]
        nb = t["n_bytes"]
        if n_expert <= 0 or nb % n_expert != 0:
            print(f"FATAL: {t['name']} n_bytes={nb} not divisible by n_expert={n_expert}")
            return 2
        rows.append({
            "layer": layer,
            "kind": kind,
            "shard_idx": shard_pos[t["shard_file"]],
            "base_offset": t["data_offset"],
            "stride": nb // n_expert,
            "n_expert": n_expert,
            "ggml_type": t["ggml_type"],
        })

    if not rows:
        print("FATAL: no routed-expert tensors found.")
        return 2

    rows.sort(key=lambda r: (r["layer"], {"gate": 0, "up": 1, "down": 2}[r["kind"]]))

    # Sanity: every layer must have all three kinds, or a replay would silently
    # read less than a real token does and report a flattering number.
    by_layer: dict[int, set] = {}
    for r in rows:
        by_layer.setdefault(r["layer"], set()).add(r["kind"])
    incomplete = {k: sorted(v) for k, v in by_layer.items() if len(v) != 3}
    if incomplete:
        print(f"FATAL: layers missing one of gate/up/down: {incomplete}")
        return 2

    # Verify every shard file the manifest references actually exists now, so
    # the C side fails at generation time here rather than mid-benchmark.
    missing = [s for s in shards if not os.path.isfile(os.path.join(MODEL_DIR, s))]
    if missing:
        print(f"FATAL: manifest references shard files that are not present: {missing}")
        return 2

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="ascii", newline="\n") as f:
        f.write("# expert slice manifest for DeepSeek-V4-Flash-0731 UD-IQ1_S\n")
        f.write("# expert e of a row lives at base_offset + e*stride, for stride bytes\n")
        for i, name in enumerate(shards):
            f.write(f"#S {i} {os.path.join(MODEL_DIR, name)}\n")
        f.write("layer,kind,shard_idx,base_offset,stride,n_expert\n")
        for r in rows:
            f.write(f"{r['layer']},{r['kind']},{r['shard_idx']},"
                    f"{r['base_offset']},{r['stride']},{r['n_expert']}\n")

    layers = sorted(by_layer.keys())
    per_layer_bytes = {}
    for r in rows:
        per_layer_bytes[r["layer"]] = per_layer_bytes.get(r["layer"], 0) + r["stride"]
    tot_one_expert_each_layer = sum(per_layer_bytes.values())

    print(f"wrote {OUT_CSV}")
    print(f"  shards      : {len(shards)}")
    print(f"  rows        : {len(rows)}  ({len(layers)} MoE layers x 3 kinds)")
    print(f"  n_expert    : {rows[0]['n_expert']}")
    print(f"  strides seen: {sorted(set(r['stride'] for r in rows))}")
    print(f"  one expert per layer, all layers = {tot_one_expert_each_layer:,} bytes")
    print(f"  top-6 routing => per token = {6*tot_one_expert_each_layer:,} bytes "
          f"({6*tot_one_expert_each_layer/(1024**3):.3f} GiB)")
    print(f"  reads per token = {len(rows)*6:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
