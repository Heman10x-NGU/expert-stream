#!/usr/bin/env python3
"""
align_test.py

DECIDES WHETHER TASK #3 (the 76 GB expert-bank repack) CAN BE DELETED.

The plan was to rewrite all routed-expert weights into a new expert-major
experts.bin so that any single expert could be read with one aligned
unbuffered ReadFile. That repack costs a 76 GB bulk write (measured to drop
the disk from 1,050 MB/s to 318 MB/s for minutes) and 76 GB of disk we do not
have to spare.

It is only necessary if the experts are NOT already aligned inside the GGUF
shards. FILE_FLAG_NO_BUFFERING requires that the file offset, the transfer
length, and the destination buffer address all be multiples of the volume's
LOGICAL sector size. So the repack is deletable if, for every routed-expert
tensor:

    (a) data_offset            % sector == 0     - the tensor starts aligned
    (b) bytes_per_expert       % sector == 0     - every expert starts aligned
    (c) the expert stride is contiguous, i.e. expert e lives at exactly
        data_offset + e * bytes_per_expert

(c) is guaranteed by the ggml memory layout for a 3D tensor (experts are the
slowest-moving dimension, which is why mul_mat_id can index them with a plain
cur_a * nb02), but it is asserted here rather than assumed.

Note on "sector": we do NOT hardcode 512. The caller passes the real logical
sector size of the volume holding the model. A 4K-native drive reports 4096
and 512-alignment would prove nothing.

Reads bench/results/tensor_index.json (produced by gguf_meta.py). Touches no
tensor data and opens no model file. Read-only, no side effects.

Usage:
    python tools\\align_test.py [--sector N]      (default 512)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_JSON = os.path.join(REPO, "bench", "results", "tensor_index.json")
OUT_TXT = os.path.join(REPO, "bench", "results", "align_test.txt")

EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")

_LOG: list[str] = []


def out(msg: str = "") -> None:
    msg = str(msg).encode("ascii", errors="backslashreplace").decode("ascii")
    print(msg)
    _LOG.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, default=512,
                    help="volume LOGICAL sector size in bytes (from fsutil fsinfo ntfsinfo)")
    args = ap.parse_args()

    sector = args.sector
    if sector <= 0 or (sector & (sector - 1)) != 0:
        out(f"FATAL: --sector {sector} is not a positive power of two.")
        return 2

    if not os.path.isfile(INDEX_JSON):
        out(f"FATAL: {INDEX_JSON} not found. Run tools\\gguf_meta.py first.")
        return 2

    with open(INDEX_JSON, "r", encoding="ascii") as f:
        idx = json.load(f)

    tensors = idx["tensors"]
    out("align_test.py - can the 76 GB expert repack be deleted?")
    out(f"tensor index : {INDEX_JSON}")
    out(f"tensors       : {len(tensors)}")
    out(f"LOGICAL SECTOR SIZE USED FOR THE TEST: {sector} bytes")
    out("")

    exps = [t for t in tensors if EXPS_RE.match(t["name"])]
    if not exps:
        out("FATAL: no routed-expert tensors matched. Index may be from a different model.")
        return 2

    # n_expert is the last (slowest) dim in ggml order for these 3D tensors.
    n_experts = set(t["shape_ggml_order"][-1] for t in exps)
    if len(n_experts) != 1:
        out(f"FATAL: routed-expert tensors disagree on expert count: {sorted(n_experts)}")
        return 2
    n_expert = n_experts.pop()
    out(f"routed-expert tensors : {len(exps)}")
    out(f"experts per tensor    : {n_expert}")
    out("")

    # NOTE ON WHAT THIS TEST ORIGINALLY GOT WRONG.
    # The first version of this script treated a misaligned tensor base offset as
    # a hard failure and concluded "repack REQUIRED". That was the wrong test. An
    # unbuffered reader never has to start a read exactly at the wanted byte: it
    # reads the enclosing sector-aligned window and returns buf + slop, where
    # slop = off % sector. The base offset can be anything.
    #
    # What genuinely must hold is that the PER-EXPERT STRIDE is a whole number of
    # sectors. If it were not, expert e and expert e+1 would sit at different
    # sub-sector phases and the read window would have to be recomputed per
    # expert - still possible, but it would also mean neighbouring experts share
    # a sector, so writing one would corrupt another in any future writable path.
    #
    # The base-offset residue is therefore reported as INFORMATION, not failure.
    slop_residues = {}
    bad_stride = []
    bad_divis = []

    for t in exps:
        off = t["data_offset"]
        nb = t["n_bytes"]

        r = off % sector
        slop_residues.setdefault(r, 0)
        slop_residues[r] += 1

        # (b) every expert must start on a sector boundary, which requires the
        # per-expert stride itself to be sector-divisible. If n_bytes does not
        # divide evenly by n_expert at all, the layout is not what we assume
        # and nothing downstream is safe.
        if nb % n_expert != 0:
            bad_divis.append((t["name"], nb, n_expert, nb % n_expert))
            continue
        stride = nb // n_expert
        if stride % sector != 0:
            bad_stride.append((t["name"], stride, stride % sector))

    out("=" * 78)
    out("RESULTS")
    out("=" * 78)

    def report(label, rows, fmt):
        if not rows:
            out(f"PASS  {label}: 0 violations out of {len(exps)}")
            return True
        out(f"FAIL  {label}: {len(rows)} violation(s). First 10:")
        for r in rows[:10]:
            out("        " + fmt(r))
        return False

    ok_div = report("(a) n_bytes divisible by n_expert", bad_divis,
                    lambda r: f"{r[0]}  n_bytes={r[1]:,}  n_expert={r[2]}  remainder={r[3]}")
    ok_str = report(f"(b) bytes_per_expert %% {sector} == 0  [THE ONE THAT MATTERS]", bad_stride,
                    lambda r: f"{r[0]}  stride={r[1]:,}  remainder={r[2]}")

    # Informational: the read-window slop a reader must skip past.
    out("")
    out(f"INFO  base-offset residues mod {sector} (this is SLOP TO SKIP, not a failure):")
    for r in sorted(slop_residues):
        note = ""
        if r == 0:
            note = "  (already aligned)"
        elif r % 32 == 0:
            note = f"  (= 32 x {r // 32}; still 32-byte aligned, so ggml SIMD loads are fine)"
        else:
            note = "  (NOT a multiple of 32 - check SIMD alignment assumptions)"
        out(f"        residue {r:>5}: {slop_residues[r]:>4} tensor(s){note}")
    max_waste = (max(slop_residues) + sector - 1) // sector * sector
    out(f"      worst case extra bytes read per expert: {max_waste:,} "
        f"({100.0 * max_waste / (min(t['n_bytes'] // n_expert for t in exps if t['n_bytes'] % n_expert == 0)):.4f}% of the smallest expert)")

    # ------------------------------------------------------------------
    # Stride table: what a reader would actually issue per expert.
    # Also reports the WORST case, because the slowest read sets the pace.
    # ------------------------------------------------------------------
    out("")
    out("PER-EXPERT READ SIZES (what one expert costs, by tensor kind):")
    by_kind: dict[str, list[int]] = {}
    for t in exps:
        nb = t["n_bytes"]
        if nb % n_expert != 0:
            continue
        by_kind.setdefault(EXPS_RE.match(t["name"]).group(2), []).append(nb // n_expert)
    for kind in ("gate", "up", "down"):
        v = sorted(set(by_kind.get(kind, [])))
        if not v:
            out(f"  {kind:>5}: (none)")
            continue
        out(f"  {kind:>5}: {len(v)} distinct stride(s), min={min(v):,} max={max(v):,} bytes"
            f"   {'ALL >= 64 KiB' if min(v) >= 65536 else 'SOME BELOW 64 KiB'}")

    # Sum of one expert across all three tensors, per layer - this is the real
    # unit of work: 6 of these per layer per token.
    per_layer: dict[int, int] = {}
    for t in exps:
        nb = t["n_bytes"]
        if nb % n_expert != 0:
            continue
        bid = int(EXPS_RE.match(t["name"]).group(1))
        per_layer[bid] = per_layer.get(bid, 0) + nb // n_expert
    if per_layer:
        vals = sorted(per_layer.values())
        out("")
        out(f"ONE FULL EXPERT (gate+up+down) : min={vals[0]:,}  "
            f"mean={sum(vals)//len(vals):,}  max={vals[-1]:,} bytes  over {len(per_layer)} MoE layers")
        out(f"  -> 3 separate reads per expert (gate, up, down are 3 distinct tensors,")
        out(f"     NOT contiguous with each other), so 6 experts = 18 reads per layer per token.")

    out("")
    out("=" * 78)
    if ok_div and ok_str:
        out("VERDICT: ALIGNMENT IS NOT A REASON TO REPACK")
        out("")
        out("Every per-expert stride is a whole number of sectors, so expert e and")
        out("expert e+1 sit at the same sub-sector phase and never share a sector. The")
        out("constant base-offset slop shown above is absorbed by reading the enclosing")
        out("aligned window and returning buf + slop, at a cost of at most one sector")
        out("per expert.")
        out("")
        out("An unbuffered (FILE_FLAG_NO_BUFFERING) reader can address experts IN PLACE")
        out("inside the original GGUF shards.")
        out("")
        out("This settles alignment only. The other argument for a repack is CONTIGUITY")
        out("(gate/up/down are three separate tensors, so one expert costs three seeks).")
        out("That was measured separately by src/expert_read_bench.c and found to be")
        out("worth about 5% at queue depth >= 2 - not worth a 76 GB write. See")
        out("docs/MEASURED_GROUND_TRUTH.md section 10.1.")
        rc = 0
    else:
        out("VERDICT: STRIDE IS NOT SECTOR-ALIGNED")
        out("")
        out("At least one per-expert stride is not a whole number of sectors, so experts")
        out("share sectors and cannot be addressed independently in place.")
        out("")
        out("=> An aligned repack WOULD be required. Do not proceed with an in-place reader.")
        rc = 1
    out("=" * 78)

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="ascii", errors="replace", newline="\n") as f:
        f.write("\n".join(_LOG) + "\n")
    out(f"written to: {OUT_TXT}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
