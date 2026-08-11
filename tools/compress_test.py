#!/usr/bin/env python3
"""
compress_test.py -- Task 0.14 KILL-OR-CONFIRM experiment: is there a free
throughput win from compressing GGUF expert weight bytes on disk?

IDEA UNDER TEST
----------------
The inference engine streams expert weights off SSD and is I/O bound (disk
busy ~90% of every step, CPU mostly idle). Disk measured at ~1.1 GB/s. ANY
lossless codec that (a) shrinks the on-disk bytes by more than ~15% and
(b) decompresses well above ~1.5 GB/s would be free throughput.

A prior doc estimate (reasoned about FP4 nibbles, i.e. "close to random
bits") says this dies at <=1.1x ratio. But this model actually uses
codebook quant formats (IQ1_S, IQ2_XXS, IQ3_XXS, MXFP4) whose *index*
streams may be skewed rather than uniform, which entropy coders CAN
exploit. So the prior estimate might not apply here. This script measures
it directly instead of re-reasoning about it.

WHAT THIS SCRIPT DOES
----------------------
1. Opens the 3 GGUF shards read-only (metadata-only memmap scan, via the
   local gguf-py -- this does not load tensor data into RAM).
2. Finds routed-expert tensors (blk.N.ffn_{gate,up,down}_exps.weight),
   which are 3-D with expert index as the OUTER (numpy axis-0) dimension,
   so one expert's raw bytes are a contiguous slice of the memmap.
3. Dynamically groups those tensors by their actual ggml quant type and
   samples ~8 expert blobs spread across several layers, covering all
   four quant types present in this model (IQ1_S, IQ2_XXS, IQ3_XXS,
   MXFP4) -- selection is derived live from the GGUF metadata, not
   hardcoded, so the script self-verifies its own coverage claim.
4. Loads each blob fully into RAM first (this read is excluded from all
   timing), then benchmarks, on the RAW QUANTIZED BYTES (never
   dequantized -- dequantizing to floats first would measure the wrong
   thing):
     - zstd level 1, zstd level 3, lz4 (default), zlib level 1
   reporting compression ratio, compression throughput, and DEcompression
   throughput (median of several repetitions -- this is the number that
   decides viability).
5. Runs an os.urandom() control of matching size through the same codecs.
   Truly random bytes compress to ~1.00x; if the real blobs score the same,
   there is no exploitable structure and the idea is dead.
6. Runs a cheap structural variant: byte-plane separation (deinterleave
   every Nth byte into its own stream, concatenate, then compress) for
   N=2 and N=4, using zstd level 3, to see if it exposes skew that is
   invisible in the interleaved on-disk layout.
7. Prints an explicit VERDICT (ALIVE / DEAD) against the stated threshold:
   ratio > 1.15 AND decompression throughput > 1.5 GB/s.

Read-only with respect to the model. Never materializes more than one
expert blob (a few MB) at a time; total bytes read off disk is a few tens
of MB.

Windows / Python 3.13. Pure ASCII output only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import statistics
import sys
import time
import traceback
import zlib

DEFAULT_MODEL_DIR = r"E:\models\ds4f-iq1s\UD-IQ1_S"
DEFAULT_GGUF_PY = r"D:\2025_Cursor_Dev\V4-local-serving\llama.cpp\gguf-py"
DEFAULT_CSV_OUT = r"D:\2025_Cursor_Dev\V4-local-serving\v4flash-16gb\bench\results\compress_test.csv"
DEFAULT_TXT_OUT = r"D:\2025_Cursor_Dev\V4-local-serving\v4flash-16gb\bench\results\compress_test.txt"

EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\.(ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight$")
KIND_LABEL = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}

REQUIRED_TYPES = ["IQ1_S", "IQ2_XXS", "IQ3_XXS", "MXFP4"]
BLOBS_PER_TYPE = 2  # -> 8 blobs total, 2 per required type, spread across layers

DECOMPRESS_REPS = 9   # timed repetitions; report median
COMPRESS_REPS = 5

RATIO_THRESHOLD = 1.15
DECOMP_GBPS_THRESHOLD = 1.5
DISK_GBPS = 1.1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--gguf-py", default=DEFAULT_GGUF_PY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    p.add_argument("--txt-out", default=DEFAULT_TXT_OUT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tee: console + text file, flushed as we go
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
# Codec wrappers: each is (name, level_label, compress(data)->bytes, decompress(cdata, orig_len)->bytes)
# ---------------------------------------------------------------------------

def build_codecs(zstd_mod, lz4_frame_mod):
    codecs = []

    def make_zstd(level):
        comp = zstd_mod.ZstdCompressor(level=level)
        decomp = zstd_mod.ZstdDecompressor()

        def c(data):
            return comp.compress(data)

        def d(cdata, orig_len):
            return decomp.decompress(cdata, max_output_size=orig_len)

        return c, d

    for lvl in (1, 3):
        c, d = make_zstd(lvl)
        codecs.append((f"zstd-{lvl}", c, d))

    def lz4_c(data):
        return lz4_frame_mod.compress(data)

    def lz4_d(cdata, orig_len):
        return lz4_frame_mod.decompress(cdata)

    codecs.append(("lz4", lz4_c, lz4_d))

    def zlib_c(data):
        return zlib.compress(data, 1)

    def zlib_d(cdata, orig_len):
        return zlib.decompress(cdata)

    codecs.append(("zlib-1", zlib_c, zlib_d))

    return codecs


def time_codec(codec_name, compress_fn, decompress_fn, data, compress_reps, decompress_reps):
    orig_len = len(data)

    # ---- compression: several reps, median time ----
    ctimes = []
    compressed = None
    for _ in range(compress_reps):
        t0 = time.perf_counter()
        compressed = compress_fn(data)
        t1 = time.perf_counter()
        ctimes.append(t1 - t0)
    ctime_median = statistics.median(ctimes)
    comp_len = len(compressed)

    # ---- decompression: warmup (untimed) + several timed reps, median ----
    decompress_fn(compressed, orig_len)  # warmup, not timed
    dtimes = []
    last_out = None
    for _ in range(decompress_reps):
        t0 = time.perf_counter()
        last_out = decompress_fn(compressed, orig_len)
        t1 = time.perf_counter()
        dtimes.append(t1 - t0)
    dtime_median = statistics.median(dtimes)

    if bytes(last_out) != data:
        raise RuntimeError(f"ROUNDTRIP MISMATCH for codec {codec_name}: decompressed output != original input")

    ratio = orig_len / comp_len if comp_len > 0 else float("inf")
    comp_mbps = (orig_len / 1e6) / ctime_median if ctime_median > 0 else float("inf")
    decomp_mbps = (orig_len / 1e6) / dtime_median if dtime_median > 0 else float("inf")

    return {
        "orig_bytes": orig_len,
        "comp_bytes": comp_len,
        "ratio": ratio,
        "compress_time_s_median": ctime_median,
        "compress_MBps": comp_mbps,
        "decompress_time_s_median": dtime_median,
        "decompress_MBps": decomp_mbps,
        "decompress_reps": decompress_reps,
        "compress_reps": compress_reps,
    }


def byte_plane_separate(data, n):
    """Deinterleave: stream i = data[i::n], concatenated stream0+stream1+...+stream(n-1)."""
    buf = bytearray(len(data))
    pos = 0
    for i in range(n):
        plane = data[i::n]
        buf[pos:pos + len(plane)] = plane
        pos += len(plane)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tee = Tee(args.txt_out)
    log = tee.log

    log("=" * 100)
    log("compress_test.py -- Task 0.14 compression kill-or-confirm experiment")
    log("=" * 100)
    log(f"model_dir = {args.model_dir}")
    log(f"gguf_py   = {args.gguf_py}")
    log(f"seed      = {args.seed}")
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
    except Exception as e:
        log(f"FAILED to import local gguf-py from {args.gguf_py}: {e}")
        log(traceback.format_exc())
        tee.close()
        sys.exit(1)

    # ---- codec availability: install what's missing, report plainly if it fails ----
    zstd_mod = None
    lz4_frame_mod = None
    missing_codecs = []

    try:
        import zstandard as zstd_mod
        log(f"zstandard: available (version {zstd_mod.__version__})")
    except ImportError:
        log("zstandard not found; attempting `python -m pip install zstandard` ...")
        rc = os.system(f'"{sys.executable}" -m pip install zstandard')
        try:
            import zstandard as zstd_mod
            log(f"zstandard: installed OK (version {zstd_mod.__version__})")
        except ImportError:
            log("zstandard: INSTALL FAILED. zstd-1 and zstd-3 will be SKIPPED (not silently -- "
                "recorded here explicitly).")
            missing_codecs.append("zstd-1")
            missing_codecs.append("zstd-3")

    try:
        import lz4.frame as lz4_frame_mod
        log("lz4: available")
    except ImportError:
        log("lz4 not found; attempting `python -m pip install lz4` ...")
        rc = os.system(f'"{sys.executable}" -m pip install lz4')
        try:
            import lz4.frame as lz4_frame_mod
            log("lz4: installed OK")
        except ImportError:
            log("lz4: INSTALL FAILED. lz4 codec will be SKIPPED (not silently -- recorded here explicitly).")
            missing_codecs.append("lz4")

    log("zlib: available (stdlib)")
    log("")

    # -----------------------------------------------------------------
    # Locate shards, open metadata-only (memmap), index expert tensors.
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
    tensor_index = {}  # name -> ReaderTensor
    for sp in shard_paths:
        t0 = time.time()
        r = gguf.GGUFReader(sp)
        log(f"  opened {os.path.basename(sp)} in {time.time() - t0:.2f}s, {len(r.tensors)} tensors")
        for t in r.tensors:
            m = EXPERT_TENSOR_RE.match(t.name)
            if m:
                tensor_index[t.name] = t
    log(f"Indexed {len(tensor_index)} routed-expert tensors across shards.")
    log("")

    if not tensor_index:
        log("No routed-expert tensors found. Aborting.")
        tee.close()
        sys.exit(1)

    sample_t = next(iter(tensor_index.values()))
    n_expert = sample_t.data.shape[0]
    log(f"Experts per routed-expert tensor (from tensor shape, axis 0): {n_expert}")
    log("")

    # -----------------------------------------------------------------
    # Group tensors by ACTUAL ggml quant type (live from GGUF metadata,
    # not from any external index file) -- this is how we prove coverage
    # of all four required quant types rather than assuming it.
    # -----------------------------------------------------------------
    by_type = {}
    for name, t in tensor_index.items():
        by_type.setdefault(t.tensor_type.name, []).append(name)

    log("Quant types found among routed-expert tensors (live scan):")
    for qt, names in sorted(by_type.items()):
        layers = sorted(set(int(EXPERT_TENSOR_RE.match(n).group(1)) for n in names))
        log(f"  {qt:10s}: {len(names)} tensor(s), layers {layers}")
    log("")

    missing_types = [t for t in REQUIRED_TYPES if t not in by_type]
    if missing_types:
        log(f"WARNING: required quant type(s) not found among routed-expert tensors: {missing_types}")
        log("Proceeding with whichever required types ARE present -- coverage claim will be adjusted "
            "in the report accordingly.")
    log("")

    # -----------------------------------------------------------------
    # Build sampling plan: for each required type present, pick up to
    # BLOBS_PER_TYPE tensors spread across distinct layers (min & max
    # layer present for that type), then one expert index per tensor
    # via seeded RNG.
    # -----------------------------------------------------------------
    py_rng = random.Random(args.seed)
    plan = []  # list of (tensor_name, layer, kind, qtype_name, expert_idx)

    for qtype in REQUIRED_TYPES:
        names = by_type.get(qtype, [])
        if not names:
            continue
        by_layer = sorted(names, key=lambda n: int(EXPERT_TENSOR_RE.match(n).group(1)))
        distinct_layers = sorted(set(int(EXPERT_TENSOR_RE.match(n).group(1)) for n in by_layer))
        if len(distinct_layers) >= BLOBS_PER_TYPE:
            idxs = [round(i * (len(distinct_layers) - 1) / (BLOBS_PER_TYPE - 1)) for i in range(BLOBS_PER_TYPE)]
            chosen_layers = sorted(set(distinct_layers[i] for i in idxs))
        else:
            chosen_layers = distinct_layers
        chosen_names = []
        for layer in chosen_layers:
            cands = [n for n in by_layer if int(EXPERT_TENSOR_RE.match(n).group(1)) == layer]
            chosen_names.append(py_rng.choice(cands))
        # if fewer distinct layers than BLOBS_PER_TYPE, top up with more tensors from same layers
        i = 0
        while len(chosen_names) < BLOBS_PER_TYPE and i < len(by_layer):
            if by_layer[i] not in chosen_names:
                chosen_names.append(by_layer[i])
            i += 1

        for name in chosen_names[:BLOBS_PER_TYPE]:
            m = EXPERT_TENSOR_RE.match(name)
            layer = int(m.group(1))
            kind = KIND_LABEL[m.group(2)]
            expert_idx = py_rng.randrange(n_expert)
            plan.append((name, layer, kind, qtype, expert_idx))

    log(f"Sampling plan: {len(plan)} blob(s), covering {len(set(p[3] for p in plan))} of "
        f"{len(REQUIRED_TYPES)} required quant types, {len(set(p[1] for p in plan))} distinct layer(s)")
    for name, layer, kind, qtype, expert_idx in plan:
        log(f"  layer={layer:3d} kind={kind:4s} qtype={qtype:10s} expert={expert_idx:3d}  tensor={name}")
    log("")

    if not plan:
        log("Sampling plan is empty -- cannot proceed.")
        tee.close()
        sys.exit(1)

    # -----------------------------------------------------------------
    # Codec list
    # -----------------------------------------------------------------
    codecs = []
    if zstd_mod is not None:
        for lvl in (1, 3):
            comp = zstd_mod.ZstdCompressor(level=lvl)
            decomp = zstd_mod.ZstdDecompressor()
            codecs.append((
                f"zstd-{lvl}",
                (lambda data, comp=comp: comp.compress(data)),
                (lambda cdata, orig_len, decomp=decomp: decomp.decompress(cdata, max_output_size=orig_len)),
            ))
    if lz4_frame_mod is not None:
        codecs.append((
            "lz4",
            (lambda data: lz4_frame_mod.compress(data)),
            (lambda cdata, orig_len: lz4_frame_mod.decompress(cdata)),
        ))
    codecs.append((
        "zlib-1",
        (lambda data: zlib.compress(data, 1)),
        (lambda cdata, orig_len: zlib.decompress(cdata)),
    ))

    log(f"Codecs under test: {[c[0] for c in codecs]}")
    if missing_codecs:
        log(f"Codecs SKIPPED due to failed install: {missing_codecs}")
    log("")

    # find the "primary" codec for the byte-plane structural test: prefer zstd-3, else first available
    primary_name = "zstd-3" if any(c[0] == "zstd-3" for c in codecs) else codecs[0][0]
    primary_codec = next(c for c in codecs if c[0] == primary_name)
    log(f"Byte-plane separation structural test will use codec: {primary_name}")
    log("")

    # -----------------------------------------------------------------
    # CSV setup
    # -----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    fieldnames = [
        "blob_id", "layer", "kind", "tensor_name", "expert_idx", "quant_type",
        "variant", "codec", "orig_bytes", "comp_bytes", "ratio",
        "compress_time_s_median", "compress_MBps",
        "decompress_time_s_median", "decompress_MBps",
        "decompress_reps", "compress_reps",
    ]
    csv_fh = open(args.csv_out, "w", newline="", encoding="ascii")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    csv_writer.writeheader()

    all_results = []  # list of dict rows (for aggregate + verdict)

    log("=" * 100)
    log("PER-BLOB RESULTS")
    log("=" * 100)

    total_bytes_read = 0

    for blob_id, (name, layer, kind, qtype, expert_idx) in enumerate(plan):
        t = tensor_index[name]
        log(f"[blob {blob_id}] layer={layer} kind={kind} qtype={qtype} expert={expert_idx} tensor={name}")

        t0 = time.time()
        arr = np.array(t.data[expert_idx], copy=True)  # load into RAM; excluded from codec timing
        load_time = time.time() - t0
        data = arr.tobytes()
        del arr
        total_bytes_read += len(data)
        log(f"  loaded {len(data)} bytes ({len(data)/1e6:.3f} MB) from memmap in {load_time:.3f}s "
            f"(disk read time, NOT counted in codec timing)")

        # ---- random control, same size ----
        random_data = os.urandom(len(data))

        # ---- byte-plane variants (real data only) ----
        plane2 = byte_plane_separate(data, 2)
        plane4 = byte_plane_separate(data, 4)

        def emit(variant, codec_name, compress_fn, decompress_fn, payload):
            res = time_codec(codec_name, compress_fn, decompress_fn, payload, COMPRESS_REPS, DECOMPRESS_REPS)
            row = {
                "blob_id": blob_id, "layer": layer, "kind": kind, "tensor_name": name,
                "expert_idx": expert_idx, "quant_type": qtype, "variant": variant, "codec": codec_name,
                "orig_bytes": res["orig_bytes"], "comp_bytes": res["comp_bytes"],
                "ratio": round(res["ratio"], 4),
                "compress_time_s_median": round(res["compress_time_s_median"], 6),
                "compress_MBps": round(res["compress_MBps"], 2),
                "decompress_time_s_median": round(res["decompress_time_s_median"], 6),
                "decompress_MBps": round(res["decompress_MBps"], 2),
                "decompress_reps": res["decompress_reps"], "compress_reps": res["compress_reps"],
            }
            csv_writer.writerow(row)
            csv_fh.flush()
            all_results.append(row)
            log(f"    [{variant:16s}] {codec_name:8s} ratio={res['ratio']:6.3f}x  "
                f"comp={res['compress_MBps']:8.1f} MB/s  decomp={res['decompress_MBps']:8.1f} MB/s "
                f"({res['decompress_MBps']/1000:.3f} GB/s, median of {res['decompress_reps']} reps)")

        for codec_name, cfn, dfn in codecs:
            emit("raw", codec_name, cfn, dfn, data)

        for codec_name, cfn, dfn in codecs:
            emit("random_control", codec_name, cfn, dfn, random_data)

        pname, pcfn, pdfn = primary_codec
        emit("byteplane_N2", pname, pcfn, pdfn, plane2)
        emit("byteplane_N4", pname, pcfn, pdfn, plane4)

        log("")

    csv_fh.close()
    log(f"Total bytes read from disk this run: {total_bytes_read} ({total_bytes_read/1e6:.1f} MB)")
    log(f"CSV written to {args.csv_out}")
    log("")

    # -----------------------------------------------------------------
    # AGGREGATE: per quant type, per codec, mean ratio / decompression MB/s
    # -----------------------------------------------------------------
    log("=" * 100)
    log("PER-QUANT-TYPE x CODEC SUMMARY (raw variant, mean across sampled blobs of that type)")
    log("=" * 100)

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    raw_rows = [r for r in all_results if r["variant"] == "raw"]
    random_rows = [r for r in all_results if r["variant"] == "random_control"]

    header = f"  {'quant_type':10s} {'codec':8s} {'n':>3s} {'ratio':>8s} {'comp_MBps':>10s} {'decomp_MBps':>12s} {'decomp_GBps':>12s}"
    log(header)
    summary_by_type_codec = {}
    for qtype in REQUIRED_TYPES:
        for codec_name, _, _ in codecs:
            rows = [r for r in raw_rows if r["quant_type"] == qtype and r["codec"] == codec_name]
            if not rows:
                continue
            mr = mean(r["ratio"] for r in rows)
            mc = mean(r["compress_MBps"] for r in rows)
            md = mean(r["decompress_MBps"] for r in rows)
            summary_by_type_codec[(qtype, codec_name)] = (mr, mc, md, len(rows))
            log(f"  {qtype:10s} {codec_name:8s} {len(rows):3d} {mr:8.3f} {mc:10.1f} {md:12.1f} {md/1000:12.3f}")
    log("")

    log("=" * 100)
    log("RANDOM CONTROL SUMMARY (os.urandom, matching sizes, mean across all blobs)")
    log("=" * 100)
    for codec_name, _, _ in codecs:
        rows = [r for r in random_rows if r["codec"] == codec_name]
        if not rows:
            continue
        mr = mean(r["ratio"] for r in rows)
        log(f"  {codec_name:8s}  n={len(rows):3d}  mean_ratio={mr:.4f}x")
    log("")

    log("=" * 100)
    log("BYTE-PLANE SEPARATION (structural test, codec=%s)" % primary_name)
    log("=" * 100)
    bp_rows = [r for r in all_results if r["variant"] in ("byteplane_N2", "byteplane_N4")]
    raw_primary_rows = {r["blob_id"]: r for r in raw_rows if r["codec"] == primary_name}
    for r in bp_rows:
        base = raw_primary_rows.get(r["blob_id"])
        base_ratio = base["ratio"] if base else float("nan")
        delta = r["ratio"] - base_ratio
        log(f"  blob {r['blob_id']:2d} ({r['quant_type']:10s}) {r['variant']:14s}: ratio={r['ratio']:.4f}x "
            f"vs raw-interleaved {base_ratio:.4f}x  (delta={delta:+.4f})")
    bp2 = [r["ratio"] - raw_primary_rows[r["blob_id"]]["ratio"] for r in bp_rows
           if r["variant"] == "byteplane_N2" and r["blob_id"] in raw_primary_rows]
    bp4 = [r["ratio"] - raw_primary_rows[r["blob_id"]]["ratio"] for r in bp_rows
           if r["variant"] == "byteplane_N4" and r["blob_id"] in raw_primary_rows]
    log("")
    log(f"  mean delta ratio, byteplane N=2 vs raw: {mean(bp2):+.4f}")
    log(f"  mean delta ratio, byteplane N=4 vs raw: {mean(bp4):+.4f}")
    helps = mean(bp2 + bp4) > 0.02  # more than 2 points of ratio improvement to call it "helps"
    log(f"  byte-plane separation {'HELPS measurably' if helps else 'does NOT help measurably'} "
        f"on this data.")
    log("")

    # -----------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------
    log("=" * 100)
    log("VERDICT")
    log("=" * 100)
    log(f"Threshold test: idea is only worth building if ratio > {RATIO_THRESHOLD:.2f} "
        f"AND decompression throughput > {DECOMP_GBPS_THRESHOLD:.2f} GB/s (measured disk = {DISK_GBPS:.2f} GB/s).")
    log("")

    best_random_ratio = max((v[0] for k, v in summary_by_type_codec.items()), default=float("nan"))
    all_random_ratios = [mean(r["ratio"] for r in random_rows if r["codec"] == c[0]) for c in codecs]
    random_ratio_overall = mean(x for x in all_random_ratios if x == x)  # filter NaN

    best_combo = None
    best_ratio_seen = -1.0
    for (qtype, codec_name), (mr, mc, md, n) in summary_by_type_codec.items():
        if mr > best_ratio_seen:
            best_ratio_seen = mr
            best_combo = (qtype, codec_name, mr, mc, md, n)

    # also find best overall by pooling across quant types per codec (data-weighted mean over all raw rows)
    best_codec_overall = None
    best_codec_ratio = -1.0
    best_codec_decomp = 0.0
    for codec_name, _, _ in codecs:
        rows = [r for r in raw_rows if r["codec"] == codec_name]
        if not rows:
            continue
        mr = mean(r["ratio"] for r in rows)
        md = mean(r["decompress_MBps"] for r in rows)
        if mr > best_codec_ratio:
            best_codec_ratio = mr
            best_codec_overall = codec_name
            best_codec_decomp = md

    log(f"Random control (os.urandom, all codecs, all blob sizes): mean ratio = {random_ratio_overall:.4f}x "
        f"(expected ~1.00x for incompressible data)")
    log("")
    if best_combo:
        qtype, codec_name, mr, mc, md, n = best_combo
        log(f"Best single (quant_type, codec) combo on RAW quantized bytes: "
            f"{qtype} + {codec_name} -> ratio={mr:.4f}x, decompress={md:.1f} MB/s ({md/1000:.4f} GB/s), n={n}")
    log(f"Best codec overall (pooled across all sampled blobs, raw variant): "
        f"{best_codec_overall} -> mean ratio={best_codec_ratio:.4f}x, mean decompress={best_codec_decomp:.1f} MB/s "
        f"({best_codec_decomp/1000:.4f} GB/s)")
    log("")

    overall_best_ratio = max(best_codec_ratio, best_ratio_seen if best_combo else -1.0)
    # decompression throughput at that best ratio point (use the (qtype,codec) combo, the more specific number)
    verdict_ratio = best_combo[2] if best_combo else best_codec_ratio
    verdict_decomp_gbps = (best_combo[4] if best_combo else best_codec_decomp) / 1000.0
    verdict_codec = best_combo[1] if best_combo else best_codec_overall

    ratio_ok = verdict_ratio > RATIO_THRESHOLD
    decomp_ok = verdict_decomp_gbps > DECOMP_GBPS_THRESHOLD
    near_random = (verdict_ratio - random_ratio_overall) < 0.05

    if ratio_ok and decomp_ok:
        log(f"ALIVE: codec {verdict_codec} gives {verdict_ratio:.2f}x at {verdict_decomp_gbps:.3f} GB/s "
            f"decompression, above the {DISK_GBPS:.1f} GB/s disk and above the {RATIO_THRESHOLD:.2f}x / "
            f"{DECOMP_GBPS_THRESHOLD:.1f} GB/s thresholds. Worth building.")
    else:
        reasons = []
        if not ratio_ok:
            reasons.append(f"ratio {verdict_ratio:.3f}x <= threshold {RATIO_THRESHOLD:.2f}x")
        if not decomp_ok:
            reasons.append(f"decompression {verdict_decomp_gbps:.3f} GB/s <= threshold {DECOMP_GBPS_THRESHOLD:.1f} GB/s")
        random_note = (f" Best real-data ratio ({verdict_ratio:.3f}x) is within 0.05 of the random-data "
                        f"control ({random_ratio_overall:.3f}x): the quantized bytes carry essentially no "
                        f"exploitable structure for a general-purpose byte-level entropy coder."
                        if near_random else
                        f" Best real-data ratio ({verdict_ratio:.3f}x) is measurably above the random-data "
                        f"control ({random_ratio_overall:.3f}x), but still not enough to clear the threshold.")
        log(f"DEAD: best ratio {verdict_ratio:.3f}x versus {random_ratio_overall:.3f}x for random control. "
            f"Failed: {'; '.join(reasons)}.{random_note} Do not revisit.")
    log("=" * 100)

    tee.close()


if __name__ == "__main__":
    main()
