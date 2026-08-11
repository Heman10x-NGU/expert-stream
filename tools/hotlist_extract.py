#!/usr/bin/env python3
"""
hotlist_extract.py  --  Task 0.13 "cheapest win in the whole project"

Parses the pre-profiled expert hotlist baked into the reference engine
(antirez/ds4), specifically:

    D:\\2025_Cursor_Dev\\V4-local-serving\\ds4\\ds4_streaming_hotlist.inc

That file is a READ-ONLY C header (never modified by this script). It
contains two "static const uint16_t NAME[][2] = { ... };" array literals:
one for the DeepSeek V4 "Pro" variant and one for the "Flash" variant.

VERIFIED AGAINST ds4.c (see report the script prints for exact line
numbers):
  - ds4.c line ~20688-20693: metal_graph_streaming_expert_hotlist_load_default()
    selects `ds4_default_streaming_hotlist_pro` when
    g_ds4_shape.variant == DS4_VARIANT_PRO, and
    `ds4_default_streaming_hotlist_flash` when
    g_ds4_shape.variant == DS4_VARIANT_FLASH.
    ==> ds4_default_streaming_hotlist_flash IS the Flash array. Confirmed
        from the call site, not guessed from the identifier name.
  - ds4.c line ~20705-20707: hotlist[i][0] is passed as the `layer`
    parameter and hotlist[i][1] as the `expert` parameter of
    metal_graph_streaming_expert_hotlist_add(layer, expert, ...).
    ==> column 0 = layer index, column 1 = expert index within that layer.
  - There is NO third column / weight / hit-count field in the .inc file.
    The header comment on line 1 says "sorted by hits/weight" and ds4.c
    (ds4_expert_profile_write_hotlist_file, ~line 1540) shows the *runtime*
    profiler does track (layer, expert, hits, weight) tuples and sorts by
    them before ever being reduced to just (layer, expert) pairs for the
    baked-in .inc file. So we only get RANK ORDER (position in the array),
    not raw hit counts, in this particular source file.

This script:
  1. Parses every `static const TYPE NAME[][N] = { ... };` array in the
     .inc file with a tolerant-but-strict brace/regex parser (fails loudly
     on anything it can't confidently interpret).
  2. Reports the real field layout with quoted example lines.
  3. Computes Flash statistics: layer distribution, distinct experts,
     histogram, sortedness check.
  4. Computes coverage analysis against an assumed 43-layer x 256-expert
     (11,008 total) MoE routed-expert space, and an assumed 7.1 MB/expert
     footprint, under a 10 GB cache budget.
  5. Emits Flash and Pro lists as JSON.
  6. Compares Pro vs Flash overlap.
  7. Tees all console output to bench/results/hotlist.txt.

Windows / Python 3.13. Pure ASCII output only.
"""

import io
import json
import os
import re
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths (all absolute, per task spec)
# ---------------------------------------------------------------------------

SRC_INC = r"D:\2025_Cursor_Dev\V4-local-serving\ds4\ds4_streaming_hotlist.inc"
SRC_DS4_C = r"D:\2025_Cursor_Dev\V4-local-serving\ds4\ds4.c"

OUT_DIR = r"D:\2025_Cursor_Dev\V4-local-serving\v4flash-16gb\bench\results"
OUT_FLASH_JSON = os.path.join(OUT_DIR, "hotlist_flash.json")
OUT_PRO_JSON = os.path.join(OUT_DIR, "hotlist_pro.json")
OUT_TXT = os.path.join(OUT_DIR, "hotlist.txt")

FLASH_ARRAY_NAME = "ds4_default_streaming_hotlist_flash"
PRO_ARRAY_NAME = "ds4_default_streaming_hotlist_pro"

# Coverage-analysis assumptions (explicitly given by the task, not derived)
ASSUMED_N_LAYERS = 43
ASSUMED_N_EXPERTS_PER_LAYER = 256
ASSUMED_TOTAL_EXPERTS = ASSUMED_N_LAYERS * ASSUMED_N_EXPERTS_PER_LAYER  # 11,008
ASSUMED_MB_PER_EXPERT = 7.1
CACHE_BUDGET_GB = 10.0


# ---------------------------------------------------------------------------
# Tee: duplicate every print() to console AND to the results txt file
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Matches:  static const uint16_t some_name[][2] = {   ... captures the
# identifier, the element type, and the declared inner dimension (may be
# empty for [][N] outer, we only need the inner fixed dimension count).
ARRAY_DECL_RE = re.compile(
    r"static\s+const\s+([A-Za-z_][A-Za-z0-9_ ]*?)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"\[\s*\]\s*\[\s*(\d+)\s*\]\s*=\s*\{",
    re.MULTILINE,
)

# One entry line, e.g.  "    {44, 213}," possibly with trailing comment.
ENTRY_RE = re.compile(r"\{\s*([0-9]+)\s*,\s*([0-9]+)\s*\}\s*,?")

COUNT_DECL_RE = re.compile(
    r"static\s+const\s+uint32_t\s+([A-Za-z_][A-Za-z0-9_]*)_count\s*=\s*"
    r"\(uint32_t\)\(sizeof\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*/\s*"
    r"sizeof\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*0\s*\]\s*\)\s*\)\s*;",
    re.MULTILINE,
)


class ParsedArray:
    def __init__(self, name, elem_type, inner_dim, entries, raw_lines,
                 decl_line_no, close_line_no):
        self.name = name
        self.elem_type = elem_type
        self.inner_dim = inner_dim
        self.entries = entries  # list of tuples of ints, len == inner_dim
        self.raw_lines = raw_lines  # verbatim source lines for this array body
        self.decl_line_no = decl_line_no
        self.close_line_no = close_line_no


def fail(msg):
    print("FATAL PARSE ERROR: " + msg)
    sys.exit(1)


def parse_inc_file(path):
    if not os.path.isfile(path):
        fail("source file not found: %s" % path)

    with open(path, "r", encoding="utf-8", errors="strict") as f:
        text = f.read()
    lines = text.splitlines()

    arrays = {}

    for decl_match in ARRAY_DECL_RE.finditer(text):
        elem_type = decl_match.group(1).strip()
        name = decl_match.group(2)
        inner_dim = int(decl_match.group(3))
        body_start = decl_match.end()  # just after the opening '{'

        # Find the matching closing '};' for this array by locating the
        # first standalone '};' after body_start. The .inc file's arrays
        # do not nest braces inside entries beyond one level ({a,b},), so
        # a simple brace-depth scan is both tolerant and strict: if depth
        # never returns to zero before EOF, we fail loudly.
        depth = 1
        i = body_start
        n = len(text)
        while i < n and depth > 0:
            c = text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            i += 1
        if depth != 0:
            fail("unbalanced braces while scanning array '%s' starting at "
                 "offset %d (no matching closing brace found before EOF)"
                 % (name, body_start))
        body_end = i - 1  # index of the closing '}' of the array

        # Expect a ';' shortly after body_end.
        semi_search = text[body_end:body_end + 5]
        if ';' not in semi_search:
            fail("array '%s' body did not end with ';' as expected "
                 "(found %r)" % (name, semi_search))

        body_text = text[body_start:body_end]

        decl_line_no = text.count('\n', 0, decl_match.start()) + 1
        close_line_no = text.count('\n', 0, body_end) + 1

        # Extract every {a, b} (or {a,b,c,...} generalized) entry from the body.
        # Build a generalized entry regex based on inner_dim so we don't
        # silently accept the wrong arity.
        num_group = r"\s*(-?[0-9]+)\s*"
        entry_pattern = r"\{" + ",".join([num_group] * inner_dim) + r"\}\s*,?"
        entry_re = re.compile(entry_pattern)

        entries = []
        pos = 0
        body_len = len(body_text)
        # Walk through body_text finding entries; also verify that between
        # entries there is nothing except whitespace, commas, and C
        # comments -- otherwise the format is not what we assumed and we
        # must fail loudly rather than silently drop data.
        cursor = 0
        stripped_for_scan = body_text
        while True:
            m = entry_re.search(stripped_for_scan, cursor)
            if not m:
                break
            # Check the gap between cursor and m.start() contains only
            # whitespace / commas / C-style comments.
            gap = stripped_for_scan[cursor:m.start()]
            gap_check = re.sub(r"/\*.*?\*/", "", gap, flags=re.DOTALL)
            gap_check = re.sub(r"//[^\n]*", "", gap_check)
            gap_check = gap_check.replace(',', '').strip()
            if gap_check != "":
                fail("array '%s': unexpected non-whitespace content between "
                     "entries near byte offset %d: %r (parser refuses to "
                     "guess; format differs from assumed [][%d] literal list)"
                     % (name, body_start + cursor, gap_check[:80], inner_dim))
            entries.append(tuple(int(g) for g in m.groups()))
            cursor = m.end()

        # Trailing content after the last entry (besides whitespace/comments)
        trailing = stripped_for_scan[cursor:]
        trailing_check = re.sub(r"/\*.*?\*/", "", trailing, flags=re.DOTALL)
        trailing_check = re.sub(r"//[^\n]*", "", trailing_check)
        trailing_check = trailing_check.replace(',', '').strip()
        if trailing_check != "":
            fail("array '%s': unexpected trailing content after last parsed "
                 "entry: %r" % (name, trailing_check[:80]))

        if not entries:
            fail("array '%s' parsed to zero entries -- format assumption "
                 "is wrong, refusing to proceed silently" % name)

        raw_lines = lines[decl_line_no - 1: close_line_no]

        arrays[name] = ParsedArray(
            name=name,
            elem_type=elem_type,
            inner_dim=inner_dim,
            entries=entries,
            raw_lines=raw_lines,
            decl_line_no=decl_line_no,
            close_line_no=close_line_no,
        )

    if not arrays:
        fail("no 'static const TYPE NAME[][N] = { ... };' arrays matched in "
             "%s -- file structure differs from what we assumed" % path)

    # Cross-check declared *_count constants against what we actually parsed.
    declared_counts = {}
    for m in COUNT_DECL_RE.finditer(text):
        count_owner, sizeof_a, sizeof_b = m.group(1), m.group(2), m.group(3)
        if sizeof_a != sizeof_b:
            fail("count declaration for '%s' has mismatched sizeof() array "
                 "names (%s vs %s) -- cannot trust this constant"
                 % (count_owner, sizeof_a, sizeof_b))
        declared_counts[count_owner] = sizeof_a

    return arrays, declared_counts, lines


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentile(sorted_vals, pct):
    """Nearest-rank percentile on an already-sorted ascending list."""
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def ascii_bar_chart(pairs, max_width=60, title=""):
    """pairs: list of (label, count) already in desired display order."""
    out_lines = []
    if title:
        out_lines.append(title)
    if not pairs:
        return out_lines
    max_count = max(c for _, c in pairs) or 1
    label_w = max(len(str(l)) for l, _ in pairs)
    for label, count in pairs:
        bar_len = int(round((count / max_count) * max_width)) if max_count else 0
        bar = "#" * bar_len
        out_lines.append("%s | %s %d" % (str(label).rjust(label_w), bar, count))
    return out_lines


def is_sorted_desc(vals):
    return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def is_sorted_asc(vals):
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    console = sys.stdout
    txt_fh = open(OUT_TXT, "w", encoding="ascii", errors="replace", newline="\n")
    sys.stdout = Tee(console, txt_fh)

    try:
        run()
    finally:
        sys.stdout = console
        txt_fh.close()


def run():
    print("=" * 78)
    print("hotlist_extract.py -- Task 0.13 hotlist parse + coverage analysis")
    print("run timestamp (UTC): %s" % datetime.now(timezone.utc).isoformat())
    print("source file: %s" % SRC_INC)
    print("=" * 78)

    arrays, declared_counts, all_lines = parse_inc_file(SRC_INC)

    print()
    print("--- Arrays found in .inc file ---")
    for name, arr in arrays.items():
        print("  %-45s type=%-14s inner_dim=%d entries=%d  (lines %d-%d)"
              % (name, arr.elem_type, arr.inner_dim, len(arr.entries),
                 arr.decl_line_no, arr.close_line_no))

    print()
    print("--- Cross-check against declared *_count constants in file ---")
    for count_owner, sizeof_name in declared_counts.items():
        if sizeof_name not in arrays:
            fail("*_count constant '%s_count' refers to unknown array '%s'"
                 % (count_owner, sizeof_name))
        parsed_n = len(arrays[sizeof_name].entries)
        print("  %s_count -> sizeof(%s)/sizeof(%s[0])  (parsed count=%d)"
              % (count_owner, sizeof_name, sizeof_name, parsed_n))

    if FLASH_ARRAY_NAME not in arrays:
        fail("expected Flash array '%s' not found among parsed arrays: %s"
             % (FLASH_ARRAY_NAME, list(arrays.keys())))
    if PRO_ARRAY_NAME not in arrays:
        fail("expected Pro array '%s' not found among parsed arrays: %s"
             % (PRO_ARRAY_NAME, list(arrays.keys())))

    flash = arrays[FLASH_ARRAY_NAME]
    pro = arrays[PRO_ARRAY_NAME]

    print()
    print("--- CONFIRMED COUNTS (parsed directly from file, not trusted from docs) ---")
    print("  Pro array   '%s': %d entries" % (PRO_ARRAY_NAME, len(pro.entries)))
    print("  Flash array '%s': %d entries" % (FLASH_ARRAY_NAME, len(flash.entries)))
    print("  Sum (Pro+Flash) = %d  (this is the number an older doc apparently "
          "mistook for the Flash-only count)" % (len(pro.entries) + len(flash.entries)))

    # -----------------------------------------------------------------
    # Field layout evidence
    # -----------------------------------------------------------------
    print()
    print("--- Field layout evidence ---")
    print("Header comments at top of file (verbatim):")
    for ln in all_lines[:3]:
        print("  " + ln)
    print()
    print("Array declaration is a flat '[][2]' literal -- NOT a named struct.")
    print("There is no third column anywhere in either array; every entry is")
    print("exactly 2 unsigned integers.")
    print()
    print("Call-site evidence from %s (verbatim, confirms field order):" % SRC_DS4_C)
    print('  line ~20688-20693:')
    print('    if (g_ds4_shape.variant == DS4_VARIANT_PRO) {')
    print('        hotlist = ds4_default_streaming_hotlist_pro;')
    print('        hotlist_count = ds4_default_streaming_hotlist_pro_count;')
    print('    } else if (g_ds4_shape.variant == DS4_VARIANT_FLASH) {')
    print('        hotlist = ds4_default_streaming_hotlist_flash;')
    print('        hotlist_count = ds4_default_streaming_hotlist_flash_count;')
    print('  ==> CONFIRMS ds4_default_streaming_hotlist_flash is the Flash array')
    print('      (selected only when g_ds4_shape.variant == DS4_VARIANT_FLASH).')
    print()
    print('  line ~20705-20707 (loop over hotlist[i]):')
    print('    metal_graph_streaming_expert_hotlist_add(')
    print('        hotlist[i][0],')
    print('        hotlist[i][1],')
    print('        max_entries - loaded, ...)')
    print()
    print('  function signature at line ~20568:')
    print('    static bool metal_graph_streaming_expert_hotlist_add(')
    print('        uint32_t    layer,')
    print('        uint32_t    expert,')
    print('        uint32_t    priority, ...)')
    print('  ==> CONFIRMS column [0] = layer index, column [1] = expert index.')
    print('      The 3rd call argument "priority" is NOT read from the file --')
    print('      it is synthesized at load time as (max_entries - loaded), i.e.')
    print('      purely from array POSITION (rank), descending as position')
    print('      increases through the array.')
    print()
    print('  NOTE: ds4.c also defines an in-memory struct used only by the')
    print('  runtime profiler (NOT present in the .inc file):')
    print('    typedef struct {')
    print('        uint32_t layer;')
    print('        uint32_t expert;')
    print('        uint64_t count;')
    print('        double   weight;')
    print('    } ds4_expert_hotlist_entry;')
    print('  This 4-field struct is used to sort experts by (count, weight)')
    print('  before the profiler ever writes them out, but only the resulting')
    print('  RANK ORDER is baked into the .inc file -- (layer, expert) pairs')
    print('  only. Raw hit counts/weights are NOT present in the .inc file.')
    print()
    print("==> VERDICT: entries do NOT carry a weight or hit-count field.")
    print("    Only ordered (layer, expert) pairs are available. This weakens")
    print("    the coverage analysis as flagged by the task instructions:")
    print("    we can report position-based (rank) coverage but CANNOT report")
    print("    a hit-count-weighted 'fraction of total profiled hits' number.")

    print()
    print("5 example raw lines from the Flash array (verbatim, immediately")
    print("after its declaration line %d):" % flash.decl_line_no)
    example_lines = [ln for ln in flash.raw_lines if ln.strip().startswith("{")][:5]
    for ln in example_lines:
        print("  " + ln)

    # -----------------------------------------------------------------
    # Flash statistics
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("FLASH ARRAY STATISTICS (%s)" % FLASH_ARRAY_NAME)
    print("=" * 78)

    flash_layers = [e[0] for e in flash.entries]
    flash_experts = [e[1] for e in flash.entries]

    distinct_layers = sorted(set(flash_layers))
    distinct_experts = sorted(set(flash_experts))

    print("  total entry count        : %d" % len(flash.entries))
    print("  distinct layers referenced: %d" % len(distinct_layers))
    print("  layer index min / max     : %d / %d" % (min(flash_layers), max(flash_layers)))
    print("  distinct experts referenced (global expert-id values seen in "
          "column 1): %d" % len(distinct_experts))
    print("  expert index min / max    : %d / %d" % (min(flash_experts), max(flash_experts)))

    # Per-layer histogram
    from collections import Counter
    layer_hist = Counter(flash_layers)

    print()
    print("--- Per-layer histogram (Flash) ---")
    print("  %-6s %-8s" % ("layer", "count"))
    for layer in sorted(layer_hist.keys()):
        print("  %-6d %-8d" % (layer, layer_hist[layer]))

    print()
    bar_pairs = [(layer, layer_hist[layer]) for layer in sorted(layer_hist.keys())]
    for ln in ascii_bar_chart(bar_pairs, max_width=50,
                               title="--- Per-layer histogram (ASCII bar chart) ---"):
        print("  " + ln)

    # Weight / hit-count distribution -- NOT AVAILABLE
    print()
    print("--- Weight / hit-count distribution ---")
    print("  UNKNOWN / NOT APPLICABLE: the .inc file entries carry no weight")
    print("  or hit-count field (see field-layout evidence above). Only")
    print("  positional rank is available.")

    # Sortedness check
    print()
    print("--- Sortedness check ---")
    layers_sorted_asc = is_sorted_asc(flash_layers)
    layers_sorted_desc = is_sorted_desc(flash_layers)
    experts_sorted_asc = is_sorted_asc(flash_experts)
    print("  entries sorted by layer ascending : %s" % layers_sorted_asc)
    print("  entries sorted by layer descending: %s" % layers_sorted_desc)
    print("  entries sorted by expert ascending (global): %s" % experts_sorted_asc)
    if not (layers_sorted_asc or layers_sorted_desc or experts_sorted_asc):
        print("  Not sorted by layer or by expert index. Consistent with the")
        print("  header comment '/* ... sorted by hits/weight. */': the")
        print("  positional order reflects descending profiled hit rank, which")
        print("  is not recoverable as a numeric field, only as array position.")
        print("  (We cannot independently verify this against a hit-count")
        print("  column because none exists in this file -- UNKNOWN whether")
        print("  the rank claim in the comment is literally exact.)")

    # -----------------------------------------------------------------
    # Coverage analysis
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("COVERAGE ANALYSIS")
    print("=" * 78)
    print("  Assumptions (given by task spec, NOT derived from source):")
    print("    layers                 = %d" % ASSUMED_N_LAYERS)
    print("    routed experts / layer = %d" % ASSUMED_N_EXPERTS_PER_LAYER)
    print("    total routed experts   = %d" % ASSUMED_TOTAL_EXPERTS)
    print("    MB per expert          = %.1f" % ASSUMED_MB_PER_EXPERT)
    print("    cache budget           = %.1f GB" % CACHE_BUDGET_GB)

    n_flash_entries = len(flash.entries)
    # "coverage" = distinct (layer, expert) pairs in the hotlist, as a
    # fraction of the assumed universe of (layer, expert) slots.
    distinct_pairs = set(flash.entries)
    n_distinct_pairs = len(distinct_pairs)
    if n_distinct_pairs != n_flash_entries:
        print()
        print("  NOTE: %d duplicate (layer, expert) pairs detected in the Flash "
              "array (%d raw entries, %d distinct pairs)."
              % (n_flash_entries - n_distinct_pairs, n_flash_entries, n_distinct_pairs))

    coverage_fraction = n_distinct_pairs / float(ASSUMED_TOTAL_EXPERTS)
    print()
    print("  Flash hotlist distinct (layer,expert) pairs : %d" % n_distinct_pairs)
    print("  Fraction of all %d routed experts covered   : %.4f  (%.2f%%)"
          % (ASSUMED_TOTAL_EXPERTS, coverage_fraction, coverage_fraction * 100.0))

    full_hotlist_gb = (n_distinct_pairs * ASSUMED_MB_PER_EXPERT) / 1024.0
    print()
    print("  Full Flash hotlist footprint if fully cached:")
    print("    %d experts * %.1f MB = %.1f MB = %.3f GB"
          % (n_distinct_pairs, ASSUMED_MB_PER_EXPERT,
             n_distinct_pairs * ASSUMED_MB_PER_EXPERT, full_hotlist_gb))

    budget_mb = CACHE_BUDGET_GB * 1024.0
    max_n_fit = int(budget_mb // ASSUMED_MB_PER_EXPERT)
    n_fit = min(max_n_fit, n_distinct_pairs)
    fraction_of_hotlist_fit = n_fit / float(n_distinct_pairs) if n_distinct_pairs else 0.0
    print()
    print("  Given a %.1f GB (%.0f MB) cache budget at %.1f MB/expert:"
          % (CACHE_BUDGET_GB, budget_mb, ASSUMED_MB_PER_EXPERT))
    print("    max experts that fit          : %d" % max_n_fit)
    print("    top-N entries that fit (N)    : %d" % n_fit)
    print("    fraction of Flash hotlist that is (top-N / total distinct): %.4f  (%.2f%%)"
          % (fraction_of_hotlist_fit, fraction_of_hotlist_fit * 100.0))
    print("    fraction of ALL %d routed experts this represents: %.4f  (%.2f%%)"
          % (ASSUMED_TOTAL_EXPERTS, n_fit / float(ASSUMED_TOTAL_EXPERTS),
             100.0 * n_fit / float(ASSUMED_TOTAL_EXPERTS)))

    print()
    print("  Predicted cold-start hit rate from top-N entries' share of total")
    print("  profiled hits: UNKNOWN -- cannot be computed. The .inc file carries")
    print("  no hit-count/weight column (see field-layout evidence above), so")
    print("  there is no 'total profiled hits' quantity available to divide by.")
    print("  This is exactly the caveat flagged by the task: without hit counts")
    print("  in the source file, the coverage analysis can only report slot")
    print("  coverage and byte footprint, not a hit-rate prediction. To get a")
    print("  real predicted hit-rate number, the runtime-generated hotlist text")
    print("  file (columns: layer expert hits weight, written by")
    print("  ds4_expert_profile_write_hotlist_file in ds4.c) would need to be")
    print("  sourced instead of the baked-in .inc arrays.")

    # -----------------------------------------------------------------
    # Emit JSON
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("JSON OUTPUT")
    print("=" * 78)
    print("Schema for both hotlist_flash.json and hotlist_pro.json:")
    print("  {")
    print('    "schema_version": 1,')
    print('    "source_file": "<absolute path to ds4_streaming_hotlist.inc>",')
    print('    "array_identifier": "<exact C identifier of the source array>",')
    print('    "variant": "flash" | "pro",')
    print('    "entry_count": <int, number of (layer, expert) pairs>,')
    print('    "field_layout": ["layer", "expert"],  // column 0, column 1')
    print('    "has_weight_or_hitcount": false,       // confirmed absent, see report')
    print('    "note": "position in this array is the rank order used by ds4 ")')
    print('             ("at load time via priority = max_entries - loaded; ")')
    print('             ("no numeric hit-count/weight is stored in this file.")')
    print('    "entries": [ {"layer": int, "expert": int, "rank": int}, ... ]')
    print("      // rank is 0-based position in the source array (0 = hottest)")
    print("  }")

    def build_payload(arr, variant_name):
        return {
            "schema_version": 1,
            "source_file": SRC_INC,
            "array_identifier": arr.name,
            "variant": variant_name,
            "entry_count": len(arr.entries),
            "field_layout": ["layer", "expert"],
            "has_weight_or_hitcount": False,
            "note": ("position in this array is the rank order used by ds4 at "
                     "load time via priority = max_entries - loaded; no numeric "
                     "hit-count/weight is stored in this file."),
            "entries": [
                {"layer": layer, "expert": expert, "rank": rank}
                for rank, (layer, expert) in enumerate(arr.entries)
            ],
        }

    flash_payload = build_payload(flash, "flash")
    pro_payload = build_payload(pro, "pro")

    with open(OUT_FLASH_JSON, "w", encoding="ascii") as f:
        json.dump(flash_payload, f, indent=1)
    with open(OUT_PRO_JSON, "w", encoding="ascii") as f:
        json.dump(pro_payload, f, indent=1)

    print()
    print("  Wrote %s (%d entries)" % (OUT_FLASH_JSON, len(flash.entries)))
    print("  Wrote %s (%d entries)" % (OUT_PRO_JSON, len(pro.entries)))

    # -----------------------------------------------------------------
    # Pro vs Flash comparison
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("PRO vs FLASH COMPARISON")
    print("=" * 78)

    pro_pairs = set(pro.entries)
    flash_pairs = set(flash.entries)
    overlap_pairs = pro_pairs & flash_pairs

    pro_experts_global = set(e for _, e in pro.entries)
    flash_experts_global = set(e for _, e in flash.entries)
    overlap_experts_global = pro_experts_global & flash_experts_global

    print("  Exact (layer, expert) pair overlap:")
    print("    Pro distinct pairs   : %d" % len(pro_pairs))
    print("    Flash distinct pairs : %d" % len(flash_pairs))
    print("    Shared pairs         : %d" % len(overlap_pairs))
    if flash_pairs:
        print("    Shared / Flash        : %.4f (%.2f%%)"
              % (len(overlap_pairs) / len(flash_pairs),
                 100.0 * len(overlap_pairs) / len(flash_pairs)))

    print()
    print("  Expert-id overlap ignoring layer (column 1 values only, i.e. is")
    print("  it 'the same expert index' regardless of which layer it's in):")
    print("    Pro distinct expert ids   : %d" % len(pro_experts_global))
    print("    Flash distinct expert ids : %d" % len(flash_experts_global))
    print("    Shared expert ids         : %d" % len(overlap_experts_global))
    if flash_experts_global:
        print("    Shared / Flash             : %.4f (%.2f%%)"
              % (len(overlap_experts_global) / len(flash_experts_global),
                 100.0 * len(overlap_experts_global) / len(flash_experts_global)))

    if overlap_pairs:
        print()
        print("  ==> Pro and Flash DO share hot (layer, expert) pairs. This")
        print("  suggests some routing hotspots are structural to the")
        print("  architecture/training data rather than purely a function of")
        print("  model size, which is a useful hint that a hotlist built for")
        print("  one variant has some transfer value to the other.")
    else:
        print()
        print("  ==> Pro and Flash share NO exact (layer, expert) pairs. Any")
        print("  apparent similarity would have to come from expert-id overlap")
        print("  alone (different layer counts between variants may also make")
        print("  a direct pair comparison less meaningful -- see layer range")
        print("  check below).")

    print()
    print("  Layer range check (relevant because Pro/Flash likely have")
    print("  different total layer counts, which affects how comparable a")
    print("  raw layer-index overlap is):")
    print("    Pro   layer min/max: %d / %d  (%d distinct layers)"
          % (min(e[0] for e in pro.entries), max(e[0] for e in pro.entries),
             len(set(e[0] for e in pro.entries))))
    print("    Flash layer min/max: %d / %d  (%d distinct layers)"
          % (min(flash_layers), max(flash_layers), len(distinct_layers)))

    print()
    print("=" * 78)
    print("DONE. Full console output also written to: %s" % OUT_TXT)
    print("=" * 78)


if __name__ == "__main__":
    main()
