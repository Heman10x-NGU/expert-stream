#!/usr/bin/env python3
"""
analyze_ngram.py -- O-8: can a free n-gram drafter supply the tokens that batching needs?

WHY THIS EXISTS
  O-6 settled the strategy question: batching N tokens beats hedging across N candidates by
  roughly a factor of N, because both read a union of the same size and only batching returns
  N tokens for it. Measured union over N real consecutive tokens, learned layers 3-42:

      N          1      2      3      4      5      6      7      8
      experts  6.00   9.29  11.62  14.22  15.94  18.32  19.61  21.49
      per tok  6.00   4.65   3.87   3.56   3.19   3.05   2.80   2.69

  So batching 8 deep would read 2.69 experts per token instead of 6 - a 55% cut in the one
  quantity the 0.503 tok/s ceiling responds to.

  There is one problem: autoregressive decoding does not hand you 8 tokens. You have to guess
  them. A neural draft model is the usual answer and is unavailable here - it would be a second
  set of weights on a machine with no RAM. An n-gram (suffix-match) drafter needs no weights,
  no training and no RAM: it looks for the current suffix earlier in the context and copies
  whatever followed it last time.

  This measures whether that is good enough to pay for itself.

THE ARITHMETIC IT IS SCORED AGAINST
  A batched round of size N reads union(N) experts and commits (1 + accepted) tokens, where
  accepted is the number of LEADING drafted tokens that match what the model would have
  produced. Greedy decoding, so "matches" is exact equality - no distribution to sample from.

      bytes per committed token = union(N) / (1 + accepted)

  It wins if that is below 6.00. Break-even acceptance is whatever makes it exactly 6.00.

WHAT THIS IS AND IS NOT
  It IS an exact-tokenizer measurement of how predictable a token stream is under suffix
  matching, per genre.
  It IS NOT a measurement of this model's output. Our own generations are a handful of
  one-word answers - at 0.5 tok/s we have never produced a long sample. Genre is reported
  separately precisely because n-gram drafting is known to live or die on repetitiveness, and
  averaging prose with source code would hide the only interesting thing here.

Usage:
  python analyze_ngram.py [file ...]      # defaults to a genre spread from this repo
"""

import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(REPO, "bench", "results")

TOKENIZE_EXE = r"E:\tools\llamacpp\llama-tokenize.exe"
SHARD1 = r"E:\models\ds4f-iq1s\UD-IQ1_S\DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf"

# Union over N real consecutive tokens, learned layers 3-42, from MEASURED_GROUND_TRUTH 18.2.
# Index by N. Layers 0-2 are excluded (frozen hash routing, 3 of 43 layers).
UNION = {1: 6.00, 2: 9.29, 3: 11.62, 4: 14.22, 5: 15.94, 6: 18.32, 7: 19.61, 8: 21.49}
BASE = UNION[1]

MAX_NGRAM = 8   # longest suffix we try to match
MIN_NGRAM = 2   # shortest suffix we will trust

OUT_TXT = os.path.join(RESULTS_DIR, "ngram_analysis.txt")
OUT_CSV = os.path.join(RESULTS_DIR, "ngram_acceptance.csv")

_log = None


def say(m=""):
    print(m)
    if _log is not None:
        _log.write(m + "\n")


def tokenize(path):
    """Exact token ids via the same prebuilt tokenizer hash_routing.py used. Vocab only."""
    proc = subprocess.run([TOKENIZE_EXE, "-m", SHARD1, "-f", path, "--ids"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("tokenizer failed on %s:\n%s" % (path, proc.stderr[-2000:]))
    txt = proc.stdout.strip()
    start = txt.find("[")
    end = txt.rfind("]")
    if start < 0 or end < 0:
        raise SystemExit("could not find an id list in tokenizer output for %s" % path)
    body = txt[start + 1:end]
    return [int(x) for x in body.replace("\n", " ").split(",") if x.strip() != ""]


def draft(tokens, i, n_draft, index):
    """n-gram drafter: longest suffix of tokens[:i+1] seen before, copy what followed.

    `index` maps a tuple-suffix to the position AFTER its most recent occurrence, built
    incrementally so this stays honest about only using the past."""
    for k in range(MAX_NGRAM, MIN_NGRAM - 1, -1):
        if i + 1 < k:
            continue
        key = tuple(tokens[i + 1 - k:i + 1])
        pos = index.get(key)
        if pos is None:
            continue
        out = tokens[pos:pos + n_draft]
        if out:
            return out
    return []


def simulate(tokens, n_batch):
    """Walk the sequence committing 1 + accepted tokens per round. Returns (rounds, committed,
    accepted_total, drafted_total)."""
    n_draft = n_batch - 1
    index = {}
    rounds_with_draft = [0]
    i = 0
    rounds = 0
    accepted_total = 0
    drafted_total = 0
    n = len(tokens)
    while i < n - 1:
        d = draft(tokens, i, n_draft, index) if n_draft > 0 else []
        drafted_total += len(d)
        if d:
            rounds_with_draft[0] += 1
        acc = 0
        for j, t in enumerate(d):
            if i + 1 + j < n and tokens[i + 1 + j] == t:
                acc += 1
            else:
                break
        accepted_total += acc
        step = 1 + acc
        # index everything we just committed, and only what we committed
        for p in range(i + 1, min(i + 1 + step, n)):
            for k in range(MIN_NGRAM, MAX_NGRAM + 1):
                if p >= k:
                    index[tuple(tokens[p - k:p])] = p
        i += step
        rounds += 1
    committed = i
    return rounds, committed, accepted_total, drafted_total, rounds_with_draft[0]


def default_corpus():
    """A deliberate genre spread. Each entry is (genre, path)."""
    c = []
    prose = os.path.join(REPO, "docs", "HONEST_ASSESSMENT.md")
    if os.path.isfile(prose):
        c.append(("prose-markdown", prose))
    code = os.path.join(HERE, "analyze_routing.py")
    if os.path.isfile(code):
        c.append(("source-python", code))
    ps = os.path.join(REPO, "bench", "run_first_output.ps1")
    if os.path.isfile(ps):
        c.append(("source-powershell", ps))
    csvf = os.path.join(RESULTS_DIR, "expert_manifest.csv")
    if os.path.isfile(csvf):
        c.append(("tabular-csv", csvf))
    return c


def main():
    global _log
    if not os.path.isfile(TOKENIZE_EXE):
        raise SystemExit("tokenizer not found: %s" % TOKENIZE_EXE)
    if not os.path.isfile(SHARD1):
        raise SystemExit("model shard 1 not found: %s" % SHARD1)

    args = sys.argv[1:]
    corpus = [("supplied", a) for a in args] if args else default_corpus()
    if not corpus:
        raise SystemExit("no corpus files found")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _log = open(OUT_TXT, "w", encoding="ascii", errors="replace", newline="\n")

    say("O-8 N-GRAM DRAFTER ACCEPTANCE")
    say("can a weightless drafter supply the tokens that batching needs?")
    say("")
    say("SCOPE, up front: this measures how predictable a TOKEN STREAM is under suffix")
    say("matching, using the model's exact tokenizer. It is not a measurement of this")
    say("model's own output - at 0.5 tok/s every generation we have is a few tokens long.")
    say("Genre is never averaged, because n-gram drafting lives or dies on repetitiveness.")
    say("")

    rows = []
    for genre, path in corpus:
        try:
            toks = tokenize(path)
        except SystemExit as e:
            say("SKIP %s: %s" % (path, e))
            continue
        if len(toks) < 200:
            say("SKIP %s: only %d tokens, too short to mean anything" % (path, len(toks)))
            continue

        say("=" * 78)
        say("%s  --  %s" % (genre, os.path.basename(path)))
        say("%d tokens" % len(toks))
        say("=" * 78)
        say("")
        say("  %-3s | %-9s | %-8s | %-9s | %-9s | %-10s | %s" %
            ("N", "accept/tok", "coverage", "tok/round", "need", "MiB/tok", "vs no batching"))
        say("  %s" % ("-" * 86))

        for n_batch in range(1, 9):
            rounds, committed, acc, drafted, rwd = simulate(toks, n_batch)
            if rounds == 0:
                continue
            tok_per_round = committed / float(rounds)
            acc_rate = (acc / float(drafted)) if drafted else 0.0
            union = UNION[n_batch]
            experts_per_tok = union / tok_per_round
            # 6.19 MiB per expert, 40 learned layers
            mib = experts_per_tok * 6.19 * 40
            base_mib = BASE * 6.19 * 40
            ratio = mib / base_mib
            verdict = "baseline" if n_batch == 1 else ("%.2fx  %s" % (ratio, "WIN" if ratio < 1.0 else "loses"))
            coverage = (rwd / float(rounds)) if rounds else 0.0
            need = union / BASE
            say("  %-3d | %9.3f | %8.3f | %9.3f | %9.3f | %10.1f | %s" %
                (n_batch, acc_rate, coverage, tok_per_round, need, mib, verdict))
            rows.append({
                "genre": genre, "file": os.path.basename(path), "tokens": len(toks),
                "n_batch": n_batch, "accept_per_drafted": round(acc_rate, 4),
                "draft_coverage": round(coverage, 4),
                "tokens_per_round": round(tok_per_round, 4),
                "tokens_per_round_needed": round(need, 4), "union": union,
                "mib_per_token": round(mib, 1), "ratio_vs_base": round(ratio, 4),
            })
        say("")

    if rows:
        import csv as _csv
        with open(OUT_CSV, "w", newline="", encoding="ascii", errors="replace") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        say("wrote %s" % OUT_CSV)

    say("")
    say("=" * 78)
    say("HOW TO READ THIS")
    say("=" * 78)
    say("  'coverage' is the fraction of rounds where the drafter could propose ANYTHING at all.")
    say("  It is the reason a genre can show 0.856 acceptance and still commit only 1.474 tokens")
    say("  per round: a suffix that has never been seen produces no draft, and that round commits")
    say("  exactly one token while still paying union(N) bytes.")
    say("")
    say("  'tok/round' is the whole game. A round always reads union(N) experts. If it commits")
    say("  1.0 tokens, batching cost you extra bytes for nothing. The break-even tok/round is")
    say("  union(N)/6.00 - so N=2 needs 1.55 tokens per round, N=8 needs 3.58.")
    say("")
    say("  A genre that clears it says nothing about a genre that does not. The spread IS the")
    say("  result: if only structured text clears the bar, then batching is a feature for")
    say("  code and data workloads and not for chat, and that is a design decision, not a bug.")
    say("")
    say("  This does NOT prove the engine would get the win. It proves whether the tokens are")
    say("  obtainable for free. The batched execution path does not exist yet.")
    say("")
    say("wrote %s" % OUT_TXT)
    _log.close()


if __name__ == "__main__":
    main()
