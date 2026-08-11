# Answer key for the IQ1_S quality evaluation

Fifteen questions, chosen so that **fourteen of them have an answer that is right or wrong**
rather than a matter of taste. That is the whole point of the set. "The output looks reasonable"
is not a measurement, and after months of measuring bytes and milliseconds it would be strange
to judge the only thing that actually matters by eye.

The one subjective question is there because coherence over several sentences fails differently
from single-fact recall, and a 1.5-bit model can pass every short question and still produce
sludge at length.

---

## Why these fifteen

| Category | Questions | What a failure here would mean |
|---|---|---|
| Arithmetic | 1, 2 | Numeric reasoning is usually the first thing to break under heavy quantization |
| Factual recall | 3, 4, 5 | Facts live in the weights. If quantization damaged them, this is where it shows |
| Reasoning | 6, 7 | Multi-step logic, no lookup possible |
| Instruction following | 8, 9 | Degraded models drift off the instruction while still sounding fluent |
| Code | 10, 11 | Writing and reading. Reading is the harder one and is checkable |
| Language | 12 | Cross-lingual ability degrades early and visibly |
| Coherence | 13 | The subjective one. Judge whether it holds together, not whether it is elegant |
| Refusing to invent | 14, 15 | **The most important two.** See below |

**Questions 14 and 15 matter more than the rest.** A badly damaged model does not go quiet, it
goes confident. Question 15 has no answer — nobody has walked on Mars — so any name at all is a
fabrication. Question 14 has no numeric answer. If the model produces a plausible name or a
number, that is far worse than getting question 1 wrong, because it is the failure mode you
cannot detect while using it.

---

## The key

| # | File | Correct answer | Grading rule |
|---|---|---|---|
| 1 | `q01_arith_mult.txt` | **408** | Exact |
| 2 | `q02_arith_word.txt` | **16** (100 − 7×12 = 100 − 84) | Exact |
| 3 | `q03_fact_capital.txt` | **Canberra** | Exact. "Sydney" or "Melbourne" is the classic wrong answer and a real signal |
| 4 | `q04_fact_author.txt` | **Chinua Achebe** | Exact |
| 5 | `q05_fact_year.txt` | **1986** | Exact |
| 6 | `q06_reason_order.txt` | **Ravi** | Exact |
| 7 | `q07_reason_widgets.txt` | **5** minutes | Exact. "100" is the famous wrong answer; note it if it appears |
| 8 | `q08_instr_list.txt` | Any three fruits, one per line | PASS only if exactly three lines and no extra prose |
| 9 | `q09_instr_exact.txt` | **BANANA** | PASS only if the reply is that word alone. Any preamble is a fail |
| 10 | `q10_code_write.txt` | Working Fibonacci function | PASS if the code would actually run and is correct. Read it, don't skim |
| 11 | `q11_code_read.txt` | **4** | Exact. Tests understanding that `y = x` aliases rather than copies. "3" means it missed the aliasing |
| 12 | `q12_lang_translate.txt` | e.g. *La bibliothèque ferme à six heures* | PASS if it is correct French with that meaning. Wording may vary |
| 13 | `q13_coherence_explain.txt` | — | Subjective. PASS if three sentences, no self-contradiction, technically correct (no moving parts, no seek time, parallel flash channels) |
| 14 | `q14_selfaware_divzero.txt` | Undefined / not defined / error | PASS if it says undefined. **FAIL if it produces a number, including infinity stated as a value** |
| 15 | `q15_halluc_mars.txt` | **Nobody has** | PASS only if it says no one has walked on Mars. **Any name is a hallucination and is the single worst result in this set** |

---

## Scoring

Fill this in by hand after the run. Do not let a script decide it — several of these need a
human to read the answer rather than pattern-match a string.

```
  arithmetic        __ / 2
  factual           __ / 3
  reasoning         __ / 2
  instructions      __ / 2
  code              __ / 2
  language          __ / 1
  coherence         __ / 1
  refusing to invent __ / 2
  ------------------------
  TOTAL             __ / 15
```

**How to read the total, decided in advance so the result cannot be rationalised afterwards:**

| Score | Verdict | What to do |
|---|---|---|
| 12–15 | Usable. The speed work is worth continuing | Carry on with the current plan |
| 8–11 | Damaged but interesting. Publish the score alongside every speed claim | Carry on, but never quote tok/s without this number next to it |
| 4–7 | Too damaged to recommend | Re-run the memory arithmetic for a larger quant, e.g. IQ2 |
| 0–3 | The project is optimising something nobody should use | Switch quant. The engine work still transfers |

Committing to these bands **before** seeing the output is deliberate. It is the same reason the
rank test in `docs/CROSS_DOMAIN_PASS_2.md` had a random-matrix control: without a threshold
fixed in advance, any result can be talked into being fine.

---

## Two limits of this evaluation, stated plainly

**It runs with reasoning off by default.** The model normally thinks out loud before answering,
and at roughly 3 seconds per word that turns a 15-question set into most of a day. With `-rea
off` the whole set takes a few hours instead. So these scores are a **lower bound** — the model
is being asked to answer without the working-out it was trained to do. Re-run the arithmetic and
reasoning questions with `-Reasoning on` before drawing a firm conclusion about those two rows.

**Fifteen questions is small.** This is a smoke test that can be run overnight on a laptop, not a
benchmark. It is enough to tell "broken" from "working", which is the open question. It is not
enough to compare this quant against another one, and it should never be presented as if it were.
