# DEMO — one delta, one grounded chat, one scorecard

A 2–4 minute walkthrough you can reproduce with the commands shown. Everything below runs
**offline** (`DELTA_LLM_PROVIDER=echo`); set `ANTHROPIC_API_KEY` for real chat answer text.

```bash
source .venv/bin/activate
make synth          # build sample pairs + ground truth (once)
```

## 1. Compute a delta  — `make run PAIR=pair1`

Rev A = provided *Lift Gas* P&ID; Rev B = Rev A with 6 authored edits. The engine recovers
**exactly** those 6 changes — typed, located, with field-level diffs and confidence:

```
Delta pair1 — 6 changes (+1 / -1 / ~4)
 op        subtype    type        tag                       change              conf
 added                instrument  PT-9099                                       1.00
 modified  attribute  line        6"-VF-43-9029-GC11S-00     spec AC21S→GC11S    1.00
 modified  attribute  line        8"-VF-43-9025-AS20S-00     size 10"→8"         1.00
 modified  attribute  setpoint    HH:300                     HH=245→HH=300       0.99
 modified  moved      valve       26BL9073                   moved 33.9pt        1.00
 removed              valve       43BL9070                                       1.00

report: data/samples/pair1/out/delta.md   (+ delta.json, delta.html)
trace:  runs/<request_id>.json
```

Note it detects a **moved** valve (same tag, shifted) and a **size/spec** change (parsed
from the line-number grammar) — not just add/remove. The report is human-readable
(`delta.md`) and machine-readable (`delta.json`), and every change has a citable id
(`DELTA:…`).

## 2. Grounded chat  — `make chat PAIR=pair1`

Real output, `claude-opus-4-8` (`DELTA_LLM_PROVIDER=anthropic`):

```
you› Did any line size change, and to what?
Yes. Line 8"-VF-43-9025-AS20S-00 changed size from 10" to 8" [DELTA:da9f45f84c9].
citations: DELTA:da9f45f84c9
retrieved: DELTA:da9f45f84c9(0.841), PID_A:9a657e556e8f(0.549), PID_B:540428b6398b(0.535)

you› Which valve was removed, and did anything move?
Valve 43BL9070 was removed [DELTA:d3f9fbe84b2]. Yes, valve 26BL9073 was moved [DELTA:db91709911a].
citations: DELTA:d3f9fbe84b2, DELTA:db91709911a
```

Answers **cite the exact delta entries**, and retrieval routes "changed?" queries to delta
chunks. Out-of-domain questions are **refused** (short-circuited before the LLM by the
retrieval gate), not hallucinated:

```
you› What is the capital of France?
REFUSED  I can't determine that from the available sources (no sufficiently relevant content was retrieved).
retrieved: PID_B:c7cdde07e26a(0.087), PID_B:1aea02492959(0.079), PID_B:739cd04da0e4(0.078)
```

Cost: a grounded answer is ~500 in / ~150 out tokens ≈ **$0.006** on Opus 4.8; the tracer
logs exact token counts + cost per call. (No key? `DELTA_LLM_PROVIDER=echo` runs the whole
pipeline offline; only the answer *prose* needs a real model.)

## 3. Eval scorecard  — `make eval`

Ground truth is exact (the authored edits), so P/R/F1 is trustworthy. All three formats —
**native PDF, scanned/OCR, DWG/DXF** — score perfectly on their labeled pairs. Real run
with `claude-opus-4-8` as the answerer + judge:

```
 pair   TP FP FN  precision recall  F1   field_acc     (format)
 pair1   6  0  0    1.00     1.00  1.00    1.00        native PDF
 pair2   3  0  0    1.00     1.00  1.00    1.00        scanned (OCR, Tesseract)
 pair3   4  0  0    1.00     1.00  1.00    1.00        DWG/DXF (ezdxf)
 Overall delta:    P=1.00  R=1.00  F1=1.00
 Judge validation: 100% agreement with gold (LLM judge)
 Chat: correctness=1.00  groundedness=1.00  citation_acc=1.00  refusal_acc=1.00
```

(Under the offline `echo` stub the delta/groundedness/refusal metrics are identical;
only *answer correctness* drops — the stub doesn't actually answer — and the scorecard
labels the mode so it isn't over-trusted.) A full `make eval` ≈ **$0.05–0.10** on Opus 4.8.
See README → Limitations for the candid failure list (dense-A3 OCR, asymmetric native-vs-scan).

## 4. Bonus — delta markup overlay  — `make markup PAIR=pair1`

Writes `data/samples/pair1/out/delta_markup.pdf`: the revised drawing with colored boxes +
redline labels on every changed region (green = added, red = removed, amber = modified) —
the manual artifact this tool replaces.

## 5. Observability  — `make serve`, then

```bash
curl -s localhost:8000/metrics            # latency, tokens, cost, delta counts, refusals
curl -s localhost:8000/trace/<id>         # full per-request trace: ingest→delta→retrieve→llm→answer
```

Every request also drops a structured JSON trace in `runs/<request_id>.json` with per-stage
timing and LLM token/cost — open one to see exactly what the system did.
```
spans: ['ingest','ingest','delta','report']  totals: {input_tokens, output_tokens, cost_usd, llm_calls, wall_ms}
```
