# Document Delta & Grounded Chat

Take two revisions of an engineering drawing (P&ID), compute the **meaningful structured
delta**, render a **delta report**, and **chat** over both revisions + the delta with
**cited, refusable** answers. Format-agnostic: **native PDF, scanned PDF (OCR), and
DWG/DXF** all plug in behind one adapter interface. Ships with **observability** (traces,
token/cost, structured logs, a metrics endpoint) and a **runnable eval harness** that
scores the delta (P/R/F1) and the chat (groundedness + a validated LLM judge).

> Domain note: a **PID** here is a *document identifier* (a handle to one revision's bytes
> + metadata), not the diagram type. The documents happen to be **P&IDs**.

**Bonuses — all covered:** ✅ delta markup overlay · ✅ all three formats end-to-end · ✅
served web UI/dashboard (`GET /`) · ✅ retrieval-quality evaluation (hit@k / MRR) · ✅
cost/latency budget analysis. See [Bonuses](#bonuses--all-covered).

---

## TL;DR — run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: make install  (pip install -e .)
cp .env.example .env                    # optional — add ANTHROPIC_API_KEY for real chat

make synth                              # (re)generate the 3 sample pairs + ground truth
make run   PAIR=pair1                   # ingest -> delta -> report (prints the change table)
make eval                               # scorecard: delta P/R/F1 + chat + judge validation
make chat  PAIR=pair1                   # interactive grounded chat (Ctrl-C to exit)
make markup PAIR=pair1                  # bonus: annotated PDF overlay of the delta
make serve                              # FastAPI: /chat /delta /metrics /trace/{id}
make test                               # unit tests
```

**No API key?** Everything runs offline with `DELTA_LLM_PROVIDER=echo` (the default falls
back gracefully): ingestion, the deterministic delta, the report, retrieval, observability,
and the *delta* half of the eval are all keyless. Only the *chat answer text* needs a real
LLM — with `echo` the pipeline still runs and the eval labels the mode so numbers aren't
over-trusted.

```bash
DELTA_LLM_PROVIDER=echo make eval        # fully offline scorecard
```

---

## What it does, in one picture

```
 two PIDs ──► FORMAT ADAPTERS ──► CANONICAL ──► DELTA ENGINE ──► REPORT ──► GROUNDED CHAT
 (rev A/B)    pdf_native          Element[]     align→classify   MD+JSON    retrieve+cite
              pdf_scanned (OCR)   (tag,bbox,     +confidence      +markup    +refuse
              dwg/dxf             attrs,conf)    (deterministic)             (LLM here only)
                    └──────────── one interface, N formats ────────────┘
        OBSERVABILITY: per-request trace (ingest→delta→retrieve→llm→answer) · token/cost · JSON logs · /metrics
        EVAL: delta P/R/F1 on labeled pairs · chat groundedness + validated judge · scorecard
```

---

## The core idea: identity-based alignment

P&IDs are **tag-based** — nearly every element carries a stable textual identity
(`10"-VF-43-9025-AS20S-00`, `PSV-9066A`, `26-KA-902`, `SP = 257 bar`). The whole system
pivots on one decision:

**The canonical representation is a flat list of typed `Element`s keyed by parsed tag
identity + geometry.** (`src/canonical/model.py`)

Alignment then matches by **identity, not pixel position** (`src/delta/align.py`):

- **Tagged types** (line, equipment, valve, instrument) match on a *stable identity key*
  that excludes mutable attributes — for a line that's `service-area-seq`, so a size change
  `10"→8"` reads as a **modification**, not delete+add. Exact key first, then fuzzy tag
  match (rapidfuzz) for near-identical tags.
- **Positional types** (setpoint, note, dimension, free text) are their own value, so they
  match by **spatial proximity** (centroid distance, greedy nearest-neighbour).

Classification (`src/delta/engine.py`) then emits `added / removed / modified` with a
subtype (`attribute` + field-level before/after diffs, `moved` + shift distance, `text`),
a location (page + bbox), and a **confidence** blending match strength with source
confidence (so an OCR-derived change is lower-confidence than a native-PDF one).

This is "smarter than a text/pixel diff": it survives moved and attribute-modified content,
parses the line-number grammar (`SIZE-SERVICE-AREA-SEQ-SPEC`) to detect *spec/size* changes,
and is **fully deterministic**.

### Where the LLM is — and isn't

The **structural delta has zero LLM** — it's deterministic and byte-reproducible (same
input → identical `delta.json`; there's a test for it). The LLM appears in exactly three
places, all isolated behind `src/chat/llm.py`:

1. **Grounded chat answering** — turning retrieved context into a cited answer.
2. *(optional)* natural-language change descriptions — off by default; the report uses
   deterministic templates so it stays reproducible.
3. **LLM-as-judge** in the eval (validated against gold labels).

This keeps the determinism requirement, and deliberately avoids "LLM where deterministic
logic is better."

---

## Format-agnostic ingestion (all three work end-to-end)

One interface (`src/ingest/base.py::FormatAdapter`); adapters register in
`src/ingest/registry.py` and are selected by sniffing bytes. All three emit the **same**
`Element` list — downstream is format-blind.

| Adapter | File | How | Status |
|---|---|---|---|
| Native PDF | `pdf_native.py` | pdfplumber words+geometry → classifier | ✅ real (2012 words/page on the samples) |
| Scanned PDF | `pdf_scanned.py` | rasterize → Tesseract OCR (px→pt) → same classifier; confidence < 1 | ✅ real |
| DWG / DXF | `dwg.py` | DWG→DXF via ODA File Converter (documented seam) → ezdxf TEXT/geometry → same classifier | ✅ real on DXF; DWG needs ODA |

The deterministic classifier (`src/ingest/classify.py`) encodes the P&ID conventions:
line-number grammar, ISA instrument bubbles (function code + 4-digit loop number, paired by
proximity), equipment/valve tags, and phrase-level setpoints/limits/elevations/notes via
token-proximity (layout-independent, so two near-identical pages classify identically —
essential for a trustworthy delta).

---

## Grounded chat

`src/chat/` — retrieval over **three** sources (PID A elements, PID B elements, delta
changes), each a chunk with a stable citation id (`PID_A:…`, `PID_B:…`, `DELTA:…`).

- **Hybrid retrieval** (`index.py`): BM25 over tokenized tags (the workhorse for short
  structured text) + a deterministic char-n-gram embedding for fuzzy prose, fused with
  Reciprocal Rank Fusion. Change-style queries ("what changed on sheet 3?") are routed
  toward delta chunks.
- **Refusal gate**: the returned score is a *raw relevance* (not rank-normalized), so an
  out-of-domain query ("who painted the Mona Lisa?") scores near zero and the system
  **refuses** instead of hallucinating.
- **Grounded answer** (`answer.py`): the model must answer only from the numbered context
  blocks and cite each claim with a block id; citations are validated against the index.

---

## Observability (homegrown, justified)

`src/observability/` — every `run`/`chat` produces one **trace** persisted to
`runs/<request_id>.json`, spanning `ingest → delta → retrieve → llm → answer` with
per-stage timing, **LLM token counts + estimated cost**, status, and **captured failures**
(errors are marked on the span and re-raised, never swallowed). Logs are **structured JSON**
with a correlation `request_id`. Aggregate metrics (latency, tokens, cost, delta counts,
refusals) are served at **`GET /metrics`**; the raw trace at **`GET /trace/{id}`**.

**Why homegrown over Langfuse/OTel/LangSmith:** the trace surface is small (a handful of
stages), the assignment explicitly accepts a well-designed homegrown tracer, and zero
dependencies keeps the repo runnable offline with no account setup. The span model is kept
OTel-shaped (name/timing/attributes/status/children) so an OTel exporter would be a small
adapter, not a rewrite.

---

## Evaluation harness

`make eval` (`eval/run_eval.py`) prints a scorecard and writes
`eval/results/<timestamp>.json` so runs are **comparable / regression-detecting**.

- **Delta P/R/F1** on labeled pairs. Ground truth is *free and exact*: Rev B is generated
  by applying **known** edits to Rev A (`src/synth*.py`), so the labels are the edits
  themselves — the numbers are honest, not hand-waved. Also reports **field-level accuracy**
  (did we get `size 10"→8"` right, not just "something changed here").
- **Chat**: groundedness (non-refused answer carries ≥1 *valid* citation), citation
  accuracy, refusal accuracy, and **answer correctness via LLM-as-judge**. The judge is
  **validated** against hand-labeled gold verdicts and the agreement is reported (so the
  judge isn't taken on faith). With `echo` the judge falls back to a lexical check and the
  scorecard labels the mode.
- **Candid failure table** printed at the end (see Limitations).

### Current scorecard (deterministic delta; `echo` provider)

Real run, `claude-opus-4-8` as answerer + judge:

```
 pair   TP FP FN  precision recall  F1   field_acc     format
 pair1   6  0  0    1.00     1.00  1.00    1.00        native PDF
 pair2   3  0  0    1.00     1.00  1.00    1.00        scanned (OCR)
 pair3   4  0  0    1.00     1.00  1.00    1.00        DWG/DXF
 Overall delta:    P=1.00 R=1.00 F1=1.00
 Judge validation: 100% agreement with gold (LLM judge)
 Chat: correctness=1.00  groundedness=1.00  citation_acc=1.00  refusal_acc=1.00
```

Under the offline `echo` stub every metric is identical except **answer correctness**
(the stub doesn't produce real prose) — that's expected and the scorecard labels the mode,
so it isn't over-trusted. A full `make eval` costs ≈ **$0.05–0.10** on Opus 4.8.

The same scorecard also reports the two bonus analyses (below).

---

## Bonuses — all covered

The assignment lists bonuses worth up to +8 (capped). All are implemented:

**1. Delta markup overlay** — `make markup PAIR=pair1` writes an annotated PDF with colored
redline boxes on every changed region (green added / red removed / amber modified) — the
manual artifact this tool replaces. (`src/markup/overlay.py`)

**2. All three formats end-to-end** — native PDF, scanned PDF (Tesseract OCR), and DWG/DXF
(ezdxf) each ingest through one adapter seam and score **P/R/F1 = 1.00** on their labeled
pairs (table above).

**3. Served web UI / dashboard** — `make serve` → open **http://localhost:8000**: pick a
pair, compute the delta table, chat with cited answers + refusal, and a live metrics footer
(requests, tokens, cost, latency). Plain FastAPI + one vanilla-JS page, no build step.
(`src/static/index.html`, routes in `src/app.py`)

**4. Retrieval-quality evaluation** — a labeled query→gold-chunk set scores whether the
answer-bearing source is retrieved, and how high (hit@k, MRR, mean rank) — the layer
grounded chat depends on, *upstream* of the LLM. Real run:

```
 Retrieval: hit@1=0.83  hit@3=1.00  hit@5=1.00  MRR=0.92  mean_rank=1.17
```

**5. Cost/latency budget analysis** — per-stage latency (p50/p95) and per-query cost against
a stated budget, with PASS/OVER verdicts. Real run (`claude-opus-4-8`):

```
 stage                p50 ms   p95 ms   budget   verdict
 ingest (per doc)     1549     2008     -        -
 delta                0.23     1.84     5000     PASS
 chat (retrieve+LLM)  2500     3364     8000     PASS
 Cost: $0.0061/query  (budget $0.02)             PASS
```

Both bonus analyses run as part of `make eval` and are written into the results JSON.

---

## Sample data & provenance (`data/samples/`)

The provided PDFs are two *different* drawings, not revisions — so pairs are **synthesized**
with recorded provenance (each pair has a `provenance.md`):

- **pair1** — native/native. Rev A = the provided *Lift Gas* P&ID; Rev B = Rev A with 6
  authored edits (size, spec, setpoint, valve remove, valve move, instrument add).
- **pair2** — scanned/scanned. A legible synthetic P&ID rasterized to noisy image-only PDFs
  and OCR'd (Tesseract). 3 edits.
- **pair3** — DWG/DXF. A synthetic P&ID authored with ezdxf. 4 edits.
- **stress** — the two provided (different) drawings as an unlabeled large-delta sanity
  check (228 changes, ~4s, doesn't explode).

---

## Design decisions & trade-offs

- **Deterministic structural delta, LLM only in chat/judge** — reproducibility + dodges the
  "LLM where deterministic logic is better" flag.
- **Identity-first alignment** — the crux; makes moved/attribute changes first-class.
- **Token-proximity phrase detection** over pdfplumber line-grouping — the latter was
  *inconsistent* between near-identical pages and injected phantom deltas; deterministic
  token proximity fixed it.
- **Homegrown tracer** — small surface, zero deps, offline-friendly, OTel-shaped.
- **Hash embedding by default** — zero-dependency and deterministic; it's *lexical*, not
  learned semantics (documented). Swap to a real embedding model via
  `DELTA_EMBEDDING_BACKEND`.

## What I deliberately cut

- **No pixel/raster diff** — the whole point is semantic/identity diff.
- **No multi-sheet-set orchestration** — samples are 1-page A3; the loop handles N pages
  but "500-sheet set" strategy lives in *Next*.
- **No vector-symbol geometry diff** — geometry counts are captured as page metadata;
  symbol-shape diffing was out of scope.
- **Lean UI** — the served dashboard is a single vanilla-JS page (no React/build step); it
  covers delta + chat + metrics, not a full multi-view admin console.

## Limitations (candid)

- **Dense-A3 OCR is poor.** Tesseract on the *original* dense A3 P&IDs (tiny text, ~2000
  tokens/page) recovers little and produces spurious churn — precision collapses. pair2
  uses a *legible* synthetic scan to demonstrate the adapter fairly. Production fix:
  high-DPI tiling or a vision model (Claude vision / PaddleOCR with layout).
- **Asymmetric native-vs-scan delta** has low precision — OCR text that doesn't match the
  native layer looks like add/remove churn. Symmetric formats (native/native, scan/scan)
  are the fair comparison.
- **Fuzzy tag matching** can mis-align very similar tags; surfaced via confidence + eval
  precision.
- **Chat correctness needs a real LLM** — `echo` is a plumbing stub.

## What I'd do next

- **Vision-based OCR** for dense P&IDs (Claude vision to read tags + bboxes) — the single
  biggest accuracy lever.
- **Connectivity-aware delta** — use the vector line graph to detect *topology* changes
  (a valve rerouted between two lines), not just per-tag changes.
- **500-sheet sets**: sheet-level fingerprinting to pair sheets across revisions first,
  then delta per sheet in parallel; index the delta report into a vector store for chat.
- **Cost/latency budget**: prompt-cache the shared context; batch the eval judge.
- **Retrieval eval**: labeled query→gold-chunk set to score retrieval hit-rate directly.

---

## Repo layout

```
src/
  ingest/    base.py registry.py pdf_native.py pdf_scanned.py dwg.py classify.py
  canonical/ model.py                 # the format-agnostic Element/Delta model
  delta/     align.py engine.py report.py
  chat/      index.py llm.py answer.py
  markup/    overlay.py               # bonus: annotated-PDF delta overlay
  observability/ tracing.py logging.py cost.py
  config.py cli.py app.py pipeline.py synth*.py
eval/  metrics.py run_eval.py datasets/ results/
data/samples/ pair1/ pair2/ pair3/ stress/ _source/
tests/
```

**Config** (`src/config.py`, env prefix `DELTA_`): model, OCR engine/DPI, alignment
thresholds, retrieval top-k and refusal gate, embedding backend — nothing load-bearing is
hardcoded. **No secrets in the repo**; credentials come from env (`.env.example` only).
