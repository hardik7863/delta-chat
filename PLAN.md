# Document Delta & Grounded Chat — Build Plan

Applied AI Engineer take-home. Two PID revisions → structured delta → delta report → grounded chat, format-agnostic (native PDF + scanned PDF + DWG), with required observability and eval harness.

## 0. Strategy (map effort to the rubric)

Internal rubric weights (leaked in the assignment HTML — treat as ground truth):

| Dimension | Weight | Where we spend |
|---|---|---|
| **Delta quality** | 20% | Identity-based alignment + attribute-diff on parsed P&ID tags. The whole thing. |
| **Evaluation rigor** | 20% | Synthetic edits = exact ground truth → trustworthy P/R/F1 + candid failure table. |
| Pipeline design | 15% | Canonical `Element` seam; all 3 adapters emit the same shape. |
| Grounded chat | 15% | Hybrid retrieval over A + B + report; cite element/delta ids; refuse when unsupported. |
| Observability | 15% | Homegrown JSON tracer, per-request trace file, LLM token/cost telemetry. |
| Eng & docs | 10% | One-command run, README trade-offs, no secrets. |
| Communication | 5% | Explicit cuts + crisp demo. |

**Delta + Eval = 40%.** Everything else protects those.

Two decisions that directly score points:
- **Structural delta is 100% deterministic.** LLM appears ONLY in: chat answering, natural-language change *descriptions*, and LLM-as-judge in eval. Kills the auto-flag "LLM used where deterministic logic was clearly better." Satisfies the determinism requirement; the report is byte-reproducible.
- **Ground truth is free.** We generate Rev B by applying *known* edits to Rev A, so the true delta is recorded as we make it. Eval precision/recall is honest, not hand-waved.

## 1. The crux: canonical representation

P&IDs are tag-based. Every meaningful element has a stable textual identity. The canonical model is a flat list of typed elements with geometry:

```python
class Attr(BaseModel):        # parsed structured attributes (line-number grammar etc.)
    size: str | None          # "10\"", "12mm"
    service: str | None       # "VF", "AI", "PV"
    seq: str | None           # "9025"
    spec: str | None          # "AS20S", "AC21S"
    setpoint: str | None      # "257 bar (g)"
    raw: dict                  # anything else parsed

class Element(BaseModel):
    id: str                   # stable within a doc: type + tag + rounded-bbox hash
    type: Literal["line","instrument","valve","equipment","note","setpoint",
                  "dimension","table_cell","text","symbol"]
    tag: str | None           # identity token, e.g. "10\"-VF-43-9025-AS20S-00", "PSV-9066A"
    text: str                 # raw text
    bbox: tuple[float,float,float,float]   # x0,y0,x1,y1 in page pts
    page: int
    attrs: Attr
    confidence: float         # 1.0 native PDF, <1 for OCR
    source: Literal["pdf_native","pdf_scanned","dwg"]

class CanonicalDoc(BaseModel):
    pid: str
    format: str
    revision_label: str | None
    sheet_count: int
    page_sizes: list[tuple[float,float]]
    elements: list[Element]
    meta: dict
```

The delta engine and chat layer consume `CanonicalDoc` and never know the source format. That is the seam the rubric rewards.

**Element typing** (deterministic classifier over the extracted tokens):
- Line number: regex `^\d+("|mm)-[A-Z]{2,3}-\d+-\d+-[A-Z0-9]+-\d+$` → `type=line`, parse `Attr`.
- Instrument bubble: 2–4 letter function code + tag number (`PSV 9066A`, `PDIT 9054`) → `type=instrument`.
- Set point: `SP =`, `HH :`, `LL :`, value+unit → `type=setpoint`.
- Note ref: `NOTE 34` → `type=note`.
- Valve tag: `\d+BL\d+`, `GT`, `CSO`, `LO` patterns → `type=valve`.
- Equipment tag: `\d{2}-[A-Z]{2}-\d{3}` (`26-KA-902`) → `type=equipment`.
- Fallback → `type=text`.

## 2. Format adapters (one interface, N formats)

```python
class FormatAdapter(Protocol):
    def sniff(self, raw: bytes, hint: str|None) -> bool: ...
    def to_canonical(self, pid: str, raw: bytes) -> CanonicalDoc: ...
```

- **pdf_native.py** — `pdfplumber`: `extract_words()` (bbox per token) + `lines`/`curves` for geometry. Confirmed: Lift Gas = 2012 words / 4304 curves, Export = 1315 / 3367. Real geometry present. This adapter is basically done by the extraction quality we already verified.
- **pdf_scanned.py** — rasterize page (pdf2image / pdfium at 200–300 dpi) → OCR. Default **PaddleOCR** (better on rotated dense CAD text; boxes + confidence); Tesseract fallback via config. Emits same `Element` list with `confidence<1`, `source="pdf_scanned"`. No vector layer → geometry elements dropped (documented degradation).
- **dwg.py** — real seam, honest converter: accept `.dwg`, convert **DWG→DXF via ODA File Converter** (documented external dep; if absent, ship `.dxf` samples and skip conversion), parse DXF with **`ezdxf`**: `TEXT`/`MTEXT` → text/tag elements with insert coords, `LINE`/`LWPOLYLINE` → geometry, `INSERT` blocks → symbols. Emits same `Element` shape. This makes "all three formats end-to-end" (+2 bonus) genuinely real, not hypothetical.

Adapter registry picks by magic bytes + extension hint.

## 3. Delta engine (deterministic — the 20%)

`align.py` → `engine.py` → `report.py`.

**Alignment (the hard part they explicitly grade):**
1. **Identity match**: bucket elements by `type`; within a type, match by `tag` exact, then `rapidfuzz` fuzzy on tag/text (≥ threshold, configurable). Identity match survives repositioning → detects *moved* not *deleted+added*.
2. **Spatial match** for untagged elements (notes, free text): Hungarian assignment on bbox IoU + centroid distance, gated by text similarity.
3. Anything unmatched in A → **removed**; unmatched in B → **added**.

**Classification per matched pair:**
- bbox moved > threshold → `modified: moved` (Δ position recorded).
- `Attr` differs (size/spec/setpoint) → `modified: attribute` with field-level diff (`size 10"→8"`). This is the standout demo.
- text differs but tag same → `modified: text`.
- else unchanged (dropped from report).

**Confidence** = f(match score, source confidence of both elements, attribute-parse certainty). Every change carries `{type, subtype, location{page,bbox}, description, confidence, evidence{a_id,b_id}}`.

```python
class Change(BaseModel):
    id: str
    op: Literal["added","removed","modified"]
    subtype: str | None            # moved | attribute | text
    element_type: str
    tag: str | None
    page: int
    bbox: tuple
    field_diffs: list[dict]        # [{field:"size", before:"10\"", after:"8\""}]
    description: str               # deterministic template; LLM polish optional
    confidence: float
    evidence: dict                 # {a_id, b_id}

class Delta(BaseModel):
    pid_a: str; pid_b: str
    changes: list[Change]
    summary: dict                  # counts by op/type
```

## 4. Delta report (`report.py`)

Emits **both**: `delta.json` (machine, canonical) and `delta.md`/`.html` (human). Report groups by sheet then by element type, leads with a counts table, lists each change with location + confidence + evidence ids. The report is itself indexed as a retrievable source for chat (each `Change` = one retrievable chunk with a stable citation id `DELTA:<id>`).

## 5. Grounded chat (`chat/`)

- **index.py** — three sources into one hybrid index: PID A elements, PID B elements, delta changes. **Hybrid retrieval**: BM25 (rank_bm25) over short structured tags + embeddings (Claude/`voyage` or local `bge-small`) over note prose; reciprocal-rank fusion. Metadata filters for `page`, `source`, `op` → routes "what changed on sheet 3" to delta chunks, "did any dimension change near the pump" to spatial+attribute delta chunks near equipment tagged pump.
- **llm.py** — provider-agnostic client, Claude default (`claude-opus-4-8` for hard/judge, cheaper Claude for routine), creds from env, token/cost captured.
- **answer.py** — retrieval-augmented answer with **mandatory citations** to `PID_A:<elem_id>` / `PID_B:<elem_id>` / `DELTA:<change_id>`; **refuses/hedges** when top retrieval score < gate (scored by rubric: "refuses when unsupported instead of hallucinating").

## 6. Observability (`observability/`) — homegrown, justified

Homegrown tracer (no heavy dep; shows we understand the mechanism — README justifies vs Langfuse/OTel). Optional OTel exporter stub.
- `Trace` with nested `Span`s: ingest_a, ingest_b, delta, retrieve, llm, answer — each with start/end/duration, status, error.
- LLM wrapper records prompt, response, model, prompt/completion tokens, **estimated cost** (per-model price table).
- Structured JSON logs, every line carries `request_id`.
- One `runs/<request_id>.json` trace file per request + a served `/metrics` endpoint (latency, tokens, cost, delta counts, retrieval hits) via the FastAPI app.
- Failures (bad OCR, unparseable DWG, LLM timeout) captured on the span, never swallowed.

## 7. Eval harness (`eval/`) — the other 20%

- **datasets/**: 3 pairs (see §8) with `expected_delta.json` (exact, from synthetic edits) + `qa.jsonl` (hand-labeled Q&A, ~10 questions).
- **Delta metrics**: precision / recall / F1, matching predicted↔expected changes by (tag, op, field). Reports per-op breakdown.
- **Chat metrics**: correctness + **groundedness/citation accuracy**. LLM-as-judge (Claude) **with a validated judge** — small human-agreement check reported so the judge isn't taken on faith.
- `make eval` prints a **scorecard**; results written to `eval/results/<timestamp>.json` so runs are **comparable / regression-detecting**.
- **Candid failure table** in README (rubric explicitly rewards honesty).

## 8. Sample data & provenance (`data/samples/`)

Given files are two *different* drawings, not revisions — so we synthesize revision pairs and document provenance:

1. **Pair 1 — native/native, small known delta.** Rev A = Lift Gas P&ID. Rev B = programmatic edit of A (change one set point `257→275 bar`, delete one `NOTE`, swap a line size `10"→8"`, move one valve). Edits applied to the source and re-exported; the edit script *is* the ground truth. Provenance = script + seed.
2. **Pair 2 — native vs scanned (cross-format).** Rev A = native Rev B above; "Rev C" = render to 250 dpi PNG + mild noise/skew, wrap as image PDF → forces the OCR adapter and cross-format delta. Tests degradation honesty.
3. **Pair 3 — DWG/DXF.** Synthesize a small P&ID in DXF via `ezdxf` (TEXT tags + LINE pipes + INSERT symbols), make Rev A/B with known edits → exercises the DWG adapter end-to-end.
4. (Stress) The two real different drawings as a "large delta" sanity pair — not scored, shows it doesn't explode.

No keys/PII committed. `.env.example` only.

## 9. Repo scaffold

```
delta-chat/
├─ README.md  .env.example  Makefile  docker-compose.yml  pyproject.toml
├─ src/
│  ├─ ingest/{base,pdf_native,pdf_scanned,dwg,registry}.py
│  ├─ canonical/model.py
│  ├─ delta/{align,engine,report}.py
│  ├─ chat/{index,llm,answer}.py
│  ├─ markup/overlay.py            # bonus
│  ├─ observability/{tracing,logging,cost}.py
│  ├─ config.py                    # pydantic-settings; thresholds/model/paths
│  └─ cli.py / app.py              # Typer CLI + FastAPI
├─ eval/{datasets,metrics.py,run_eval.py,results}/
├─ data/samples/  tests/
```
Commands: `make run PAIR=pair1` · `make chat` · `make eval` · `make markup`.

## 10. Milestones & time budget (~1.5 days to Sat 23:59)

1. **Scaffold + config + canonical model + tracer skeleton** (~1h)
2. **pdf_native adapter + element classifier + line-grammar parser** (~2h) ← unblocks everything
3. **Synthetic edit tool + Pair 1** (~1.5h) ← ground truth
4. **Delta engine: align + classify + confidence** (~3h) ← the 20%, protect this
5. **Delta report MD+JSON** (~1h)
6. **Chat: index + retrieval + Claude client + grounded answer w/ citations** (~3h)
7. **Observability wiring: traces/logs/cost/metrics endpoint** (~1.5h)
8. **Eval harness: delta P/R/F1 + Q&A judge + scorecard** (~2.5h) ← the other 20%
9. **README + DEMO + walkthrough recording** (~1.5h)
10. Bonus in priority order if time remains: **markup overlay** → **scanned adapter** → **DWG/DXF adapter**.

**Cut-first if behind:** DWG (keep honest stub) → scanned → markup. Core A–D + obs + eval never get cut. Every cut written down in README §"what we cut".

## 11. Risks & mitigations

- **All-three-formats is ambitious for the window.** → Core-first ordering; DWG/scanned are steps 10, explicitly droppable to stub. Seam stays real either way.
- **OCR box quality on dense CAD text.** → 250–300 dpi, PaddleOCR, report degraded confidence honestly; scanned pair is small.
- **Alignment false-positives (fuzzy tag matches).** → conservative thresholds + confidence surfaced; eval catches over/under-matching via P/R.
- **LLM judge trust.** → validate judge against a few human labels, report agreement.
- **Determinism.** → structural delta has zero LLM; only descriptions optionally polished (flagged, and default-off for reproducible report).

## 12. What we deliberately cut (initial)

- No pixel-level raster diff (we do semantic/identity diff — the whole point).
- No multi-sheet set handling beyond N pages loop (samples are 1-page A3); "500-sheet" answer lives in README §what's-next.
- No fine-tuned models; off-the-shelf OCR + Claude.
- Minimal UI: CLI + FastAPI `/chat` + `/metrics`; a served dashboard only if bonus time remains.
