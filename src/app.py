"""FastAPI app — a minimal served surface for chat + observability.

  GET  /healthz              liveness
  POST /delta   {pair}       run ingest→delta→report, return the delta + report paths
  POST /chat    {pair, q}    grounded answer with citations
  GET  /metrics              inspectable metrics (latency, tokens, cost, delta counts, refusals)
  GET  /trace/{request_id}   the full per-request trace JSON

Run: python -m uvicorn src.app:api --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import pipeline
from .config import settings

api = FastAPI(title="Delta-Chat", version="0.1.0")
_STATIC = Path(__file__).parent / "static"


@api.get("/", response_class=HTMLResponse)
def home():
    """Minimal served UI: compute a delta and chat over it in the browser."""
    return (_STATIC / "index.html").read_text()


@api.get("/report/{pair}", response_class=HTMLResponse)
def report(pair: str):
    run, _ = _get(pair)
    p = Path(run.report_paths.get("html", ""))
    if not p.exists():
        raise HTTPException(404, "report not found")
    return p.read_text()
_INDICES: dict = {}  # pair -> (DeltaRun, RetrievalIndex) cache


class DeltaReq(BaseModel):
    pair: str = "pair1"


class ChatReq(BaseModel):
    pair: str = "pair1"
    q: str


def _get(pair: str):
    if pair not in _INDICES:
        try:
            run = pipeline.run_delta(pair, write=True)
        except FileNotFoundError as e:
            raise HTTPException(404, f"unknown pair '{pair}': {e}")
        _INDICES[pair] = (run, pipeline.build_index(run))
    return _INDICES[pair]


@api.get("/healthz")
def healthz():
    return {"ok": True}


@api.post("/delta")
def delta(req: DeltaReq):
    run, _ = _get(req.pair)
    return {"pair": req.pair, "summary": run.delta.summary,
            "report": run.report_paths,
            "changes": [c.model_dump() for c in run.delta.changes],
            "trace": run.trace.request_id}


@api.post("/chat")
def chat(req: ChatReq):
    run, index = _get(req.pair)
    ans, trace = pipeline.answer(req.pair, req.q, run=run, index=index)
    return {"question": ans.question, "answer": ans.text, "refused": ans.refused,
            "citations": ans.citations, "retrieved": ans.retrieved,
            "model": ans.model, "cost_usd": ans.cost_usd, "trace": trace.request_id}


@api.get("/metrics")
def metrics():
    return pipeline.METRICS.snapshot()


@api.get("/trace/{request_id}")
def trace(request_id: str):
    p = settings.runs_dir / f"{request_id}.json"
    if not p.exists():
        raise HTTPException(404, "trace not found")
    return json.loads(p.read_text())
