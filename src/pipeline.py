"""Pipeline orchestration — ties ingest → delta → report → index → chat under a single
traced request, and exposes a tiny in-process metrics registry the FastAPI app serves.

This is the seam observability hangs off: every `run_delta` / `answer` produces one Trace
persisted to runs/<request_id>.json with per-stage timing, LLM token/cost, and captured
failures. Metrics are aggregated in-memory for the /metrics endpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .canonical.model import CanonicalDoc, Delta
from .chat.answer import Answer, answer_question
from .chat.index import RetrievalIndex
from .chat.llm import LLMClient
from .config import settings
from .delta.engine import compute_delta
from .delta.report import write_report
from .ingest.registry import ingest
from .observability.tracing import Trace


# ---- in-memory metrics registry (served at /metrics) ---------------------------

@dataclass
class Metrics:
    requests: int = 0
    errors: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    total_wall_ms: float = 0.0
    delta_changes: int = 0
    chat_refusals: int = 0
    per_request: list[dict] = field(default_factory=list)

    def record(self, trace: Trace) -> None:
        t = trace.totals()
        self.requests += 1
        self.errors += 1 if trace.status == "error" else 0
        self.llm_calls += t["llm_calls"]
        self.input_tokens += t["input_tokens"]
        self.output_tokens += t["output_tokens"]
        self.cost_usd = round(self.cost_usd + t["cost_usd"], 6)
        self.total_wall_ms += t["wall_ms"]
        if "delta_counts" in trace.metrics:
            self.delta_changes += trace.metrics["delta_counts"].get("total", 0)
        self.per_request.append({"request_id": trace.request_id, "name": trace.name,
                                 "status": trace.status, **t})
        self.per_request[:] = self.per_request[-50:]  # keep last 50

    def snapshot(self) -> dict:
        avg = round(self.total_wall_ms / self.requests, 2) if self.requests else 0
        return {
            "requests": self.requests, "errors": self.errors,
            "llm_calls": self.llm_calls, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 6),
            "avg_wall_ms": avg, "delta_changes": self.delta_changes,
            "chat_refusals": self.chat_refusals,
            "recent": self.per_request[-10:],
        }


METRICS = Metrics()


# ---- pair resolution -----------------------------------------------------------

def pair_paths(pair: str) -> tuple[str, str, Path]:
    d = settings.samples_dir / pair
    a, b = d / "revA.pdf", d / "revB.pdf"
    # allow non-pdf revisions (dxf/scanned) by globbing revA.* / revB.*
    if not a.exists():
        a = next(iter(sorted(d.glob("revA.*"))), a)
    if not b.exists():
        b = next(iter(sorted(d.glob("revB.*"))), b)
    return str(a), str(b), d


# ---- operations ----------------------------------------------------------------

@dataclass
class DeltaRun:
    doc_a: CanonicalDoc
    doc_b: CanonicalDoc
    delta: Delta
    report_paths: dict
    trace: Trace


def run_delta(pair: str, write: bool = True) -> DeltaRun:
    pid_a, pid_b, out_dir = pair_paths(pair)
    trace = Trace(f"delta:{pair}", runs_dir=settings.runs_dir)
    try:
        a = ingest(pid_a, "A", trace=trace)
        b = ingest(pid_b, "B", trace=trace)
        delta = compute_delta(a, b, trace=trace)
        paths = {}
        if write:
            with trace.span("report"):
                paths = write_report(delta, out_dir / "out")
                paths = {k: str(v) for k, v in paths.items()}
    finally:
        trace.save()
        METRICS.record(trace)
    return DeltaRun(a, b, delta, paths, trace)


def build_index(run: DeltaRun) -> RetrievalIndex:
    return RetrievalIndex().build(run.doc_a, run.doc_b, run.delta)


def answer(pair: str, question: str, run: Optional[DeltaRun] = None,
           index: Optional[RetrievalIndex] = None,
           llm: Optional[LLMClient] = None) -> tuple[Answer, Trace]:
    if run is None:
        run = run_delta(pair, write=True)
    if index is None:
        index = build_index(run)
    llm = llm or LLMClient()
    trace = Trace(f"chat:{pair}", runs_dir=settings.runs_dir)
    try:
        with trace.span("index", chunks=len(index.chunks)):
            pass
        ans = answer_question(question, index, llm, trace=trace)
        if ans.refused:
            METRICS.chat_refusals += 1
    finally:
        trace.save()
        METRICS.record(trace)
    return ans, trace
