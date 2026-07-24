"""Grounded answering: retrieve → prompt the LLM with cited context → return an answer
that carries citations, or refuse when retrieval is too weak to support one.

Grounding contract:
  - The model is instructed to answer ONLY from the provided context blocks and to cite
    each claim with the block's citation id ([PID_A:..], [PID_B:..], [DELTA:..]).
  - If the top retrieval score is below the configured gate, we short-circuit and refuse
    rather than letting the model answer from imagination (scored by the rubric).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..config import settings
from ..observability.tracing import Trace
from .index import Chunk, RetrievalIndex
from .llm import LLMClient

SYSTEM = (
    "You are a precise assistant for engineering-drawing (P&ID) revision analysis. "
    "Answer ONLY from the numbered context blocks provided. Every factual statement MUST "
    "cite the source block id in square brackets, e.g. [DELTA:d1a2] or [PID_A:ab12]. "
    "If the context does not support an answer, say you cannot determine it from the "
    "available sources. Never invent tags, values, or changes. Be concise."
)


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str]
    retrieved: list[dict]
    refused: bool = False
    model: str = ""
    cost_usd: float = 0.0


def _format_context(hits: list[tuple[Chunk, float]]) -> str:
    lines = []
    for c, s in hits:
        loc = f"sheet {c.page + 1}"
        lines.append(f"[{c.cite}] ({c.source}, {loc}, score {s:.2f}) {c.text}")
    return "\n".join(lines)


def answer_question(question: str, index: RetrievalIndex, llm: LLMClient,
                    trace: Optional[Trace] = None) -> Answer:
    def _run() -> Answer:
        hits = index.search(question)
        retrieved = [{"cite": c.cite, "source": c.source, "score": round(s, 3),
                      "page": c.page} for c, s in hits]
        top = hits[0][1] if hits else 0.0
        if trace is not None:
            trace.set_metric("retrieval_top_score", round(top, 3))
            trace.set_metric("retrieval_hits", len(hits))

        # refusal gate: too weak to ground an answer
        if not hits or top < settings.retrieval_min_score:
            return Answer(question=question,
                          text="I can't determine that from the available sources "
                               "(no sufficiently relevant content was retrieved).",
                          citations=[], retrieved=retrieved, refused=True)

        context = _format_context(hits)
        user = (f"Context blocks:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer using only the context above and cite block ids in [brackets].")
        res = llm.complete(SYSTEM, user, purpose="grounded_answer", trace=trace)
        cites = sorted(set(re.findall(r"\[(PID_[AB]:[0-9a-f]+|DELTA:[0-9a-z]+)\]", res.text)))
        return Answer(question=question, text=res.text, citations=cites,
                      retrieved=retrieved, refused=False, model=res.model,
                      cost_usd=res.cost_usd)

    if trace is None:
        return _run()
    with trace.span("retrieve+answer", question=question[:80]) as sp:
        ans = _run()
        sp.attributes.update(refused=ans.refused, citations=len(ans.citations))
        return ans
