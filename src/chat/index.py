"""Retrieval index over the three grounded sources: PID A elements, PID B elements, and
the delta report entries. Every retrievable chunk carries a stable citation id so answers
can point back to an exact source.

Hybrid retrieval:
  - BM25 over tokenized text (the workhorse for short, structured tags like line numbers).
  - A deterministic lexical embedding (character n-gram hashing) for fuzzy/semantic-ish
    matching on note prose. `embedding_backend="hash"` keeps the repo zero-dependency and
    fully offline; swap to a real embedding model via config for stronger semantics.
  - Scores fused with Reciprocal Rank Fusion (RRF), so neither signal dominates.

Chat queries that mention change/added/removed/etc. get delta chunks boosted, routing
"what changed?" questions to the delta report and content questions to the PID chunks.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from ..canonical.model import CanonicalDoc, Delta
from ..config import settings

_CHANGE_WORDS = re.compile(r"\b(chang|delta|diff|add|remov|delet|modif|mov|revis|"
                           r"new|different|between|rev\b)", re.I)


@dataclass
class Chunk:
    cite: str            # citation id, e.g. PID_A:ab12 / DELTA:d34
    source: str          # pid_a | pid_b | delta
    text: str            # retrievable/answerable text
    page: int
    bbox: tuple
    meta: dict = field(default_factory=dict)


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic char-3gram hashing embedding. Captures lexical overlap (incl.
    substrings/typos), not learned semantics — documented as such in the README."""
    v = np.zeros(dim, dtype=np.float32)
    s = f"  {text.lower()}  "
    for i in range(len(s) - 2):
        g = s[i:i + 3]
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _elem_text(e) -> str:
    parts = [e.type, e.tag or "", e.text or ""]
    a = e.attrs
    for f in ("size", "service", "spec", "setpoint", "value"):
        if getattr(a, f):
            parts.append(f"{f}={getattr(a, f)}")
    return " ".join(p for p in parts if p)


class RetrievalIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._emb: Optional[np.ndarray] = None

    def build(self, a: CanonicalDoc, b: CanonicalDoc, delta: Delta) -> "RetrievalIndex":
        chunks: list[Chunk] = []
        for e in a.elements:
            chunks.append(Chunk(f"PID_A:{e.id}", "pid_a", _elem_text(e), e.page, e.bbox,
                                {"type": e.type, "tag": e.tag}))
        for e in b.elements:
            chunks.append(Chunk(f"PID_B:{e.id}", "pid_b", _elem_text(e), e.page, e.bbox,
                                {"type": e.type, "tag": e.tag}))
        for c in delta.changes:
            txt = c.description + " " + " ".join(
                f"{d['field']} {d['before']}->{d['after']}" for d in c.field_diffs)
            chunks.append(Chunk(f"DELTA:{c.id}", "delta", txt.strip(), c.page, c.bbox,
                                {"op": c.op, "type": c.element_type, "tag": c.tag,
                                 "confidence": c.confidence}))
        self.chunks = chunks
        corpus = [_tok(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None
        self._emb = np.vstack([_hash_embed(c.text) for c in chunks]) if chunks else None
        return self

    def search(self, query: str, top_k: Optional[int] = None) -> list[tuple[Chunk, float]]:
        """Return top_k (chunk, relevance) pairs. Ordering uses RRF (robust rank fusion);
        the returned score is a RAW relevance in 0..1 (blend of normalized BM25 and
        embedding cosine) so the refusal gate reflects actual match strength, not just
        rank position — an off-domain query scores near zero and gets refused."""
        top_k = top_k or settings.retrieval_top_k
        if not self.chunks:
            return []
        n = len(self.chunks)
        bm = self._bm25.get_scores(_tok(query)) if self._bm25 else np.zeros(n)
        q = _hash_embed(query)
        emb = self._emb @ q if self._emb is not None else np.zeros(n)

        bm_n = bm / bm.max() if bm.max() > 0 else bm            # 0..1 keyword signal
        rel = 0.6 * bm_n + 0.4 * np.clip(emb, 0, 1)             # raw relevance 0..1

        def rrf(scores):
            order = np.argsort(-scores)
            rank = np.empty(n, dtype=np.float32)
            for r, idx in enumerate(order):
                rank[idx] = 1.0 / (60 + r)
            return rank
        fused = rrf(bm) + rrf(emb)                              # ranking quality only
        if _CHANGE_WORDS.search(query):                        # route "what changed" -> delta
            for i, c in enumerate(self.chunks):
                if c.source == "delta":
                    fused[i] *= 1.6
                    rel[i] = min(1.0, rel[i] * 1.15)
        order = np.argsort(-fused)[:top_k]
        return [(self.chunks[i], round(float(rel[i]), 3)) for i in order]
