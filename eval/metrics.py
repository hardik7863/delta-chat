"""Eval metrics: delta precision/recall/F1, and chat groundedness/correctness.

Delta matching. A predicted `Change` matches an expected label when they share the same
`op` and `element_type` and the expected `identity` string appears in the predicted
change's "haystack" (tag + every field before/after value + description). Matching is
greedy one-to-one, so duplicates don't inflate the score. Field-level correctness is
tracked separately as a secondary signal.

Chat. Groundedness = did a non-refused answer carry >=1 valid citation (a citation id
that actually exists in the index)? Correctness = LLM-as-judge over expected key facts,
with a lexical fallback when no real LLM is configured. Refusal questions score correct
iff the system refused.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _haystack(change) -> str:
    parts = [change.tag or "", change.description, change.subtype or ""]
    for d in change.field_diffs:
        parts += [str(d.get("field")), str(d.get("before")), str(d.get("after"))]
    return _norm(" ".join(parts))


@dataclass
class DeltaScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    field_correct: int = 0
    field_total: int = 0
    matched: list = field(default_factory=list)
    missed: list = field(default_factory=list)     # false negatives (labels not found)
    spurious: list = field(default_factory=list)   # false positives (extra predictions)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def field_accuracy(self) -> float:
        return self.field_correct / self.field_total if self.field_total else 1.0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 3), "recall": round(self.recall, 3),
                "f1": round(self.f1, 3), "field_accuracy": round(self.field_accuracy, 3),
                "missed": self.missed, "spurious": self.spurious}


def score_delta(predicted: list, expected: list[dict]) -> DeltaScore:
    s = DeltaScore()
    used = set()
    for label in expected:
        ident = _norm(label["identity"])
        found = None
        for i, c in enumerate(predicted):
            if i in used:
                continue
            if c.op == label["op"] and c.element_type == label["element_type"] \
               and ident and ident in _haystack(c):
                found = i
                break
        if found is not None:
            used.add(found)
            s.tp += 1
            s.matched.append(label["identity"])
            # field-level check for attribute changes
            for fd in label.get("field_diffs", []):
                s.field_total += 1
                pred_fields = predicted[found].field_diffs
                ok = any(pf.get("field") == fd["field"]
                         and _norm(str(pf.get("after"))) == _norm(str(fd["after"]))
                         for pf in pred_fields)
                s.field_correct += 1 if ok else 0
        else:
            s.fn += 1
            s.missed.append(label["identity"])
    s.fp = len(predicted) - len(used)
    for i, c in enumerate(predicted):
        if i not in used:
            s.spurious.append(f"{c.op}:{c.element_type}:{c.tag}")
    return s


# ---- chat scoring --------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict grader. Given a QUESTION, a set of REQUIRED FACTS the correct answer "
    "must convey, and a CANDIDATE ANSWER, respond with exactly 'CORRECT' if the candidate "
    "conveys all required facts (allowing paraphrase), otherwise 'INCORRECT'. One word only."
)


def judge_correct(question: str, key_facts: list[str], answer: str, llm) -> bool:
    """LLM-as-judge with a lexical fallback for keyless/echo runs."""
    if not key_facts:
        return True
    if llm is None or llm.provider == "echo":
        # lexical fallback: all key facts present (normalized substring)
        hay = _norm(answer)
        return all(_norm(k) in hay for k in key_facts)
    user = (f"QUESTION: {question}\nREQUIRED FACTS: {key_facts}\n"
            f"CANDIDATE ANSWER: {answer}\nVerdict:")
    res = llm.complete(JUDGE_SYSTEM, user, purpose="judge", max_tokens=8)
    return "CORRECT" in res.text.upper() and "INCORRECT" not in res.text.upper()


def valid_citations(citations: list[str], valid_ids: set[str]) -> tuple[int, int]:
    """(# valid citations, # total citations)."""
    return sum(1 for c in citations if c in valid_ids), len(citations)


# ---- retrieval quality ---------------------------------------------------------

def retrieval_rank(index, query: str, gold_all: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk that contains ALL gold substrings
    (case-insensitive), scanning the full ranking. None if never retrieved.

    Measures whether retrieval surfaces the answer-bearing source at all, and how high —
    which is what grounded chat depends on, upstream of the LLM."""
    hits = index.search(query, top_k=len(index.chunks))
    gold = [g.lower() for g in gold_all]
    for rank, (chunk, _score) in enumerate(hits, start=1):
        t = chunk.text.lower()
        if all(g in t for g in gold):
            return rank
    return None


def retrieval_report(ranks: list[int | None], ks=(1, 3, 5)) -> dict:
    """Aggregate hit@k, MRR, and mean rank over a set of queries."""
    n = len(ranks)
    found = [r for r in ranks if r is not None]
    out = {"n": n, "found": len(found),
           "mrr": round(sum(1.0 / r for r in found) / n, 3) if n else 0.0,
           "mean_rank": round(sum(found) / len(found), 2) if found else None}
    for k in ks:
        out[f"hit@{k}"] = round(sum(1 for r in found if r <= k) / n, 3) if n else 0.0
    return out
