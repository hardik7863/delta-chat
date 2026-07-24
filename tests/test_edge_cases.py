"""Aggressive edge-case + regression coverage.

Locks behavior on the nasty cases: identical revisions, swapped direction (delta must
invert), corrupt / missing / unknown inputs (visible failures, not swallowed), confidence
bounds, id uniqueness, report round-trip, OCR-confidence propagation, retrieval bounds and
the refusal boundary, and cross-format determinism.
"""
from __future__ import annotations

import json

import pytest

from src.canonical.model import Delta
from src.chat.index import RetrievalIndex
from src.delta.engine import compute_delta
from src.delta.report import to_markdown, write_report
from src.ingest.base import resolve_pid
from src.ingest.registry import ingest


A_PATH = "data/samples/pair1/revA.pdf"
B_PATH = "data/samples/pair1/revB.pdf"


@pytest.fixture(scope="module")
def docs():
    return ingest(A_PATH, "A"), ingest(B_PATH, "B")


# ---- delta edge cases ----------------------------------------------------------

def test_identical_revisions_zero_delta(docs):
    a, _ = docs
    d = compute_delta(a, a)
    assert d.summary["total"] == 0, "a document vs itself must have no changes"


def test_swapped_direction_inverts_ops(docs):
    a, b = docs
    fwd = compute_delta(a, b)
    rev = compute_delta(b, a)
    # counts of added/removed swap; modified count is stable
    assert fwd.summary["by_op"]["added"] == rev.summary["by_op"]["removed"]
    assert fwd.summary["by_op"]["removed"] == rev.summary["by_op"]["added"]
    assert fwd.summary["by_op"]["modified"] == rev.summary["by_op"]["modified"]


def test_change_ids_unique(docs):
    a, b = docs
    d = compute_delta(a, b)
    ids = [c.id for c in d.changes]
    assert len(ids) == len(set(ids)), "change ids must be unique (citable)"


def test_confidence_bounds(docs):
    a, b = docs
    d = compute_delta(a, b)
    assert all(0.0 <= c.confidence <= 1.0 for c in d.changes)


def test_report_roundtrips_to_delta(docs, tmp_path):
    a, b = docs
    d = compute_delta(a, b)
    paths = write_report(d, tmp_path)
    reloaded = Delta.model_validate_json(paths["json"].read_text())
    assert reloaded.summary == d.summary
    assert "Delta Report" in to_markdown(d)


# ---- ingestion failure visibility ----------------------------------------------

def test_missing_pid_raises():
    with pytest.raises(FileNotFoundError):
        resolve_pid("data/samples/does_not_exist.pdf")


def test_unknown_format_raises(tmp_path):
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not a pdf or dxf at all")
    with pytest.raises(ValueError):
        ingest(str(junk))


def test_corrupt_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 truncated garbage")
    with pytest.raises(Exception):
        ingest(str(bad))


# ---- cross-format determinism --------------------------------------------------

@pytest.mark.parametrize("pair,pa,pb", [
    ("pair1", A_PATH, B_PATH),
    ("pair2", "data/samples/pair2/revA.pdf", "data/samples/pair2/revB.pdf"),
    ("pair3", "data/samples/pair3/revA.dxf", "data/samples/pair3/revB.dxf"),
])
def test_delta_deterministic_all_formats(pair, pa, pb):
    d1 = compute_delta(ingest(pa, "A"), ingest(pb, "B"))
    d2 = compute_delta(ingest(pa, "A"), ingest(pb, "B"))
    assert [c.id for c in d1.changes] == [c.id for c in d2.changes]
    assert d1.summary == d2.summary


def test_scanned_confidence_below_one():
    b = ingest("data/samples/pair2/revB.pdf", "B")
    assert b.format == "pdf_scanned"
    assert b.elements and all(e.confidence < 1.0 for e in b.elements)


def test_dxf_routes_to_dwg_adapter():
    d = ingest("data/samples/pair3/revA.dxf", "A")
    assert d.format == "dwg" and len(d.elements) > 0


# ---- retrieval / refusal boundary ----------------------------------------------

@pytest.fixture(scope="module")
def index(docs):
    a, b = docs
    return RetrievalIndex().build(a, b, compute_delta(a, b))


def test_retrieval_respects_top_k(index):
    hits = index.search("valve", top_k=3)
    assert len(hits) <= 3


def test_empty_query_does_not_crash(index):
    hits = index.search("")
    assert isinstance(hits, list)


@pytest.mark.parametrize("q", [
    "who painted the mona lisa?",
    "what is the meaning of life?",
    "recipe for pasta carbonara",
])
def test_offdomain_below_gate(index, q):
    from src.config import settings
    assert index.search(q)[0][1] < settings.retrieval_min_score


@pytest.mark.parametrize("q", [
    "did any line size change?",
    "which valve was removed?",
    "what setpoint changed?",
])
def test_indomain_above_gate(index, q):
    from src.config import settings
    assert index.search(q)[0][1] >= settings.retrieval_min_score


def test_retrieval_finds_answer_chunk_high(index):
    from eval.metrics import retrieval_rank, retrieval_report
    rk = retrieval_rank(index, "which valve was removed?", ["43BL9070", "removed"])
    assert rk is not None and rk <= 3, "answer-bearing chunk must rank in the top 3"
    rep = retrieval_report([1, 2, None])
    assert rep["hit@1"] == round(1 / 3, 3) and rep["found"] == 2
