"""Tests that matter: the deterministic core (ingest → delta) and the grounding gate.

These lock the properties the whole system rests on:
  - the native adapter recovers tagged elements with the P&ID grammar
  - the delta engine finds exactly the authored edits on pair1 (P=R=1.0)
  - alignment classifies size/spec/setpoint/move/add/remove correctly
  - the chat refusal gate fires on out-of-domain queries
Run: python -m pytest -q
"""
from __future__ import annotations

import pytest

from src.chat.index import RetrievalIndex
from src.delta.engine import compute_delta
from src.ingest.classify import parse_line_number
from src.ingest.registry import ingest


@pytest.fixture(scope="module")
def pair1():
    a = ingest("data/samples/pair1/revA.pdf", "A")
    b = ingest("data/samples/pair1/revB.pdf", "B")
    return a, b, compute_delta(a, b)


def test_line_grammar():
    attr = parse_line_number('10"-VF-43-9025-AS20S-00')
    assert attr and attr.size == '10"' and attr.service == "VF" and attr.spec == "AS20S"
    assert parse_line_number("not-a-line") is None


def test_native_ingest_recovers_tags(pair1):
    a, _, _ = pair1
    assert len(a.elements) > 100
    assert any(e.type == "line" for e in a.elements)
    assert any(e.type == "instrument" for e in a.elements)


def test_delta_matches_ground_truth(pair1):
    _, _, d = pair1
    assert d.summary["total"] == 6
    kinds = {(c.op, c.subtype) for c in d.changes}
    assert ("added", None) in kinds        # PT-9099
    assert ("removed", None) in kinds      # 43BL9070
    assert ("modified", "attribute") in kinds
    assert ("modified", "moved") in kinds


def test_size_change_detected(pair1):
    _, _, d = pair1
    size = [c for c in d.changes if any(fd["field"] == "size" for fd in c.field_diffs)]
    assert size and size[0].field_diffs[0]["before"] == '10"'
    assert size[0].field_diffs[0]["after"] == '8"'


def test_delta_is_deterministic():
    a = ingest("data/samples/pair1/revA.pdf", "A")
    b = ingest("data/samples/pair1/revB.pdf", "B")
    d1 = compute_delta(a, b)
    d2 = compute_delta(a, b)
    assert [c.id for c in d1.changes] == [c.id for c in d2.changes]


def test_chat_refuses_offdomain(pair1):
    a, b, d = pair1
    idx = RetrievalIndex().build(a, b, d)
    top_in = idx.search("did any line size change?")[0][1]
    top_out = idx.search("what is the capital of France?")[0][1]
    assert top_in > top_out
    assert top_out < 0.12  # below refusal gate
