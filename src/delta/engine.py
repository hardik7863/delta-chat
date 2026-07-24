"""Delta engine: turn alignment matches into a structured, typed, located, confidence-
scored Delta. Deterministic — the reproducible core of the system. No LLM.

Classification per match:
  a and not b            -> removed
  b and not a            -> added
  a and b, attrs differ  -> modified/attribute  (field-level before/after diffs)
  a and b, moved > thr   -> modified/moved       (centroid shift)
  a and b, text differ   -> modified/text
  else                   -> unchanged (dropped)

Confidence blends match strength with the source confidence of both elements (so an
OCR-recovered element yields a lower-confidence change than a native-PDF one).
"""
from __future__ import annotations

from typing import Optional

from ..canonical.model import Change, CanonicalDoc, Delta, Element
from ..config import settings
from ..observability.tracing import Trace
from .align import Match, _centroid_dist, align


def _describe(op: str, subtype: Optional[str], e: Element, field_diffs: list[dict]) -> str:
    """Deterministic, template-generated natural-language description (no LLM, so the
    report is reproducible). The chat layer may paraphrase these, but the artifact text
    is fixed."""
    tag = e.tag or e.text.strip() or e.type
    if op == "added":
        return f"Added {e.type} {tag}."
    if op == "removed":
        return f"Removed {e.type} {tag}."
    if subtype == "attribute":
        parts = [f"{d['field']} {d['before']}→{d['after']}" for d in field_diffs]
        return f"Modified {e.type} {tag}: " + "; ".join(parts) + "."
    if subtype == "moved":
        return f"Moved {e.type} {tag}."
    if subtype == "text":
        return f"Changed text of {e.type} {tag}."
    return f"Modified {e.type} {tag}."


def _confidence(match_score: float, a: Optional[Element], b: Optional[Element]) -> float:
    src = min([e.confidence for e in (a, b) if e is not None] or [1.0])
    return round(max(0.0, min(1.0, 0.5 * match_score + 0.5 * src)), 3)


def _change_from_match(m: Match) -> Optional[Change]:
    a, b = m.a, m.b
    if a is not None and b is None:
        c_conf = _confidence(1.0, a, None)
        return Change(
            id=Change.make_id("removed", a.tag, a.type, a.id, None),
            op="removed", element_type=a.type, tag=a.tag, page=a.page, bbox=a.bbox,
            description=_describe("removed", None, a, []), confidence=c_conf,
            evidence={"a_id": a.id},
        )
    if a is None and b is not None:
        c_conf = _confidence(1.0, None, b)
        return Change(
            id=Change.make_id("added", b.tag, b.type, None, b.id),
            op="added", element_type=b.type, tag=b.tag, page=b.page, bbox=b.bbox,
            description=_describe("added", None, b, []), confidence=c_conf,
            evidence={"b_id": b.id},
        )
    # matched pair -> is it modified?
    field_diffs = a.attrs.diff_fields(b.attrs)
    moved = _centroid_dist(a, b) > settings.spatial_move_threshold
    text_changed = (a.text.strip() != b.text.strip()) and not field_diffs

    subtype = None
    if field_diffs:
        subtype = "attribute"
    elif moved:
        subtype = "moved"
    elif text_changed:
        subtype = "text"
    else:
        return None  # unchanged

    ev = {"a_id": a.id, "b_id": b.id, "match_score": round(m.score, 3), "method": m.method}
    if subtype == "moved":
        ev["shift_pts"] = round(_centroid_dist(a, b), 1)
    return Change(
        id=Change.make_id("modified", b.tag or a.tag, b.type, a.id, b.id),
        op="modified", subtype=subtype, element_type=b.type, tag=b.tag or a.tag,
        page=b.page, bbox=b.bbox, field_diffs=field_diffs,
        description=_describe("modified", subtype, b, field_diffs),
        confidence=_confidence(m.score, a, b), evidence=ev,
    )


def compute_delta(a: CanonicalDoc, b: CanonicalDoc, trace: Optional[Trace] = None) -> Delta:
    def _run() -> Delta:
        matches = align(a, b)
        changes = [c for m in matches if (c := _change_from_match(m)) is not None]
        # stable ordering: by page, then op, then tag — reproducible reports
        changes.sort(key=lambda c: (c.page, c.op, c.element_type, c.tag or ""))
        return Delta(pid_a=a.pid, pid_b=b.pid, rev_a=a.revision_label,
                     rev_b=b.revision_label, changes=changes)

    if trace is None:
        return _run()
    with trace.span("delta", elements_a=len(a.elements), elements_b=len(b.elements)) as sp:
        d = _run()
        sp.attributes.update(changes=len(d.changes), **{f"n_{k}": v
                             for k, v in d.summary["by_op"].items()})
        trace.set_metric("delta_counts", d.summary)
        return d
