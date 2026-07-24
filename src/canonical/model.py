"""The format-agnostic canonical representation — the seam of the whole system.

Every ingestion adapter (native PDF, scanned PDF, DWG) normalizes its input into a
`CanonicalDoc`: a flat list of typed `Element`s with geometry. The delta engine and the
chat layer consume `CanonicalDoc` only and never learn what the source format was.

Design note: P&IDs (and most engineering drawings) are *tag-based*. Almost every
meaningful element carries a stable textual identity — a line number, an instrument
bubble, an equipment tag. That identity, not pixel position, is what we align on. So the
canonical unit is an `Element` keyed by a parsed `tag` plus a geometry `bbox`.
"""
from __future__ import annotations

import hashlib
from typing import Literal, Optional

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]  # (x0, y0, x1, y1) in page points, origin top-left

ElementType = Literal[
    "line",        # process/pipe line number, e.g. 10"-VF-43-9025-AS20S-00
    "instrument",  # instrument bubble, e.g. PDIT-9054, PSV-9066A
    "equipment",   # equipment tag, e.g. 26-KA-902
    "valve",       # valve / fitting tag, e.g. 63BL9022
    "setpoint",    # SP = 257 bar (g), HH: 150
    "note",        # NOTE 34
    "dimension",   # explicit dimension / elevation, e.g. EL + 47.4 M
    "table_cell",  # a cell in a titleblock / revision table
    "text",        # generic uncategorized text token
    "symbol",      # vector symbol / block with no text identity
]

Op = Literal["added", "removed", "modified"]


class Attr(BaseModel):
    """Structured attributes parsed out of a tag/text. Populated best-effort; a field is
    None when it doesn't apply. `raw` holds anything parsed but not modeled explicitly."""
    size: Optional[str] = None       # "10\"", "12mm", "3/4\""
    service: Optional[str] = None    # line service code: VF, AI, PV, GT ...
    seq: Optional[str] = None        # sequence number within the line tag
    spec: Optional[str] = None       # piping spec class: AS20S, AC21S ...
    setpoint: Optional[str] = None   # "257 bar (g)"
    value: Optional[str] = None      # generic numeric value (dimensions, HH/LL limits)
    raw: dict = Field(default_factory=dict)

    def diff_fields(self, other: "Attr") -> list[dict]:
        """Field-level diff against another Attr. Returns [{field, before, after}]."""
        out = []
        for f in ("size", "service", "seq", "spec", "setpoint", "value"):
            a, b = getattr(self, f), getattr(other, f)
            if a != b and (a is not None or b is not None):
                out.append({"field": f, "before": a, "after": b})
        return out


class Element(BaseModel):
    """One atomic, located piece of a document."""
    id: str
    type: ElementType
    tag: Optional[str] = None   # canonical identity token, normalized (see normalize_tag)
    text: str = ""
    bbox: BBox
    page: int = 0
    attrs: Attr = Field(default_factory=Attr)
    confidence: float = 1.0     # 1.0 for native PDF; <1 for OCR-recovered content
    source: str = "pdf_native"  # which adapter produced it

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @staticmethod
    def make_id(etype: str, tag: str | None, text: str, bbox: BBox, page: int,
                bbox_round: int = 1) -> str:
        """Stable within-document id. Deterministic so runs are reproducible and so an
        unchanged element keeps the same id across a re-ingest."""
        key = (
            f"{etype}|{tag or ''}|{text.strip()}|{page}|"
            + ",".join(str(round(c, bbox_round)) for c in bbox)
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


class CanonicalDoc(BaseModel):
    """A single document revision, normalized. `pid` is the persistent identifier the
    system resolves to bytes+metadata (per the assignment's domain language)."""
    pid: str
    format: str                      # pdf_native | pdf_scanned | dwg
    revision_label: Optional[str] = None
    sheet_count: int = 1
    page_sizes: list[tuple[float, float]] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)

    def by_type(self, etype: ElementType) -> list[Element]:
        return [e for e in self.elements if e.type == etype]

    def by_id(self) -> dict[str, Element]:
        return {e.id: e for e in self.elements}


# ---- delta types -----------------------------------------------------------------

class Change(BaseModel):
    """One meaningful change from PID A (base) to PID B (revised)."""
    id: str                                  # citable id, referenced as DELTA:<id>
    op: Op
    subtype: Optional[str] = None            # moved | attribute | text | None
    element_type: ElementType
    tag: Optional[str] = None
    page: int = 0
    bbox: BBox
    field_diffs: list[dict] = Field(default_factory=list)  # [{field,before,after}]
    description: str = ""                     # deterministic, template-generated
    confidence: float = 1.0
    evidence: dict = Field(default_factory=dict)  # {a_id, b_id, match_score}

    @staticmethod
    def make_id(op: str, tag: str | None, element_type: str, a_id: str | None,
                b_id: str | None) -> str:
        key = f"{op}|{element_type}|{tag or ''}|{a_id or ''}|{b_id or ''}"
        return "d" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


class Delta(BaseModel):
    pid_a: str
    pid_b: str
    rev_a: Optional[str] = None
    rev_b: Optional[str] = None
    changes: list[Change] = Field(default_factory=list)

    @property
    def summary(self) -> dict:
        by_op: dict[str, int] = {"added": 0, "removed": 0, "modified": 0}
        by_type: dict[str, int] = {}
        for c in self.changes:
            by_op[c.op] += 1
            by_type[c.element_type] = by_type.get(c.element_type, 0) + 1
        return {"total": len(self.changes), "by_op": by_op, "by_type": by_type}
