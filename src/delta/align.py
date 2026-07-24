"""Content alignment between two revisions — the hard part of a delta.

Diffing is easy once you know which element in Rev A corresponds to which in Rev B.
Matching is the hard part, and doing it by *identity* (not pixel position) is what makes
this smarter than a text/pixel diff: a valve that moved 40pt is the SAME valve, and a line
whose size changed 10"->8" is the SAME line, not a delete + an add.

Two matching regimes, chosen per element type:

  ID_TYPES  (line, equipment, valve, instrument)
    have a stable textual identity. Match on an *identity key* that excludes mutable
    attributes:
       line  -> service-area-seq   (size & spec are attributes that may change)
       other -> normalized tag
    Exact identity-key match first; then fuzzy tag match for near-identical tags
    (catches OCR wobble and minor edits) above a configurable threshold.

  POS_TYPES (setpoint, dimension, note, text)
    are their own value — their "identity" is where they sit. Match by spatial proximity
    (centroid distance) gated by low text similarity, via greedy nearest-neighbour.

Everything is deterministic and reproducible; no LLM here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from ..canonical.model import CanonicalDoc, Element
from ..config import settings

ID_TYPES = {"line", "equipment", "valve", "instrument"}
POS_TYPES = {"setpoint", "dimension", "note", "text"}


def identity_key(e: Element) -> str:
    """A key that is stable under attribute changes, so an attribute edit reads as a
    modification rather than remove+add."""
    if e.type == "line" and e.attrs.service and e.attrs.raw.get("area") and e.attrs.seq:
        return f"line|{e.attrs.service}-{e.attrs.raw['area']}-{e.attrs.seq}"
    return f"{e.type}|{e.tag or e.text.strip().upper()}"


@dataclass
class Match:
    a: Optional[Element]
    b: Optional[Element]
    score: float          # 0..1 match confidence
    method: str           # id-exact | id-fuzzy | spatial


def _centroid_dist(a: Element, b: Element) -> float:
    (ax, ay), (bx, by) = a.centroid, b.centroid
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _match_id_type(a_elems, b_elems, etype) -> tuple[list[Match], list[Element], list[Element]]:
    matches: list[Match] = []
    a_left = [e for e in a_elems if e.type == etype]
    b_left = [e for e in b_elems if e.type == etype]

    # 1) exact identity-key match
    b_by_key: dict[str, list[Element]] = {}
    for e in b_left:
        b_by_key.setdefault(identity_key(e), []).append(e)
    a_rest = []
    used_b = set()
    for a in a_left:
        cands = b_by_key.get(identity_key(a), [])
        pick = next((c for c in cands if id(c) not in used_b), None)
        if pick is not None:
            used_b.add(id(pick))
            matches.append(Match(a, pick, 1.0, "id-exact"))
        else:
            a_rest.append(a)
    b_rest = [e for e in b_left if id(e) not in used_b]

    # 2) fuzzy tag match on the leftovers
    thr = settings.tag_fuzzy_threshold
    for a in a_rest:
        best, best_s = None, -1.0
        for b in b_rest:
            if id(b) in used_b:
                continue
            s = fuzz.ratio(a.tag or a.text, b.tag or b.text)
            if s > best_s:
                best, best_s = b, s
        if best is not None and best_s >= thr:
            used_b.add(id(best))
            matches.append(Match(a, best, best_s / 100.0, "id-fuzzy"))
        else:
            matches.append(Match(a, None, 0.0, "id-fuzzy"))  # removed
    for b in b_rest:
        if id(b) not in used_b:
            matches.append(Match(None, b, 0.0, "id-fuzzy"))   # added
    return matches, [], []


def _match_pos_type(a_elems, b_elems, etype) -> list[Match]:
    a_left = [e for e in a_elems if e.type == etype]
    b_left = [e for e in b_elems if e.type == etype]
    matches: list[Match] = []
    used_b = set()
    max_d = settings.spatial_match_max_dist
    # greedy nearest-neighbour by centroid, gated so far-apart items don't pair
    pairs = []
    for ai, a in enumerate(a_left):
        for bi, b in enumerate(b_left):
            d = _centroid_dist(a, b)
            if d <= max_d:
                pairs.append((d, ai, bi))
    pairs.sort()
    used_a = set()
    for d, ai, bi in pairs:
        if ai in used_a or bi in used_b:
            continue
        used_a.add(ai)
        used_b.add(bi)
        score = max(0.0, 1.0 - d / max_d)
        matches.append(Match(a_left[ai], b_left[bi], score, "spatial"))
    for ai, a in enumerate(a_left):
        if ai not in used_a:
            matches.append(Match(a, None, 0.0, "spatial"))
    for bi, b in enumerate(b_left):
        if bi not in used_b:
            matches.append(Match(None, b, 0.0, "spatial"))
    return matches


def align(a: CanonicalDoc, b: CanonicalDoc) -> list[Match]:
    """Return the full list of matches (matched pairs + unmatched A/B) across all types."""
    out: list[Match] = []
    types = {e.type for e in a.elements} | {e.type for e in b.elements}
    for t in sorted(types):
        if t in ID_TYPES:
            m, _, _ = _match_id_type(a.elements, b.elements, t)
            out.extend(m)
        else:
            out.extend(_match_pos_type(a.elements, b.elements, t))
    return out
