"""Deterministic P&ID element classifier + tag-grammar parser.

This is where raw extracted tokens (from ANY adapter) become typed, identity-bearing
`Element`s. It is pure, deterministic, and format-independent: give it tokens with
bounding boxes and it returns canonical elements. No LLM — the structural layer must be
reproducible.

The engineering-drawing conventions it encodes (validated against the sample P&IDs):

  line number   SIZE-SERVICE-AREA-SEQ-SPEC-INSUL   10"-VF-43-9025-AS20S-00
  equipment     AREA-TYPE-NUM[-SUB]                26-KA-901 / 26-KA-901-M01
  valve/fitting AREA + TYPE + NUM                  43BL9070 / 64GT9004
  instrument    ISA function code + tag number     PSV 9066A / PDIT 9054 (2 stacked tokens)
  setpoint      SP = / HH: / LL: + value           SP = 257 bar (g)
  dimension     EL +/- value M                     EL + 47.4 M
  note ref      NOTE <n>                           NOTE 34
"""
from __future__ import annotations

import re
from typing import Optional

from ..canonical.model import Attr, BBox, Element

# ---- token grammars --------------------------------------------------------------

# 10"-VF-43-9025-AS20S-00 | 12mm-PV-26-9118-FD70X-00 | 3/4"-...  (insul group optional)
LINE_RE = re.compile(
    r"^(?P<size>\d+(?:/\d+)?(?:\"|mm))-"
    r"(?P<service>[A-Z]{1,4})-"
    r"(?P<area>\d+)-"
    r"(?P<seq>\d+)-"
    r"(?P<spec>[A-Z0-9]+)"
    r"(?:-(?P<insul>[A-Z0-9]+))?$"
)
# 26-KA-901 | 26-FV-9076 | 26-KA-901-M01
EQUIP_RE = re.compile(r"^(?P<area>\d{2})-(?P<type>[A-Z]{2})-(?P<num>\d{3,4})(?:-(?P<sub>[A-Z0-9]+))?$")
# 43BL9070 | 64GT9004 | 40GB9023
VALVE_RE = re.compile(r"^(?P<area>\d{2})(?P<type>[A-Z]{2,3})(?P<num>\d{3,4}[A-Z]?)$")
NOTE_RE = re.compile(r"^NOTE$", re.I)
NUM_RE = re.compile(r"^\d{2,4}[A-Z]?$")            # bare number token
# Instrument loop numbers in this domain are 4-digit (optionally suffixed): 9066A, 9054.
# Requiring 4 digits is what separates real bubbles (PSV-9066A) from coincidental
# function-code-near-number pairs (DUE 50, ALARM 40, DP 257, SKID 16).
INSTR_NUM_RE = re.compile(r"^\d{4}[A-Z]?$")
# ISA function code: measured-variable first letter + >=1 function/modifier letter.
FUNC_RE = re.compile(r"^[PTFLAWSZ][A-Z]{1,3}$")

# words that look like function codes but aren't instruments
STOPWORDS = {
    "TO", "IS", "GAS", "THE", "AND", "FOR", "OPEN", "SEAL", "LAST", "TAKEN",
    "DRAIN", "FROM", "WITH", "HP", "LP", "VENDOR", "NOTE", "FLARE", "DSS", "EL",
    "CSO", "CSC", "LO", "LC", "NO", "NC", "TYP", "OF", "BY", "ON", "AT", "SEE",
    "DECK", "TOP", "REF", "MIN", "MAX", "DWG", "REV", "NTS", "PSD", "ALARM",
    "SKID", "TAG", "TYPE", "PART", "STD", "PLAN", "AREA", "LINE", "FLOW", "LEVEL",
}

# phrase-level (run over reconstructed text lines)
# Capture value = number + unit word (+ optional parenthesised qualifier), no more, so a
# two-column layout that concatenates a neighbouring line number can't leak in.
SETPOINT_RE = re.compile(r"\bSP\s*=\s*(?P<v>[-\d.]+\s*[A-Za-z]+(?:\s*\([a-zA-Z]\))?)", re.I)
LIMIT_RE = re.compile(r"\b(?P<k>HH|LL|H|L)\s*[:=]\s*(?P<v>[-\d.]+)")
ELEV_RE = re.compile(r"\bEL\s*(?P<v>[+\-]\s*[\d.]+\s*M)\b", re.I)


def normalize_tag(s: str) -> str:
    """Canonicalize a tag so trivial punctuation/whitespace differences don't look like
    changes. Uppercase, collapse spaces, unify separators to single hyphens."""
    s = s.strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_line_number(tok: str) -> Optional[Attr]:
    m = LINE_RE.match(tok)
    if not m:
        return None
    g = m.groupdict()
    return Attr(size=g["size"], service=g["service"], seq=g["seq"], spec=g["spec"],
               raw={"area": g["area"], "insul": g.get("insul")})


def _mk(etype, tag, text, bbox, page, source, confidence, attrs=None) -> Element:
    tag_n = normalize_tag(tag) if tag else None
    return Element(
        id=Element.make_id(etype, tag_n, text, bbox, page),
        type=etype, tag=tag_n, text=text, bbox=bbox, page=page,
        attrs=attrs or Attr(), confidence=confidence, source=source,
    )


def _classify_token(tok: str) -> tuple[str, Optional[str], Attr]:
    """Return (type, tag, attrs) for a single token. Order matters: most specific first."""
    attr = parse_line_number(tok)
    if attr is not None:
        return "line", tok, attr
    if EQUIP_RE.match(tok):
        m = EQUIP_RE.match(tok)
        return "equipment", tok, Attr(raw=m.groupdict())
    if VALVE_RE.match(tok):
        m = VALVE_RE.match(tok)
        return "valve", tok, Attr(raw=m.groupdict())
    return "text", None, Attr()


def _dist(a: Element | tuple, b_centroid) -> float:
    ac = a.centroid if isinstance(a, Element) else a
    return ((ac[0] - b_centroid[0]) ** 2 + (ac[1] - b_centroid[1]) ** 2) ** 0.5


def classify_words(words: list[dict], page: int, source: str, base_conf: float) -> list[Element]:
    """words: list of {text,x0,x1,top,bottom}. Returns typed elements.

    Two passes: (1) classify atomic tokens; (2) pair ISA function codes with a nearby
    number token to form instrument bubbles (e.g. 'PSV' + '9066A' -> instrument PSV-9066A).
    """
    toks = []
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        bbox = (float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))
        toks.append({"text": t, "bbox": bbox, "cx": (bbox[0] + bbox[2]) / 2,
                     "cy": (bbox[1] + bbox[3]) / 2})

    consumed = set()
    elements: list[Element] = []

    # pass 1: atomic classification, remember candidate instrument codes & numbers
    prelim = []
    for i, t in enumerate(toks):
        etype, tag, attr = _classify_token(t["text"])
        prelim.append((etype, tag, attr))

    # pass 2: instrument bubble pairing (function code + nearest number below/near)
    for i, t in enumerate(toks):
        if i in consumed:
            continue
        etype, tag, attr = prelim[i]
        if etype != "text":
            continue
        w = t["text"]
        if FUNC_RE.match(w) and w not in STOPWORDS:
            # find nearest unconsumed bare-number token within radius
            best_j, best_d = None, 1e9
            for j, u in enumerate(toks):
                if j == i or j in consumed:
                    continue
                if prelim[j][0] != "text" or not INSTR_NUM_RE.match(u["text"]):
                    continue
                d = _dist((t["cx"], t["cy"]), (u["cx"], u["cy"]))
                # bubble stacks code above number: number is near & roughly below/right
                if d < best_d and d <= 28 and abs(u["cx"] - t["cx"]) <= 20:
                    best_d, best_j = d, j
            if best_j is not None:
                u = toks[best_j]
                num = u["text"]
                bbox = (min(t["bbox"][0], u["bbox"][0]), min(t["bbox"][1], u["bbox"][1]),
                        max(t["bbox"][2], u["bbox"][2]), max(t["bbox"][3], u["bbox"][3]))
                tag = f"{w}-{num}"
                elements.append(_mk("instrument", tag, f"{w} {num}", bbox, page, source,
                                    base_conf, Attr(raw={"func": w, "num": num})))
                consumed.add(i)
                consumed.add(best_j)

    # emit remaining atomic elements (skip generic single text tokens to cut noise, but
    # keep tagged ones and multi-char words that carry meaning)
    for i, t in enumerate(toks):
        if i in consumed:
            continue
        etype, tag, attr = prelim[i]
        if etype == "text":
            if NOTE_RE.match(t["text"]):
                continue  # handled at phrase level (NOTE + number)
            continue      # drop bare generic tokens; phrase pass captures meaningful text
        elements.append(_mk(etype, tag, t["text"], t["bbox"], page, source, base_conf, attr))

    return elements


LIMIT_KEYS = {"HH", "LL", "H", "L"}
NUM_VAL_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def _row_neighbors(toks: list[dict], i: int, max_dx: float = 42, max_dy: float = 4):
    """Yield indices of tokens on the same visual row to the right of tok i, nearest first.
    Row = small vertical centroid delta; deterministic and layout-independent (no reliance
    on pdfplumber line grouping, which proved inconsistent between near-identical pages)."""
    ti = toks[i]
    cands = []
    for j, u in enumerate(toks):
        if j == i:
            continue
        if abs(u["cy"] - ti["cy"]) <= max_dy and 0 < (u["cx"] - ti["cx"]) <= max_dx:
            cands.append((u["cx"] - ti["cx"], j))
    return [j for _, j in sorted(cands)]


def classify_phrases(words: list[dict], page: int, source: str, base_conf: float) -> list[Element]:
    """Token-proximity phrase detection over the raw word tokens (setpoints, limits,
    elevations, NOTE refs). Deterministic: two near-identical pages yield the same
    elements, which is essential for a trustworthy delta."""
    toks = []
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        bbox = (float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))
        toks.append({"text": t, "bbox": bbox, "cx": (bbox[0] + bbox[2]) / 2,
                     "cy": (bbox[1] + bbox[3]) / 2})

    out: list[Element] = []
    for i, t in enumerate(toks):
        raw = t["text"]
        key = raw.rstrip(":").upper()  # "HH:" -> "HH"

        # ---- pressure/level limits: HH|LL|H|L [:] <value> ----
        if key in LIMIT_KEYS:
            for j in _row_neighbors(toks, i):
                cand = toks[j]["text"].lstrip(":").strip()
                if cand == ":":
                    continue
                if NUM_VAL_RE.match(cand):
                    bbox = _union(t["bbox"], toks[j]["bbox"])
                    out.append(_mk("setpoint", f"{key}:{cand}", f"{key}: {cand}", bbox,
                                   page, source, base_conf,
                                   Attr(setpoint=f"{key}={cand}", value=cand)))
                    break
                if cand and not cand in (":",):
                    break  # first non-colon neighbour isn't a number -> not a limit

        # ---- SP = <num> <unit> [(q)] ----
        elif key == "SP":
            parts, vb = [], t["bbox"]
            for j in _row_neighbors(toks, i, max_dx=70):
                c = toks[j]["text"]
                if c == "=":
                    continue
                parts.append(c)
                vb = _union(vb, toks[j]["bbox"])
                if len(parts) >= 3:
                    break
            val = " ".join(p for p in parts if p != "=").strip()
            val = re.sub(r"^=\s*", "", val)
            if val and re.search(r"\d", val):
                out.append(_mk("setpoint", f"SP@{val}", f"SP = {val}", vb, page, source,
                               base_conf, Attr(setpoint=val)))

        # ---- NOTE <n> ----
        elif key == "NOTE":
            for j in _row_neighbors(toks, i, max_dx=30):
                c = toks[j]["text"]
                if c.isdigit():
                    bbox = _union(t["bbox"], toks[j]["bbox"])
                    out.append(_mk("note", f"NOTE-{c}", f"NOTE {c}", bbox, page, source,
                                   base_conf, Attr(value=c)))
                break

        # ---- EL +/- <num> M (elevation/dimension) ----
        elif key == "EL":
            nbrs = _row_neighbors(toks, i, max_dx=60)
            frag = " ".join(toks[j]["text"] for j in nbrs[:3])
            m = re.search(r"([+\-]\s*[\d.]+)\s*M", frag)
            if m:
                v = re.sub(r"\s+", "", m.group(1)) + "M"
                bb = t["bbox"]
                for j in nbrs[:3]:
                    bb = _union(bb, toks[j]["bbox"])
                out.append(_mk("dimension", f"EL{v}", f"EL {v}", bb, page, source,
                               base_conf, Attr(value=v)))
    return out


def _union(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def merge(word_elems: list[Element], phrase_elems: list[Element]) -> list[Element]:
    """Combine token and phrase elements, dropping token elements fully contained in a
    phrase element (so a setpoint phrase supersedes its constituent number token)."""
    kept = list(phrase_elems)
    for we in word_elems:
        if any(_iou(we.bbox, pe.bbox) > 0.6 for pe in phrase_elems):
            continue
        kept.append(we)
    return kept
