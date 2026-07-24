"""Synthesize a DXF revision pair (pair3) with known edits, exercising the DWG/DXF adapter
end-to-end. We author a small P&ID-style DXF (TEXT tags + LINE pipes) as Rev A, then apply
controlled edits for Rev B and record the exact expected delta.
"""
from __future__ import annotations

import json

import re

import ezdxf

from .config import settings

_INSTR = re.compile(r"^([PTFLAWSZ][A-Z]{1,3})\s+(\d{4}[A-Z]?)$")

# (tag_text, x, y) — a compact synthetic P&ID in drawing units (mm).
BASE_TAGS = [
    ('10"-VF-43-9025-AS20S-00', 20, 180),
    ('6"-PV-26-9044-GC11S-38', 20, 160),
    ('2"-WC-40-9014-AC21-00', 20, 140),
    ("26-KA-902", 90, 175),
    ("26-CX-9021", 90, 150),
    ("PSV 9066A", 150, 178),
    ("PDIT 9054", 150, 150),
    ("40BL9021", 60, 120),
    ("43BL9070", 120, 120),
    ("NOTE 34", 30, 100),
]
PIPES = [((30, 182), (85, 182)), ((30, 162), (85, 162)), ((95, 175), (145, 178))]


def _write_dxf(path, tags, pipes):
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    for txt, x, y in tags:
        m = _INSTR.match(txt)
        if m:  # instrument bubble: code above number, as separate TEXT entities
            msp.add_text(m.group(1), dxfattribs={"height": 2.5, "insert": (x, y + 4)})
            msp.add_text(m.group(2), dxfattribs={"height": 2.5, "insert": (x, y)})
        else:
            msp.add_text(txt, dxfattribs={"height": 2.5, "insert": (x, y)})
    for (x0, y0), (x1, y1) in pipes:
        msp.add_line((x0, y0), (x1, y1))
    doc.saveas(path)


def build_pair3():
    out_dir = settings.samples_dir / "pair3"
    out_dir.mkdir(parents=True, exist_ok=True)
    revA = out_dir / "revA.dxf"
    revB = out_dir / "revB.dxf"

    _write_dxf(revA, BASE_TAGS, PIPES)

    # Rev B edits: line size 10"->8" (VF-43-9025); remove valve 43BL9070;
    # add instrument PT 9099; move valve 40BL9021.
    tags_b = []
    for txt, x, y in BASE_TAGS:
        if txt == '10"-VF-43-9025-AS20S-00':
            tags_b.append(('8"-VF-43-9025-AS20S-00', x, y))
        elif txt == "43BL9070":
            continue  # removed
        elif txt == "40BL9021":
            tags_b.append((txt, x + 25, y - 8))  # moved
        else:
            tags_b.append((txt, x, y))
    tags_b.append(("PT 9099", 150, 120))  # added instrument
    _write_dxf(revB, tags_b, PIPES)

    expected = {
        "pair": "pair3", "pid_a": str(revA), "pid_b": str(revB),
        "rev_a": "A", "rev_b": "B",
        "changes": [
            {"op": "modified", "element_type": "line", "subtype": "attribute",
             "identity": "VF-43-9025",
             "field_diffs": [{"field": "size", "before": '10"', "after": '8"'}]},
            {"op": "removed", "element_type": "valve", "identity": "43BL9070",
             "field_diffs": []},
            {"op": "added", "element_type": "instrument", "identity": "PT-9099",
             "field_diffs": []},
            {"op": "modified", "element_type": "valve", "subtype": "moved",
             "identity": "40BL9021", "field_diffs": []},
        ],
    }
    (out_dir / "expected_delta.json").write_text(json.dumps(expected, indent=2))
    (out_dir / "provenance.md").write_text(
        "# pair3 — synthetic DXF pair (DWG/DXF adapter)\n\n"
        "- **Rev A / Rev B**: authored with `ezdxf` (`src/synth_dwg.py`) — TEXT tags + "
        "LINE pipes in a compact P&ID layout.\n"
        "- **DWG note**: the adapter accepts `.dwg` and converts DWG→DXF via ODA File "
        "Converter when present; we ship `.dxf` directly so the pair runs with no external "
        "tool. The conversion seam is real (see `src/ingest/dwg.py`).\n"
        "- **Ground truth**: 4 edits — size 10\"→8\", valve 43BL9070 removed, PT-9099 "
        "added, valve 40BL9021 moved.\n"
    )
    return out_dir


if __name__ == "__main__":
    print("built", build_pair3())
