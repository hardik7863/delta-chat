"""Synthetic revision-pair generator + ground-truth labeller.

The assignment provides two *different* drawings, not two revisions of one document. To
evaluate a delta engine you need pairs whose true delta is known. So we take a real native
P&ID as Rev A and apply a small set of *controlled* edits to produce Rev B — and because
we author the edits, we know the exact ground-truth delta for free. That labelled delta is
what the eval harness scores against (precision/recall/F1), so the numbers are honest.

Edits are applied at the PDF content level with PyMuPDF (redact old text, insert new), so
Rev B is a genuinely different born-digital PDF, not an overlay.

Provenance is written to each pair's provenance.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import fitz  # PyMuPDF

from .config import settings

SOURCE = settings.samples_dir / "_source" / "lift_gas.pdf"

# Each edit is data + its own ground-truth label. `find` is the exact text in Rev A.
# op/element_type/field_diffs describe the Change the delta engine SHOULD emit.
EDITS = [
    {
        "kind": "replace",
        "find": '10"-VF-43-9025-AS20S-00',
        "replace": '8"-VF-43-9025-AS20S-00',
        "label": {"op": "modified", "element_type": "line", "subtype": "attribute",
                  "identity": "VF-43-9025",
                  "field_diffs": [{"field": "size", "before": '10"', "after": '8"'}]},
    },
    {
        "kind": "replace",
        "find": '6"-VF-43-9029-AC21S-00',
        "replace": '6"-VF-43-9029-GC11S-00',
        "label": {"op": "modified", "element_type": "line", "subtype": "attribute",
                  "identity": "VF-43-9029",
                  "field_diffs": [{"field": "spec", "before": "AC21S", "after": "GC11S"}]},
    },
    {
        "kind": "replace",
        "find": "245",
        "replace": "300",
        "label": {"op": "modified", "element_type": "setpoint", "subtype": "attribute",
                  "identity": "HH=245",
                  "field_diffs": [{"field": "setpoint", "before": "HH=245",
                                   "after": "HH=300"}]},
    },
    {
        "kind": "delete",
        "find": "43BL9070",
        "label": {"op": "removed", "element_type": "valve", "identity": "43BL9070",
                  "field_diffs": []},
    },
    {
        "kind": "move",
        "find": "26BL9073",
        "dx": 0.0, "dy": 34.0,
        "label": {"op": "modified", "element_type": "valve", "subtype": "moved",
                  "identity": "26BL9073", "field_diffs": []},
    },
    {
        "kind": "add",
        "tokens": ["PT", "9099"],
        "auto_empty": True, "line_gap": 7.5, "fontsize": 5.5,
        "label": {"op": "added", "element_type": "instrument", "identity": "PT-9099",
                  "field_diffs": []},
    },
]


def _find_span(page: fitz.Page, text: str) -> dict | None:
    """Locate the text span containing `text`, returning its exact baseline origin, font
    size and full span text — so a replacement can be re-inserted at the identical
    baseline (otherwise line reconstruction splits it and reads the wrong neighbour)."""
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if text in s["text"]:
                    return s
    return None


def _empty_point(page: fitz.Page, w: float = 64, h: float = 24) -> tuple[float, float]:
    """Find a genuinely empty spot on the page so an 'added' element doesn't collide
    with (and get mis-associated with) existing text."""
    for y in range(120, 780, 18):
        for x in range(60, 1120, 48):
            if page.get_textbox(fitz.Rect(x, y, x + w, y + h)).strip() == "":
                # double-check a slightly larger halo is clear too
                if page.get_textbox(fitz.Rect(x - 10, y - 10, x + w + 10, y + h + 10)).strip() == "":
                    return float(x), float(y)
    return 70.0, 803.0


def _apply(page: fitz.Page, edit: dict) -> None:
    kind = edit["kind"]
    if kind in ("replace", "delete", "move"):
        span = _find_span(page, edit["find"])
        if not span:
            raise RuntimeError(f"synth: target not found in Rev A: {edit['find']!r}")
        r = fitz.Rect(span["bbox"])  # clear the WHOLE span; we re-insert its full text
        ox, oy = span["origin"]
        fs = span["size"]
        page.add_redact_annot(r, fill=(1, 1, 1))
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        if kind == "replace":
            # replace just the changed token within the span's full text, insert whole
            # span text at the original baseline so grouping is preserved.
            new_text = span["text"].replace(edit["find"], edit["replace"])
            page.insert_text((ox, oy), new_text, fontsize=fs, color=(0, 0, 0))
        elif kind == "move":
            page.insert_text((ox + edit["dx"], oy + edit["dy"]), span["text"],
                             fontsize=fs, color=(0, 0, 0))
        # delete => nothing re-inserted
    elif kind == "add":
        x, y = edit.get("at") or _empty_point(page)
        if edit.get("auto_empty", True):
            x, y = _empty_point(page)
        for i, tok in enumerate(edit["tokens"]):
            page.insert_text((x, y + i * edit["line_gap"]), tok,
                             fontsize=edit["fontsize"], color=(0, 0, 0))
        edit["_placed_at"] = (x, y)
    else:
        raise ValueError(f"unknown edit kind {kind}")


def build_pair1() -> Path:
    out_dir = settings.samples_dir / "pair1"
    out_dir.mkdir(parents=True, exist_ok=True)
    revA = out_dir / "revA.pdf"
    revB = out_dir / "revB.pdf"

    # Rev A = the source, copied in verbatim (self-contained repo).
    src = fitz.open(SOURCE)
    src.save(revA)
    src.close()

    doc = fitz.open(SOURCE)
    page = doc[0]
    for edit in EDITS:
        _apply(page, edit)
    doc.save(revB, garbage=4, deflate=True)
    doc.close()

    expected = {
        "pair": "pair1",
        "pid_a": str(revA),
        "pid_b": str(revB),
        "rev_a": "A", "rev_b": "B",
        "changes": [e["label"] for e in EDITS],
    }
    (out_dir / "expected_delta.json").write_text(json.dumps(expected, indent=2))

    (out_dir / "provenance.md").write_text(
        "# pair1 — native/native, controlled edits\n\n"
        "- **Rev A**: `revA.pdf` — verbatim copy of the provided *Lift Gas compressor* "
        "P&ID (AutoCAD Plant 3D plot, born-digital).\n"
        "- **Rev B**: `revB.pdf` — Rev A with 6 authored edits applied via PyMuPDF "
        "(`src/synth.py`).\n"
        "- **Ground truth**: `expected_delta.json` — the 6 edits, labelled, used by "
        "`make eval`.\n\n"
        "Edits: line size 10\"→8\" (VF-43-9025); line spec AC21S→GC11S (VF-43-9029); "
        "setpoint HH 245→300; valve 43BL9070 removed; valve 26BL9073 moved +34pt; "
        "instrument PT-9099 added.\n"
    )
    return out_dir


def build_stress() -> Path:
    """The two provided (different) drawings as a large-delta sanity pair. Not labelled —
    it only demonstrates the engine doesn't explode on a big, real diff."""
    out_dir = settings.samples_dir / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(settings.samples_dir / "_source" / "lift_gas.pdf", out_dir / "revA.pdf")
    shutil.copy(settings.samples_dir / "_source" / "export_gas.pdf", out_dir / "revB.pdf")
    (out_dir / "provenance.md").write_text(
        "# stress — two different real drawings (unlabelled)\n\n"
        "Rev A = Lift Gas compressor P&ID, Rev B = Export Gas compressor P&ID. Different "
        "drawings, so the 'delta' is huge — used only to check the pipeline scales and "
        "stays sane on a large diff, not for scored eval.\n"
    )
    return out_dir


def build_all() -> list[Path]:
    settings.ensure_dirs()
    out = [build_pair1(), build_stress()]
    try:
        from .synth_dwg import build_pair3
        out.append(build_pair3())
    except Exception:
        pass
    try:
        from .synth_scan import build_pair2
        out.append(build_pair2())
    except Exception:
        pass
    return out


if __name__ == "__main__":
    for p in build_all():
        print("built", p)
