"""Delta markup overlay (bonus) — draw the computed delta back onto the revised drawing.

Renders an annotated PDF: colored boxes + redline labels on each changed region of Rev B
(added = green, removed = red drawn at the Rev-A location, modified = amber). This is the
classic manual artifact the tool replaces — a reviewer sees at a glance what moved.

Uses PyMuPDF drawing primitives; deterministic (no LLM).
"""
from __future__ import annotations

from pathlib import Path

import fitz

from ..pipeline import DeltaRun

_COLORS = {
    "added": (0.10, 0.60, 0.15),     # green
    "removed": (0.85, 0.12, 0.12),   # red
    "modified": (0.90, 0.55, 0.05),  # amber
}


def render_markup(run: DeltaRun) -> Path:
    # Overlay onto the revised drawing (Rev B) using its own resolved path.
    src = run.doc_b.pid
    out_dir = Path(src).parent
    if run.doc_b.format not in ("pdf_native", "pdf_scanned"):
        raise ValueError(f"markup overlay supports PDF drawings only; Rev B is "
                         f"'{run.doc_b.format}'. (DXF/DWG would render via a CAD export "
                         f"step — out of scope for this bonus.)")
    doc = fitz.open(src)

    for c in run.delta.changes:
        page = doc[min(c.page, len(doc) - 1)]
        x0, y0, x1, y1 = c.bbox
        rect = fitz.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2)
        color = _COLORS.get(c.op, (0, 0, 0))
        page.draw_rect(rect, color=color, width=1.2)
        label = c.op.upper()[:3]
        if c.subtype:
            label += f"/{c.subtype[:4]}"
        # small redline caption above the box
        page.insert_text((rect.x0, max(6, rect.y0 - 2.5)), label, fontsize=4.5,
                         color=color)

    # legend
    p0 = doc[0]
    ly = 12
    for op, col in _COLORS.items():
        p0.draw_rect(fitz.Rect(12, ly, 20, ly + 6), color=col, width=1.2)
        p0.insert_text((24, ly + 5), op, fontsize=6, color=col)
        ly += 10

    out = out_dir / "out" / "delta_markup.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out, garbage=3, deflate=True)
    doc.close()
    return out
