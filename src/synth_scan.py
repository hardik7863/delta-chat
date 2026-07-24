"""Synthesize a scanned pair (pair2): BOTH revisions rasterized to noisy image-only PDFs.

To fairly demonstrate the OCR adapter, pair2 uses a *legible* synthetic P&ID (large tags,
generous spacing) rather than the dense A3 originals — Tesseract recovers large text
reliably, so the delta reflects the adapter working, not OCR falling over. (Dense A3
P&IDs OCR poorly; that's documented as a known limitation in the README, with high-DPI
tiling / a vision model as the production fix.) Both revisions are scanned so OCR noise is
symmetric — unchanged tags cancel and the authored edits surface.
"""
from __future__ import annotations

import io
import json
import re

import fitz
import numpy as np
from PIL import Image

from .config import settings

# Legible synthetic P&ID: (text, x, y) at 13pt on an A4-landscape page.
TAGS_A = [
    ('10"-VF-43-9025-AS20S-00', 60, 80),
    ('6"-PV-26-9044-GC11S-38', 60, 130),
    ('2"-WC-40-9014-AC21-00', 60, 180),
    ("26-KA-902", 430, 90),
    ("26-CX-9021", 430, 140),
    ("PSV 9066A", 620, 90),
    ("PDIT 9054", 620, 150),
    ("43BL9070", 430, 220),
    ("NOTE 34", 60, 260),
]
# Rev B edits vs A: size 10"->8"; remove 43BL9070; add PT 9099.
TAGS_B = [
    ('8"-VF-43-9025-AS20S-00', 60, 80),
    ('6"-PV-26-9044-GC11S-38', 60, 130),
    ('2"-WC-40-9014-AC21-00', 60, 180),
    ("26-KA-902", 430, 90),
    ("26-CX-9021", 430, 140),
    ("PSV 9066A", 620, 90),
    ("PDIT 9054", 620, 150),
    ("NOTE 34", 60, 260),
    ("PT 9099", 620, 220),
]
EXPECTED = [
    {"op": "modified", "element_type": "line", "subtype": "attribute",
     "identity": "VF-43-9025",
     "field_diffs": [{"field": "size", "before": '10"', "after": '8"'}]},
    {"op": "removed", "element_type": "valve", "identity": "43BL9070", "field_diffs": []},
    {"op": "added", "element_type": "instrument", "identity": "PT-9099", "field_diffs": []},
]


_INSTR = re.compile(r"^([PTFLAWSZ][A-Z]{1,3})\s+(\d{4}[A-Z]?)$")


def _make_pdf(tags) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)  # A4 landscape
    for txt, x, y in tags:
        m = _INSTR.match(txt)
        if m:  # render instrument bubbles stacked (code above number), like real P&IDs
            page.insert_text((x, y), m.group(1), fontsize=13, color=(0, 0, 0))
            page.insert_text((x, y + 16), m.group(2), fontsize=13, color=(0, 0, 0))
        else:
            page.insert_text((x, y), txt, fontsize=13, color=(0, 0, 0))
    return doc.tobytes()


def _rasterize_noisy(pdf_bytes, dpi=300, rotate_deg=0.2, noise=3) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = img.rotate(rotate_deg, expand=False, fillcolor=(255, 255, 255))
    arr = np.asarray(img).astype(np.int16)
    arr += np.random.default_rng(0).integers(-noise, noise + 1, arr.shape, dtype=np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("L").convert("RGB")
    out = fitz.open()
    pg = out.new_page(width=page.rect.width, height=page.rect.height)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pg.insert_image(pg.rect, stream=buf.getvalue())
    return out.tobytes(garbage=3, deflate=True)


def build_pair2():
    out_dir = settings.samples_dir / "pair2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "revA.pdf").write_bytes(_rasterize_noisy(_make_pdf(TAGS_A), settings.ocr_dpi))
    (out_dir / "revB.pdf").write_bytes(_rasterize_noisy(_make_pdf(TAGS_B), settings.ocr_dpi))
    (out_dir / "expected_delta.json").write_text(json.dumps(
        {"pair": "pair2", "pid_a": str(out_dir / "revA.pdf"),
         "pid_b": str(out_dir / "revB.pdf"), "rev_a": "A", "rev_b": "B",
         "changes": EXPECTED}, indent=2))
    (out_dir / "provenance.md").write_text(
        "# pair2 — both revisions SCANNED (OCR adapter)\n\n"
        "- **Rev A / Rev B**: a legible synthetic P&ID rendered to PDF, then rasterized to "
        "noisy, slightly rotated, image-only PDFs (`src/synth_scan.py`) — no text layer, "
        "so both are OCR'd (Tesseract).\n"
        "- **Ground truth**: 3 edits — size 10\"→8\", valve 43BL9070 removed, PT-9099 "
        "added.\n"
        "- Dense A3 originals OCR poorly (tiny text); see README → Limitations.\n"
    )
    return out_dir


if __name__ == "__main__":
    print("built", build_pair2())
