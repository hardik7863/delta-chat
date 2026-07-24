"""Native (born-digital) PDF adapter.

Born-digital PDFs (e.g. AutoCAD Plant 3D plots) carry an extractable text layer with per
-token bounding boxes plus vector geometry (lines/curves = pipes and symbols). We read
words + reconstructed text-lines with pdfplumber and hand them to the deterministic
classifier. Vector line/curve counts are recorded as page metadata (used later as a
coarse geometry signal); full symbol-level vector diffing is intentionally out of scope.
"""
from __future__ import annotations

import io
import warnings
from typing import Optional

import pdfplumber

from ..canonical.model import CanonicalDoc
from . import classify

warnings.filterwarnings("ignore")


class NativePDFAdapter:
    name = "pdf_native"

    def sniff(self, raw: bytes, hint: Optional[str]) -> bool:
        if not raw.startswith(b"%PDF"):
            return False
        # native == has an extractable text layer on page 1
        try:
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                return len(pdf.pages[0].extract_words()) > 0
        except Exception:
            return False

    def to_canonical(self, pid: str, raw: bytes, rev_label: Optional[str] = None) -> CanonicalDoc:
        elements = []
        page_sizes = []
        geom = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for pi, page in enumerate(pdf.pages):
                page_sizes.append((float(page.width), float(page.height)))
                words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
                we = classify.classify_words(words, pi, self.name, base_conf=1.0)
                pe = classify.classify_phrases(words, pi, self.name, base_conf=1.0)
                elements.extend(classify.merge(we, pe))
                geom.append({"lines": len(page.lines), "curves": len(page.curves),
                             "rects": len(page.rects)})
        return CanonicalDoc(
            pid=pid, format=self.name, revision_label=rev_label,
            sheet_count=len(page_sizes), page_sizes=page_sizes, elements=elements,
            meta={"geometry": geom, "producer": "pdf_native"},
        )
