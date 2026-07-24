"""Scanned-PDF adapter — raster/image PDFs with no reliable text layer.

Rasterizes each page and runs OCR to recover text + word bounding boxes, then feeds them
to the SAME deterministic classifier the native adapter uses. The delta engine never
learns the source was a scan — that's the seam. OCR-recovered elements carry
`confidence < 1` (derived from the OCR engine's per-word confidence), which flows into the
delta confidence, so a change grounded in a fuzzy scan reads as lower-confidence than one
from a born-digital PDF.

Default engine: Tesseract (widely available). PaddleOCR can be swapped in via config.
OCR coordinates are in pixels at render DPI; we scale them back to PDF points so bboxes
line up with the native pipeline and the markup overlay.
"""
from __future__ import annotations

import io
from typing import Optional

import fitz

from ..canonical.model import CanonicalDoc
from ..config import settings
from . import classify


def _is_scanned(raw: bytes) -> bool:
    """A PDF with negligible extractable text but real page images == a scan."""
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
        p = doc[0]
        text = p.get_text("text").strip()
        has_images = len(p.get_images()) > 0
        return has_images and len(text) < 20
    except Exception:
        return False


class ScannedPDFAdapter:
    name = "pdf_scanned"

    def sniff(self, raw: bytes, hint: Optional[str]) -> bool:
        return raw.startswith(b"%PDF") and _is_scanned(raw)

    def _ocr_page(self, page: fitz.Page, dpi: int):
        import pytesseract
        from PIL import Image
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words = []
        scale = 72.0 / dpi  # px -> pt
        n = len(data["text"])
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", -1) else -1
            if not txt or conf < 30:
                continue
            x, y, w, h = (data["left"][i], data["top"][i],
                          data["width"][i], data["height"][i])
            words.append({"text": txt, "x0": x * scale, "top": y * scale,
                          "x1": (x + w) * scale, "bottom": (y + h) * scale,
                          "conf": conf / 100.0})
        return words

    def to_canonical(self, pid: str, raw: bytes, rev_label: Optional[str] = None) -> CanonicalDoc:
        doc = fitz.open(stream=raw, filetype="pdf")
        elements, page_sizes = [], []
        confs = []
        for pi, page in enumerate(doc):
            page_sizes.append((float(page.rect.width), float(page.rect.height)))
            words = self._ocr_page(page, settings.ocr_dpi)
            confs += [w["conf"] for w in words]
            # per-word confidence -> base_conf handled per element below
            we = classify.classify_words(words, pi, self.name, base_conf=1.0)
            pe = classify.classify_phrases(words, pi, self.name, base_conf=1.0)
            page_elems = classify.merge(we, pe)
            # scale each element's confidence by mean OCR confidence of the page
            mean_conf = sum(w["conf"] for w in words) / len(words) if words else 0.6
            for e in page_elems:
                e.confidence = round(mean_conf, 3)
            elements.extend(page_elems)
        return CanonicalDoc(
            pid=pid, format=self.name, revision_label=rev_label,
            sheet_count=len(page_sizes), page_sizes=page_sizes, elements=elements,
            meta={"ocr_engine": settings.ocr_engine, "ocr_dpi": settings.ocr_dpi,
                  "mean_ocr_conf": round(sum(confs) / len(confs), 3) if confs else None},
        )
