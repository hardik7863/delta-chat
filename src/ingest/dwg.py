"""DWG/DXF adapter — native CAD drawings.

DWG is a binary AutoCAD format. The real-world seam is: accept a `.dwg`, convert it to
`.dxf` with the ODA File Converter (a free, widely-used external tool), then parse the DXF
with `ezdxf`. That conversion is a single documented call behind this adapter — if ODA
isn't installed we operate directly on `.dxf` samples and log that DWG→DXF conversion was
skipped. Either way the adapter emits the SAME canonical `Element` list as the PDF
adapters, so the delta engine is format-blind.

From DXF we read TEXT/MTEXT entities (tags, notes) with their insert points, and count
LINE/LWPOLYLINE geometry (pipes) as page metadata. TEXT insert coords are drawing units
with a bottom-left origin; we flip Y so bboxes are top-left like the PDF pipeline.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..canonical.model import CanonicalDoc
from ..observability import logging as slog
from . import classify


def _dwg_to_dxf(dwg_bytes: bytes) -> Optional[bytes]:
    """Convert DWG→DXF via ODA File Converter if available. Returns None if unavailable
    (the seam is real; the tool is an external, documented dependency)."""
    oda = shutil.which("ODAFileConverter")
    if not oda:
        slog.warn("dwg.no_oda", msg="ODAFileConverter not found; expecting .dxf input")
        return None
    with tempfile.TemporaryDirectory() as ind, tempfile.TemporaryDirectory() as outd:
        (Path(ind) / "in.dwg").write_bytes(dwg_bytes)
        subprocess.run([oda, ind, outd, "ACAD2018", "DXF", "0", "1", "*.dwg"],
                       check=False, capture_output=True)
        dxfs = list(Path(outd).glob("*.dxf"))
        return dxfs[0].read_bytes() if dxfs else None


class DWGAdapter:
    name = "dwg"

    def sniff(self, raw: bytes, hint: Optional[str]) -> bool:
        if hint in ("dwg", "dxf"):
            return True
        return raw[:4] == b"AC10" or raw[:2] == b"AC" or raw.lstrip()[:2] == b"0\n"

    def to_canonical(self, pid: str, raw: bytes, rev_label: Optional[str] = None) -> CanonicalDoc:
        import ezdxf

        is_dwg = raw[:2] == b"AC"
        dxf_bytes = raw
        if is_dwg:
            converted = _dwg_to_dxf(raw)
            if converted is None:
                raise RuntimeError("DWG input requires ODAFileConverter (DWG→DXF); "
                                   "install it or supply a .dxf")
            dxf_bytes = converted

        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
            f.write(dxf_bytes)
            tmp = f.name
        doc = ezdxf.readfile(tmp)
        msp = doc.modelspace()

        # collect text entities with insert points; derive a page extent to flip Y
        raw_words = []
        max_y = 0.0
        for e in msp:
            if e.dxftype() in ("TEXT", "MTEXT"):
                txt = (e.plain_text() if e.dxftype() == "MTEXT" else e.dxf.text) or ""
                txt = txt.strip()
                if not txt:
                    continue
                ins = e.dxf.insert
                h = float(getattr(e.dxf, "height", 2.5) or 2.5)
                w = max(len(txt) * h * 0.6, h)
                raw_words.append({"_x": float(ins[0]), "_y": float(ins[1]),
                                  "w": w, "h": h, "text": txt})
                max_y = max(max_y, float(ins[1]) + h)

        n_lines = sum(1 for e in msp if e.dxftype() in ("LINE", "LWPOLYLINE", "POLYLINE"))
        n_blocks = sum(1 for e in msp if e.dxftype() == "INSERT")

        # flip Y (CAD origin is bottom-left) so bboxes match the PDF top-left convention
        words = []
        for w in raw_words:
            top = max_y - w["_y"] - w["h"]
            words.append({"text": w["text"], "x0": w["_x"], "top": top,
                          "x1": w["_x"] + w["w"], "bottom": top + w["h"]})

        we = classify.classify_words(words, 0, self.name, base_conf=1.0)
        pe = classify.classify_phrases(words, 0, self.name, base_conf=1.0)
        elements = classify.merge(we, pe)

        return CanonicalDoc(
            pid=pid, format=self.name, revision_label=rev_label, sheet_count=1,
            page_sizes=[(0.0, 0.0)], elements=elements,
            meta={"source": "dwg" if is_dwg else "dxf", "text_entities": len(raw_words),
                  "geometry": {"lines": n_lines, "blocks": n_blocks}},
        )
