"""Adapter registry + the single ingest entry point.

`ingest()` resolves a PID, sniffs the format across registered adapters, and normalizes.
Adapters are tried in registration order; the first that sniffs True wins. Scanned-PDF
and DWG adapters register themselves lazily (their heavy deps are optional) so the core
runs without them installed — but the seam is always present.
"""
from __future__ import annotations

from typing import Optional

from ..canonical.model import CanonicalDoc
from ..observability.tracing import Trace
from .base import FormatAdapter, resolve_pid
from .pdf_native import NativePDFAdapter


def _build_registry() -> list[FormatAdapter]:
    adapters: list[FormatAdapter] = [NativePDFAdapter()]
    # Optional adapters — import lazily; absence must not break the core.
    try:
        from .pdf_scanned import ScannedPDFAdapter
        adapters.append(ScannedPDFAdapter())
    except Exception:
        pass
    try:
        from .dwg import DWGAdapter
        adapters.append(DWGAdapter())
    except Exception:
        pass
    return adapters


REGISTRY = _build_registry()


def ingest(pid: str, rev_label: Optional[str] = None, trace: Optional[Trace] = None,
           force_format: Optional[str] = None) -> CanonicalDoc:
    """Resolve + normalize a PID into a CanonicalDoc. Wrapped in a span when a trace is
    supplied so ingestion timing and failures are observable."""
    def _run() -> CanonicalDoc:
        raw, hint = resolve_pid(pid)
        chosen: Optional[FormatAdapter] = None
        if force_format:
            chosen = next((a for a in REGISTRY if a.name == force_format), None)
        if chosen is None:
            for a in REGISTRY:
                try:
                    if a.sniff(raw, hint):
                        chosen = a
                        break
                except Exception:
                    continue
        if chosen is None:
            raise ValueError(f"No adapter can handle PID {pid} (hint={hint})")
        doc = chosen.to_canonical(pid, raw, rev_label)
        return doc

    if trace is None:
        return _run()
    with trace.span("ingest", pid=pid, rev=rev_label) as sp:
        doc = _run()
        sp.attributes.update(format=doc.format, elements=len(doc.elements),
                             sheets=doc.sheet_count)
        return doc
