"""FormatAdapter interface — the one seam every input format plugs into.

A PID is a persistent identifier that resolves to raw bytes + a format hint. An adapter
sniffs whether it can handle those bytes and, if so, normalizes them into a CanonicalDoc.
The delta engine and chat layer depend ONLY on CanonicalDoc, so adding a 4th format means
writing one adapter and registering it — nothing downstream changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from ..canonical.model import CanonicalDoc


@runtime_checkable
class FormatAdapter(Protocol):
    name: str

    def sniff(self, raw: bytes, hint: Optional[str]) -> bool:
        """Cheap check: can this adapter handle these bytes? `hint` is usually the file
        extension or MIME type. Must not raise."""
        ...

    def to_canonical(self, pid: str, raw: bytes, rev_label: Optional[str] = None) -> CanonicalDoc:
        """Normalize raw bytes into the canonical representation. May raise; the caller
        wraps this in a traced span so failures are visible, not swallowed."""
        ...


def resolve_pid(pid: str) -> tuple[bytes, str]:
    """Resolve a PID to (bytes, format_hint). Here a PID is a filesystem path; in a real
    system this would hit a document store / DMS. Kept behind a function so the resolver
    is swappable without touching adapters."""
    p = Path(pid)
    if not p.exists():
        raise FileNotFoundError(f"PID does not resolve to a file: {pid}")
    return p.read_bytes(), p.suffix.lower().lstrip(".")
