"""Delta report rendering: one delta, three artifacts.

  delta.json  — machine-parseable, the canonical Delta (source of truth for eval + chat)
  delta.md    — human-readable, grouped, with a counts summary and per-change locations
  delta.html  — same content as HTML for quick browser viewing

Every change is addressable as `DELTA:<change_id>`, so the report is a first-class
retrievable source for the chat layer (a change is one retrieval chunk).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..canonical.model import Delta

_OP_EMOJI = {"added": "➕", "removed": "➖", "modified": "✎"}


def _loc(c) -> str:
    x0, y0, x1, y1 = (round(v) for v in c.bbox)
    return f"sheet {c.page + 1} · bbox({x0},{y0},{x1},{y1})"


def _fields(c) -> str:
    if not c.field_diffs:
        return ""
    return " — " + "; ".join(f"`{d['field']}` {d['before']} → {d['after']}"
                             for d in c.field_diffs)


def to_markdown(delta: Delta) -> str:
    s = delta.summary
    lines = [
        f"# Delta Report — {Path(delta.pid_a).name} (Rev {delta.rev_a}) → "
        f"{Path(delta.pid_b).name} (Rev {delta.rev_b})",
        "",
        "## Summary",
        "",
        f"- **Total changes:** {s['total']}",
        f"- **Added:** {s['by_op']['added']} · **Removed:** {s['by_op']['removed']} · "
        f"**Modified:** {s['by_op']['modified']}",
        "",
        "| Element type | Changes |",
        "| --- | --- |",
    ]
    for t, n in sorted(s["by_type"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {t} | {n} |")
    lines.append("")

    # group by sheet then element type
    by_sheet: dict[int, list] = {}
    for c in delta.changes:
        by_sheet.setdefault(c.page, []).append(c)

    for page in sorted(by_sheet):
        lines.append(f"## Sheet {page + 1}")
        lines.append("")
        for c in sorted(by_sheet[page], key=lambda c: (c.element_type, c.op, c.tag or "")):
            emoji = _OP_EMOJI.get(c.op, "")
            head = f"- {emoji} **{c.op}**"
            if c.subtype:
                head += f"/{c.subtype}"
            head += f" · {c.element_type} · `{c.tag or c.description}`"
            head += _fields(c)
            head += f"  \n  ↳ {_loc(c)} · confidence {c.confidence:.2f} · `DELTA:{c.id}`"
            if c.evidence.get("shift_pts"):
                head += f" · moved {c.evidence['shift_pts']}pt"
            lines.append(head)
        lines.append("")
    return "\n".join(lines)


def to_html(delta: Delta) -> str:
    import html
    md = to_markdown(delta)
    body = html.escape(md)
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Delta Report</title>"
        "<style>body{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:900px;"
        "margin:2rem auto;padding:0 1rem}pre{white-space:pre-wrap}</style>"
        f"<pre>{body}</pre>"
    )


def write_report(delta: Delta, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "delta.json",
        "md": out_dir / "delta.md",
        "html": out_dir / "delta.html",
    }
    paths["json"].write_text(delta.model_dump_json(indent=2))
    paths["md"].write_text(to_markdown(delta))
    paths["html"].write_text(to_html(delta))
    return paths
