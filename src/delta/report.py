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


_REPORT_CSS = """
:root{
  --bg:#0a0c11;--bg2:#0d1017;--surface:#12151d;--surface-2:#161a23;--raised:#1b202b;
  --line:#232a37;--line-2:#2c3444;--ink:#eef1f7;--dim:#909aad;--faint:#5c6577;
  --accent:#6aa8ff;--accent-soft:rgba(106,168,255,.12);
  --add:#3fb950;--add-soft:rgba(63,185,80,.13);--rem:#ff6b6b;--rem-soft:rgba(255,107,107,.13);
  --mod:#e3b341;--mod-soft:rgba(227,179,65,.13);
  --sans:'IBM Plex Sans',ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;font-family:var(--sans);color:var(--ink);font-size:14px;line-height:1.55;
  background:radial-gradient(1000px 480px at 82% -8%,rgba(106,168,255,.10),transparent 60%),
             linear-gradient(180deg,#0b0e14,#0a0c11 40%);min-height:100vh}
body::before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.5;
  background-image:linear-gradient(rgba(255,255,255,.022) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(255,255,255,.022) 1px,transparent 1px);
  background-size:28px 28px;mask-image:radial-gradient(1100px 640px at 50% 0%,#000 40%,transparent 92%)}
.wrap{position:relative;max-width:960px;margin:0 auto;padding:0 24px 56px}
.top{display:flex;align-items:center;gap:14px;padding:24px 2px 20px;position:sticky;top:0;
  backdrop-filter:blur(8px);background:linear-gradient(180deg,rgba(10,12,17,.9),transparent)}
.mark{width:34px;height:34px;border-radius:9px;display:grid;place-items:center;
  background:linear-gradient(150deg,#1c2432,#12151d);border:1px solid var(--line-2)}
.mark svg{width:18px;height:18px}
.top h1{margin:0;font-size:15px;font-family:var(--mono);font-weight:600;letter-spacing:.02em}
.top .rev{color:var(--dim);font-size:12px;font-family:var(--mono)}
.top .rev b{color:var(--ink);font-weight:500}
.top .spacer{flex:1}
.back{color:var(--dim);text-decoration:none;font-size:12.5px;transition:color .15s}
.back:hover{color:var(--accent)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:6px 0 26px}
.stat{background:var(--surface-2);border:1px solid var(--line);border-radius:11px;padding:14px 15px;position:relative;overflow:hidden}
.stat::after{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent)}
.stat.added::after{background:var(--add)}.stat.removed::after{background:var(--rem)}.stat.modified::after{background:var(--mod)}
.stat .n{font-family:var(--mono);font-size:27px;font-weight:600;line-height:1}
.stat.added .n{color:var(--add)}.stat.removed .n{color:var(--rem)}.stat.modified .n{color:var(--mod)}
.stat .l{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-top:6px}
h2.sheet{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);
  margin:26px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.change{display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;
  background:var(--surface-2);border:1px solid var(--line);border-radius:11px;padding:13px 15px;margin-bottom:9px}
.badge{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  padding:5px 8px;border-radius:7px;min-width:66px;text-align:center;align-self:start}
.badge.added{color:var(--add);background:var(--add-soft)}.badge.removed{color:var(--rem);background:var(--rem-soft)}
.badge.modified{color:var(--mod);background:var(--mod-soft)}
.badge small{display:block;font-size:8.5px;font-weight:500;opacity:.75}
.change .type{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);margin-bottom:2px}
.change .tag{font-family:var(--mono);font-size:12.5px;word-break:break-all}
.change .diff{margin-top:4px;font-family:var(--mono);font-size:11.5px;color:var(--dim);display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.was{color:var(--rem);text-decoration:line-through;text-decoration-color:rgba(255,107,107,.5)}
.arrow{color:var(--faint)}.now{color:var(--add)}.moved{color:var(--mod)}
.loc{font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:3px}
.conf{display:flex;flex-direction:column;align-items:flex-end;gap:5px;align-self:start}
.conf .v{font-family:var(--mono);font-size:11px;color:var(--dim)}
.conf .bar{width:54px;height:4px;border-radius:3px;background:var(--line);overflow:hidden}
.conf .bar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#9bc2ff)}
"""


def _diff_html(c) -> str:
    import html as _h
    if c.field_diffs:
        parts = []
        for d in c.field_diffs:
            b = _h.escape(str(d.get("before") if d.get("before") is not None else "∅"))
            a = _h.escape(str(d.get("after") if d.get("after") is not None else "∅"))
            parts.append(f'<span>{_h.escape(str(d["field"]))}</span> '
                         f'<span class="was">{b}</span><span class="arrow">→</span>'
                         f'<span class="now">{a}</span>')
        return '<div class="diff">' + "&nbsp;&nbsp;".join(parts) + "</div>"
    if c.evidence.get("shift_pts"):
        return f'<div class="diff"><span class="moved">moved {c.evidence["shift_pts"]}pt</span></div>'
    return ""


def to_html(delta: Delta) -> str:
    import html as _h
    s = delta.summary
    by = s["by_op"]
    mark = ("<svg viewBox='0 0 24 24' fill='none'><path d='M12 3 L20 19 L4 19 Z' "
            "stroke='#6aa8ff' stroke-width='1.6' stroke-linejoin='round'/>"
            "<path d='M12 9 L15.5 16 L8.5 16 Z' fill='#6aa8ff' fill-opacity='.85'/></svg>")

    by_sheet: dict[int, list] = {}
    for c in delta.changes:
        by_sheet.setdefault(c.page, []).append(c)

    sections = []
    for page in sorted(by_sheet):
        rows = []
        for c in sorted(by_sheet[page], key=lambda c: (c.element_type, c.op, c.tag or "")):
            pct = round(c.confidence * 100)
            sub = f"<small>{_h.escape(c.subtype)}</small>" if c.subtype else ""
            rows.append(
                f'<div class="change"><div class="badge {c.op}">{c.op}{sub}</div>'
                f'<div class="main"><div class="type">{_h.escape(c.element_type)}</div>'
                f'<div class="tag">{_h.escape(c.tag or c.description)}</div>{_diff_html(c)}'
                f'<div class="loc">sheet {c.page + 1} · DELTA:{c.id}</div></div>'
                f'<div class="conf"><span class="v">{pct}%</span>'
                f'<span class="bar"><i style="width:{pct}%"></i></span></div></div>')
        sections.append(f'<h2 class="sheet">Sheet {page + 1}</h2>' + "".join(rows))

    tiles = (
        f'<div class="stat total"><div class="n">{s["total"]}</div><div class="l">changes</div></div>'
        f'<div class="stat added"><div class="n">+{by["added"]}</div><div class="l">added</div></div>'
        f'<div class="stat removed"><div class="n">−{by["removed"]}</div><div class="l">removed</div></div>'
        f'<div class="stat modified"><div class="n">~{by["modified"]}</div><div class="l">modified</div></div>')

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Delta Report</title>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
        "&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        f"<style>{_REPORT_CSS}</style></head><body><div class='wrap'>"
        f"<div class='top'><div class='mark'>{mark}</div>"
        f"<div><h1>Delta Report</h1><div class='rev'>"
        f"<b>{_h.escape(Path(delta.pid_a).name)}</b> (Rev {delta.rev_a}) → "
        f"<b>{_h.escape(Path(delta.pid_b).name)}</b> (Rev {delta.rev_b})</div></div>"
        f"<div class='spacer'></div><a class='back' href='/'>← back to console</a></div>"
        f"<div class='stats'>{tiles}</div>{''.join(sections)}</div></body></html>"
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
