"""Command-line interface. One documented command runs ingest → delta → report; another
opens a grounded chat. Uses Typer + Rich for a readable terminal experience.

    python -m src.cli synth                 # (re)generate sample pairs + ground truth
    python -m src.cli run --pair pair1      # ingest -> delta -> report artifacts
    python -m src.cli chat --pair pair1     # interactive grounded chat
    python -m src.cli markup --pair pair1   # delta overlay (bonus)
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from . import pipeline
from .chat.llm import LLMClient
from .config import settings

app = typer.Typer(add_completion=False, help="Document Delta & Grounded Chat")
console = Console()


@app.command()
def synth():
    """(Re)generate synthetic revision pairs + ground truth."""
    from .synth import build_all
    for p in build_all():
        console.print(f"[green]built[/] {p}")


@app.command()
def run(pair: str = typer.Option("pair1", help="sample pair name under data/samples/")):
    """Ingest a pair, compute the delta, and write the delta report."""
    r = pipeline.run_delta(pair, write=True)
    s = r.delta.summary
    console.print(f"\n[bold]Delta {pair}[/] — {s['total']} changes "
                  f"([green]+{s['by_op']['added']}[/] / [red]-{s['by_op']['removed']}[/] / "
                  f"[yellow]~{s['by_op']['modified']}[/])")
    t = Table("op", "subtype", "type", "tag", "change", "conf")
    for c in r.delta.changes:
        fd = "; ".join(f"{d['field']} {d['before']}→{d['after']}" for d in c.field_diffs)
        t.add_row(c.op, c.subtype or "", c.element_type, str(c.tag or "")[:24],
                  fd or (c.evidence.get("shift_pts") and f"moved {c.evidence['shift_pts']}pt" or ""),
                  f"{c.confidence:.2f}")
    console.print(t)
    console.print(f"\nreport: [cyan]{r.report_paths.get('md')}[/]")
    console.print(f"trace:  [cyan]{settings.runs_dir}/{r.trace.request_id}.json[/]")


@app.command()
def chat(pair: str = typer.Option("pair1"),
         question: str = typer.Option(None, "--q", help="one-shot question; omit for REPL")):
    """Grounded chat over both PIDs + the delta report (citations + refusal)."""
    run = pipeline.run_delta(pair, write=True)
    index = pipeline.build_index(run)
    llm = LLMClient()
    console.print(f"[dim]provider={llm.provider} model={llm.model} · "
                  f"{len(index.chunks)} chunks indexed. Ctrl-C to exit.[/]\n")

    def ask(q: str):
        ans, _ = pipeline.answer(pair, q, run=run, index=index, llm=llm)
        tag = "[red]REFUSED[/] " if ans.refused else ""
        console.print(f"{tag}[bold]{ans.text}[/]")
        if ans.citations:
            console.print(f"[dim]citations: {', '.join(ans.citations)}[/]")
        retr = ", ".join("{}({})".format(r["cite"], r["score"]) for r in ans.retrieved[:4])
        console.print(f"[dim]retrieved: {retr}[/]\n")

    if question:
        ask(question)
        return
    try:
        while True:
            q = console.input("[cyan]you› [/]").strip()
            if q:
                ask(q)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]bye[/]")


@app.command()
def markup(pair: str = typer.Option("pair1")):
    """Render the delta as an annotated PDF overlay (bonus)."""
    from .markup.overlay import render_markup
    run = pipeline.run_delta(pair, write=True)
    try:
        out = render_markup(run)
        console.print(f"[green]markup written[/] {out}")
    except ValueError as e:
        console.print(f"[yellow]skipped:[/] {e}")


if __name__ == "__main__":
    app()
