"""Evaluation harness. One command prints a scorecard and writes a comparable results
file, so a change can be shown to help or hurt.

  python -m eval.run_eval            # full scorecard (delta P/R/F1 + chat + judge check)

Delta metrics run keyless (deterministic). Chat correctness uses the configured LLM as a
judge; with the offline `echo` provider it falls back to a lexical check and the scorecard
labels the mode so the numbers aren't over-trusted. A judge-validation probe reports how
well the judge agrees with hand-labeled gold verdicts.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src import pipeline
from src.chat.llm import LLMClient
from src.config import settings
from . import metrics

console = Console()

# Hand-labeled gold verdicts to validate the judge itself.
JUDGE_PROBES = [
    {"q": "Did the size change?", "facts": ["10", "8"],
     "answer": "Yes, line VF-43-9025 went from 10\" to 8\".", "gold": True},
    {"q": "Did the size change?", "facts": ["10", "8"],
     "answer": "No lines changed size in this revision.", "gold": False},
    {"q": "Which valve was removed?", "facts": ["43BL9070"],
     "answer": "Valve 43BL9070 was removed.", "gold": True},
    {"q": "Which valve was removed?", "facts": ["43BL9070"],
     "answer": "A new pump was added.", "gold": False},
]


def _labeled_pairs() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(settings.samples_dir / "*" / "expected_delta.json"))):
        out.append(json.loads(Path(f).read_text()))
    return out


def _load_qa() -> list[dict]:
    out = []
    for f in sorted(glob.glob("eval/datasets/*_qa.jsonl")):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def run() -> dict:
    llm = LLMClient()
    results = {"provider": llm.provider, "model": llm.model, "delta": {}, "chat": {},
               "judge_validation": {}}

    # ---- delta P/R/F1 on labeled pairs ----
    console.rule("[bold]Delta metrics (deterministic)")
    dtable = Table("pair", "TP", "FP", "FN", "precision", "recall", "F1", "field_acc")
    agg = {"tp": 0, "fp": 0, "fn": 0}
    for label in _labeled_pairs():
        pair = label["pair"]
        run = pipeline.run_delta(pair, write=False)
        sc = metrics.score_delta(run.delta.changes, label["changes"])
        results["delta"][pair] = sc.as_dict()
        agg["tp"] += sc.tp; agg["fp"] += sc.fp; agg["fn"] += sc.fn
        dtable.add_row(pair, str(sc.tp), str(sc.fp), str(sc.fn),
                       f"{sc.precision:.2f}", f"{sc.recall:.2f}", f"{sc.f1:.2f}",
                       f"{sc.field_accuracy:.2f}")
    console.print(dtable)
    P = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) else 0
    R = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) else 0
    F1 = 2 * P * R / (P + R) if (P + R) else 0
    results["delta"]["_overall"] = {"precision": round(P, 3), "recall": round(R, 3),
                                    "f1": round(F1, 3)}
    console.print(f"[bold]Overall delta: P={P:.2f} R={R:.2f} F1={F1:.2f}[/]\n")

    # ---- judge validation ----
    console.rule("[bold]Judge validation")
    correct = 0
    for p in JUDGE_PROBES:
        v = metrics.judge_correct(p["q"], p["facts"], p["answer"], llm)
        correct += 1 if v == p["gold"] else 0
    agree = correct / len(JUDGE_PROBES)
    results["judge_validation"] = {"agreement": round(agree, 3), "n": len(JUDGE_PROBES),
                                   "mode": "lexical" if llm.provider == "echo" else "llm"}
    console.print(f"judge agreement with gold: [bold]{agree:.0%}[/] "
                  f"({'lexical fallback' if llm.provider=='echo' else 'LLM judge'})\n")

    # ---- chat metrics ----
    console.rule("[bold]Chat metrics")
    qa = _load_qa()
    by_pair_index: dict = {}
    n = grounded = correct_ans = refusal_ok = cite_valid = cite_total = 0
    failures = []
    ctable = Table("question", "refuse?", "grounded", "correct", "cites")
    for item in qa:
        pair = item["pair"]
        if pair not in by_pair_index:
            r = pipeline.run_delta(pair, write=False)
            by_pair_index[pair] = (r, pipeline.build_index(r))
        r, index = by_pair_index[pair]
        valid_ids = {c.cite for c in index.chunks}
        from src.chat.answer import answer_question
        ans = answer_question(item["question"], index, llm)
        n += 1

        if item["expect_refusal"]:
            ok = ans.refused
            refusal_ok += 1 if ok else 0
            correct = ok
            g = "-"
        else:
            g = bool(ans.citations) and not ans.refused
            grounded += 1 if g else 0
            vc, tc = metrics.valid_citations(ans.citations, valid_ids)
            cite_valid += vc; cite_total += tc
            correct = metrics.judge_correct(item["question"], item["key_facts"], ans.text, llm)
        correct_ans += 1 if correct else 0
        if not correct:
            failures.append({"q": item["question"], "answer": ans.text[:160],
                             "refused": ans.refused, "citations": ans.citations})
        ctable.add_row(item["question"][:44], "yes" if item["expect_refusal"] else "no",
                       "-" if item["expect_refusal"] else ("✓" if g else "✗"),
                       "✓" if correct else "✗",
                       ",".join(ans.citations[:2]) or "-")
    console.print(ctable)
    results["chat"] = {
        "n": n,
        "answer_correctness": round(correct_ans / n, 3) if n else 0,
        "groundedness": round(grounded / max(1, n - refusal_ok_denom(qa)), 3),
        "citation_accuracy": round(cite_valid / cite_total, 3) if cite_total else 1.0,
        "refusal_accuracy": round(refusal_ok / max(1, sum(1 for q in qa if q["expect_refusal"])), 3),
        "failures": failures,
    }
    console.print(f"[bold]Chat: correctness={results['chat']['answer_correctness']:.2f} "
                  f"groundedness={results['chat']['groundedness']:.2f} "
                  f"citation_acc={results['chat']['citation_accuracy']:.2f} "
                  f"refusal_acc={results['chat']['refusal_accuracy']:.2f}[/]")

    # ---- candid failure table (rubric rewards honesty) ----
    if failures:
        console.rule("[bold red]Failures (candid)")
        for f in failures:
            console.print(f"[red]✗[/] {f['q']}\n   → {f['answer']}")

    return results


def refusal_ok_denom(qa: list[dict]) -> int:
    return sum(1 for q in qa if q["expect_refusal"])


def main():
    results = run()
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = settings.repo_root / "eval" / "results" / f"{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    console.print(f"\n[dim]scorecard written to {out}[/]")
    console.print("[dim]compare across runs to detect regressions.[/]")


if __name__ == "__main__":
    main()
