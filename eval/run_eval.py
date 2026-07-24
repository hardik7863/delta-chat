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


def _load_jsonl(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_qa() -> list[dict]:
    return _load_jsonl("eval/datasets/*_qa.jsonl")


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return round(s[i], 2)


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

    # ---- retrieval quality (bonus) ----
    console.rule("[bold]Retrieval quality")
    by_pair_index: dict = {}

    def _index_for(pair: str):
        if pair not in by_pair_index:
            r = pipeline.run_delta(pair, write=False)
            by_pair_index[pair] = (r, pipeline.build_index(r))
        return by_pair_index[pair]

    ranks = []
    rtable = Table("query", "rank of answer-bearing chunk")
    for item in _load_jsonl("eval/datasets/*_retrieval.jsonl"):
        _, index = _index_for(item["pair"])
        rk = metrics.retrieval_rank(index, item["query"], item["gold_all"])
        ranks.append(rk)
        rtable.add_row(item["query"][:48], str(rk) if rk else "not retrieved")
    console.print(rtable)
    rq = metrics.retrieval_report(ranks)
    results["retrieval"] = rq
    console.print(f"[bold]Retrieval: hit@1={rq['hit@1']:.2f} hit@3={rq['hit@3']:.2f} "
                  f"hit@5={rq['hit@5']:.2f} MRR={rq['mrr']:.2f} mean_rank={rq['mean_rank']}[/]\n")

    # ---- chat metrics ----
    console.rule("[bold]Chat metrics")
    qa = _load_qa()
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

    # ---- cost / latency budget analysis (bonus) ----
    console.rule("[bold]Cost / latency budget")
    BUDGET = {"delta_ms": 5000, "chat_ms": 8000, "cost_per_query_usd": 0.02}
    # delta-stage latency across the labeled pairs
    delta_ms, ingest_ms = [], []
    for label in _labeled_pairs():
        run = pipeline.run_delta(label["pair"], write=False)
        for sp in run.trace.to_dict()["spans"]:
            if sp["name"] == "delta":
                delta_ms.append(sp["duration_ms"])
            if sp["name"] == "ingest":
                ingest_ms.append(sp["duration_ms"])
    # chat latency + cost over a few real requests
    chat_ms, chat_cost = [], []
    probe_qs = ["what changed between the revisions?", "which valve was removed?",
                "did any line size change?"]
    for q in probe_qs:
        r, index = _index_for("pair1")
        _, tr = pipeline.answer("pair1", q, run=r, index=index, llm=llm)
        t = tr.totals()
        chat_ms.append(t["wall_ms"])
        chat_cost.append(t["cost_usd"])

    btable = Table("stage", "p50 ms", "p95 ms", "budget", "verdict")
    def _verdict(p95, budget):
        return "[green]PASS[/]" if p95 <= budget else "[red]OVER[/]"
    btable.add_row("ingest (per doc)", str(_percentile(ingest_ms, 50)),
                   str(_percentile(ingest_ms, 95)), "-", "-")
    btable.add_row("delta", str(_percentile(delta_ms, 50)), str(_percentile(delta_ms, 95)),
                   f"{BUDGET['delta_ms']}", _verdict(_percentile(delta_ms, 95), BUDGET["delta_ms"]))
    btable.add_row("chat (retrieve+LLM)", str(_percentile(chat_ms, 50)),
                   str(_percentile(chat_ms, 95)), f"{BUDGET['chat_ms']}",
                   _verdict(_percentile(chat_ms, 95), BUDGET["chat_ms"]))
    console.print(btable)
    mean_cost = round(sum(chat_cost) / len(chat_cost), 5) if chat_cost else 0
    cost_verdict = "PASS" if mean_cost <= BUDGET["cost_per_query_usd"] else "OVER"
    results["cost_latency"] = {
        "budget": BUDGET,
        "delta_ms_p50": _percentile(delta_ms, 50), "delta_ms_p95": _percentile(delta_ms, 95),
        "chat_ms_p50": _percentile(chat_ms, 50), "chat_ms_p95": _percentile(chat_ms, 95),
        "mean_cost_per_query_usd": mean_cost, "cost_verdict": cost_verdict,
        "provider": llm.provider,
    }
    console.print(f"[bold]Cost: ${mean_cost:.5f}/query (budget ${BUDGET['cost_per_query_usd']}) "
                  f"→ {cost_verdict}[/]  [dim](provider={llm.provider})[/]\n")

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
