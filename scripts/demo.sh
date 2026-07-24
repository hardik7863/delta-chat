#!/usr/bin/env bash
# Narrated end-to-end walkthrough for the 2-4 min demo recording.
# Usage:  bash scripts/demo.sh        (uses your .env; anthropic if key set, else echo)
set -e
pause(){ echo; read -rp "  ↵ press enter…" _; echo; }
say(){ printf "\n\033[1;36m# %s\033[0m\n" "$*"; }

say "Delta-Chat — document-revision delta + grounded chat (P&IDs)"
say "1) Compute the structured delta for pair1 (native PDF, 6 authored edits)"
pause
python -m src.cli run --pair pair1

say "2) Grounded chat — cited answer + off-domain refusal (real Anthropic if key set)"
pause
python -m src.cli chat --pair pair1 --q "Which valve was removed, and did anything move?"
python -m src.cli chat --pair pair1 --q "What is the capital of France?"

say "3) Delta markup overlay (bonus) — annotated PDF of the changes"
pause
python -m src.cli markup --pair pair1
echo "  -> open data/samples/pair1/out/delta_markup.pdf"

say "4) Eval scorecard — delta P/R/F1 across native + scanned + DWG, validated judge"
pause
python -m eval.run_eval

say "5) Observability — one per-request trace (ingest -> delta -> retrieve -> llm -> answer)"
pause
ls -t runs/*.json | head -1 | xargs -I{} sh -c 'echo "trace: {}"; python -m json.tool "{}" | head -40'

say "Done. README.md = design/trade-offs/cuts; DEMO.md = this walkthrough in text."
