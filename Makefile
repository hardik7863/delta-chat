# Delta-Chat — one-command entry points. Assumes an activated venv (see README).
PY ?= python
PAIR ?= pair1

.PHONY: help install run report chat markup eval serve test synth clean

help:
	@echo "make install   - install deps into current environment"
	@echo "make synth      - (re)generate synthetic revision pairs + ground truth"
	@echo "make run PAIR=pair1   - ingest a pair -> canonical -> delta -> report artifacts"
	@echo "make report PAIR=pair1 - alias for run"
	@echo "make chat PAIR=pair1  - interactive grounded chat over a pair + its delta report"
	@echo "make markup PAIR=pair1 - render delta overlay (annotated PDF) [bonus]"
	@echo "make eval        - run the eval harness, print a scorecard"
	@echo "make serve       - launch FastAPI (/chat, /metrics, /healthz)"
	@echo "make test        - run unit tests"

install:
	$(PY) -m pip install -e .

synth:
	$(PY) -m src.cli synth

run report:
	$(PY) -m src.cli run --pair $(PAIR)

chat:
	$(PY) -m src.cli chat --pair $(PAIR)

markup:
	$(PY) -m src.cli markup --pair $(PAIR)

eval:
	$(PY) -m eval.run_eval

serve:
	$(PY) -m uvicorn src.app:api --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

clean:
	rm -rf runs/*.json data/samples/*/out eval/results/*.json
