# Conveyor monitoring. `make demo` is the one command a reviewer needs.
PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup demo-data demo data features train score frontend dev test lint clean

setup:  ## create the virtual environment on Python 3.12 and install pinned packages
	@$(PY) -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 'Python 3.12 is required, found ' + sys.version.split()[0])"
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -r requirements.txt

demo-data: $(VENV)  ## build the demo scenario from its seed (about 10 s)
	$(BIN)/python -m simulator.scenario_set --only demo_01

demo: demo-data  ## run the backend and the replayer, then open http://127.0.0.1:8000
	$(BIN)/python -m scripts.demo

data: $(VENV)  ## build the full scenario set (about 7 minutes, 1.8 GB)
	$(BIN)/python -m simulator.scenario_set

features: $(VENV)  ## build the model tables from the scenario set
	$(BIN)/python -m scripts.make_model_data

train: $(VENV)  ## train the models and tune thresholds on the validation scenarios
	$(BIN)/python -m model.train

score: $(VENV)  ## retrain over three seeds and write model/artifacts/eval_report.md
	$(BIN)/python -m scripts.score --seeds 1 2 3

frontend:  ## rebuild the dashboard into frontend/dist (needs Node 20 or newer)
	cd frontend && npm ci && npm run build

dev:  ## Vite dev server on port 5173, proxying /api to the backend on port 8000
	cd frontend && npm run dev

test: $(VENV)
	$(BIN)/python -m pytest -q

lint: $(VENV)
	$(BIN)/ruff check .

clean:
	rm -rf data/app.sqlite*

$(VENV):
	@echo "Run 'make setup' first." && exit 1
