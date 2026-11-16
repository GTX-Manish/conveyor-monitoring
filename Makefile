# Conveyor monitoring. `make demo` is the one command a reviewer needs.
PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup demo-data demo data features train score frontend dev test lint clean

# Finds Python 3.12 in this order: $(PY) (python3.12) on the PATH, else uv (which can fetch 3.12), else stops with a
# message saying what to install. It never falls back to another Python version.
setup:  ## create the virtual environment on Python 3.12 and install pinned packages
	@set -e; \
	if command -v $(PY) >/dev/null 2>&1; then \
		echo "make setup: using $$(command -v $(PY))"; \
		$(PY) -m venv $(VENV); \
		INSTALL="$(BIN)/python -m pip install -r requirements.txt"; \
	elif command -v uv >/dev/null 2>&1; then \
		echo "make setup: $(PY) not found on the PATH; using uv to create a Python 3.12 environment"; \
		uv venv --python 3.12 $(VENV); \
		INSTALL="uv pip install --python $(BIN)/python -r requirements.txt"; \
	else \
		echo "make setup: Python 3.12 is required, and neither $(PY) nor uv was found on the PATH." >&2; \
		echo "  Install Python 3.12 (https://www.python.org/downloads/ or: sudo apt install python3.12 python3.12-venv)," >&2; \
		echo "  or install uv (https://docs.astral.sh/uv/getting-started/installation/), then run make setup again." >&2; \
		exit 1; \
	fi; \
	$(BIN)/python -c "import sys; v = sys.version_info[:2]; sys.exit(0 if v == (3, 12) else 'make setup: .venv uses Python %d.%d, but 3.12 is required. Delete .venv and run make setup again.' % v)"; \
	$$INSTALL; \
	echo "make setup: done ($$($(BIN)/python --version))"

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
