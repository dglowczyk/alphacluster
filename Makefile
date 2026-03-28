VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup download-data train evaluate evaluate-model tournament test lint format clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

download-data:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

evaluate-model:
	$(PYTHON) scripts/evaluate.py --model-path $(MODEL_PATH)

tournament:
	$(PYTHON) scripts/train.py --tournament

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(VENV)/bin/ruff check src/ tests/
	$(VENV)/bin/ruff format --check src/ tests/

format:
	$(VENV)/bin/ruff check --fix src/ tests/
	$(VENV)/bin/ruff format src/ tests/

signals:
	@ssh $(HOST) "journalctl --user -u opencluster-monitor --no-pager -n 5000" \
		| grep -E "SIGNAL|EXECUTED|CLOSED|AUTO-TUNE" \
		| tail -n $(or $(N),50)

clean:
	rm -rf $(VENV) data/ models/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
