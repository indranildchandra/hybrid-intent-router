.PHONY: help setup setup-cpu run run-cpu check check-cpu test test-unit test-live test-live-cpu test-installer calibration lint clean clean-all

PY_GPU := .venv-gpu/bin/python
PY_CPU := .venv-cpu/bin/python
PY     := $(if $(wildcard $(PY_GPU)),$(PY_GPU),$(if $(wildcard $(PY_CPU)),$(PY_CPU),python3))
export PYTHONPATH := $(CURDIR)/src

help:
	@echo "hybrid-intent-router: targets"
	@echo ""
	@echo "  Setup (one time; idempotent)"
	@echo "    make setup         GPU (default, falls back to CPU): deps, Ollama, models, CLM; preflight"
	@echo "    make setup-cpu     Same, forced CPU"
	@echo ""
	@echo "  Run"
	@echo "    make run           Route the Appendix B demo set (GPU, CPU fallback)"
	@echo "    make run-cpu       Route the Appendix B demo set on forced CPU"
	@echo "    make calibration   ECE + threshold sweep on the article's synthetic holdout"
	@echo ""
	@echo "  Verify"
	@echo "    make check         Preflight doctor via run.sh (starts/stops Ollama only if needed)"
	@echo "    make test          Every suite; the live suite skips unless Ollama is already up"
	@echo "    make test-unit     Offline unit tests: no GPU, Ollama or Hugging Face"
	@echo "    make test-live     Unit + live suites via run.sh (make test-live-cpu for forced CPU)"
	@echo "    make test-installer  run.sh end to end against a stand-in Ollama and Laya (Linux)"
	@echo "    make lint          shellcheck run.sh (if installed)"
	@echo ""
	@echo "    make clean         Remove caches and Ollama logs"
	@echo "    make clean-all     clean + both venvs and the downloaded CLM encoder"
	@echo ""
	@echo "  Anything more specific: ./run.sh --help"

setup:
	./run.sh --setup-only

setup-cpu:
	./run.sh --cpu --setup-only

run:
	./run.sh --no-setup

run-cpu:
	./run.sh --cpu --no-setup

check:
	./run.sh --no-setup --setup-only

check-cpu:
	./run.sh --cpu --no-setup --setup-only

test:
	$(PY) -m pytest

test-unit:
	$(PY) -m pytest -m unit

test-live:
	./run.sh --no-setup --setup-only --test

test-live-cpu:
	./run.sh --cpu --no-setup --setup-only --test

test-installer:
	bash tests/installer/run_scenarios.sh

calibration:
	$(PY) -m hybrid_intent_router.calibration

lint:
	@command -v shellcheck >/dev/null && shellcheck -S warning run.sh || echo "shellcheck not installed; skipping"

clean:
	rm -rf .pytest_cache .run catboost_info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-all: clean
	rm -rf .venv-gpu .venv-cpu models
