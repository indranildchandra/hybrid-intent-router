.PHONY: help setup setup-cpu run run-cpu check check-cpu test test-unit test-live calibration lint clean clean-all

PY_GPU := .venv-gpu/bin/python
PY_CPU := .venv-cpu/bin/python
PY     := $(if $(wildcard $(PY_GPU)),$(PY_GPU),$(if $(wildcard $(PY_CPU)),$(PY_CPU),python3))
export PYTHONPATH := $(CURDIR)/src

help:
	@echo "hybrid-intent-router: targets"
	@echo ""
	@echo "  Setup (one time; idempotent)"
	@echo "    make setup         GPU: install deps, Ollama, models, CLM encoder; run preflight"
	@echo "    make setup-cpu     Same, CPU only (CPU torch wheel, private CPU Ollama on :11435)"
	@echo ""
	@echo "  Run"
	@echo "    make run           Route the Appendix B demo set on GPU"
	@echo "    make run-cpu       Route the Appendix B demo set on CPU"
	@echo "    make calibration   ECE + threshold sweep on the article's synthetic holdout"
	@echo ""
	@echo "  Verify"
	@echo "    make check         Preflight doctor against the running Ollama"
	@echo "    make test          Every suite this machine can run (live suite auto-skips)"
	@echo "    make test-unit     Offline unit tests: no GPU, Ollama or Hugging Face"
	@echo "    make test-live     The real cascade against Ollama and Laya"
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
	$(PY) -m hybrid_intent_router.doctor

check-cpu:
	HIR_OLLAMA_URL=http://127.0.0.1:11435 HIR_DEVICE=cpu $(PY) -m hybrid_intent_router.doctor

test:
	$(PY) -m pytest

test-unit:
	$(PY) -m pytest -m unit

test-live:
	$(PY) -m pytest -m live

calibration:
	$(PY) -m hybrid_intent_router.calibration

lint:
	@command -v shellcheck >/dev/null && shellcheck -S warning run.sh || echo "shellcheck not installed; skipping"

clean:
	rm -rf .pytest_cache .run catboost_info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-all: clean
	rm -rf .venv-gpu .venv-cpu models
