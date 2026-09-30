.PHONY: help setup setup-gpu run run-gpu check check-gpu test test-unit test-live test-live-gpu calibration lint clean clean-all

PY_GPU := .venv-gpu/bin/python
PY_CPU := .venv-cpu/bin/python
PY     := $(if $(wildcard $(PY_CPU)),$(PY_CPU),$(if $(wildcard $(PY_GPU)),$(PY_GPU),python3))
export PYTHONPATH := $(CURDIR)/src

help:
	@echo "hybrid-intent-router: targets"
	@echo ""
	@echo "  Setup (one time; idempotent)"
	@echo "    make setup         CPU (default): install deps, Ollama, models, CLM encoder; run preflight"
	@echo "    make setup-gpu     Same, on GPU (CUDA or Apple Silicon)"
	@echo ""
	@echo "  Run"
	@echo "    make run           Route the Appendix B demo set on CPU"
	@echo "    make run-gpu       Route the Appendix B demo set on GPU"
	@echo "    make calibration   ECE + threshold sweep on the article's synthetic holdout"
	@echo ""
	@echo "  Verify"
	@echo "    make check         Preflight doctor (CPU setup); make check-gpu for GPU"
	@echo "    make test          Every suite; the live suite skips unless Ollama is already up"
	@echo "    make test-unit     Offline unit tests: no GPU, Ollama or Hugging Face"
	@echo "    make test-live     Unit + live suites with Ollama started by run.sh (CPU; -gpu for GPU)"
	@echo "    make lint          shellcheck run.sh (if installed)"
	@echo ""
	@echo "    make clean         Remove caches and Ollama logs"
	@echo "    make clean-all     clean + both venvs and the downloaded CLM encoder"
	@echo ""
	@echo "  Anything more specific: ./run.sh --help"

setup:
	./run.sh --setup-only

setup-gpu:
	./run.sh --gpu --setup-only

run:
	./run.sh --no-setup

run-gpu:
	./run.sh --gpu --no-setup

check:
	HIR_OLLAMA_URL=http://127.0.0.1:11435 HIR_DEVICE=cpu $(PY_CPU) -m hybrid_intent_router.doctor

check-gpu:
	$(PY_GPU) -m hybrid_intent_router.doctor

test:
	$(PY) -m pytest

test-unit:
	$(PY) -m pytest -m unit

test-live:
	./run.sh --no-setup --setup-only --test

test-live-gpu:
	./run.sh --gpu --no-setup --setup-only --test

calibration:
	$(PY) -m hybrid_intent_router.calibration

lint:
	@command -v shellcheck >/dev/null && shellcheck -S warning run.sh || echo "shellcheck not installed; skipping"

clean:
	rm -rf .pytest_cache .run catboost_info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-all: clean
	rm -rf .venv-gpu .venv-cpu models
