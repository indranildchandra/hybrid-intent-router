.PHONY: lock help setup setup-cpu run run-cpu check check-cpu test test-unit test-live test-live-cpu test-installer calibration lint clean clean-all

PY_GPU := .venv-gpu/bin/python
PY_CPU := .venv-cpu/bin/python
PY     := $(if $(wildcard $(PY_GPU)),$(PY_GPU),$(if $(wildcard $(PY_CPU)),$(PY_CPU),python3))
# Prepend src and keep any existing path. The comment sits on its own line: make would keep the
# spaces before an inline comment as part of the value.
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

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
	@echo "    make clean         Remove caches and every log under .run/"
	@echo "    make clean-all     clean + both venvs and the downloaded CLM encoder"
	@echo ""
	@echo "  Dependencies"
	@echo "    make lock          Regenerate the pinned requirements.txt from pyproject.toml (needs uv)"
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

# torch and its GPU-specific dependencies stay out of the lock: run.sh installs the torch build
# that matches the mode (TORCH_VERSION in run.sh), and each build brings its own GPU libraries.
TORCH_GPU_DEPS := cuda-bindings cuda-pathfinder cuda-toolkit nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc \
	nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
	nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx triton
lock:
	@command -v uv >/dev/null || { echo "make lock needs uv: https://docs.astral.sh/uv/"; exit 1; }
	@head -n 8 requirements.txt > .requirements.header
	echo "torch==$$(sed -n 's/^TORCH_VERSION="$${HIR_TORCH_VERSION:-\(.*\)}"/\1/p' run.sh)" > .torch-constraint.txt
	uv pip compile pyproject.toml --extra dev --universal --python-version 3.10 --no-header \
		-c .torch-constraint.txt --no-emit-package torch $(addprefix --no-emit-package ,$(TORCH_GPU_DEPS)) \
		-o .requirements.body
	cat .requirements.header .requirements.body > requirements.txt
	@rm -f .requirements.header .requirements.body .torch-constraint.txt

clean:
	rm -rf .pytest_cache .run catboost_info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-all: clean
	rm -rf .venv-gpu .venv-cpu models
