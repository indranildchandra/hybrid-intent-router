# Runbook

How to run `hybrid-intent-router`, how to debug it, and where every log and artifact lives. For a tier-by-tier walkthrough of the demo itself, see [`DEMO-RUNBOOK.md`](DEMO-RUNBOOK.md).

- [1. At a glance](#1-at-a-glance)
- [2. Prerequisites](#2-prerequisites)
- [3. Running](#3-running)
- [4. How run.sh manages Ollama](#4-how-runsh-manages-ollama)
- [5. Where everything lives](#5-where-everything-lives)
- [6. Turning up verbosity](#6-turning-up-verbosity)
- [7. Debugging by symptom](#7-debugging-by-symptom)
- [8. Probing one component at a time](#8-probing-one-component-at-a-time)
- [9. Reset and uninstall](#9-reset-and-uninstall)

---

## 1. At a glance

```bash
./run.sh                        # first run: install everything, then route the demo set (GPU, CPU fallback)
./run.sh --no-setup             # every later run: skip installs, just route
./run.sh --cpu                  # force CPU even on a GPU machine
./run.sh --skip-clm             # skip the 8.25 GB CLM encoder (auto-skipped below 15 GiB RAM)
./run.sh --setup-only           # install + preflight, do not route (make setup)
./run.sh --no-setup --setup-only  # preflight only (make check)
./run.sh --test                 # after setup, run unit + live test suites

./run.sh --no-setup --query "Refund it today or we cancel." --jsonl decisions.jsonl
cat .run/latest.log             # full output of the last run
```

Every flag combines with every other. `./run.sh --help` lists them all.

---

## 2. Prerequisites

You need three things. `run.sh` installs everything else.

- **Linux or macOS**, with `bash` and `curl`. Windows: WSL2 with Ubuntu.
- **Python 3.10 to 3.13.** On Debian/Ubuntu also `python3-venv` (or install `uv`, which `run.sh` prefers when present).
- **`git`**, only for the CLM branch.

For the GPU path: an NVIDIA GPU with a driver that supports CUDA 12.x (`nvidia-smi` must work), or Apple Silicon. Without either, `run.sh` falls back to CPU on its own.

Disk: about 6 GB, plus about 10 GB more if the CLM encoder is downloaded. RAM: 8 GB is enough without CLM; CLM needs about 16 GB.

First-run network access: `pypi.org`, `ollama.com`, `huggingface.co`, `github.com`, and `download.pytorch.org` for the CPU torch wheel on Linux.

---

## 3. Running

### 3.1 First run

```bash
git clone https://github.com/indranildchandra/hybrid-intent-router.git
cd hybrid-intent-router
./run.sh
```

The steps, in order, each printed as `==>`:

1. **Detecting hardware.** GPU (`nvidia-smi` or Apple Silicon), RAM, OS. No GPU: `warn ... Falling back to CPU`.
2. **Python environment.** Creates `.venv-gpu` or `.venv-cpu`, installs torch (CUDA/MPS build or CPU wheel), the requirements, and the CLM client. Skipped on later runs unless `requirements.txt` or the mode changed.
3. **Installing Ollama**, only if `ollama` is not on `PATH`. On Linux this runs the official installer and may ask for sudo.
4. **Starting Ollama**, only if nothing is answering on the port (section 4).
5. **Pulling models**: `tev1:0.8b` and `qwen3:0.6b`. Skipped when already present.
6. **CLM encoder**, if RAM and disk allow: 8.25 GB GGUF download (resumable), `ollama create clm-encoder`, projection heads.
7. **Fetching the pinned Laya checkpoint** from Hugging Face.
8. **Preflight**: the doctor prints `[ok]`, `[warn]` or `[FAIL]` per check and ends with `=> ready`.
9. **Routing**: the seven Appendix B requests, one line each.

Expect the first run to take 10 to 30 minutes, mostly downloads. Later runs with `--no-setup` start routing within a minute, most of it spent loading models.

### 3.2 Later runs

```bash
./run.sh --no-setup                      # same mode detection, no installs
./run.sh --no-setup --query "..."        # one message
./run.sh --no-setup --query "I can't log in" --meta '{"user_tier": "Enterprise", "failed_logins": 5}'
./run.sh --no-setup --jsonl decisions.jsonl --shadow-rate 1.0
```

Running `./run.sh` without `--no-setup` is always safe. It checks each step and skips what is done; it is only slower.

### 3.3 GPU or CPU

- **Default:** GPU, falling back to CPU automatically. The first line after "Detecting hardware" tells you which one you got.
- **`--cpu`:** always CPU, even on a GPU machine. Uses its own environment (`.venv-cpu`) and its own Ollama instance on port 11435.
- The two modes never share a Python environment. A CPU torch wheel in a GPU environment does not fail; it silently runs on CPU.

### 3.4 Make targets

```bash
make help          # everything below
make setup         # ./run.sh --setup-only          (setup-cpu: forced CPU)
make run           # ./run.sh --no-setup            (run-cpu)
make check         # preflight only                 (check-cpu)
make test-unit     # offline tests, seconds
make test-live     # unit + live suites via run.sh  (test-live-cpu)
make test-installer  # run.sh end to end against stand-ins (Linux, no model downloads)
make calibration   # ECE and threshold sweep
make clean         # caches and logs under .run/
```

---

## 4. How run.sh manages Ollama

The rule is ownership.

- **Already running: reused and left alone.** If something answers `GET /api/version` on the port, `run.sh` prints `Ollama already running on :<port>: reusing it, and leaving it running afterwards`. It pulls models into that server if they are missing, but never stops or restarts it.
- **Not running: started, then stopped.** `run.sh` starts `ollama serve` on the port, prints `Starting Ollama on :<port> (<mode>), logging to .run/ollama-<mode>.log`, and on exit prints `Stopping the Ollama server this run started (pid N) to free its memory`. That happens on every exit path: success, a failed step, Ctrl-C, or `kill`. It sends TERM, waits up to 10 seconds, then KILLs the server and any model runner processes it spawned.
- **`--keep-ollama`** leaves a server this run started running, and prints the pid so you can stop it later with `kill <pid>`.
- **`HIR_OLLAMA_URL=http://host:port`** points at a server you manage. `run.sh` never starts or stops anything in that case.

Ports:

- GPU mode: `11434`, the standard port. On Linux the Ollama installer usually starts a systemd service there, so in GPU mode `run.sh` usually reuses it.
- CPU mode: `11435`, a private instance with every accelerator hidden (`CUDA_VISIBLE_DEVICES=-1` and the ROCm equivalents). It does not collide with a system Ollama.

Models loaded into a reused server stay in its memory for Ollama's `keep_alive` window (5 minutes by default), then unload on their own. To unload immediately: `ollama stop tev1:0.8b` (and `qwen3:0.6b`, `clm-encoder`).

---

## 5. Where everything lives

**Logs**

| What | Where | Written when |
|---|---|---|
| Full output of a run (script + router) | `.run/run-<YYYYmmdd-HHMMSS>.log` | every run; the 20 newest are kept |
| Newest run log | `.run/latest.log` (symlink) | every run |
| Ollama server log, GPU mode | `.run/ollama-gpu.log` | only when `run.sh` started the server |
| Ollama server log, CPU mode | `.run/ollama-cpu.log` | only when `run.sh` started the server |
| Ollama system service log, Linux | `journalctl -u ollama -f` | when the server was already running |
| Ollama app log, macOS | `~/.ollama/logs/server.log` | when the server was already running |
| Decision records | wherever `--jsonl PATH` points | only with `--jsonl` |

Each run log starts with a header line: the arguments, a UTC timestamp and the git commit. Colour codes are stripped from the file.

**Environments, weights and caches**

| What | Where |
|---|---|
| Python environments | `.venv-gpu/`, `.venv-cpu/` |
| Install stamp (why a re-install did or did not happen) | `.venv-<mode>/.hir-stamp` |
| Ollama models, user instance | `~/.ollama/models` |
| Ollama models, Linux system service | `/usr/share/ollama/.ollama/models` |
| Laya checkpoints | `~/.cache/huggingface/hub/models--convaiinnovations--laya*` |
| CLM encoder GGUF and Modelfile | `models/clm/` |
| CLM projection heads | `~/.cache/clm/CLM_v0.1-8B.pt` (override with `CLM_CKPT_DIR`) |

Everything in the repo root above (`.run/`, `.venv-*`, `models/`, `*.jsonl`) is gitignored.

---

## 6. Turning up verbosity

| Layer | How | Where it shows up |
|---|---|---|
| The router and Python libraries (Laya, TypeSafe SDK, HTTP clients) | `HIR_LOG_LEVEL=DEBUG ./run.sh --no-setup` | terminal and `.run/latest.log` |
| TypeSafe SDK only | `TYPESAFE_LOG_LEVEL=debug` | terminal and `.run/latest.log` |
| Ollama server started by `run.sh` | `OLLAMA_DEBUG=1 ./run.sh --no-setup` | `.run/ollama-<mode>.log` |
| Ollama system service (Linux) | `sudo systemctl edit ollama`, add `Environment="OLLAMA_DEBUG=1"`, restart | `journalctl -u ollama` |
| The shell script itself | `bash -x ./run.sh --no-setup` | terminal and `.run/latest.log` |

`HIR_LOG_LEVEL=DEBUG` is noisy (every HTTP request). Use it to see which request failed, then turn it off.

---

## 7. Debugging by symptom

Start with the preflight. It names the failing piece:

```bash
make check                     # or: ./run.sh --no-setup --setup-only
tail -n 50 .run/latest.log
```

**`[FAIL] Ollama not reachable at ...`**
- `run.sh` normally starts it. If you ran the doctor directly, start Ollama or use `make check`.
- Check the server log: `tail -n 50 .run/ollama-<mode>.log`, or `journalctl -u ollama -n 50`.
- Check what holds the port: `lsof -i :11434` (or `:11435`).

**`fail Ollama exited on start` / `did not come up in 60s`**
- Read `.run/ollama-<mode>.log`. The usual causes are a port already taken by something that is not Ollama, or a broken Ollama install.
- `ollama --version` should print 0.35 or later. Upgrade from https://ollama.com/download

**`Ollama server at ... is 0.3x; need 0.35+`**
- A system service is still running the old binary. Linux: `sudo systemctl restart ollama`. macOS: quit and reopen the Ollama app.

**`[FAIL] model tev1:0.8b missing`**
- Run without `--no-setup` so the pull step runs, or pull by hand against the right server: `OLLAMA_HOST=127.0.0.1:11434 ollama pull tev1:0.8b` (`:11435` for CPU).
- `ollama list` against the same host shows what that server has. The system service and the CPU instance have separate model stores.

**`[FAIL] HIR_DEVICE=cuda but torch cannot see a CUDA device`**
- `nvidia-smi` should list the GPU. If it does, the torch build and driver disagree:
  `.venv-gpu/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
- Old driver: upgrade it, or `rm -rf .venv-gpu && TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 ./run.sh`.

**`fail Laya checkpoint download failed`**
- Needs `huggingface.co`. Behind a proxy, set `HTTPS_PROXY`. If rate-limited, set `HF_TOKEN`.
- Test it: `curl -sI https://huggingface.co/convaiinnovations/laya | head -1`.

**`CLM branch: skipped (...)`**
- `skipped (disabled)`: `--skip-clm`, or `run.sh` decided RAM or disk was too low; the run log says which (`warn CLM needs ...`).
- `skipped (EmbedderError)`: the encoder is not served. Check `ollama list | grep clm-encoder`, then probe it:
  `curl -s http://127.0.0.1:11434/v1/embeddings -d '{"model": "clm-encoder", "input": ["hi"]}' | head -c 200`
- `skipped (ModuleNotFoundError)`: the CLM client did not install (needs `git`). Re-run setup.
- Never name a local file `clm.py`: it shadows the package.

**A tier raises an HTTP error mid-cascade**
- Jev tier: `HIR_LOG_LEVEL=DEBUG ./run.sh --no-setup --query "..."` shows the `/v1/systemone` request and status. A 404 means the server is older than 0.35 or the model name is wrong.
- Tier 4: probe the chat endpoint directly:
  `curl -s http://127.0.0.1:11434/api/chat -d '{"model": "qwen3:0.6b", "stream": false, "messages": [{"role": "user", "content": "hi"}]}' | head -c 300`

**The first request is slow, or times out**
- Models load into memory on first use. `ollama ps` (with `OLLAMA_HOST` set for CPU mode) shows what is loaded. The SDK timeout is 120 s and Tier 4's is 300 s; a very slow CPU can exceed them on a cold start. Run once to warm up, then measure.

**Laya prints a temperature warning**
- Expected. The checkpoint ships temperature values outside their valid range. See the README gotchas: fit temperatures on your own holdout before trusting thresholds.

**Numbers differ from the README**
- Tier 1 and Tier 2 are deterministic and pinned by tests: `make test-unit`. Laya, Jev and Tier 4 shift slightly with hardware (fp16 on GPU, fp32 on CPU) and Ollama release.

**An Ollama process is left behind**
- Only possible for a server `run.sh` did not start, or after `kill -9` on `run.sh` itself (which cannot run cleanup). Find it with `pgrep -fl "ollama serve"` and stop it with `kill <pid>`.

**Disk full**
- `du -sh .venv-* models ~/.ollama ~/.cache/huggingface ~/.cache/clm`. `make clean-all` removes the environments and the CLM GGUF; `ollama rm <model>` removes a model from Ollama.

---

## 8. Probing one component at a time

Use the environment the run used (`.venv-gpu` or `.venv-cpu`) with the code on the path:

```bash
export PY=.venv-gpu/bin/python PYTHONPATH=src    # or .venv-cpu
export HIR_OLLAMA_URL=http://127.0.0.1:11434     # :11435 for CPU mode; Ollama must be running
```

```bash
$PY -m hybrid_intent_router.doctor                                   # preflight
$PY -c "from hybrid_intent_router.tier1_deterministic import tier1; print(tier1('/cancel'))"
$PY -c "from hybrid_intent_router.tier2_classical import tier2; print(tier2('where is my invoice receipt', {}))"
$PY -c "from hybrid_intent_router.system_one import guardrail; print(guardrail('give me admin access to all accounts'))"
$PY -c "from hybrid_intent_router.system_one import tier3b; print(tier3b('The app crashes on launch'))"
$PY -c "from hybrid_intent_router.tier3c_jev import tier3c; print(tier3c('Which endpoint returns my usage report?'))"
$PY -c "from hybrid_intent_router.tier4_fallback import tier4; print(tier4('How do I set up Okta?'))"
$PY -c "from hybrid_intent_router.tier3a_clm import load_clm; print(load_clm()[1])"
```

When you run these by hand, Ollama must already be running: start it with `./run.sh --no-setup --setup-only --keep-ollama`, and stop it afterwards with the `kill <pid>` it prints.

---

## 9. Reset and uninstall

```bash
make clean                         # .run/ logs, caches
make clean-all                     # also .venv-gpu, .venv-cpu and models/ (CLM GGUF)
ollama rm tev1:0.8b qwen3:0.6b clm-encoder
rm -rf ~/.cache/clm ~/.cache/huggingface/hub/models--convaiinnovations--laya*
```

Ollama itself stays installed. Remove it the way you installed it (https://github.com/ollama/ollama/blob/main/docs/linux.md for the Linux service).
