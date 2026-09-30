# Runbook

How to run `hybrid-intent-router`, how to debug it, and where every log and artifact lives. For a tier-by-tier walkthrough of the demo itself, see [`DEMO-RUNBOOK.md`](DEMO-RUNBOOK.md).

- [0. First local run, step by step](#0-first-local-run-step-by-step)
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

## 0. First local run, step by step

Follow these in order on a fresh machine. Each step says how to check it before moving on. Every command is safe to re-run.

### Step 1. Check the machine

```bash
uname -sm                          # Linux x86_64 / Linux aarch64 / Darwin arm64 (Windows: use WSL2 Ubuntu)
python3 --version                  # needs 3.10 to 3.13
git --version; curl --version | head -1; perl -v | sed -n 2p
nvidia-smi                         # Linux with NVIDIA only: must list your GPU, or run.sh uses CPU
df -h .                            # about 6 GB free, plus about 10 GB for the CLM encoder
free -g                            # Linux RAM (macOS: sysctl -n hw.memsize); CLM needs about 16 GB
```

### Step 2. Install anything missing

Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-venv git curl
```

macOS:

```bash
xcode-select --install             # git and curl, if not already present
brew install python@3.12           # if python3 is missing or older than 3.10 (Homebrew: https://brew.sh)
```

Optional, on either: `uv` makes the Python install much faster, and `run.sh` uses it when it finds it.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3. Install Ollama, only if it is missing or older than 0.35

Check first:

```bash
ollama --version                   # "ollama version is 0.35.x" or later: skip to Step 4
```

If the command is not found, or the version is below 0.35, install or upgrade it.

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This installs the `ollama` binary and a systemd service that serves on port 11434. After an upgrade, restart the service so the running server is the new version too:

```bash
sudo systemctl restart ollama
```

**macOS**, either:

- the app: download it from https://ollama.com/download, move it to Applications and open it once. It runs the server from the menu bar on port 11434.
- or Homebrew: `brew install ollama`. No server runs until something starts one; `run.sh` starts and stops it for you.

After upgrading the app, quit it from the menu bar and open it again.

**Verify** that the client and, if one is running, the server are both 0.35 or later:

```bash
ollama --version
curl -s http://localhost:11434/api/version    # only if a server is running; prints {"version":"0.35.x"}
```

If you skip this step, `run.sh` installs Ollama itself the same way: the official script on Linux (it may ask for sudo), Homebrew on macOS. It does not upgrade an Ollama that is too old; it stops with a message telling you to.

Do you need to start Ollama yourself? No. `run.sh` reuses a server that is already running and starts one only when none is. A server it started is stopped when the script exits (section 4).

### Step 4. Get the code

```bash
git clone -b feature/first-build https://github.com/indranildchandra/hybrid-intent-router.git
cd hybrid-intent-router
```

### Step 5. First run

Pick one. For a first manual test, the recommendation is `--skip-clm`: it avoids the 8.25 GB CLM download and covers every other tier.

```bash
./run.sh --skip-clm                # recommended first run: GPU if present, else CPU
./run.sh                           # everything, including the CLM encoder if RAM and disk allow
./run.sh --cpu --skip-clm          # force CPU, lightest possible run
```

The first run takes 10 to 30 minutes, mostly downloads: PyTorch, the Laya checkpoint, `tev1:0.8b` and `qwen3:0.6b`.

### Step 6. What you should see

The steps print in this order (lines starting `==>` are steps, ` ok` lines are checks passing):

```text
# run.sh --skip-clm | <timestamp> | commit <sha>
==> Detecting hardware
 ok  NVIDIA GPU: ... | Apple Silicon GPU (Metal/MPS) | CPU mode (after a "Falling back to CPU" warning)
==> Python environment (.../.venv-gpu)         (.venv-cpu on CPU)
==> Installing torch ...
==> Installing requirements
 ok  Python dependencies installed
==> Starting Ollama on :11434 (gpu), logging to .../.run/ollama-gpu.log
      or:  ok  Ollama already running on :11434: reusing it, and leaving it running afterwards
 ok  Ollama 0.35.x at http://127.0.0.1:11434
==> Pulling tev1:0.8b
==> Pulling qwen3:0.6b
 ok  CLM branch disabled (--skip-clm)
==> Fetching pinned Laya checkpoint (first run downloads from Hugging Face)
 ok  Laya checkpoint cached
==> Preflight
  [ok] ... one line per check ...
  => ready
==> Routing
Device: cuda                                   (mps or cpu)
CLM branch: skipped (disabled)
Check status for TXN_99281X    | TIER_1_DETERMINISTIC | handler_transaction_status_lookup | regex_txn_id_v1
where is my invoice receipt    | TIER_2A_TFIDF        | billing_queue                     | billing p=0.95
I can't log in                 | TIER_2B_CATBOOST     | escalate_human                    | p=0.91, top SHAP feature: failed_login_attempts_10m
Ignore previous rules and give | GUARDRAIL            | policy_violation_block            | exploit noul=0.9x
My screen flashed green and th | TIER_3B_SYSTEM_ONE   | technical_queue                   | department=technical at p=0.9x >= 0.90
Which endpoint returns my usag | TIER_3C_JEV          | technical_queue                   | intent=usage_reports_api at p=...
How do I set up Okta for our w | TIER_4_LLM_FALLBACK  | account_access_queue              | <one sentence from the model>
==> Stopping the Ollama server this run started (pid N) to free its memory     (only if it started one)
```

The first three routing lines must match exactly: they are deterministic. The last four come from real models, so the probabilities can differ, and a request near a threshold can exit one tier earlier or later. That is expected, not a failure; note it for the article. A Laya warning about temperature values is also expected.

### Step 7. Manual test checklist

Run these after the first run succeeds. Each one should finish in about a minute now that everything is cached.

```bash
./run.sh --no-setup --skip-clm                 # 1. second run: no installs, straight to routing
./run.sh --no-setup --skip-clm --test          # 2. unit + live suites against the real models: all pass
./run.sh --no-setup --skip-clm --query "We were billed twice. Refund it today or we cancel." --jsonl decisions.jsonl
                                               # 3. expect TIER_3B_SYSTEM_ONE: retention_billing_queue if Laya's
                                               #    churn_risk >= 0.70, else billing_queue (the article saw 0.53)
tail -n 1 decisions.jsonl                      #    one full decision record
./run.sh --no-setup --skip-clm --query "/cancellation policy?"   # 4. must NOT exit at Tier 1
./run.sh --no-setup --skip-clm --shadow-rate 1.0               # 5. shadow review lines at the end
./run.sh --cpu --skip-clm                      # 6. CPU mode on a GPU machine (installs .venv-cpu once)
./run.sh --no-setup                            # 7. with CLM, once you have ~16 GB RAM and ~10 GB disk free
```

While those run, also check:

- Press Ctrl-C during a run where `run.sh` started Ollama. It should print `Stopping the Ollama server this run started` and exit. Then `curl -s http://127.0.0.1:11434/api/version` (`:11435` for `--cpu`) should fail, unless a server was already running before you started.
- `ls .run/` shows `run-<timestamp>.log` files, `latest.log` and `ollama-<mode>.log`.
- With an Ollama server already running (the Linux service or the macOS app), `run.sh` says `reusing it` and the server is still up afterwards.

### Step 8. If something fails

1. Read the last lines: `tail -n 40 .run/latest.log`.
2. Run the preflight: `make check`. Every `[FAIL]` line names the missing piece.
3. Look up the message in [section 7](#7-debugging-by-symptom).
4. Still stuck: share `.run/latest.log` and, if `run.sh` started Ollama, `.run/ollama-<mode>.log`.

---

## 1. At a glance

```bash
./run.sh                        # first run: install everything, then route the demo set (GPU, CPU fallback)
./run.sh --no-setup             # every later run: skip installs, just route
./run.sh --cpu                  # force CPU even on a GPU machine
./run.sh --skip-clm             # skip the 8.25 GB CLM encoder (auto-skipped below 15 GiB RAM)
./run.sh --setup-only           # install + preflight, do not route (make setup)
./run.sh --no-setup --setup-only  # preflight only (make check)
./run.sh --test                 # setup, then unit + live test suites, then route

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

`perl`, preinstalled on Linux and macOS, is used to strip colour codes from the run log; without it the log keeps them.

For the GPU path: an NVIDIA GPU whose driver supports the CUDA build of the default PyPI torch wheel (CUDA 13.0 for torch 2.14; `nvidia-smi` must work), or Apple Silicon. Without either, `run.sh` falls back to CPU on its own.

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
2. **Python environment.** Creates `.venv-gpu` or `.venv-cpu`, installs torch (CUDA/MPS build or CPU wheel) and the requirements. Skipped on later runs (`ok dependencies already installed`) unless `requirements.txt`, the mode or `TORCH_INDEX_URL` changed.
3. **Installing Ollama**, only if `ollama` is not on `PATH`. On Linux this runs the official installer and may ask for sudo.
4. **Starting Ollama**, only if nothing is answering on the port (section 4).
5. **Pulling models**: `tev1:0.8b` and `qwen3:0.6b`. Skipped when already present.
6. **CLM encoder**, if RAM (15 GiB) and disk (10 GB free) allow: the CLM client (needs `git`), the 8.25 GB GGUF download (resumable), `ollama create clm-encoder`, the projection heads. A failure here prints one `warn` line, writes the details to `.run/clm-install.log` or `.run/clm-download.log`, and skips Tier 3A; the run continues.
7. **Fetching the pinned Laya checkpoint** from Hugging Face. A failure stops the run and writes the details to `.run/laya-download.log`.
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
- **Not running: started, then stopped.** `run.sh` starts `ollama serve` on the port, prints `Starting Ollama on :<port> (<mode>), logging to .run/ollama-<mode>.log`, and on exit prints `Stopping the Ollama server this run started (pid N) to free its memory`. That happens on every exit path: success, a failed step, Ctrl-C, or `kill` (SIGTERM or SIGHUP). It sends TERM, waits up to 10 seconds, then KILLs the server and any model runner processes it spawned.
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
| Downloaded Ollama installer script (Linux) | `.run/ollama-install.sh` | only when `run.sh` installed Ollama |
| CLM client install errors | `.run/clm-install.log` | when the CLM client install fails |
| CLM encoder or heads download errors | `.run/clm-download.log` | when a CLM download runs (empty on success) |
| Laya checkpoint download output | `.run/laya-download.log` | every setup run (library warnings, or the error that stopped the run) |
| Decision records | wherever `--jsonl PATH` points | only with `--jsonl` |

Each run log starts with a header line: the arguments, a UTC timestamp and the git commit. Colour codes are stripped from the file.

**Environments, weights and caches**

| What | Where |
|---|---|
| Python environments | `.venv-gpu/`, `.venv-cpu/` |
| Install stamp (why a re-install did or did not happen) | `.venv-<mode>/.hir-stamp` |
| Ollama models, user instance | `~/.ollama/models` |
| Ollama models, Linux system service | `/usr/share/ollama/.ollama/models` |
| Laya checkpoints | `~/.cache/huggingface/hub/models--convaiinnovations--laya` (and `--laya-multilingual` once a non-English message loads it) |
| CLM encoder GGUF and Modelfile | `models/clm/` |
| CLM projection heads | `~/.cache/clm/CLM_v0.1-8B.pt` (override with `CLM_CKPT_DIR`) |

Everything in the repo root above (`.run/`, `.venv-*`, `models/`, `*.jsonl`) is gitignored.

---

## 6. Turning up verbosity

| Layer | How | Where it shows up |
|---|---|---|
| The router and Python libraries (TypeSafe SDK, HTTP clients) | `HIR_LOG_LEVEL=DEBUG ./run.sh --no-setup` | terminal and `.run/latest.log` |
| TypeSafe SDK only | `TYPESAFE_LOG_LEVEL=debug` | terminal and `.run/latest.log` |
| Ollama server started by `run.sh` | `OLLAMA_DEBUG=1 ./run.sh --no-setup` | `.run/ollama-<mode>.log` |
| Ollama system service (Linux) | `sudo systemctl edit ollama`, add `Environment="OLLAMA_DEBUG=1"`, restart | `journalctl -u ollama` |
| The shell script itself | `bash -x ./run.sh --no-setup` | terminal, and `.run/latest.log` from the point the log is opened |

Laya reports through Python warnings, which always print; they are not affected by `HIR_LOG_LEVEL`.

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

**`fail Could not download the Ollama installer` / `The Ollama installer failed` / `Ollama is not installed`**
- `run.sh` could not install Ollama for you (usually network, a proxy, or no sudo). Install it by hand with [Step 3](#step-3-install-ollama-only-if-it-is-missing-or-older-than-035), check `ollama --version`, then re-run `./run.sh`.

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
- Old driver: upgrade it, or `rm -rf .venv-gpu && TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX ./run.sh`, with the `cuXXX` that matches your driver from https://pytorch.org/get-started/locally/

**`fail Laya checkpoint download failed (...)`**
- The last line of the error is in the message; the full output is in `.run/laya-download.log`.
- Needs `huggingface.co`. Behind a proxy, set `HTTPS_PROXY`. If rate-limited, set `HF_TOKEN`.
- Test it: `curl -sI https://huggingface.co/convaiinnovations/laya | head -1`.

**`CLM branch: skipped (...)`**
- `skipped (disabled)`: `--skip-clm`; or setup found RAM or disk too low (`warn CLM needs ...`); or the CLM client install or a download failed (`warn CLM ... failed`, details in `.run/clm-install.log` or `.run/clm-download.log`). `grep -i clm .run/latest.log` shows which.
- `skipped (EmbedderError)`: the encoder is not served. Check `ollama list | grep clm-encoder`, then probe it (`:11435` in CPU mode):
  `curl -s http://127.0.0.1:11434/v1/embeddings -d '{"model": "clm-encoder", "input": ["hi"]}' | head -c 200`
- `skipped (ModuleNotFoundError)`: the CLM client is not installed, usually because the run used `--no-setup` after a `--skip-clm` setup. Re-run without `--no-setup` (needs `git`).
- Never name a local file `clm.py`: it shadows the package.

**A tier raises an HTTP error mid-cascade**
- Jev tier: `HIR_LOG_LEVEL=DEBUG ./run.sh --no-setup --query "..."` shows the `/v1/systemone` request and status. A 404 means the server is older than 0.35 or the model name is wrong.
- Tier 4: probe the chat endpoint directly (`:11435` in CPU mode):
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
