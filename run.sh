#!/usr/bin/env bash
# hybrid-intent-router: one command from a clean machine to a routed decision.
#
#   ./run.sh            GPU (default): CUDA on Linux, Metal/MPS on Apple Silicon. Falls back to CPU
#                       automatically when the machine has no usable GPU.
#   ./run.sh --cpu      Force CPU: CPU torch wheel, private CPU-only Ollama instance on port 11435.
#   ./run.sh --skip-clm Either mode, without the 8.25 GB CLM encoder.
#
# Logs, all under .run/ at the repo root (created by this script, gitignored):
#   run-<timestamp>.log / latest.log   full output of every run (last 20 kept)
#   ollama-<mode>.log                  Ollama server log, only when this script started the server
#   clm-install.log, clm-download.log, laya-download.log   setup errors, summarised in one line
# Ollama: started only if nothing is running on the port; a server this script started is stopped
# on exit (success, failure, Ctrl-C, SIGTERM). A server that was already running is reused and left alone.
#
# Idempotent: re-running skips everything already installed, pulled or downloaded.
set -Eeuo pipefail

# ----------------------------------------------------------------------------- constants
MIN_OLLAMA="0.35.0"
JEV_MODEL="${HIR_JEV_MODEL:-tev1:0.8b}"
FALLBACK_MODEL="${HIR_FALLBACK_MODEL:-qwen3:0.6b}"
CLM_MODEL="${HIR_CLM_MODEL:-clm-encoder}"
CLM_GGUF_REPO="czl/CLM-v0.1-8B-GGUF"
CLM_GGUF_FILE="Qwen3-8B-Q8_0-outq2.gguf"
CLM_GIT="git+https://github.com/Contrastive-LM/CLM.git@bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
CLM_HEADS="${CLM_CKPT_DIR:-$HOME/.cache/clm}/CLM_v0.1-8B.pt"
CLM_MIN_RAM_GB=15   # a "16 GB" machine reports ~15.x GiB
CLM_MIN_DISK_GB=10  # 8.25 GB GGUF plus Ollama's own copy of the blob is written on create
GPU_PORT=11434
CPU_PORT=11435

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$ROOT/.run"
mkdir -p "$STATE_DIR"

# ----------------------------------------------------------------------------- output helpers
if [[ -t 1 ]]; then B=$'\033[1m'; BLUE=$'\033[1;34m'; YEL=$'\033[1;33m'; RED=$'\033[1;31m'; GRN=$'\033[1;32m'; R=$'\033[0m'
else B=""; BLUE=""; YEL=""; RED=""; GRN=""; R=""; fi
log()  { printf '%s==>%s %s\n' "$BLUE" "$R" "$*"; }
ok()   { printf '%s ok%s  %s\n' "$GRN" "$R" "$*"; }
warn() { printf '%swarn%s %s\n' "$YEL" "$R" "$*" >&2; }
die()  { printf '%sfail%s %s\n' "$RED" "$R" "$*" >&2; exit 1; }

usage() {
  cat <<EOF
${B}Usage:${R} ./run.sh [options] [-- router args]

${B}Modes${R}
  (default)       GPU: CUDA on Linux/NVIDIA, Metal/MPS on Apple Silicon, Ollama on port $GPU_PORT.
                  Falls back to CPU automatically if no GPU is found.
  --cpu           Force CPU: CPU torch wheel, Laya/CLM on CPU, private CPU-only Ollama on port $CPU_PORT.
  --gpu           Explicit GPU (same as the default).

${B}Options${R}
  --skip-clm      Do not download or serve the 8.25 GB CLM encoder (Tier 3A is skipped at runtime).
  --setup-only    Install and download everything, run the preflight doctor, then stop.
  --no-setup      Skip installation; just start Ollama if needed and run the router.
  --test          Run all test suites (unit, plus live against the Ollama this run started).
  --keep-ollama   Leave an Ollama server this script started running after exit.
                  (A server that was already running is always reused and never stopped.)
  -h, --help      Show this help.

${B}Router args${R} (anything else is passed to python -m hybrid_intent_router)
  --query TEXT    Route one message instead of the Appendix B demo set.
  --meta JSON     Metadata for that message, e.g. '{"user_tier": "Enterprise", "failed_logins": 5}'.
  --jsonl PATH    Append every full decision record to a JSONL file.
  --shadow-rate X Share of confident Tier 2/3 decisions re-checked by Tier 4 (default 0.02).

${B}Examples${R}
  ./run.sh
  ./run.sh --skip-clm
  ./run.sh --cpu
  ./run.sh --query "We were billed twice. Refund it today or we cancel." --jsonl decisions.jsonl
EOF
}

# ----------------------------------------------------------------------------- arguments
ORIG_ARGS="$*"
FORCED_CPU=0; MODE="gpu"; SKIP_CLM=0; SETUP_ONLY=0; NO_SETUP=0; RUN_TESTS=0; KEEP_OLLAMA=0; PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE="cpu"; FORCED_CPU=1 ;;
    --gpu) MODE="gpu"; FORCED_CPU=0 ;;
    --skip-clm) SKIP_CLM=1 ;;
    --setup-only) SETUP_ONLY=1 ;;
    --no-setup) NO_SETUP=1 ;;
    --test) RUN_TESTS=1 ;;
    --keep-ollama) KEEP_OLLAMA=1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; PY_ARGS+=("$@"); break ;;
    *) PY_ARGS+=("$1") ;;
  esac
  shift
done

# ----------------------------------------------------------------------------- run log
# Everything this script and the router print also lands in .run/run-<timestamp>.log, with
# .run/latest.log pointing at the newest one. tee ignores INT/TERM so that on Ctrl-C it stays
# alive long enough to record the cleanup (and the cleanup never dies on a broken pipe).
RUN_LOG="$STATE_DIR/run-$(date +%Y%m%d-%H%M%S).log"
if command -v perl >/dev/null 2>&1; then  # colours on the terminal, plain text in the file
  exec > >(trap '' INT TERM HUP; exec tee >(exec perl -pe 'BEGIN { $| = 1 } s/\e\[[0-9;]*m//g' >>"$RUN_LOG")) 2>&1
else
  exec > >(trap '' INT TERM HUP; exec tee -a "$RUN_LOG") 2>&1
fi
ln -sf "$(basename "$RUN_LOG")" "$STATE_DIR/latest.log"
# keep the 20 most recent run logs
# shellcheck disable=SC2012
ls -1t "$STATE_DIR"/run-*.log 2>/dev/null | tail -n +21 | while read -r old; do rm -f "$old"; done
echo "# run.sh $ORIG_ARGS | $(date -u +%Y-%m-%dT%H:%M:%SZ) | commit $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# ----------------------------------------------------------------------------- utilities
version_ge() {  # version_ge 0.35.1 0.35.0 -> true
  awk -v a="$1" -v b="$2" 'BEGIN { split(a, x, "."); split(b, y, ".");
    for (i = 1; i <= 3; i++) { if (x[i] + 0 > y[i] + 0) exit 0; if (x[i] + 0 < y[i] + 0) exit 1 } exit 0 }'
}
have() { command -v "$1" >/dev/null 2>&1; }
http_ok() { curl -fsS --max-time 3 "$1" >/dev/null 2>&1; }

OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
  Linux|Darwin) ;;
  MINGW*|MSYS*|CYGWIN*) die "Native Windows is not supported. Use WSL2 (Ubuntu) and run ./run.sh inside it." ;;
  *) die "Unsupported OS: $OS" ;;
esac
have curl || die "curl is required (apt install curl / brew install curl)."

ram_gb() {
  if [[ "$OS" == "Linux" ]]; then awk '/MemTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo
  else echo $(( $(sysctl -n hw.memsize) / 1073741824 )); fi
}
disk_free_gb() { df -Pk "$1" | awk 'NR == 2 { printf "%d", $4 / 1048576 }'; }

# ----------------------------------------------------------------------------- 1. hardware
log "Detecting hardware"
GPU_KIND="none"
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  GPU_KIND="mps"
elif have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  GPU_KIND="cuda"
fi
RAM_GB="$(ram_gb)"
if [[ "$MODE" == "gpu" && "$GPU_KIND" == "none" ]]; then
  warn "No NVIDIA GPU (nvidia-smi) or Apple Silicon found. Falling back to CPU (same as --cpu)."
  warn "AMD/ROCm users: Ollama will still use your GPU; Laya runs on CPU with the default torch wheel."
  MODE="cpu"
fi
if [[ "$MODE" == "gpu" ]]; then
  HIR_DEVICE="$GPU_KIND"
  if [[ "$GPU_KIND" == "cuda" ]]; then
    ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  else
    ok "Apple Silicon GPU (Metal/MPS)"
  fi
else
  HIR_DEVICE="cpu"
  ok "CPU mode"
fi
ok "RAM: ${RAM_GB} GiB  |  OS: $OS $ARCH"

# ----------------------------------------------------------------------------- 2. python
find_python() {
  local c
  for c in python3.12 python3.11 python3.13 python3.10 python3; do
    if have "$c" && "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}
VENV="$ROOT/.venv-$MODE"   # separate envs: a CPU torch wheel in a GPU env silently runs on CPU
PIP=()
PY="$VENV/bin/python"

setup_python() {
  log "Python environment ($VENV)"
  local base
  base="$(find_python)" || die "Python 3.10-3.13 is required. Install one (e.g. apt install python3.12 python3.12-venv, brew install python@3.12)."
  local torch_index="${TORCH_INDEX_URL:-}"
  if [[ -z "$torch_index" && "$OS" == "Linux" && "$MODE" == "cpu" ]]; then
    torch_index="https://download.pytorch.org/whl/cpu"  # avoids a ~2.5 GB CUDA download you will not use
  fi
  local stamp_file="$VENV/.hir-stamp"
  local stamp
  stamp="$(cat "$ROOT/requirements.txt" <(echo "$MODE|$torch_index") | cksum | awk '{print $1}')"
  if [[ -x "$PY" && -f "$stamp_file" && "$(cat "$stamp_file")" == "$stamp" ]]; then
    ok "dependencies already installed"
    return
  fi

  if have uv; then
    [[ -x "$PY" ]] || uv venv -q --python "$base" "$VENV"
    PIP=(uv pip install -q --python "$PY")
  else
    if [[ ! -x "$PY" ]]; then
      "$base" -m venv "$VENV" 2>/dev/null || die "Could not create a venv. On Debian/Ubuntu: sudo apt install python3-venv"
    fi
    "$PY" -m pip install -q --upgrade pip
    PIP=("$PY" -m pip install -q)
  fi

  log "Installing torch ($([[ -n "$torch_index" ]] && echo "$torch_index" || echo "default wheel"))"
  if [[ -n "$torch_index" ]]; then "${PIP[@]}" torch --index-url "$torch_index"; else "${PIP[@]}" torch; fi
  log "Installing requirements"
  "${PIP[@]}" -r "$ROOT/requirements.txt" pytest
  echo "$stamp" > "$stamp_file"
  ok "Python dependencies installed"
}

# ----------------------------------------------------------------------------- 3. ollama
install_ollama() {
  if have ollama; then return; fi
  log "Installing Ollama"
  local manual="install it by hand (RUNBOOK.md, section 0, Step 3), then re-run ./run.sh"
  if [[ "$OS" == "Linux" ]]; then
    warn "Running the official installer (https://ollama.com/install.sh). It may ask for sudo."
    # Download first, so a failed download stops here with a clear message instead of a curl code
    local installer="$STATE_DIR/ollama-install.sh"
    curl -fsSL https://ollama.com/install.sh -o "$installer" \
      || die "Could not download the Ollama installer from ollama.com (network or proxy?). Or $manual"
    sh "$installer" || die "The Ollama installer failed (see the lines above). Or $manual"
  elif have brew; then
    brew install ollama || die "brew install ollama failed. Download the app from https://ollama.com/download, or $manual"
  else
    die "Ollama is not installed and Homebrew is not available. Download the app from https://ollama.com/download, or $manual"
  fi
  have ollama || die "The Ollama install finished but 'ollama' is not on PATH. Open a new terminal and re-run ./run.sh, or $manual"
}

check_client_version() {
  local v
  v="$(ollama --version 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | tail -1 || true)"
  if [[ -n "$v" ]] && ! version_ge "$v" "$MIN_OLLAMA"; then
    die "Ollama client $v is too old; the /v1/systemone API needs $MIN_OLLAMA+. Upgrade: https://ollama.com/download"
  fi
}

# Ownership rule: a server this script finds already running is never touched. A server this
# script starts is stopped on every exit path (success, failure, Ctrl-C, kill), runners included.
OLLAMA_PID=""

stop_pid_tree() {  # graceful TERM, then KILL anything left, including model runner children
  local pid="$1" children=""
  have pgrep && children="$(pgrep -P "$pid" 2>/dev/null || true)"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
  local c
  for c in $children; do kill -KILL "$c" 2>/dev/null || true; done
  return 0
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$OLLAMA_PID" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    if [[ "$KEEP_OLLAMA" == "1" ]]; then
      log "Leaving the Ollama server this run started (pid $OLLAMA_PID) running (--keep-ollama). Stop it with: kill $OLLAMA_PID"
    else
      log "Stopping the Ollama server this run started (pid $OLLAMA_PID) to free its memory"
      stop_pid_tree "$OLLAMA_PID"
    fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

start_ollama() {
  local port="$1"
  local url="http://127.0.0.1:$port"
  if [[ -n "${HIR_OLLAMA_URL:-}" ]]; then
    url="${HIR_OLLAMA_URL%/}"
    http_ok "$url/api/version" || die "HIR_OLLAMA_URL=$url is not reachable."
    ok "using Ollama at $url (HIR_OLLAMA_URL): not managed by this script"
  elif http_ok "$url/api/version"; then
    ok "Ollama already running on :$port: reusing it, and leaving it running afterwards"
  else
    log "Starting Ollama on :$port ($MODE), logging to $STATE_DIR/ollama-$MODE.log"
    if [[ "$MODE" == "cpu" ]]; then
      # Hide every accelerator from this instance so decision models and the fallback run on CPU.
      # On Apple Silicon Ollama still uses Metal; Tier 4 additionally sends num_gpu=0 per request.
      OLLAMA_HOST="127.0.0.1:$port" CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
        GPU_DEVICE_ORDINAL=-1 nohup ollama serve >"$STATE_DIR/ollama-$MODE.log" 2>&1 &
    else
      OLLAMA_HOST="127.0.0.1:$port" nohup ollama serve >"$STATE_DIR/ollama-$MODE.log" 2>&1 &
    fi
    OLLAMA_PID=$!
    for _ in $(seq 1 60); do
      http_ok "$url/api/version" && break
      kill -0 "$OLLAMA_PID" 2>/dev/null || die "Ollama exited on start. See $STATE_DIR/ollama-$MODE.log (port $port in use?)"
      sleep 1
    done
    http_ok "$url/api/version" || die "Ollama did not come up in 60s. See $STATE_DIR/ollama-$MODE.log"
  fi
  local sv
  sv="$(curl -fsS "$url/api/version" | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  version_ge "$sv" "$MIN_OLLAMA" || die "Ollama server at $url is $sv; need $MIN_OLLAMA+ (a running system service may be older than your client: restart it after upgrading)."
  ok "Ollama $sv at $url"
  HIR_OLLAMA_URL="$url"
  OLLAMA_HOSTPORT="${url#http://}"; OLLAMA_HOSTPORT="${OLLAMA_HOSTPORT#https://}"
}

model_present() {  # exact tag or tag:latest in /api/tags
  local tags  # captured first: grep -q closing the pipe early would trip pipefail
  tags="$(curl -fsS "$HIR_OLLAMA_URL/api/tags")" || return 1
  grep -Eq "\"name\": *\"$1(:latest)?\"" <<<"$tags"
}

pull_models() {
  local m
  for m in "$JEV_MODEL" "$FALLBACK_MODEL"; do
    if [[ "${HIR_USE_TYPESAFE:-0}" == "1" && "$m" == "$JEV_MODEL" ]]; then continue; fi
    if model_present "$m"; then ok "model $m"; else
      log "Pulling $m"
      OLLAMA_HOST="$OLLAMA_HOSTPORT" ollama pull "$m"
    fi
  done
}

# ----------------------------------------------------------------------------- 4. CLM encoder (optional)
setup_clm() {
  if [[ "$SKIP_CLM" == "1" ]]; then
    export HIR_DISABLE_CLM=1; ok "CLM branch disabled (--skip-clm)"; return
  fi
  if (( RAM_GB < CLM_MIN_RAM_GB )); then
    export HIR_DISABLE_CLM=1
    warn "CLM needs ~16 GB RAM (found ${RAM_GB} GiB). Skipping Tier 3A; the cascade runs without it."
    return
  fi
  if ! "$PY" -c "import clm" 2>/dev/null; then
    if ! have git; then
      export HIR_DISABLE_CLM=1; warn "git not found, so the CLM client cannot be installed; skipping Tier 3A."; return
    fi
    # --no-deps: CLM lists vLLM as a dependency; the Ollama-served encoder does not need it
    log "Installing CLM client (no vLLM)"
    if have uv; then PIP=(uv pip install -q --python "$PY"); else PIP=("$PY" -m pip install -q); fi
    "${PIP[@]}" --no-deps "$CLM_GIT" 2>"$STATE_DIR/clm-install.log" \
      || { export HIR_DISABLE_CLM=1; warn "CLM client install failed; skipping Tier 3A. Details: $STATE_DIR/clm-install.log"; return; }
  fi
  if model_present "$CLM_MODEL"; then
    ok "model $CLM_MODEL"
  else
    local dir="$ROOT/models/clm"
    mkdir -p "$dir"
    if [[ ! -f "$dir/$CLM_GGUF_FILE" ]]; then
      local free; free="$(disk_free_gb "$dir")"
      if (( free < CLM_MIN_DISK_GB )); then
        export HIR_DISABLE_CLM=1
        warn "CLM needs ~${CLM_MIN_DISK_GB} GB free disk (found ${free} GB). Skipping Tier 3A."; return
      fi
      log "Downloading CLM encoder ($CLM_GGUF_REPO/$CLM_GGUF_FILE, 8.25 GB, resumable)"
      "$PY" - "$CLM_GGUF_REPO" "$CLM_GGUF_FILE" "$dir" 2>"$STATE_DIR/clm-download.log" <<'PYEOF' || { export HIR_DISABLE_CLM=1; warn "CLM download failed ($(tail -n 1 "$STATE_DIR/clm-download.log" | cut -c1-120)); skipping Tier 3A. Details: $STATE_DIR/clm-download.log"; return; }
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], local_dir=sys.argv[3])
PYEOF
    fi
    # The GGUF already declares last-token pooling, which is what lets Ollama serve it as an embedder
    echo "FROM ./$CLM_GGUF_FILE" > "$dir/Modelfile"
    log "Registering $CLM_MODEL with Ollama"
    (cd "$dir" && OLLAMA_HOST="$OLLAMA_HOSTPORT" ollama create "$CLM_MODEL" -f Modelfile) \
      || { export HIR_DISABLE_CLM=1; warn "ollama create failed; skipping Tier 3A."; return; }
  fi
  if [[ ! -f "$CLM_HEADS" ]]; then
    log "Downloading CLM projection heads into ${CLM_HEADS%/*}"
    "$VENV/bin/clm-download" >/dev/null 2>"$STATE_DIR/clm-download.log" || { export HIR_DISABLE_CLM=1; warn "clm-download failed; skipping Tier 3A. Details: $STATE_DIR/clm-download.log"; return; }
  fi
  ok "CLM projection heads"
}

# ----------------------------------------------------------------------------- 5. laya warm-up
warm_laya() {
  log "Fetching pinned Laya checkpoint (first run downloads from Hugging Face)"
  HIR_DEVICE="$HIR_DEVICE" "$PY" -c "from hybrid_intent_router.system_one import laya; laya().load('english')" 2>"$STATE_DIR/laya-download.log" \
    || die "Laya checkpoint download failed ($(tail -n 1 "$STATE_DIR/laya-download.log" | cut -c1-120)). Check access to huggingface.co (HF_TOKEN if rate-limited). Details: $STATE_DIR/laya-download.log"
  ok "Laya checkpoint cached"
}

# ----------------------------------------------------------------------------- run
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"  # code lives in src/

if [[ "$NO_SETUP" == "0" ]]; then
  setup_python
  install_ollama
  check_client_version
fi
[[ -x "$PY" ]] || die "No environment at $VENV. Run ./run.sh$([[ "$FORCED_CPU" == "1" ]] && echo " --cpu") without --no-setup first."
have ollama || die "Ollama is not installed. Run ./run.sh without --no-setup to install it, or install it by hand (RUNBOOK.md, section 0, Step 3)."

if [[ "$MODE" == "cpu" ]]; then start_ollama "$CPU_PORT"; else start_ollama "$GPU_PORT"; fi
export HIR_OLLAMA_URL HIR_DEVICE

if [[ "$NO_SETUP" == "0" ]]; then
  pull_models
  setup_clm
  warm_laya
elif [[ "$SKIP_CLM" == "1" ]]; then
  export HIR_DISABLE_CLM=1
fi

log "Preflight"
"$PY" -m hybrid_intent_router.doctor || die "Preflight failed (see above)."

if [[ "$RUN_TESTS" == "1" ]]; then
  log "Tests (unit, plus live against this Ollama)"
  "$PY" -m pytest "$ROOT/tests"
fi

if [[ "$SETUP_ONLY" == "1" ]]; then
  ok "Setup complete. Run ./run.sh$([[ "$FORCED_CPU" == "1" ]] && echo " --cpu") --no-setup to route."
  exit 0
fi

log "Routing"
"$PY" -m hybrid_intent_router ${PY_ARGS[@]+"${PY_ARGS[@]}"}
