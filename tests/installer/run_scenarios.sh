#!/usr/bin/env bash
# Installer scenarios: drives the real run.sh end to end against a stand-in Ollama (HTTP server
# with the same endpoints) and a stand-in Laya, so no model downloads are needed. Checks install
# idempotency, the Ollama ownership rules (reuse vs start/stop, Ctrl-C, SIGTERM, failures), logs,
# pull retries, argument pass-through, error paths and the make targets. Prints PASS/FAIL per check.
#
#   bash tests/installer/run_scenarios.sh     (Linux; needs setsid, pgrep/pkill and script)
#
# It uses the real .venv-cpu and .run/ of this checkout, and ports 11435, 11777 and 11999. A CLM
# encoder already in models/clm/ skips the download S1 expects to fail: move it aside first.
# Set TORCH_INDEX_URL if download.pytorch.org is blocked on your network.
# shellcheck disable=SC2034  # A/B/R are read inside the eval'd check strings
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; cd "$REPO" || exit 1
S="$(mktemp -d)"; mkdir -p "$S/stubstate"
export PATH="$HERE/fakebin:$PATH" PYTHONPATH="$HERE/fakepy" STUB_STATE="$S/stubstate"
# The CLM encoder download must fail, fast and the same way everywhere: a revision that does not
# exist answers 404 (or a network error offline), never an 8.25 GB download on a CI runner.
export HIR_CLM_GGUF_REVISION="hir-installer-scenarios-no-such-revision"
PASS=0; FAIL=0
# check NAME CONDITION [OUTPUT]: on FAIL, the warn/error/CLM lines of OUTPUT show what run.sh said instead
check() {
  if eval "$2"; then echo "PASS  $1"; PASS=$((PASS+1)); return; fi
  echo "FAIL  $1"; FAIL=$((FAIL+1))
  [[ -n "${3:-}" && -f "$3" ]] && grep -iE "warn|error|fail|clm" "$3" | tail -n 8 | sed 's/^/      | /'
  return 0
}
up() { curl -fsS --max-time 2 "http://127.0.0.1:${1:-11435}/api/version" >/dev/null 2>&1; }
ours() { pgrep -f "stub_ollama_server.py" >/dev/null; }
reset_models() { rm -f "$STUB_STATE"/*; touch "$STUB_STATE/tev1__0.8b" "$STUB_STATE/qwen3__0.6b"; }
reset_models; pkill -f "^sleep 1000$"

echo "--- S1 idempotent setup"
./run.sh --setup-only --skip-clm > "$S/s1a.out" 2>&1; A=$?
./run.sh --setup-only > "$S/s1b.out" 2>&1; B=$?
check "setup exits 0 twice" '[[ $A == 0 && $B == 0 ]]'
check "second run skips installs even when --skip-clm toggles" 'grep -q "dependencies already installed" "$S/s1b.out"'
check "CLM failure is one line and points at a log" 'grep -q "CLM download failed (.*). skipping Tier 3A. Details: .*clm-download.log" "$S/s1b.out" && ! grep -q Traceback "$S/s1b.out"' "$S/s1b.out"
check "fallback hint does not force --cpu" 'grep -q "Run ./run.sh --no-setup to route" "$S/s1b.out"'
check "server stopped after setup" '! up && ! ours'

echo "--- S2 CLM client install path"
.venv-cpu/bin/python -m pip uninstall -y -q contrastive-lm >/dev/null 2>&1 || uv pip uninstall -q --python .venv-cpu/bin/python contrastive-lm >/dev/null 2>&1
./run.sh --setup-only > "$S/s2.out" 2>&1
check "CLM client reinstalled on demand" 'grep -q "Installing CLM client" "$S/s2.out" && .venv-cpu/bin/python -c "import clm" 2>/dev/null'

echo "--- S3 pre-existing server is reused and left running"
(OLLAMA_HOST=127.0.0.1:11435 setsid ollama serve >/dev/null 2>&1 &); sleep 2
./run.sh --no-setup > "$S/s3.out" 2>&1; R=$?
check "run succeeds against existing server" '[[ $R == 0 ]]'
check "says it is reusing" 'grep -q "already running on :11435: reusing it" "$S/s3.out"'
check "does not stop it" '! grep -q "Stopping the Ollama" "$S/s3.out" && up'
pkill -f stub_ollama_server.py; pkill -f "^sleep 1000$"; sleep 1

echo "--- S4 --keep-ollama"
./run.sh --no-setup --setup-only --keep-ollama > "$S/s4.out" 2>&1
check "started server left running with pid hint" 'up && grep -q "Leaving the Ollama server this run started (pid" "$S/s4.out"'
pkill -f stub_ollama_server.py; pkill -f "^sleep 1000$"; sleep 1

echo "--- S5 Ctrl-C mid-run"
# a real Ctrl-C: a ^C keystroke typed into a pseudo-terminal, delivered by the line discipline
# to the whole foreground process group, exactly as in an interactive terminal
(sleep 14; printf '\003'; sleep 30) | STUB_DELAY=60 script -qefc "./run.sh --no-setup" /dev/null > "$S/s5.out" 2>&1; R=$?
check "exit 130 and server stopped" '[[ $R == 130 ]] && grep -q "Stopping the Ollama" "$S/s5.out" && ! up && ! ours'
check "cleanup message reached the run log" 'grep -q "Stopping the Ollama" .run/latest.log'

echo "--- S6 SIGTERM"
./run.sh --no-setup > "$S/s6.out" 2>&1 & P=$!; for _ in $(seq 1 30); do up && break; sleep 0.5; done; sleep 1; kill -TERM $P; wait $P; R=$?
check "exit 143 and server stopped" '[[ $R == 143 ]] && ! up && ! ours'

echo "--- S7 failure after start (server too old)"
STUB_VERSION=0.30.0 ./run.sh --no-setup > "$S/s7.out" 2>&1; R=$?
check "fails with version message, server stopped" '[[ $R == 1 ]] && grep -q "need 0.35.0+" "$S/s7.out" && ! up && ! ours'

echo "--- S8/S9 HIR_OLLAMA_URL"
HIR_OLLAMA_URL=http://127.0.0.1:11999 ./run.sh --no-setup > "$S/s8.out" 2>&1; R=$?
check "unreachable URL fails fast, nothing started" '[[ $R == 1 ]] && grep -q "is not reachable" "$S/s8.out" && ! ours'
(OLLAMA_HOST=127.0.0.1:11777 setsid ollama serve >/dev/null 2>&1 &); sleep 2
HIR_OLLAMA_URL=http://127.0.0.1:11777 ./run.sh --no-setup --setup-only > "$S/s9.out" 2>&1; R=$?
check "managed URL used and left running" '[[ $R == 0 ]] && grep -q "not managed by this script" "$S/s9.out" && up 11777'
pkill -f stub_ollama_server.py; pkill -f "^sleep 1000$"; sleep 1

echo "--- S10 missing model"
rm -f "$STUB_STATE/tev1__0.8b"
./run.sh --no-setup > "$S/s10.out" 2>&1; R=$?
check "doctor names the missing model, exit 1, server stopped" '[[ $R == 1 ]] && grep -q "model tev1:0.8b missing: run ./run.sh without --no-setup" "$S/s10.out" && ! up'
./run.sh --setup-only > "$S/s10b.out" 2>&1
check "setup pulls it back" 'grep -q "Pulling tev1:0.8b" "$S/s10b.out" && [[ -f "$STUB_STATE/tev1__0.8b" ]]'

echo "--- S10c pull retries"
rm -f "$STUB_STATE/tev1__0.8b" "$STUB_STATE/.pull_attempts"
STUB_PULL_FAIL=2 ./run.sh --setup-only > "$S/s10c.out" 2>&1; R=$?
check "two connection resets, then the pull succeeds on retry" '[[ $R == 0 ]] && [[ $(grep -c "retrying in" "$S/s10c.out") == 2 ]] && [[ -f "$STUB_STATE/tev1__0.8b" ]]'
rm -f "$STUB_STATE/tev1__0.8b" "$STUB_STATE/.pull_attempts"
STUB_PULL_FAIL=always HIR_PULL_ATTEMPTS=2 ./run.sh --setup-only > "$S/s10d.out" 2>&1; R=$?
check "persistent failure: clear message, exit 1, server stopped" '[[ $R == 1 ]] && grep -q "Could not pull tev1:0.8b from registry.ollama.ai after 2 attempts" "$S/s10d.out" && ! up && ! ours'
rm -f "$STUB_STATE/.pull_attempts"; touch "$STUB_STATE/tev1__0.8b"

echo "--- S11 router args pass through"
rm -f "$S/d.jsonl"
./run.sh --no-setup --query "I can't log in" --meta '{"user_tier": "Enterprise", "failed_logins": 5}' --jsonl "$S/d.jsonl" > "$S/s11.out" 2>&1
check "single query routed with metadata, record written" 'grep -q "TIER_2B_CATBOOST" "$S/s11.out" && [[ $(wc -l < "$S/d.jsonl") == 1 ]] && grep -q policy_version "$S/d.jsonl"'
./run.sh --no-setup --shadow-rate 1.0 > "$S/s11b.out" 2>&1
check "shadow review runs for confident tiers" 'grep -q "shadow review, TIER_2A_TFIDF" "$S/s11b.out"'
./run.sh --no-setup --bogus-flag > "$S/s11c.out" 2>&1; R=$?
check "bad router arg: argparse error, server still stopped" '[[ $R == 2 ]] && grep -q "unrecognized arguments" "$S/s11c.out" && ! up && ! ours'

echo "--- S12 run logs"
for i in $(seq 1 25); do touch -d "2020-01-01 00:00:$(printf %02d "$i")" ".run/run-2020010100$(printf %04d "$i").log"; done
script -qec "./run.sh --no-setup --setup-only" /dev/null > "$S/s12.out" 2>&1
check "terminal output has colour" 'grep -q $'"'"'\e\[1;34m'"'"' "$S/s12.out"'
check "run log has no colour codes" '! grep -q $'"'"'\e\['"'"' .run/latest.log'
check "run log header records args and commit" 'head -1 .run/latest.log | grep -q "^# run.sh --no-setup --setup-only | .* | commit "'
check "only 20 run logs kept" '[[ $(ls .run/run-*.log | wc -l) -le 20 ]]'

echo "--- S13 --test runs unit and live suites"
./run.sh --no-setup --setup-only --test > "$S/s13.out" 2>&1; R=$?
check "all tests pass, none skipped (live suite ran)" '[[ $R == 0 ]] && grep -qE "^[0-9]+ passed in" "$S/s13.out"'

echo "--- S14 port taken by something that is not Ollama"
(setsid python3 -m http.server 11435 --bind 127.0.0.1 >/dev/null 2>&1 &); sleep 1
./run.sh --no-setup > "$S/s14.out" 2>&1; R=$?
check "clear failure pointing at the log" '[[ $R == 1 ]] && grep -q "Ollama exited on start. See .*ollama-cpu.log" "$S/s14.out"'
pkill -f "http.server 11435"; sleep 1

echo "--- S15 explicit flags"
./run.sh --gpu --no-setup --setup-only > "$S/s15.out" 2>&1
check "--gpu on a GPU-less box falls back with a warning" 'grep -q "Falling back to CPU" "$S/s15.out"'
./run.sh --cpu --no-setup --setup-only > "$S/s15b.out" 2>&1
check "--cpu: no fallback warning, hint keeps --cpu" '! grep -q "Falling back" "$S/s15b.out" && grep -q "Run ./run.sh --cpu --no-setup" "$S/s15b.out"'
./run.sh --help > "$S/s15c.out" 2>&1
check "--help shows GPU as default" 'grep -q "(default)       GPU" "$S/s15c.out"'

echo "--- S16 make targets"
make -s test-unit > "$S/m1.out" 2>&1; check "make test-unit" '[[ $? == 0 ]] || grep -q passed "$S/m1.out"'
make -s calibration > "$S/m2.out" 2>&1; check "make calibration" 'grep -q "ECE = 0.061" "$S/m2.out"'
make -s check > "$S/m3.out" 2>&1; check "make check (via run.sh, server stopped after)" 'grep -q "=> ready" "$S/m3.out" && ! up'
make -s test-live > "$S/m4.out" 2>&1; check "make test-live" 'grep -qE "[0-9]+ passed" "$S/m4.out" && ! up'

echo "--- leftovers"
sleep 2
check "no stub servers or runner children left" '! ours && [[ $(ps -eo stat=,args= | awk "\$2 == \"sleep\" && \$3 == \"1000\" && \$1 !~ /Z/" | wc -l) == 0 ]]'
echo; echo "PASSED $PASS  FAILED $FAIL"
rm -rf "$S"
[[ $FAIL == 0 ]]
