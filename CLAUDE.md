# CLAUDE.md

Context for coding agents working in `hybrid-intent-router`.

## What this repo is

The runnable companion to Appendix B of the article "Making Intent Routing Explainable". A four-tier intent-routing cascade ordered by cost: guardrail (Laya, every request) -> Tier 1 regex/trie -> Tier 2 TF-IDF or CatBoost -> Tier 3A CLM (optional) -> Tier 3B Laya -> Tier 3C Jev (tev1 via Ollama) -> Tier 4 generative fallback. Every exit writes a decision record.

## Invariants (do not break)

- Tiers only fall through on abstention. Nothing climbs back up.
- The guardrail runs on every request in parallel with Tier 1, and a positive verdict overrides any route. Never move it into the fall-through.
- Tier 1 matches the whole first token, never a prefix (`/cancellation policy?` must return None), and caps input at 2,000 chars before any regex.
- Every exit returns `records.record(...)`: `ts`, `state_sha256` (16 hex chars over the message and its metadata), `tier`, `model`, `policy_version`, `target`, `reason`, `answers` (the full distributions on the path, the guardrail's verdict always included) and `rules_fired`; Tier 3B adds `page_on_call`. Each tier function hands back its distribution as its last return value, and the cascade passes it to `record()`. Tier 3B's policy is the `RULES` table in `system_one.py`; changing a rule means bumping `POLICY_VERSION`, and `tests/test_tier2_and_policy.py` pins the article's example. Reasons are rendered from rules and probabilities, never generated (Tier 4 excepted, and labelled as such).
- Models and checkpoints stay pinned (Laya revisions in `config.py`, loaded with `standalone_repos=True` because the SHAs are per repository; laya==0.3.22; CLM commit in `run.sh`).
- Versions stay pinned: `requirements.txt` is the single, generated lock (edit ranges in `pyproject.toml`, then `make lock`; never hand-edit it); torch and Ollama versions live in `run.sh` (`TORCH_VERSION`, `OLLAMA_VERSION`).
- Tier 1 and Tier 2 outputs must reproduce the article exactly; `tests/test_tier2_and_policy.py` pins them.

## Layout

- `run.sh` (root): one-click installer and runner. GPU default with automatic CPU fallback, `--cpu` to force CPU, `--skip-clm` in either mode. Starts Ollama only if none is running; stops only a server it started. Keep it shellcheck-clean and idempotent.
- `src/hybrid_intent_router/`: the package. `cascade.py` wires tiers; one module per tier; `config.py` holds every `HIR_*` knob. Tier 3A reaches the CLM encoder through Ollama's native `/api/embed` (`ollama_embedder` in `tier3a_clm.py`), not the vLLM-style `/v1/embeddings` the CLM client defaults to. `gguf_pooling.py` renames the published GGUF's bare `pooling_type` key to `qwen3.pooling_type` in place before `ollama create`; without it Ollama registers the encoder for completion only. Tier 3A only runs when `in_catalog_scope()` (a regex over the catalog's vocabulary) passes, so CLM's 8B pass stays a branch, not a step before Laya. It ranks short page descriptions plus a "none of these" candidate and abstains when that wins: CLM's softmax always has a winner, so this is how it says out of catalog.
- `tests/`: `unit` (offline, model tiers faked) and `live` (auto-skips without Ollama). `tests/installer/` drives the real `run.sh` against a stand-in Ollama and Laya.
- Runtime state (gitignored): `.run/` (run logs, Ollama server log, CLM/Laya download logs), `.venv-gpu/`, `.venv-cpu/`, `models/`.

## Commands

```bash
make test-unit                       # must pass before any commit
make lint                            # shellcheck run.sh (pinned shellcheck-py, in the venv via requirements.txt)
make test-installer                  # after touching run.sh: 37 end-to-end checks (Linux)
./run.sh --cpu --skip-clm --test     # full local run on the lightest path
```

## Style

No emojis and no em-dashes in docs or code comments. Match the surrounding code's comment density.
