# CLAUDE.md

Context for coding agents working in `hybrid-intent-router`.

## What this repo is

The runnable companion to Appendix B of the article "Making Intent Routing Explainable". A four-tier intent-routing cascade ordered by cost: guardrail (Laya, every request) -> Tier 1 regex/trie -> Tier 2 TF-IDF or CatBoost -> Tier 3A CLM (optional) -> Tier 3B Laya -> Tier 3C Jev (tev1 via Ollama) -> Tier 4 generative fallback. Every exit writes a decision record.

## Invariants (do not break)

- Tiers only fall through on abstention. Nothing climbs back up.
- The guardrail runs on every request in parallel with Tier 1, and a positive verdict overrides any route. Never move it into the fall-through.
- Tier 1 matches the whole first token, never a prefix (`/cancellation policy?` must return None), and caps input at 2,000 chars before any regex.
- Every exit returns `records.record(...)`: `ts`, `state_sha256`, `tier`, `model`, `policy_version`, `target`, `reason`. Reasons are rendered from rules and probabilities, never generated (Tier 4 excepted, and labelled as such).
- Models and checkpoints stay pinned (Laya revisions in `config.py`, loaded with `standalone_repos=True` because the SHAs are per repository; laya==0.3.22; CLM commit in `run.sh`).
- Tier 1 and Tier 2 outputs must reproduce the article exactly; `tests/test_tier2_and_policy.py` pins them.

## Layout

- `run.sh` (root): one-click installer and runner. GPU default with automatic CPU fallback, `--cpu` to force CPU, `--skip-clm` in either mode. Starts Ollama only if none is running; stops only a server it started. Keep it shellcheck-clean and idempotent.
- `src/hybrid_intent_router/`: the package. `cascade.py` wires tiers; one module per tier; `config.py` holds every `HIR_*` knob.
- `tests/`: `unit` (offline, model tiers faked) and `live` (auto-skips without Ollama). `tests/installer/` drives the real `run.sh` against a stand-in Ollama and Laya.
- Runtime state (gitignored): `.run/` (run logs, Ollama server log, CLM/Laya download logs), `.venv-gpu/`, `.venv-cpu/`, `models/`.

## Commands

```bash
make test-unit                       # must pass before any commit
make lint                            # shellcheck run.sh
make test-installer                  # after touching run.sh: 35 end-to-end scenarios (Linux)
./run.sh --cpu --skip-clm --test     # full local run on the lightest path
```

## Style

No emojis and no em-dashes in docs or code comments. Match the surrounding code's comment density.
