"""Tier 3A: CLM, a contrastive dual-encoder, as a branch for large static catalogs.

State and action are embedded in separate towers that never cross-attend, so the catalog is
embedded once and cached; each request costs one state pass plus a similarity op. The price:
no cross-attention, so "cancel my order" and "do not cancel my order" can land close together.
The top-2 margin is the explanation, and a small margin hands off to Tier 3B.

The encoder is an 8-bit Qwen3-8B served by Ollama as `clm-encoder` (about 16 GB of RAM).
When it is not served, this branch is skipped instead of failing the cascade.
"""
import os
from typing import Optional, Tuple

from .config import CLM_MARGIN, CLM_MODEL, DISABLE_CLM, OLLAMA, torch_device

CATALOG = ["docs_usage_reports_api", "docs_billing_api", "docs_auth_tokens", "docs_webhooks", "docs_rate_limits"]


def load_clm():
    """Returns (engine or None, status string)."""
    if DISABLE_CLM:
        return None, "skipped (disabled)"
    try:
        os.environ.setdefault("CLM_DEVICE", torch_device())  # where the projection heads run
        from clm import Engine  # do not name any local file clm.py

        engine = Engine(emb_url=f"{OLLAMA}/v1/embeddings", emb_model=CLM_MODEL)
        engine.rank("warm-up", CATALOG[:2])  # fails fast if clm-encoder is missing or cannot load
        return engine, "enabled"
    except Exception as exc:  # noqa: BLE001
        return None, f"skipped ({type(exc).__name__})"


def tier3a(engine, query: str) -> Tuple[Optional[str], str]:
    ranked = engine.rank(query, CATALOG, instructions="Which documentation page answers this?")
    margin = ranked[0]["prob"] - ranked[1]["prob"]
    why = f"top-2 margin {margin:.2f} over {ranked[1]['candidate']}"
    return (ranked[0]["candidate"] if margin >= CLM_MARGIN else None), why
