"""Tier 3A: CLM, a contrastive dual-encoder, as a branch for large static catalogs.

State and action are embedded in separate towers that never cross-attend, so the catalog is
embedded once and cached; each request costs one state pass plus a similarity op. The price:
no cross-attention, so "cancel my order" and "do not cancel my order" can land close together.
The top-2 margin is the explanation, and a small margin hands off to Tier 3B.

The candidates are short page descriptions, the plain answer text the action head is trained on, plus
an explicit "none of these". CLM's probabilities are a softmax over the candidates, so without that
option a request no page answers (an Okta setup question) still gets a confident winner; with it,
CLM itself says the request is out of catalog, and Tier 3A abstains.

The encoder is an 8-bit Qwen3-8B served by Ollama as `clm-encoder` (about 16 GB of RAM), reached
through Ollama's native /api/embed (see ollama_embedder).
When it is not served, this branch is skipped instead of failing the cascade.
"""
import os
from typing import Optional, Tuple

from .config import CLM_MARGIN, CLM_MODEL, DISABLE_CLM, OLLAMA, torch_device

CATALOG = {  # page id -> what the action head embeds
    "docs_usage_reports_api": "Usage reports endpoint: account usage and consumption data",
    "docs_billing_api": "Billing API: invoices, charges, payment methods",
    "docs_auth_tokens": "API tokens: create, rotate, revoke",
    "docs_webhooks": "Webhooks: event notifications to your HTTPS endpoint",
    "docs_rate_limits": "Rate limits: request quotas and HTTP 429 errors",
}
NONE_OF_THESE = "none_of_these"
CANDIDATES = {**CATALOG, NONE_OF_THESE: "None of these: not a question about the developer API docs"}
QUESTION = "Which documentation page answers this?"


def ollama_embedder():
    """CLM's embedder, speaking Ollama's native /api/embed instead of the OpenAI-style endpoint.

    CLM's client was written for vLLM: it sends `encoding_format: base64` and `truncate_prompt_tokens`
    to /v1/embeddings, vLLM options that Ollama's OpenAI compatibility layer may reject. /api/embed takes
    {"model", "input", "truncate"} and returns plain float vectors. Batching, caching and the L2
    normalisation the projection heads expect stay in CLM's Embedder; only the HTTP call changes.
    """
    import numpy as np
    import requests
    from clm.embedder import Embedder, EmbedderError, l2

    class OllamaEmbedder(Embedder):
        def _fetch(self, texts):
            body = {"model": self.model, "input": texts, "truncate": True}
            try:
                r = self.session.post(self.url, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                raise EmbedderError(f"Ollama unreachable at {self.url}: {e}") from e
            if r.status_code != 200:
                raise EmbedderError(f"Ollama /api/embed error {r.status_code}: {r.text[:300]}")
            j = r.json()
            vecs = j.get("embeddings") or []
            if len(vecs) != len(texts):
                raise EmbedderError(f"Ollama /api/embed returned {len(vecs)} vectors for {len(texts)} inputs")
            return [l2(np.asarray(v, dtype=np.float32)) for v in vecs], int(j.get("prompt_eval_count", 0) or 0)

        def healthy(self):
            try:
                return self.session.get(f"{OLLAMA}/api/version", timeout=5).status_code == 200
            except requests.RequestException:
                return False

    return OllamaEmbedder(url=f"{OLLAMA}/api/embed", model=CLM_MODEL)


def load_clm():
    """Returns (engine or None, status string)."""
    if DISABLE_CLM:
        return None, "skipped (disabled)"
    try:
        os.environ.setdefault("CLM_DEVICE", torch_device())  # where the projection heads run
        from clm import Engine  # do not name any local file clm.py

        engine = Engine(embedder=ollama_embedder())
        engine.rank("warm-up", list(CATALOG.values())[:2])  # fails fast if clm-encoder is missing or cannot load
        return engine, "enabled"
    except Exception as exc:  # noqa: BLE001
        detail = " ".join(str(exc).split())[:240]  # the embedder's error carries Ollama's HTTP status and reply
        return None, f"skipped ({type(exc).__name__}{': ' + detail if detail else ''})"


def tier3a(engine, query: str) -> Tuple[Optional[str], str]:
    ids, texts = list(CANDIDATES), list(CANDIDATES.values())
    ranked = [(ids[texts.index(r["candidate"])], r["prob"])
              for r in engine.rank(query, texts, instructions=QUESTION)]
    (top, p1), (second, p2) = ranked[0], ranked[1]
    if top == NONE_OF_THESE:
        return None, f"out of catalog: {NONE_OF_THESE} p={p1:.2f} over {second} p={p2:.2f}"
    why = f"top-2 margin {p1 - p2:.2f} over {second}"
    return (top if p1 - p2 >= CLM_MARGIN else None), why
