"""Tier 3A talks to Ollama's native /api/embed, and its failures explain themselves: the status line
keeps the embedder's message, and the preflight probe sends the same request and reports Ollama's
answer."""
import sys
import types

import numpy as np
import pytest

from hybrid_intent_router import doctor, tier3a_clm

pytestmark = pytest.mark.unit


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body
        self.text = str(body)

    def json(self):
        return self._body


def test_status_keeps_the_embedder_message(monkeypatch):
    class EmbedderError(RuntimeError):
        pass

    class Engine:
        def __init__(self, **kw):
            pass

        def rank(self, *a, **kw):
            raise EmbedderError('Ollama /api/embed error 400: {"error":"model does not support embeddings"}')

    monkeypatch.setitem(sys.modules, "clm", types.SimpleNamespace(Engine=Engine))
    monkeypatch.setattr(tier3a_clm, "ollama_embedder", lambda: None)
    monkeypatch.setattr(tier3a_clm, "DISABLE_CLM", False)
    engine, status = tier3a_clm.load_clm()
    assert engine is None
    assert status.startswith("skipped (EmbedderError: Ollama /api/embed error 400")
    assert "does not support embeddings" in status


def test_ollama_embedder_speaks_native_api():
    pytest.importorskip("clm.embedder")  # the CLM client is installed by run.sh, not in CI
    emb = tier3a_clm.ollama_embedder()
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, body=json)
        return _Resp(200, {"embeddings": [[3.0, 4.0], [0.0, 2.0]], "prompt_eval_count": 7})

    emb.session.post = post
    vecs, tokens = emb._fetch(["a", "b"])
    assert sent["url"].endswith("/api/embed")
    assert sent["body"] == {"model": "clm-encoder", "input": ["a", "b"], "truncate": True}
    assert np.allclose(vecs[0], [0.6, 0.8]) and np.allclose(vecs[1], [0.0, 1.0])  # L2-normalised
    assert tokens == 7


@pytest.mark.parametrize("resp, expect", [
    (_Resp(200, {"embeddings": [[0.1] * 4096]}), "ok  CLM embeddings answer"),
    (_Resp(400, {"error": "model does not support embeddings"}), "registered without embedding support"),
    (_Resp(500, {"error": "boom"}), "CLM embeddings fail: HTTP 500"),
])
def test_embedding_probe(monkeypatch, resp, expect):
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url)
        return resp

    monkeypatch.setattr(doctor.requests, "post", post)
    lines = []
    doctor.clm_embedding_probe(lambda status, msg: lines.append(f"{status}  {msg}"))
    assert sent["url"].endswith("/api/embed")
    assert any(expect in line for line in lines), lines


class _Engine:
    """Ranks the candidates in the order given, with the given probabilities."""
    def __init__(self, order):
        self.order = order

    def rank(self, query, candidates, instructions=None):
        assert candidates == list(tier3a_clm.CANDIDATES.values())
        return [{"candidate": tier3a_clm.CANDIDATES[k], "prob": p} for k, p in self.order]


def test_tier3a_routes_a_clear_winner():
    target, why, answers = tier3a_clm.tier3a(_Engine([("docs_rate_limits", 0.89), ("none_of_these", 0.06)]), "q")
    assert target == "docs_rate_limits"
    assert why == "top-2 margin 0.83 over none_of_these"
    assert answers == {"page": {"choice": "docs_rate_limits",
                                "probabilities": {"docs_rate_limits": 0.89, "none_of_these": 0.06}}}


def test_tier3a_abstains_when_no_page_answers():
    # A softmax over pages alone would hand the Okta question to some page; "none of these" lets CLM say no
    target, why, _ = tier3a_clm.tier3a(_Engine([("none_of_these", 0.86), ("docs_auth_tokens", 0.07)]), "q")
    assert target is None
    assert why == "out of catalog: none_of_these p=0.86 over docs_auth_tokens p=0.07"


def test_tier3a_abstains_on_a_tie():
    target, why, _ = tier3a_clm.tier3a(_Engine([("docs_billing_api", 0.46), ("docs_webhooks", 0.41)]), "q")
    assert target is None and why == "top-2 margin 0.05 over docs_webhooks"


@pytest.mark.parametrize("query, in_scope", [
    ("Why am I getting HTTP 429 responses?", True),
    ("Which endpoint returns my usage report?", True),
    ("How do I rotate my API token?", True),
    ("How do I register a webhook endpoint?", True),
    ("What are the API rate limits?", True),
    ("How do I set up Okta for our workspace?", False),
    ("My screen flashed green and the app uninstalled itself", False),
    ("We were billed twice. Refund it today or we cancel.", False),
])
def test_catalog_scope_gates_the_8b_pass(query, in_scope):
    assert tier3a_clm.in_catalog_scope(query) is in_scope
