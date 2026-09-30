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
    (_Resp(400, {"error": "model does not support embeddings"}), "not created as an embedding model"),
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
