import json

import pytest

from hybrid_intent_router import __main__ as cli
from hybrid_intent_router import cascade

pytestmark = pytest.mark.unit


class FakeRouter:
    def __init__(self, shadow_rate):
        self.device, self.clm_status, self.shadow_queue = "cpu", "skipped (test)", []

    def route(self, query, meta):
        from hybrid_intent_router.records import record
        return record(query, "TIER_1_DETERMINISTIC", "handler_x", f"meta={sorted(meta)}")


def test_single_query_writes_jsonl(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cascade, "HybridRouter", FakeRouter)
    out = tmp_path / "decisions.jsonl"
    assert cli.main(["--query", "hello", "--meta", '{"user_tier": "Pro"}', "--jsonl", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["query"] == "hello" and rows[0]["meta"] == {"user_tier": "Pro"}
    assert "TIER_1_DETERMINISTIC" in capsys.readouterr().out


def test_demo_set_routes_all_eight(monkeypatch, capsys):
    monkeypatch.setattr(cascade, "HybridRouter", FakeRouter)
    assert cli.main([]) == 0
    assert capsys.readouterr().out.count("TIER_1_DETERMINISTIC") == len(cli.DEMO_REQUESTS) == 8
