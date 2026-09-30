import numpy as np
import pytest

from hybrid_intent_router.calibration import ece, threshold_sweep
from hybrid_intent_router.records import record
from hybrid_intent_router.tier2_classical import tier2
from hybrid_intent_router.tier3c_jev import INTENT_QUEUE, INTENTS

pytestmark = pytest.mark.unit


def test_tfidf_routes_billing():
    target, p, tier, why = tier2("where is my invoice receipt", {})
    assert (target, tier) == ("billing_queue", "TIER_2A_TFIDF")
    assert p >= 0.85, why


def test_catboost_uses_metadata_not_text():
    target, p, tier, why = tier2("I can't log in", {"user_tier": "Enterprise", "failed_logins": 5})
    assert (target, tier) == ("escalate_human", "TIER_2B_CATBOOST")
    assert "failed_login_attempts_10m" in why


def test_every_intent_has_a_queue():
    assert set(INTENTS) - {"other"} == set(INTENT_QUEUE)


def test_record_shape():
    r = record("hello", "TIER_1_DETERMINISTIC", "x", "rule")
    assert {"ts", "state_sha256", "tier", "model", "policy_version", "target", "reason"} <= set(r)


def test_calibration_matches_article():
    rng = np.random.default_rng(7)
    conf = rng.beta(5, 2, size=5_000)
    correct = (rng.random(5_000) < conf ** 1.3).astype(float)
    assert round(ece(conf, correct), 3) == 0.061
    theta, coverage, precision = threshold_sweep(conf, correct, [0.90])[0]
    assert (round(coverage, 3), round(precision, 3)) == (0.112, 0.927)
