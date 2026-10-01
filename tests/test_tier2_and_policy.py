import numpy as np
import pytest

from hybrid_intent_router.calibration import ece, threshold_sweep
from hybrid_intent_router.records import record
from hybrid_intent_router.system_one import policy
from hybrid_intent_router.tier2_classical import shap_values, tier2
from hybrid_intent_router.tier3c_jev import INTENT_QUEUE, INTENTS

pytestmark = pytest.mark.unit


def test_tfidf_routes_billing():
    target, p, tier, why, _ = tier2("where is my invoice receipt", {})
    assert (target, tier) == ("billing_queue", "TIER_2A_TFIDF")
    assert p >= 0.85, why


def test_documented_threshold_edges():
    # RUNBOOK 3.5 and Step 8 rely on these: the invoice variant clears 0.85 at
    # Tier 2, the variant without it abstains and reaches Laya's churn question
    _, p_invoice, _, _, _ = tier2("We were billed twice for invoice #99281. Refund it today or we cancel.", {})
    _, p_plain, _, _, _ = tier2("We were billed twice. Refund it today or we cancel.", {})
    assert round(p_invoice, 3) == 0.854 and p_invoice >= 0.85
    assert round(p_plain, 3) == 0.648


def test_catboost_uses_metadata_not_text():
    target, p, tier, why, _ = tier2("I can't log in", {"user_tier": "Enterprise", "failed_logins": 5})
    assert (target, tier) == ("escalate_human", "TIER_2B_CATBOOST")
    assert "failed_login_attempts_10m" in why


def test_tfidf_abstains_on_the_green_screen_crash():
    # The article's point: TF-IDF's best guess is billing, wrong, and the threshold stops it
    target, p, tier, why, _ = tier2("My screen flashed green and the app uninstalled itself", {})
    assert (why, round(p, 2)) == ("billing p=0.54", 0.54) and p < 0.85


def test_catboost_matches_article():
    # Same sentence, different metadata, different route; the login counter makes the call
    e_target, e_p, _, _, _ = tier2("can't log in", {"user_tier": "Enterprise", "failed_logins": 5})
    f_target, f_p, _, _, _ = tier2("can't log in", {"user_tier": "Free", "failed_logins": 1})
    assert (e_target, round(e_p, 2)) == ("escalate_human", 0.91)
    assert (f_target, round(f_p, 2)) == ("account_access_queue", 0.90)
    assert shap_values("can't log in", {"user_tier": "Enterprise", "failed_logins": 5}) == {
        "query_text": 0.149, "user_tier": 0.177, "failed_login_attempts_10m": 1.691}
    assert shap_values("can't log in", {"user_tier": "Free", "failed_logins": 1}) == {
        "query_text": 0.078, "user_tier": 0.451, "failed_login_attempts_10m": 1.381}


def test_every_intent_has_a_queue():
    assert set(INTENTS) - {"other"} == set(INTENT_QUEUE)


def test_record_shape():
    r = record("hello", "TIER_1_DETERMINISTIC", "x", "rule")
    assert {"ts", "state_sha256", "tier", "model", "policy_version", "target", "reason"} <= set(r)


# The article's illustrative answers for the refund-or-cancel message ("Keep the policy in versioned code")
ARTICLE_ANSWERS = {
    "department": {"choice": "billing", "probabilities": {"billing": 0.93, "technical": 0.04, "sales": 0.03}},
    "urgency": {"score": 1.78},
    "churn_risk": {"noul": 0.84},
}


def test_policy_matches_article():
    target, reason, fired, page = policy(ARTICLE_ANSWERS)
    assert fired == ["churn_priority", "confident_department", "page_on_call"]
    assert reason == ("billing with churn_risk 0.84 >= 0.70 -> retention_billing_queue; "
                      "department=billing at p=0.93 >= 0.90; urgency 1.78 >= 1.50 (near blocking) -> page on-call")
    assert (target, page) == ("retention_billing_queue", True)


def test_record_fingerprints_the_state_as_in_the_article():
    r = record("We were billed twice for invoice #<redacted>. Refund it today or we cancel.", "TIER_3B_LAYA",
               "retention_billing_queue", "rule", meta={"user_tier": "Enterprise", "open_tickets": 2})
    assert r["state_sha256"] == "4f008132db77c300"


def test_policy_abstains_when_no_routing_rule_fires():
    calm = {"department": {"choice": "sales", "probabilities": {"billing": 0.2, "technical": 0.3, "sales": 0.5}},
            "urgency": {"score": 0.4}, "churn_risk": {"noul": 0.1}}
    assert policy(calm) == (None, "no rule cleared its threshold", [], False)


def test_page_on_call_without_a_route_still_abstains():
    # Urgency alone pages on-call but picks no queue, so the request falls through
    urgent = {"department": {"choice": "technical", "probabilities": {"billing": 0.3, "technical": 0.6, "sales": 0.1}},
              "urgency": {"score": 1.9}, "churn_risk": {"noul": 0.2}}
    target, reason, fired, page = policy(urgent)
    assert (target, fired, page) == (None, ["page_on_call"], True)
    assert reason == "urgency 1.90 >= 1.50 (near blocking) -> page on-call"


def test_calibration_matches_article():
    rng = np.random.default_rng(7)
    conf = rng.beta(5, 2, size=5_000)
    correct = (rng.random(5_000) < conf ** 1.3).astype(float)
    assert round(ece(conf, correct), 3) == 0.061
    theta, coverage, precision = threshold_sweep(conf, correct, [0.90])[0]
    assert (round(coverage, 3), round(precision, 3)) == (0.112, 0.927)
