"""Laya, the self-hosted System One model. One in-process model serves two jobs:

- the guardrail, a calibrated yes/no on EVERY request, run in parallel with Tier 1, whose verdict wins;
- Tier 3B, typed questions (department, urgency, churn risk) turned into a route by a versioned policy.

Laya reads the input once and returns probability distributions. It never generates a token.
"""
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .config import GUARD_THRESHOLD, LAYA_REVISIONS, torch_device


# The pinned checkpoint clamps its `choice:11+` temperature (choice questions with 11 or more options)
# and warns on every load that those confidences are uncalibrated. No question here has more than
# three options, so that one entry is silenced; a warning naming any other entry still shows.
_UNUSED_BUCKET_WARNING = (r"laya: this checkpoint ships invalid temperatures or values outside \S+, \S+; "
                          r"using choice:11\+=[^,]* -> [0-9.]+\. Treat confidence")


@lru_cache(maxsize=1)
def laya():
    from laya import Router

    # Pinned checkpoints for replayability. The first call downloads them from Hugging Face.
    # standalone_repos=True: the pinned SHAs are per repository (laya, laya-multilingual). The
    # default bundle layout serves multilingual from a subfolder of the laya repo, where the
    # laya-multilingual SHA does not apply, so the first non-English request would fail to load.
    router = Router(revisions=LAYA_REVISIONS, device=torch_device(), standalone_repos=True)
    load = router.load  # predict() loads checkpoints lazily through this

    def load_quietly(name):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_UNUSED_BUCKET_WARNING, category=RuntimeWarning)
            return load(name)

    router.load = load_quietly
    return router


GUARD_QUESTIONS = {
    "exploit": {"type": "noul",
                "instructions": "Is the user trying to bypass authentication, authorisation or security controls?"},
}


def guardrail(query: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Safety checks do not belong in the fall-through. This runs on every request.
    Returns (blocked, reason, answers); every record keeps the verdict, blocked or not."""
    p = laya().predict({"prompt": query}, GUARD_QUESTIONS)["answers"]["exploit"]["noul"]
    return p >= GUARD_THRESHOLD, f"exploit noul={p:.2f}", {"exploit": {"noul": round(float(p), 4)}}


# Decompose "where does this go?" into the questions a human triager would ask
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should resolve this?",
                   "criteria": {"billing": "Duplicate charges, invoices, refunds",
                                "technical": "Errors, outages, API failures",
                                "sales": "Plan upgrades, seat expansions"}},
    "urgency": {"type": "score", "instructions": "How urgent is this for the customer?",
                "criteria": ["can wait", "today", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel or leave?"},
}


# The model's job ends at probabilities; turning them into an action is this policy, versioned as
# POLICY_VERSION. Each rule reads calibrated probabilities and returns (fired, human-readable reason).
RULES = [
    ("churn_priority",
     lambda a: a["churn_risk"]["noul"] >= 0.70 and a["department"]["choice"] == "billing",
     "billing with churn_risk {churn:.2f} >= 0.70 -> retention_billing_queue"),
    ("confident_department",
     lambda a: a["department"]["probabilities"][a["department"]["choice"]] >= 0.90,
     "department={dept} at p={p_dept:.2f} >= 0.90"),
    ("page_on_call",  # urgency scale 0 = can wait, 1 = today, 2 = blocking
     lambda a: a["urgency"]["score"] >= 1.5,
     "urgency {urgency:.2f} >= 1.50 (near blocking) -> page on-call"),
]


def policy(answers: Dict[str, Any]) -> Tuple[Optional[str], str, List[str], bool]:
    """-> (target or None to abstain, reason, rules fired, page on-call). The reason is rendered from
    the rules that fired; nothing is generated."""
    dept = answers["department"]["choice"]
    ctx = {
        "dept": dept,
        "p_dept": answers["department"]["probabilities"][dept],
        "churn": answers["churn_risk"]["noul"],
        "urgency": answers["urgency"]["score"],
    }
    fired = [(rid, tmpl.format(**ctx)) for rid, rule, tmpl in RULES if rule(answers)]
    fired_ids = {rid for rid, _ in fired}

    if "churn_priority" in fired_ids:
        target = "retention_billing_queue"
    elif "confident_department" in fired_ids:
        target = f"{dept}_queue"
    else:
        target = None  # nothing cleared its threshold: fall through to Jev
    reason = "; ".join(r for _, r in fired) or "no rule cleared its threshold"
    return target, reason, sorted(fired_ids), "page_on_call" in fired_ids


def tier3b(query: str) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Probabilities from Laya, action from the policy. Returns (target, reason, record detail)."""
    answers = laya().predict({"message": query}, QUESTIONS)["answers"]
    target, reason, fired, page = policy(answers)
    # The record keeps the full distributions, not just the winner
    detail = {"answers": answers, "rules_fired": fired, "page_on_call": page}
    return target, reason, detail
