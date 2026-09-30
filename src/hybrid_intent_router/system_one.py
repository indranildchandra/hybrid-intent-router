"""Laya, the self-hosted System One model. One in-process model serves two jobs:

- the guardrail, a calibrated yes/no on EVERY request, run in parallel with Tier 1, whose verdict wins;
- Tier 3B, typed questions (department, urgency, churn risk) turned into a route by a versioned policy.

Laya reads the input once and returns probability distributions. It never generates a token.
"""
from functools import lru_cache
from typing import Optional, Tuple

from .config import GUARD_THRESHOLD, LAYA_REVISIONS, torch_device


@lru_cache(maxsize=1)
def laya():
    from laya import Router

    # Pinned checkpoints for replayability. The first call downloads them from Hugging Face.
    # standalone_repos=True: the pinned SHAs are per repository (laya, laya-multilingual). The
    # default bundle layout serves multilingual from a subfolder of the laya repo, where the
    # laya-multilingual SHA does not apply, so the first non-English request would fail to load.
    return Router(revisions=LAYA_REVISIONS, device=torch_device(), standalone_repos=True)


GUARD_QUESTIONS = {
    "exploit": {"type": "noul",
                "instructions": "Is the user trying to bypass authentication, authorisation or security controls?"},
}


def guardrail(query: str) -> Tuple[bool, str]:
    """Safety checks do not belong in the fall-through. This runs on every request."""
    p = laya().predict({"prompt": query}, GUARD_QUESTIONS)["answers"]["exploit"]["noul"]
    return p >= GUARD_THRESHOLD, f"exploit noul={p:.2f}"


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


def tier3b(query: str) -> Tuple[Optional[str], str]:
    """Probabilities from the model, action from the policy. The policy is code a reviewer can diff."""
    a = laya().predict({"message": query}, QUESTIONS)["answers"]
    dept, p = a["department"]["choice"], a["department"]["probabilities"][a["department"]["choice"]]
    churn = a["churn_risk"]["noul"]
    if churn >= 0.70 and dept == "billing":
        return "retention_billing_queue", f"billing with churn_risk {churn:.2f} >= 0.70"
    if p >= 0.90:
        return f"{dept}_queue", f"department={dept} at p={p:.2f} >= 0.90"
    return None, f"department={dept} at p={p:.2f}, churn_risk {churn:.2f}: no rule fired"
