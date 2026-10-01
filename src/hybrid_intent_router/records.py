"""The decision record. Every exit from the cascade writes the same shape, so "why did this go
there" is answered from logs alone, months later, without re-running a model."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import POLICY_VERSION


def record(query: str, tier: str, target: str, reason: str, model: str = "-",
           meta: Optional[Dict[str, Any]] = None, answers: Optional[Dict[str, Any]] = None,
           rules_fired: Optional[List[str]] = None, **detail: Any) -> Dict[str, Any]:
    """Every exit writes the same decision record. `answers` holds the full probability distributions
    the router computed (not just the winner) and `rules_fired` the rules that turned them into this
    exit, so a decision can be explained from the record alone. `detail` adds a tier's own fields
    (Tier 3B: page_on_call)."""
    state = {"message": query, **(meta or {})}
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # What the router saw: a fingerprint of the (redacted) state. Store the state itself separately.
        "state_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16],
        "tier": tier,
        "model": model,  # the engine and version that decided
        "policy_version": POLICY_VERSION,
        "target": target,
        "reason": reason,  # rendered from the rule or probabilities that fired, never generated
        "answers": answers or {},
        "rules_fired": list(rules_fired or []),
        **detail,
    }
