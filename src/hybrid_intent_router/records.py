"""The decision record. Every exit from the cascade writes the same shape, so "why did this go
there" is answered from logs alone, months later, without re-running a model."""
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict

from .config import POLICY_VERSION


def record(query: str, tier: str, target: str, reason: str, model: str = "-") -> Dict[str, Any]:
    """Every exit writes the same decision record."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # What the router saw: a fingerprint of the (redacted) state. Store the state itself separately.
        "state_sha256": hashlib.sha256(query.encode()).hexdigest()[:12],
        "tier": tier,
        "model": model,  # the engine and version that decided
        "policy_version": POLICY_VERSION,
        "target": target,
        "reason": reason,  # rendered from the rule or probabilities that fired, never generated
    }
