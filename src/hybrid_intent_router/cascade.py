"""The four-tier cascade. Tiers are ordered by cost; each one either answers above its threshold
or passes the request down. Nothing climbs back up. The guardrail sees everything."""
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .config import CLM_MODEL, FALLBACK_MODEL, SHADOW_RATE, THRESHOLD, torch_device
from .records import record
from .system_one import guardrail, laya, tier3b
from .tier1_deterministic import tier1
from .tier2_classical import catboost_model, tfidf_model, tier2
from .tier3a_clm import in_catalog_scope, load_clm, tier3a
from .tier3c_jev import tier3c
from .tier4_fallback import tier4


class HybridRouter:
    def __init__(self, shadow_rate: float = SHADOW_RATE, seed: Optional[int] = 0):
        # Load everything up front so the first request does not pay for model downloads
        tfidf_model()
        catboost_model()
        laya()
        self.device = torch_device()
        self.clm, self.clm_status = load_clm()
        self.shadow_rate = shadow_rate
        self.shadow_queue: List[Tuple[str, Dict[str, Any]]] = []
        self._rng = random.Random(seed)

    def route(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = meta or {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            guard = pool.submit(guardrail, query)  # every request, in parallel with Tier 1
            t1 = tier1(query)
            blocked, why, guard_answers = guard.result()
        # The guardrail's verdict goes into every record: it shows why a request was not blocked, too
        if blocked:
            return record(query, "GUARDRAIL", "policy_violation_block", why, "laya", meta,
                          guard_answers, ["guardrail_exploit"])
        if t1:
            return record(query, "TIER_1_DETERMINISTIC", t1[0], t1[1], "-", meta, guard_answers, [t1[1]])

        target, p, tier, why, answers = tier2(query, meta)
        if p >= THRESHOLD:
            return self._shadow(record(query, tier, target, why, tier.lower(), meta,
                                       {**guard_answers, **answers}, ["tier2_confident"]), query)

        if self.clm is not None and in_catalog_scope(query):  # a branch for its catalog, not a step for all
            target, why, answers = tier3a(self.clm, query)
            if target:
                return self._shadow(record(query, "TIER_3A_CLM", target, why, CLM_MODEL, meta,
                                           {**guard_answers, **answers}, ["clm_margin"]), query)

        target, why, detail = tier3b(query)
        if target:
            return self._shadow(record(query, "TIER_3B_LAYA", target, why, "laya", meta,
                                       {**guard_answers, **detail["answers"]}, detail["rules_fired"],
                                       page_on_call=detail["page_on_call"]), query)

        target, why, model, answers = tier3c(query)
        if target:
            return self._shadow(record(query, "TIER_3C_JEV", target, why, model, meta,
                                       {**guard_answers, **answers}, ["jev_confident"]), query)

        target, why = tier4(query)  # generated: no distribution and no rule, and the record says so
        return record(query, "TIER_4_LLM_FALLBACK", target, why, FALLBACK_MODEL, meta, guard_answers, [])

    def _shadow(self, decision: Dict[str, Any], query: str) -> Dict[str, Any]:
        # Confident decisions never reach a later tier, so a small random share is re-checked
        if self._rng.random() < self.shadow_rate:
            self.shadow_queue.append((query, decision))
        return decision

    @staticmethod
    def shadow_review(query: str, decision: Dict[str, Any]) -> str:
        """Runs off the request path. Disagreements go to a person, not straight to a metric."""
        second, _ = tier4(query)
        return "agree" if second == decision["target"] else f"disagree (Tier 4 says {second}): send to review"
