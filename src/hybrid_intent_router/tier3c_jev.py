"""Tier 3C: Jev on the fine-grained 25-intent catalog.

Past roughly 20 options Laya degrades and Jev holds up, so the large question goes here once
Laya's three-way department question has abstained. Locally, Ollama serves the Jev-compatible
tev1 model behind the same /v1/systemone API, so the official TypeSafe SDK works unchanged.
Moving to managed Jev is configuration, not code: HIR_USE_TYPESAFE=1 plus TYPESAFE_API_KEY.
"""
from functools import lru_cache
from typing import Optional, Tuple

from .config import JEV_MODEL, OLLAMA, THRESHOLD, TYPESAFE_MODEL, USE_TYPESAFE

INTENTS = {
    "usage_reports_api": "Questions about API endpoints for usage or consumption reports",
    "billing_api": "Questions about invoices or payments through the API",
    "auth_tokens": "API keys, tokens, OAuth, authentication setup",
    "webhooks": "Webhook setup, delivery failures, retries",
    "rate_limits": "429 errors, quotas, throttling",
    "sdk_install": "Installing or upgrading SDKs and client libraries",
    "data_export": "Exporting data, CSV downloads, bulk export",
    "sso_setup": "SAML, SSO, identity provider configuration",
    "user_management": "Adding, removing or changing roles of users",
    "plan_upgrade": "Upgrading plans, adding seats",
    "plan_comparison": "Comparing plans or competitors, migration advice",
    "invoice_copy": "Requesting a copy of an invoice or receipt",
    "refund_request": "Asking for a refund",
    "duplicate_charge": "Charged twice or duplicate payments",
    "tax_documents": "VAT, GST, tax invoices",
    "password_reset": "Forgot password or reset links",
    "two_factor": "Two-factor or OTP problems",
    "account_locked": "Account locked or suspended",
    "app_crash": "App crashes, freezes or uninstalls itself",
    "ui_bug": "Visual glitches, broken buttons, display issues",
    "performance": "Slow pages, timeouts, latency",
    "data_loss": "Missing or deleted data",
    "feature_request": "Asking for a new feature",
    "cancellation": "Wants to cancel the subscription",
    "other": "None of the above",
}
INTENT_QUEUE = {
    **dict.fromkeys(["usage_reports_api", "billing_api", "auth_tokens", "webhooks", "rate_limits", "sdk_install",
                     "data_export", "app_crash", "ui_bug", "performance", "data_loss"], "technical_queue"),
    **dict.fromkeys(["invoice_copy", "refund_request", "duplicate_charge", "tax_documents"], "billing_queue"),
    **dict.fromkeys(["sso_setup", "user_management", "password_reset", "two_factor", "account_locked"],
                    "account_access_queue"),
    **dict.fromkeys(["plan_upgrade", "plan_comparison", "feature_request", "cancellation"], "sales_queue"),
}


@lru_cache(maxsize=1)
def jev():
    from typesafe_sdk import TypeSafeClient

    if USE_TYPESAFE:
        # Managed Jev: TYPESAFE_API_KEY comes from the environment
        return TypeSafeClient(model=TYPESAFE_MODEL, timeout=120)
    # The SDK requires a non-empty key; Ollama ignores it. The first call loads the model into memory.
    return TypeSafeClient(base_url=OLLAMA, api_key="ollama", model=JEV_MODEL, timeout=120)


def tier3c(query: str) -> Tuple[Optional[str], str, str]:
    """Returns (target or None, reason, model tag that actually answered)."""
    from typesafe_sdk import Choice

    resp = jev().system_one(state={"message": query}, questions={
        "intent": Choice(instructions="Which support intent best describes this message?", criteria=INTENTS)})
    a = resp.answers["intent"]
    p = a.probabilities[a.choice]
    runner_up = sorted(a.probabilities.items(), key=lambda kv: -kv[1])[1]
    why = f"intent={a.choice} at p={p:.2f} (next: {runner_up[0]} {runner_up[1]:.2f})"
    model = getattr(resp, "model", None) or (TYPESAFE_MODEL if USE_TYPESAFE else JEV_MODEL)
    if a.choice != "other" and p >= THRESHOLD:
        return INTENT_QUEUE[a.choice], why, model
    return None, why, model
