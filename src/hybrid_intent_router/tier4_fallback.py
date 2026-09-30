"""Tier 4: the generative fallback, the reasoning layer of last resort.

A small local model stands in for your frontier model. It is reached only when every tier above
abstained, and its output is constrained to the handler enum with a JSON schema. Its "reason" is
a story about the decision, not a record of it: that is exactly why it sits at the bottom.
"""
import json
from typing import Tuple

import requests

from .config import DEVICE, FALLBACK_MODEL, HANDLERS, OLLAMA

SYSTEM_PROMPT = (
    "You route customer support messages. Handlers:\n"
    "- billing_queue: charges, invoices, refunds\n"
    "- technical_queue: errors, bugs, APIs, endpoints, integrations, how-to questions about the product\n"
    "- sales_queue: plans, pricing, upgrades\n"
    "- account_access_queue: logging in, passwords, two-factor codes, SSO and identity providers\n"
    "- human_triage: anything that fits none of these\n"
    "Reply with the handler and a one-sentence reason about the message."
)


def tier4(query: str) -> Tuple[str, str]:
    schema = {"type": "object", "properties": {"handler": {"type": "string", "enum": HANDLERS},
                                               "reason": {"type": "string"}}, "required": ["handler", "reason"]}
    options = {"temperature": 0, "seed": 7}
    if DEVICE == "cpu":
        options["num_gpu"] = 0  # keep every layer off the GPU even if this Ollama server can see one
    r = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
        "model": FALLBACK_MODEL, "stream": False, "think": False, "format": schema,
        "options": options,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]})
    r.raise_for_status()
    out = json.loads(r.json()["message"]["content"])
    return out["handler"], out["reason"]
