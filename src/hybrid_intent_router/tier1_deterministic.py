"""Tier 1: deterministic matching. If a pattern is invariant, match it. Do not predict it."""
import re
from typing import Optional, Tuple

MAX_INPUT_CHARS = 2_000  # bound regex work on untrusted input

COMMANDS = {"/cancel": "handler_cancel_subscription", "/help": "handler_display_help"}
TXN = re.compile(r"\bTXN_[A-Za-z0-9]{5,10}\b")


def tier1(query: str) -> Optional[Tuple[str, str]]:
    """Returns (handler, rule_id) or None. The rule id is the whole explanation."""
    query = query[:MAX_INPUT_CHARS]
    # Match the whole first token, never a prefix of it, so "/cancellation policy?"
    # does not fire the /cancel handler in the one tier that promises zero false positives.
    tokens = query.strip().lower().split(maxsplit=1)
    if tokens and tokens[0] in COMMANDS:
        return COMMANDS[tokens[0]], "cmd_trie_exact"
    if TXN.search(query):
        return "handler_transaction_status_lookup", "regex_txn_id_v1"
    return None
