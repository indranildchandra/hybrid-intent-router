import pytest

from hybrid_intent_router.tier1_deterministic import MAX_INPUT_CHARS, tier1

pytestmark = pytest.mark.unit


def test_exact_command():
    assert tier1("/cancel") == ("handler_cancel_subscription", "cmd_trie_exact")
    assert tier1("/cancel please")[0] == "handler_cancel_subscription"
    assert tier1("  /HELP ") == ("handler_display_help", "cmd_trie_exact")


def test_prefix_does_not_fire():
    # Regression guard: a naive trie walk would route this to the cancel handler
    assert tier1("/cancellation policy?") is None


def test_transaction_id():
    assert tier1("Check status for TXN_99281X") == ("handler_transaction_status_lookup", "regex_txn_id_v1")
    assert tier1("TXN_1 is too short") is None


def test_natural_language_falls_through():
    assert tier1("where is my invoice receipt") is None
    assert tier1("") is None


def test_input_is_capped():
    # An ID past the cap is never scanned: bounded work on untrusted input
    assert tier1("x" * MAX_INPUT_CHARS + " TXN_99281X") is None
