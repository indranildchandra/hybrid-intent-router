"""Tier 2: classical ML on CPU cores you already pay for.

2A: TF-IDF + logistic regression for text-only requests.
2B: CatBoost when metadata carries the signal (same text, different route).

The training data is deliberately tiny, as in the article. Train on your own tickets.
"""
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TFIDF_TARGET = {"billing": "billing_queue", "account_access": "account_access_queue", "bug_report": "technical_queue"}

_TFIDF_X = [
    "where is my invoice receipt", "invoice receipt missing", "send me the invoice", "billing receipt for last month",
    "need a copy of my invoice", "charged twice on my card", "refund the extra charge", "my receipt is wrong",
    "cannot log into my account", "password reset link expired", "locked out of my account", "reset my password",
    "two factor code not arriving", "account access denied", "forgot my login email", "sign in fails",
    "app crashing on checkout", "getting 500 error on api call", "page will not load", "export button broken",
    "sync job fails every night", "webhook returns timeout", "dashboard shows blank chart", "upload keeps failing",
]
_TFIDF_Y = ["billing"] * 8 + ["account_access"] * 8 + ["bug_report"] * 8

# Toy-sized corpus: let every token into CatBoost's text dictionary. The default dictionary drops
# rare tokens and fails outright on a handful of rows. Remove this block once you train on real volume.
_CAT_TEXT_PROCESSING = {
    "tokenizers": [{"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " "}],
    "dictionaries": [{"dictionary_id": "Word", "occurrence_lower_bound": "1", "gram_order": "1"}],
    "feature_processing": {"default": [{"dictionaries_names": ["Word"], "feature_calcers": ["BoW"],
                                        "tokenizers_names": ["Space"]}]},
}
_CAT_TRAIN = pd.DataFrame({
    "query_text": ["can't log in", "need refund", "can't log in", "locked out of account", "refund my last charge",
                   "forgot password"],
    "user_tier": ["Enterprise", "Free", "Free", "Enterprise", "Pro", "Pro"],
    "failed_login_attempts_10m": [5, 0, 1, 6, 0, 1],
    "label": ["escalate_human", "billing_queue", "account_access_queue", "escalate_human", "billing_queue",
              "account_access_queue"],
})


@lru_cache(maxsize=1)
def tfidf_model() -> Pipeline:
    # multi_class="multinomial" is gone from current scikit-learn; lbfgs is multinomial by default
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("clf", LogisticRegression(C=20.0))])
    model.fit(_TFIDF_X, _TFIDF_Y)
    return model


@lru_cache(maxsize=1)
def catboost_model() -> CatBoostClassifier:
    model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, loss_function="MultiClass",
                               verbose=False, random_seed=0, thread_count=-1,
                               text_processing=_CAT_TEXT_PROCESSING, allow_writing_files=False)
    model.fit(_CAT_TRAIN.drop(columns=["label"]), _CAT_TRAIN["label"],
              text_features=["query_text"], cat_features=["user_tier"])
    return model


def tier2(query: str, meta: Dict[str, Any]) -> Tuple[Optional[str], float, str, str]:
    """Returns (target, confidence, tier, reason). The caller applies the threshold."""
    if meta:
        cat = catboost_model()
        x = pd.DataFrame([{"query_text": query, "user_tier": meta.get("user_tier", "Free"),
                           "failed_login_attempts_10m": meta.get("failed_logins", 0)}])
        probs = cat.predict_proba(x)[0]
        i = int(probs.argmax())
        # Per-decision SHAP attributions for the predicted class, straight from CatBoost
        shap = cat.get_feature_importance(Pool(x, text_features=["query_text"], cat_features=["user_tier"]),
                                          type="ShapValues")[0][i][:-1]
        top = x.columns[int(np.argmax(shap))]
        return cat.classes_[i], float(probs[i]), "TIER_2B_CATBOOST", f"p={probs[i]:.2f}, top SHAP feature: {top}"
    tfidf = tfidf_model()
    probs = tfidf.predict_proba([query])[0]
    i = int(probs.argmax())
    label = tfidf.classes_[i]
    return TFIDF_TARGET[label], float(probs[i]), "TIER_2A_TFIDF", f"{label} p={probs[i]:.2f}"
