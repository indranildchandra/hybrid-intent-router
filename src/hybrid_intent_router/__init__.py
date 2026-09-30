"""Explainable hybrid intent routing: a four-tier cascade (regex/trie, TF-IDF and CatBoost,
System One decision models Laya/Jev/CLM, and a generative fallback) where every exit writes a
decision record with calibrated probabilities, a pinned model and a versioned policy."""

__version__ = "0.1.0"
