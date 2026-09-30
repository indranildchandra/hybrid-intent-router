"""Runtime configuration. Every knob is an environment variable so run.sh, CI and production
share one code path. Defaults reproduce Appendix B of the article exactly."""
import os


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


# Ollama serves the Jev-compatible decision model, the CLM encoder and the Tier 4 stand-in.
# run.sh points this at a private CPU-only instance (port 11435) when invoked with --cpu.
OLLAMA = _env("HIR_OLLAMA_URL", "http://localhost:11434").rstrip("/")

# Device for the in-process models (Laya, CLM heads): auto | cuda | mps | cpu
DEVICE = _env("HIR_DEVICE", "auto").lower()

# Thresholds. Each one is a trade between precision and coverage: sweep them on a labelled
# holdout (see hybrid_intent_router.calibration) before trusting them in production.
THRESHOLD = _float("HIR_THRESHOLD", 0.85)              # calibrated-probability cut-off, Tiers 2 and 3
GUARD_THRESHOLD = _float("HIR_GUARD_THRESHOLD", 0.80)  # guardrails lean towards recall
CLM_MARGIN = _float("HIR_CLM_MARGIN", 0.08)            # top-2 probability gap for Tier 3A
SHADOW_RATE = _float("HIR_SHADOW_RATE", 0.02)          # share of confident decisions re-checked by Tier 4

POLICY_VERSION = _env("HIR_POLICY_VERSION", "routing-policy@v14")

# Model tags. Pin them: a decision you cannot replay is a decision you cannot explain.
JEV_MODEL = _env("HIR_JEV_MODEL", "tev1:0.8b")
FALLBACK_MODEL = _env("HIR_FALLBACK_MODEL", "qwen3:0.6b")
CLM_MODEL = _env("HIR_CLM_MODEL", "clm-encoder")

# Set HIR_USE_TYPESAFE=1 (and TYPESAFE_API_KEY) to send Tier 3C to managed Jev instead of Ollama.
USE_TYPESAFE = _env("HIR_USE_TYPESAFE", "0") == "1"
TYPESAFE_MODEL = _env("HIR_TYPESAFE_MODEL", "jev-1.13.0")

# Set HIR_DISABLE_CLM=1 to skip Tier 3A without probing the encoder (run.sh --skip-clm).
DISABLE_CLM = _env("HIR_DISABLE_CLM", "0") == "1"

# Laya checkpoints pinned to reviewed commits (laya.PINNED_REVISIONS in laya 0.3.22)
LAYA_REVISIONS = {
    "english": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "multilingual": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
}

HANDLERS = ["billing_queue", "technical_queue", "sales_queue", "account_access_queue", "human_triage"]


def torch_device() -> str:
    """Resolve HIR_DEVICE to a concrete torch device string."""
    if DEVICE in ("cpu", "cuda", "mps"):
        return DEVICE
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
