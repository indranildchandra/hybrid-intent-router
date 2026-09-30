"""Preflight checks: python -m hybrid_intent_router.doctor

Fails loudly on anything that would otherwise surface as a confusing stack trace mid-cascade.
Exit code 0 means the cascade can run; the CLM branch is optional and only warns.
"""
import sys

import requests

from .config import CLM_MODEL, DEVICE, DISABLE_CLM, FALLBACK_MODEL, JEV_MODEL, OLLAMA, USE_TYPESAFE, torch_device

MIN_OLLAMA = (0, 35, 0)


def _version_tuple(v: str):
    parts = []
    for p in v.split("-")[0].split(".")[:3]:
        parts.append(int("".join(ch for ch in p if ch.isdigit()) or 0))
    return tuple(parts + [0] * (3 - len(parts)))


def main() -> int:
    ok = True

    def report(status: str, msg: str):
        print(f"  [{status}] {msg}")

    print("hybrid-intent-router doctor")
    try:
        import torch

        dev = torch_device()
        report("ok", f"torch {torch.__version__}, HIR_DEVICE={DEVICE} -> {dev}")
        if DEVICE == "cuda" and not torch.cuda.is_available():
            report("FAIL", "HIR_DEVICE=cuda but torch cannot see a CUDA device (CPU-only wheel or driver issue)")
            ok = False
    except ImportError:
        report("FAIL", "torch is not installed")
        ok = False

    for mod in ("laya", "typesafe_sdk", "catboost", "sklearn"):
        try:
            __import__(mod)
            report("ok", f"import {mod}")
        except ImportError as exc:
            report("FAIL", f"import {mod}: {exc}")
            ok = False

    try:
        v = requests.get(f"{OLLAMA}/api/version", timeout=5).json()["version"]
        if _version_tuple(v) < MIN_OLLAMA:
            report("FAIL", f"Ollama {v} at {OLLAMA}; the /v1/systemone API needs 0.35 or later")
            ok = False
        else:
            report("ok", f"Ollama {v} at {OLLAMA}")
        # name -> digest. Tags can be re-pointed upstream; the digest logged here is what ran.
        tags = {m["name"]: m.get("digest", "") for m in requests.get(f"{OLLAMA}/api/tags", timeout=5).json().get("models", [])}

        def digest_of(model: str) -> str:
            d = tags.get(model) or tags.get(f"{model}:latest")
            return f" (digest {d[:12]})" if d else ""

        needed = [FALLBACK_MODEL] + ([] if USE_TYPESAFE else [JEV_MODEL])
        for model in needed:
            present = model in tags or f"{model}:latest" in tags
            report("ok" if present else "FAIL",
                   f"model {model}" + (digest_of(model) if present else
                                        " missing: run ./run.sh without --no-setup to pull it (or RUNBOOK section 7)"))
            ok &= present
        if not DISABLE_CLM:
            present = CLM_MODEL in tags or f"{CLM_MODEL}:latest" in tags
            report("ok" if present else "warn",
                   f"model {CLM_MODEL}" + (digest_of(CLM_MODEL) if present else " not served: Tier 3A (CLM) will be skipped"))
    except requests.RequestException as exc:
        report("FAIL", f"Ollama not reachable at {OLLAMA} ({type(exc).__name__}); start it with `ollama serve`")
        ok = False

    print("  => ready" if ok else "  => not ready")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
