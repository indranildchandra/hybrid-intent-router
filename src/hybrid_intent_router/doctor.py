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


def clm_embedding_probe(report) -> None:
    """Send the request Tier 3A sends (Ollama's native /api/embed). Warns, since CLM is optional."""
    url = f"{OLLAMA}/api/embed"
    try:
        r = requests.post(url, json={"model": CLM_MODEL, "input": ["ping"], "truncate": True},
                          timeout=300)  # the first call loads the 8 GB encoder
        if r.status_code == 200 and r.json().get("embeddings"):
            report("ok", f"CLM embeddings answer at {url} (dim {len(r.json()['embeddings'][0])})")
            return
        answer = " ".join(r.text.split())[:200]
        hint = (" (the model was registered without embedding support: run ./run.sh again, which renames "
                "the GGUF pooling key and registers it anew)" if "embed" in answer.lower() and "support" in answer.lower() else "")
        report("warn", f"CLM embeddings fail: HTTP {r.status_code}: {answer}{hint}")
    except requests.RequestException as exc:
        report("warn", f"CLM embeddings: no answer from {url} ({type(exc).__name__}); the first call loads 8 GB, retry once")
    report("warn", "Tier 3A (CLM) will be skipped; the rest of the cascade is unaffected")


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
            if present:
                clm_embedding_probe(report)
    except requests.RequestException as exc:
        report("FAIL", f"Ollama not reachable at {OLLAMA} ({type(exc).__name__}); start it with `ollama serve`")
        ok = False

    print("  => ready" if ok else "  => not ready")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
