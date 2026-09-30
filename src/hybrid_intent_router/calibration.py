"""Set theta from a labelled holdout, never from a gut feeling.

python -m hybrid_intent_router.calibration   # runs the synthetic example from the article

Feed it your own holdout: each tier's top-class confidence and whether the label was right.
Read the output as a menu. There is no correct row, only the one your error budget and your
cost budget can both live with, written down next to the policy version.
"""
from typing import Iterable

import numpy as np


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error: how far stated confidence sits from observed accuracy."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(total)


def threshold_sweep(conf: np.ndarray, correct: np.ndarray, thetas: Iterable[float]):
    """Coverage = share of traffic this tier keeps. Precision = accuracy on what it keeps."""
    rows = []
    for theta in thetas:
        kept = conf >= theta
        coverage = float(kept.mean())
        precision = float(correct[kept].mean()) if kept.any() else float("nan")
        rows.append((theta, coverage, precision))
    return rows


def main() -> None:
    # Synthetic here; in production this comes from a human-labelled sample
    rng = np.random.default_rng(7)
    conf = rng.beta(5, 2, size=5_000)
    correct = (rng.random(5_000) < conf ** 1.3).astype(float)  # slightly overconfident model

    print(f"ECE = {ece(conf, correct):.3f}")
    for theta, coverage, precision in threshold_sweep(conf, correct, [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]):
        print(f"theta={theta:.2f}  coverage={coverage:6.1%}  precision={precision:6.1%}")


if __name__ == "__main__":
    main()
