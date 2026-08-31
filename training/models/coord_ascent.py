"""Gradient-free Coordinate Ascent, directly optimizing mean MRR (not a
proxy loss). Starts at BASELINE_WEIGHTS; for each dimension, try w[d]+-delta,
keep whichever improves MRR, halve delta when neither does; stop when
delta < tolerance or max_iter is reached. Subsamples whole query groups
(never splits one) when the data is larger than max_rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from starter.reranker.base import BASELINE_WEIGHTS, LinearRanker, mrr_single  # noqa: E402

N_FEATURES = len(BASELINE_WEIGHTS)


def _mean_mrr(w: np.ndarray, X: np.ndarray, y: np.ndarray, groups: np.ndarray, group_ids: np.ndarray) -> float:
    scores = X @ w
    total = 0.0
    n = 0
    for g in group_ids:
        mask = groups == g
        g_y = y[mask]
        if not np.any(g_y > 0):
            continue
        order = np.argsort(-scores[mask])
        total += mrr_single(g_y[order])
        n += 1
    return total / n if n else 0.0


class CoordinateAscentRanker:
    def __init__(self, max_iter: int = 300, initial_delta: float = 0.1, tolerance: float = 1e-6,
                 max_rows: int | None = 50_000):
        self.max_iter = max_iter
        self.initial_delta = initial_delta
        self.tolerance = tolerance
        self.max_rows = max_rows
        self.weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "CoordinateAscentRanker":
        rng = np.random.default_rng(42)
        unique_groups = np.unique(groups)
        if self.max_rows is not None and len(y) > self.max_rows:
            avg_rows_per_group = len(y) / len(unique_groups)
            keep_n = max(1, int(self.max_rows / max(1.0, avg_rows_per_group)))
            unique_groups = rng.choice(unique_groups, size=min(keep_n, len(unique_groups)), replace=False)

        w = np.array(BASELINE_WEIGHTS, dtype=np.float64)
        delta = self.initial_delta
        best_score = _mean_mrr(w, X, y, groups, unique_groups)
        it = 0
        while it < self.max_iter and delta >= self.tolerance:
            improved = False
            for d in range(N_FEATURES):
                for sign in (1.0, -1.0):
                    candidate = w.copy()
                    candidate[d] = max(0.0, candidate[d] + sign * delta)
                    score = _mean_mrr(candidate, X, y, groups, unique_groups)
                    if score > best_score:
                        w, best_score, improved = candidate, score, True
                it += 1
                if it >= self.max_iter:
                    break
            if not improved:
                delta /= 2.0
        self.weights_ = w
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.weights_

    def to_linear_ranker(self) -> LinearRanker:
        return LinearRanker(self.weights_, 0.0, name="coord_ascent")
