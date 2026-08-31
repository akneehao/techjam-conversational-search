"""Simplex-constrained linear ranker: w >= 0, sum(w) = 1, score = w . x.

Loss = class-balanced weighted BCE + 0.01*||w||^2 (L2) + 0.5*pairwise
log-loss (log(1 + exp(-(w.x_pos - w.x_neg))) over (pos, neg) pairs per query
group). Optimized with SciPy SLSQP, warm-started from BASELINE_WEIGHTS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from starter.reranker.base import BASELINE_WEIGHTS, LinearRanker, sigmoid  # noqa: E402

N_FEATURES = len(BASELINE_WEIGHTS)


class SimplexRanker:
    def __init__(self, l2_reg: float = 0.01, ranking_loss_weight: float = 0.5, max_iter: int = 500):
        self.l2_reg = l2_reg
        self.ranking_loss_weight = ranking_loss_weight
        self.max_iter = max_iter
        self.weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "SimplexRanker":
        X = np.asarray(X, dtype=np.float64)
        y_bin = (np.asarray(y) > 0).astype(np.float64)
        n_pos = max(1.0, y_bin.sum())
        n_neg = max(1.0, len(y_bin) - y_bin.sum())
        pos_weight = n_neg / n_pos  # class-balanced BCE weighting

        group_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for g in np.unique(groups):
            mask = groups == g
            idx = np.nonzero(mask)[0]
            pos_idx = idx[y[idx] > 0]
            neg_idx = idx[y[idx] == 0]
            if len(pos_idx) and len(neg_idx):
                group_pairs.append((X[pos_idx], X[neg_idx]))

        def loss(w: np.ndarray) -> float:
            logits = X @ w
            probs = sigmoid(logits)
            eps = 1e-9
            bce = -np.mean(
                pos_weight * y_bin * np.log(probs + eps) + (1 - y_bin) * np.log(1 - probs + eps)
            )
            l2 = self.l2_reg * float(np.dot(w, w))
            # Pairwise log-loss: for each group, every (pos, neg) combination
            # contributes log(1 + exp(-(w.pos - w.neg))).
            pairwise = 0.0
            n_pairs = 0
            for pos_x, neg_x in group_pairs:
                pos_scores = pos_x @ w          # (P,)
                neg_scores = neg_x @ w          # (N,)
                delta = pos_scores[:, None] - neg_scores[None, :]   # (P, N)
                pairwise += float(np.sum(np.log1p(np.exp(-np.clip(delta, -30, 30)))))
                n_pairs += delta.size
            pairwise = pairwise / n_pairs if n_pairs else 0.0
            return bce + l2 + self.ranking_loss_weight * pairwise

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * N_FEATURES
        w0 = np.array(BASELINE_WEIGHTS, dtype=np.float64)
        result = minimize(
            loss, w0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": 1e-9},
        )
        self.weights_ = np.clip(result.x, 0.0, None)
        self.weights_ = self.weights_ / self.weights_.sum() if self.weights_.sum() > 0 else w0
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.weights_

    def to_linear_ranker(self) -> LinearRanker:
        return LinearRanker(self.weights_, 0.0, name="simplex")
