"""Pairwise RankSVM: build difference vectors dx = x_pos - x_neg per query
group (capped at max_pairs_per_query), train sklearn LinearSVC(C=1.0) on
{(dx, +1)}; the SVC's normal vector becomes the (unconstrained, no-simplex)
ranking weight vector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from starter.reranker.base import LinearRanker  # noqa: E402


class RankSVMRanker:
    def __init__(self, max_pairs_per_query: int = 50, C: float = 1.0, max_iter: int = 5000):
        self.max_pairs_per_query = max_pairs_per_query
        self.C = C
        self.max_iter = max_iter
        self.weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "RankSVMRanker":
        rng = np.random.default_rng(42)
        diffs: list[np.ndarray] = []
        for g in np.unique(groups):
            idx = np.nonzero(groups == g)[0]
            pos_idx = idx[y[idx] > 0]
            neg_idx = idx[y[idx] == 0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            pairs = [(p, n) for p in pos_idx for n in neg_idx]
            if len(pairs) > self.max_pairs_per_query:
                sel = rng.choice(len(pairs), size=self.max_pairs_per_query, replace=False)
                pairs = [pairs[i] for i in sel]
            for p, n in pairs:
                diffs.append(X[p] - X[n])

        if not diffs:
            self.weights_ = np.zeros(X.shape[1])
            return self

        dX = np.array(diffs)
        # Symmetrize: for every (pos - neg, +1) example, also add its
        # negation as a -1 example so LinearSVC has both classes to learn a
        # meaningful separating hyperplane through the origin.
        dX_full = np.concatenate([dX, -dX], axis=0)
        labels = np.concatenate([np.ones(len(dX)), -np.ones(len(dX))])
        clf = LinearSVC(C=self.C, max_iter=self.max_iter, fit_intercept=False)
        clf.fit(dX_full, labels)
        self.weights_ = clf.coef_.reshape(-1)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.weights_

    def to_linear_ranker(self) -> LinearRanker:
        return LinearRanker(self.weights_, 0.0, name="ranksvm")
