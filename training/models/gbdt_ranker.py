"""GBDT ranker: LightGBM LambdaRank objective. label_gain=[0,1,3] maps grade
0/1/2 -> gain 0/1/3, matching the 2^grade - 1 NDCG gain formula so
LambdaRank's pairwise gradients are calibrated to the 3-level graded scheme.

An optional small grid search (`grid_search`) is provided but is genuinely
optional -- the fixed defaults below are what the source methodology shipped
with; only run the sweep if there's time left after all 7 models + the
agent.py integration are done (see the plan's risk-flags section).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from starter.reranker.base import FEATURE_NAMES, sigmoid  # noqa: E402
from training.common import group_kfold_splits  # noqa: E402
from training.evaluate import evaluate_predictions  # noqa: E402

LABEL_GAIN = (0, 1, 3)


def _make_dataset(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[lgb.Dataset, np.ndarray]:
    order = np.argsort(groups, kind="stable")
    X_sorted, y_sorted, groups_sorted = X[order], y[order], groups[order]
    _, group_sizes = np.unique(groups_sorted, return_counts=True)
    dataset = lgb.Dataset(X_sorted, label=y_sorted, group=group_sizes, feature_name=list(FEATURE_NAMES))
    return dataset, order


class GBDTRanker:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05, num_leaves: int = 31,
                 min_data_in_leaf: int = 5, min_sum_hessian_in_leaf: float = 5.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.booster_ = None
        self.feature_importances_: dict[str, float] = {}

    def _params(self) -> dict:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_sum_hessian_in_leaf": self.min_sum_hessian_in_leaf,
            "label_gain": list(LABEL_GAIN),
            "verbosity": -1,
        }

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "GBDTRanker":
        dataset, _ = _make_dataset(X, y, groups)
        self.booster_ = lgb.train(self._params(), dataset, num_boost_round=self.n_estimators)
        gains = self.booster_.feature_importance(importance_type="gain")
        self.feature_importances_ = dict(zip(FEATURE_NAMES, gains.tolist()))
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        raw = self.booster_.predict(np.asarray(X, dtype=np.float64))
        return sigmoid(np.asarray(raw))

    def save(self, model_path: str | Path, importances_path: str | Path | None = None) -> None:
        self.booster_.save_model(str(model_path))
        if importances_path is not None:
            Path(importances_path).write_text(
                json.dumps(self.feature_importances_, indent=2), encoding="utf-8",
            )


def grid_search(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
    learning_rates=(0.01, 0.05, 0.10), num_leaves_opts=(15, 31, 63),
    min_data_opts=(5, 20, 50), n_estimators: int = 500, n_folds: int = 5, seed: int = 42,
) -> tuple[dict, list[dict]]:
    """3x3x3 sweep, mean NDCG@5 across n_folds group-K-folds. Returns
    (best_params, all_results) -- the caller decides whether to actually
    retrain with best_params or just keep the fixed defaults."""
    folds = list(group_kfold_splits(groups, n_folds=n_folds, seed=seed))
    results = []
    for lr in learning_rates:
        for leaves in num_leaves_opts:
            for min_data in min_data_opts:
                fold_ndcg5 = []
                for train_mask, val_mask in folds:
                    model = GBDTRanker(
                        n_estimators=n_estimators, learning_rate=lr,
                        num_leaves=leaves, min_data_in_leaf=min_data,
                    ).fit(X[train_mask], y[train_mask], groups[train_mask])
                    scores = model.predict_scores(X[val_mask])
                    metrics = evaluate_predictions(scores, y[val_mask], groups[val_mask])
                    fold_ndcg5.append(metrics.get("ndcg@5", 0.0))
                mean_ndcg5 = float(np.mean(fold_ndcg5)) if fold_ndcg5 else 0.0
                results.append({
                    "learning_rate": lr, "num_leaves": leaves, "min_data_in_leaf": min_data,
                    "mean_ndcg@5": mean_ndcg5,
                })
    best = max(results, key=lambda r: r["mean_ndcg@5"])
    return best, results
