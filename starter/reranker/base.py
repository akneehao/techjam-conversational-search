"""Shared feature schema, the linear-ranker persistence format, and the
ranking-metric primitives reused by both the runtime package and the
dev-only training scripts under ``training/``.

Numpy-only. No torch/scipy/scikit-learn/lightgbm imports here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

# Fixed order -- every feature matrix in this project is (N, 11) in this
# exact column order. See training/label_generation.py / reranker/features.py
# for how each column is computed.
FEATURE_NAMES: tuple[str, ...] = (
    "node_profile_sim",
    "max_desc_sim",
    "avg_top5_desc_sim",
    "sibling_coherence",
    "lexical_score",
    "path_score",
    "sibling_max_sim",
    "parent_sim",
    "query_token_count",
    "query_max_node_sim",
    "query_embedding_norm",
    # First-stage retrieval signals. Without these the re-ranker throws away
    # the exact evidence that makes the BM25+dense pipeline strong and has no
    # floor -- it cannot even reproduce the RRF ordering it is re-ranking.
    # With them, "copy RRF" is learnable, so the re-ranker can only add.
    "bm25_rank_score",
    "dense_rank_score",
    "rrf_score",
)

# v2 prior: mostly the first-stage retrieval signal (which is what actually
# works), topped up with the semantic/lexical features. Deliberately NOT the
# weights ported from .pg/SIMILARITY_LEARNING_V2.md -- those were tuned for
# ranking ontology *categories*, and their heavy weighting of neighbourhood
# features (path_score, sibling_max_sim) is wrong for ranking individual
# product instances that retrieval has already narrowed to one category.
BASELINE_WEIGHTS: tuple[float, ...] = (
    0.12, 0.04, 0.03, 0.02, 0.06, 0.05, 0.03, 0.02, 0.01, 0.02, 0.0,
    0.20, 0.15, 0.25,
)


class BaseRanker(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "BaseRanker": ...
    def predict_scores(self, X: np.ndarray) -> np.ndarray: ...
    def get_weights(self) -> dict: ...
    def save(self, path: str | Path) -> None: ...


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LinearRanker:
    """score = w . x (+ optional bias). Shared save/load/predict for the
    Simplex / RankSVM / Coordinate-Ascent / REINFORCE weight vectors -- they
    differ only in how ``weights``/``bias`` are fit, not in how they're
    persisted or scored."""

    def __init__(self, weights: np.ndarray, bias: float = 0.0, name: str = "linear") -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.name = name

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.weights + self.bias

    def get_weights(self) -> dict:
        return {"weights": self.weights.tolist(), "bias": self.bias, "name": self.name}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.get_weights(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearRanker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(data["weights"], dtype=np.float64), data.get("bias", 0.0), data.get("name", "linear"))


class BaselineRanker(LinearRanker):
    """Fixed (not learned) reference: dot product with BASELINE_WEIGHTS."""

    def __init__(self) -> None:
        super().__init__(np.array(BASELINE_WEIGHTS, dtype=np.float64), 0.0, name="baseline")

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "BaselineRanker":
        return self  # not learned


# --------------------------------------------------------------------------- #
# Ranking metrics. Each function takes `grades` already sorted by descending
# predicted score (caller's responsibility -- see rank_metrics() below, which
# does the sort once and computes everything off of it).
# --------------------------------------------------------------------------- #

def hits_at_k(sorted_grades: np.ndarray, k: int) -> float:
    return float(np.any(sorted_grades[:k] > 0)) if len(sorted_grades) else 0.0


def mrr_single(sorted_grades: np.ndarray) -> float:
    hits = np.nonzero(sorted_grades > 0)[0]
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def ndcg_at_k(sorted_grades: np.ndarray, k: int) -> float:
    grades = sorted_grades[:k]
    if len(grades) == 0:
        return 0.0
    ranks = np.arange(1, len(grades) + 1)
    dcg = float(np.sum((2.0 ** grades - 1.0) / np.log2(ranks + 1)))
    ideal = np.sort(sorted_grades)[::-1][:k]
    idcg = float(np.sum((2.0 ** ideal - 1.0) / np.log2(np.arange(1, len(ideal) + 1) + 1)))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(sorted_grades: np.ndarray, k: int) -> float:
    relevant = (sorted_grades[:k] > 0).astype(np.float64)
    total_relevant = int(np.sum(sorted_grades > 0))
    if total_relevant == 0 or relevant.sum() == 0:
        return 0.0
    cum_hits = np.cumsum(relevant)
    precision_at_i = cum_hits / np.arange(1, len(relevant) + 1)
    return float(np.sum(precision_at_i * relevant) / min(total_relevant, k))


def rank_metrics(grades: np.ndarray, scores: np.ndarray, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Sort one query group's candidates by descending score and compute the
    full metric suite in one place, so training/evaluate.py and the
    notebooks never re-implement the sort."""
    order = np.argsort(-scores)
    sorted_grades = np.asarray(grades)[order]
    out: dict = {"mrr": mrr_single(sorted_grades)}
    for k in ks:
        out[f"hits@{k}"] = hits_at_k(sorted_grades, k)
        out[f"ndcg@{k}"] = ndcg_at_k(sorted_grades, k)
        out[f"map@{k}"] = average_precision_at_k(sorted_grades, k)
    return out
