"""Threshold-union ensembling of the GBDT and MLP score arrays."""

from __future__ import annotations

import numpy as np


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def threshold_union(
    gbdt_scores: np.ndarray,
    mlp_scores: np.ndarray,
    candidate_ids: list[str],
    *,
    gbdt_threshold: float = 0.72,
    mlp_threshold: float = 0.85,
    fallback_top_n: int = 10,
) -> list[str]:
    """Admit a candidate if its normalized GBDT score >= gbdt_threshold OR
    its normalized MLP score >= mlp_threshold; rank admitted candidates by
    the higher of the two normalized scores. If nothing clears either
    threshold, fall back to the top-N by GBDT score (or MLP if GBDT is
    unavailable)."""
    n = len(candidate_ids)
    gbdt_norm = minmax_normalize(gbdt_scores) if gbdt_scores is not None else np.zeros(n)
    mlp_norm = minmax_normalize(mlp_scores) if mlp_scores is not None else np.zeros(n)
    max_score = np.maximum(gbdt_norm, mlp_norm)

    admitted = np.nonzero((gbdt_norm >= gbdt_threshold) | (mlp_norm >= mlp_threshold))[0]
    if len(admitted) == 0:
        primary = gbdt_scores if gbdt_scores is not None else mlp_scores
        order = np.argsort(-primary)[:fallback_top_n]
        return [candidate_ids[i] for i in order]

    order = admitted[np.argsort(-max_score[admitted])]
    rest = [i for i in np.argsort(-max_score) if i not in set(order.tolist())]
    full_order = list(order) + rest
    return [candidate_ids[i] for i in full_order]
