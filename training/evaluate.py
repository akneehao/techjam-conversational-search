"""Offline model-comparison metrics: Hits@K / MRR / NDCG@K / MAP@K,
group-macro-averaged. This is a DIAGNOSTIC tool for picking/tuning models in
the notebooks -- it is not, and must never be confused with, the official
scorer. The only authoritative score is `python -m evaluator.local_evaluator`
run unmodified against the live agent (see the repo README / the
verification steps in the implementation plan).
"""

from __future__ import annotations

import numpy as np

from starter.reranker.base import rank_metrics


def evaluate_predictions(
    scores: np.ndarray, y: np.ndarray, groups: np.ndarray, ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Macro-average rank_metrics() across query groups. Groups with no
    positive label are skipped (nothing to rank correctly)."""
    scores, y, groups = np.asarray(scores), np.asarray(y), np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        mask = groups == g
        g_y = y[mask]
        if not np.any(g_y > 0):
            continue
        rows.append(rank_metrics(g_y, scores[mask], ks=ks))
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def evaluate_subsets(
    scores: np.ndarray, y: np.ndarray, groups: np.ndarray, source: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, dict]:
    """Same as evaluate_predictions but broken out by the `source` column
    (e.g. "in_catalog" vs "synthetic"), plus an overall "full" row."""
    out = {"full": evaluate_predictions(scores, y, groups, ks)}
    for tag in sorted(set(source.tolist())):
        mask = source == tag
        if mask.any():
            out[tag] = evaluate_predictions(scores[mask], y[mask], groups[mask], ks)
    return out
