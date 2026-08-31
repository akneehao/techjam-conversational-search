"""Learned re-ranking layer for the hybrid retrieval agent.

Pure-numpy at import/inference time except for the optional LightGBM GBDT
sub-model (``gbdt_inference.py`` guards that import itself). Nothing in this
package ever imports torch, scipy, or scikit-learn -- those are training-time
only dependencies that live under ``training/`` at the repo root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaselineRanker, LinearRanker
from .ensemble import minmax_normalize, threshold_union
from .gbdt_inference import load_gbdt
from .mlp_inference import MLPWeights, load_mlp

__all__ = ["RerankerBundle", "load_reranker"]


class RerankerBundle:
    """Wraps whichever trained model(s) are available and exposes .rank()."""

    def __init__(
        self,
        *,
        model_name: str,
        linear=None,
        mlp: Optional[MLPWeights] = None,
        gbdt=None,
        thresholds: Optional[dict] = None,
    ) -> None:
        self.model_name = model_name
        self.linear = linear
        self.mlp = mlp
        self.gbdt = gbdt
        self.thresholds = thresholds or {"gbdt": 0.72, "mlp": 0.85}

    def rank(self, X: np.ndarray, candidate_ids: list[str]) -> list[str]:
        if X.shape[0] == 0:
            return list(candidate_ids)

        if self.model_name == "ensemble":
            gbdt_scores = self.gbdt.predict(X) if self.gbdt is not None else None
            mlp_scores = self.mlp.predict(X) if self.mlp is not None else None
            if mlp_scores is None and gbdt_scores is None:
                return list(candidate_ids)
            if mlp_scores is None:
                order = np.argsort(-gbdt_scores)
                return [candidate_ids[i] for i in order]
            return threshold_union(
                gbdt_scores, mlp_scores, candidate_ids,
                gbdt_threshold=self.thresholds.get("gbdt", 0.72),
                mlp_threshold=self.thresholds.get("mlp", 0.85),
            )

        if self.model_name == "gbdt" and self.gbdt is not None:
            scores = self.gbdt.predict(X)
        elif self.model_name == "mlp" and self.mlp is not None:
            scores = self.mlp.predict(X)
        elif self.linear is not None:
            scores = self.linear.predict_scores(X)
        else:
            return list(candidate_ids)

        order = np.argsort(-scores)
        return [candidate_ids[i] for i in order]


def load_reranker(artifacts_dir: str | Path, model: str = "ensemble") -> Optional[RerankerBundle]:
    """Load whichever artifacts ``model`` needs. Returns None on any failure
    or missing artifact -- callers must treat that as "no re-ranking"."""
    artifacts_dir = Path(artifacts_dir)
    try:
        thresholds = None
        thresholds_path = artifacts_dir / "ensemble_thresholds.json"
        if thresholds_path.is_file():
            thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))

        if model == "baseline":
            return RerankerBundle(model_name="linear", linear=BaselineRanker(), thresholds=thresholds)

        if model in ("simplex", "ranksvm", "coord_ascent", "reinforce"):
            path = artifacts_dir / f"{model}_weights.json"
            if not path.is_file():
                return None
            return RerankerBundle(model_name="linear", linear=LinearRanker.load(path), thresholds=thresholds)

        if model == "mlp":
            mlp = load_mlp(artifacts_dir / "mlp_ranker.npz", artifacts_dir / "mlp_ranker.json")
            if mlp is None:
                return None
            return RerankerBundle(model_name="mlp", mlp=mlp, thresholds=thresholds)

        if model == "gbdt":
            gbdt = load_gbdt(artifacts_dir / "gbdtranker.txt")
            if gbdt is None:
                return None
            return RerankerBundle(model_name="gbdt", gbdt=gbdt, thresholds=thresholds)

        if model == "ensemble":
            mlp = load_mlp(artifacts_dir / "mlp_ranker.npz", artifacts_dir / "mlp_ranker.json")
            gbdt = load_gbdt(artifacts_dir / "gbdtranker.txt")
            if mlp is None and gbdt is None:
                return None
            return RerankerBundle(model_name="ensemble", mlp=mlp, gbdt=gbdt, thresholds=thresholds)

        return None
    except Exception:
        return None
