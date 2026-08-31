"""Optional GBDT (LightGBM) inference. Guarded import: if lightgbm isn't
installed (the official scoring environment may not have it), load_gbdt()
returns None and RerankerBundle degrades to MLP-only / plain RRF -- the same
graceful-degradation pattern starter/agent.py already uses for the dense
retrieval track (see _init_dense).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import lightgbm as _lgb
except ImportError:
    _lgb = None

from .base import sigmoid


class GBDTModel:
    def __init__(self, booster) -> None:
        self.booster = booster

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.booster.predict(np.asarray(X, dtype=np.float64))
        # LambdaRank raw scores are unbounded; squash to [0,1] so ensemble
        # threshold-union can compare them on the same footing as the MLP's
        # sigmoid output before per-query min-max normalization.
        return sigmoid(np.asarray(raw))


def load_gbdt(path: str | Path) -> Optional[GBDTModel]:
    if _lgb is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    try:
        booster = _lgb.Booster(model_file=str(path))
        return GBDTModel(booster)
    except Exception:
        return None
