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
    """Load the LightGBM booster, tolerating CRLF-mangled model files.

    The model is a .txt, so a Windows checkout with ``core.autocrlf=true``
    rewrites its ~2,000 line endings to CRLF.  LightGBM's parser rejects that
    with a **native abort** ("Model format error, expect a tree here") which
    kills the interpreter outright -- ``except Exception`` cannot catch it, so
    every downstream fallback in the agent is bypassed and the process dies at
    ``Agent()`` construction.

    Loading via ``model_str`` with normalised newlines avoids the abort
    entirely, and works on checkouts that are already mangled.  ``.gitattributes``
    marks these artifacts ``-text`` so fresh clones stay clean; this is the
    belt-and-braces half, since it also repairs existing bad working copies.
    """
    if _lgb is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        if not raw.lstrip().startswith(b"tree"):
            return None          # not a LightGBM text model -- do not hand it over
        text = raw.replace(b"\r\n", b"\n").decode("utf-8", "replace")
        booster = _lgb.Booster(model_str=text)
        return GBDTModel(booster)
    except Exception:
        return None
