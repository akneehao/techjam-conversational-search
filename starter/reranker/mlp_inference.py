"""Pure-numpy forward pass for the MLP ranker. Deliberately contains no
`import torch` anywhere -- the model is trained with torch under
training/models/mlp_ranker.py, then its weights are extracted to plain numpy
arrays (mlp_ranker.npz) so the submitted agent never needs torch installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .base import sigmoid


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class MLPWeights:
    """11 -> 32 (ReLU) -> 16 (ReLU) -> 1 (sigmoid). Dropout is identity at
    eval time, so it doesn't appear here at all."""

    def __init__(
        self,
        W1: np.ndarray, b1: np.ndarray,
        W2: np.ndarray, b2: np.ndarray,
        Wout: np.ndarray, bout: np.ndarray,
        feat_mean: np.ndarray, feat_std: np.ndarray,
    ) -> None:
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.Wout, self.bout = Wout, bout
        self.feat_mean = feat_mean
        self.feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = (np.asarray(X, dtype=np.float64) - self.feat_mean) / self.feat_std
        h1 = _relu(Xn @ self.W1.T + self.b1)
        h2 = _relu(h1 @ self.W2.T + self.b2)
        out = sigmoid(h2 @ self.Wout.T + self.bout)
        return out.reshape(-1)

    def save(self, npz_path: str | Path, json_path: str | Path) -> None:
        np.savez(
            npz_path,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            Wout=self.Wout, bout=self.bout,
        )
        Path(json_path).write_text(
            json.dumps({
                "feat_mean": self.feat_mean.tolist(),
                "feat_std": self.feat_std.tolist(),
                "architecture": "11-32-16-1",
            }, indent=2),
            encoding="utf-8",
        )


def load_mlp(npz_path: str | Path, json_path: str | Path) -> Optional[MLPWeights]:
    npz_path, json_path = Path(npz_path), Path(json_path)
    if not npz_path.is_file() or not json_path.is_file():
        return None
    try:
        blob = np.load(npz_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        return MLPWeights(
            W1=blob["W1"], b1=blob["b1"],
            W2=blob["W2"], b2=blob["b2"],
            Wout=blob["Wout"], bout=blob["bout"],
            feat_mean=np.array(meta["feat_mean"], dtype=np.float64),
            feat_std=np.array(meta["feat_std"], dtype=np.float64),
        )
    except Exception:
        return None
