"""MLP ranker: 11 -> 32 (ReLU) -> Dropout(0.2) -> 16 (ReLU) -> Dropout(0.2)
-> 1. Loss = BCE-with-logits + 0.3 * ApproxNDCGLoss (listwise, sharpness
alpha=10). Adam, lr=1e-3, weight_decay=1e-4, group-aware mini-batches
(~512 rows), early stopping (patience 15) on a held-out slice of groups.

Trained with torch (dev-only). After training, weights are extracted to
plain numpy arrays via `to_mlp_weights()` -- see starter/reranker/
mlp_inference.py for the numpy-only forward pass used at serving time; torch
is never imported there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from starter.reranker.base import FEATURE_NAMES  # noqa: E402
from starter.reranker.mlp_inference import MLPWeights  # noqa: E402

N_FEATURES = len(FEATURE_NAMES)


class _TorchMLP(nn.Module):
    def __init__(self, in_dim: int = N_FEATURES, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _approx_ndcg_loss(logits: torch.Tensor, grades: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    n = logits.shape[0]
    if n < 2:
        return logits.new_zeros(())
    diff = logits.unsqueeze(0) - logits.unsqueeze(1)          # (n, n): s_i - s_j
    sig = torch.sigmoid(alpha * diff)
    sig = sig - torch.diag(torch.diagonal(sig))                 # exclude the i == j term
    approx_rank = 1.0 + sig.sum(dim=1)
    gains = torch.pow(2.0, grades) - 1.0
    dcg = (gains / torch.log2(1.0 + approx_rank)).sum()
    ideal_grades, _ = torch.sort(grades, descending=True)
    ranks = torch.arange(1, n + 1, dtype=torch.float32, device=logits.device)
    idcg = ((torch.pow(2.0, ideal_grades) - 1.0) / torch.log2(1.0 + ranks)).sum()
    if idcg <= 0:
        return logits.new_zeros(())
    return -(dcg / idcg)


class MLPRanker:
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 512,
                 max_epochs: int = 75, patience: int = 15, dropout: float = 0.2,
                 approxndcg_weight: float = 0.3, seed: int = 42):
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.dropout, self.approxndcg_weight = dropout, approxndcg_weight
        self.seed = seed
        self.model_: _TorchMLP | None = None
        self.feat_mean_: np.ndarray | None = None
        self.feat_std_: np.ndarray | None = None
        self.history_: list[float] = []  # per-epoch validation BCE, for plotting

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "MLPRanker":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y_bin = (np.asarray(y) > 0).astype(np.float32)
        grades = np.asarray(y, dtype=np.float32)

        self.feat_mean_ = X.mean(axis=0)
        self.feat_std_ = X.std(axis=0)
        self.feat_std_[self.feat_std_ < 1e-8] = 1.0
        Xn = (X - self.feat_mean_) / self.feat_std_

        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        n_val = max(1, int(round(len(unique_groups) * 0.1)))
        val_groups, train_groups = set(unique_groups[:n_val].tolist()), unique_groups[n_val:]
        train_mask = np.isin(groups, train_groups)
        val_mask = ~train_mask

        model = _TorchMLP(dropout=self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss()

        Xn_t = torch.tensor(Xn)
        y_bin_t = torch.tensor(y_bin)
        grades_t = torch.tensor(grades)

        def group_batches(mask: np.ndarray, shuffle: bool):
            gids = list({int(g) for g in groups[mask]})
            if shuffle:
                rng.shuffle(gids)
            batch: list[int] = []
            for g in gids:
                idx = np.nonzero((groups == g) & mask)[0]
                if len(batch) + len(idx) > self.batch_size and batch:
                    yield np.array(batch)
                    batch = []
                batch.extend(idx.tolist())
            if batch:
                yield np.array(batch)

        best_val, best_state, stale = float("inf"), None, 0
        for _epoch in range(self.max_epochs):
            model.train()
            for batch_idx in group_batches(train_mask, shuffle=True):
                optimizer.zero_grad()
                logits = model(Xn_t[batch_idx])
                loss = bce(logits, y_bin_t[batch_idx])
                for g in np.unique(groups[batch_idx]):
                    sub = batch_idx[groups[batch_idx] == g]
                    if len(sub) >= 2:
                        loss = loss + self.approxndcg_weight * _approx_ndcg_loss(
                            model(Xn_t[sub]), grades_t[sub],
                        ) / max(1, len(np.unique(groups[batch_idx])))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(Xn_t[val_mask])
                val_loss = float(bce(val_logits, y_bin_t[val_mask])) if val_mask.any() else 0.0
            self.history_.append(val_loss)

            if val_loss < best_val - 1e-5:
                best_val, stale = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        Xn = (np.asarray(X, dtype=np.float32) - self.feat_mean_) / self.feat_std_
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(Xn))
            return torch.sigmoid(logits).numpy()

    def to_mlp_weights(self) -> MLPWeights:
        layers = list(self.model_.net.children())
        lin1, lin2, lin3 = layers[0], layers[3], layers[6]
        return MLPWeights(
            W1=lin1.weight.detach().numpy().astype(np.float64),
            b1=lin1.bias.detach().numpy().astype(np.float64),
            W2=lin2.weight.detach().numpy().astype(np.float64),
            b2=lin2.bias.detach().numpy().astype(np.float64),
            Wout=lin3.weight.detach().numpy().astype(np.float64),
            bout=lin3.bias.detach().numpy().astype(np.float64),
            feat_mean=self.feat_mean_.astype(np.float64),
            feat_std=self.feat_std_.astype(np.float64),
        )
