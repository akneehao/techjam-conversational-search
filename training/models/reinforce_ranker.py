"""REINFORCE / policy-gradient ranker (pure numpy, deliberately not torch --
this is the lowest-expected-payoff model of the seven, so it isn't worth a
second training framework).

Policy: linear scores s = w.x + b define a Plackett-Luce distribution over
permutations (sequential softmax sampling without replacement). Reward =
NDCG@5 of the sampled permutation's grades. Trained with the REINFORCE
gradient estimator and an EMA baseline for variance reduction, plus a small
per-step entropy bonus to discourage policy collapse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from starter.reranker.base import BASELINE_WEIGHTS, LinearRanker, ndcg_at_k  # noqa: E402

N_FEATURES = len(BASELINE_WEIGHTS)


def _sample_plackett_luce(scores: np.ndarray, rng: np.random.Generator):
    """One sequential-softmax sample of a full permutation. Returns
    (perm, grad_w_coefs, grad_b, entropy) where grad_w_coefs is the (n, n)
    per-step (indicator - prob) matrix needed to accumulate d(log P)/dw."""
    n = len(scores)
    remaining = list(range(n))
    perm: list[int] = []
    entropy = 0.0
    grad_contrib = np.zeros(n)   # coefficient on each original index's features
    grad_b = 0.0
    s = scores - scores.max()
    for _ in range(n):
        rem = np.array(remaining)
        rem_scores = s[rem]
        rem_scores = rem_scores - rem_scores.max()
        exp_s = np.exp(rem_scores)
        probs = exp_s / exp_s.sum()
        choice_local = rng.choice(len(remaining), p=probs)
        chosen = remaining[choice_local]
        perm.append(chosen)
        entropy += float(-np.sum(probs * np.log(probs + 1e-12)))
        indicator = np.zeros(len(remaining))
        indicator[choice_local] = 1.0
        coef = indicator - probs
        grad_contrib[rem] += coef
        grad_b += float(coef.sum())
        del remaining[choice_local]
    return perm, grad_contrib, grad_b, entropy


class ReinforceRanker:
    def __init__(self, lr: float = 0.01, epochs: int = 30, momentum: float = 0.9,
                 entropy_coef: float = 0.01, seed: int = 42, grad_clip: float = 1.0):
        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum
        self.entropy_coef = entropy_coef
        self.seed = seed
        self.grad_clip = grad_clip
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.feat_mean_: np.ndarray | None = None
        self.feat_std_: np.ndarray | None = None
        self.reward_history_: list[float] = []  # one entry per group update, for plotting

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "ReinforceRanker":
        rng = np.random.default_rng(self.seed)
        # Feature standardisation is essential here: the raw features span
        # very different scales (lexical_score ~1.0 vs rrf_score ~0.015), and
        # unnormalised gradient ascent on a softmax policy diverges -- it
        # drove the weights strongly negative on the most informative
        # features and produced a random/inverted ranking.
        X = np.asarray(X, dtype=np.float64)
        self.feat_mean_ = X.mean(axis=0)
        self.feat_std_ = X.std(axis=0)
        self.feat_std_[self.feat_std_ < 1e-8] = 1.0
        X = (X - self.feat_mean_) / self.feat_std_

        w = np.array(BASELINE_WEIGHTS, dtype=np.float64)
        b = 0.0
        baseline = 0.0
        unique_groups = np.unique(groups)

        for _epoch in range(self.epochs):
            order = unique_groups.copy()
            rng.shuffle(order)
            for g in order:
                idx = np.nonzero(groups == g)[0]
                if len(idx) < 2 or not np.any(y[idx] > 0):
                    continue
                Xg, yg = X[idx], y[idx]
                scores = Xg @ w + b
                perm, grad_contrib, grad_b, entropy = _sample_plackett_luce(scores, rng)
                reward = ndcg_at_k(yg[perm], 5)
                self.reward_history_.append(reward)
                shaped_reward = reward + self.entropy_coef * entropy
                advantage = shaped_reward - baseline
                baseline = self.momentum * baseline + (1 - self.momentum) * reward

                grad_w = grad_contrib @ Xg   # (n,) . (n, D) -> (D,)
                update = self.lr * advantage * grad_w
                norm = float(np.linalg.norm(update))
                if norm > self.grad_clip:     # keep a single group from blowing up the policy
                    update *= self.grad_clip / norm
                w += update
                b += float(np.clip(self.lr * advantage * grad_b, -self.grad_clip, self.grad_clip))

        self.weights_, self.bias_ = w, b
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        Xn = (np.asarray(X, dtype=np.float64) - self.feat_mean_) / self.feat_std_
        return Xn @ self.weights_ + self.bias_

    def to_linear_ranker(self) -> LinearRanker:
        """Fold the feature standardisation into the weight vector so the
        artifact stays a plain `score = w.x + b` linear model at serving
        time (LinearRanker applies no normalisation of its own)."""
        w = self.weights_ / self.feat_std_
        b = self.bias_ - float(np.dot(self.weights_, self.feat_mean_ / self.feat_std_))
        return LinearRanker(w, b, name="reinforce")
