"""Shared loading/splitting helpers for the training scripts and notebooks.

Deliberately thin: rather than re-implementing catalog loading, the dense
embedding cache, tokenization, or category indexing a second time, this just
instantiates `starter.agent.Agent` (LLM off) and reuses whatever it already
builds -- the exact same FTS5 index, `data/dense_*.npz` cache, catalog dict,
category index, and per-product token sets that the live agent uses at
serving time. That guarantees training-time features are computed identically
to serving-time features.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# starter.agent reads RERANK_* into module-level constants at import time, and
# ships with RERANK_ENABLED=0 (see the Day 5 comment there). Training always
# needs the category index / doc tokens that _init_reranker builds, so opt in
# BEFORE the import -- this does not change the serving default.
os.environ.setdefault("RERANK_ENABLED", "1")

from starter.agent import Agent  # noqa: E402


def load_agent(catalog_path: str | Path = "data/catalog.jsonl") -> Agent:
    """Build (or reuse the cached) dense index + category index. Raises if
    dense retrieval isn't available -- the reranker needs embeddings, so
    there's no degraded offline path for *training* (unlike serving, where
    everything gracefully falls back)."""
    agent = Agent(str(catalog_path), use_llm=False)
    if not agent._dense_on():
        raise RuntimeError(
            "Dense retrieval unavailable (need numpy + sentence-transformers, "
            "and a downloadable/local all-MiniLM-L6-v2). Training the "
            "re-ranker requires embeddings."
        )
    if agent._cat_index is None:
        raise RuntimeError(
            "agent._cat_index was not built -- check RERANK_ENABLED isn't "
            "set to 0 in the environment used to run this script."
        )
    return agent


def group_split(groups: np.ndarray, test_frac: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Group-level train/test split: never splits one query's rows across
    both sides. Returns (train_mask, test_mask) boolean arrays aligned to
    `groups`."""
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_frac)))
    test_groups = set(shuffled[:n_test].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    return ~test_mask, test_mask


def dense_group_ids(groups: np.ndarray) -> np.ndarray:
    """Remap arbitrary group ids to a dense 0..G-1 range (used after
    filtering/concatenating .npz files so group ids stay contiguous)."""
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    return inverse.astype(np.int32)


def group_kfold_splits(groups: np.ndarray, n_folds: int = 5, seed: int = 42):
    """Group-level K-fold (a plain simplification of the source
    methodology's tier-stratified K-fold -- with only in-catalog,
    "high"-confidence labels in this v1, there's no tier to stratify by).
    Yields (train_mask, val_mask) pairs."""
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)
    for i in range(n_folds):
        val_groups = set(folds[i].tolist())
        val_mask = np.isin(groups, list(val_groups))
        yield ~val_mask, val_mask
