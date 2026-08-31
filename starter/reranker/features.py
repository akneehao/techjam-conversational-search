"""The 11-dim feature vector, ported from .pg/SIMILARITY_LEARNING_V2.md's
ontology-node features to this catalog's category-path structure. See
catalog_index.py for how "descendants"/"siblings"/"parent" are defined
without a graph library.

Pure numpy. Reused identically by starter/agent.py at serving time and by
training/label_generation.py + the notebooks at training time, so the
learned models see the exact same feature computation in both places.
"""

from __future__ import annotations

import numpy as np

from .base import FEATURE_NAMES
from .catalog_index import CategoryIndex, descendants_of, immediate_children_of

TOP5 = 5


def _sims_to(query_vec: np.ndarray, doc_vecs: np.ndarray, doc_id_row: dict[str, int], asins) -> np.ndarray:
    rows = [doc_id_row[a] for a in asins if a in doc_id_row]
    if not rows:
        return np.empty(0, dtype=np.float64)
    return doc_vecs[rows] @ query_vec


def _widen_siblings(cat_index: CategoryIndex, path: tuple[str, ...], exclude: str) -> set[str]:
    if len(path) == 0:
        return set()
    return cat_index.by_prefix.get(path[:-1], set()) - {exclude}


def retrieval_rank_scores(
    candidate_asins: list[str],
    bm25_ranked: list[str] | None,
    dense_ranked: list[str] | None,
    rrf_k: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three first-stage features: 1/(k + rank) for each track (0 when a
    candidate is absent from that track) and their sum, which is exactly the
    RRF score the fused ordering is built from."""
    bm25_pos = {a: i for i, a in enumerate(bm25_ranked or [], start=1)}
    dense_pos = {a: i for i, a in enumerate(dense_ranked or [], start=1)}
    bm25 = np.array([1.0 / (rrf_k + bm25_pos[a]) if a in bm25_pos else 0.0 for a in candidate_asins])
    dense = np.array([1.0 / (rrf_k + dense_pos[a]) if a in dense_pos else 0.0 for a in candidate_asins])
    return bm25, dense, bm25 + dense


def compute_feature_matrix(
    query_vec_norm: np.ndarray,
    query_norm_raw: float,
    query_tokens: list[str],
    candidate_asins: list[str],
    cat_index: CategoryIndex,
    doc_vecs: np.ndarray,
    doc_id_row: dict[str, int],
    doc_tokens: dict[str, frozenset[str]],
    bm25_ranked: list[str] | None = None,
    dense_ranked: list[str] | None = None,
    rrf_k: int = 60,
) -> np.ndarray:
    n = len(candidate_asins)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    if n == 0:
        return X

    query_vec_norm = np.asarray(query_vec_norm, dtype=np.float64)
    q_tokens = set(query_tokens)
    q_token_count = len(q_tokens)

    candidate_rows = [doc_id_row.get(a, -1) for a in candidate_asins]
    valid_mask = np.array([r >= 0 for r in candidate_rows])
    candidate_vecs = np.zeros((n, doc_vecs.shape[1]), dtype=np.float64)
    if valid_mask.any():
        candidate_vecs[valid_mask] = doc_vecs[[r for r in candidate_rows if r >= 0]]

    # Feature 10 (query_max_node_sim) and 9/11 are constant across the whole
    # candidate set for a given query -- compute once, not per candidate.
    node_profile_sim = candidate_vecs @ query_vec_norm  # feature 1, vectorized
    query_max_node_sim = float(node_profile_sim.max()) if valid_mask.any() else 0.0
    query_token_count_feat = q_token_count / 10.0

    bm25_feat, dense_feat, rrf_feat = retrieval_rank_scores(
        candidate_asins, bm25_ranked, dense_ranked, rrf_k,
    )

    for i, asin in enumerate(candidate_asins):
        if not valid_mask[i]:
            continue
        path = cat_index.path_by_asin.get(asin, ())

        desc = descendants_of(cat_index, path) if path else set()
        if not desc:
            desc = cat_index.siblings_by_path.get(path, set()) - {asin} if path else set()
        desc_sims = _sims_to(query_vec_norm, doc_vecs, doc_id_row, desc)
        max_desc_sim = float(desc_sims.max()) if desc_sims.size else 0.0
        avg_top5_desc_sim = float(np.sort(desc_sims)[::-1][:TOP5].mean()) if desc_sims.size else 0.0

        children = immediate_children_of(cat_index, path) if path else set()
        child_sims = _sims_to(query_vec_norm, doc_vecs, doc_id_row, children)
        sibling_coherence = float(child_sims.mean()) if child_sims.size else 0.0

        lex_tokens = doc_tokens.get(asin, frozenset())
        lexical_score = (len(q_tokens & lex_tokens) / q_token_count) if q_token_count else 0.0

        path_score = 0.0
        if path:
            weights_sum = 0.0
            acc = 0.0
            for depth in range(1, len(path) + 1):
                centroid = cat_index.prefix_centroid.get(path[:depth])
                if centroid is None:
                    continue
                sim = float(centroid.astype(np.float64) @ query_vec_norm)
                acc += depth * sim
                weights_sum += depth
            path_score = acc / weights_sum if weights_sum else 0.0

        siblings = (cat_index.siblings_by_path.get(path, set()) - {asin}) if path else set()
        if not siblings and path:
            siblings = _widen_siblings(cat_index, path, asin)
        sib_sims = _sims_to(query_vec_norm, doc_vecs, doc_id_row, siblings)
        sibling_max_sim = float(sib_sims.max()) if sib_sims.size else 0.0

        parent_sim = 0.0
        if len(path) >= 1:
            centroid = cat_index.prefix_centroid.get(path[:-1])
            if centroid is not None:
                parent_sim = float(centroid.astype(np.float64) @ query_vec_norm)

        X[i] = (
            node_profile_sim[i],
            max_desc_sim,
            avg_top5_desc_sim,
            sibling_coherence,
            lexical_score,
            path_score,
            sibling_max_sim,
            parent_sim,
            query_token_count_feat,
            query_max_node_sim,
            query_norm_raw,
            bm25_feat[i],
            dense_feat[i],
            rrf_feat[i],
        )

    return X
