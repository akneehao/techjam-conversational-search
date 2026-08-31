"""Self-supervised label generation, v2 -- aligned with the actual task.

v1 (superseded) built labels from category structure: the "query" was a
product's full title+categories+features text, and the positives were that
product's category *siblings*, with the grade-2 positive chosen as the
sibling most cosine-similar to the query. Evaluated against the official
evaluator, every model trained on it scored below the plain RRF ordering
(see docs/reranker_eval_results.md). Three defects caused that:

  1. Label leakage. Picking the grade-2 positive by "most cosine-similar
     sibling" made the `sibling_max_sim` feature nearly encode the labelling
     rule -- it took 68% of GBDT's gain and 53% of Simplex's weight. The
     models learned how labels were made, not what makes a product relevant.
  2. Task mismatch. Labelling same-category products as relevant teaches a
     category matcher, but the evaluator asks for ONE specific hidden target,
     where same-category-wrong-product is precisely the distractor to beat.
  3. No first-stage signal. BM25/dense/RRF scores were not features, so the
     re-ranker discarded the evidence that makes retrieval strong and had no
     floor -- it could not even reproduce the ordering it was re-ranking.

v2 fixes all three by mirroring deployment exactly:
  * query   = a SHORT, slot-style string built from the product (the shape
              Agent._dense_query_text produces at serving time), not its full
              description text.
  * candidates = whatever the real BM25 + dense + RRF pipeline retrieves for
              that query -- i.e. genuinely hard, plausible distractors.
  * positive = the product itself (grade 2). Everything else retrieved is a
              negative. No feature participates in choosing the label.
  * features = the 11 semantic/structural ones plus bm25/dense/rrf rank
              scores, so "reproduce RRF" is learnable as a floor.

Queries whose own product is not retrieved are skipped: the re-ranker only
reorders retrieved candidates, so those cases are unlearnable and unfixable
at serving time either way.

CLI:
    python -m training.label_generation --catalog data/catalog.jsonl \\
        --out data/reranker_training_data.npz --seed 42 --num-queries 2500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must precede the starter.agent import: it freezes RERANK_* into module-level
# constants at import time and ships with RERANK_ENABLED=0, but training needs
# the category index that _init_reranker builds. Does not affect serving.
os.environ.setdefault("RERANK_ENABLED", "1")

from starter.agent import (  # noqa: E402
    DENSE_WEIGHT,
    LEX_K,
    RERANK_DEPTH,
    RRF_DEPTH,
    RRF_K,
    Agent,
    _tokens,
)
from starter.reranker.base import FEATURE_NAMES  # noqa: E402
from starter.reranker.features import compute_feature_matrix  # noqa: E402
from training.common import load_agent  # noqa: E402

MAX_QUERY_TOKENS = 14
TOKEN_KEEP_RATE = 0.7

# Generic shopper vocabulary. Real customers describe a product with words
# that are not all drawn from its own listing text, so a few of these are
# mixed in. Without them the query would be a strict SUBSET of the product's
# own tokens, which pins lexical_score to exactly 1.0 for every positive --
# a label leak of the same kind that sank v1 (see this module's docstring).
FILLER_TOKENS = (
    "comfortable", "everyday", "casual", "gift", "quality", "affordable",
    "durable", "stylish", "lightweight", "favourite", "work", "travel",
)


def build_query_text(product: dict, rng: np.random.Generator) -> str:
    """A short, slot-style query in the same shape Agent._dense_query_text
    emits at serving time (fragments joined by ' ; '), e.g.
    "hoop earrings ; women jewelry ; stainless steel gift".  Deliberately a
    NOISY, PARTIAL view of the product: tokens are randomly dropped and a
    couple of generic shopper words mixed in, so the positive does not become
    trivially identifiable by perfect lexical overlap."""
    categories = [str(c) for c in (product.get("categories") or []) if str(c).strip()]
    leaf = " ".join(_tokens(" ".join(categories[-2:]))) if categories else ""
    context = " ".join(_tokens(" ".join(categories[1:-2]))) if len(categories) > 3 else ""

    cat_tokens = set(_tokens(" ".join(categories)))
    title_tokens = [t for t in _tokens(str(product.get("title") or "")) if t not in cat_tokens]
    feature_text = " ".join(str(f) for f in (product.get("features") or [])[:4])
    feature_tokens = [
        t for t in _tokens(feature_text) if t not in cat_tokens and t not in set(title_tokens)
    ]

    def drop(tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        keep = [t for t in tokens if rng.random() < TOKEN_KEEP_RATE]
        return keep or tokens[:1]

    parts = [p for p in (leaf, context) if p]
    kept_title = drop(title_tokens[:5])
    if kept_title:
        parts.append(" ".join(kept_title[:4]))
    kept_features = drop(feature_tokens[:4])
    if kept_features:
        parts.append(" ".join(kept_features[:3]))

    n_filler = int(rng.integers(1, 3))
    fillers = rng.choice(np.array(FILLER_TOKENS), size=n_filler, replace=False)
    parts.append(" ".join(str(f) for f in fillers))

    text = " ; ".join(parts)
    tokens = text.split()
    if len(tokens) > MAX_QUERY_TOKENS * 2:
        text = " ".join(tokens[: MAX_QUERY_TOKENS * 2])
    return text[:400]


def bm25_expressions(query_tokens: list[str]) -> list[str]:
    """Tiered FTS5 expressions mirroring Agent._build_queries' shape: a
    narrow AND tier over the most specific terms, then a broad OR net."""
    terms = [t for t in dict.fromkeys(query_tokens) if len(t) > 2][:12]
    if not terms:
        return []
    core = terms[:3]
    tiers = []
    if len(core) >= 2:
        tiers.append("(" + " AND ".join(f'"{t}"' for t in core) + ")")
    tiers.append("(" + " OR ".join(f'"{t}"' for t in terms) + ")")
    return tiers


def retrieve_candidates(agent: Agent, query_text: str) -> tuple[list[str], list[str], list[str]]:
    """Run the real first stage, whichever one is configured.

    Returns ``(candidates, bm25_ranked, dense_ranked)``.

    This MUST mirror ``Agent._rank``: the ordering it produces here is the
    ordering the re-ranker will be asked to improve at serving time, and the
    ``bm25_rank_score`` / ``dense_rank_score`` / ``rrf_score`` features are
    computed from these two lists.  If the two diverge, the model is trained on
    a candidate distribution it never sees in production.

    It previously hardcoded ``_rrf_fuse``, which silently ignored FUSION_MODE.
    With the first stage rewritten (BM25 gates, dense re-ranks within), that
    left training pinned to the old symmetric-RRF ordering while serving used
    the new one -- a train/serve mismatch that no amount of extra training data
    would fix.
    """
    query_tokens = _tokens(query_text)
    expressions = bm25_expressions(query_tokens)
    mode = getattr(agent, "_fusion_mode", "rrf")

    if mode == "rrf":       # legacy symmetric fusion
        bm25_ranked = agent._search(expressions, RRF_DEPTH, limit=RRF_DEPTH) if expressions else []
        dense_ranked = agent._dense_rank(query_text, RRF_DEPTH)
        return Agent._rrf_fuse(bm25_ranked, dense_ranked, k=RRF_K, top_k=None), bm25_ranked, dense_ranked

    # rerank / bm25: BM25 fixes the candidate set, dense only reorders within it
    depth = getattr(agent, "_rerank_depth", None) or RERANK_DEPTH
    bm25_ranked = agent._search(expressions, depth, limit=depth) if expressions else []
    if not bm25_ranked:
        return [], [], []
    if mode == "bm25":
        return list(bm25_ranked), bm25_ranked, []

    head = 1.0 / (LEX_K + 1)
    scores = {a: (1.0 / (LEX_K + r)) / head for r, a in enumerate(bm25_ranked, start=1)}
    dense_ranked: list[str] = []
    if DENSE_WEIGHT > 0.0 and agent._dense_on():
        dense_scores = agent._dense_scores(bm25_ranked, query_text)
        dense_ranked = sorted(dense_scores, key=lambda a: dense_scores[a], reverse=True)
        for asin, sem in Agent._minmax(dense_scores).items():
            scores[asin] += DENSE_WEIGHT * sem
    ordered = sorted(bm25_ranked, key=lambda a: scores.get(a, 0.0), reverse=True)
    return ordered, bm25_ranked, dense_ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out", default="data/reranker_training_data.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-queries", type=int, default=2500,
                         help="Number of catalog products to use as self-supervised queries.")
    parser.add_argument("--max-candidates", type=int, default=60)
    args = parser.parse_args()

    t0 = time.time()
    agent = load_agent(args.catalog)
    cat_index = agent._cat_index
    doc_vecs = agent._doc_vecs
    doc_tokens = agent._doc_tokens
    print(f"Loaded agent + dense index ({len(agent._dense_ids)} vectors) in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    queryable = sorted(cat_index.path_by_asin.keys())
    by_top: dict[str, list[str]] = defaultdict(list)
    for a in queryable:
        by_top[cat_index.path_by_asin[a][0]].append(a)
    # Stratify by top-level category, then redistribute the shortfall.
    #
    # A flat `num_queries // len(by_top)` per bucket silently under-delivers
    # whenever the buckets are uneven, and this catalog is extremely uneven:
    # two top-level nodes sized 10 and 49,990.  Asking for 2,500 used to yield
    # 1,260 (10 from the small bucket + 1,250 from the large one) -- roughly
    # half the requested set, with no warning.  Small buckets are still taken
    # whole (that is the point of stratifying), but their unused quota now
    # flows to buckets that still have products left, so --num-queries means
    # what it says.
    remaining = args.num_queries
    buckets = sorted(by_top.items(), key=lambda kv: len(kv[1]))  # smallest first
    sampled: list[str] = []
    for position, (_top, asins) in enumerate(buckets):
        share = max(1, remaining // (len(buckets) - position))
        take = min(share, len(asins))
        idx = rng.choice(len(asins), size=take, replace=False)
        sampled.extend(asins[i] for i in idx)
        remaining -= take
    query_list = sorted(sampled)[: args.num_queries]
    print(f"Query set: {len(query_list)} products (stratified by top-level category, "
          f"{len(buckets)} buckets)")

    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    all_groups: list[int] = []
    all_query_ids: list[str] = []
    all_candidate_ids: list[str] = []
    accepted: list[str] = []
    not_retrieved = 0
    group_id = 0

    for i, q in enumerate(query_list):
        product = agent._catalog.get(q)
        if not product:
            continue
        query_text = build_query_text(product, rng)
        if not query_text.strip():
            continue

        fused, bm25_ranked, dense_ranked = retrieve_candidates(agent, query_text)
        if q not in fused:
            not_retrieved += 1     # unlearnable: a re-ranker only reorders what retrieval returned
            continue
        candidates = fused[: args.max_candidates]
        if q not in candidates:
            candidates = candidates[: args.max_candidates - 1] + [q]

        raw_qv = agent._embedder.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=False,
        )[0].astype("float32")
        q_norm = float(np.linalg.norm(raw_qv)) or 1.0
        qv = (raw_qv / q_norm).astype("float32")

        X = compute_feature_matrix(
            qv, q_norm, _tokens(query_text), candidates,
            cat_index, doc_vecs, cat_index.id_row, doc_tokens,
            bm25_ranked=bm25_ranked, dense_ranked=dense_ranked, rrf_k=RRF_K,
        )
        all_X.append(X)
        all_y.extend(2 if a == q else 0 for a in candidates)
        all_groups.extend([group_id] * len(candidates))
        all_query_ids.extend([query_text] * len(candidates))
        all_candidate_ids.extend(candidates)
        accepted.append(q)
        group_id += 1

        if (i + 1) % 250 == 0:
            print(f"  ... {i + 1}/{len(query_list)} queries, {len(accepted)} kept, "
                  f"{not_retrieved} not-retrieved, {time.time()-t0:.0f}s")

    X = np.concatenate(all_X, axis=0).astype(np.float32) if all_X else np.zeros((0, len(FEATURE_NAMES)), np.float32)
    y = np.array(all_y, dtype=np.int8)
    groups = np.array(all_groups, dtype=np.int32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X=X, y=y, groups=groups,
        source=np.array(["retrieved"] * len(y), dtype="U16"),
        confidence_tier=np.array(["high"] * len(y), dtype="U8"),
        query_ids=np.array(all_query_ids, dtype=object),
        candidate_ids=np.array(all_candidate_ids, dtype=object),
        query_list=np.array(accepted, dtype=object),
        meta=np.array([{
            "version": 3,
            "num_queries": len(accepted),
            "num_pairs": int(len(y)),
            "pos_rate": float(np.mean(y > 0)) if len(y) else 0.0,
            "not_retrieved": not_retrieved,
            "seed": args.seed,
            "feature_names": list(FEATURE_NAMES),
            # v3: the first stage these candidates came from. A model trained on
            # one mode and served under another sees a different candidate
            # distribution -- record it so artifacts stay traceable.
            "fusion_mode": getattr(agent, "_fusion_mode", "rrf"),
            "rrf_k": RRF_K,
            "rerank_depth": RERANK_DEPTH,
            "dense_weight": DENSE_WEIGHT,
        }], dtype=object),
    )
    print(f"Wrote {out_path}: {len(accepted)} queries, {len(y)} pairs, "
          f"{not_retrieved} skipped (target not retrieved), total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
