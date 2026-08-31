# Learned Re-Ranking Layer for the Conversational Search Agent

## Context

`starter/agent.py` currently fuses BM25 (FTS5) and dense (sentence-transformers)
retrieval with plain, unweighted Reciprocal Rank Fusion (`_rrf_fuse`,
`starter/agent.py:841-849`) — there is no learned component anywhere in the
ranking path. The competition's scoring formula
(`TechnicalScore = 0.50×HitRate@10 + 0.30×MRR + 0.20×Efficiency`) weights MRR
at 30%: a re-ranker that reorders the existing top-N candidates to push the
true target higher directly improves MRR without needing to touch recall at
all. That's the highest-leverage lever currently unused in the codebase.

This plan ports the ranking methodology documented (prose only, no source
code) in `.pg/SIMILARITY_LEARNING_V2.md` (an internship project that ranked
olfactory-ontology nodes against natural-language queries) to this catalog's
product/category structure. The user explicitly wants **all** model types
from that doc (baseline + 6 learned rankers — not just GBDT/MLP), a real
feature-extraction pipeline built from `data/catalog.jsonl`, self-supervised
label generation, and Jupyter notebooks to inspect model performance —
verified end-to-end against the **unmodified** `evaluator/local_evaluator.py`
so the result stays compliant with hackathon rules (no evaluator edits, no
full foundational-model training, in-memory only, graceful offline/CPU-only
fallback).

**Scope decision (confirmed with user):** synthetic out-of-distribution
(free-text) query generation via LLM annotation is **out of scope for this
pass** — no `GEMINI_API_KEY` is currently configured in this repo, and the
original project's 6,000-query multi-annotator Dawid-Skene pipeline is too
costly to replicate for a hackathon timeline. Training uses **only the
in-catalog self-supervised labels** (derived from the 50,000-product
category structure, zero API cost). A scoped-down synthetic-query approach is
documented as a **future-work recommendation** (§7) rather than built now.

Every technical claim about `starter/agent.py` below (line numbers, function
signatures, data shapes) has been verified directly against the current file
contents, not just the prior internship docs.

---

## Category-structure adaptation (no graph DB needed)

Every product's `categories` field is already an ordered root→leaf path
(e.g. `["Clothing, Shoes & Jewelry","Women","Jewelry","Earrings","Hoop"]`).
Unlike the internship's true multi-parent ontology DAG (which needed a real
graph DB + BFS), this is a simple tree path per product — a couple of
`defaultdict`s built once at index time replace the whole graph layer:

- `siblings_by_path[full_path] -> set[parent_asin]` (exact full-path match)
- `by_prefix[path[:d]] -> set[parent_asin]`, for every realized depth `d`
- `prefix_centroid[path[:d]] -> L2-normalized mean doc vector`, precomputed
  once per realized prefix (avoids re-scanning thousands of same-parent
  products per candidate per query for `parent_sim`/`path_score`)

Feature adaptations (11-dim vector, names/order ported verbatim from the doc):

| # | Feature | Adaptation |
|---|---|---|
| 1 | `node_profile_sim` | cosine(query_vec, candidate_vec) — direct port |
| 2 | `max_desc_sim` | max cosine to products whose path is a strict extension of candidate's path; falls back to same-full-path siblings when none exist (most products sit at max realized depth) |
| 3 | `avg_top5_desc_sim` | mean cosine over top-5 most similar such products |
| 4 | `sibling_coherence` | mean cosine to products one level deeper than candidate (`len(path)+1`, prefix match); expected near-zero for most leaf-depth products — document as expected, not a bug |
| 5 | `lexical_score` | reuse `starter.agent._tokens` — query-token overlap fraction against candidate's title+features+category+description tokens (precomputed once per product) |
| 6 | `path_score` | depth-weighted mean cosine to `prefix_centroid[path[:d]]` for every `d`, deeper weighted more |
| 7 | `sibling_max_sim` | max cosine to `siblings_by_path[path] - {candidate}`; widens to same-parent-prefix set if the exact-path sibling set is a singleton |
| 8 | `parent_sim` | cosine to `prefix_centroid[path[:-1]]` |
| 9 | `query_token_count` | `len(query_tokens) / 10` |
| 10 | `query_max_node_sim` | max cosine over the *current candidate set* (~60-120 items, not full 50k) |
| 11 | `query_embedding_norm` | L2 norm of the raw (non-normalized) query embedding |

Baseline weight vector (reference, sums to 1.0):
`[0.15, 0.12, 0.10, 0.08, 0.06, 0.27, 0.10, 0.06, 0.02, 0.02, 0.02]`.

**Efficiency note:** encode the query once with `normalize_embeddings=False`,
derive both the raw norm (feature 11) and the normalized vector (all cosine
features) from that single call — don't call `.encode()` twice.

**First validation step before building anything else:** a data-exploration
notebook cell histogramming category-path depths and sibling/descendant/
parent group sizes across all 50k products, to confirm these adaptations
aren't degenerate (mostly-empty) before the rest of the pipeline depends on
them. This is the single highest-risk assumption in the whole plan.

---

## File layout

```
starter/
  agent.py                    # modified in place — integration only, see below
  reranker/                   # NEW — ships with the submission (runtime package)
    __init__.py                # load_reranker(), RerankerBundle
    base.py                    # BaseRanker protocol, FEATURE_NAMES, BASELINE_WEIGHTS,
                                #   BaselineRanker, ndcg_at_k/mrr/hits_at_k (numpy-only)
    catalog_index.py           # CategoryIndex, build_category_index()
    features.py                 # compute_feature_matrix() (numpy-only)
    mlp_inference.py            # pure-numpy MLP forward pass — NEVER imports torch
    gbdt_inference.py           # guarded `import lightgbm`; None on failure
    ensemble.py                  # minmax_normalize, threshold_union
    artifacts/                  # small persisted model files, checked into git
      simplex_weights.json, ranksvm_weights.json, coord_ascent_weights.json,
      reinforce_weights.json, mlp_ranker.npz, mlp_ranker.json,
      gbdtranker.txt, gbdtranker_feature_importances.json,
      ensemble_thresholds.json

training/                     # NEW — dev-only, not needed to run the submitted agent
  common.py                    # load_catalog_and_vectors(), group_split(seed=42)
  evaluate.py                  # Hits@K/MRR/NDCG@K/MAP@K, evaluate_groups()
  label_generation.py          # self-supervised in-catalog label builder (CLI)
  train_all.py                 # trains all 7 models + baseline, writes artifacts (CLI)
  models/
    simplex_ranker.py, ranksvm.py, coord_ascent.py, mlp_ranker.py,
    reinforce_ranker.py, gbdt_ranker.py

notebooks/                    # NEW
  training_pipeline.ipynb
  model_comparison.ipynb

data/
  reranker_training_data.npz   # generated — in-catalog self-supervised labels

tests/
  test_reranker_fallback.py    # NEW — graceful-degradation + wiring checks

requirements.txt               # extended (runtime: + lightgbm, optional)
requirements-dev.txt            # NEW — training/notebook-only deps
```

`training/`, `notebooks/`, `requirements-dev.txt` are dev tooling and can be
dropped from a minimal submission zip without breaking `agent.py` — matches
`docs/submission_rules.md`'s recommended layout while keeping the methodology
visible in the full repo for judges.

---

## Label generation (in-catalog only, per scope decision)

`training/label_generation.py` (CLI: `--catalog --out --seed 42 --neg-per-query --query-sample-frac`).
Every catalog product is a self-supervised "query" (all 50k, or a stratified
sample bounded by `--query-sample-frac`, stratified by top-level category):

```
siblings = siblings_by_path[path] - {q}
if siblings:
    primary   = argmax_s cosine(vec[q], vec[s])                    -> grade 2
    secondary = (siblings - {primary}) + top-3 parent-level extras -> grade 1
else:
    parent_candidates = by_prefix[path[:-1]] - {q}
    if parent_candidates:
        primary = argmax by sim -> grade 2; remaining (top 10) -> grade 1
    else:
        skip q (degenerate top-level orphan), log it

negatives = sample_negatives(q, exclude=positives∪{q}, k=NEG_PER_QUERY)
    60% hard: same top-level category, different subcategory
    40% easy: uniform random from catalog
    cap total candidates/query at ~60
```

Output `.npz` schema (mirrors the internship's `training_data_v2.npz`):

| Key | dtype | shape |
|---|---|---|
| `X` | float32 | (N_pairs, 11) |
| `y` | int8 | (N_pairs,) — 0/1/2 |
| `groups` | int32 | (N_pairs,) |
| `query_ids`, `candidate_ids` | object | (N_pairs,) |
| `query_list` | object | (N_queries,) |
| `meta` | object | (1,) — counts, pos_rate, seed, catalog hash |

Train/test split: 80/20, **group-level** (never split one query's candidates
across train/test), seeded (`rng=42`), matching the internship's procedure.

---

## Models (all 7, per explicit user request)

Shared protocol (`starter/reranker/base.py`):
`fit(X, y, groups)`, `predict_scores(X)`, `get_weights()`, `save(path)`,
`load(path)`.

| # | Model | Formulation | Train-time dep | Persist format | Runtime path |
|---|---|---|---|---|---|
| 0 | Baseline | fixed dot product, `BASELINE_WEIGHTS` | none | hardcoded | `starter/reranker/base.py` |
| 1 | Simplex-Constrained Linear | `w≥0, Σw=1`; loss = class-balanced BCE + 0.01·‖w‖² + 0.5·pairwise log-loss; scipy SLSQP, warm-started from baseline, max_iter=500 | scipy (dev) | `simplex_weights.json` | numpy dot — no scipy at runtime |
| 2 | Pairwise RankSVM | `Δx=x_pos-x_neg` per group (cap 50 pairs/query) → `sklearn.svm.LinearSVC(C=1.0)` | sklearn (dev) | `ranksvm_weights.json` | numpy dot — no sklearn at runtime |
| 3 | Coordinate Ascent | gradient-free, directly optimizes MRR; coordinate-wise ±δ search from baseline, halve δ on no improvement | pure numpy | `coord_ascent_weights.json` | numpy dot |
| 4 | MLP | 11→32(ReLU)→Dropout(0.2)→16(ReLU)→Dropout(0.2)→1; loss = BCE + 0.3·ApproxNDCG (α=10); Adam lr=1e-3, wd=1e-4, batch≤512, ~75 epochs, early stop patience 15 | torch (dev only) | `mlp_ranker.npz` (extracted W/b arrays) + `mlp_ranker.json` (feature mean/std) | **hand-written numpy forward pass** in `mlp_inference.py` — no `import torch` anywhere in the runtime package |
| 5 | REINFORCE | linear+softmax policy, reward=NDCG@5 of sampled ranking, REINFORCE gradient + EMA baseline (momentum 0.9), entropy bonus 0.01 | pure numpy (deliberately, not torch — avoids a second training framework for the lowest-expected-payoff model) | `reinforce_weights.json` | numpy dot + bias |
| 6 | GBDT | LightGBM LambdaRank, `label_gain=[0,1,3]`; defaults `n_estimators=100, learning_rate=0.05, num_leaves=31, min_data_in_leaf=5`; optional small 3×3 grid search only if time remains | lightgbm (dev + optional runtime) | `gbdtranker.txt` (native booster) + feature-importances JSON | `gbdt_inference.py` — guarded `try: import lightgbm except ImportError: None` |

**Operationally concrete point:** MLP is numpy-only at inference by
construction (weights extracted via `state_dict()` → numpy → `.npz`, forward
pass hand-written). GBDT is the *only* model with a real optional runtime
dependency, isolated to one small guarded-import file. Neither torch nor
lightgbm nor scipy nor sklearn is ever imported by `starter/agent.py` itself.

**Ensembling** (`starter/reranker/ensemble.py`): threshold-union of
GBDT+MLP scores, min-max normalized per query. The internship's `0.72/0.85`
thresholds were tuned on a different dataset — **re-derive** them by sweeping
a small grid against held-out validation NDCG@5, save winners to
`ensemble_thresholds.json`. If GBDT is unavailable at runtime, degrade to
MLP-only-with-threshold; if nothing clears either threshold, fall back to
top-10 by GBDT (or MLP) score.

---

## `starter/agent.py` integration (verified against current file)

**A. New constants** (near line 156, alongside `RRF_DEPTH`/`RRF_K`):
`RERANK_ENABLED` (env, default on), `RERANK_MODEL` (default `"ensemble"`),
`RERANK_ARTIFACTS_DIR`, `RERANK_CANDIDATES` (default 60).

**B. Guarded import** (near lines 12-19, same pattern as existing optional imports):
`try: from .reranker import load_reranker except ImportError: load_reranker = None`.

**C. Fold `self._catalog: dict[str, dict]` into the *existing* `_build_index`
loop** (`starter/agent.py:439-461`) — at line 452 where
`product = json.loads(line)` is already parsed, add
`self._catalog[str(product.get("parent_asin") or "")] = product` right
there. This avoids a third full pass over the 50k-row catalog (the file is
already read once in `_build_index` and, on a dense-cache miss, a second time
in `_init_dense`; folding into the first loop keeps `self._catalog`
populated exactly once regardless of cache state).

**D. `Agent.__init__`** (after line 421, `self._init_dense(use_dense)`):
add `self._cat_index = None`, `self._dense_id_row: dict[str, int] = {}`,
`self._reranker = None`, then call `self._init_reranker()`.

**E. New `_init_reranker()` method**, mirroring `_init_dense`'s exact
try/except/no-op structure (guard on `RERANK_ENABLED`, `load_reranker`
availability, and `self._dense_on()` since features need dense vectors);
any exception leaves `self._reranker = None`.

**F. Modify the hybrid-retrieval block, `starter/agent.py:976-979`:**
change `fused = self._rrf_fuse(bm25_ranked, dense_ranked, k=RRF_K, top_k=top_k)`
to `top_k=None` (verified: `_rrf_fuse` already returns the *full* unsliced
order when `top_k` is falsy — `starter/agent.py:849` — so this is a safe,
minimal change, not new fusion logic). Then, if `self._reranker is not
None`, re-score `fused[:RERANK_CANDIDATES]` via a new `_rerank()` method and
splice the reordered head back in front of the remaining tail; wrap in
try/except so a bad turn falls back to the original RRF order; finally slice
to `top_k` before building `recommendations`.

**G. New `_rerank(candidates, query_text)` method**: one embedder call
(`normalize_embeddings=False`) to get raw query vector + norm, tokenize via
existing `_tokens`, call `compute_feature_matrix(...)`, call
`self._reranker.rank(X, candidates)`.

**H. Fallback behavior** (matches `_init_dense`'s philosophy exactly):
`RERANK_ENABLED=0`, dense off, numpy/sentence-transformers missing,
artifacts missing/corrupt, or lightgbm missing for gbdt/ensemble mode →
`self._reranker is None` → `respond()` behaves identically to the current
unmodified code, zero behavior change. Any exception during a specific
turn's `_rerank()` call is caught locally; that turn keeps its original RRF
order.

---

## Notebooks

**`notebooks/training_pipeline.ipynb`** — sections, in order:
1. Setup/imports from `starter.reranker.*`, `training.*`.
2. **Data-exploration cell first**: category-path depth histogram, sibling/
   descendant/parent group-size distributions — validates the feature
   adaptations aren't degenerate before anything else is built.
3. Load catalog + dense vectors (reuse `data/dense_*.npz` cache convention).
4. Build `CategoryIndex`, run label generation, show class balance / group
   size histogram.
5. 80/20 group-level split (seed 42).
6. Train each of the 7 models in its own section (Baseline, Simplex,
   RankSVM, CoordAscent, MLP with loss curve, REINFORCE with reward curve,
   GBDT with feature-importance chart; optional grid search behind a
   `RUN_GRID_SEARCH = False` flag).
7. Persist all artifacts to `starter/reranker/artifacts/`.
8. Derive + save ensemble thresholds via validation sweep.
9. Preview table: baseline vs. each model's Hits@1/NDCG@5 on held-out split.

**`notebooks/model_comparison.ipynb`** — sections, in order:
1. Load data + trained artifacts; recompute the *identical* held-out split
   (shared `group_split(seed=42)` helper — must match the training notebook).
2. Score every model + ensemble; **equivalence-check cell** asserting
   `np.allclose(training_time_scores, runtime_numpy_scores)` for MLP and
   GBDT — proves the extracted-weights runtime path matches the trained model.
3. Comparison table (pandas): Hits@1/3/5/10, MRR, NDCG@5/10, MAP@5/10 per
   model, sorted by NDCG@5. Bar/line plots (matplotlib).
4. Qualitative spot-checks: 5-10 sample queries, top-10 titles per model
   side by side.
5. GBDT feature-importance chart + short interpretation.
6. Final recommendation cell: which `RERANK_MODEL` to default to, with
   justification.
7. Explicit disclaimer cell: these are offline proxy metrics over
   engineered-feature grades, **not** the official TechnicalScore — final
   validation must run `evaluator/local_evaluator.py` unmodified (below).

---

## `requirements.txt` / `requirements-dev.txt`

`requirements.txt` (runtime, append with a Day-5-labeled comment like the
existing Day 1-4 comments): `lightgbm>=4.0` — optional; without it (or with
`RERANK_ENABLED=0`) the agent falls back to the Day 4 RRF-fused order
exactly as today.

`requirements-dev.txt` (NEW, dev/training/notebook-only, never required for
official scoring): `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.0`,
`lightgbm>=4.0`, `jupyter`, `ipykernel`, `pandas`, `matplotlib`.
(`scipy`/`scikit-learn`/`torch` are already installed in `tiktokEnv`;
`lightgbm`/`jupyter`/`ipykernel`/`pandas`/`matplotlib` are not and need
`pip install`.)

---

## Verification plan (sequenced)

1. `python -m training.label_generation --catalog data/catalog.jsonl --out data/reranker_training_data.npz` — sanity-print group count / positive rate.
2. Run `notebooks/training_pipeline.ipynb` end-to-end → produces all
   `starter/reranker/artifacts/*`.
3. Run `notebooks/model_comparison.ipynb` → confirms runtime-numpy path
   matches training-time scores (MLP/GBDT equivalence asserts), produces the
   comparison table, picks the `RERANK_MODEL` default.
4. Add `tests/test_reranker_fallback.py`: (a) with artifacts missing,
   `agent._reranker is None` and `respond()` still returns valid
   recommendations; (b) with real artifacts present, reranked order differs
   from pure-RRF order on at least one query.
5. **Control**: `RERANK_ENABLED=0 python -m evaluator.local_evaluator --output results_no_rerank.json` (unmodified evaluator).
6. **Treatment**: `RERANK_ENABLED=1 python -m evaluator.local_evaluator --output results_with_rerank.json`.
7. Compare `technical_score` across: `docs/baseline_results.json` (0.10671,
   weak-BM25 floor) → no-rerank control (current hybrid agent) → with-rerank
   treatment. **Require treatment ≥ control before defaulting
   `RERANK_ENABLED=1`**; if it regresses, ship with `RERANK_ENABLED=0` as
   default and document the finding honestly as a limitation.
8. `python -m unittest discover tests` — confirm nothing else broke.
9. Update `README.md` with model choice, artifact sizes, the GBDT
   optional-with-fallback disclosure, and the before/after `TechnicalScore`
   numbers from steps 5-6 (satisfies the hackathon's cost/latency/token
   disclosure requirement — note training is entirely offline and adds zero
   live inference cost/latency/tokens).

Hackathon-compliance notes baked into this plan: `evaluator/` is never
modified (read-only reference only); no foundational-LLM training occurs
(all 7 models are small local rankers over engineered features); everything
runs in-memory with no external vector DB; the reranker degrades gracefully
under CPU-only/no-network/missing-artifact conditions, following the exact
pattern already established by `_init_dense`.

---

## Scope/effort risk flags

- **Highest risk**: category-path depth may be too uniform, making
  `max_desc_sim`/`sibling_coherence`/`parent_sim` degenerate. Mitigated by
  running the depth/group-size histogram *first*, before committing to the
  rest of the pipeline.
- **REINFORCE**: lowest expected payoff (same linear expressiveness as
  Simplex/CoordAscent/RankSVM) but explicitly requested — pure numpy,
  budget ~1-2 hours.
- **GBDT grid search**: optional; timebox to a small 3×3 sweep only if time
  remains after all 7 models + integration are done; otherwise ship fixed
  defaults.
- **Priority order if time runs short**: (1) label gen + features — everything
  depends on it → (2) Baseline/Simplex/RankSVM/CoordAscent (cheap, fast) →
  (3) MLP + GBDT (likely best performers) → (4) `agent.py` integration +
  evaluator validation (must-do) → (5) REINFORCE, notebook polish, optional
  grid search.

---

## Future work: synthetic natural-language OOD queries (not built now)

The in-catalog labels only teach the models "which products are structurally
similar," not "how a real shopper's free-text phrasing maps to those
products." Recommended follow-up, scoped down from the internship's
6,000-query/7-annotator Dawid-Skene pipeline:

1. Set up `GEMINI_API_KEY` (same env var `starter/agent.py`'s Day-2 router
   already reads — no new config plumbing needed).
2. Generate ~150-300 synthetic natural-language queries: mostly free
   template-filled variations (e.g. `"I'm looking for {category}. A key
   requirement is: {constraint}."`, styled after — but not imported from —
   `evaluator/local_evaluator.py`'s `initial_message`/`customer_reply`
   phrasing patterns), optionally paraphrased by a small Gemini pass on a
   subset.
3. For each synthetic query, retrieve ~20-30 candidates via the **already-
   built** `Agent._bm25_ranked`/`_dense_ranked`/`_rrf_fuse` (instantiate one
   `Agent`, reuse it — don't rescan the catalog).
4. **One batched Gemini call per query** (all candidate titles in a single
   prompt) asking for a direct graded 0/1/2 relevance label per candidate —
   same `urllib` REST pattern as `_call_router`, reimplemented standalone in
   `training/synthetic_queries.py`. This directly yields hard grades,
   **skipping Dawid-Skene entirely** — a deliberate, disclosed simplification,
   not a partial multi-annotator implementation.
5. Cache raw judgments to `data/gemini_ood_queries.json` (idempotent reruns),
   fold into a `data/reranker_synthetic_data.npz` with the same schema as
   the in-catalog data (`source="synthetic"`), append **only to the training
   split** after the 80/20 group split (test set stays untouched).
6. This step is entirely offline/dev-time — never called from `agent.py` —
   so it adds no live inference cost/latency/tokens; disclose the one-time
   API cost (~150-300 calls at a low-cost Gemini tier) in the README if
   implemented.
7. If pursued further, true Dawid-Skene multi-annotator denoising (multiple
   LLM providers/prompts per query, joint sensitivity/specificity + latent
   label estimation) remains the documented stretch goal beyond this
   scoped version.
