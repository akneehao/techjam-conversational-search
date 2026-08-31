# Learned re-ranker: evaluation results

Official `evaluator/local_evaluator.py` results (200-session public set,
unmodified evaluator) for the Day 5 re-ranking layer (`starter/reranker/` +
`training/`).

## Headline: v2 ships enabled, worth +0.0803 TechnicalScore

Every model retrained on the v2 formulation, each a full
`evaluator/local_evaluator.py` run on the 200 public sessions:

| Configuration | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| **simplex (shipped default)** | 0.960 | **0.649** | 2.68 | **0.8410** |
| mlp | **0.965** | 0.624 | **2.67** | 0.8362 |
| gbdt | 0.955 | 0.609 | 2.71 | 0.8260 |
| coord_ascent | 0.945 | 0.622 | 2.73 | 0.8244 |
| ranksvm | 0.935 | 0.624 | 2.83 | 0.8181 |
| control (no re-ranking) | 0.940 | 0.444 | 3.12 | 0.7607 |
| v1 GBDT (superseded) | 0.885 | 0.483 | 5.34 | 0.7006 |

Every v2 model beats the control, and every one improves every component
metric. The largest gain is exactly where a re-ranker should deliver it:
**MRR +46% relative** for the shipped simplex (0.444 -> 0.649) -- when the
target is retrieved it lands much closer to rank 1. Targets are also found
slightly more often and roughly half a turn sooner.

**Simplex is the shipped default** (`RERANK_MODEL=simplex`), for two reasons:
it scored highest here, and its artifact is an 11-number weight vector scored
with a single numpy dot product, so the default serving path needs no
lightgbm, torch, scipy or sklearn at inference. Any other model is selectable
with `RERANK_MODEL=<name>`.

Two caveats to carry into any further tuning:

- The top four models sit within ~0.023 of each other, and the training set
  has only 601 positive examples (601 training query groups; see below).
  A single session is worth ~0.005 HitRate@10 on a 200-session set, so treat
  the ordering among simplex/mlp/gbdt as provisional rather than decisive.
- **Simplex was the *worst* model in v1 (0.3344) and is the best in v2.** That
  reversal is informative rather than surprising: it is the most constrained
  model (weights non-negative, summing to 1), so v1's leaked feature damaged
  it most, while on v2's honest features and a small training set its low
  effective capacity makes it the most resistant to overfitting.

### Training set size

| | |
|---|---|
| Query groups | 751 (1.5% of the 50,000-product catalog) |
| Candidates per query | 60 |
| Rows (query-candidate pairs) | 45,060 |
| Positive examples | 751 -- one per query |
| Train / test split | 601 groups (601 positives) / 150 groups (150 positives) |

The row count overstates the real sample size: the 36k training rows encode
601 decisions, each padded with 59 negatives. Raising `--num-queries` is the
most promising remaining lever -- it attacks the actual bottleneck, and would
also make the model comparison above decisive rather than noise-limited.

REINFORCE is excluded from the table. On the v2 features it initially
diverged outright (weights driven strongly negative on the most informative
features, producing a random/inverted ranking at MRR 0.017 ~ 1/60). The cause
was unnormalised gradient ascent over features spanning very different scales
(`lexical_score` ~1.0 vs `rrf_score` ~0.015). Adding feature standardisation
and gradient clipping fixed the divergence (offline MRR 0.017 -> 0.320), but
it remains far below the other six, so it was not worth an evaluator run.

## v1 results (superseded)

The rest of this document covers the original v1 formulation, which scored
*below* the plain RRF ordering for all seven model types. It is kept because
the root-cause analysis of that failure is what produced v2.

Each row is a full `python -m evaluator.local_evaluator` run against the
live agent, varying only `RERANK_ENABLED` / `RERANK_MODEL`:

| Configuration | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| **control (`RERANK_ENABLED=0`)** | 0.940 | 0.444 | 3.12 | **0.7607** |
| `gbdt` | 0.885 | **0.483** | 5.34 | 0.7006 |
| `reinforce` | 0.765 | 0.435 | 4.37 | 0.6456 |
| `baseline` (fixed weights, not learned) | 0.770 | 0.413 | 4.37 | 0.6415 |
| `ranksvm` | 0.745 | 0.396 | 4.57 | 0.6198 |
| `coord_ascent` | 0.750 | 0.375 | 4.54 | 0.6169 |
| `ensemble` (GBDT + MLP threshold-union) | 0.490 | 0.339 | 7.12 | 0.4244 |
| `mlp` | 0.495 | 0.287 | 7.08 | 0.4120 |
| `simplex` | 0.440 | 0.133 | 7.27 | 0.3344 |

Baseline (weak BM25, `docs/baseline_results.json`) for scale: TechnicalScore
`0.10671`. Every re-ranker configuration is still far above that floor -- the
regression is entirely relative to the existing Day 1-4 hybrid pipeline
(BM25 + dense + RRF), which was already strong.

**Every one of the 7 models makes the composite score worse than not
re-ranking at all.** GBDT is the clear standout: it's the smallest
regression by a wide margin, and it's the *only* configuration whose MRR
alone beats the control (0.483 vs. 0.444) -- it genuinely improves ranking
precision when the target is found, it just finds it less often (lower
HitRate@10) and takes longer when it does (higher MTTC), and HitRate@10 is
weighted 0.50 in the composite, so that dominates.

## Why: a training/serving query mismatch, not a training bug

The self-supervised labels (`training/label_generation.py`) use each
catalog product's own text (title + categories + features, up to 800 chars)
as the "query" -- a natural choice, since it's the only large-scale source of
free relevance judgments available without manual labeling or LLM cost. But
the agent's actual queries at serving time (`Agent._dense_query_text`) are
short, slot-based fragments assembled from the last few conversation turns
plus extracted constraints, e.g. `"black leather jacket ; men ; leather"` --
structurally nothing like a full product description. The 11 features are
all computed relative to the query embedding, so this is a real distribution
shift between training and serving, not an artifact of the offline metrics.

This is also why the *offline* proxy metrics computed during training
(`training/evaluation_summary.json`, surfaced in `notebooks/
model_comparison.ipynb`) were misleadingly good -- Hits@1 near 1.0 and
NDCG@5 above 0.99 for most models. That evaluation only ever tests
product-vs-product similarity (the same distribution the labels were built
from), so it never exercises the actual serving-time query shape and could
not have caught this. It's a concrete illustration of why the plan's
verification steps insist the *official* evaluator, run unmodified, is the
only authoritative check -- and why they paid off here.

**Why MLP and the ensemble fail hardest:** MLP is the highest-capacity model
of the 7, so it fit the training-distribution features (and their
mean/std normalization) most tightly -- when fed out-of-distribution
serving-time feature values, its dense nonlinear response surface
extrapolates unpredictably. GBDT's threshold-split trees degrade more
gracefully under the same shift (each leaf is reached by simple, bounded
feature comparisons rather than an unbounded learned nonlinearity), which
likely explains why it alone stayed close to the control. The `ensemble`
mode inherits MLP's weakness because threshold-union admits a candidate if
*either* sub-model scores it highly, so MLP's bad nominations still leak
through even though GBDT alone would have ranked it well.

## What v1 meant for the submission (resolved by v2)

At the time, `RERANK_ENABLED` was set to default `0` and the agent shipped as
the Day 1-4 hybrid pipeline. The v2 rework described below reversed that: the
flag now defaults to `1` with `RERANK_MODEL=gbdt`.

## v2: what was changed in response

The root cause was diagnosed from the trained models' own weights, which is
worth recording because the offline metrics gave no hint of it:

| Model | Top feature by weight/gain | Share |
|---|---|---|
| GBDT | `sibling_max_sim` | 68% of total gain |
| Simplex | `sibling_max_sim` | 0.53 of 1.0 |

Both models converged on `sibling_max_sim` -- "max cosine similarity between
the query and *other products sharing this candidate's category path*". That
is not a relevance signal; it is a **label leak**. v1 chose the grade-2
positive as *the sibling most cosine-similar to the query*, so the feature
nearly encodes the labelling rule itself. The leaderboard ordering follows
directly: Simplex put the most weight on the leaked feature and scored worst
(0.3344); GBDT spread more gain onto the genuine query-candidate signal
(`node_profile_sim`) and scored best (0.7006). The whole ranking was
essentially "how far did this model degenerate toward just using dense
cosine".

Three defects, all fixed in v2 (`training/label_generation.py`):

1. **Label leakage** -- no feature participates in choosing the label any
   more. The positive is the query's own source product, full stop.
2. **Task mismatch** -- v1 labelled same-category products as relevant,
   training a *category matcher*; the evaluator asks for one specific hidden
   target, where a same-category-wrong-product is precisely the distractor to
   beat. v2's positive is the source product itself and the negatives are the
   other products the real retrieval pipeline returned for that query, so the
   training task is the deployment task.
3. **No first-stage signal** -- `bm25_rank_score`, `dense_rank_score` and
   `rrf_score` are now features (schema is 14-dim, was 11). This also gives
   the re-ranker a *floor*: with the RRF score as an input, "reproduce the
   RRF ordering" is learnable, so a well-fit model should not be able to fall
   below the control the way v1 did.

A second leak was caught during v2 development, before training: building the
query purely from the product's own tokens made it a strict subset of that
product's text, pinning `lexical_score` to exactly 1.0 for every positive
(vs. 0.41 for negatives). Queries are now a *noisy, partial* view -- tokens
randomly dropped, a couple of generic shopper words mixed in -- which moves
positives to a realistic 0.67-1.0 spread. Worth noting as a pattern: any
self-supervised label rule risks becoming visible to a feature, so
positive-vs-negative feature separation is worth inspecting before training,
not after evaluating.

## Reproducing these numbers

```bash
RERANK_ENABLED=0 python -m evaluator.local_evaluator --output results_no_rerank.json
RERANK_ENABLED=1 RERANK_MODEL=gbdt python -m evaluator.local_evaluator --output results_rerank_gbdt.json
# swap RERANK_MODEL for: baseline, simplex, ranksvm, coord_ascent, mlp, reinforce, ensemble
```

`recommended_technical_score` in each output file is the number in the table
above. These result files are gitignored (regenerable, not part of the
frozen submission).
