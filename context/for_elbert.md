# Handoff: learned re-ranking layer

Everything added on top of your Day 1-4 hybrid agent: what it does, where it
lives, what you need to regenerate, and the traps I hit so you don't repeat
them.

## TL;DR

A learned re-ranker now re-scores the top 60 of the RRF-fused candidate list
before the top 10 are returned. All numbers below are full runs of the
**unmodified** `evaluator/local_evaluator.py` on the 200 public sessions.

| Configuration | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| 3k simplex | **0.970** | 0.640 | 2.65 | **0.8440** |
| **3k gbdt (shipped default)** | 0.960 | **0.649** | 2.65 | 0.8418 |
| 751q simplex | 0.960 | **0.649** | 2.68 | 0.8410 |
| 3k mlp | 0.965 | 0.627 | **2.63** | 0.8380 |
| 751q mlp | 0.965 | 0.624 | 2.67 | 0.8362 |
| 751q gbdt | 0.955 | 0.609 | 2.71 | 0.8260 |
| without re-ranking (your pipeline) | 0.940 | 0.444 | 3.12 | 0.7607 |

**+0.0811 TechnicalScore** for the shipped default, improving every component
metric. The biggest gain is MRR (+46% relative) -- the target lands much
closer to rank 1 when found.

Ships **enabled**: `RERANK_ENABLED=1`, `RERANK_MODEL=gbdt`,
`RERANK_ARTIFACTS_DIR=starter/reranker/artifacts_3k`. Turn it off with
`RERANK_ENABLED=0`; it also falls back to the plain RRF order on its own if
lightgbm, the artifact, or the dense track is unavailable.

### Why GBDT and not simplex, when simplex scores 0.0022 higher

Deliberate. 0.0022 is under one session on a 200-session set, and:

- **Simplex is degenerate.** It converges to `lexical_score = 1.0` with all
  13 other weights *exactly* 0.0 -- at both training sizes. It is purely a
  lexical-overlap ranker that ignores the dense, BM25 and RRF signals, so it
  has no graceful behaviour if the hidden 800 sessions paraphrase rather than
  quote product wording. That is a real risk on a private set we can't see.
- **The bigger test set disagrees with the evaluator.** On the 594-group
  offline test split (roughly 3x the statistical power of 200 sessions) the
  ordering is gbdt ~ mlp > coord_ascent > ranksvm > simplex *last*. When a
  larger, cleaner test set contradicts a smaller one by 0.002, trust the
  larger.

If you disagree, switching is one env var: `RERANK_MODEL=simplex`.

## Committed to git -- you get these on pull

```
starter/reranker/                    runtime code, ships with the agent
├── __init__.py                      RerankerBundle, load_reranker()
├── base.py                          feature schema (14 names), baseline weights, metrics
├── catalog_index.py                 category-tree indexing (siblings / prefixes / centroids)
├── features.py                      >> FEATURE EXTRACTION -- compute_feature_matrix()
├── gbdt_inference.py                loads the model; guarded lightgbm import
├── mlp_inference.py                 pure-numpy MLP forward pass (no torch at serving time)
├── ensemble.py                      threshold-union of GBDT + MLP (not used by default)
├── artifacts_3k/                    >> SHIPPED WEIGHTS (trained on 2,972 queries)
│   ├── gbdtranker.txt               the default model
│   ├── gbdtranker_feature_importances.json
│   ├── simplex_weights.json, ranksvm_weights.json,
│   ├── coord_ascent_weights.json, mlp_ranker.npz/.json
│   └── ensemble_thresholds.json
└── artifacts/                       older weights (trained on 751 queries), kept for comparison

training/                            dev-only; NOT needed to run the agent
├── label_generation.py              >> BUILDS THE TRAINING DATA
├── train_all.py                     trains models, writes artifacts/ (--models to subset)
├── common.py                        agent loading + group-level train/test splits
├── evaluate.py                      offline Hits@K / MRR / NDCG / MAP
└── models/                          the 7 trainable model types
    ├── gbdt_ranker.py   simplex_ranker.py   ranksvm.py
    ├── mlp_ranker.py    coord_ascent.py     reinforce_ranker.py

notebooks/training_pipeline.ipynb    train + inspect, with plots
notebooks/model_comparison.ipynb     side-by-side model comparison
docs/reranker_eval_results.md        full results + the v1 -> v2 root-cause write-up
tests/test_reranker_fallback.py      graceful-degradation tests
requirements-dev.txt                 training-only dependencies
```

Changes to existing files: `starter/agent.py` (Day 5 constants,
`_init_reranker`, `_rerank`, and the re-rank step in `respond()`),
`requirements.txt` (+lightgbm), `.gitignore`, `README.md`.

## NOT in git -- regenerate locally

| File | Size | How |
|---|---|---|
| `data/catalog.jsonl` | 58 MB | download from the GitHub Release |
| `data/dense_*.npz` | 74 MB | automatic on first `Agent()` -- ~20 min, CPU embedding of 50k products |
| `data/reranker_training_data_3k.npz` | 30 MB | `python -m training.label_generation --num-queries 3000 --out data/reranker_training_data_3k.npz` (~16 min) |
| `data/reranker_training_data.npz` | 7.5 MB | same, `--num-queries 751` (the older set) |
| `training/evaluation_summary*.json` | 4 KB | written by `train_all.py` |
| `results_*.json` | 40 KB | `python -m evaluator.local_evaluator --output ...` |
| `.env` | 1 KB | your own Gemini key -- correctly excluded, never commit it |

**You do not need to retrain anything.** The weights are committed, so the
agent works right after a pull -- you only need `catalog.jsonl` plus the
one-time dense-embedding build.

Full retrain, if you want to change the model:

```bash
python -m training.label_generation --num-queries 3000 \
    --out data/reranker_training_data_3k.npz          # ~16 min
python -m training.train_all \
    --data data/reranker_training_data_3k.npz \
    --artifacts-dir starter/reranker/artifacts_3k \
    --summary-out training/evaluation_summary_3k.json \
    --models baseline,simplex,ranksvm,coord_ascent,mlp,gbdt   # ~3 min
python -m evaluator.local_evaluator --output results.json      # ~6 min
```

Timing notes: label generation runs at ~0.38 s/query and dominates
everything. Simplex/RankSVM/GBDT train in seconds even at 10k queries; MLP
and REINFORCE are what make a full sweep slow (~2.5 min each at 3k).
Coordinate Ascent self-caps at 50k rows so it stays ~28 s regardless.

## Scaling to a larger training set

Everything is parameterised, so going bigger is just three commands with new
paths. **Always write to new paths** rather than overwriting -- that is how
`artifacts/` (751q) and `artifacts_3k/` (2,972q) both still exist and can be
compared.

```bash
N=10000     # pick your size

python -m training.label_generation \
    --num-queries $N \
    --out data/reranker_training_data_${N}.npz

python -m training.train_all \
    --data data/reranker_training_data_${N}.npz \
    --artifacts-dir starter/reranker/artifacts_${N} \
    --summary-out training/evaluation_summary_${N}.json \
    --models baseline,simplex,ranksvm,coord_ascent,mlp,gbdt

RERANK_ENABLED=1 RERANK_MODEL=gbdt \
RERANK_ARTIFACTS_DIR=starter/reranker/artifacts_${N} \
    python -m evaluator.local_evaluator --output results_${N}_gbdt.json
```

Then point the agent at it by editing the two defaults near the top of
`starter/agent.py` (`RERANK_MODEL`, `RERANK_ARTIFACTS_DIR`), or just set those
env vars.

**Time budget** (measured at ~0.38 s/query for generation):

| Queries | Label gen | Train gbdt+simplex | Train all 6 | Evaluate 1 model |
|---|---|---|---|---|
| 3,000 *(current)* | ~16 min | ~4 s | ~3 min | ~6 min |
| 5,000 | ~32 min | ~7 s | ~5 min | ~6 min |
| 10,000 | ~63 min | ~15 s | ~10 min | ~6 min |
| 25,000 | ~2.6 h | ~40 s | ~25 min | ~6 min |
| 50,000 (whole catalog) | ~5.3 h | ~90 s | ~50 min | ~6 min |

Evaluator runs parallelise fine -- I ran four at once without trouble.

**If generation is the bottleneck**, it is embarrassingly parallel (each query
is independent, and the dense vectors are already cached). Sharding it across
4 processes would cut a 63-min run to ~16 min. `label_generation.py` has no
shard flag yet; adding one is small -- slice `query_list` by
`[shard::num_shards]`, write `..._shard{i}.npz`, then concatenate the `X`/`y`
arrays and re-number `groups` with `training.common.dense_group_ids`.

**Two things to know before spending hours on this:**

- **Diminishing returns are already visible.** 751 -> 2,972 queries (4x) moved
  the evaluator score by only +0.0158 for gbdt, +0.0030 for simplex and
  +0.0018 for mlp, even though the offline metrics improved a lot. Another 4x
  may buy very little.
- **The sampling bug is fixed, but know why it existed.** The catalog has two
  top-level buckets of wildly different size (49,990 vs 10), and the original
  even-per-bucket quota silently capped `--num-queries 1500` at 760 actual
  queries. It now allocates proportionally and tops up the shortfall, so the
  count you ask for is the count you get (minus targets that retrieval never
  surfaces, ~1%). If you change the sampling, re-check the printed
  "Query set: N products" line actually matches what you asked for.

## How it works

1. Retrieval is unchanged: BM25 (FTS5) + dense (all-MiniLM-L6-v2), fused by RRF.
2. The top 60 fused candidates each get a 14-dim feature vector: embedding
   similarity, lexical overlap, category-tree structure (sibling / parent /
   descendant / path similarity), and **the first-stage BM25, dense and RRF
   rank scores**.
3. A LightGBM LambdaRank model scores each candidate; the list is re-sorted
   and sliced to the top 10.

Training labels are self-supervised from the catalog -- no manual labelling,
no paid API calls. For each sampled product: build a short slot-style query
from it, run the *real* retrieval pipeline, and label that product as the
positive with the other retrieved products as (hard) negatives.

## Three traps I hit -- read before touching label generation

The **first version scored worse than no re-ranking at all** (0.7006, and as
low as 0.3344 for other model types). Root cause, diagnosed from the trained
models' own feature weights:

1. **Label leakage.** v1 chose the positive as "the category sibling most
   cosine-similar to the query", which made the `sibling_max_sim` feature
   nearly encode the labelling rule itself -- 68% of GBDT's gain, 53% of
   Simplex's weight. The models learned how I made labels, not what makes a
   product relevant.
2. **The training task was nearly the opposite of the real one.** v1 labelled
   same-category products as relevant, which teaches a *category matcher* --
   but the evaluator asks for one specific hidden target, where a
   same-category-wrong-product is exactly the distractor to beat.
3. **The first-stage signals weren't features.** BM25/dense/RRF scores weren't
   passed in, so the re-ranker discarded the evidence that makes retrieval
   strong and had no floor -- it couldn't even reproduce the ordering it was
   re-ranking. With `rrf_score` as an input, "copy RRF" is learnable, so a
   well-fit model shouldn't be able to fall below the control.

A **fourth trap** appeared during the rework: building the query purely from a
product's own tokens made it a strict subset of that product's text, pinning
`lexical_score` to exactly 1.0 for every positive (vs 0.41 for negatives).
Queries are now a noisy, partial view (random token dropout + a couple of
generic shopper words), which moves positives to a realistic 0.67-1.0 spread.

**General lesson:** any self-supervised labelling rule risks becoming visible
to a feature. Check positive-vs-negative feature separation *before* training,
not after evaluating -- the offline metrics looked near-perfect (0.99+ NDCG)
the entire time v1 was actually regressing on the real evaluator.

## Training set sizes

| | 751-query set | 3k set (shipped) |
|---|---|---|
| Query groups | 751 | 2,972 |
| Candidates per query | 60 | 60 |
| Rows (query-candidate pairs) | 45,060 | 178,320 |
| Positive examples | 751 (one per query) | 2,972 |
| Train / test split | 601 / 150 groups | 2,378 / 594 groups |
| Share of the 50k catalog | 1.5% | 5.9% |

The row count overstates the real sample size: each positive is padded with 59
negatives, so the number that governs learning is the positive count.

Going 751 -> 2,972 queries lifted the offline metrics substantially but moved
the *evaluator* score much less: gbdt +0.0158, simplex +0.0030, mlp +0.0018.
Diminishing returns have clearly set in, so I would not expect much from
another 4x on its own.

## Open items / caveats

- **The top three models are within 0.006 of each other** on the evaluator
  (0.8380-0.8440), which is about one session. Don't read the ordering as
  settled, and be wary of tuning the default on public-set margins this thin.
- **Possible optimism vs the private 800 sessions.** Training queries come
  from the same catalog, and all tuning decisions were made against the 200
  public sessions. Expect some shrinkage.
- **REINFORCE is excluded from the shipped comparison.** On the 14-feature
  schema it initially diverged outright (weights driven strongly negative on
  the most informative features -> random/inverted ranking, MRR 0.017 ~ 1/60),
  caused by unnormalised gradient ascent over features spanning very different
  scales. I added feature standardisation and gradient clipping, which fixed
  the divergence (offline MRR 0.017 -> 0.320), but it remains far below the
  other six and isn't worth an evaluator run.
- **The Gemini LLM path has never been evaluated end-to-end.** `AGENT_USE_LLM`
  defaults on, so with a key present the Day 2 router runs -- but every score
  in this document was measured with no key configured, i.e. via the
  deterministic heuristic fallback. Worth one evaluator run with the key
  active to see the real number, and to check nothing breaks.
- `CLARIFY_ENABLED` (Day 3 proactive guidance) is still off by default with
  its original ~0.06-score-cost note -- also never verified against a working
  key.
- **An ensemble is probably not worth it.** Four of the seven models
  (simplex, ranksvm, coord_ascent, reinforce) are the same linear hypothesis
  class over identical features, so combining them adds little; only
  GBDT/MLP are genuinely decorrelated. If you do try, RRF over model *ranks*
  is the right form (their raw scores are on incompatible scales), not the
  existing threshold-union ensemble, which scored worst in v1.
- Commits on this branch are authored as `Ripleyyyyy <10203191h@gmail.com>`
  (a stale global `git config user.name/email`), not saffronlim5 -- worth
  noting in the write-up's contributions section since GitHub attributes by
  email.
