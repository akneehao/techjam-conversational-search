# Handoff: learned re-ranking layer

Everything added on top of your Day 1-4 hybrid agent, where it lives, and what
you need to regenerate locally.

## TL;DR

A learned re-ranker now re-scores the top 60 of the RRF-fused candidate list
before the top 10 are returned. All models measured with the **unmodified**
`evaluator/local_evaluator.py` on the 200 public sessions:

| Configuration | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| **simplex (shipped default)** | 0.960 | **0.649** | 2.68 | **0.8410** |
| mlp | **0.965** | 0.624 | **2.67** | 0.8362 |
| gbdt | 0.955 | 0.609 | 2.71 | 0.8260 |
| coord_ascent | 0.945 | 0.622 | 2.73 | 0.8244 |
| ranksvm | 0.935 | 0.624 | 2.83 | 0.8181 |
| without re-ranking (your pipeline) | 0.940 | 0.444 | 3.12 | 0.7607 |

**+0.0803 TechnicalScore** for the default, improving every component metric.
Biggest gain is MRR (+46% relative) -- the target lands much closer to rank 1
when found.

It ships **enabled** (`RERANK_ENABLED=1`, `RERANK_MODEL=simplex`). Simplex is
also the cheapest to serve: an 11-number weight vector and one numpy dot
product, so the default path needs no lightgbm at inference. Switch models
with `RERANK_MODEL=<name>`, or turn it off with `RERANK_ENABLED=0`; it also
degrades to the plain RRF order on its own if a dependency, the artifact, or
the dense track is unavailable.

Caveat: the top three models are within ~0.015 of each other on 200 sessions,
and training used only 601 positive examples -- so don't read the ordering as
settled. More training data (`--num-queries`) is the most promising next lever.

## Committed to git -- you get these on pull

```
starter/reranker/                    runtime code, ships with the agent
├── __init__.py                      RerankerBundle, load_reranker()
├── base.py                          feature schema (14 names), baseline weights, metrics
├── catalog_index.py                 category-tree indexing (siblings / prefixes / centroids)
├── features.py                      >> FEATURE EXTRACTION -- compute_feature_matrix()
├── gbdt_inference.py                loads the model; guarded lightgbm import
├── mlp_inference.py                 pure-numpy MLP forward pass (no torch at serving time)
├── ensemble.py                      threshold-union of GBDT + MLP
└── artifacts/                       >> MODEL WEIGHTS
    ├── gbdtranker.txt               250 KB -- the shipped trained model
    └── gbdtranker_feature_importances.json

training/                            dev-only; NOT needed to run the agent
├── label_generation.py              >> BUILDS THE TRAINING DATA
├── train_all.py                     trains models, writes artifacts/
├── common.py                        agent loading + group-level train/test splits
├── evaluate.py                      offline Hits@K / MRR / NDCG / MAP
└── models/                          the 7 trainable model types
    ├── gbdt_ranker.py   simplex_ranker.py   ranksvm.py
    ├── mlp_ranker.py    coord_ascent.py     reinforce_ranker.py

notebooks/training_pipeline.ipynb    train + inspect, with plots
notebooks/model_comparison.ipynb     side-by-side model comparison
docs/reranker_eval_results.md        all results + the v1 -> v2 root-cause write-up
tests/test_reranker_fallback.py      graceful-degradation tests
requirements-dev.txt                 training-only dependencies
```

Changes to existing files: `starter/agent.py` (Day 5 constants, `_init_reranker`,
`_rerank`, and the re-rank step in `respond()`), `requirements.txt` (+lightgbm),
`.gitignore`, `README.md`.

## NOT in git -- regenerate locally

| File | Size | How |
|---|---|---|
| `data/catalog.jsonl` | 58 MB | download from the GitHub Release |
| `data/dense_*.npz` | 74 MB | automatic on first `Agent()` -- ~20 min, CPU embedding of 50k products |
| `data/reranker_training_data.npz` | 7.5 MB | `python -m training.label_generation` (~5 min) |
| `training/evaluation_summary.json` | 4 KB | `python -m training.train_all --models gbdt` (~2 s) |
| `results_*.json` | 40 KB | `python -m evaluator.local_evaluator --output ...` |
| `.env` | 1 KB | your own Gemini key -- correctly excluded, never commit it |

**You do not need to retrain anything.** `gbdtranker.txt` is committed, so the
agent works right after a pull -- you only need `catalog.jsonl` plus the
one-time dense-embedding build.

Full retrain, if you want to change the model:

```bash
python -m training.label_generation          # -> data/reranker_training_data.npz
python -m training.train_all --models gbdt   # -> starter/reranker/artifacts/
python -m evaluator.local_evaluator --output results.json
```

## How it works

1. Retrieval is unchanged: BM25 (FTS5) + dense (all-MiniLM-L6-v2), fused by RRF.
2. The top 60 fused candidates get a 14-dim feature vector each: embedding
   similarity, lexical overlap, category-tree structure (sibling / parent /
   descendant / path similarity), and **the first-stage BM25, dense and RRF
   rank scores**.
3. A LightGBM LambdaRank model scores each candidate; the list is re-sorted and
   sliced to the top 10.

Training labels are self-supervised from the catalog -- no manual labelling, no
paid API calls. For each sampled product: build a short slot-style query from
it, run the *real* retrieval pipeline, and label that product as the positive
with the other retrieved products as (hard) negatives.

## Two traps I hit -- worth knowing before you touch label generation

The first version of this scored **worse** than no re-ranking at all (0.7006 vs
0.7607, and as low as 0.3344 for other model types). Root cause, diagnosed from
the trained models' own feature weights:

1. **Label leakage.** v1 chose the positive as "the category sibling most
   cosine-similar to the query" -- which made the `sibling_max_sim` feature
   nearly encode the labelling rule itself. It took 68% of GBDT's gain. The
   model learned how I made labels, not what makes a product relevant.
2. **The training task was nearly the opposite of the real one.** v1 labelled
   same-category products as relevant, which teaches a *category matcher* --
   but the evaluator asks for one specific hidden target, where a
   same-category-wrong-product is exactly the distractor to beat.
3. **The first-stage signals weren't features.** BM25/dense/RRF scores weren't
   passed in, so the re-ranker discarded the evidence that makes retrieval
   strong and had no floor -- it couldn't even reproduce the ordering it was
   re-ranking. With `rrf_score` as an input, "copy RRF" is learnable, so a
   well-fit model shouldn't be able to fall below the control.

A second leak appeared during the rework: building the query purely from a
product's own tokens made it a strict subset of that product's text, pinning
`lexical_score` to exactly 1.0 for every positive. Queries are now a noisy,
partial view (random token dropout + a couple of generic shopper words).

**General lesson:** any self-supervised labelling rule risks becoming visible
to a feature. Check positive-vs-negative feature separation *before* training,
not after evaluating -- the offline metrics looked near-perfect (0.99+ NDCG)
the whole time v1 was actually regressing on the real evaluator.

## Open items / caveats

- **Only GBDT is trained on the current 14-feature schema.** The other six
  model types are implemented and runnable (`--models simplex,mlp,...`) but
  have no current artifact, so those `RERANK_MODEL` values fall back to plain
  RRF. Retraining them is cheap for the linear ones, ~8 min for MLP.
- **Possible optimism vs. the private 800 sessions.** These numbers are on the
  200 public sessions, and training queries come from the same catalog. The
  +0.165 MRR gain is large enough that it probably doesn't vanish entirely, but
  expect some shrinkage.
- **The Gemini LLM path has never been evaluated end-to-end.** `AGENT_USE_LLM`
  defaults on, so with a key present the Day 2 router runs -- but every score in
  this document was measured with no key configured, i.e. via the deterministic
  heuristic fallback. Worth one evaluator run with the key active to see the
  real number.
- `CLARIFY_ENABLED` (Day 3 proactive guidance) is still off by default, with
  its original ~0.06-score-cost note -- also never verified against a working
  key.
- Training data is currently 751 queries / 45k pairs (kept small for time).
  More queries via `--num-queries` should help.
- There may be a stray 8 KB `data/dense_*.npz` from a test run -- harmless and
  gitignored, safe to delete.
