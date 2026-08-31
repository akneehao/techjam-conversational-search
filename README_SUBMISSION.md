# TechJam Conversational E-Commerce Search — Submission

Team repository: `akneehao/techjam-conversational-search` (branch `amogh`)

---

## 1. Project Overview

We build a conversational shopping agent. The agent talks with a customer for
up to 10 turns and must return the customer's hidden target product inside a
top-10 list.

The system has seven layers. Each layer is optional and fails safely: if a
layer cannot run, the agent still works using the layers below it.

| Layer | What it does | Needs |
|---|---|---|
| 1. Sparse retrieval | SQLite FTS5 + BM25 over the 50k catalog. Tiered queries (strict AND then broad OR) fused by weighted RRF. | Python stdlib only |
| 2. LLM state tracker | Google Gemini extracts intent (buying / browsing) and 8 constraint slots, and flags intent override. | `GEMINI_API_KEY` (optional) |
| 3. Proactive guidance | Asks one clarifying question when the candidate pool is too large, while still returning results. | Off by default (see section 7) |
| 4. Dense re-ranking | `all-MiniLM-L6-v2` embeddings reorder the BM25 candidate set. Dense never selects candidates. | `numpy`, `sentence-transformers` |
| 5. Learned re-ranker | A trained model re-scores the top 60 candidates using 14 features. | `numpy` (LightGBM only for `gbdt`) |
| 6. Dual-track routing | Intent chooses the retrieval weights and truncation depth per turn, re-selected from live state. | Python stdlib only |
| 7. Web UI & API | A Flask server hosts a single-page web app for interactive demos. | `flask` |

Layers 5 and 6 are the main contributions of this submission.

### Results (200 public sessions, unmodified official evaluator)

| Configuration | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Weak BM25 starter (organizer baseline) | 0.125 | 0.068 | 9.81 | 0.1067 |
| Retrieval only, no learned re-ranker | 0.960 | 0.643 | 2.71 | 0.8387 |
| **Full system (submitted)** | **0.970** | **0.670** | **2.65** | **0.8530** |

Per scenario type (full system):

| Scenario | Sessions | HitRate@10 | MRR | MTTC |
|---|---|---|---|---|
| Buying | 80 | 0.950 | 0.634 | 2.38 |
| Browsing | 80 | 0.988 | 0.656 | 2.27 |
| Intent override | 30 | 0.967 | 0.743 | 4.13 |
| Boundary | 10 | 1.000 | 0.845 | 3.30 |

---

## 2. Setup and Installation

### Requirements

- **Python 3.12** (developed and tested on 3.12.6). Python 3.10+ should work.
- About **2 GB** free disk space (catalog + embedding cache).
- No GPU needed. Everything runs on CPU.

### Step 1 — Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

| Package | Why | If missing |
|---|---|---|
| `numpy` | dense vectors, feature math, model inference | agent falls back to BM25 only |
| `sentence-transformers` | query and document embeddings | agent falls back to BM25 only |
| `lightgbm` | only for `RERANK_MODEL=gbdt` | the shipped `ranksvm` does not need it |
| `flask` | web server for the UI demo | `app.py` will not run |

`requirements-dev.txt` is only needed to RE-TRAIN the models
(`scipy`, `scikit-learn`, `torch`, `jupyter`, `pandas`, `matplotlib`).
You do not need it to reproduce our score.

### Step 2 — Download the catalog

```bash
gzip -dk data/catalog.jsonl.gz          # -> data/catalog.jsonl
sha256sum -c data/SHA256SUMS            # verify
```

Expected: 50,000 rows.

### Step 3 — Environment variables (all optional)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | unset | Enables the Gemini intent router (layer 2). Without it, a deterministic parser is used. |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Which Gemini model to call. |
| `AGENT_USE_LLM` | `1` | Set to `0` to force the deterministic parser. |
| `LLM_MIN_INTERVAL` | `0.0` | Seconds between router calls. Set to `4.5` on a free-tier key, which is limited to ~15 requests/minute. |
| `FUSION_MODE` | `rerank` | `rerank` (BM25 gates, dense reorders), `rrf` (legacy symmetric fusion), `bm25`. |
| `RERANK_ENABLED` | `1` | Set to `0` to disable the learned re-ranker. |
| `RERANK_MODEL` | `ranksvm` | One of `ranksvm`, `simplex`, `coord_ascent`, `mlp`, `gbdt`. |
| `RERANK_ARTIFACTS_DIR` | `starter/reranker/artifacts_5k` | Which trained weights to load. |
| `DUAL_TRACK_ENABLED` | `1` | Intent-conditional weights and truncation. |
| `CLARIFY_ENABLED` | `0` | Proactive clarifying questions (see section 7). |
| `PROFILE_INJECT` | `0` | `0` profile never reaches retrieval; `1` inject discriminating tags; `2` inject all tags. |

**No API key is required to reproduce our reported score.** Put keys in a
`.env` file, which is git-ignored. We never commit secrets.

### Step 4 — First run builds an embedding cache

The first time you create an `Agent`, it encodes all 50,000 products with
`all-MiniLM-L6-v2`. This takes **about 12 minutes on CPU** and needs network
access once, to download the model (about 90 MB) from Hugging Face.

The result is cached to `data/dense_*.npz` (74 MB). Every later run loads the
cache in a few seconds and needs no network; set `HF_HUB_OFFLINE=1` to stop
`sentence-transformers` contacting Hugging Face for a model already on disk.

> **On a network that intercepts TLS**, the one-time model download can fail
> with `CERTIFICATE_VERIFY_FAILED`. Install `truststore` and call
> `truststore.inject_into_ssl()` before constructing the `Agent`; this uses the
> OS certificate store and keeps verification enabled. Do not disable it.

---

## 3. Steps to Reproduce Our Results

### One command

```bash
python -m evaluator.local_evaluator --output results.json
```

This uses the **unmodified** official evaluator. We never edited any file in
`evaluator/`. Expected result in `results.json`:

```json
"recommended_technical_score": 0.853,
"hit_rate_at_10": 0.97, "mrr": 0.669733, "mttc": 2.645
```

Runtime: about **6 minutes** after the embedding cache exists, plus about
**100 seconds** of startup (see section 6).

With `AGENT_USE_LLM=0` the run is fully deterministic. With the Gemini router
enabled, results can vary slightly because quota or network failures change how
many turns actually reach the LLM.

### Ablations

```bash
# no learned re-ranker
RERANK_ENABLED=0 python -m evaluator.local_evaluator --output r_control.json   # 0.8387

# no dual-track routing
DUAL_TRACK_ENABLED=0 python -m evaluator.local_evaluator --output r_uniform.json  # 0.8485

# the original symmetric-RRF first stage
FUSION_MODE=rrf RRF_K=60 RERANK_ENABLED=0 python -m evaluator.local_evaluator     # 0.7617
```

### Run the interactive web UI

```bash
python app.py      # then open http://localhost:5001
```

### Re-train the re-ranker from scratch (optional)

Trained weights are already committed, so this is **not** required.

```bash
pip install -r requirements-dev.txt

# 1. Build self-supervised training data   (about 16 min)
python -m training.label_generation --num-queries 5000

# 2. Train the models                      (about 12 min)
python -m training.train_all --models all \
    --artifacts-dir starter/reranker/artifacts_5k \
    --summary-out training/evaluation_summary_5k.json

# 3. Score it                              (about 6 min)
python -m evaluator.local_evaluator --output results.json
```

`--models all` trains six models and excludes REINFORCE, which ranks worse than
the first stage it is given (see section 7). Use `--models everything` to
include it.

### Tests

```bash
python -m pytest tests/ -q
```

8 tests. They check the official metric maths and, importantly, that the agent
still returns valid recommendations when the re-ranker, its weights, or
`lightgbm` are missing.

---

## 4. Method

### 4.1 Retrieval: BM25 gates, dense re-ranks (layers 1 and 4)

**BM25 alone decides which documents are candidates. Every other component
reorders that set; none can add or remove a member.** Recall is therefore a
pure function of the lexical gate, and no learned component can cost a hit.

1. **BM25 / FTS5.** Tiered FTS5 queries built from the accumulated slots. The
   strictest tier is `leaf_category AND at least two specific constraints`;
   broader tiers follow. Tiers are fused by weighted RRF. Field weights favour
   category (11.0), title (8.0) and a normalised keyword bag (8.0). The top
   120 form the candidate set.
2. **Dense.** The conversation state is embedded and cosine-scored **against
   those candidates only**, min-max normalised per turn, and blended under a
   lexical rank prior that spans 1.0 → 0.085. Semantics reorder neighbours but
   cannot overturn a decisive lexical match.

This replaced a symmetric RRF fusion of two 60-deep lists at `k=60`, which was
degenerate: rank 1 and rank 60 differed by only 1.97x, so a document ranked
last in *both* lists outscored the top hit of either, and the fused top-10 was
an exact rank-merge of `BM25[1..5]` with `dense[1..5]`. Fixing it is worth
**+0.077** (0.7617 → 0.8387). Retuning `k` alone recovers only +0.018; the rest
requires the architectural change.

### 4.2 Dual-track routing (layer 6)

`_select_strategy()` picks the retrieval weights and truncation depth per turn
from live session state:

| Track | Trigger | Dense weight | Depth |
|---|---|---|---|
| precision | buying | 0.10 | 120 |
| discovery | browsing | 0.45 | 200 |
| recovery | constraint set unchanged ≥2 turns | +0.20 | 200 |

Intent has a **deterministic floor** (`_infer_intent`) derived from the message
templates and accumulated constraints, so the tracks work with no API key; the
LLM router overrides it when available. Worth **+0.0045**, entirely on the
buying track (HitRate 0.938 → 0.950). In practice browsing sessions flip to
buying at turn 2 once a real constraint arrives, so the mechanism is better
described as *"tighten as soon as real constraints arrive"* than as a standing
buying/browsing split.

### 4.3 Learned re-ranker (layer 5)

For each of the top 60 candidates we compute **14 features**:

| Group | Features |
|---|---|
| Semantic | `node_profile_sim`, `max_desc_sim`, `avg_top5_desc_sim`, `sibling_coherence`, `sibling_max_sim`, `parent_sim`, `path_score` |
| Lexical | `lexical_score` (query-token coverage of the product text) |
| Query context | `query_token_count`, `query_max_node_sim`, `query_embedding_norm` |
| First stage | `bm25_rank_score`, `dense_rank_score`, `rrf_score` |

Category features use the catalog's own `categories` path, so sibling, parent
and descendant sets come from prefix dictionaries built once at startup — no
graph database. Including the first-stage scores gives the model a **floor**:
"reproduce the first-stage order" is learnable, so a well-fitted model should
not do worse than not re-ranking.

Worth **+0.0098** on a sound first stage. Measured against the *broken* first
stage the same layer appeared to be worth +0.080 — most of which was the model
routing around the fusion defect rather than adding signal.

### 4.4 Personalized context distillation (layer 2)

`reset()` distils the aggregate `user_profile` once per session. It reaches the
agent by exactly two routes: a compact `PROFILE:` line in the router prompt,
and a small additive re-rank boost (`PROFILE_WEIGHT=0.05`). **Neither route can
add or remove a candidate.**

That separation is deliberate. Across the 200 public sessions `preference_tags`
is nearly constant — fit 82%, material 77%, comfort 72%, style 50%,
durability 24% — and all five are already in the agent's `GENERIC` stop set.
They describe essentially every apparel product, and they come from the
shopper's *prior* purchases, which the evaluator never links to the target.
Gating on them would inject noise into 100% of sessions. The filter keeps only
rare, discriminating tags (`warmth`, `weather`, `performance`).
`PROFILE_INJECT=1|2` exposes the injection variants so the claim is measurable
rather than assumed.

### 4.5 User interface (layer 7)

A Flask server (`app.py`) exposes a single `/api/chat` endpoint that calls
`agent.respond()` and returns enriched recommendations as JSON. A standalone
`templates/index.html` renders the chat experience in pure HTML/CSS/JS with
light/dark support and product cards showing title, brand, price and category
tags.

### 4.6 Training data (self-supervised, no labelling cost)

We have no human relevance labels, so we generate them from the catalog:

1. Sample 5,000 products, stratified by top-level category with proportional
   allocation and top-up.
2. Build a **short, slot-style query** from each product's category and a few
   of its tokens, dropping ~30% at random and mixing in generic shopper words,
   so the query is a noisy, partial description.
3. Run the **real** first stage for that query.
4. Label the source product as the positive; every other retrieved product is a
   hard negative.

Result: **4,965 queries, 60 candidates each, 297,900 rows**, split by group
into 3,972 train / 993 test. Zero API calls, no manual annotation.

`retrieve_candidates()` mirrors whichever `FUSION_MODE` is configured, and the
saved metadata records it. This matters: models trained against the old RRF
ordering and served on the current one lose ground, which is most of why
`gbdt` trained on the older set scores 0.8135 here.

---

## 5. Model Choice — and why offline metrics did not decide it

We implemented seven ranking models on a shared interface and measured each
with the official evaluator, all on the current first stage:

| Model | Offline MRR (993 groups) | Evaluator | Note |
|---|---|---|---|
| **ranksvm (submitted)** | 0.9476 (2nd) | **0.84849** | pure numpy, no lightgbm |
| simplex | 0.9398 (5th) | 0.84848 | degenerate single-feature solution |
| coord_ascent | 0.9464 (3rd) | 0.84675 | |
| mlp | 0.9444 (4th) | 0.84311 | 14-32-16-1, exported to numpy |
| gbdt | **0.9477 (1st)** | 0.82499 | LightGBM LambdaRank |
| *first-stage baseline* | *0.7298* | *0.8387* | control |
| reinforce | 0.6810 | — | **below the baseline**; excluded |

**The offline test set is blind at the tails.** It is a held-out split of the
same self-supervised distribution used for training, while the evaluator uses
templated shopper messages. The middle of the table transfers perfectly —
ranksvm / coord_ascent / mlp hold offline ranks 2,3,4 and evaluator ranks
1,3,4. Both extremes invert:

- **gbdt** is 1st offline and **last** on the evaluator, 0.024 behind. It is
  the only tree model, so it fits sharp splits on the synthetic query style
  that do not survive the distribution shift.
- **simplex** is last offline and 2nd on the evaluator. It converges to a
  one-hot solution (`lexical_score = 1.0`, all 13 other weights exactly `0.0`)
  at every training size, which is too degenerate to fit the synthetic style
  and therefore invariant to the shift.

Capacity buys offline score and costs transfer. Since a default is chosen from
the tail, **selecting on offline metrics alone would have shipped the worst of
the five models.**

The top four are inside 0.005 — about one session on a 200-session set — so
score does not choose between them. We submit `ranksvm` on grounds that are not
noise: best coverage (HitRate 0.965 vs 0.960, and HitRate carries 0.50 of the
composite); strong on *both* metrics rather than one; trained on candidates
drawn from the real serving pipeline using all 14 features; and a pure-numpy
artifact — a weight vector and a dot product — with no LightGBM at inference.

That last point is not stylistic. Loading the LightGBM booster from a
CRLF-converted checkout aborts the interpreter **natively**, which no
`try/except` can catch (see section 6).

---

## 6. Negative Results

We report these because the diagnoses shaped the final design.

### 6.1 The first re-ranker scored worse than no re-ranking

0.7006, and as low as 0.3344 for other models. We found the cause by reading
the trained models' own feature weights: `sibling_max_sim` took **68% of
GBDT's total gain**. Three defects:

1. **Label leakage.** v1 chose the positive as "the category sibling most
   similar to the query", so that feature nearly encoded the labelling rule.
   The models learned *how we made labels*, not what makes a product relevant.
2. **Wrong task.** v1 labelled same-category products as relevant, training a
   *category matcher* — but the evaluator asks for one specific product, where
   a same-category-wrong-product is exactly the distractor to beat.
3. **Missing first-stage signals**, so the re-ranker had no floor.

A fourth leak appeared during the rewrite: building the query only from the
product's own words pinned `lexical_score` to exactly `1.0` for every positive.
Fixed with token dropout and filler words.

**Lesson:** any self-supervised labelling rule can become visible to a feature.
Check positive-versus-negative separation *before* training. Our offline
metrics read above 0.99 NDCG throughout the period the real score was
regressing.

### 6.2 Proactive clarification costs 0.064 and cannot be made free

`CLARIFY_ENABLED=1` scores 0.7843 against 0.8530. The FAQ (§5) permits asking a
question *and* returning recommendations in the same turn, and we do — but that
fixed only the smaller half. The dominant cost is the narrow `ask_attribute`:

```python
matches = [v for v in constraints if v not in disclosed
           and (attribute == "other" or classify_constraint(v) == attribute)]
```

`"other"` matches any undisclosed constraint, so the shopper always volunteers
something. A targeted slot matches only that type and often returns *"I don't
have an additional preference for color"* — a turn with zero information.
Browsing HitRate falls 0.988 → 0.812 and MTTC rises 2.27 → 3.84.

This is a property of an unconditionally cooperative simulator, not of bad
conversational design. We kept the feature and documented the trade-off.

### 6.3 A native crash that no fallback could catch

`gbdtranker.txt` is a `.txt`, and with `core.autocrlf=true` git rewrites its
~2,055 newlines on checkout. LightGBM rejects that with a **native abort**, not
a Python exception, so `except Exception` cannot catch it and the process dies
at `Agent()` construction. The blob stored in git was always LF — the corruption
happens entirely at checkout, so it hits any Windows clone while leaving the
committed bytes innocent.

Fixed in two layers: `.gitattributes` marks the artifacts `-text`, and
`load_gbdt` normalises newlines and loads via `model_str`, repairing an
already-mangled working copy. **A C-extension crash defeats Python-level
robustness entirely**; the only defences are validating input before handing it
to native code, and preferring pure-numpy artifacts.

---

## 7. Disclosure: Latency, Token Usage, and Cost

Measured on a Windows 11 laptop, Python 3.12.6, CPU only, no GPU.

### Latency

**Startup (once per process):**

| Phase | Time |
|---|---|
| Import (`torch`, `sentence-transformers`) | ~67 s |
| `Agent()` init: FTS5 index, vector cache, category index | ~34 s |
| **Total startup** | **~100 s** |
| First run only: encode 50k products | ~12 min (one time, then cached) |

**Per turn:** retrieval only ~161 ms; plus the re-ranker ~353 ms; plus the
Gemini call ~613 ms (network round trip). A full 200-session evaluation takes
about **6 minutes** after startup.

### Token usage

The LLM is **optional** and used only for intent and slot extraction. We never
send catalog text or product lists to it. Usage is bounded by three levers: a
terse prompt with single-letter JSON keys, skipping calls on messages that
carry no new information, and a hard per-session call cap.

| Measurement | Value |
|---|---|
| Full 200-session run, LLM enabled | **36,820 tokens** (30,976 prompt + 5,844 completion) |
| Per router call | 283 prompt + 59 completion = **342** |
| Calls in a full run | ~109 |
| Full 200-session run, `AGENT_USE_LLM=0` | **0 tokens** |

Against the original Day 2 prompt this is −28% on the system prompt, −22% on
the completion, **−26% per call** and **−55% per session** once the call cap is
applied.

### Estimated model cost

**Under USD 0.01 for a complete 200-session evaluation**, and USD 0.00 with no
key set. Training costs nothing: label generation and model training run on
local CPU and make no API calls.

### Does the LLM help?

Measured, on the full shipped configuration: **0.8487 with the router versus
0.8530 without** — it costs 0.0043 and 36,820 tokens. The loss is concentrated
in `intent_override`, the scenario it exists for, because the LLM's override
handler was more destructive than the deterministic one. We made it
conservative (retire the category gate, keep accumulated constraints), but
**that fix is unvalidated**: the validating run exhausted the free-tier daily
quota and fell back to the deterministic path.

Our reported score is measured with `AGENT_USE_LLM=0`.

### Network dependency and fallback behaviour

| Situation | Behaviour |
|---|---|
| No `GEMINI_API_KEY` | Deterministic parser replaces the router. Our reported score was measured this way. |
| Gemini returns 429 / 5xx / times out | Exponential backoff 2s → 60s; the router recovers on its own. |
| Gemini returns 400 / 401 / 403 | Circuit breaker opens permanently for the process; agent continues offline. |
| **Gemini quota exhausted mid-run** | Measured: the run completes at **0.8530**, identical to LLM off. |
| No `lightgbm` | Only affects `RERANK_MODEL=gbdt`. The shipped `ranksvm` is pure numpy. |
| No `numpy` / `sentence-transformers` | Dense track and re-ranker skipped. Agent is pure BM25. |
| Malformed FTS expression | That query tier contributes no rows; the turn still answers. |
| Any other error in `respond()` | Staged degradation: full → BM25-only → last good list → empty-but-valid. |
| No network at all (after first run) | Everything works; vectors are cached on disk. |

**Our reported score of 0.8530 does not depend on any network access or paid
service.**

---

## 8. Limitations

1. **Tuned on 200 public sessions.** Training queries come from the same
   catalog and every decision was validated on the public set. Expect the
   private 800 to score somewhat lower.
2. **Margins between the top four models are inside the noise** — 0.005, about
   one session. Our choice of `ranksvm` rests on coverage, dual-metric
   agreement, training alignment and dependency footprint, not on a
   statistically significant gap.
3. **Training queries are synthetic**, built from product text rather than real
   customer language. The FAQ (§1) confirms the private sessions use the same
   deterministic templates with no paraphrasing, which limits this risk here
   but would not hold for real shoppers.
4. **Diminishing returns from more training data.** Going from 751 to 4,965
   queries improved HitRate@10 (0.960 → 0.965) but barely moved the composite
   for the linear models, even though offline metrics improved substantially.
5. **Proactive guidance is disabled** (see 6.2). Real conversational behaviour
   that this metric punishes.
6. **The LLM override fix is unvalidated** (see section 7). Re-run with quota
   available before relying on it.
7. **Slow startup**, about 100 seconds, mostly importing `torch`. A first-ever
   run needs ~12 minutes to build the embedding cache.
8. **REINFORCE is unusable.** It ranks worse than the first stage it is handed
   (MRR 0.681 vs a 0.730 baseline), so it can only destroy the ordering. It is
   excluded from `--models all`.
9. **Recall ceiling.** HitRate@10 is 0.970, so about 6 sessions never retrieve
   the target at all. No re-ranker can fix those.

---

## 9. What We Would Improve With More Time

**Short term:**

1. **Re-run the LLM evaluation with quota available**, to settle whether the
   conservative override fix turns the router from −0.0043 into a gain.
2. **Bootstrap confidence intervals** over the 200 sessions, so the choice
   among the top four models rests on evidence rather than judgement.
3. **Analyse the ~6 unreachable sessions** to see whether the recall ceiling is
   tokenisation, category drift, or vocabulary mismatch.

**Medium term:**

4. **Train on real conversational queries.** Harvest the actual query state at
   each turn from the 200 labelled sessions and use it as a validation set that
   reflects the true query distribution. This would also have caught the v1
   failure much earlier — and would fix the offline metric's blindness at the
   tails, which is the deeper problem behind section 5.
5. **Fine-tune the embedding model.** Retrieval uses a generic
   `all-MiniLM-L6-v2`; contrastive fine-tuning on catalog pairs should raise
   the recall ceiling that limits everything downstream.
6. **Speed up intent override**, still our slowest scenario at 4.13 turns.

**Longer term:**

7. **Add a cross-encoder re-ranker.** Implemented behind
   `CROSS_ENCODER_ENABLED` but off by default: 30–60 forward passes per turn on
   CPU contradicts the latency budget. Worth measuring properly.
8. **Learned fusion weights** conditioned on intent, extending dual-track from
   two hand-set weights to a learned policy.

---

## 10. Repository Map

```text
starter/agent.py                     the Agent class (entry point)
starter/reranker/                    re-ranker runtime code
  base.py, features.py               feature schema and extraction
  catalog_index.py                   category-tree indexes
  gbdt_inference.py                  LightGBM loading (guarded, CRLF-tolerant)
  mlp_inference.py                   numpy-only MLP forward pass
  artifacts_5k/                      SUBMITTED WEIGHTS (4,965-query training)
  artifacts_3k/, artifacts/          earlier weights, kept for comparison
training/                            training scripts (dev only)
notebooks/                           training and comparison notebooks
tests/                               unit tests
docs/reranker_eval_results.md        re-ranker experiment log
docs/first_stage_ablation.md         first-stage, personalization, token budget
evaluator/                           official evaluator - NEVER MODIFIED
.gitattributes                       stops git corrupting model artifacts
app.py, templates/index.html         Flask web UI
```

We did not modify any file under `evaluator/`, and we did not include any
organizer-only files, evaluation labels, or secrets.
