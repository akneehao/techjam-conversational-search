# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions unreleased until the Devpost submission deadline. After the deadline, the final evaluation package will be released and teams will run the unmodified official evaluator in their own environments using their frozen submitted commit.

See [`docs/final_evaluation_faq.md`](docs/final_evaluation_faq.md) for the final evaluation, network, credentials, hardware, data, and scoring policy.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Learned Re-Ranking Layer

`starter/reranker/` + `training/` add a learned re-ranking stage that
re-scores the top 60 of the BM25+dense RRF-fused list over 14 features:
semantic similarity, lexical overlap, category-tree structure, and the
first-stage BM25/dense/RRF rank scores. It is trained on self-supervised
labels built from the catalog itself -- no manual labeling and no paid API
calls.

**Enabled by default, worth +0.0803 TechnicalScore** on the public set,
measured with the unmodified `evaluator/local_evaluator.py`:

| Configuration | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| **simplex (default)** | 0.960 | **0.649** | 2.68 | **0.8410** |
| mlp | **0.965** | 0.624 | **2.67** | 0.8362 |
| gbdt | 0.955 | 0.609 | 2.71 | 0.8260 |
| coord_ascent | 0.945 | 0.622 | 2.73 | 0.8244 |
| ranksvm | 0.935 | 0.624 | 2.83 | 0.8181 |
| no re-ranking | 0.940 | 0.444 | 3.12 | 0.7607 |

Every trained model beats the plain RRF ordering on every component metric,
with the biggest gain in MRR (+46% relative for the default) -- the target
lands much closer to rank 1 when found. The default is the simplex-constrained
linear model: it scored highest here, and its artifact is an 11-number weight
vector scored with a single numpy dot product, so the default serving path
needs no lightgbm, torch, scipy or sklearn. Any other model is selectable with
`RERANK_MODEL=<name>`; note the top three sit within ~0.015 of each other on a
200-session sample, so see `docs/reranker_eval_results.md` before reading much
into that ordering.

Seven model types are implemented (fixed baseline, Simplex-constrained
linear, RankSVM, Coordinate Ascent, MLP, REINFORCE, GBDT, plus a GBDT+MLP
ensemble). An earlier training formulation scored *below* plain RRF for all
seven -- the root-cause analysis (label leakage, and a training task that was
nearly the opposite of the real one) is documented in
`docs/reranker_eval_results.md`, along with caveats on the small margins
between the top models. See `notebooks/training_pipeline.ipynb` and
`notebooks/model_comparison.ipynb` to reproduce training and inspect
performance.

Reproduce: `python -m training.label_generation` then
`python -m training.train_all`. Select a model with `RERANK_MODEL=<name>`;
disable entirely with `RERANK_ENABLED=0`.

Runtime cost/dependency disclosure: training and label generation run
entirely offline -- no external API calls, no added inference-time token cost
-- using only local compute (`lightgbm`, `scikit-learn`, `scipy`, `torch` are
training-time only; see `requirements-dev.txt`). At serving time the
re-ranker adds numpy operations plus one `lightgbm` inference call per turn,
and degrades gracefully to the plain RRF order if any dependency, the trained
artifact, or the flag is missing.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/final_evaluation_faq.md      final evaluation and judging clarifications
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
starter/reranker/                 optional learned re-ranking layer (disabled by default)
training/                         re-ranker training scripts (dev-only, not needed to run agent.py)
notebooks/                        training + model-comparison notebooks (dev-only)
docs/reranker_eval_results.md     re-ranker evaluation results and why it's disabled by default
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Final evaluation FAQ: `docs/final_evaluation_faq.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
