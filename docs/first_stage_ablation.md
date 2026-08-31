# First-stage retrieval, personalization, and the token budget

Companion to `docs/reranker_eval_results.md`. That document covers the learned
re-ranking layer; this one covers the retrieval stage underneath it, the
aggregate-profile personalization, and the LLM cost budget.

All numbers below are full `python -m evaluator.local_evaluator` runs on the
200 public sessions with `AGENT_USE_LLM=0` (no Gemini key), so they are
directly comparable to the tables in `reranker_eval_results.md`.

## Headline ablation

| Configuration | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| **`FUSION_MODE=rerank` + `RERANK_ENABLED=1`** | 0.960 | **0.682** | 2.75 | **0.8498** |
| `FUSION_MODE=rrf` + `RERANK_ENABLED=1` (previous ship) | 0.960 | 0.649 | 2.68 | 0.8410 |
| `FUSION_MODE=bm25`, dense off, no re-rank | 0.960 | 0.647 | 2.71 | 0.8398 |
| `FUSION_MODE=rerank` + `RERANK_ENABLED=0` | 0.960 | 0.643 | 2.71 | 0.8387 |
| `FUSION_MODE=rrf`, `RRF_K=10`, no re-rank | 0.930 | 0.517 | 3.04 | 0.7794 |
| `FUSION_MODE=rrf`, `RRF_K=60`, no re-rank (old control) | 0.940 | 0.448 | 3.13 | 0.7617 |

The shipped configuration is the top row: the rewritten first stage with the
learned re-ranker on top, **+0.0088 over the previous default**.

## The Day 4 fusion defect

Day 4 fused two equal-length (60) ranked lists with symmetric RRF at `k=60`.
That is degenerate in two provable ways.

**1. `k == depth` flattens the curve.** Rank 1 scores `1/61 = .01639`, rank 60
scores `1/120 = .00833` — a 1.97x span. A document ranked *last in both lists*
(`2/120 = .01667`) therefore outranks the #1 BM25 hit that dense missed
(`.01639`). Fusion had collapsed into a co-occurrence vote.

**2. Equal weights + equal depth make the fused order an exact rank-merge.**
For a document in only one list the score is `1/(60 + rank)` regardless of
*which* list, so the top-10 was `BM25[1..5]` interleaved with `DENSE[1..5]`.
Dense was silently taking half the top-10, and its picks bypassed the tiered
category gate in `_build_queries` entirely.

It is also the wrong prior for this task. `intent_card()` in the evaluator
builds the shopper's constraints by copying verbatim substrings out of the
target product's own `features`/`details`, so the shopper quotes the target
document. This is known-item **lexical** retrieval with one relevant document
in 50,000 — BM25 owns it, and a bi-encoder cannot separate 50k near-duplicate
apparel items.

Measured cost of the defect: **0.8398 -> 0.7617, i.e. -0.078**. Note the damage
is concentrated in MRR (0.647 -> 0.448, -31%), not HitRate@10 (0.960 -> 0.940):
the flat curve did not push targets out of the top ten so much as shuffle them
down *within* it. Retuning `k` 60 -> 10 recovers only +0.018 of that; the
remaining +0.059 requires the architectural change. **The parameterisation was
not the main problem — letting a weak track inject candidates as a peer was.**

### The fix

`FUSION_MODE=rerank` (default): BM25 alone decides *which* documents are
candidates (`RERANK_DEPTH=120`); dense only reorders *within* that set.

    score = lex_prior + DENSE_WEIGHT * cosine + PROFILE_WEIGHT * profile_match

`lex_prior` is `1/(LEX_K + rank)` renormalised so rank 1 == 1.0, spanning
1.0 -> 0.085 over 120 candidates. Learned terms are min-max normalised per turn
and weighted well below that span, so semantics reorder neighbours but cannot
overturn a decisive lexical match.

The consequence worth noting: **HitRate@10 is 0.960 in every `rerank` row**,
with dense on or off, re-ranker on or off. Recall is now a pure function of the
lexical gate, so no learned component can cost a hit.

## Dense retrieval is negative here; dense *features* are not

As a retriever the bi-encoder is measurably harmful — even in its safest
configuration it costs -0.001 (0.8398 -> 0.8387), and under the Day 4 fusion it
cost -0.078. This is expected given the verbatim-quoting setup above.

As a *feature source* for the learned re-ranker it is fine, and that is the use
`starter/reranker/` makes of it. The distinction matters: the fix constrains
dense from choosing candidates, it does not discard the embeddings.

## The simplex artifact independently confirms this

The shipped `simplex_weights.json` is **one-hot on `lexical_score`** — weight
1.0 there, and numerically zero (1e-16 or exact 0.0) on all thirteen other
features, including `rrf_score`, `bm25_rank_score`, `dense_rank_score` and
every embedding / category-tree feature.

This is not an artifact of the simplex constraint. The loss is nonlinear
(sigmoid BCE + pairwise log-loss), and its L2 term actively penalises one-hot
solutions: `0.01 * ||w||^2` costs 0.01 at a vertex versus 0.0007 at uniform
weights. SLSQP paid that penalty anyway, which means `lexical_score` dominated
every mixture by more than the regularisation gap.

So a constrained optimiser over 45k training pairs reached the same conclusion
as the structural analysis above: **this task is lexical; discard the fusion.**
That is also why the previous ship (0.8410) and a plain BM25 pipeline (0.8398)
land 0.0012 apart — a quarter of the ~0.005/session noise floor. Two different
routes to the same place.

### But the layers still compose

The obvious inference — that the two fixes are redundant — is wrong, and the
measurement says so: fixed stage alone 0.8387, fixed stage + simplex **0.8498**,
a **+0.0111** contribution from the learned layer on a sound first stage.

`lexical_score` is not a cruder BM25. BM25 is IDF-weighted and field-weighted
over the FTS columns; `lexical_score` is raw query-token *coverage*
(`|q ∩ doc| / |q|`) against title + categories + features. Weighted-match and
coverage are different signals and they add, almost entirely through MRR
(0.643 -> 0.682). Measuring the re-ranker against a broken control understated
what it actually contributes.

## Personalized context distillation

`reset()` distills the aggregate `user_profile` (`preference_tags`, `summary`)
once per session. It reaches the agent by exactly two routes: a compact
`PROFILE:` line in the router prompt, and a small additive re-rank boost.

**Neither route can add or remove a retrieval candidate.** That separation is
deliberate. Over the 200 public sessions `preference_tags` is close to constant:

| tag | share of sessions |
|---|---|
| fit | 82% |
| material | 77% |
| comfort | 72% |
| style | 50% |
| durability | 24% |

All five are already in `GENERIC` — they describe essentially every apparel
product — and they derive from the shopper's *prior* purchases, which the
evaluator never links to the target. AND-ing them into the FTS query, or even
OR-ing them into the recall net, would inject noise into 100% of sessions.

`_distill_profile()` therefore filters tags through `GENERIC`, which drops the
five above and keeps the rare discriminating ones (`warmth` 9%, `weather` 6%,
`performance` 13%). Anything the LLM proposes in its `pf` field lands in the
same boost lane, never in `keywords`. `PROFILE_WEIGHT=0.05` is smaller than one
rank step at the head of the lexical prior (0.083), so it breaks ties and
nothing more.

This is the spec's "**safe** personalization using the aggregate profile":
the mechanism is real and demonstrable, and it is wired so that a stale or
hallucinated profile term cannot filter out the target.

## Token and latency budget

Measured on a representative mid-conversation turn (chars/4 estimate):

| | Day 2 | Day 5 | change |
|---|---|---|---|
| system prompt | ~302 tok | ~217 tok | -28% |
| context block | ~60 tok | ~50 tok | -17% |
| completion | ~59 tok | ~46 tok | -22% |
| **per call** | **~421 tok** | **~313 tok** | **-26%** |
| **per session** | ~4210 tok | ~1880 tok | **-55%** |

Levers: a terse router prompt, single-letter JSON keys (`i`/`sl`/`kw`/`ov`/`pf`)
pinned by `responseSchema`, `LLM_MAX_TOKENS` 512 -> 192, `_LLM_TIMEOUT`
20s -> 8s, and `LLM_MAX_CALLS_PER_SESSION=6`. `_normalize_route()` accepts both
the compact and the original long-key shapes, so schema drift cannot break
parsing.

## Failure behaviour

| condition | result |
|---|---|
| no Gemini key / timeout / bad JSON | deterministic parser only |
| 3 consecutive LLM errors | breaker trips for the run |
| per-session LLM budget exceeded | deterministic parser only |
| no numpy / sentence-transformers | pure BM25 ranking |
| encode or cross-encode failure | that term contributes 0.0 |
| malformed FTS expression | that tier contributes no rows |
| re-ranker artifact/dep missing | candidate order passes through |
| anything else in `respond()` | `_last_resort` BM25 response |

Verified against corrupted session state, hostile FTS metacharacters, a raising
encoder, LLM timeouts, and one genuine unplanned fault: the MiniLM download
failed with `CERTIFICATE_VERIFY_FAILED` on a TLS-intercepting network and the
agent served BM25 results throughout.

`_last_resort` and the `last_ranked` backstop mean an unparseable turn returns
the previous good list rather than nothing — an empty turn cannot score.

## Reproducing

```bash
# one-time: 50k dense vectors (~12 min CPU), needs the model downloaded once
python -c "from starter.agent import Agent; Agent('data/catalog.jsonl')"

AGENT_USE_LLM=0 HF_HUB_OFFLINE=1 FUSION_MODE=rerank RERANK_ENABLED=1 \
  python -m evaluator.local_evaluator --output results.json
```

On a TLS-intercepting network the one-time model download needs the OS trust
store: `pip install truststore`, then `import truststore;
truststore.inject_into_ssl()` before constructing the `Agent`. Certificate
verification stays enabled — do not disable it. After the cache exists,
`HF_HUB_OFFLINE=1` avoids the network entirely.

## The Gemini router, measured

One run with a live key, same shipped configuration:

| | HitRate@10 | MRR | MTTC | TechnicalScore | tokens |
|---|---|---|---|---|---|
| `AGENT_USE_LLM=0` | 0.960 | 0.682 | 2.75 | 0.8498 | 0 |
| `AGENT_USE_LLM=1` | 0.955 | 0.682 | 2.77 | 0.8468 | 5,536 |

**Do not read that second row as "the router costs 0.003".** The run is not a
valid measurement of the LLM path, for a reason worth recording.

The free-tier key returns HTTP 429 after ~16 calls. `reported_token_usage` of
5,536 confirms it: at ~342 tokens/call that is 16 calls across 200 sessions,
not the ~1,200 a full run needs. Three consecutive 429s then tripped the
circuit breaker, and because the evaluator constructs one `Agent` for all 200
sessions, the breaker stayed tripped for the remaining ~184. So the row above
is roughly 8 LLM-routed sessions followed by 192 deterministic ones -- within
noise of the `AGENT_USE_LLM=0` row, which is exactly what it mostly is.

**The router's real contribution remains unmeasured**, and needs a key with
enough quota for ~1,200 calls (or a slow run pacing under the rate limit).
Verified working in isolation: a single live call returns the compact schema
correctly and the profile round-trips
(`{"i":"buying","sl":{"category":"jackets","gender":"men","material":"wool"},
"kw":["100%","wool"],"ov":false,"pf":["warmth"]}` for a `warmth` profile).

### What that exposed

The Day 2 breaker tripped **permanently** after any 3 consecutive failures,
regardless of cause. In a 200-session batch run against one `Agent`, that turns
a transient rate limit into a silently disabled feature for the rest of the
run -- and the run still looks like it had an LLM.

Failures are now classified:

| cause | behaviour |
|---|---|
| 400 / 401 / 403 (bad key, bad request, API disabled) | trip permanently -- retrying cannot help |
| 429 / 5xx / timeout / transport | exponential backoff 2s -> 60s, recovers on its own |
| success | streak and cooldown reset |

Cost per call, measured live: 283 prompt + 59 completion = **342 tokens**
(estimate was 313).

## Open items
- The re-ranker requires the dense track (`_init_reranker` gates on
  `_dense_on()`), because its feature extractor needs the document vectors —
  even for simplex, whose one live feature is lexical. Dense therefore cannot
  be disabled in the shipped configuration.
- The re-ranker was trained on features computed from the *RRF-fused* ordering.
  It is applied here to the re-ranked ordering without retraining. It still
  gains +0.0111, but retraining `label_generation` against the new first stage
  is untested and might add more.
- Public-set optimism applies to the learned layer as noted in
  `reranker_eval_results.md`; the first-stage fix has no learned parameters and
  nothing to overfit.
