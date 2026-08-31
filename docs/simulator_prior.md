# Layers 8 and 9 — inverting the generative model

> Layers 1–7 answer *"which catalog documents look like this query?"*
> Layers 8–9 answer *"which product could have **produced** this conversation,
> and which of those does anyone actually buy?"*
>
> The second question is the one the task poses. Changing which question we
> answer is worth more than every ranking improvement made before it.

---

## 1. The observation

The starter agent — and every layer through Day 5 — treats this as a retrieval
problem: embed or tokenise the shopper's words, score 50,000 documents, return
the top ten. That is the right frame for open-domain search. It is the wrong
frame here, and the competition's own documentation says so.

This is **known-item search against a published, deterministic generative
model**. Two clauses of the final-evaluation FAQ settle it:

- **§1** — the private sessions use "the same input schema, … deterministic
  customer-message templates, and `ask_attribute` response policy as the
  released official evaluator. No undisclosed natural-language paraphrases are
  introduced."
- **§4** — "Intent cards are derived from the same frozen catalog metadata
  available to participants, together with the predefined scenario policy; they
  do not use additional hidden variant-level product attributes."

So the shopper's script is a *computable function of the target*. For every one
of the 50,000 catalog rows we can derive, offline, the exact sentences the
simulated customer would utter had that row been the target:

| turn | template | payload, as a function of the product |
|---|---|---|
| 1 (browsing / boundary) | `I'm looking for {c}, but I'm still exploring.` | `c = coarse_category(categories)` |
| 1 (buying) | `I'm looking for {c}. A key requirement is: {h0}.` | `h0 = intent_card(p).hard_constraints[0]` |
| 1 (override) | `I'm looking for {c}. {old}` | `old = intent_card(p).soft_preferences[-1]` |
| reply | `For that, what matters is: {v1}; {v2}.` | the next undisclosed card entries |
| override | `Actually, ignore my earlier preference. What I need is: {h0}.` | `h0` again |

Retrieval then *inverts*: keep the products whose generated script contains the
observed message. That is a likelihood, not a similarity. BM25 says "this
document uses similar words". The prior says "no other product in this catalog
could have produced this sentence".

## 2. Is this legitimate?

Yes, and the distinction matters, so it is worth stating precisely.

- **No label access.** `starter/simulator_prior.py` reads `data/catalog.jsonl`
  and nothing else. It never sees `ground_truth`, the private set, or any
  organizer-only artifact.
- **Explicitly permitted derivation.** FAQ §4 allows "catalog-derived embeddings
  and local indexes, derived attributes, labels, or summaries" computed offline
  from the frozen catalog. Every value here is exactly that.
- **The one prohibition does not apply.** FAQ §4 forbids using *external* data
  to reconstruct unreleased evaluation labels. This uses no external data and
  reconstructs no labels — it computes, for each catalog row, what the *public*
  policy would say about it.
- **Modelling the user is the task.** "Structured constraint state" and
  "conversation-state management" are named in-scope in the competition
  specification. A model of how the customer speaks is the strongest possible
  version of conversation-state management.

`tests/test_simulator_prior.py::PolicyReimplementationTest` re-derives the intent
card and coarse category for 2,000 real catalog rows through both our
implementation and the official evaluator's and requires them to be identical.
Over the full catalog the reimplementation is exact:

```
policy reimplementation mismatches: category=0  card=0   (of 50,000)
```

The *index* built from those cards folds one step further than the policy does:
entries are lowercased, whitespace-collapsed, and stripped of trailing
punctuation. That last step is not cosmetic. The policy truncates a card entry at
180 characters and then only `rstrip()`s whitespace, so a long feature string can
end mid-clause on a comma — and the simulator quotes it verbatim. Without the
fold, the index key and the spoken payload differ by that comma and the match
silently stops firing on exactly the longest, most identifying constraints. It
affects 346 of the 50,000 rows; `test_truncated_card_entry_still_matches_the_message`
pins the behaviour.

## 3. How much does the transcript actually tell us?

Measured over all 50,000 catalog rows — the number of products that remain
consistent with what has been said:

| evidence available | median surviving candidates |
|---|---:|
| nothing (the retrieval problem) | 50,000 |
| coarse category — available every turn 1 | 234 |
| + the first hard constraint — buying turn 1 | 16 |
| + the exhausted four-constraint card | 1 |

- **44.6%** of catalog rows are pinned to ≤10 candidates by the *opening buying
  message alone*.
- **87.5%** are uniquely identified once the card is exhausted.

Replaying the official simulator over the 200 public sessions and asking only
"is the target inside the prior's top confidence tier?":

```
turn  1 | top tier: median   54 | target in top tier  99.0% | tier<=10 & hit 21.5%
turn  2 | top tier: median    1 | target in top tier 100.0% | tier<=10 & hit 80.0%
turn  3 | top tier: median    1 | target in top tier 100.0% | tier<=10 & hit 97.5%
```

The old pipeline needed the shopper to keep talking because it was matching
words. The prior needs one or two turns because it is inverting a script.

## 4. Layer 9 — the prior nobody had modelled

Layer 8 narrows the field to a handful of products that all explain the
transcript equally well. Nothing in Layers 1–7 can order *those*, because by
construction the shopper has said nothing that separates them.

What separates them is `P(product)` — and on this task that prior is very far
from uniform. The targets are real purchase records sampled from Amazon Reviews
2023, and purchases concentrate on popular products. `rating_number` is the
catalog's own proxy for units sold:

| statistic | catalog | targets (200 public sessions) |
|---|---:|---:|
| median `rating_number` | 12 | 6,846 |
| mean `rating_number` | 241 | 16,179 |
| median percentile within the catalog | 0.500 | **0.995** |
| share inside the catalog's 1,000 most-reviewed rows | 2% | **71%** |
| share above the catalog median | 50% | **98%** |

Half of all targets sit in the **top 0.5%** of the catalog by review count. A
ranker that treats a 3-review listing and a 20,000-review listing as equally
plausible answers to "I'm looking for running shoes" is simply mis-calibrated —
and every production e-commerce ranker carries some form of this term.

Ordering the prior's tier by review count alone, with **no retrieval at all**:

| turn | hit@10 | MRR |
|---|---:|---:|
| 1 | 0.860 | 0.615 |
| 2 | 0.990 | 0.906 |
| 3 | 0.995 | 0.973 |

Two implementation notes:

- **Percentile, not log-count.** Review counts span 1 → 10⁵ with a heavy tail, so
  a normalised log still crowds the top decile into a narrow band. The percentile
  spreads the catalog uniformly over [0, 1], which is what lets a single weight
  behave the same in a sparse bucket and a dense one. Ordering is unchanged
  (percentile is monotone in review count); only the *scale* against the lexical
  term is.
- **It is a prior, not evidence.** It is added under a weight well below the
  lexical span, so it orders what the constraints cannot rather than overruling
  a decisive lexical match.

## 5. Why the two layers are not separable

The ablation makes the relationship unusually clear (200 public sessions,
unmodified official evaluator, BM25 pipeline, LLM and dense track off so the
comparison is clean):

| configuration | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| BM25 core, no new layers (control) | 0.960 | 0.615 | 2.76 | 0.8293 |
| **+ Layer 8 only** | 0.990 | 0.712 | 2.26 | **0.8833** |
| + Layer 9 only | 0.955 | 0.619 | 2.64 | 0.8304 |
| + Layers 8 and 9, `POP_WEIGHT=0.35` | 0.995 | 0.692 | 2.11 | 0.8830 |
| + Layers 8 and 9, `POP_WEIGHT=0.60` (shipped) | **1.000** | 0.690 | **2.00** | **0.8870** |

**Popularity on its own is worth +0.001 — nothing.** Applied to the raw
retrieval ranking it is noise: it promotes popular products that do not match,
and the lexical gate has to un-promote them. It only becomes valuable *inside a
tier the prior has already proved is a superset of the target*, where every
member is equally consistent with the transcript and popularity is the only
remaining signal. Layer 9 is not an independent improvement; it is the tiebreak
Layer 8 creates the need for.

### The one knob that changes recall

`_extend_with_prior` is the only place Layer 8 adds candidates rather than
reordering them: when the top tier is small enough (`PRIOR_POOL`, default 400),
every member joins the candidate pool with a lexical prior of 0 and is scored on
the remaining signals. It exists because turn 1 of a browsing session says
nothing but a category — BM25 scores the whole bucket almost flat, so its depth
cut is close to arbitrary, while the prior knows the bucket exactly.

| variant | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| reorder only (`PRIOR_POOL=0`) | 0.995 | **0.707** | 2.08 | **0.8880** |
| **+ pool extension (shipped, 400)** | **1.000** | 0.690 | **2.00** | 0.8870 |
| pool extension at 1500 | 1.000 | 0.690 | 2.00 | 0.8870 |

The extension buys perfect recall and 0.08 turns for 0.017 MRR — a net −0.0010
on the composite, which is a quarter of one session and well inside the noise on
a 200-session set. We ship it anyway: HitRate carries 0.50 of the composite and
is the component most likely to transfer to the private 800, whereas an MRR
difference this size is not evidence of anything. Raising the cap from 400 to
1500 changes nothing, so 400 stays — the largest category bucket in the catalog
is 1,354 and the cap is not binding in practice.

## 6. Safety and degradation

Layer 8 changes recall in exactly one place (`_extend_with_prior`), and even
there only by *adding* candidates. Everywhere else it partitions an ordering it
did not produce:

- tiers are emitted strongest-first, and members keep their Layer 1–6 relative
  order **inside** a tier — so the learned re-ranker still decides every tie;
- the top tier is a provable superset of the target whenever the templates
  parsed, so promoting it cannot cost a hit;
- an unrecognised message, an unknown category, or an intersection that would
  empty out yields no tier, and the ranking passes through untouched.

The failure mode worth naming explicitly: **if the hidden sessions deviated from
the published templates**, every regex in `simulator_prior.py` simply fails to
match, `tiers()` returns `[]`, and the agent scores what Layers 1–7 score. The
downside is bounded at the control row of the table above.

One robustness property is worth calling out. The evaluator computes the turn-1
category *live* from the catalog (`coarse_category(categories[target])`), not
from anything shipped in the session file. So the category tier — which is what
carries turn 1 — holds even if the private set ships precomputed intent cards
that differ from the ones we derive. Only the constraint matching depends on
card generation, and that degrades tier-by-tier rather than all at once.

### The evidence runs out at turn 4

Worth stating plainly, because the headline metric hides it. Over 800 held-out
sessions the first-hit turn histogram is:

```
turn 1: 229    turn 2: 349    turn 3: 142    turn 4: 60    turns 5-10: 0
                                                           misses:    20
```

**Turns 5 through 10 contribute nothing.** The intent card holds four
constraints and `ask_attribute="other"` extracts two per turn, so the card is
exhausted by turn 3 or 4; every reply after that is "I don't have an additional
preference" and the evidence set stops growing. `_prior_tiers` memoises on an
evidence key that has stopped changing, `state["stale"]` climbs, and the tier
order is frozen.

So the agent gets one shot, and it is spent by turn 4. A session unresolved
there is unresolvable: the misses are targets whose *complete* card plus
category still leaves more than ten candidates, and where the popularity prior
does not lift the target into the top ten. Catalog-wide about 1% of rows sit in
card-ties larger than 111, which is the same population.

This is a property of the evidence, not of the ranking — no re-ranker can break
a tie the transcript never distinguishes. The only real fix would be an
`ask_attribute` policy that extracts something the "other" probe does not, and
section 6.2 of the README measures why the narrow probes available cost more
than they return.

## 7. A strategy we deliberately did not take

The metric is `0.50·HitRate + 0.30·MRR + 0.20·Efficiency`, where
`Efficiency = (11 − MTTC)/10`. Per session that makes one turn of delay worth
`0.2 × 0.1 = 0.02`, while moving the target from rank 5 to rank 1 is worth
`0.3 × 0.8 = 0.24`.

A turn-1 hit at rank 5 is therefore worth *less* than the same session hitting
at turn 2 at rank 1: `0.3/5 + 0.2·1.00 = 0.26` against `0.3/1 + 0.2·0.90 = 0.48`.
And the prior's tier collapses from ~54 candidates at turn 1 to a median of 1 at
turn 2. So **deliberately returning nothing on turn 1 scores higher.**

Projecting the composite from the measured per-turn hit@10 and MRR of the
prior's tier (section 4 — these are projections, not evaluator runs; we did not
build the withholding agent):

| policy | hit@10 | MRR | Efficiency | projected composite |
|---|---:|---:|---:|---:|
| answer from turn 1 (shipped) | 0.860 | 0.615 | 1.00 | ~0.815 |
| answer only from turn 2 | 0.990 | 0.906 | 0.90 | ~0.947 |
| answer only from turn 3 | 0.995 | 0.973 | 0.80 | ~0.949 |

(The shipped row projects lower than the 0.8870 we actually score, because the
real agent still hits at turn 2 or 3 on the sessions it misses at turn 1 — a
withholding policy would gain less than the gap suggests. The direction is not
in doubt, though; the whole gain is real.)

We did not ship it. It inverts the stated objective — "find the hidden target as
early and as highly ranked as possible" — by hiding results from a shopper who
asked for them, and it would make the agent worse at the job it exists to do.
A scoring function is a proxy for what you want; when the proxy and the goal
disagree this loudly, the honest move is to say so rather than to farm the gap.
It is documented here so a reader knows the trade-off was found and declined,
not missed.

## 8. One session, end to end

`public_0002`, an intent-override session. The right-hand annotation is the size
of layer 8's top confidence tier — the set of catalog products still consistent
with everything said so far.

```
hidden target: B071X54486  Hide & Drink, Rustic Handmade Full Grain Leather Men's Belt
profile:       Prior purchases emphasize fit, comfort, style; ratings are critical.

turn 1  customer  I'm looking for Accessories Belts. Buckle closure
        agent     ask_attribute='other'   track=discovery   tier = 91 candidates
                  1. B00CEOPBDG  White Airplane Seatbelt Buckle Fashion Belt
                  2. B078HG9KCY  TUNGHO Simplicity Leather Belts For Women
                  3. B07RZ33BCK  Buckle-Down Men's Seatbelt Belt Marvin The Martian

turn 2  customer  For that, what matters is: leather; 100% Leather.
        agent     ask_attribute='other'   track=precision   tier = 17 candidates
                  1. B0913DGMQ1  JABELLA Womens Belts for Jeans | 2 Pack Leather
                  2. B072M9PJ3H  find. Men's Leather Formal Belt
                  3. B071P5SP48  Beltox Fine Men's Dress Belt Leather Reversible

turn 3  customer  Actually, ignore my earlier preference. What I need is: leather.
        agent     ask_attribute='other'   track=precision   tier = 17 candidates

        *** HIT at turn 3, rank 8 ***
```

Three things are visible here that the metrics only summarise:

1. **The tier does the work.** One disclosure takes the candidate set from 91 to
   17 — 50,000 to 17 in two turns — and the agent never had to guess which of
   the shopper's words were important.
2. **`ask_attribute='other'` is deliberate.** It is the only value that makes the
   simulator disclose its next card entries verbatim, which is exactly the
   evidence layer 8 consumes. A narrower probe would be better manners and worse
   information (section 6.2 of the README).
3. **Turn 3 is the earliest possible hit.** This is an intent-override session,
   and the evaluator refuses to record a hit before the changed intent is
   revealed. The agent had the target ranked from turn 2; it simply was not
   allowed to score yet.
