"""Held-out generalization harness for layers 8 and 9.

HitRate@10 = 1.000 on the public set is a point estimate on 200 sessions, and a
metric that pins at its ceiling deserves to be attacked rather than reported.
This builds fresh sessions from catalog products that were never involved in any
tuning decision and scores them with the UNMODIFIED official ``evaluate()``.

Two target populations, because they answer different questions:

  * ``uniform``  -- drawn uniformly from the catalog. Deliberately unfair: real
    targets are purchase records and sit at a median review-count percentile of
    0.995, so a uniform draw strips layer 9 of its prior and stress-tests the
    inversion on its own.
  * ``matched``  -- drawn to match the public set's review-count distribution,
    which is the honest proxy for the unreleased evaluation sessions.

Nothing here is used at serving time; it is a development tool.

    python -m training.heldout_generalization 800
"""

from __future__ import annotations

import bisect
import math
import random
import sys

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent

CATALOG = "data/catalog.jsonl"
PUBLIC = "data/public_set.jsonl"
SEED = 20260901

# The official mix is 40/40/15/5.  Spelled as an integer pattern on purpose:
# accumulating the floats puts 0.40 + 0.40 + 0.15 at 0.9500000000000001, so the
# final bucket never fires and every boundary session silently becomes an
# intent_override one.
MIX_PATTERN = ["buying"] * 8 + ["browsing"] * 8 + ["intent_override"] * 3 + ["boundary"] * 1


def wilson_lower_bound(hits: int, n: int, z: float = 1.96) -> float:
    """Lower end of the Wilson score interval -- correct at p = 1, unlike Wald."""
    if n == 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def build_sessions(targets: list[str], profiles: list[dict], tag: str) -> list[dict]:
    rng = random.Random(SEED)
    return [
        {
            "sample_id": f"{tag}_{index:04d}",
            "scenario_type": MIX_PATTERN[index % len(MIX_PATTERN)],
            "user_profile": profiles[rng.randrange(len(profiles))],
            "ground_truth": {"parent_asin": target},
        }
        for index, target in enumerate(targets)
    ]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = random.Random(SEED)

    catalog_ids, categories, products = catalog_index(CATALOG)
    public = load_jsonl(PUBLIC)
    public_targets = {str(item["ground_truth"]["parent_asin"]) for item in public}
    profiles = [item["user_profile"] for item in public]

    reviews = {asin: float(p.get("rating_number") or 0) for asin, p in products.items()}
    sorted_counts = sorted(reviews.values())
    # Every catalog row EXCEPT the 200 the system was developed against.
    pool = [asin for asin in products if asin not in public_targets]
    by_count = sorted(pool, key=lambda asin: reviews[asin])

    def percentile(asin: str) -> float:
        return bisect.bisect_left(sorted_counts, reviews[asin]) / len(sorted_counts)

    uniform_targets = rng.sample(pool, n)

    # Resample the public targets' empirical percentiles and draw a catalog
    # product from the same neighbourhood, so the popularity profile matches.
    public_percentiles = [percentile(t) for t in public_targets]
    matched_targets: list[str] = []
    taken: set[str] = set()
    while len(matched_targets) < n:
        q = public_percentiles[rng.randrange(len(public_percentiles))]
        index = int(q * len(by_count)) + rng.randint(-40, 40)
        candidate = by_count[min(len(by_count) - 1, max(0, index))]
        if candidate not in taken:
            taken.add(candidate)
            matched_targets.append(candidate)

    agent = Agent(CATALOG, use_llm=False)
    print(
        f"agent ready: dense={agent._dense_on()} "
        f"reranker={agent._reranker is not None} prior={agent._prior is not None}",
        flush=True,
    )

    for tag, targets in (("uniform", uniform_targets), ("matched", matched_targets)):
        percentiles = sorted(percentile(t) for t in targets)
        sessions = build_sessions(targets, profiles, tag)
        agent._state.clear()
        result = evaluate(agent, sessions, catalog_ids, categories, products)
        hits = sum(1 for s in result["sessions"] if s["hit"])

        print(
            f"\n=== {tag} targets (n={len(sessions)}, "
            f"median review-count percentile {percentiles[len(percentiles) // 2]:.3f}) ===",
            flush=True,
        )
        print(
            "  hit@10 %.4f   MRR %.4f   MTTC %.3f   score %.4f   "
            "(95%% Wilson lower bound on hit@10: %.4f)"
            % (
                result["hit_rate_at_10"], result["mrr"], result["mttc"],
                result["recommended_technical_score"], wilson_lower_bound(hits, len(sessions)),
            ),
            flush=True,
        )
        for scenario in sorted(result["scenario_metrics"]):
            m = result["scenario_metrics"][scenario]
            print(
                "    %-16s n=%3d  hit %.3f  MRR %.3f  MTTC %.2f"
                % (scenario, m["sample_count"], m["hit_rate_at_10"], m["mrr"], m["mttc"]),
                flush=True,
            )
        turns: dict[int, int] = {}
        for s in result["sessions"]:
            if s["hit"]:
                turns[s["first_hit_turn"]] = turns.get(s["first_hit_turn"], 0) + 1
        print("    first-hit turn histogram:", dict(sorted(turns.items())), flush=True)
        missed = [s for s in result["sessions"] if not s["hit"]]
        if missed:
            print(
                "    %d misses, e.g. %s"
                % (len(missed), [(s["sample_id"], s["scenario_type"]) for s in missed[:8]]),
                flush=True,
            )


if __name__ == "__main__":
    main()
