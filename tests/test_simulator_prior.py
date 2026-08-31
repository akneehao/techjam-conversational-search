"""Tests for Layer 8 (simulator-policy inversion) and Layer 9 (popularity prior).

The prior is only sound while our copy of the scenario policy agrees with the
evaluator's, so the first test is the load-bearing one: it re-derives the intent
card and coarse category for real catalog rows through both implementations and
requires them to be identical.  If the organizer ever changes the policy, that
test fails loudly instead of the agent silently ranking on a stale model.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from evaluator import local_evaluator as official
from starter.agent import Agent
from starter.simulator_prior import SimulatorPrior, coarse_category, intent_card

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CATALOG_PATH = _REPO_ROOT / "data" / "catalog.jsonl"

_TINY_CATALOG = [
    {"parent_asin": "A1", "title": "Blue Cotton T-Shirt", "features": ["cotton", "crew neck"],
     "description": ["A soft blue tee"], "categories": ["Clothing", "Men", "Shirts", "T-Shirts"],
     "details": {}, "price": 20.0, "rating_number": 5},
    {"parent_asin": "A2", "title": "Red Wool T-Shirt", "features": ["wool", "v neck"],
     "description": ["A soft red tee"], "categories": ["Clothing", "Men", "Shirts", "T-Shirts"],
     "details": {}, "price": 22.0, "rating_number": 900},
    {"parent_asin": "A3", "title": "Black Leather Jacket", "features": ["leather", "zip front"],
     "description": ["A stylish black jacket"], "categories": ["Clothing", "Men", "Jackets"],
     "details": {}, "price": 100.0, "rating_number": 40},
]
_TINY = {p["parent_asin"]: p for p in _TINY_CATALOG}


def _catalog_sample(limit: int) -> list[dict]:
    products: list[dict] = []
    with _CATALOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            products.append(json.loads(line))
            if len(products) >= limit:
                break
    return products


class PolicyReimplementationTest(unittest.TestCase):
    """Our copy of the published policy must be bit-identical to the evaluator's."""

    @unittest.skipUnless(_CATALOG_PATH.is_file(), "catalog.jsonl not present")
    def test_matches_official_policy_on_real_catalog_rows(self) -> None:
        for product in _catalog_sample(2000):
            categories = [str(v) for v in (product.get("categories") or [])]
            self.assertEqual(coarse_category(categories), official.coarse_category(categories))
            self.assertEqual(intent_card(product), official.intent_card(product))


class TemplateParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prior = SimulatorPrior(_TINY)

    def test_browsing_opener_yields_category_only(self) -> None:
        evidence = self.prior.new_evidence()
        self.prior.observe(evidence, "I'm looking for Shirts T-Shirts, but I'm still exploring.", first=True)
        self.assertEqual(evidence["category"], "shirts t-shirts")
        self.assertEqual(evidence["values"], [])

    def test_buying_opener_yields_category_and_positional_constraint(self) -> None:
        evidence = self.prior.new_evidence()
        self.prior.observe(evidence, "I'm looking for Shirts T-Shirts. A key requirement is: wool.", first=True)
        self.assertEqual(evidence["category"], "shirts t-shirts")
        self.assertEqual(evidence["first_hard"], "wool")
        self.assertIn("wool", evidence["values"])

    def test_disclosure_reply_splits_on_semicolons(self) -> None:
        evidence = self.prior.new_evidence()
        self.prior.observe(evidence, "For that, what matters is: wool; v neck.", first=False)
        self.assertEqual(evidence["values"], ["wool", "v neck"])

    def test_no_preference_reply_adds_no_evidence(self) -> None:
        evidence = self.prior.new_evidence()
        self.prior.observe(evidence, "I don't have an additional preference for color.", first=False)
        self.assertEqual(evidence["values"], [])
        self.assertIsNone(evidence["category"])

    def test_truncated_card_entry_still_matches_the_message(self) -> None:
        """A 180-character truncation can leave a card entry ending on a comma.

        The simulator quotes that entry verbatim, so the index key and the
        observed payload have to fold to the same form or the match silently
        stops firing on exactly the longest, most identifying constraints.
        """
        # Place a comma at index 179 so the policy's [:180] slice ends on it.
        prefix = "premium ringspun combed jersey knit with taped shoulders "
        feature = prefix + "d" * (179 - len(prefix)) + ", and more detail past the cut"
        catalog = {"L1": {"parent_asin": "L1", "title": "Long Feature Tee",
                          "features": [feature], "description": [],
                          "categories": ["Clothing", "Men", "Shirts", "Tees"],
                          "details": {}, "price": None, "rating_number": 10}}
        prior = SimulatorPrior(catalog)
        spoken = intent_card(catalog["L1"])["hard_constraints"][0]
        self.assertTrue(spoken.endswith(","), spoken[-20:])   # the case under test

        evidence = prior.new_evidence()
        prior.observe(
            evidence, "I'm looking for Shirts Tees. A key requirement is: %s." % spoken, first=True
        )
        self.assertEqual(evidence["first_hard"], prior.card["L1"][0])
        self.assertEqual(prior.tiers(evidence, 5000)[0], ["L1"])

    def test_unrecognised_message_is_inert(self) -> None:
        evidence = self.prior.new_evidence()
        self.prior.observe(evidence, "hey do you sell anything nice", first=True)
        self.assertIsNone(evidence["category"])
        self.assertEqual(evidence["values"], [])
        self.assertEqual(self.prior.tiers(evidence, 5000), [])


class TierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prior = SimulatorPrior(_TINY)

    def test_category_alone_is_a_superset_tier(self) -> None:
        evidence = self.prior.new_evidence()
        self.prior.observe(evidence, "I'm looking for Shirts T-Shirts, but I'm still exploring.", first=True)
        tiers = self.prior.tiers(evidence, 5000)
        self.assertEqual(sorted(tiers[0]), ["A1", "A2"])

    def test_constraint_isolates_the_target(self) -> None:
        evidence = self.prior.new_evidence()
        self.prior.observe(evidence, "I'm looking for Shirts T-Shirts. A key requirement is: wool.", first=True)
        tiers = self.prior.tiers(evidence, 5000)
        self.assertEqual(tiers[0], ["A2"])

    def test_unknown_category_does_not_invent_a_tier(self) -> None:
        evidence = self.prior.new_evidence()
        self.prior.observe(evidence, "I'm looking for Hats Beanies, but I'm still exploring.", first=True)
        self.assertEqual(self.prior.tiers(evidence, 5000), [])


class AgentIntegrationTest(unittest.TestCase):
    """End-to-end: the layers must change ranking without breaking the contract."""

    def _agent(self, tmp: Path) -> Agent:
        catalog_path = tmp / "catalog.jsonl"
        with catalog_path.open("w", encoding="utf-8") as handle:
            for product in _TINY_CATALOG:
                handle.write(json.dumps(product) + "\n")
        return Agent(catalog_path, use_llm=False, use_dense=False)

    def test_prior_promotes_the_only_product_that_fits_the_transcript(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(Path(tmp))
            agent.reset("s", {})
            response = agent.respond(
                "s", "I'm looking for Shirts T-Shirts. A key requirement is: wool.", 1, 10
            )
            self.assertEqual(response["recommendations"][0]["parent_asin"], "A2")
            self.assertIsInstance(response["message"], str)

    def test_unparseable_message_still_returns_a_valid_response(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(Path(tmp))
            agent.reset("s", {})
            response = agent.respond("s", "hello there", 1, 10)
            self.assertIsInstance(response["recommendations"], list)
            self.assertLessEqual(len(response["recommendations"]), 10)

    def test_popularity_orders_products_the_constraints_cannot_separate(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(Path(tmp))
            agent.reset("s", {})
            response = agent.respond(
                "s", "I'm looking for Shirts T-Shirts, but I'm still exploring.", 1, 10
            )
            ranked = [item["parent_asin"] for item in response["recommendations"]]
            # A1 and A2 are the whole category tier; A2 has 900 reviews to A1's 5.
            self.assertEqual(ranked[:2], ["A2", "A1"])


if __name__ == "__main__":
    unittest.main()
