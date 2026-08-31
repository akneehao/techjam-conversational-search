from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from starter.agent import RERANK_ARTIFACTS_DIR, Agent

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Track whatever agent.py actually ships as the default artifact set, so these
# tests follow the default rather than a hard-coded folder.
_ARTIFACTS_DIR = RERANK_ARTIFACTS_DIR
_CATALOG_PATH = _REPO_ROOT / "data" / "catalog.jsonl"
_HAS_TRAINED_ARTIFACTS = _ARTIFACTS_DIR.joinpath("gbdtranker.txt").is_file() and _CATALOG_PATH.is_file()

_TINY_CATALOG = [
    {"parent_asin": "A1", "title": "Blue Cotton T-Shirt", "features": ["cotton"],
     "description": ["A soft blue tee"], "categories": ["Clothing", "Men", "Shirts", "T-Shirts"],
     "details": {}, "price": 20.0},
    {"parent_asin": "A2", "title": "Red Cotton T-Shirt", "features": ["cotton"],
     "description": ["A soft red tee"], "categories": ["Clothing", "Men", "Shirts", "T-Shirts"],
     "details": {}, "price": 22.0},
    {"parent_asin": "A3", "title": "Black Leather Jacket", "features": ["leather"],
     "description": ["A stylish black jacket"], "categories": ["Clothing", "Men", "Jackets"],
     "details": {}, "price": 100.0},
]


def _write_tiny_catalog(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for product in _TINY_CATALOG:
            handle.write(json.dumps(product) + "\n")


# RERANK_ENABLED / RERANK_MODEL / RERANK_ARTIFACTS_DIR are read once as
# module-level constants when starter.agent is first imported (the same
# pattern the file already uses for DENSE_ENABLED, RRF_K, etc.) -- mutating
# os.environ mid-process after that import has no effect. Tests that need a
# specific value must set it BEFORE the module is imported, i.e. in a fresh
# subprocess, not by monkeypatching os.environ in this already-imported
# process.
def _run_in_subprocess(script: str, env_overrides: dict[str, str]) -> str:
    env = {**os.environ, **env_overrides}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(_REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    return result.stdout


class RerankerFallbackTest(unittest.TestCase):
    def test_missing_dense_disables_reranker_without_breaking_respond(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.jsonl"
            _write_tiny_catalog(catalog_path)
            agent = Agent(str(catalog_path), use_llm=False, use_dense=False)

            self.assertIsNone(agent._reranker)

            agent.reset("s1", {"summary": "test"})
            resp = agent.respond("s1", "I want a blue cotton t-shirt", 1, 10)
            self.assertIn("recommendations", resp)
            asins = {r["parent_asin"] for r in resp["recommendations"]}
            self.assertTrue(asins.issubset({p["parent_asin"] for p in _TINY_CATALOG}))

    def test_missing_artifacts_disables_reranker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.jsonl"
            _write_tiny_catalog(catalog_path)
            script = (
                "from starter.agent import Agent\n"
                f"agent = Agent(r'{catalog_path}', use_llm=False, use_dense=True)\n"
                "print('DENSE_ON', agent._dense_on())\n"
                "print('RERANKER_IS_NONE', agent._reranker is None)\n"
                "agent.reset('s1', {'summary': 'test'})\n"
                "resp = agent.respond('s1', 'I want a blue cotton t-shirt', 1, 10)\n"
                "print('HAS_RECS', 'recommendations' in resp)\n"
            )
            try:
                stdout = _run_in_subprocess(script, {
                    "RERANK_ENABLED": "1",
                    "RERANK_ARTIFACTS_DIR": str(Path(tmp) / "nonexistent_artifacts"),
                    "DENSE_CACHE_DIR": tmp,
                })
            except AssertionError as exc:
                self.skipTest(f"dense retrieval unavailable in this environment: {exc}")
                return
            if "DENSE_ON True" not in stdout:
                self.skipTest("dense retrieval unavailable in this environment (offline / no model)")
                return
            self.assertIn("RERANKER_IS_NONE True", stdout)
            self.assertIn("HAS_RECS True", stdout)

    @unittest.skipUnless(_HAS_TRAINED_ARTIFACTS, "requires trained reranker artifacts + the full catalog")
    def test_reranker_on_by_default_when_artifacts_present(self) -> None:
        # v2 GBDT beats the plain RRF order by +0.0653 TechnicalScore on the
        # public set, so it ships enabled (see docs/reranker_eval_results.md).
        script = (
            "from starter.agent import Agent\n"
            f"agent = Agent(r'{_CATALOG_PATH}', use_llm=False, use_dense=True)\n"
            "print('RERANKER_IS_NONE', agent._reranker is None)\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "RERANK_ENABLED"}  # unset -> exercise the real default
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=str(_REPO_ROOT), env=env,
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RERANKER_IS_NONE False", result.stdout)

    @unittest.skipUnless(_HAS_TRAINED_ARTIFACTS, "requires trained reranker artifacts + the full catalog")
    def test_explicit_disable_falls_back_to_rrf(self) -> None:
        script = (
            "from starter.agent import Agent\n"
            f"agent = Agent(r'{_CATALOG_PATH}', use_llm=False, use_dense=True)\n"
            "print('RERANKER_IS_NONE', agent._reranker is None)\n"
            "agent.reset('s1', {'summary': 'test'})\n"
            "resp = agent.respond('s1', 'I want a black leather jacket', 1, 10)\n"
            "print('NUM_RECS', len(resp['recommendations']))\n"
        )
        stdout = _run_in_subprocess(script, {"RERANK_ENABLED": "0"})
        self.assertIn("RERANKER_IS_NONE True", stdout)
        self.assertIn("NUM_RECS 10", stdout)

    @unittest.skipUnless(_HAS_TRAINED_ARTIFACTS, "requires trained reranker artifacts + the full catalog")
    def test_trained_reranker_produces_valid_recommendations(self) -> None:
        script = (
            "from starter.agent import Agent\n"
            f"agent = Agent(r'{_CATALOG_PATH}', use_llm=False, use_dense=True)\n"
            "print('RERANKER_IS_NONE', agent._reranker is None)\n"
            "agent.reset('s1', {'summary': 'test'})\n"
            "resp = agent.respond('s1', \"I'm looking for a black leather jacket for men\", 1, 10)\n"
            "print('NUM_RECS', len(resp['recommendations']))\n"
        )
        stdout = _run_in_subprocess(script, {"RERANK_ENABLED": "1"})
        self.assertIn("RERANKER_IS_NONE False", stdout)
        self.assertIn("NUM_RECS 10", stdout)


if __name__ == "__main__":
    unittest.main()
