1. Hackathon problem statement, requirements, judging, constraints (context/hackathon.md)
Problem statement: Build a conversational e-commerce shopping agent (TikTok TechJam) that goes beyond static keyword search, using an Amazon Reviews 2023 "Clothing_Shoes_and_Jewelry" catalog. Four required pillars:

I. Core Architecture — Dual-track intent routing (Buying = strict-filter track vs. Browsing = diverse dense-retrieval track); pipeline = Multi-Route Retrieval → LLM Semantic Ranking (keyword + category + vector).
II. Dialog Strategy — A dynamic conversational state tracker handling incremental slot accumulation and abrupt intent override (slot erasure/rewrite); "Proactive Guidance" — cut off retrieval and ask a clarifying question when facing "Over-Generality" (candidate pool overload).
III. Self-Evolution — Dynamic Context Programming: personalized context distillation across session state + long-term profile, runtime workflow re-orchestration.
IV. Evaluation — Coverage (Hit Rate@K), Precision (MRR / Top-K), Efficiency (MTTC — Mean Turns to Conversion). Formula (from docs/): TechnicalScore = 0.50×HitRate@10 + 0.30×MRR + 0.20×Efficiency, Efficiency = clip((11-MTTC)/10, 0, 1).
Constraints:

Hard limit 10 turns/session (zero score if exceeded).
Catalog is read-only (no mutation, no mock ASINs).
No UI/UX required (backend/API only), no full LLM fine-tuning, no heavy external vector DBs (must run in-memory), text-only (no multimodal).
Inputs are pre-cleaned text (no typo/ASR handling needed).
No paid LLM required; organizer provides no API keys/credits — teams use/pay for their own external services and must disclose cost, latency, token usage, and must not commit secrets.
Data: 50,000 frozen products, 200 public labeled dev sessions, 800 private eval sessions (separate users/targets).
Deliverables: Devpost write-up, public GitHub repo with README (setup, reproduction, limitations, contributions), and a demo video (walkthrough acceptable for backend-only tracks).
2. Repo structure

techjam-conversational-search/
├── .gitignore
├── DATA_ATTRIBUTION.md
├── README.md                      # participant-facing project README
├── requirements.txt
├── context/
│   └── hackathon.md                # full challenge brief (read above)
├── data/
│   ├── README.md
│   ├── public_set.jsonl             # 200 labeled dev sessions (present)
│   └── catalog.jsonl                # NOT PRESENT — must be downloaded (gitignored)
├── docs/
│   ├── agent_api_contract.json      # JSON-schema contract for reset/respond
│   ├── baseline_results.json        # weak-BM25 reference scores
│   ├── competition_specification.md # fuller spec (session protocol, scenario mix, etc.)
│   ├── evaluation_config.json       # scoring config
│   └── submission_rules.md
├── evaluator/
│   ├── __init__.py
│   └── local_evaluator.py           # the simulator + scorer
├── starter/
│   ├── __init__.py
│   └── agent.py                     # the Agent implementation (1002 lines)
├── tests/
│   ├── __init__.py
│   └── test_evaluator.py            # 3 unit tests for the evaluator itself
└── tiktokEnv/                       # a local Python venv (numpy, torch installed; NOT gitignored — see note below)
Notable: README also references organizer/JUDGING_RUNBOOK.md, organizer/private_release_checklist.md, organizer/JUDGING_DAY_SOP.md and docs/audits/, docs/data_selection_audit.md, docs/participant_release_checklist.md, tests/test_5core_builder.py, tests/test_organizer_pipeline.py, secure/ — these are all listed in .gitignore as "organizer-only" and do not exist in this participant repo (expected; they're for the organizer's private copy, not something missing from your teammate's work).

3. Key implementation files
starter/agent.py (1002 lines) — the whole system
Architecture is explicitly commented in four "Day" layers:

Module-level constants/helpers (lines 1–380): tokenizer (_tokens), gender/synonym/material/color canonicalization maps, GENERIC stopword-like set (attributes too generic to gate a strict match), BM25 per-field weights (BM25_WEIGHTS), Gemini prompt templates (_ROUTER_SYSTEM, _CLARIFY_SYSTEM, _ROUTER_SCHEMA), .env loader (_load_dotenv), doc-row builders for the FTS5 index (_doc_row, _tags_from, _details_text), and _dense_doc_text for the embedding track.
class Agent (line 381 onward) — "Hybrid conversational retrieval agent" with 4 layers:
Day 1 — Sparse/BM25 retrieval: builds an in-memory SQLite FTS5 (sqlite3.connect(":memory:")) virtual table over title/category/features/details/store/description/tags, tiered multi-query construction (_build_queries) with internal reciprocal-rank fusion across tiers (_search), plus a deterministic heuristic parser (_ingest) that handles the simulator's templated phrasing ("I'm looking for X, a key requirement is Y", "actually...", "what matters is...", etc.) as an offline backstop when the LLM is unavailable.
Day 2 — LLM-powered state tracker (Gemini): _route() / _call_router() / _apply_route() call Google Gemini (GEMINI_MODEL env var, default gemini-3.5-flash-lite — note: commit message says "Gemini 3.5 Flash" but the actual default model string is gemini-3.5-flash-lite) via raw urllib REST calls (stdlib only, no SDK dependency) with a structured JSON schema (_ROUTER_SCHEMA) to classify intent (buying/browsing), extract SLOT_KEYS (category, gender, color, material, style, brand, use_case, budget), and detect intent_override. This is the "dynamic state machine" / "Information Accumulation vs Intent Override" pillar from the brief: on intent_override=True + new category, it wipes category/style/use_case slots and category/constraint term lists (lines 707–727) while preserving colour/material/brand/budget. Has a circuit breaker (_llm_broken, trips after 3 consecutive failures) and a response cache (_route_cache).
Day 3 — Proactive Guidance / Over-Generality handling: _clarify_slot() (lines 863–890) fires when a category is pinned but no discriminating constraint exists AND the match count for that category exceeds OVERGENERAL_MATCHES (default 1500); _clarify_question() generates a natural clarifying question (Gemini, with a deterministic fallback string) asking about the highest-priority missing slot (CLARIFY_PRIORITY). Important: this feature is disabled by default — CLARIFY_ENABLED = os.environ.get("CLARIFY_ENABLED", "0") — with an explicit code comment (lines 127–132) explaining why: on the public set, withholding a turn to ask a clarifying question costs ~0.06 TechnicalScore (0.840→0.778) versus just returning 10 results and asking "other". So the commit message's "implemented Proactive Guidance" feature exists in code but is turned off for scoring runs unless CLARIFY_ENABLED=1 is set.
Day 4 — Dense retrieval + hybrid RRF fusion: optional sentence-transformers embedder (all-MiniLM-L6-v2 by default), cosine similarity over cached doc vectors (data/dense_{hash}.npz), fused with the BM25 ranked list via textbook Reciprocal Rank Fusion (_rrf_fuse, score = Σ 1/(RRF_K+rank)), unconditionally run every turn when available (degrades gracefully to pure BM25 if numpy/sentence-transformers/model aren't available).
Public contract: reset(session_id, user_profile) and respond(session_id, user_message, turn, top_k) -> dict matching docs/agent_api_contract.json exactly (message/ask_attribute/recommendations/usage).
evaluator/local_evaluator.py (313 lines) — simulator + scorer (should not be edited per README/rules)
intent_card() / behavior_for(): derive a synthetic "intent card" (hard constraints, soft preferences) and scenario behavior (e.g. intent-override turn/message) from a catalog product when the public sessions file omits them.
catalog_index(): loads data/catalog.jsonl into id set / categories / full product dict.
initial_message() / customer_reply(): the deterministic simulated-customer policy — generates turn-1 messages per scenario type (buying/browsing/intent_override) and subsequent replies based on the agent's ask_attribute, disclosing one new constraint at a time; boundary scenario replies "I don't have a preference... please use your judgment" once.
evaluate(): runs the full session loop per sample (up to MAX_TURNS=10), calls agent.respond(), normalizes/scores recommendations, tracks hit, first_hit_turn, best_rank, reciprocal_rank, accumulates token usage, computes overall + per-scenario metrics and recommended_technical_score.
main(): CLI entrypoint (python3 -m evaluator.local_evaluator), writes results.json.
tests/test_evaluator.py — 3 unit tests
test_normalization_preserves_first_valid_unique_order, test_metric_summary_assigns_turn_11_to_miss, test_evaluate_derives_hidden_fields_when_public_set_omits_them. These test the evaluator, not the agent — there are currently no tests for starter/agent.py itself.

4. README / dependency / config files
README.md (repo root): project overview, catalog download instructions (gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl), run command (python3 -m evaluator.local_evaluator), Agent interface spec, metrics formulas, baseline scores (Hit@10 0.125, MRR 0.068034, MTTC 9.81), model-cost disclosure requirement. Contains a typo (see §6).
requirements.txt: comment-documented as "Day 1/3 = stdlib only; Day 2 (Gemini) = stdlib urllib, no package needed; Day 4 (dense) optional." Lists numpy>=1.26, sentence-transformers>=3.0, with a comment about installing CPU-only torch separately. No google-generativeai SDK dependency (calls Gemini via raw REST/urllib).
No package.json / pyproject.toml — pure pip/stdlib project.
docs/agent_api_contract.json: formal JSON Schema for reset_request, turn_request, turn_response (matches agent.py's actual return shape).
docs/evaluation_config.json: top_k=10, max_turns=10, miss_turn_value=11, scenario metrics list, efficiency formula, composite weights (0.5/0.3/0.2).
docs/baseline_results.json: weak-BM25 reference (Hit@10 0.125, MRR 0.068, MTTC 9.81, TechnicalScore 0.10671) — the bar to beat.
docs/submission_rules.md: what must be submitted, disallowed contents (no private eval data, no secrets, no evaluator edits), model policy (network may be disabled for official scoring — document offline fallback), recommended file layout.
docs/competition_specification.md: fuller spec — scenario mix is 40% Buying / 40% Browsing / 15% Intent Override / 5% Boundary; session protocol details; visible catalog fields (parent_asin, title, features, description, price, categories, details, average_rating, rating_number, store).
No .env or .env.example file exists anywhere in the repo (confirmed via glob) — .env is gitignored. This means GEMINI_API_KEY / GOOGLE_API_KEY are currently unset in this checkout; the agent's _gemini_api_key() will return None and the whole Day 2/Day 3 LLM layer will silently be inactive, falling back to the Day 1 deterministic parser. You will need to create a .env yourself (with GEMINI_API_KEY=...) to exercise the LLM router/clarification features.
.gitignore: excludes .env, results.json, data/catalog.jsonl(.gz), data/dense_*.npz, organizer/, secure/, plus several organizer-only doc/test files.
tiktokEnv/ is a full Python virtualenv checked into the working directory (not currently gitignored — worth checking git status/git ls-files before committing further, to avoid accidentally publishing a huge venv folder to the public submission repo).
5. Git log and mapping commit messages to code

2e039ef  (HEAD) "Modified Agent to Improve Hit10 and MTTC Score, Included a LLM-powered
                state tracker using Gemini 3.5 Flash, and implemented a 'Proactive
                Guidance' feature while handleing 'Over-Generality.'"
e917c37  "Yes"                                    — trivial/likely-accidental README edit
3407835  "Clarify TechnicalScore judging role"     — organizer commit
9a35be5  "Clarify participant model API costs"     — organizer commit
2a6cc8e  "Publish conversational search challenge" — organizer initial publish
2e039ef is the only substantive commit from your teammate (author akneehao, dated "2026-08-31" — matches the environment's current date). Diff stat: starter/agent.py: +971/-56 (essentially the entire hybrid system), plus requirements.txt (+13, added numpy/sentence-transformers) and .gitignore (+3, added dense-cache exclusion). Before this commit, agent.py was presumably the "weak BM25 starter" (~56-97 lines); this single commit built the whole 1002-line Day 1–4 architecture described above.
e917c37 ("Yes"): a one-line, essentially meaningless commit — it changed README.md line 12 from ...for final evaluation. to ...for final evaluation.iii, i.e. it appears to be an accidental stray keystroke committed with a placeholder message "Yes". This is a bug/typo still present in the current README (see §6).
Mapping requested features to code:
"LLM-powered state tracker using Gemini" → Agent._route() / _call_router() / _apply_route() (agent.py lines 620–739), using LLM_MODEL = "gemini-3.5-flash-lite" (default; commit message says "3.5 Flash" — the actual default is the lite variant, controllable via GEMINI_MODEL env var).
"Proactive Guidance" / "Over-Generality" → Agent._clarify_slot() / _clarify_question() (lines 851–916), gated by CLARIFY_ENABLED env var — off by default with an explicit rationale comment that it costs TechnicalScore on the public eval set.
"Improve Hit10 and MTTC" → the tiered BM25 query builder (_build_queries), RRF fusion (_search, _rrf_fuse), and the "stale/rotate" mechanism (state["stale"], rotate param) that cycles retrieval strategy when the constraint set stops changing so a buried target gets re-ranked each turn (lines 936–949, 796–802).
6. Incomplete / stubbed / broken / needs-attention items
data/catalog.jsonl is missing — only data/README.md and data/public_set.jsonl exist in data/. Per README/data/README.md instructions, you must download catalog.jsonl.gz from the GitHub Release (https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit), verify against SHA256SUMS, decompress, and place it at data/catalog.jsonl before the agent or evaluator can run at all (default catalog_path="data/catalog.jsonl" in Agent.__init__).
No .env file / no GEMINI_API_KEY configured — the Day 2 LLM router and Day 3 clarification-question generation are currently inert in this checkout (silent fallback to deterministic parsing / fallback question string). Not "broken" per se (graceful degradation is intentional), but it means the flashy "LLM-powered" and "Proactive Guidance" features have likely never actually been exercised locally unless your teammate had a .env on their own machine that wasn't committed (correctly, since it's gitignored — but also means there's no record of it having worked).
sentence-transformers (and transformers) not installed in the checked-in tiktokEnv venv, even though torch and numpy are present. requirements.txt lists sentence-transformers>=3.0 as a dependency but pip install was apparently not completed for it, so the Day 4 dense-retrieval track (_init_dense) will silently no-op (_SentenceTransformer = None → _embedder = None → pure-BM25 fallback) in this environment as it currently stands. Need pip install sentence-transformers (plus CPU-only torch per the requirements.txt comment, though torch is already present) to actually activate it.
No data/dense_*.npz cache present — consistent with #3; dense embeddings have never been built/cached here.
CLARIFY_ENABLED defaults to "0" (off) — the "Proactive Guidance" pillar exists but is explicitly disabled by default per the author's own comment because it costs ~0.06 TechnicalScore on the public set as currently tuned. This is a deliberate trade-off, not a bug, but worth knowing before claiming it's "implemented and active" in a Devpost writeup — you'd want to either re-tune it to be a net positive or explicitly discuss it as an optional/demo-only feature in the report (the write-up requirement explicitly allows describing limitations).
README.md typo from the "Yes" commit: line 12 currently reads The organizer keeps 800 additional sessions private for final evaluation.iii — stray "iii" appended, should be cleaned up before submission since the README is a graded deliverable.
No results.json in the repo (correctly gitignored) — meaning there is no evidence anyone has run the evaluator end-to-end against the new Day 2–4 agent and the real catalog; the only recorded scores are the old weak-BM25 baseline in docs/baseline_results.json. Given item #1 (catalog missing), it's likely the new agent has never been run end-to-end in this environment.
No tests at all for starter/agent.py — tests/ only covers the evaluator (test_evaluator.py). None of the tokenizer, query-builder, slot-extraction, RRF fusion, or clarification logic in agent.py has unit test coverage.
Model name mismatch: commit message says "Gemini 3.5 Flash" but code default is gemini-3.5-flash-lite (env-overridable via GEMINI_MODEL) — minor discrepancy worth confirming/aligning for the Devpost "APIs used" disclosure.
tiktokEnv/ venv folder is present in the working tree and (per a quick look at .gitignore) is not explicitly excluded — worth checking git status/git ls-files before your next commit to avoid bloating the public submission repo with a full virtualenv.
7. Tests / evaluation scripts and tracked metrics
Evaluation script: evaluator/local_evaluator.py, run via python3 -m evaluator.local_evaluator [--catalog PATH] [--dataset PATH] [--output results.json]. It is the official local scorer and must not be edited (per rules).
Metrics tracked (matches the hackathon brief's "Evaluation Matrix"):
hit_rate_at_10 — fraction of the 200 public sessions where the target parent_asin appears in the top-10 within 10 turns.
mrr — mean reciprocal rank (0 for misses).
mttc — mean first-hit turn (miss = 11).
efficiency — clip((11-mttc)/10, 0, 1).
recommended_technical_score — the 0.50/0.30/0.20 weighted composite.
reported_token_usage — summed prompt/completion tokens from the agent's usage field.
scenario_metrics — the same metrics broken out per scenario type (buying, browsing, intent_override, boundary).
Per-session detail array (sessions) with hit, first_hit_turn, best_rank, reciprocal_rank.
Unit tests: tests/test_evaluator.py (3 tests, evaluator-only, described in §3). Run via python -m unittest / pytest (both should discover it; no pytest-specific config seen — no pytest.ini/pyproject.toml).
Reference numbers: docs/baseline_results.json gives the weak-BM25 starter's public-set scores (Hit@10 0.125, MRR 0.068034, MTTC 9.81, Efficiency 0.119, TechnicalScore 0.10671) as the baseline your teammate's 2e039ef commit was presumably trying to beat — but as noted, there's no committed results.json proving the new agent's actual score, and it likely can't even run yet in this checkout without the catalog download.
Suggested immediate next steps (read-only observation, no action taken)
Download/verify/place data/catalog.jsonl per data/README.md.
Create a .env with GEMINI_API_KEY (or GOOGLE_API_KEY) if you want to exercise the LLM router/clarification path; otherwise the agent runs in pure-BM25 fallback mode.
pip install sentence-transformers in tiktokEnv if you want the Day 4 dense track active.
Run python -m evaluator.local_evaluator to get a real, current results.json baseline for the new agent before making further changes.
Fix the stray README typo from the "Yes" commit.
Decide/document whether to flip CLARIFY_ENABLED=1 given its measured score trade-off, for the Devpost write-up.
The survey's done. Here's the picture of what you're inheriting:

The challenge
Build a conversational e-commerce shopping agent for TikTok's Amazon Clothing/Shoes/Jewelry catalog. Four required pillars: dual-track intent routing (buying vs. browsing), a dynamic dialog state tracker with proactive clarifying questions, self-evolving context across sessions, and evaluation via TechnicalScore = 0.50×HitRate@10 + 0.30×MRR + 0.20×Efficiency (efficiency penalizes taking many turns, capped at 10 turns/session). No paid LLM required, no external vector DBs, in-memory only.

What's implemented (starter/agent.py, 1002 lines — all from one commit 2e039ef by your teammate)
A layered hybrid retrieval agent:

Day 1 — Sparse retrieval: in-memory SQLite FTS5 index, tiered query building with reciprocal-rank fusion, plus a deterministic heuristic parser as an offline backstop.
Day 2 — LLM state tracker: calls Gemini via raw urllib (no SDK) to classify intent, extract slots (category/gender/color/material/style/brand/use_case/budget), and detect intent overrides — this is the "dynamic state machine" pillar. Has a circuit breaker and response cache.
Day 3 — Proactive Guidance: generates clarifying questions when a category is under-constrained and the match pool is too large. This is implemented but disabled by default (CLARIFY_ENABLED=0) — there's an explicit code comment saying it costs ~0.06 TechnicalScore on the public set as currently tuned.
Day 4 — Dense retrieval: optional sentence-transformers embeddings fused with BM25 via RRF, degrades gracefully if the library isn't installed.
Gaps to close before this can run or be judged fairly
data/catalog.jsonl is missing — needs to be downloaded from the release and placed in data/. Without it, nothing runs.
No .env/GEMINI_API_KEY — the LLM router and clarification features have likely never actually been exercised; the agent silently falls back to deterministic parsing.
sentence-transformers isn't installed in the venv (tiktokEnv/) despite being in requirements.txt — the dense track is currently a no-op.
No results.json — no evidence the new agent has been scored end-to-end; only the old weak-BM25 baseline (Hit@10 0.125, TechnicalScore 0.107) is on record.
No tests for agent.py — only the evaluator itself has tests.
Minor: a stray typo in README.md from a trivial "Yes" commit, and a model-name mismatch (commit says "Gemini 3.5 Flash", code defaults to gemini-3.5-flash-lite).
tiktokEnv/ (the venv) is sitting in the working tree and doesn't look gitignored — worth checking before your next commit so you don't publish a full venv to the public submission repo.
Suggested first move: get the catalog downloaded and run the evaluator once to see where the current agent actually stands, since right now there's no verified score for any of this work.