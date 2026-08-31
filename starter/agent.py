from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

try:  # optional -- Day 4 dense track; the BM25 core never imports these
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:  # pragma: no cover
    _SentenceTransformer = None
try:  # optional -- Day 5 learned re-ranking layer
    from .reranker import load_reranker
    from .reranker.catalog_index import build_category_index
    from .reranker.features import compute_feature_matrix
except ImportError:  # pragma: no cover
    load_reranker = None
    build_category_index = None
    compute_feature_matrix = None


# --------------------------------------------------------------------------- #
# Tokenisation / normalisation
# --------------------------------------------------------------------------- #
#
# Every document field AND every query string is pushed through ``_tokens`` so
# the two sides always agree.  FTS5's own ``porter`` stemmer then runs on top of
# that (plurals / verb forms), so we only have to normalise what Porter cannot:
# irregular plurals and department / gender vocabulary.

TOKEN_RE = re.compile(r"[a-z0-9]+")
# insert a break between letter<->digit runs so "100%Cotton" -> "100 cotton"
_ALNUM_BREAK_RE = re.compile(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])")
_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "im", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "your", "looking",
    # simulator function words (safe to drop, never product attributes)
    "still", "exploring", "requirement", "requirements", "preference", "preferences",
    "actually", "ignore", "earlier", "matters", "additional", "judgment", "need",
    "key", "item", "date", "available", "quite", "right", "yet", "tell", "about",
    "one", "specific", "attribute", "closure", "instead", "they", "them", "their",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

# raw token -> canonical token (applied to both index and query text)
GENDER_MAP = {
    "mens": "men", "men": "men", "man": "men", "mans": "men", "male": "men",
    "males": "men", "guy": "men", "guys": "men", "gentleman": "men",
    "gentlemen": "men", "boyfriend": "men", "husband": "men", "dad": "men",
    "womens": "women", "women": "women", "woman": "women", "womans": "women",
    "female": "women", "females": "women", "lady": "women", "ladies": "women",
    "gal": "women", "gals": "women", "girlfriend": "women", "wife": "women",
    "mom": "women",
    "kids": "kids", "kid": "kids", "child": "kids", "children": "kids",
    "childrens": "kids", "childs": "kids", "toddler": "kids", "toddlers": "kids",
    "junior": "kids", "juniors": "kids", "youth": "kids",
    "girl": "girls", "girls": "girls", "boy": "boys", "boys": "boys",
    "baby": "baby", "infant": "baby", "infants": "baby", "newborn": "baby",
    "unisex": "unisex",
}
SYN_MAP = {"grey": "gray", "pjs": "pajamas", "pj": "pajamas", "hoody": "hoodie"}

MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "linen", "denim", "cashmere", "satin", "velvet", "suede", "mesh", "fleece",
    "acrylic", "modal", "viscose", "bamboo", "canvas", "chiffon", "lace", "jersey",
    "elastane", "lycra", "microfiber", "faux", "sherpa", "corduroy", "flannel",
    "tweed", "neoprene", "terry", "alloy", "sterling", "brass", "platinum",
    "titanium", "rhinestone", "crystal", "pearl", "rubber", "latex", "eva",
    "plastic", "metal", "wood", "ceramic", "fabric", "knit", "cork",
}
COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "tan", "gold", "silver",
    "ivory", "cream", "maroon", "burgundy", "teal", "olive", "khaki", "coral",
    "turquoise", "lavender", "charcoal", "rose", "multicolor",
}
# generic constraint tokens that must NOT drive a strict AND match
GENERIC = {
    "imported", "closure", "machine", "wash", "hand", "only", "quality", "high",
    "made", "usa", "soft", "comfortable", "comfort", "day", "long", "keep", "cool",
    "dry", "fit", "perfect", "adjustable", "great", "care", "pull", "snap", "zipper",
    "button", "drawstring", "buckle", "lightweight", "durable", "durability", "style",
    "material", "department", "brand", "manufacturer", "number", "model", "product",
    "package", "dimensions", "measures", "approximately", "features", "featuring",
    "new", "favorite", "set", "pack", "size", "color", "colors",
}
GENDER_CANON = ("men", "women", "kids", "girls", "boys", "baby", "unisex")

# tokens that must never appear in a query at all (pure noise)
_CAT_DROP = {"clothing", "item", "more"}
# "Clothing, Shoes & Jewelry" is every sparse item's root; comma-splitting leaks
# "shoes"/"jewelry" into the stated category. Fine as OR terms, too weak to gate.
_CAT_JUNK = {"shoes", "jewelry"}

DETAIL_TEXT_KEYS = frozenset({
    "department", "manufacturer", "brand", "brand name", "material", "fabric type",
    "style", "color", "closure type", "sole material", "outer material", "pattern",
    "shape", "occasion", "sport", "sport type", "fit type", "neck style",
    "age range (description)", "theme", "special feature", "item model number",
    "model name", "part number", "country of origin",
})

# column order: parent_asin, title, category, features, details, store, description, tags
# `category` leads: the shopper's stated category is the most reliable signal and
# always present from turn 1.  `tags` (normalised keyword bag) and `title` next.
BM25_WEIGHTS = (0.0, 8.0, 11.0, 5.0, 3.0, 1.0, 1.5, 8.0)

# --------------------------------------------------------------------------- #
# LLM layer -- Gemini intent router (Day 2) + proactive clarification (Day 3)
# --------------------------------------------------------------------------- #
# Live LLM = Google Gemini via the REST API (stdlib urllib, no extra dependency).
# The key is read from GEMINI_API_KEY / GOOGLE_API_KEY (see .env).  When no key /
# no network is available the agent falls back to the deterministic parser -- the
# submission rules allow network to be disabled during official scoring.
LLM_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
LLM_MAX_TOKENS = 512
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_LLM_TIMEOUT = 20.0

SLOT_KEYS = ("category", "gender", "color", "material", "style", "brand", "use_case", "budget")

# Day 3: when the query is this broad and the shopper has given no discriminating
# constraint, ask a clarifying question instead of returning 10 near-random hits.
# Default OFF: the local evaluator's shopper is always cooperative, so withholding
# a turn's results to ask a *targeted* question costs ~0.06 TechnicalScore
# (0.840 -> 0.778 on the public set) vs. "show 10 + ask 'other'".  It is real,
# useful conversational UX -- set CLARIFY_ENABLED=1 for demos / production.
OVERGENERAL_MATCHES = int(os.environ.get("OVERGENERAL_MATCHES", "1500"))
CLARIFY_MAX_TURN = int(os.environ.get("CLARIFY_MAX_TURN", "1"))
CLARIFY_ENABLED = os.environ.get("CLARIFY_ENABLED", "0") not in ("0", "false", "False")
# most-important-missing-slot order; every value is a valid `ask_attribute`
CLARIFY_PRIORITY = ("category", "color", "material", "style", "use_case", "budget")

# --------------------------------------------------------------------------- #
# Day 4 -- dense retrieval track (sentence-transformers) + hybrid RRF fusion
# --------------------------------------------------------------------------- #
# Optional: needs `numpy` + `sentence-transformers`.  Without them (or with
# DENSE_ENABLED=0) the agent is pure BM25 and scores exactly as Day 1-3.
#
# Fusion is UNCONDITIONAL: every turn runs both tracks and combines them with
# textbook Reciprocal Rank Fusion -- take the top RRF_DEPTH from each list and
# add 1 / (RRF_K + rank) per appearance, equal weight.  (Measured cost vs. the
# pure-BM25 0.840 baseline: see the Day 4 notes / eval table.)
DENSE_ENABLED = os.environ.get("DENSE_ENABLED", "1") not in ("0", "false", "False")
DENSE_MODEL = os.environ.get("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DENSE_CACHE_DIR = Path(os.environ.get("DENSE_CACHE_DIR", "data"))
_DENSE_TEXT_VERSION = 2          # bump to invalidate cached vectors when doc text changes
DENSE_ENCODE_BATCH = 256
RRF_DEPTH = int(os.environ.get("RRF_DEPTH", "60"))   # top-N pulled from each track
RRF_K = int(os.environ.get("RRF_K", "60"))           # the "60" in 1 / (60 + rank)

# --------------------------------------------------------------------------- #
# Day 5 -- learned re-ranking layer (optional)
# --------------------------------------------------------------------------- #
# Re-scores the top RERANK_CANDIDATES of the RRF-fused list with a small,
# locally-trained ranker (see starter/reranker/ + training/). Fully optional:
# with RERANK_ENABLED=0, missing artifacts, or a missing dependency, the
# agent falls straight back to the plain RRF order -- identical to Day 4
# behaviour. Never trains or calls any external service at serving time.
#
# ENABLED BY DEFAULT (simplex). On the public set this is worth +0.0803
# TechnicalScore over the plain RRF ordering -- 0.8410 vs. 0.7607 -- and it
# improves every component metric: HitRate@10 0.960 vs 0.940, MRR 0.649 vs
# 0.444, MTTC 2.68 vs 3.12.  Other trained models, same evaluator run:
# mlp 0.8362, gbdt 0.8260, coord_ascent 0.8244, ranksvm 0.8181.  Select any
# of them with RERANK_MODEL=<name>; the margins between the top three are
# small relative to a 200-session sample, so treat that ordering as
# provisional (see docs/reranker_eval_results.md).
#
# Simplex is also the cheapest to serve: the artifact is an 11-number JSON
# weight vector scored with one numpy dot product, so the default path needs
# no lightgbm (nor torch/scipy/sklearn) at inference time.
#
# All of this holds only for the v2 training formulation. An earlier v1
# (labels built from category siblings, no first-stage retrieval features)
# scored BELOW the plain RRF order for every model type -- best 0.7006,
# worst 0.3344, and simplex was that worst case. See
# docs/reranker_eval_results.md for the root cause (label leakage into
# sibling_max_sim, plus a training task that was nearly the opposite of the
# real one) and what changed.
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1") not in ("0", "false", "False")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "simplex")
RERANK_ARTIFACTS_DIR = Path(
    os.environ.get("RERANK_ARTIFACTS_DIR", str(Path(__file__).resolve().parent / "reranker" / "artifacts"))
)
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "60"))

# -- Day 5b: personalized context distillation ----------------------------- #
# The aggregate ``user_profile`` is distilled once per session in ``reset()``.
# PROFILE_INJECT decides whether those distilled tags may reach *retrieval*:
#   0 (default) -- never. The profile only conditions how the router reads a
#                  message; it cannot add or remove candidates.
#   1           -- inject only tags outside GENERIC (performance/warmth/...).
#   2           -- inject every tag, i.e. the literal "append the profile to
#                  the search constraints" reading.
# Measured on the 200 public sessions, GENERIC already covers 91% of all tag
# occurrences (fit 82%, material 77%, comfort 72%, style 50%, durability 24%),
# so level 2 mostly feeds high-frequency noise to the query builder. Levels 1
# and 2 exist to make that claim measurable rather than assumed.
PROFILE_INJECT = int(os.environ.get("PROFILE_INJECT", "0") or 0)


def _distil_profile(user_profile: dict) -> dict:
    """Reduce the aggregate profile to the few fields worth carrying per session.

    ``salient_tags`` drops anything in GENERIC plus multi-word tags: those are
    too common in clothing text to discriminate between candidates, which is
    exactly why they must never be allowed to gate a match.
    """
    empty = {"tags": [], "salient_tags": [], "summary": "", "rating_style": ""}
    if not isinstance(user_profile, dict):
        return empty

    tags: list[str] = []
    for tag in user_profile.get("preference_tags") or ():
        if not isinstance(tag, str):
            continue
        cleaned = tag.strip().lower()
        if cleaned and cleaned not in tags:
            tags.append(cleaned)

    def _text(key: str) -> str:
        value = user_profile.get(key)
        return value.strip() if isinstance(value, str) else ""

    return {
        "tags": tags,
        "salient_tags": [t for t in tags if t not in GENERIC and " " not in t],
        "summary": _text("summary"),
        "rating_style": _text("rating_style"),
    }


# messages that carry no new constraint -- never worth an LLM call
_NO_INFO_MARKERS = (
    "additional preference", "please use your judgment", "please use your judgement",
    "not quite right yet",
)

_ROUTER_SYSTEM = (
    "You are the intent router for a Clothing/Shoes/Jewelry shopping search agent.\n"
    "Given the conversation state and the newest customer message, return the "
    "routing object. The response schema is enforced by the API, so do not "
    "restate or explain it.\n"
    "Rules:\n"
    "- intent=\"buying\" when the customer gives firm, specific requirements; "
    "\"browsing\" when vague, exploring, or only a broad category is known.\n"
    "- Fill a slot only with something the customer actually says in THIS message "
    "(new or explicitly restated). Use null otherwise. Lowercase all values.\n"
    "- gender: men / women / kids / girls / boys / baby / unisex, only if stated.\n"
    "- keywords: other salient constraint words not covered by a slot (max 6).\n"
    "- intent_override=true ONLY when the customer abandons the previously stated "
    "product CATEGORY for a different one (\"actually I want a dress instead\"). "
    "A changed colour/material/size/fit/preference is NOT an override.\n"
    "- shopper_prior_aspects describes what the shopper's PAST purchases "
    "emphasised. It is background, NOT a request: never copy it into a slot or "
    "keyword. Use it only to disambiguate wording used in THIS message.\n"
    "Output JSON only, no prose."
)

_CLARIFY_SYSTEM = (
    "You are a friendly shopping assistant. The catalog search for the shopper's "
    "request is too broad to show good results yet. Write ONE short, warm question "
    "(<=25 words) asking them to narrow it down by the given attribute. Mention how "
    "many items matched if given. Plain text only, no lists, no quotes."
)

# Gemini structured-output schema (OpenAPI subset -- uses `nullable`, not JSON-Schema unions)
_ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["buying", "browsing"]},
        "extracted_slots": {
            "type": "object",
            "properties": {
                **{k: {"type": "string", "nullable": True} for k in SLOT_KEYS},
                "keywords": {"type": "array", "items": {"type": "string"}},
            },
            "required": [*SLOT_KEYS, "keywords"],
        },
        "intent_override": {"type": "boolean"},
    },
    "required": ["intent", "extracted_slots", "intent_override"],
}


def _load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (stdlib only) -- fills os.environ, never overrides it."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # repo-root .env too


def _gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


_CREATE_PORTER = (
    "CREATE VIRTUAL TABLE products USING fts5("
    "parent_asin UNINDEXED, title, category, features, details, store, description, tags, "
    "tokenize='porter unicode61 remove_diacritics 2')"
)
_CREATE_PLAIN = _CREATE_PORTER.replace("'porter unicode61", "'unicode61")

_BOILERPLATE_RE = re.compile(
    r"i'?m looking for|looking for|but i'?m still exploring|a key requirement is|"
    r"actually|ignore my earlier preference|what i need is|for that|what matters is|"
    r"please use your judgment|i don'?t have an additional preference for|"
    r"i don'?t have a preference for|those options are not quite right yet|"
    r"ask me about one specific attribute"
)


def _tokens(text: str) -> list[str]:
    low = text.lower()
    if any(ch.isdigit() for ch in low):
        low = _ALNUM_BREAK_RE.sub(" ", low)  # "100%Cotton" -> "100 cotton"
    out: list[str] = []
    for raw in TOKEN_RE.findall(low):
        if len(raw) < 2 or _YEAR_RE.match(raw):
            continue
        term = GENDER_MAP.get(raw) or SYN_MAP.get(raw, raw)
        if term not in STOPWORDS:
            out.append(term)
    return out


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _flatten(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key} {item}" for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _details_text(details: object) -> str:
    if not isinstance(details, dict):
        return ""
    parts: list[str] = []
    for key, value in details.items():
        if str(key).strip().lower() in DETAIL_TEXT_KEYS:
            parts.append(f"{key} {value}")
    return " ".join(parts)


def _tags_from(title_t: list[str], cat_entries: list[str], feat_t: list[str],
               det_t: list[str], store_t: list[str], model_t: list[str]) -> str:
    """A deduped, normalised keyword bag: the 'maximum vocabulary' merge.

    Kept as its own column (rather than a giant duplicated ``all_text``) so the
    per-field BM25 weights above stay meaningful instead of double counting.
    Built from token lists already computed for the other columns.
    """
    blob = set(title_t) | set(feat_t) | set(det_t)
    tags: list[str] = [g for g in GENDER_CANON if g in blob]
    tags += [m for m in MATERIALS if m in blob]
    tags += [c for c in COLORS if c in blob]
    for entry in cat_entries[-3:]:
        tags += _tokens(entry)
    tags += store_t[:4]
    tags += model_t
    return " ".join(_dedupe(tags))


def _doc_row(product: dict) -> tuple[str, str, str, str, str, str, str, str]:
    cat_entries = [str(c) for c in (product.get("categories") or [])]
    details = product.get("details") if isinstance(product.get("details"), dict) else {}
    title_t = _tokens(str(product.get("title") or ""))
    cat_t = _tokens(" ".join(cat_entries))
    feat_t = _tokens(" ".join(_flatten(product.get("features"))))
    det_t = _tokens(_details_text(details))
    store_t = _tokens(str(product.get("store") or ""))
    desc_t = _tokens(" ".join(_flatten(product.get("description"))))
    model_t: list[str] = []
    for key in ("Item model number", "Model Name", "Part Number", "Manufacturer", "Brand"):
        if key in details:
            model_t += _tokens(str(details[key]))[:4]
    return (
        str(product.get("parent_asin") or ""),
        " ".join(title_t),
        " ".join(cat_t),
        " ".join(feat_t),
        " ".join(det_t),
        " ".join(store_t),
        " ".join(desc_t),
        _tags_from(title_t, cat_entries, feat_t, det_t, store_t, model_t),
    )


def _dense_doc_text(product: dict) -> str:
    """Natural-language document text for the dense encoder: title + categories +
    features (raw, not tokenised -- transformers want real language)."""
    title = str(product.get("title") or "").strip()
    categories = ", ".join(str(c) for c in (product.get("categories") or []))
    features = " ".join(_flatten(product.get("features")))
    return re.sub(r"\s+", " ", f"{title}. {categories}. {features}").strip()[:800]


def _split_first_message(low: str) -> tuple[str, str]:
    """Parse the templated turn-1 message into (category_text, extra_constraint)."""
    anchor = low
    for lead in ("i'm looking for ", "im looking for ", "looking for "):
        if lead in low:
            anchor = low.split(lead, 1)[1]
            break
    for expl in (", but i'm still exploring", ", but im still exploring", "but i'm still exploring"):
        if expl in anchor:
            return anchor.split(expl, 1)[0].strip(" .,"), ""
    if ". a key requirement is" in anchor:
        left, right = anchor.split(". a key requirement is", 1)
        return left.strip(" .,"), right.strip(" :.\"'")
    if ". " in anchor:  # intent_override: "{category}. {old_value}"
        left, right = anchor.split(". ", 1)
        return left.strip(" .,"), right.strip(" .")
    return anchor.strip(" .,"), ""


def _parse_router_json(text: str) -> dict:
    """Parse the router's reply, tolerating stray prose / code fences around the JSON."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class Agent:
    """Hybrid conversational retrieval agent.

    Layers:
      * Day 1 -- self-contained FTS5 / BM25 sparse retrieval with tiered RRF.
      * Day 2 -- a Google Gemini intent router (REST, stdlib only): Buying vs
        Browsing, constraint-slot extraction, category-override detection.
      * Day 3 -- proactive clarification when the query is far too broad.
      * Day 4 -- a dense retrieval track (sentence-transformers, cached in-memory
        vectors).  Every turn runs BM25 and dense; the two ranked lists are
        combined by unconditional Reciprocal Rank Fusion.
      * Day 5 -- an optional learned re-ranking layer (see starter/reranker/)
        that re-scores the top of the RRF-fused list with a small locally
        trained ranker.  Falls back to the plain Day 4 order if disabled,
        untrained, or missing a dependency.

    Every LLM / dense / re-ranking component is optional.  With no Gemini key
    and no numpy/sentence-transformers (the offline final-scoring case the
    rules allow) the agent is pure BM25 and behaves exactly as Day 1-3,
    ``usage`` all zeros.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        use_llm: bool | None = None,
        model: str = LLM_MODEL,
        use_dense: bool | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._state: dict[str, dict] = {}
        self._order_by = "bm25(products, " + ", ".join(str(w) for w in BM25_WEIGHTS) + ")"
        self._llm_model = model
        self._llm_key = self._init_llm(use_llm)
        self._llm_broken = False           # circuit breaker (permanent once tripped)
        self._llm_fail_streak = 0          # consecutive failures; 3 -> trip the breaker
        self._route_cache: dict[tuple, dict] = {}
        self.llm_usage_total = {"prompt_tokens": 0, "completion_tokens": 0}  # for disclosure
        # dense track (Day 4)
        self._embedder = None
        self._doc_vecs = None              # (N, dim) float32, L2-normalised
        self._dense_ids: list[str] = []
        self._catalog: dict[str, dict] = {}   # asin -> raw product dict, filled by _build_index
        self._build_index()
        self._init_dense(use_dense)
        # re-ranking layer (Day 5)
        self._cat_index = None
        self._dense_id_row: dict[str, int] = {}
        self._doc_tokens: dict[str, frozenset[str]] = {}
        self._reranker = None
        self._init_reranker()

    def _llm_on(self) -> bool:
        """True when the Gemini router is configured and hasn't hard-failed."""
        return bool(self._llm_key) and not self._llm_broken

    def _note_llm_failure(self) -> None:
        self._llm_fail_streak += 1
        if self._llm_fail_streak >= 3:  # bad key / offline / quota -> stop trying this run
            self._llm_broken = True

    @staticmethod
    def _init_llm(use_llm: bool | None) -> str | None:
        if use_llm is None:
            use_llm = os.environ.get("AGENT_USE_LLM", "1") not in ("0", "false", "False")
        return _gemini_api_key() if use_llm else None

    # -- indexing ---------------------------------------------------------- #
    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(_CREATE_PORTER)
        except sqlite3.OperationalError:
            cursor.execute(_CREATE_PLAIN)
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    product = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = str(product.get("parent_asin") or "")
                if asin:
                    self._catalog[asin] = product
                batch.append(_doc_row(product))
                if len(batch) >= 2000:
                    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", batch)
        self.connection.commit()

    # -- Day 4: dense retrieval track -------------------------------------- #
    def _dense_on(self) -> bool:
        return self._embedder is not None and self._doc_vecs is not None

    def _catalog_signature(self) -> str:
        try:
            stat = self.catalog_path.stat()
            raw = f"{self.catalog_path.name}:{stat.st_size}:{int(stat.st_mtime)}"
        except OSError:
            raw = str(self.catalog_path)
        raw += f":{DENSE_MODEL}:v{_DENSE_TEXT_VERSION}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _init_dense(self, use_dense: bool | None) -> None:
        if use_dense is None:
            use_dense = DENSE_ENABLED
        if not use_dense or _np is None or _SentenceTransformer is None:
            return
        try:
            self._embedder = _SentenceTransformer(DENSE_MODEL)
        except Exception:            # no local model + no network -> stay pure BM25
            self._embedder = None
            return

        cache = DENSE_CACHE_DIR / f"dense_{self._catalog_signature()}.npz"
        if cache.is_file():
            try:
                blob = _np.load(cache, allow_pickle=True)
                self._dense_ids = list(blob["ids"])
                self._doc_vecs = blob["vecs"].astype("float32")
                return
            except Exception:
                pass  # corrupt cache -> rebuild

        ids: list[str] = []
        texts: list[str] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    product = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ids.append(str(product.get("parent_asin") or ""))
                texts.append(_dense_doc_text(product))
        try:
            vecs = self._embedder.encode(
                texts, batch_size=DENSE_ENCODE_BATCH, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ).astype("float32")
        except Exception:
            self._embedder = None
            return
        self._dense_ids, self._doc_vecs = ids, vecs
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            _np.savez(cache, ids=_np.array(ids, dtype=object), vecs=vecs)
        except OSError:
            pass  # read-only fs -> keep the in-memory vectors, just don't cache

    # -- Day 5: learned re-ranking layer ------------------------------------ #
    def _init_reranker(self) -> None:
        """Best-effort setup of the category index + trained re-ranker.  Any
        missing dependency, missing artifact, or unexpected error leaves
        self._reranker as None -- respond() then behaves exactly like Day 4."""
        if not RERANK_ENABLED or load_reranker is None or not self._dense_on():
            return
        try:
            self._dense_id_row = {asin: i for i, asin in enumerate(self._dense_ids)}
            self._cat_index = build_category_index(self._catalog, self._doc_vecs, self._dense_ids)
            self._doc_tokens = {
                asin: frozenset(_tokens(_dense_doc_text(product)))
                for asin, product in self._catalog.items()
                if asin in self._dense_id_row
            }
            self._reranker = load_reranker(RERANK_ARTIFACTS_DIR, model=RERANK_MODEL)
        except Exception:
            self._cat_index = None
            self._doc_tokens = {}
            self._reranker = None

    def _rerank(
        self, candidates: list[str], query_text: str,
        bm25_ranked: list[str] | None = None, dense_ranked: list[str] | None = None,
    ) -> list[str]:
        if not candidates or self._reranker is None or self._cat_index is None:
            return candidates
        try:
            raw = self._embedder.encode(
                [query_text], convert_to_numpy=True, normalize_embeddings=False,
            )[0].astype("float32")
            q_norm = float(_np.linalg.norm(raw)) or 1.0
            qv = (raw / q_norm).astype("float32")
            X = compute_feature_matrix(
                qv, q_norm, _tokens(query_text), candidates,
                self._cat_index, self._doc_vecs, self._dense_id_row, self._doc_tokens,
                bm25_ranked=bm25_ranked, dense_ranked=dense_ranked, rrf_k=RRF_K,
            )
            return self._reranker.rank(X, candidates)
        except Exception:
            return candidates

    def _dense_rank(self, query_text: str, top_n: int) -> list[str]:
        if not self._dense_on() or not query_text.strip():
            return []
        try:
            qv = self._embedder.encode(
                [query_text], convert_to_numpy=True, normalize_embeddings=True,
            )[0].astype("float32")
        except Exception:
            return []
        sims = self._doc_vecs @ qv                      # cosine (both L2-normalised)
        top_n = min(top_n, sims.shape[0])
        part = _np.argpartition(-sims, top_n - 1)[:top_n]
        order = part[_np.argsort(-sims[part])]
        return [self._dense_ids[i] for i in order]

    @staticmethod
    def _dense_query_text(state: dict, user_message: str) -> str:
        """The 'vibe' string to embed: recent shopper wording + structured slots."""
        parts: list[str] = []
        for message in state["history"][-3:]:
            cleaned = _BOILERPLATE_RE.sub(" ", message.lower())
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:")
            if cleaned:
                parts.append(cleaned)
        parts.extend(v for v in state["slots"].values() if v)
        parts.extend(state["keywords"])
        seen: set[str] = set()
        uniq = [p for p in parts if not (p in seen or seen.add(p))]
        return " ; ".join(uniq)[:400] or user_message.strip()

    # -- session lifecycle ---------------------------------------------------- #
    @staticmethod
    def _new_state() -> dict:
        return {
            # ---- conversational memory (Day 2) --------------------------------
            "intent": "browsing",                     # "buying" | "browsing"
            "slots": {key: None for key in SLOT_KEYS},  # color / category / style / ...
            "keywords": [],                            # freeform constraint words
            "history": [],                             # raw customer messages
            # ---- heuristic backstop (Day 1) ---------------------------------
            "seen_first": False,
            "category_terms": [],
            "constraint_terms": [],
            "exhausted": False,
            "stale": 0,
            "last_signature": None,
            # ---- distilled long-term profile (Day 5b) -----------------------
            "profile": {"tags": [], "salient_tags": [], "summary": "", "rating_style": ""},
        }

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Fresh conversational memory for the session (intent, constraint slots,
        # keyword bag, message history -- see ``_new_state``).
        state = self._new_state()
        # Personalized context distillation: reduce the aggregate profile once,
        # here, then let it *interpret* later turns.  It stays out of sparse
        # retrieval by default -- its preference tags ("fit", "comfort", ...)
        # are high-frequency noise, 91% of them already in GENERIC -- unless
        # PROFILE_INJECT says otherwise (see the constant for the levels).
        state["profile"] = _distil_profile(user_profile)
        if PROFILE_INJECT:
            key = "tags" if PROFILE_INJECT >= 2 else "salient_tags"
            state["keywords"] = list(state["profile"][key])
        self._state[session_id] = state

    # -- per-turn ingestion of the simulated customer's message -------------- #
    def _ingest(self, state: dict, message: str, turn: int) -> None:
        low = message.strip().lower()

        if not state["seen_first"]:
            state["seen_first"] = True
            if "looking for" in low:  # templated simulator opener -> parse the category
                category_text, extra = _split_first_message(low)
                state["category_terms"] = _tokens(category_text)
                if extra:
                    state["constraint_terms"] += _tokens(extra)
            else:  # free text -> let the LLM's category slot drive the gate
                state["constraint_terms"] += _tokens(_BOILERPLATE_RE.sub(" ", low))
            return

        if "ignore my earlier" in low or "what i need is" in low or low.startswith("actually"):
            # Intent override: the *new* intent is what counts, but keep the
            # already-accumulated tokens too -- the evaluator only blocks
            # conversion before this turn, and stale tokens often still point at
            # the target (e.g. a product name mentioned in the old preference).
            new_value = low.split("what i need is", 1)[1] if "what i need is" in low else low
            state["constraint_terms"] = _tokens(new_value.strip(" :.\"'")) + state["constraint_terms"]
            return

        if "additional preference" in low:  # customer has nothing left to add
            state["exhausted"] = True
            return

        if "please use your judgment" in low or "not quite right" in low:
            return  # no new information this turn

        if "what matters is" in low:
            payload = low.split("what matters is", 1)[1].strip(" :.")
            for chunk in payload.split(";"):
                state["constraint_terms"] += _tokens(chunk)
            return

        state["constraint_terms"] += _tokens(_BOILERPLATE_RE.sub(" ", low))

    # -- LLM intent router (Day 2) --------------------------------------------- #
    def _route(self, state: dict, message: str) -> tuple[int, int]:
        """Classify intent + extract slots via the LLM, merge into ``state``.

        Returns ``(prompt_tokens, completion_tokens)`` billed *this turn* (0 when
        the call is skipped, served from cache, or the LLM is unavailable).
        """
        if not self._llm_on():
            return 0, 0
        low = message.strip().lower()
        if state["seen_first"] and any(marker in low for marker in _NO_INFO_MARKERS):
            return 0, 0  # boilerplate turn -- nothing to extract, save the call

        cache_key = (
            message.strip(),
            state["slots"].get("category"),
            tuple(sorted(state["keywords"])),
        )
        if cache_key in self._route_cache:
            self._apply_route(state, self._route_cache[cache_key])
            return 0, 0

        try:
            parsed, prompt_tokens, completion_tokens = self._call_router(state, message)
        except Exception:
            self._note_llm_failure()  # trips the breaker after a few consecutive fails
            return 0, 0

        self._llm_fail_streak = 0
        self._route_cache[cache_key] = parsed
        if len(self._route_cache) > 4096:
            self._route_cache.clear()
        self._apply_route(state, parsed)
        self.llm_usage_total["prompt_tokens"] += prompt_tokens
        self.llm_usage_total["completion_tokens"] += completion_tokens
        return prompt_tokens, completion_tokens

    def _gemini_generate(
        self,
        system: str,
        user: str,
        *,
        schema: dict | None = None,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> tuple[str, int, int]:
        """One Gemini ``generateContent`` call. Returns (text, prompt_toks, completion_toks)."""
        generation_config: dict = {"temperature": 0.0, "maxOutputTokens": max_tokens}
        if schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = schema
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }).encode("utf-8")

        url = _GEMINI_URL.format(model=self._llm_model) + "?key=" + self._llm_key
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=_LLM_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))

        parts = payload["candidates"][0]["content"].get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
        usage = payload.get("usageMetadata", {})
        return (
            text,
            int(usage.get("promptTokenCount", 0) or 0),
            int(usage.get("candidatesTokenCount", 0) or 0),
        )

    def _call_router(self, state: dict, message: str) -> tuple[dict, int, int]:
        context = {
            "current_intent": state["intent"],
            "known_slots": {k: v for k, v in state["slots"].items() if v},
            "known_keywords": state["keywords"][-12:],
        }
        # Read-only personalization. Only the tags travel: `summary` is itself
        # generated from those same tags ("Prior purchases emphasize material,
        # fit; ..."), so sending both would spend prompt tokens restating them.
        # _ROUTER_SYSTEM forbids turning these into slots or keywords.
        prior_aspects = (state.get("profile") or {}).get("tags") or []
        if prior_aspects:
            context["shopper_prior_aspects"] = prior_aspects
        user_block = (
            "CONVERSATION STATE:\n" + json.dumps(context, ensure_ascii=False)
            + "\n\nNEW CUSTOMER MESSAGE:\n" + message.strip()
            + "\n\nReturn the routing JSON."
        )
        text, prompt_tokens, completion_tokens = self._gemini_generate(
            _ROUTER_SYSTEM, user_block, schema=_ROUTER_SCHEMA
        )
        return _parse_router_json(text), prompt_tokens, completion_tokens

    def _apply_route(self, state: dict, parsed: dict) -> None:
        intent = str(parsed.get("intent", "")).strip().lower()
        if intent in ("buying", "browsing"):
            state["intent"] = intent

        slots = parsed.get("extracted_slots")
        slots = slots if isinstance(slots, dict) else {}
        new_category = slots.get("category")
        new_category = new_category.strip().lower() if isinstance(new_category, str) and new_category.strip() else None

        # Requirement 4: a genuine category change wipes the conflicting slots.
        # Guard on "the LLM told us what to switch to" so a mis-fire on a mere
        # preference swap can't erase a good category gate.
        if parsed.get("intent_override") is True and new_category:
            for key in ("category", "style", "use_case"):
                state["slots"][key] = None
            state["keywords"] = []
            state["category_terms"] = []      # drop the abandoned category gate
            state["constraint_terms"] = []    # and its now-stale constraint tokens
            # colour / material / brand / budget slots survive (not category-bound);
            # the LLM nulls them itself if it judged them stale.

        for key in SLOT_KEYS:
            value = slots.get(key)
            if isinstance(value, str) and value.strip():
                state["slots"][key] = value.strip().lower()

        extra = slots.get("keywords")
        if isinstance(extra, list):
            for word in extra:
                if isinstance(word, str) and word.strip():
                    state["keywords"].append(word.strip().lower())
        state["keywords"] = _dedupe(state["keywords"])[-20:]

    # -- query construction ------------------------------------------------- #
    def _slot_terms(self, state: dict) -> tuple[list[str], list[str]]:
        """Merge the LLM state-machine slots with the heuristic token lists.

        With the LLM disabled every slot is ``None`` and this returns exactly the
        Day 1 ``category_terms`` / ``constraint_terms``.
        """
        slots = state["slots"]
        cat_src = list(state["category_terms"])
        if slots.get("category"):
            cat_src += _tokens(slots["category"])

        con_src = list(state["constraint_terms"])
        for key in ("gender", "color", "material", "style", "brand", "use_case", "budget"):
            if slots.get(key):
                con_src += _tokens(slots[key])
        for word in state["keywords"]:
            con_src += _tokens(word)
        return cat_src, con_src

    def _build_queries(self, state: dict, rotate: int = 0) -> list[str]:
        cat_src, con_src = self._slot_terms(state)
        cat = [t for t in _dedupe(cat_src) if t not in _CAT_DROP]
        cat_set = set(cat)
        con = [t for t in _dedupe(con_src)[-60:] if t not in cat_set]

        def group(terms: list[str], op: str) -> str:
            return "(" + f" {op} ".join(f'"{t}"' for t in terms) + ")" if terms else ""

        # The AND gate uses only the last few (most specific / leaf) category
        # tokens, minus broad top-level nodes that leak in ("shoes jewelry") and
        # that no single product satisfies together.
        core = [t for t in cat if t not in _CAT_JUNK][-3:] or cat[-2:]
        core_and, cat_and = group(core, "AND"), group(cat, "AND")
        cat_or, con_or = group(cat, "OR"), group(con, "OR")
        specific = [t for t in con if t not in GENERIC and len(t) > 2][:4]

        strict = f"{core_and} AND {group(specific, 'AND')}" if core_and and len(specific) >= 2 else ""

        tiers: list[str] = []
        if con_or:
            # "buying" = firm requirements -> lead with the strict specific-term AND.
            if strict and state.get("intent") == "buying":
                tiers.append(strict)
            if core_and:
                tiers.append(f"{core_and} AND {con_or}")   # leaf category + any constraint (primary)
            if cat_or and cat_or != core_and:
                tiers.append(f"{cat_or} AND {con_or}")     # any category token + any constraint
            tiers.append(f"{cat_or} OR {con_or}" if cat_or else con_or)  # broad recall net
            if strict:                                     # category + specific terms AND-ed
                tiers.append(strict)
        elif cat_or:                                        # pure-category (browsing, pre-constraint)
            tiers.append(cat_and)
            if cat_and != cat_or:
                tiers.append(cat_or)
        tiers = _dedupe([t for t in tiers if t])
        # Once the customer has nothing left to add, cycle which recall strategy
        # leads so a buried target gets a fresh ranking pass each turn instead of
        # the same frozen list.
        if rotate and len(tiers) > 1:
            shift = rotate % len(tiers)
            tiers = tiers[shift:] + tiers[:shift]
        return tiers

    def _search(self, expressions: list[str], top_k: int, *, limit: int | None = None) -> list[str]:
        # Weighted reciprocal-rank fusion across the tiers: a product that ranks
        # well under the precise tier *and* shows up in the broad net beats junk
        # that only appears once.  This avoids an early tier starving a later one.
        depth = max(top_k * 3, 30, (limit or 0))
        fused: dict[str, float] = {}
        for position, expression in enumerate(expressions):
            if not expression:
                continue
            try:
                rows = self.connection.execute(
                    f"SELECT parent_asin FROM products WHERE products MATCH ? "
                    f"ORDER BY {self._order_by} LIMIT ?",
                    (expression, depth),
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            weight = 1.0 / (1 + position)
            for rank, (asin,) in enumerate(rows):
                fused[asin] = fused.get(asin, 0.0) + weight / (10 + rank)
        ordered = sorted(fused, key=lambda a: fused[a], reverse=True)
        return ordered[: (limit or top_k)]

    def _bm25_ranked(self, state: dict) -> list[str]:
        """The FTS5 BM25 track: tiered query -> internal RRF -> top RRF_DEPTH ids."""
        expressions = self._build_queries(state, rotate=state["stale"])
        if not expressions:
            return []
        return self._search(expressions, RRF_DEPTH, limit=RRF_DEPTH)

    def _dense_ranked(self, state: dict, user_message: str) -> list[str]:
        """The Sentence-Transformer track: cosine top RRF_DEPTH ids (or [] if off)."""
        if not self._dense_on():
            return []
        return self._dense_rank(self._dense_query_text(state, user_message), RRF_DEPTH)

    @staticmethod
    def _rrf_fuse(*ranked_lists: list[str], k: int = RRF_K, top_k: int | None = None) -> list[str]:
        """Textbook Reciprocal Rank Fusion: score(d) = sum_l 1 / (k + rank_l(d))."""
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, doc_id in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        order = sorted(scores, key=lambda d: scores[d], reverse=True)
        return order[:top_k] if top_k else order

    # -- Day 3: over-generality -> proactive clarification ------------------- #
    def _count_matches(self, expression: str) -> int:
        if not expression:
            return 0
        try:
            row = self.connection.execute(
                "SELECT count(*) FROM products WHERE products MATCH ?", (expression,)
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row else 0

    def _clarify_slot(self, state: dict, turn: int) -> tuple[str, int] | None:
        """Return ``(attribute_to_ask, match_count)`` when the query is too broad
        to answer usefully, else ``None`` to proceed with results.

        Fires only when the shopper has pinned a category but nothing that
        discriminates within it, and that category alone matches a huge slice of
        the catalog -- i.e. showing 10 rows now would be close to random.
        """
        if not CLARIFY_ENABLED or state["exhausted"] or turn > CLARIFY_MAX_TURN:
            return None

        cat_src, con_src = self._slot_terms(state)
        cat = [t for t in _dedupe(cat_src) if t not in _CAT_DROP]
        con = [t for t in _dedupe(con_src) if t not in set(cat)]
        discriminating = [t for t in con if t not in GENERIC and len(t) > 2]
        if not cat or discriminating:
            return None  # nothing to gate on, or the shopper already narrowed it

        gate = "(" + " AND ".join(f'"{t}"' for t in (cat[-3:] or cat)) + ")"
        matches = self._count_matches(gate)
        if matches < OVERGENERAL_MATCHES:
            return None

        known = {slot for slot, value in state["slots"].items() if value} | {"category"}
        for slot in CLARIFY_PRIORITY:
            if slot not in known:
                return slot, matches
        return None

    def _clarify_question(self, state: dict, slot: str, matches: int) -> tuple[str, int, int]:
        """Return (natural question, prompt_tokens, completion_tokens)."""
        category = state["slots"].get("category") or " ".join(
            t for t in _dedupe(self._slot_terms(state)[0]) if t not in _CAT_DROP
        ) or "items"
        fallback = (
            f"I found a lot of {category} ({matches:,}+ matches). To narrow it down, "
            f"do you have a {slot} in mind?"
        )
        if not self._llm_on():
            return fallback, 0, 0
        prompt = (
            f"Category: {category}\nMatches: {matches}\nAttribute to ask about: {slot}\n"
            "Write the question."
        )
        try:
            text, prompt_tokens, completion_tokens = self._gemini_generate(
                _CLARIFY_SYSTEM, prompt, max_tokens=80
            )
        except Exception:
            self._note_llm_failure()
            return fallback, 0, 0
        self.llm_usage_total["prompt_tokens"] += prompt_tokens
        self.llm_usage_total["completion_tokens"] += completion_tokens
        return " ".join(text.split()) or fallback, prompt_tokens, completion_tokens

    # -- public contract -------------------------------------------------- #
    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._state.get(session_id)
        if state is None:
            state = self._state[session_id] = self._new_state()
        state["history"].append(user_message)

        # 1. Deterministic parse (Day 1) -- always runs; the offline backstop.
        self._ingest(state, user_message, turn)
        # 2. LLM state machine (Day 2) -- intent routing + slot extraction.
        prompt_tokens, completion_tokens = self._route(state, user_message)

        # Track how many turns the constraint set has been unchanged; once the
        # customer has nothing left to add, rotate the recall strategy so a
        # buried target gets a fresh ranking pass instead of a frozen list.
        signature = (
            tuple(state["category_terms"]),
            tuple(state["constraint_terms"]),
            tuple(sorted(f"{k}:{v}" for k, v in state["slots"].items() if v)),
            tuple(state["keywords"]),
        )
        if signature == state.get("last_signature"):
            state["stale"] += 1
        else:
            state["stale"] = 0
            state["last_signature"] = signature

        # 3. Day 3 -- over-generality guard.  If the request is still far too broad
        #    to rank meaningfully, ask for the most useful missing attribute
        #    instead of returning ~random hits.
        clarify = self._clarify_slot(state, turn)
        if clarify is not None:
            slot, matches = clarify
            question, clarify_prompt_tokens, clarify_completion_tokens = self._clarify_question(
                state, slot, matches
            )
            return {
                "message": question,
                "ask_attribute": slot,
                "recommendations": [],
                "usage": {
                    "prompt_tokens": prompt_tokens + clarify_prompt_tokens,
                    "completion_tokens": completion_tokens + clarify_completion_tokens,
                },
            }

        # 4. Hybrid retrieval -- run BOTH tracks every turn and combine them with
        #    Reciprocal Rank Fusion.
        #      * BM25 track  : FTS5 tiered query -> top RRF_DEPTH (=60) ids
        #      * Dense track : all-MiniLM-L6-v2 cosine top RRF_DEPTH (=60) ids
        #    RRF: score(d) = sum over lists of  1 / (RRF_K + rank_in_list(d)).
        #    (Dense list is empty -> this degrades to the pure BM25 ranking.)
        bm25_ranked = self._bm25_ranked(state)
        dense_ranked = self._dense_ranked(state, user_message)
        fused = self._rrf_fuse(bm25_ranked, dense_ranked, k=RRF_K, top_k=None)
        if self._reranker is not None and fused:
            head = self._rerank(
                fused[:RERANK_CANDIDATES], self._dense_query_text(state, user_message),
                bm25_ranked=bm25_ranked, dense_ranked=dense_ranked,
            )
            fused = head + fused[len(head):]
        fused = fused[:top_k]
        recommendations = [{"parent_asin": asin} for asin in fused]

        if state["exhausted"]:
            ask_attribute = None
            message = "Here are the closest matches I found."
        else:
            # "other" makes the simulator disclose its next hidden constraints
            # verbatim, which is the strongest signal we can add each turn.
            ask_attribute = "other"
            message = (
                "Here are some options that fit so far. Any other must-haves "
                "— material, color, brand, or how you'll use it?"
            )

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
