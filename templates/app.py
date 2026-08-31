from __future__ import annotations

import json
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from starter.agent import Agent

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "data" / "catalog.jsonl"

app = Flask(__name__)
_agent = None
_agent_error = None
_sessions: dict[str, dict] = {}
_catalog_by_asin: dict[str, dict] | None = None


def load_catalog_index() -> dict[str, dict]:
    global _catalog_by_asin
    if _catalog_by_asin is not None:
        return _catalog_by_asin

    catalog: dict[str, dict] = {}
    if CATALOG_PATH.exists():
        with CATALOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    product = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = str(product.get("parent_asin") or "").strip()
                if asin:
                    catalog[asin] = product
    _catalog_by_asin = catalog
    return catalog


def enrich_recommendations(recommendations):
    catalog = load_catalog_index()
    enriched = []
    for item in recommendations or []:
        asin = str(item.get("parent_asin") or "").strip()
        product = catalog.get(asin, {})
        entry = dict(item)
        entry["parent_asin"] = asin
        entry["title"] = product.get("title") or "Product"
        entry["brand"] = product.get("brand") or product.get("store") or "Unknown brand"
        entry["price"] = product.get("price")
        entry["description"] = (product.get("description") or "")[:280]
        entry["categories"] = product.get("categories") or []
        entry["store"] = product.get("store") or ""
        entry["link"] = product.get("link") or (
            f"https://www.amazon.com/s?k={asin}" if asin else ""
        )
        enriched.append(entry)
    return enriched


def default_profile() -> dict:
    return {
        "purchase_frequency": "occasionally",
        "average_prior_rating": None,
        "rating_style": "balanced",
        "preference_tags": [],
        "summary": "General shopping assistant",
    }


def get_agent() -> Agent | None:
    global _agent, _agent_error
    if _agent is not None:
        return _agent
    if not CATALOG_PATH.exists():
        _agent_error = (
            "Catalog missing. Download and unzip the catalog to "
            f"{CATALOG_PATH} before running the chatbot."
        )
        return None
    try:
        _agent = Agent(str(CATALOG_PATH), use_llm=False)
        _agent_error = None
        return _agent
    except Exception as exc:  # pragma: no cover - defensive
        _agent_error = str(exc)
        return None


def ensure_session(session_id: str | None = None) -> str:
    session_id = session_id or str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = {"turn": 0, "profile": default_profile()}
    return session_id


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "catalog_present": CATALOG_PATH.exists(),
        "agent_ready": get_agent() is not None,
        "message": _agent_error or "ready",
    })


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "A message is required."}), 400

    session_id = ensure_session(payload.get("session_id"))
    profile = payload.get("profile") or _sessions[session_id]["profile"] or default_profile()
    agent = get_agent()

    if agent is None:
        return jsonify({
            "session_id": session_id,
            "message": _agent_error or "The agent could not be created.",
            "ask_attribute": None,
            "recommendations": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        })

    if _sessions[session_id]["turn"] == 0:
        agent.reset(session_id, profile)

    _sessions[session_id]["profile"] = profile
    _sessions[session_id]["turn"] += 1

    response = agent.respond(session_id, message, _sessions[session_id]["turn"], top_k=10)
    response["recommendations"] = enrich_recommendations(response.get("recommendations", []))
    response["session_id"] = session_id
    return jsonify(response)


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
