"""web_research — find authoritative learning material for a topic via DuckDuckGo."""

from __future__ import annotations
import json
from ddgs import DDGS


def web_research(topic: str, max_results: int = 5) -> str:
    """Search DuckDuckGo for `topic` and return up to `max_results` links,
    each with title / url / snippet. Free, no API key required.
    """
    max_results = max(1, min(int(max_results), 10))
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(topic, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href") or r.get("link", ""),
                    "snippet": (r.get("body") or r.get("snippet", ""))[:300],
                })
    except Exception as exc:  # noqa: BLE001
        results.append({"error": f"web_research failed: {exc!r}"})
    return json.dumps({"query": topic, "results": results[:max_results]}, ensure_ascii=False)


WEB_RESEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_research",
        "description": (
            "Search the public web (DuckDuckGo) for learning material on a "
            "topic. Returns title/url/snippet for each result. Use this when "
            "the learner mentions a new topic or you want authoritative "
            "references to ground your explanations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "the search query"},
                "max_results": {
                    "type": "integer",
                    "description": "how many links to return (1..10, default 5)",
                    "default": 5,
                },
            },
            "required": ["topic"],
        },
    },
}
