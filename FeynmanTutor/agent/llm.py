"""llm — minimal OpenAI-compatible client wrapper.

The FeynmanTutor agent always talks to a **local vLLM server** that exposes
an OpenAI-compatible API. We use the official `openai` Python SDK pointed at
`base_url` (default http://127.0.0.1:8000/v1).

This module exposes two functions:
- `client` — the shared SDK client
- `chat(system, user, max_tokens)` — a one-shot completion used by the Feynman /
  quiz / notes tool prompts that need a focused, single-turn response
  (distinct from the multi-turn ReAct loop in `core.py`).
"""

from __future__ import annotations
import os
from openai import OpenAI

_BASE_URL = os.environ.get("FT_BASE_URL", "http://127.0.0.1:8000/v1")
_API_KEY = os.environ.get("FT_API_KEY", "not-needed-for-local")
MODEL = os.environ.get("FT_MODEL", "Qwen2.5-14B-Instruct")

client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)


def chat(system: str, user: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
    """One-shot completion, returns the assistant's message text."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
