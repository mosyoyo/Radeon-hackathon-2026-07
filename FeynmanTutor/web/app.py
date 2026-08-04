"""web/app.py — FastAPI web front-end for FeynmanTutor.

Exposes:
  GET  /            -> chat UI (single-page, no build step)
  POST /api/session -> create/reset a learning session (optional topic)
  POST /api/chat    -> SSE stream; send a learner message, get back the
                       agent's events (tool calls + assistant text) as they
                       happen. Session state is kept server-side per session
                       id (in-memory), so the same session can continue.

Run (from repo root):
    uvicorn web.app:app --host 127.0.0.1 --port 8500
(or use server/start_web.sh)
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.core import Agent

app = FastAPI(title="FeynmanTutor Web")

# in-memory sessions: session_id -> Agent
_SESSIONS: dict[str, Agent] = {}
_LAST_SEEN: dict[str, float] = {}
SESSION_TTL = 3600  # seconds of inactivity before a session is dropped


class SessionCreate(BaseModel):
    topic: str = Field(default="", description="optional seed topic for a scripted journey")


class ChatMessage(BaseModel):
    session_id: str
    message: str = Field(default="", description="learner reply / instruction; empty on first send of a scripted topic")


def _cleanup() -> None:
    now = time.time()
    dead = [sid for sid, t in _LAST_SEEN.items() if now - t > SESSION_TTL]
    for sid in dead:
        _SESSIONS.pop(sid, None)
        _LAST_SEEN.pop(sid, None)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "model": os.environ.get("FT_MODEL", "Qwen2.5-14B-Instruct")}


@app.post("/api/session")
async def create_session(req: SessionCreate) -> dict:
    _cleanup()
    sid = uuid.uuid4().hex[:12]
    _SESSIONS[sid] = Agent(topic=req.topic)
    _LAST_SEEN[sid] = time.time()
    return {"session_id": sid}


@app.post("/api/chat")
async def chat(req: ChatMessage) -> StreamingResponse:
    _cleanup()
    agent = _SESSIONS.get(req.session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="session not found (expired?) — start a new one")

    _LAST_SEEN[req.session_id] = time.time()
    if req.message:
        agent.add_user_message(req.message)

    return StreamingResponse(
        _agent_stream(agent),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _agent_stream(agent: Agent):
    """Yield SSE events: tool events, assistant text, and a final done event."""
    max_rounds = 10
    last_text = ""
    try:
        for _ in range(max_rounds):
            out = agent.step()
            kind = out.get("kind")

            if kind == "error":
                yield _sse("error", {"content": out["content"]})
                break

            if kind == "tool":
                # preamble text the model wrote before calling tools
                preamble = out.get("content") or ""
                if preamble:
                    yield _sse("assistant", {"content": preamble})
                for name in out["tool_names"]:
                    yield _sse("tool", {"name": name})
                # loop continues — the tool results have been fed back to the
                # model, and the next step() picks up its narration.
                last_text = ""
                continue

            if kind == "text":
                content = out["content"]
                # stop if the model is repeating itself (degenerate loop)
                if content == last_text:
                    yield _sse("done", {"ok": True, "note": "repetition-stop"})
                    return
                yield _sse("assistant", {"content": content})
                last_text = content
                # If the model is asking the learner something (probe), stop
                # and wait for the next user message.
                if _ends_with_question(content):
                    yield _sse("done", {"ok": True, "waiting": True})
                    return

        yield _sse("done", {"ok": True, "note": "max-rounds"})
    except Exception as exc:  # noqa: BLE001
        yield _sse("error", {"content": f"internal error: {exc!r}"})


def _ends_with_question(text: str) -> bool:
    """Heuristic: a Feynman probe usually ends in '?'. If the text looks like a
    question we stop the loop and let the learner answer."""
    t = text.strip()
    return "?" in t[-60:] or t.endswith("?：")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
