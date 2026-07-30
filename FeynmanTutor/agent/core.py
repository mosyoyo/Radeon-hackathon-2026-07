"""core — the FeynmanTutor agent loop.

A dependency-free ReAct loop over OpenAI tool-calling. The agent is *driven*:
once given a goal, it tries to decompose, research, teache probe, quiz,
schedule review, and summarize without Further prompting. The user can
interject at any moment and the loop will treat that as a new instruction.

Stages of a session:
  plan           — make_plan()
  research       — web_research(), parse_uploaded_material(), index_material()
  feynman(x N)   — feynman_explain() → feynman_probe() → waits for learner answer
  quiz           — quiz() → grade → record_mastery(x N) → schedule_review(x N)
  notes          — cornell_notes() for each sub-skill
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field

from .llm import client, MODEL
from . import prompts
from .tools import TOOL_SCHEMAS, TOOL_FUNCTIONS


SYSTEM_PROMPT = prompts.prompt_for_stage("base")


@dataclass
class Agent:
    topic: str
    history: list = field(default_factory=list)
    transcript_path: str | None = None  # if set, append each event as jsonl

    def __post_init__(self) -> None:
        if self.topic:
            self.history.append({"role": "system", "content": SYSTEM_PROMPT})
            user_msg = (
                f"I want to learn: \"{self.topic}\".\n"
                "Walk me through the full workflow end-to-end. "
                "Decompose the skill, gather materials, then for each sub-skill: "
                "give a Feynman explanation, probe my understanding with one "
                "question, wait for my answer, then quiz me and record the score. "
                "When all sub-skills are done, write Cornell notes and schedule reviews. "
                "At each step, tell me what you're doing and why."
            )
            self.history.append({"role": "user", "content": user_msg})

    def step(self, max_tokens: int = 1200) -> dict:
        """One iteration of the ReAct loop.

        Returns a dict describing the round:
          {"kind": "text"|"tool", "content": "...", "tool_name": "...", "tool_args": {...}}
        """
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=self.history,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=0.3,
            )
        except Exception as exc:  # noqa: BLE001
            return {"kind": "error", "content": f"LLM call failed: {exc!r}"}

        msg = resp.choices[0].message
        # Convert the SDK message into a dict we can push back as the message
        msg_dict = _msg_to_dict(msg)
        self.history.append(msg_dict)
        self._persist({"role": "assistant", **msg_dict})

        if not msg.tool_calls:
            return {"kind": "text", "content": msg.content or ""}

        # Execute each requested tool
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            handler = TOOL_FUNCTIONS.get(name)
            if handler is None:
                result = json.dumps({"error": f"unknown tool: {name}"})
            else:
                try:
                    result = handler(**args)
                except TypeError as exc:
                    result = json.dumps({"error": f"bad arguments: {exc!r}"})
                except Exception as exc:  # noqa: BLE001
                    result = json.dumps({"error": f"tool {name} failed: {exc!r}"})

            tool_event = {
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": result,
            }
            self.history.append(tool_event)
            self._persist({"role": "tool", "tool": name, "arguments": args, "result": result})

        return {"kind": "tool", "tool_names": [c.function.name for c in msg.tool_calls]}

    def add_user_message(self, text: str) -> None:
        """Trampoline for the CLI runner to inject a learner reply."""
        self.history.append({"role": "user", "content": text})
        self._persist({"role": "user", "content": text})

    def _persist(self, event: dict) -> None:
        if not self.transcript_path:
            return
        event_with_ts = {"t": time.time(), **event}
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_with_ts, ensure_ascii=False) + "\n")


def _msg_to_dict(msg) -> dict:
    """The OpenAI SDK's ChatCompletionMessage doesn't serialize cleanly, do it by hand."""
    d = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d
