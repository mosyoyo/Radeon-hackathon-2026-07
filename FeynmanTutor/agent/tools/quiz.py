"""quiz — generate questions, then the agent grades the answers."""

from __future__ import annotations
import json

from ..llm import chat


QUIZ_INSTRUCTIONS = """\
Generate {n} quiz questions on the given sub-skill for an intermediate
learner. Constraints:
- A mix: short-answer, multiple-choice, and "explain in one sentence".
- Each question must be answerable from what we just covered.
- Avoid overlapping questions (no two questions test the same fact).
- Output STRICT JSON. No prose, no markdown. Schema:
  {{"questions": [{{"q": "...", "type": "short_answer|multiple_choice",
      "options": ["a", "b", "c", "d"] or null,
      "answer": "model answer"}}]}}
- For short_answer, the model answer should be concise enough to grade
  in one line.
"""


def quiz(subskill: str, n: int = 5, context: str = "") -> str:
    """Generate `n` questions about `subskill`. Returns JSON ready to be
    presented to the learner and graded against their answers.
    """
    n = max(1, min(int(n), 10))
    sys_prompt = QUIZ_INSTRUCTIONS.format(n=n)
    user = f"Sub-skill: {subskill}\n"
    if context:
        user += f"\nReference material:\n{context[:1500]}\n"
    raw = chat(system=sys_prompt, user=user, max_tokens=800)
    # best-effort JSON parse; the agent loops on grading
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # try to find the JSON object inside the text
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(raw[start: end + 1])
            except json.JSONDecodeError:
                payload = {"questions": [{"q": raw, "type": "short_answer", "answer": ""}]}
        else:
            payload = {"questions": [{"q": raw, "type": "short_answer", "answer": ""}]}
    return json.dumps(payload, ensure_ascii=False)


QUIZ_SCHEMA = {
    "type": "function",
    "function": {
        "name": "quiz",
        "description": "Generate quiz questions on a sub-skill for the learner.",
        "parameters": {
            "type": "object",
            "properties": {
                "subskill": {"type": "string"},
                "n": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                "context": {"type": "string", "description": "optional reference material from RAG"},
            },
            "required": ["subskill"],
        },
    },
}
