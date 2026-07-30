"""feynman — Feynman technique: plain-words explanation + probing questions.

The two `tools` here don't do heavy lifting themselves — they're really
*structured prompts* that nudge the main LLM into doing the Feynman thing
inside a focused completion. Splitting them into tool calls (rather than
just sticking everything in `prompts.py`) gives us three wins:

1. Tracability: the transcript shows *exactly* when we entered explain mode
   vs. probe mode. The demo video can highlight each one.
2. Composability: tools can be called in any order by the agent — Feynman,
   probe, go back to Feynman again.
3. Testability: we can unit-test the prompt strings + LLM call without
   touching the rest of the agent.
"""

from __future__ import annotations
import json

from ..llm import chat


FEYNMAN_EXPLAIN_INSTRUCTIONS = """\
Explain the given sub-skill to a motivated but non-expert learner using the
Feynman technique. Constraints:
- Plain words. No jargon without immediate unpacking.
- One strong analogy. If the analogy breaks, say where.
- Keep it under 200 words.
- End with one sentence that names *the one concept* the learner must walk
  away with.
"""


def feynman_explain(subskill: str, context: str = "") -> str:
    """Produce a Feynman-style plain-words explanation of `subskill`.

    The optional `context` (a string the agent composes from RAG hits / web
    research snippets) is appended to the prompt so the model grounds its
    explanation in actual material rather than hallucinating.
    """
    user = f"Sub-skill to explain: {subskill}\n"
    if context:
        user += f"\nContext material (use this):\n{context[:1500]}\n"
    reply = chat(
        system=FEYNMAN_EXPLAIN_INSTRUCTIONS,
        user=user,
        max_tokens=512,
    )
    return json.dumps({"subskill": subskill, "explanation": reply}, ensure_ascii=False)


FEYNMAN_PROBE_INSTRUCTIONS = """\
Pose ONE question that tests whether the learner can re-explain the just-
covered sub-skill in their own words. Constraints:
- Ask about a single, well-defined sub-concept (not a vague open question).
- The question should be answerable without external references.
- Use the second person ("In your own words, what is…?").
- Under 30 words. Do NOT reveal the answer.
"""


def feynman_probe(subskill: str, level: int = 1) -> str:
    """Return a single probing question that tests the learner's understanding.

    `level` 1 asks about the *what*, 2 asks about the *why*, 3 asks about an
    *edge case*. The agent increments levels across rounds.
    """
    level = max(1, min(int(level), 3))
    user = f"Sub-skill: {subskill}\nProbe level: {level} "
    user += ["(ask what it IS)", "(ask WHY it works)", "(ask about an EDGE CASE)"][level - 1]
    reply = chat(
        system=FEYNMAN_PROBE_INSTRUCTIONS,
        user=user,
        max_tokens=128,
    )
    return json.dumps({"subskill": subskill, "level": level, "question": reply}, ensure_ascii=False)


FEYNMAN_EXPLAIN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "feynman_explain",
        "description": "Return a plain-words Feynman-style explanation of a sub-skill.",
        "parameters": {
            "type": "object",
            "properties": {
                "subskill": {"type": "string"},
                "context": {"type": "string", "description": "optional grounding passages from RAG or web search"},
            },
            "required": ["subskill"],
        },
    },
}

FEYNMAN_PROBE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "feynman_probe",
        "description": "Pose a single Feynman follow-up question to test learner understanding.",
        "parameters": {
            "type": "object",
            "properties": {
                "subskill": {"type": "string"},
                "level": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1},
            },
            "required": ["subskill"],
        },
    },
}
