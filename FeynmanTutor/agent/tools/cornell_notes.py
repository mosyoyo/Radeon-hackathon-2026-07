"""cornell_notes — synthesize a Cornell-style markdown note for a sub-skill."""

from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime

from ..llm import chat


NOTES_ROOT = Path(os.environ.get(
    "FT_NOTES_ROOT",
    "/workspace/persistence/hackathon/FeynmanTutor/data/notes",
))
NOTES_ROOT.mkdir(parents=True, exist_ok=True)


CORNELL_INSTRUCTIONS = """\
Synthesize a Cornell-style study note from the conversation so far for the
given sub-skill. The note must have THREE sections, separated by horizontal
rules:

1. **Cues** — a bulleted list of key terms / questions (the left column).
2. **Notes** — the plain-words explanation in 1-3 short paragraphs (the right
   column). Include the analogy we used. No filler.
3. **Summary** — a single 2-3 sentence recapitulation at the bottom.

Format strictly with Markdown. No headings other than 'Cues', 'Notes',
'Summary'. Total length < 500 words.
"""


def cornell_notes(subskill: str, context: str = "") -> str:
    """Write a Cornell-style markdown note for `subskill` to disk and return
    the path / preview. The optional `context` (RAG hits or web snippets)
    grounds the note.
    """
    user = f"Sub-skill: {subskill}\n"
    if context:
        user += f"\nReference material:\n{context[:2000]}\n"
    md = chat(system=CORNELL_INSTRUCTIONS, user=user, max_tokens=900)

    safe = subskill.replace(" ", "-").lower()[:40]
    fname = NOTES_ROOT / f"{datetime.utcnow():%Y%m%d-%H%M%S}_{safe}.md"
    fname.write_text(f"# {subskill}\n\n{md}\n", encoding="utf-8")
    return json.dumps({
        "subskill": subskill,
        "path": str(fname),
        "preview": md[:500],
    }, ensure_ascii=False)


CORNELL_NOTES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "cornell_notes",
        "description": (
            "Synthesize a Cornell-style study note for a sub-skill and save "
            "it to disk as Markdown. Call this at the end of each sub-skill "
            "block so the learner has a permanent, organized note set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subskill": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["subskill"],
        },
    },
}
