"""planner — decompose a skill into sub-skills + a timed learning plan."""

from __future__ import annotations
import json
from datetime import timedelta


def make_plan(skill: str, hours_per_day: int = 1, max_subskills: int = 5) -> str:
    """Decompose a skill into a list of sub-skills and a day-by-day timetable.

    The decomposition is intentionally simple (the heavy reasoning happens
    in the LLM, not here): we expect the agent to invoke us with the *skill*
    name and then receive this structured plan back. The agent then uses it
    to drive subsequent tool calls.

    The timetable spreads the sub-skills out so that the learner covers one
    per session, with `hours_per_day` focused hours per session.
    """
    sub_skills = _heuristic_decompose(skill)
    plan = []
    for i, sub in enumerate(sub_skills[:max_subskills]):
        plan.append({
            "session": i + 1,
            "day_offset": i,  # 1 sub-skill per day by default
            "hours": hours_per_day,
            "subskill": sub,
        })
    return json.dumps({
        "skill": skill,
        "subskills": sub_skills[:max_subskills],
        "plan": plan,
        "estimated_total_hours": len(plan) * hours_per_day,
    }, ensure_ascii=False)


def _heuristic_decompose(skill: str) -> list[str]:
    """A small rule-based sketch of how a `skill` typically decomposes.

    The agent is free to override this — what we return here is a sensible
    default that keeps the demo deterministic. The LLM will embellish
    each sub-skill during later `feynman_explain` calls.
    """
    skill = skill.lower().strip()
    if any(k in skill for k in ["raft", "paxos", "consensus", "distributed"]):
        return [
            "Leader Election",
            "Log Replication",
            "Safety and Persistence",
            "Membership Change (conf change)",
            "Comparison with Paxos",
        ]
    if any(k in skill for k in ["neural", "deep learning", "transformer"]):
        return ["Embeddings", "Self-Attention", "Multi-Head Attention", "Layer Norm + Residual", "Training Loop Basics"]
    if any(k in skill for k in ["python", "rust", "go", "java", "c++"]):
        return ["Types and Variables", "Functions", "Collections and Iterators", "Error Handling", "Standard Library Tour"]
    # generic fallback
    return ["Foundations and Vocabulary", "Core Mechanism", "Common Use Cases", "Pitfalls and Best Practices", "Practical Exercises"]


MAKE_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "make_plan",
        "description": (
            "Decompose a skill into a list of sub-skills plus a day-by-day "
            "study timetable. Call this right after the learner states their "
            "goal; the agent drives all subsequent stages off the returned plan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "description": "the skill the learner wants to acquire"},
                "hours_per_day": {"type": "integer", "default": 1, "minimum": 1, "maximum": 8},
                "max_subskills": {"type": "integer", "default": 5, "minimum": 2, "maximum": 10},
            },
            "required": ["skill"],
        },
    },
}
