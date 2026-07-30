"""memory_curve — SM-2 spaced repetition based on the Ebbinghaus curve.

SM-2 (SuperMemo-2) is the algorithm Anki is based on. The full algorithm
is parameter-free and runs as plain Python; we annotate every returned
date with the Ebbinghaus-curve "memory retention" at that upcoming review
time, so the agent can explain to the learner why the dates make sense.

References:
- P. Wozniak, "The SuperMemo algorithm: SM-2" (1987).
- H. Ebbinghaus, "Memory: A Contribution to Experimental Psychology" (1885).
"""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import os

DB_PATH = Path(os.environ.get(
    "FT_LEARNER_DB",
    "/workspace/persistence/hackathon/FeynmanTutor/data/learner.sqlite",
))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mastery (
            subskill TEXT NOT NULL,
            skill    TEXT,
            score    REAL NOT NULL,
            ts       TEXT NOT NULL,
            ef       REAL DEFAULT 2.5,
            interval INTEGER DEFAULT 0,
            rep      INTEGER DEFAULT 0
        )
    """)
    return conn


def _ebbinghaus_retention(interval_days: float) -> float:
    """Memory retention fraction at `interval_days` after the last review,
    per the Ebbinghaus forgetting curve. Stability constant = 1 day.
    """
    return float(round(1.0 / (1.0 + max(interval_days, 0.01) ** 1.25), 3))


def record_mastery(subskill: str, score: float, skill: str = "") -> str:
    """Record the mastery `score` (a 0..1 ratio) for `subskill` and compute
    the next review spot using SM-2.

    SM-2 quality ∈ {0..5} — we map the 0..1 mastery score to that range:
        q = round(score * 5)
    """
    score = max(0.0, min(float(score), 1.0))
    q = int(round(score * 5))

    conn = _db()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT ef, interval, rep FROM mastery WHERE subskill=? "
        "ORDER BY ts DESC LIMIT 1", (subskill,)
    ).fetchall()
    prev_ef, prev_i, prev_rep = (rows[0] if rows else (2.5, 0, 0))

    # SM-2 update
    if q < 3:
        new_rep = 0
        new_i = 1
        new_ef = max(1.3, prev_ef - 0.2)
    else:
        new_rep = prev_rep + 1
        if new_rep == 1:
            new_i = 1
        elif new_rep == 2:
            new_i = 6
        else:
            new_i = round(prev_i * prev_ef)
        new_ef = max(1.3, prev_ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))

    now = datetime.utcnow()
    next_dt = now + timedelta(days=max(new_i, 1))
    conn.execute(
        "INSERT INTO mastery (subskill, skill, score, ts, ef, interval, rep) VALUES (?,?,?,?,?,?,?)",
        (subskill, skill, score, now.isoformat(), new_ef, new_i, new_rep),
    )
    conn.commit()
    conn.close()

    return json.dumps({
        "subskill": subskill,
        "score": score,
        "quality": q,
        "interval_days": new_i,
        "next_review_at": next_dt.date().isoformat(),
        "retention_at_next_review": _ebbinghaus_retention(new_i)
    }, ensure_ascii=False)


def schedule_review(subskill: str) -> str:
    """Return the *current* next review spot for `subskill` plus the upcoming
    four future intervals (the typical 1-6-16-35-day Ebbinghaus schedule) so
    the agent can plan ahead."""
    conn = _db()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT score, ts, ef, interval, rep FROM mastery WHERE subskill=? "
        "ORDER BY ts DESC LIMIT 1", (subskill,)
    ).fetchall()
    if not rows:
        conn.close()
        return json.dumps({
            "subskill": subskill,
            "status": "no mastery recorded yet; call record_mastery first",
            "next_review_at": None
        }, ensure_ascii=False)
    score, ts, ef, interval, rep = rows[0]
    next_dt = datetime.fromisoformat(ts) + timedelta(days=interval)
    schedule = []
    cur_i = max(interval, 1)
    for k in range(1, 6):
        cur_i = round(cur_i * ef) if k > 1 else cur_i
        schedule.append({
            "k": k,
            "interval_days": cur_i,
            "retention_at_review": _ebbinghaus_retention(cur_i),
        })
    conn.close()
    return json.dumps({
        "subskill": subskill,
        "last_score": score,
        "next_review_at": next_dt.date().isoformat(),
        "future_schedule": schedule,
    }, ensure_ascii=False)


RECORD_MASTERY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "record_mastery",
        "description": (
            "Record the learner's 0..1 mastery score for a sub-skill, then "
            "compute the next SM-2 spaced-repetition interval. Call this "
            "after grading a quiz."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subskill": {"type": "string"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "skill": {"type": "string", "description": "parent skill, optional"},
            },
            "required": ["subskill", "score"],
        },
    },
}

SCHEDULE_REVIEW_SCHEMA = {
    "type": "function",
    "function": {
        "name": "schedule_review",
        "description": "Return the upcoming spaced-repetition schedule for a sub-skill.",
        "parameters": {"type": "object", "properties": {"subskill": {"type": "string"}}, "required": ["subskill"]},
    },
}
