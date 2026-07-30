"""
Tools for FeynmanTutor.

Every tool is a plain Python function with:
  - a docstring the agent reads (the LLM figures out what it does from here)
  - a `schema()` function returning the OpenAI tool-call JSON schema.

We keep tools dependency-free where possible: search uses DuckDuckGo via
`ddgs`, vector store uses `faiss-cpu`, embeddings use `transformers` (the
/opt/venv already has ROCm torch).
"""

from .web_research import web_research, WEB_RESEARCH_SCHEMA
from .material_loader import parse_uploaded_material, MATERIAL_LOADER_SCHEMA
from .knowledge_base import index_material, INDEX_MATERIAL_SCHEMA
from .knowledge_base import rag_lookup, RAG_LOOKUP_SCHEMA
from .planner import make_plan, MAKE_PLAN_SCHEMA
from .feynman import feynman_explain, FEYNMAN_EXPLAIN_SCHEMA
from .feynman import feynman_probe, FEYNMAN_PROBE_SCHEMA
from .quiz import quiz, QUIZ_SCHEMA
from .memory_curve import record_mastery, RECORD_MASTERY_SCHEMA
from .memory_curve import schedule_review, SCHEDULE_REVIEW_SCHEMA
from .cornell_notes import cornell_notes, CORNELL_NOTES_SCHEMA


TOOL_SCHEMAS = [
    WEB_RESEARCH_SCHEMA,
    MATERIAL_LOADER_SCHEMA,
    INDEX_MATERIAL_SCHEMA,
    RAG_LOOKUP_SCHEMA,
    MAKE_PLAN_SCHEMA,
    FEYNMAN_EXPLAIN_SCHEMA,
    FEYNMAN_PROBE_SCHEMA,
    QUIZ_SCHEMA,
    RECORD_MASTERY_SCHEMA,
    SCHEDULE_REVIEW_SCHEMA,
    CORNELL_NOTES_SCHEMA,
]


TOOL_FUNCTIONS = {
    "web_research": web_research,
    "parse_uploaded_material": parse_uploaded_material,
    "index_material": index_material,
    "rag_lookup": rag_lookup,
    "make_plan": make_plan,
    "feynman_explain": feynman_explain,
    "feynman_probe": feynman_probe,
    "quiz": quiz,
    "record_mastery": record_mastery,
    "schedule_review": schedule_review,
    "cornell_notes": cornell_notes,
}
