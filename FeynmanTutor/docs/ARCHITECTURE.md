# System Architecture — FeynmanTutor

FeynmanTutor is a **single-process Python agent driving a local ROCm-hosted
LLM**. There are two physically separate runtime components:

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                        User (Terminal / Web)                       │
   └──────────────────────────────┬───────────────────────────────────┘
                                  │ text in / text out
                                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                        Agent process (Python)                     │
   │                                                                   │
   │   ┌─────────────┐    tool_call    ┌──────────────────────────┐   │
   │   │   ReAct      │ ─────────────▶ │     Tool dispatcher      │   │
   │   │   loop       │                │                          │   │
   │   │  (core.py)   │ ◀───────────── │  feynman / memory_curve  │   │
   │   │              │    tool_result  │  planner / quiz / notes   │   │
   │   │              │                │  web_research / loader    │   │
   │   └─────┬───────┘                │  knowledge_base (RAG)     │   │
   │         │ chat.completions         └──────────────────────────┘   │
   │         │ (OpenAI-compatible HTTP)                                │
   └─────────┼───────────────────────────────────────────────────────┘
             │
             ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │     vLLM server on ROCm (gfx1100, 48 GB VRAM)                     │
   │     model: Qwen2.5-14B-Instruct (bf16)                            │
   │     endpoint: http://127.0.0.1:8000/v1                            │
   └──────────────────────────────────────────────────────────────────┘
```

Two heavy components sit on the GPU:

1. the main Qwen2.5-14B LLM served by vLLM (interactive chat + tool calling)
2. a small embedding model (`BAAI/bge-base-en-v1.5`) used by the RAG store

They share the 48 GB VRAM: Qwen2.5-14B in bf16 takes ~28 GB, leaving
~16 GB for KV cache + the ~0.4 GB embedding model — comfortable on gfx1100.

---

## The Agent loop

We implement a small, dependency-free ReAct loop in `agent/core.py`. It
follows the standard **thought → tool → observation → thought** pattern but
uses the OpenAI tool-calling schema (Qwen2.5 supports it natively) instead
of prompt-parsed ReAct.

```python
# core.py (simplified)
def run(topic: str):
    messages = [system_prompt, user_prompt(topic)]
    while True:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            yield msg.content
            return
        for call in msg.tool_calls:
            result = dispatch(call.name, json.loads(call.arguments))
            messages.append({  # tool result back to the model
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": result,
            })
            yield tool_event(call.name, result)
```

Max iterations are bounded (default 16) to avoid runaway loops.

---

## Tools (the agent's "hands")

Every tool is a pure Python function with a JSON-Schema signature so the
model can call it through the OpenAI tool-calling protocol. No external paid
APIs are required.

| Tool                     | Purpose                                                              | Inputs                              | Outputs                                |
| ------------------------ | -------------------------------------------------------------------- | ----------------------------------- | -------------------------------------- |
| `web_research`           | Find learning material for a topic                                   | `topic: str`, `max_results: int`    | list of `{title, url, snippet}`        |
| `parse_uploaded_material`| Parse a user-uploaded PDF/DOCX/MD file                               | `file_path: str`                   | `{n_pages, n_chunks, preview}`         |
| `index_material`         | Embed parsed material into the local FAISS store                     | `corpus_id: str`                    | `{n_vectors}`                          |
| `rag_lookup`             | Retrieve top-K passages from the store                               | `query: str`, `corpus_id: str`     | list of `{passage, score}`             |
| `make_plan`              | Decompose a skill into sub-skills + a timed plan                     | `skill: str`, `hours_per_day: int` | `{subskills: [...], plan: [...]}`      |
| `feynman_explain`        | Explain a sub-skill in plain words                                   | `subskill: str`                    | `(see agent prompts; returns text)`   |
| `feynman_probe`          | Pose a follow-up question to test the learner's understanding       | `subskill: str`, `level: int`      | a question string                      |
| `quiz`                   | Generate N multiple-choice / short-answer questions                  | `subskill: str`, `n: int`          | list of `Q/A`                          |
| `record_mastery`         | Record a 0–1 mastery score for a sub-skill and trigger a review date | `subskill: str`, `score: float`    | `{next_review_at: date}`               |
| `schedule_review`        | SM-2 + Ebbinghaus curve scheduler                                    | `subskill: str`, `history: [...]`  | `{intervals, next_review_at}`         |
| `cornell_notes`          | Build a Cornell-style note from the session                          | `subskill: str`                    | markdown notes                          |

### Why some "tools" are just calls back into the LLM

The Feynman explanation, the quiz generation, and the Cornell-note synthesis
are essentially extra LLM completions with tuned system prompts. We expose
them as tools so the agent can invoke them *autonomously* at the right moment
(by the ReAct loop) rather than the model trying to do everything in one
giant completion. Splitting into tools gives us three properties:

1. **Tracability** — each step is a discrete event in the transcript.
2. **Caching / replay** — a tool result can be cached and replayed offline
   during the demo video.
3. **Testability** — each tool is unit-testable without the LLM.

---

## Knowledge base (the agent's "memory")

![RAG diagram](../assets/rag.png) <!-- TODO add diagram -->

- All uploaded material and the search snippets (after the user opts in) are
  chunked (~512 tokens, 64-token overlap) and embedded with
  `BAAI/bge-base-en-v1.5` (110M params, ~0.4 GB VRAM).
- Vectors are stored in a local FAISS index, persisted per corpus under
  `FeynmanTutor/data/kb/<corpus_id>`.
- `rag_lookup` is how the agent pulls facts back when answering Feynman
  follow-ups or building quizzes — this is what makes it a *learning* agent
  rather than a pure LLM tutor.

The learner's mastery and review schedule live in a tiny SQLite file
(`FeynmanTutor/data/learner.sqlite`) — no server, no external DB.

---

## How each learning methodology maps onto the loop

Below is one concrete trace for the demo topic **Raft consensus algorithm**.
See `examples/session_raft.jsonl` for the full recorded transcript.

```
1) user: "I want to learn the Raft consensus algorithm"
2) agent: make_plan("Raft")
        → subskills: [Leader Election, Log Replication, Safety, Membership Change]
        → plan: 4 sessions × 1h each, spread over 1 week
3) agent: web_research("Raft consensus algorithm", max_results=5)
        → 5 reference links (the Raft paper, MIT 6.824 lecture notes, …)
4) agent: feynman_explain("Leader Election")
        → "Imagine you're a clerk in a board meeting …"
5) agent: feynman_probe("Leader Election", level=1)
        → "In your own words — what's a 'split vote' and how does Raft avoid it?"
6) user answers
7) agent: quiz("Leader Election", n=5) → grades answers → record_mastery(0.72)
        → schedule_review → next_review_in = 3 days (Ebbinghaus day-3 revisit)
        → cornell_notes("Leader Election") → notes/raft-leader-election.md
8) loop back to step 4 with the next subskill
```

---

## Why this design for an AMD Radeon hackathon

- **Self-contained on one GPU.** No external API key needed (we use DuckDuckGo
  for search, local embeddings for RAG, local LLM for everything else). A
  judge can reproduce the project on a Radeon Cloud instance with zero extra
  accounts.
- **Uses vLLM-the-ROCm-path.** vLLM is the **only** inference engine the
  Radeon Cloud *Model API* template supports natively, so we play to the
  platform's strength.
- **Demonstrates real GPU usage, not just a token response.** Embedding +
  generation cohabit on the same card, which is a more interesting showcase
  for a learning agent than a thin wrapper around an external LLM API.
