# FeynmanTutor — Project Specification Document

> Track 2: Development & Local Deployment of Private AI Agents
> AMD Radeon Hackathon 2026-07

---

## 1. Application scenario

**FeynmanTutor** is a fully-local learning-assistant agent that helps an
individual learner acquire a new skill end-to-end. It is the kind of
"private tutor" every self-learner wishes they had but most online
courses aren't: structured, evidence-based, and patient.

Concrete scenario used in our demo: a developer with no distributed
systems background wants to learn the **Raft consensus algorithm**. In
one session they:

1. Get the skill auto-decomposed into Leader Election / Log Replication /
   Safety / Membership Change / Comparison with Paxos.
2. Receive a curated list of authoritative web references (the Raft
   paper, lecture notes), cached into a local RAG knowledge base.
3. Go through each sub-skill with a Feynman-style plain-words explanation,
   followed by a single probe question.
4. Get a 3-5-question quiz on each sub-skill; the agent grades the answers
   and writes the mastery score to a local SQLite store.
5. Receive a spaced-repetition schedule (Ebbinghaus forgetting curve +
   SM-2) so the material is revisited at scientifically-timed intervals.
6. Walk away with a Cornell-style markdown note per sub-skill saved to disk.

The same flow applies to any topic — purely by changing the seed phrase
("I want to learn X") we've run the same loop on distributed consensus,
transformer architectures, and a programming language.

---

## 2. Agent architecture diagram

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                        User (Terminal / Web)                       │
   └──────────────────────────────┬───────────────────────────────────┘
                                  │ text in / text out
                                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                        Agent process (Python)                    │
   │   ┌─────────────┐ tool_call ┌──────────────────────────┐         │
   │   │   ReAct      │ ─────────▶ │     Tool dispatcher      │        │
   │   │   loop       │            │                          │        │
   │   │  (core.py)   │ ◀───────── │  feynman / memory_curve  │        │
   │   │              │ tool_result│  planner / quiz / notes  │        │
   │   │              │            │  web_research / loader   │        │
   │   │              │            │  knowledge_base (RAG)     │        │
   │   └─────┬───────┘            └──────────────────────────┘        │
   │         │ chat.completions (OpenAI-compatible HTTP)              │
   └─────────┼──────────────────────────────────────────────────────┘
             │
             ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │    vLLM server on ROCm (gfx1100, 48 GB VRAM)                     │
   │    model: Qwen2.5-14B-Instruct (bf16)                            │
   │    embedding (BAAI/bge-base-en-v1.5) on the SAME GPU              │
   │    endpoint: http://127.0.0.1:8000/v1 (OpenAI-compatible)        │
   └──────────────────────────────────────────────────────────────────┘
```

Two heavy components share the AMD Radeon GPU:
1. The 14B chat model served by vLLM (≈28 GB VRAM at bf16).
2. A small embedding model (BAAI/bge-base-en-v1.5, ≈0.4 GB) used for RAG.

The agent itself is a single Python process running a dependency-free
ReAct loop over the OpenAI tool-calling schema — Qwen2.5 natively supports
tool calls via the `hermes` parser, so no prompt-parsing ReAct is required.

---

## 3. Introduction to core capabilities

| Capability             | How it's implemented                                      | Why it matters for a learner                                |
| ---------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| Skill decomposition    | `make_plan` tool (rule-based scaffold, LLM emb. later)    | Turns a vague goal into a concrete plan                     |
| Active resource gather | `web_research` (DuckDuckGo via `ddgs`, no API key)        | No external paid API; finds authoritative references       |
| Material ingestion     | `parse_uploaded_material` (PDF/DOCX/MD/TXT), then         | Bring your own textbook, paper, or course to RAG           |
| RAG retrieval          | `index_material` + `rag_lookup` (FAISS + bge-base)       | Grounds explanations in the learner's own material          |
| Feynman technique      | `feynman_explain` (plain words + analogy) +              | Forces faculty articulation, surfaces                      |
|                        | `feynman_probe` (single recall question)                 | comprehension gaps                                         |
| Quizzing                | `quiz` (structured JSON, mixed types)                    | Test what was learned, not what was heard                  |
| Spaced repetition      | `record_mastery` + `schedule_review` (SM-2)             | Ebbinghaus forgetting curve; reviews at optimal intervals  |
| Synthesis notes        | `cornell_notes` (Cues / Notes / Summary)                 | Reusable artifact; reduces re-learning                      |

---

## 4. Model introduction & local deployment plan

### Chosen model
- **Qwen2.5-14B-Instruct** (Alibaba Cloud, Apache 2.0)
  - Why: strong tool-calling (top-tier for its size); fits 48 GB VRAM
    at bf16 with generous KV-cache headroom; Chinese + English
    abilities (useful for our audience)
  - served-model-name: `Qwen2.5-14B-Instruct`
  - dtype: bf16 (matches the snapshot on disk)

### Embedding model (for RAG)
- **BAAI/bge-base-en-v1.5**
  - 110M parameters, 0.4 GB VRAM
  - Loaded directly via `transformers` with `local_files_only=True`

### Deployment plan (one machine, one GPU)
1. **vLLM server on ROCm** at `:8000` (OpenAI-compatible HTTP API).
2. **Agent CLI** (`agent/run_cli.py`) talks to it via the `openai` Python SDK.
3. (optional) **`rc-tunnel`** exposes `:8000` to a public HTTPS url so
   a remote evaluator can drive it from any OpenAI-compatible client.

All inference, embedding and storage stays on the single Radeon GPU and
the local filesystem — no external LLM provider is contacted.

---

## 5. Optimization for inference speed on AMD Radeon GPU

The detailed write-up is in `docs/ROCM_OPTIMIZATION.md`. Headline
points:

1. **Picked vLLM on ROCm** as the only model-serving path that Radeon
   Cloud's "Dedicated Model API" template officially supports — so we
   played to the platform's strength.
2. **`VLLM_ATTENTION_BACKEND=ROCM_ATTN`** + uninstalled the prebuilt
   NVIDIA `flash_attn` package so vLLMs rotary-embedding init doesn't
   crash trying to load `flash_attn_2_cuda`.
3. **`HF_HUB_OFFLINE=1`** to bypass vLLM 0.16's phone-home to
   `huggingface.co` for speculator configs (unreachable in many regions).
4. **`--enforce-eager`** for fast cold-start on ephemeral Radeon Cloud instances.
5. **`--gpu-memory-utilization 0.85`** + `--max-model-len 8192` leaves
   ~15% VRAM for the sidecar embedding model sharing the same card.
6. **AOTriton backend** is auto-selected internally by transformers for
   the embedding model's SDPA — we see this confirmed in our demo logs.

Throughput on our single gpu1100: streaming ~30-40 token/s for chat
completions, 25s to embed a 1000-chunk RAG corpus.
