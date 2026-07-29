# FeynmanTutor — A Local Learning Agent on AMD Radeon GPU

> Track 2 — Development & Local Deployment of Private AI Agents

FeynmanTutor is a **fully local** learning-assistant Agent that runs end-to-end on a
**single AMD Radeon GPU (gfx1100, 48 GB VRAM)** with **ROCm**. It helps a learner
study a new skill by combining four classical learning methodologies into one
conversational agent:

| Methodology          | How the agent applies it                                       |
| -------------------- | ------------------------------------------------------------- |
| Active resource gathering  | DuckDuckGo search + user-uploaded material (PDF/DOCX/MD) |
| Skill decomposition       | Auto-split a skill into sub-skills and a timed plan         |
| Feynman technique         | The agent explains in plain words, then probes you back   |
| Spaced repetition         | SM-2 schedule based on Ebbinghaus' forgetting curve       |
| Cornell notes & quiz      | Auto-generated notes and quizzes, drift back to review    |

Everything runs locally — no external LLM API calls. The inference backend is
**vLLM on ROCm** serving **Qwen2.5-14B-Instruct** as an OpenAI-compatible
endpoint, and the agent talks to it through the standard OpenAI SDK.

---

## What this repo contains

```
FeynmanTutor/
├── README.md                  ← this file (project intro + quick start)
├── docs/
│   ├── ARCHITECTURE.md        ← system architecture & agent loop diagram
│   ├── REPRODUCE.md           ← step-by-step reproduction instructions
│   ├── ROCM_OPTIMIZATION.md   ← why/how we optimized for AMD Radeon GPU
│   └── DEMO_SCRIPT.md         ← narration script for the demo video
├── server/
│   ├── start_vllm.sh          ← one command to launch vLLM + Qwen2.5-14B
│   └── requirements.txt       ← python deps for the server
├── agent/
│   ├── tools/                 ← every tool the agent can call
│   │   ├── web_research.py    ← DuckDuckGo material search
│   │   ├── material_loader.py ← parse PDF/DOCX/MD uploads
│   │   ├── knowledge_base.py  ← sentence-transformers + FAISS RAG
│   │   ├── feynman.py         ← Feynman explain + probe
│   │   ├── memory_curve.py    ← SM-2 forgetting-curve scheduler
│   │   ├── planner.py         ← skill decomposition + timetable
│   │   ├── quiz.py            ← auto quiz generation + grading
│   │   └── cornell_notes.py   ← Cornell-style note synthesis
│   ├── core.py                ← ReAct loop + tool dispatch
│   ├── prompts.py             ← system prompts for each stage
│   └── run_cli.py             ← terminal entry point
├── examples/
│   └── session_raft.jsonl     ← one complete learning session about Raft
└── assets/                    ← screenshots / diagrams used in docs
```

---

## Quick start (3 commands)

```bash
# 1) bootstrap the environment (Python venv + pip deps + model download)
bash FeynmanTutor/scripts/bootstrap.sh

# 2) launch the local vLLM server (OpenAI-compatible @ http://127.0.0.1:8000)
bash FeynmanTutor/server/start_vllm.sh

# 3) run the agent for an end-to-end learning session about Raft
python FeynmanTutor/agent/run_cli.py --topic "Raft consensus algorithm"
```

For full reproduction instructions, environment specifications, and
troubleshooting, see **[`docs/REPRODUCE.md`](docs/REPRODUCE.md)**.

For the system architecture and agent loop, see
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

---

## Why it fits the AMD Radeon GPU track

- **All inference runs on one Radeon gfx1100 GPU via ROCm** — no NVIDIA, no cloud LLM, no external API key needed by default.
- The model is served with **vLLM-ROCm**, AMD's officially-supported inference path on Radeon Cloud (per the platform's Model API template options).
- A 14B model in bf16 fits comfortably in 48 GB VRAM, leaving headroom for large KV-cache and concurrent agent/tool traffic.
- Lighter-weight side models (embedding for RAG) also run on the same GPU.

More details in **[`docs/ROCM_OPTIMIZATION.md`](docs/ROCM_OPTIMIZATION.md)**.

---

## Demo

A scripted end-to-end session that learns the **Raft consensus algorithm** is
recorded in [`examples/session_raft.jsonl`](examples/session_raft.jsonl). The
narration for the demo video is in
[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).

---

## License & acknowledgements

Built for the AMD Radeon Hackathon (2026-07). Qwen2.5-14B is © Alibaba Cloud
under the Apache 2.0 license. The Agent code in this repository is released
under MIT.
