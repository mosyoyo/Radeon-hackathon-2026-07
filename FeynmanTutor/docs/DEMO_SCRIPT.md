# Demo Video Script — FeynmanTutor

> Recommended length: 3–5 minutes. Show the actual operation, from CLI to final
> result, on the AMD Radeon GPU.

The narration below is matched one-to-one against
[`examples/session_raft.jsonl`](../examples/session_raft.jsonl) so the
evaluator can play both together.

---

## T+0:00 — Title card

"FeynmanTutor — a local learning Agent running entirely on one AMD Radeon
gfx1100 GPU via ROCm. We learn **the Raft consensus algorithm** end-to-end
in this demo."

Screen: terminal, agent banner + prompt `I want to learn "Raft consensus algorithm"`.

## T+0:15 — Verify the GPU

```bash
rocm-smi --showproductname --showmeminfo vram
```

Narration:
"First — the GPU is real and empty: one AMD gfx1100 with 48 GB of VRAM,
the whole ROCm 6.16 stack loaded."

## T+0:30 — Verify vLLM is up

```bash
curl http://127.0.0.1:8000/v1/models
```

Narration:
"The model is Qwen2.5-14B-Instruct, served with vLLM on ROCm — OpenAI-compatible.

## T+0:45 — Start the agent

```bash
python agent/run_cli.py --topic "Raft consensus algorithm"
```

Scroll the agent while it streams.

## T+1:00 — Stage 1: planning

Narration:
"The agent uses `make_plan` to break Raft down into four sub-skills —
*Leader Election*, *Log Replication*, *Safety*, *Membership Change* — and
a 4-session × 1-hour weekly plan."

On-screen: the JSON tool-call event from `make_plan`.

## T+1:30 — Stage 2: web research

Narration:
"It then calls `web_research` against DuckDuckGo — no API key, no external
LLM provider — and pulls five reference links including the original Raft
paper and the MIT 6.824 lecture notes. Those snippets are stored in the
local RAG index for later lookups."

On-screen: `web_research` tool event + five URLs.

## T+2:00 — Stage 3: Feynman explanation + probing

Narration:
"Now the agent uses `feynman_explain` on the first sub-skill, *Leader Election*,
framing it as a real-world board-meeting vote. Then `feynman_probe` asks:
'in your own words, what is a split vote and how does Raft avoid it?' —
this is the Feynman technique: forcing recall."

Show the user typing their answer.

## T+2:45 — Stage 4: quiz + scoring

Narration:
"Five quiz questions, scored automatically. `record_mastery` stores 0.72
for Leader Election and triggers the SM-2 spaced-repetition scheduler."

Show the `quiz` output and the schedule.

## T+3:15 — Stage 5: spaced repetition

Narration:
"Based on the Ebbinghaus forgetting curve, the agent schedules the next
review of Leader Election in **3 days**, the next at the usual 7-day / 16-day
rebound points."

On-screen: `schedule_review` event with the date.

## T+3:30 — Stage 6: Cornell notes + RAG lookup

Narration:
"Finally, `cornell_notes` synthesises a Cornell-style note for the session,
and along the way we exercise the RAG store by asking the agent to *'summarize
Section 5.4 of the Raft paper'* — the call goes through `rag_lookup`."

On-screen: `notes/raft-leader-election.md` on the right pane.

## T+3:45 — Summary

"FeynmanTutor ran six classical learning methodologies end-to-end —
decomposition, web research, Feynman technique, quiz, spaced repetition,
Cornell notes — entirely on one AMD Radeon GPU. No NVIDIA. No cloud LLM
provider. No paid API. Thank you."

End card: project URL + OpenAI-compatible endpoint URL (optionally tunnelled
via `rc-tunnel`).

---

## Things to prepare before recording

- Clear the cache so the demo starts cold but completes fast:
  ```bash
  rm -rf FeynmanTutor/data/kb FeynmanTutor/data/learner.sqlite
  ```
- Pre-warm the vLLM server with a `curl` ping so the first chat completes
  quickly on camera.
- Pick a non-overlapping terminal width (≥ 132 columns) so JSON events render
  cleanly.
- If showing the tunnel endpoint for the public URL, set it up ~5 min before
  recording so the domain is resolvable.
