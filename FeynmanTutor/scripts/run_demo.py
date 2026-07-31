#!/usr/bin/env python
"""Drive the FeynmanTutor agent in a fully scripted non-interactive mode
for demo purposes. Used to generate examples/session_raft.jsonl."""

import os
import sys
import json
import time
from pathlib import Path

# Set environment for local model paths (offline)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

FT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FT_ROOT))

from agent.core import Agent


# A scripted set of learner answers to common Feynman probes — the agent
# never blocks waiting for input from the keyboard this way.
SCRIPTED_ANSWERS = [
    "A leader in Raft is elected by randomized timeouts — when a server doesn't hear from the leader for a random period, it becomes a candidate and asks for votes. Random timeouts make sure only one candidate typically wins per term, avoiding split votes.",
    "Log replication works because the leader sends AppendEntries RPCs to followers with the new log entry. Followers write the entry and reply with ack. Once a majority ack, the leader commits the entry and notifies followers.",
    "Safety means a committed entry is never lost and all committed entries are identical across servers, as long as a majority is up. This comes from the leader's write barrier requiring majority quorum before commit.",
    "Membership change in Raft uses joint consensus — old and new configs overlap so neither side can independently commit. Once the joint config is committed, the leader proposes the new-only config and the transition completes safely.",
    "Paxos is more general and harder to understand. Raft redesigned Paxos around a strong leader for clarity, sacrificing some generality. Raft is widely used because clarity reduces implementation bugs more than theoretical generality does.",
]
TOPIC = os.environ.get("FT_TOPIC", "Raft consensus algorithm")


def main():
    agent = Agent(topic=TOPIC, transcript_path="examples/session_raft.jsonl")
    print(f"=== FeynmanTutor scripted demo: topic = '{TOPIC}' ===")
    state = {"iter": 0, "answer_idx": 0}

    for step in range(40):
        round_ = agent.step()
        kind = round_.get("kind")
        if kind == "error":
            print(f"[step {step}] ERROR: {round_['content'][:200]}")
            continue
        if kind == "tool":
            print(f"[step {step}] TOOL: {round_['tool_names']}")
            continue
        if kind == "text":
            txt = round_["content"]
            print(f"[step {step}] ASSISTANT: {txt[:300]}")
            # Heuristic: if the message looks like a question, feed the next scripted answer.
            if "?" in txt and ("your" in txt.lower() or "you" in txt.lower() or "explain" in txt.lower() or "why" in txt.lower()):
                if state["answer_idx"] < len(SCRIPTED_ANSWERS):
                    ans = SCRIPTED_ANSWERS[state["answer_idx"]]
                    print(f"\n[LEARNER ANSWER]: {ans[:300]} ...\n")
                    agent.add_user_message(ans)
                    state["answer_idx"] += 1
                    time.sleep(0.2)  # be nice to the server

    print("\n=== Transcript saved to examples/session_raft.jsonl ===")


if __name__ == "__main__":
    main()
