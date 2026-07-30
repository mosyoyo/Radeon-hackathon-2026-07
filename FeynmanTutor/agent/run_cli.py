"""run_cli — terminal entry point for the FeynmanTutor agent.

Usage:
    # driven / scripted demo
    python -m agent.run_cli --topic "Raft consensus algorithm"

    # interactive REPL
    python -m agent.run_cli

The agent prints each round with rich-formatted boxes:
- assistant text -> blue box
- tool event     -> yellow box (name + result)
- learner prompt -> green, when the model says "wait for answer"
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from agent.core import Agent

console = Console(width=132)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="", help="the skill the learner wants to study")
    parser.add_argument("--transcript", default="", help="path to append a JSONL transcript to")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--base-url", default="", help="override FT_BASE_URL on the fly")
    args = parser.parse_args()

    if args.base_url:
        import os
        os.environ["FT_BASE_URL"] = args.base_url
        # reload to pick up the new env var
        import importlib
        import agent.llm as _llm
        importlib.reload(_llm)
        _llm.MODEL  # touch to ensure reload

    transcript = args.transcript or None
    agent = Agent(topic=args.topic, transcript_path=transcript)

    if not args.topic:
        topic = Prompt.ask("[bold]What skill do you want to learn?[/bold]")
        agent.topic = topic
        # Reuse __post_init__ by calling it manually
        Agent.__post_init__(agent)

    console.rule(f"[bold cyan]FeynmanTutor — learning '{agent.topic}'[/bold cyan]")

    for step in range(args.max_steps):
        try:
            round_ = agent.step()
        except KeyboardInterrupt:
            console.print("[yellow]interrupted by user[/yellow]")
            return 130

        if round_["kind"] == "error":
            console.print(Panel(round_["content"], title="error", border_style="red"))
            continue
        if round_["kind"] == "text":
            content = round_["content"]
            # Detect if the message ends with a question (the Feynman probe
            # waits for the learner's reply). Crude heuristic: contains '?'.
            console.print(Panel(content, title="FeynmanTutor", border_style="blue"))
            if "?" in content and ("you" in content.lower() or "your" in content.lower()):
                # likely a probe — wait for learner input
                answer = Prompt.ask("[bold green]your answer[/bold green]")
                agent.add_user_message(answer)
                continue
            # Otherwise, just one more step. The model may stop by itself.
            continue
        if round_["kind"] == "tool":
            console.print(Panel(
                ", ".join(round_["tool_names"]),
                title="tool call", border_style="yellow",
            ))
            continue

    console.print("[bold] Reached max steps — exiting. Bye![/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
