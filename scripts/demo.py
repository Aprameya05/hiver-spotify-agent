"""
Interactive CLI demo for the Spotify support agent.

Usage:
    python scripts/demo.py
    python scripts/demo.py --message "my app keeps crashing"
    python scripts/demo.py --batch data/golden_eval/examples.json --n 10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import SpotifyAgent
from src.utils import load_config

app = typer.Typer(help="Spotify AI support agent demo")
console = Console()


def print_response(agent, resp):
    esc_color = "red" if resp.escalation.should_escalate else "green"
    esc_label = "ESCALATE TO HUMAN" if resp.escalation.should_escalate else "AUTO-HANDLE"

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Field", style="bold cyan", width=18)
    table.add_column("Value")

    table.add_row("Message", resp.original_message)
    table.add_row("Cleaned", resp.cleaned_message)
    table.add_row("Intent", f"[bold]{resp.intent.intent}[/bold] ({resp.intent.confidence:.1%})")
    table.add_row("Decision", f"[bold {esc_color}]{esc_label}[/bold {esc_color}] (score={resp.escalation.score:.2f})")
    table.add_row("Reply draft", f"[italic]{resp.reply.reply}[/italic]")
    table.add_row("Reasons", "\n".join(f"- {r}" for r in resp.escalation.reasons))
    table.add_row("Latency", f"{resp.latency_ms:.0f}ms")

    if resp.intent.refinement_used:
        table.add_row("Note", "[yellow]LLM refinement used for intent[/yellow]")

    console.print(Panel(table, title="[bold white]Spotify Support Agent[/bold white]", border_style="blue"))


@app.command()
def interactive(
    config: str = typer.Option(None, help="Path to config YAML"),
    message: str = typer.Option(None, "--message", "-m", help="Single message to process"),
):
    """Run the agent interactively or on a single message."""
    cfg = load_config(config)
    console.print("\n[bold green]Loading Spotify support agent...[/bold green]")
    agent = SpotifyAgent(cfg)
    agent.warm_up()
    console.print("[bold green]Agent ready![/bold green]\n")

    if message:
        resp = agent.handle(message)
        print_response(agent, resp)
        return

    console.print("Type a customer message (or 'quit' to exit):\n")
    while True:
        try:
            msg = console.input("[bold cyan]Customer> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg.lower() in ("quit", "exit", "q"):
            break
        resp = agent.handle(msg)
        print_response(agent, resp)
        print()


@app.command()
def batch(
    input_file: str = typer.Argument(..., help="JSON file with examples"),
    n: int = typer.Option(10, help="Number of examples to process"),
    config: str = typer.Option(None, help="Path to config YAML"),
    output: str = typer.Option(None, help="Output JSON file for results"),
):
    """Run the agent on a batch of examples from a JSON file."""
    cfg = load_config(config)
    console.print("[bold green]Loading Spotify support agent...[/bold green]")
    agent = SpotifyAgent(cfg)
    agent.warm_up()

    with open(input_file) as f:
        examples = json.load(f)

    examples = examples[:n]
    console.print(f"Processing {len(examples)} examples...\n")

    results = []
    correct_intent = 0
    auto_count = 0

    for ex in examples:
        resp = agent.handle(ex["text"])
        is_correct = resp.intent.intent == ex.get("intent", "")
        if is_correct:
            correct_intent += 1
        if not resp.escalation.should_escalate:
            auto_count += 1

        results.append({
            "id": ex.get("id", ""),
            "text": ex["text"],
            "true_intent": ex.get("intent", "N/A"),
            "pred_intent": resp.intent.intent,
            "correct": is_correct,
            "confidence": resp.intent.confidence,
            "escalate": resp.escalation.should_escalate,
            "escalation_score": resp.escalation.score,
            "reply": resp.reply.reply,
        })
        print_response(agent, resp)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Intent accuracy: {correct_intent/len(examples):.1%}")
    console.print(f"  Auto-handled:    {auto_count/len(examples):.1%}")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"  Results saved to: {output}")


if __name__ == "__main__":
    app()
