"""Interactive real-time episode analysis CLI."""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich import box

from src import config
from src.analysis import llm_analyzer, scoring
from src.prediction import model as prediction_model
from src.market import kalshi
from src.output import cli as cli_output

console = Console()


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _delta_str(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f}%"


def run_live_episode(
    show_slug: str,
    season: int,
    episode_number: int,
) -> None:
    """
    Interactive loop for real-time episode analysis.
    At each act break, user enters a text summary; model updates predictions.
    Uses Haiku for speed, Sonnet runs the "official" post-episode pass.
    """
    console.print(
        Panel(
            f"[bold red]🔴 LIVE MODE — {show_slug.upper()} Season {season}, Episode {episode_number}[/bold red]\n"
            "[dim]At each act break, enter a brief summary. Type 'done' when episode ends. "
            "Type 'reddit' to pull live Reddit thread. Type 'skip' to skip an act.[/dim]",
            border_style="red",
        )
    )

    # Load current contestants
    active = prediction_model.get_active_contestants(show_slug, season)
    contestant_names = [c["name"] for c in active]

    # Try to load existing cumulative scores for context
    cum_scores = scoring.compute_cumulative_scores(show_slug, season)
    previous_scores_str = json.dumps(
        {name: cum_scores.get(name, {}) for name in contestant_names}, indent=2
    )

    # Fetch initial Kalshi odds
    market_odds = kalshi.fetch_survivor_odds(
        show_slug, season, episode_number, contestant_names, cache=False
    )

    # Start Kalshi polling thread
    live_dir = config.live_dir(show_slug, season)
    live_dir.mkdir(parents=True, exist_ok=True)
    timeline: list[dict] = []
    stop_polling = threading.Event()

    def _poll_kalshi() -> None:
        import time
        while not stop_polling.is_set():
            try:
                odds = kalshi.fetch_survivor_odds(
                    show_slug, season, episode_number, contestant_names, cache=False
                )
                if odds:
                    timeline.append({"timestamp": datetime.utcnow().isoformat(), "odds": odds})
            except Exception:
                pass
            time.sleep(60)

    poll_thread = threading.Thread(target=_poll_kalshi, daemon=True)
    poll_thread.start()

    # Running score deltas (accumulated across acts)
    running_deltas: dict[str, dict[str, float]] = {}
    act_number = 0

    while True:
        act_number += 1
        console.print(f"\n[bold cyan][Act {act_number} break][/bold cyan] Enter summary (or 'done'/'skip'/'reddit'):")
        user_input = Prompt.ask(">")

        if user_input.strip().lower() == "done":
            break
        if user_input.strip().lower() == "skip":
            continue
        if user_input.strip().lower() == "reddit":
            show_config = config.load_show_config(show_slug)
            subreddits = show_config["show"]["subreddits"]
            # Pull very recent Reddit comments
            import praw
            try:
                r = praw.Reddit(
                    client_id=config.REDDIT_CLIENT_ID,
                    client_secret=config.REDDIT_CLIENT_SECRET,
                    user_agent=config.REDDIT_USER_AGENT,
                )
                sub = r.subreddit(subreddits[0])
                new_posts = list(sub.new(limit=5))
                live_comments: list[str] = []
                for post in new_posts:
                    if "episode" in post.title.lower() or "live" in post.title.lower():
                        post.comments.replace_more(limit=0)
                        for c in list(post.comments)[:20]:
                            live_comments.append(c.body)
                        break
                if live_comments:
                    user_input = "\n".join(live_comments[:20])
                    console.print(f"[dim]Pulled {len(live_comments)} Reddit comments[/dim]")
                else:
                    console.print("[yellow]No live thread found. Enter summary manually:[/yellow]")
                    user_input = Prompt.ask(">")
            except Exception as e:
                console.print(f"[red]Reddit error: {e}[/red]")
                continue

        # Run quick Haiku analysis
        with console.status("[yellow]Analyzing...[/yellow]"):
            try:
                result = llm_analyzer.quick_live_analysis(
                    show_slug=show_slug,
                    act_summary=user_input,
                    contestant_list=contestant_names,
                    previous_scores=previous_scores_str,
                )
            except Exception as e:
                console.print(f"[red]Analysis error: {e}[/red]")
                continue

        updates = result.get("quick_updates", {})
        act_sum = result.get("act_summary", "")

        if act_sum:
            console.print(f"[dim italic]{act_sum}[/dim italic]")

        # Refresh market odds
        try:
            market_odds = kalshi.fetch_survivor_odds(
                show_slug, season, episode_number, contestant_names, cache=False
            )
        except Exception:
            pass

        # Display updated table
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta", expand=False)
        table.add_column("Contestant", min_width=18)
        table.add_column("Elim Δ", justify="right")
        table.add_column("Win Δ", justify="right")
        table.add_column("Kalshi Win%", justify="right")
        table.add_column("Note")

        for name in contestant_names:
            upd = updates.get(name, {})
            elim_d = upd.get("elimination_risk_delta", 0)
            win_d = upd.get("winner_signals_delta", 0)
            note = upd.get("note", "")
            market_p = market_odds.get(name)

            elim_color = "red" if elim_d > 0 else "green" if elim_d < 0 else "white"
            win_color = "green" if win_d > 0 else "red" if win_d < 0 else "white"

            table.add_row(
                name,
                f"[{elim_color}]{'+' if elim_d >= 0 else ''}{elim_d:+d}[/]",
                f"[{win_color}]{'+' if win_d >= 0 else ''}{win_d:+d}[/]",
                f"{market_p * 100:.1f}%" if market_p else "—",
                note[:60],
            )

        console.print(table)

        # Flag large market movements
        if market_odds:
            sorted_m = sorted(market_odds.items(), key=lambda x: x[1], reverse=True)
            console.print("\n[bold]📊 Market odds:[/bold]", " | ".join(f"{n}: {p:.0%}" for n, p in sorted_m[:6]))

    # Episode ended — save live timeline
    stop_polling.set()
    timeline_path = live_dir / f"episode_{episode_number:02d}_kalshi_timeline.json"
    with open(timeline_path, "w") as f:
        json.dump({"episode": episode_number, "timeline": timeline}, f, indent=2)

    console.print(
        f"\n[green]Live mode complete. Timeline saved. "
        f"Now run: python -m src.main --episode {episode_number} --sources manual to run the full post-episode analysis.[/green]"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live episode mode")
    parser.add_argument("--show", default="survivor")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args()

    run_live_episode(args.show, args.season, args.episode)
