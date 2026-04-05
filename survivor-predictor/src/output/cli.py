"""Rich-formatted terminal output for predictions and market comparison."""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns

from src import config
from src.analysis.llm_analyzer import get_cost_summary

console = Console()


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _delta_str(delta: float) -> str:
    """Format a probability delta with sign and color indication."""
    pct = delta * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def print_odds_table(
    predictions: dict,
    market_odds: Optional[dict[str, float]] = None,
    mispricing_threshold: float = config.MISPRICING_THRESHOLD,
) -> None:
    """
    Print the full odds table to the terminal.

    Parameters
    ----------
    predictions:   output of model.build_predictions()
    market_odds:   { contestant_name: implied_probability } from Kalshi
    """
    episode = predictions.get("episode_number", "?")
    season = predictions.get("season", "?")
    show = predictions.get("show", "survivor").upper()
    n_remaining = predictions.get("n_remaining", "?")

    console.print(
        Panel(
            f"[bold cyan]{show} Season {season} — Episode {episode}[/bold cyan]\n"
            f"[dim]{n_remaining} contestants remaining[/dim]",
            expand=False,
        )
    )

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        expand=False,
    )
    table.add_column("Contestant", style="bold", min_width=20)
    table.add_column("Win% (Model)", justify="right")
    if market_odds:
        table.add_column("Win% (Market)", justify="right")
        table.add_column("Delta", justify="right")
    table.add_column("Elim%", justify="right")
    table.add_column("Flag", justify="center")

    contestants = predictions.get("contestants", {})
    # Sort by win_prob descending
    sorted_c = sorted(contestants.items(), key=lambda x: x[1]["win_prob"], reverse=True)

    mispricings: list[tuple[str, float, float, float]] = []  # (name, model, market, delta)

    for name, data in sorted_c:
        win_m = data["win_prob"]
        elim_m = data["elim_prob"]
        market_p = market_odds.get(name) if market_odds else None

        flag = ""
        row_style = ""
        delta_text = ""

        if market_p is not None:
            delta = win_m - market_p
            delta_text = _delta_str(delta)
            if abs(delta) >= mispricing_threshold:
                flag = "⚠️" if delta > 0 else "📉"
                row_style = "on dark_green" if delta > 0 else "on dark_red"
                mispricings.append((name, win_m, market_p, delta))

        row = [
            Text(name, style=row_style),
            Text(_pct(win_m), style=row_style),
        ]
        if market_odds:
            row.append(Text(_pct(market_p) if market_p is not None else "—", style=row_style))
            row.append(Text(delta_text if delta_text else "—", style=row_style))
        row.append(Text(_pct(elim_m), style=row_style))
        row.append(Text(flag, style=row_style))

        table.add_row(*row)

    console.print(table)

    if mispricings:
        console.print("\n[bold yellow]⚡ Potential Mispricings[/bold yellow]")
        for name, model_p, market_p, delta in sorted(mispricings, key=lambda x: abs(x[3]), reverse=True):
            direction = "underpriced" if delta > 0 else "overpriced"
            console.print(
                f"  [bold]{name}[/bold] — model [cyan]{_pct(model_p)}[/cyan], "
                f"market [yellow]{_pct(market_p)}[/yellow] "
                f"→ [{'green' if delta > 0 else 'red'}]{direction} by {_pct(abs(delta))}[/]"
            )
    else:
        console.print("[dim]No mispricings above threshold.[/dim]")


def print_cost_summary() -> None:
    """Print running LLM cost totals."""
    stats = get_cost_summary()
    console.print(
        f"\n[dim]LLM usage this run: "
        f"{stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out — "
        f"${stats['total_cost_usd']:.4f} ({stats['calls']} call(s))[/dim]"
    )


def print_elimination_forecast(predictions: dict) -> None:
    """Print a focused next-elimination bar-chart-style view."""
    contestants = predictions.get("contestants", {})
    sorted_c = sorted(contestants.items(), key=lambda x: x[1]["elim_prob"], reverse=True)

    console.print("\n[bold]🔥 Next Elimination Forecast[/bold]")
    for name, data in sorted_c[:8]:
        prob = data["elim_prob"]
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        color = "red" if prob > 0.3 else "yellow" if prob > 0.15 else "green"
        console.print(f"  [{color}]{bar}[/] {name} — {_pct(prob)}")
